
import "dotenv/config";
import express from "express";
import { prisma } from "./lib/prisma.js";

const app = express();
app.use(express.json({ limit: "5mb" }));

app.get("/health", async (_req, res) => {
  try {
    await prisma.$queryRaw`SELECT 1`;
    res.json({ status: "ok", database: "connected" });
  } catch (error) {
    console.error("Database health check failed:", error);
    res.status(503).json({ status: "error", database: "unavailable" });
  }
});

// Create an analysis record.
app.post("/api/analyses", async (req, res) => {
  try {
    const {
      pcapFilename,
      totalFlows = 0,
      normalFlows = 0,
      attackFlows = 0,
      durationSeconds,
    } = req.body;

    if (typeof pcapFilename !== "string" || !pcapFilename.trim()) {
      res.status(400).json({ error: "pcapFilename is required" });
      return;
    }

    const analysis = await prisma.analysis.create({
      data: {
        pcapFilename,
        totalFlows: Number(totalFlows),
        normalFlows: Number(normalFlows),
        attackFlows: Number(attackFlows),
        durationSeconds: durationSeconds ?? null,
        status: "PROCESSING",
      },
    });

    res.status(201).json({ analysisId: analysis.id });
  } catch (error) {
    console.error("Create analysis failed:", error);
    res.status(500).json({ error: "Could not create analysis" });
  }
});

// Save metrics and mark an analysis as completed.
app.post("/api/analyses/:id/complete", async (req, res) => {
  try {
    const id = req.params.id;
    const {
      totalFlows,
      normalFlows,
      attackFlows,
      durationSeconds,
      metrics,
    } = req.body;

    const analysis = await prisma.analysis.findUnique({
      where: { id },
      select: { id: true },
    });

    if (!analysis) {
      res.status(404).json({ error: "Analysis not found" });
      return;
    }

    const updated = await prisma.analysis.update({
      where: { id },
      data: {
        status: "COMPLETED",
        completedAt: new Date(),
        ...(totalFlows !== undefined && { totalFlows: Number(totalFlows) }),
        ...(normalFlows !== undefined && { normalFlows: Number(normalFlows) }),
        ...(attackFlows !== undefined && { attackFlows: Number(attackFlows) }),
        ...(durationSeconds !== undefined && {
          durationSeconds: durationSeconds == null ? null : Number(durationSeconds),
        }),
        ...(metrics && {
          metrics: {
            upsert: {
              create: {
                accuracy: metrics.accuracy ?? null,
                precision: metrics.precision ?? null,
                recall: metrics.recall ?? null,
                f1Score: metrics.f1Score ?? null,
                rocAuc: metrics.rocAuc ?? null,
              },
              update: {
                accuracy: metrics.accuracy ?? null,
                precision: metrics.precision ?? null,
                recall: metrics.recall ?? null,
                f1Score: metrics.f1Score ?? null,
                rocAuc: metrics.rocAuc ?? null,
              },
            },
          },
        }),
      },
    });

    res.json({ analysisId: updated.id, status: updated.status });
  } catch (error) {
    console.error("Complete analysis failed:", error);
    res.status(500).json({ error: "Could not complete analysis" });
  }
});

app.post("/api/analyses/:id/fail", async (req, res) => {
  try {
    const id = req.params.id;
    const { errorMessage } = req.body;

    if (typeof errorMessage !== "string" || !errorMessage.trim()) {
      res.status(400).json({ error: "errorMessage is required" });
      return;
    }

    const failed = await prisma.analysis.updateMany({
      where: { id, status: "PROCESSING" },
      data: {
        status: "FAILED",
        errorMessage: errorMessage.trim().slice(0, 1000),
        completedAt: new Date(),
      },
    });

    if (failed.count === 0) {
      const analysis = await prisma.analysis.findUnique({
        where: { id },
        select: { status: true },
      });

      if (!analysis) {
        res.status(404).json({ error: "Analysis not found" });
        return;
      }

      res.status(409).json({
        error: `Analysis is already ${analysis.status.toLowerCase()}`,
      });
      return;
    }

    res.json({ analysisId: id, status: "FAILED" });
  } catch (error) {
    console.error("Fail analysis update failed:", error);
    res.status(500).json({ error: "Could not mark analysis as failed" });
  }
});

// Save up to 1000 flow predictions per request.
app.post("/api/analyses/:id/predictions", async (req, res) => {
  try {
    const analysisId = req.params.id;
    const rows = req.body.predictions;

    if (!Array.isArray(rows) || rows.length > 1000) {
      res.status(400).json({
        error: "predictions must contain at most 1000 rows",
      });
      return;
    }

    const analysis = await prisma.analysis.findUnique({
      where: { id: analysisId },
      select: { id: true },
    });

    if (!analysis) {
      res.status(404).json({ error: "Analysis not found" });
      return;
    }

    const predictions = rows.map((p: any) => ({
      analysisId,
      flowId: p.flowId == null ? null : String(p.flowId),
      srcIp: p.srcIp == null ? null : String(p.srcIp),
      dstIp: p.dstIp == null ? null : String(p.dstIp),
      srcPort: p.srcPort == null ? null : Number(p.srcPort),
      dstPort: p.dstPort == null ? null : Number(p.dstPort),
      protocol: p.protocol == null ? null : String(p.protocol),
      predictionLabel: String(p.predictionLabel),
      normalProbability: p.normalProbability ?? null,
      attackProbability: p.attackProbability ?? null,
    }));

    if (predictions.some(p => !p.predictionLabel.trim())) {
      res.status(400).json({ error: "predictionLabel is required" });
      return;
    }

    const result = await prisma.prediction.createMany({ data: predictions });
    res.status(201).json({ saved: result.count });
  } catch (error) {
    console.error("Save predictions failed:", error);
    res.status(500).json({ error: "Could not save predictions" });
  }
});

// Retrieve recent analysis history.
app.get("/api/analyses", async (req, res) => {
  try {
    const rawLimit = Number(req.query.limit ?? 20);
    const limit = Number.isFinite(rawLimit)
      ? Math.min(Math.max(Math.floor(rawLimit), 1), 100)
      : 20;

    const history = await prisma.analysis.findMany({
      take: limit,
      orderBy: { createdAt: "desc" },
      include: { metrics: true },
    });

    res.json(history);
  } catch (error) {
    console.error("Load history failed:", error);
    res.status(500).json({ error: "Could not load history" });
  }
});

// Retrieve predictions for one analysis.
app.get("/api/analyses/:id/predictions", async (req, res) => {
  try {
    const rawSkip = Number(req.query.skip ?? 0);
    const rawLimit = Number(req.query.limit ?? 100);

    const skip = Number.isFinite(rawSkip)
      ? Math.max(Math.floor(rawSkip), 0)
      : 0;
    const take = Number.isFinite(rawLimit)
      ? Math.min(Math.max(Math.floor(rawLimit), 1), 500)
      : 100;

    const predictions = await prisma.prediction.findMany({
      where: { analysisId: req.params.id },
      orderBy: { id: "asc" },
      skip,
      take,
    });

    
    res.json(
      predictions.map((prediction) => ({
        ...prediction,
        id: prediction.id.toString(),
      }))
    );

  } catch (error) {
    console.error("Load predictions failed:", error);
    res.status(500).json({ error: "Could not load predictions" });
  }
});

const port = Number(process.env.PORT ?? 3000);
const host = process.env.HOST ?? "127.0.0.1";
const server = app.listen(port, host, () => {
  console.log(`TabPFN database API: http://${host}:${port}`);
});

async function shutdown() {
  server.close();
  await prisma.$disconnect();
}

process.on("SIGINT", () => void shutdown());
process.on("SIGTERM", () => void shutdown());
