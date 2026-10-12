-- CreateEnum
CREATE TYPE "AnalysisStatus" AS ENUM ('PROCESSING', 'COMPLETED', 'FAILED');

-- CreateTable
CREATE TABLE "Analysis" (
    "id" TEXT NOT NULL,
    "pcapFilename" TEXT NOT NULL,
    "status" "AnalysisStatus" NOT NULL DEFAULT 'PROCESSING',
    "totalFlows" INTEGER NOT NULL DEFAULT 0,
    "normalFlows" INTEGER NOT NULL DEFAULT 0,
    "attackFlows" INTEGER NOT NULL DEFAULT 0,
    "durationSeconds" DOUBLE PRECISION,
    "errorMessage" TEXT,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,
    "completedAt" TIMESTAMP(3),

    CONSTRAINT "Analysis_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "EvaluationMetric" (
    "id" TEXT NOT NULL,
    "analysisId" TEXT NOT NULL,
    "accuracy" DOUBLE PRECISION,
    "precision" DOUBLE PRECISION,
    "recall" DOUBLE PRECISION,
    "f1Score" DOUBLE PRECISION,
    "rocAuc" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "EvaluationMetric_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "Prediction" (
    "id" BIGSERIAL NOT NULL,
    "analysisId" TEXT NOT NULL,
    "flowId" TEXT,
    "srcIp" TEXT,
    "dstIp" TEXT,
    "srcPort" INTEGER,
    "dstPort" INTEGER,
    "protocol" TEXT,
    "predictionLabel" TEXT NOT NULL,
    "normalProbability" DOUBLE PRECISION,
    "attackProbability" DOUBLE PRECISION,
    "createdAt" TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "Prediction_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE INDEX "Analysis_createdAt_idx" ON "Analysis"("createdAt");

-- CreateIndex
CREATE INDEX "Analysis_status_idx" ON "Analysis"("status");

-- CreateIndex
CREATE UNIQUE INDEX "EvaluationMetric_analysisId_key" ON "EvaluationMetric"("analysisId");

-- CreateIndex
CREATE INDEX "Prediction_analysisId_idx" ON "Prediction"("analysisId");

-- CreateIndex
CREATE INDEX "Prediction_predictionLabel_idx" ON "Prediction"("predictionLabel");

-- AddForeignKey
ALTER TABLE "EvaluationMetric" ADD CONSTRAINT "EvaluationMetric_analysisId_fkey" FOREIGN KEY ("analysisId") REFERENCES "Analysis"("id") ON DELETE CASCADE ON UPDATE CASCADE;

-- AddForeignKey
ALTER TABLE "Prediction" ADD CONSTRAINT "Prediction_analysisId_fkey" FOREIGN KEY ("analysisId") REFERENCES "Analysis"("id") ON DELETE CASCADE ON UPDATE CASCADE;
