"""Client for persisting TabPFN-NIDS results through the TypeScript database API."""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)


def _request(
    path: str,
    *,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    base_url = os.getenv(
        "TABPFN_DATABASE_API_URL",
        "http://127.0.0.1:3000",
    ).rstrip("/")

    request = Request(
        f"{base_url}{path}",
        data=(
            json.dumps(payload, allow_nan=False).encode("utf-8")
            if payload is not None
            else None
        ),
        headers=(
            {"Content-Type": "application/json"}
            if payload is not None
            else {}
        ),
        method=method,
    )

    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Database API returned HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not reach database API: {exc.reason}"
        ) from exc


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return _request(path, method="POST", payload=payload)


def get_analysis_history(limit: int) -> list[dict[str, Any]]:
    """Load recent saved analyses from the database API."""
    query = urlencode({"limit": limit})
    result = _request(f"/api/analyses?{query}")
    if not isinstance(result, list):
        raise RuntimeError("Database API returned an invalid analysis history")
    return result


def get_analysis_by_id(analysis_id: str) -> dict[str, Any]:
    """Load a single saved analysis from the database API."""
    result = _request(f"/api/analyses/{quote(analysis_id, safe='')}")
    if not isinstance(result, dict):
        raise RuntimeError("Database API returned an invalid analysis record")
    return result


def get_analysis_predictions(
    analysis_id: str,
    skip: int,
    limit: int,
) -> list[dict[str, Any]]:
    """Load a page of saved flow predictions for an analysis."""
    query = urlencode({"skip": skip, "limit": limit})
    result = _request(
        f"/api/analyses/{quote(analysis_id, safe='')}/predictions?{query}"
    )
    if not isinstance(result, list):
        raise RuntimeError("Database API returned invalid predictions")
    return result


def _value(row: pd.Series, *names: str) -> Any:
    for name in names:
        if name not in row.index:
            continue

        value = row[name]

        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except (TypeError, ValueError):
            pass

        return value

    return None


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None

    try:
        number = float(value)
        if not math.isfinite(number) or not number.is_integer():
            return None
        return int(number)
    except (TypeError, ValueError, OverflowError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None

    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def persist_analysis(
    pcap_filename: str,
    output: pd.DataFrame,
    duration_seconds: float,
) -> str:
    """Save a completed PCAP analysis and its flow predictions."""

    if "prediction_label" not in output.columns:
        raise ValueError("Missing required prediction_label column")

    total = len(output)
    labels = output["prediction_label"].astype(str)

    normal_count = int((labels == "Normal").sum())
    attack_count = int((labels == "Attack").sum())

    created = _post(
        "/api/analyses",
        {
            "pcapFilename": pcap_filename,
            "totalFlows": total,
            "normalFlows": normal_count,
            "attackFlows": attack_count,
            "durationSeconds": float(duration_seconds),
        },
    )

    analysis_id = created.get("analysisId")
    if not analysis_id:
        raise RuntimeError(
            f"Database API did not return analysisId: {created}"
        )

    batch_size = 500

    try:
        for start in range(0, total, batch_size):
            batch = output.iloc[start : start + batch_size]
            predictions = []

            for index, row in batch.iterrows():
                flow_id = _value(row, "flow_id")

                if flow_id is None:
                    flow_id = index + 1

                predictions.append(
                    {
                        "flowId": str(flow_id),
                        "srcIp": _string_or_none(
                            _value(row, "src_ip", "source_ip", "ip_src")
                        ),
                        "dstIp": _string_or_none(
                            _value(
                                row,
                                "dst_ip",
                                "dest_ip",
                                "destination_ip",
                                "ip_dst",
                            )
                        ),
                        "srcPort": _optional_int(
                            _value(row, "src_port", "source_port", "sport")
                        ),
                        "dstPort": _optional_int(
                            _value(
                                row,
                                "dst_port",
                                "dest_port",
                                "destination_port",
                                "dport",
                            )
                        ),
                        "protocol": _string_or_none(
                            _value(row, "protocol", "Protocol", "proto")
                        ),
                        "predictionLabel": str(row["prediction_label"]),
                        "normalProbability": _optional_float(
                            _value(row, "normal_probability")
                        ),
                        "attackProbability": _optional_float(
                            _value(row, "attack_probability")
                        ),
                    }
                )

            _post(
                f"/api/analyses/{analysis_id}/predictions",
                {"predictions": predictions},
            )

        _post(
            f"/api/analyses/{analysis_id}/complete",
            {
                "totalFlows": total,
                "normalFlows": normal_count,
                "attackFlows": attack_count,
                "durationSeconds": float(duration_seconds),
            },
        )
    except Exception:
        try:
            _post(
                f"/api/analyses/{analysis_id}/fail",
                {"errorMessage": "Analysis result persistence failed."},
            )
        except Exception:
            logger.exception(
                "Could not mark database analysis %s as failed",
                analysis_id,
            )
        raise

    logger.info(
        "Persisted analysis %s: %d flows",
        analysis_id,
        total,
    )

    return str(analysis_id)
