from __future__ import annotations

from urllib.parse import parse_qs, urlparse

from fastapi.testclient import TestClient
import pandas as pd
import pytest

from tabpfn_nids.api import database_client


def test_get_analysis_history_uses_database_api(monkeypatch):
    requested: list[str] = []
    expected = [{"id": "analysis-1", "pcapFilename": "capture.pcap"}]

    def fake_request(path: str):
        requested.append(path)
        return expected

    monkeypatch.setattr(database_client, "_request", fake_request)

    result = database_client.get_analysis_history(limit=25)

    assert result == expected
    assert requested == ["/api/analyses?limit=25"]


def test_get_analysis_predictions_quotes_id_and_pages(monkeypatch):
    requested: list[str] = []
    expected = [{"flowId": "1", "predictionLabel": "Normal"}]

    def fake_request(path: str):
        requested.append(path)
        return expected

    monkeypatch.setattr(database_client, "_request", fake_request)

    result = database_client.get_analysis_predictions(
        analysis_id="id/with?reserved",
        skip=50,
        limit=50,
    )

    assert result == expected
    parsed_url = urlparse(requested[0])
    assert parsed_url.path == "/api/analyses/id%2Fwith%3Freserved/predictions"
    assert parse_qs(parsed_url.query) == {"skip": ["50"], "limit": ["50"]}


def test_saved_analysis_api_routes_proxy_to_database(monkeypatch):
    from tabpfn_nids.api import main

    history = [{"id": "analysis-1", "pcapFilename": "capture.pcap"}]
    predictions = [{"flowId": "1", "predictionLabel": "Normal"}]
    monkeypatch.setattr(main, "get_analysis_history", lambda limit: history)
    monkeypatch.setattr(
        main,
        "get_analysis_predictions",
        lambda analysis_id, skip, limit: predictions,
    )

    with TestClient(main.app) as client:
        history_response = client.get("/api/analyses?limit=25")
        predictions_response = client.get(
            "/api/analyses/analysis-1/predictions?skip=50&limit=50"
        )

    assert history_response.status_code == 200
    assert history_response.json() == history
    assert predictions_response.status_code == 200
    assert predictions_response.json() == predictions


def test_saved_analysis_api_rejects_invalid_page_size():
    from tabpfn_nids.api import main

    with TestClient(main.app) as client:
        response = client.get("/api/analyses?limit=101")

    assert response.status_code == 422


def test_persist_analysis_marks_partial_database_write_as_failed(monkeypatch):
    requests: list[tuple[str, dict]] = []

    def fake_post(path: str, payload: dict):
        requests.append((path, payload))
        if path == "/api/analyses":
            return {"analysisId": "analysis-1"}
        if path.endswith("/predictions"):
            raise RuntimeError("prediction write failed")
        return {"status": "FAILED"}

    monkeypatch.setattr(database_client, "_post", fake_post)

    with pytest.raises(RuntimeError, match="prediction write failed"):
        database_client.persist_analysis(
            pcap_filename="capture.pcap",
            output=pd.DataFrame({"prediction_label": ["Normal"]}),
            duration_seconds=1.5,
        )

    assert requests[-1] == (
        "/api/analyses/analysis-1/fail",
        {"errorMessage": "Analysis result persistence failed."},
    )
