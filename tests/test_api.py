import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from api.app import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_predict_valid_input(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1
    assert 0 <= body["probability_disease"] <= 1


def test_predict_missing_field_returns_422(client, sample_patient):
    sample_patient.pop("age")
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_predict_out_of_range_returns_422(client, sample_patient):
    sample_patient["cp"] = 9
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_metrics_endpoint_counts_requests(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "api_requests_total" in text
    assert "api_predictions_total" in text
