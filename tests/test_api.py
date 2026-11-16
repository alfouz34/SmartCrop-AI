import pytest
from fastapi.testclient import TestClient
from backend.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c

def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "SmartCrop AI"
    assert data["model_loaded"] is True

def test_predict_valid_inputs(client):
    # Rice-like profile
    payload = {
        "N": 90.0,
        "P": 42.0,
        "K": 43.0,
        "temperature": 20.88,
        "humidity": 82.0,
        "ph": 6.5,
        "rainfall": 202.94
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "recommendations" in data
    assert len(data["recommendations"]) == 3
    assert data["recommendations"][0]["crop"] == "rice"
    assert data["recommendations"][0]["score"] > 0.5
    assert "note" in data

def test_predict_apple_profile(client):
    # Apple-like profile (high P and K)
    payload = {
        "N": 20.0,
        "P": 134.0,
        "K": 200.0,
        "temperature": 22.0,
        "humidity": 92.0,
        "ph": 6.0,
        "rainfall": 112.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["recommendations"][0]["crop"] == "apple"

def test_predict_invalid_input_ph_out_of_range(client):
    # pH must be between 0 and 14
    payload = {
        "N": 90.0, "P": 42.0, "K": 43.0,
        "temperature": 25.0, "humidity": 80.0,
        "ph": -1.0, "rainfall": 100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422

def test_predict_invalid_input_missing_field(client):
    payload = {
        "P": 42.0, "K": 43.0,
        "temperature": 25.0, "humidity": 80.0,
        "ph": 6.5, "rainfall": 100.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
