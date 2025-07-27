import pytest
from fastapi.testclient import TestClient
from src.predict.app import app
import numpy as np

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

def test_valid_prediction():
    test_data = {
        "AF3": 4329.23,
        "F7": 4009.23,
        "F3": 4289.23,
        "FC5": 4148.21,
        "T7": 4350.26,
        "P7": 4586.15,
        "O1": 4096.00,
        "O2": 4129.23,
        "P8": 4356.41,
        "T8": 4216.41,
        "FC6": 4088.97,
        "F4": 4273.85,
        "F8": 4148.72,
        "AF4": 4163.08
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    assert "eye_state" in response.json()
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], int)
    assert response.json()["eye_state"] in ["Open", "Closed"]

def test_invalid_prediction():
    test_data = {
        "AF3": np.inf,  # Invalid value
        "F7": 4009.23,
        "F3": 4289.23,
        "FC5": 4148.21,
        "T7": 4350.26,
        "P7": 4586.15,
        "O1": 4096.00,
        "O2": 4129.23,
        "P8": 4356.41,
        "T8": 4216.41,
        "FC6": 4088.97,
        "F4": 4273.85,
        "F8": 4148.72,
        "AF4": 4163.08
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 422

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    metrics = response.json()
    assert "total_predictions" in metrics
    assert "avg_prediction_time" in metrics
    assert "feature_statistics" in metrics
