import pytest
from fastapi.testclient import TestClient
from src.serving.serve import app

@pytest.fixture
def client():
    with TestClient(app) as client:
        yield client

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "v1" in data["ml_model_version"]
    assert "vector_index_version" in data

def test_predict_endpoint(client):
    payload = {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "duration": 150,
        "campaign": 2,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0
    }
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert "v1" in data["model_version"]

def test_batch_score_endpoint(client):
    payload = {
        "features": [
            {
                "age": 35,
                "job": "admin.",
                "marital": "married",
                "education": "university.degree",
                "default": "no",
                "housing": "yes",
                "loan": "no",
                "contact": "cellular",
                "month": "may",
                "day_of_week": "mon",
                "duration": 150,
                "campaign": 2,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp.var.rate": 1.1,
                "cons.price.idx": 93.994,
                "cons.conf.idx": -36.4,
                "euribor3m": 4.857,
                "nr.employed": 5191.0
            },
            {
                "age": 45,
                "job": "management",
                "marital": "single",
                "education": "university.degree",
                "default": "no",
                "housing": "no",
                "loan": "no",
                "contact": "telephone",
                "month": "jun",
                "day_of_week": "tue",
                "duration": 200,
                "campaign": 1,
                "pdays": 999,
                "previous": 0,
                "poutcome": "nonexistent",
                "emp.var.rate": 1.4,
                "cons.price.idx": 94.465,
                "cons.conf.idx": -41.8,
                "euribor3m": 4.961,
                "nr.employed": 5228.1
            }
        ]
    }
    
    response = client.post("/batch-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_scored"] == 2
    
    # Check that counts sum to 2
    counts = data["conversion_counts"]
    total = sum(counts.values())
    assert total == 2
