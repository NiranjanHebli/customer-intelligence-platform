from fastapi.testclient import TestClient
from src.serving.serve import app

client = TestClient(app)

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "error_rate" in data
    assert "avg_latency_ms" in data
    assert "prediction_distribution" in data
    assert "rag_refusal_rate" in data

def test_customer_intel_endpoint():
    payload = {
        "features": {
            "age": 30,
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
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "poutcome": "nonexistent",
            "emp.var.rate": 1.1,
            "cons.price.idx": 93.994,
            "cons.conf.idx": -36.4,
            "euribor3m": 4.857,
            "nr.employed": 5191.0
        },
        "product_filter": "Checking or savings account"
    }
    
    response = client.post("/customer-intel", json=payload)
    
    # Check if the model is loaded or not in the CI environment
    if response.status_code == 503:
        assert response.json()["detail"] == "ML Model is not loaded"
    else:
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "complaint_themes_answer" in data
        assert "cited_evidence_ids" in data
        assert isinstance(data["cited_evidence_ids"], list)
