import os
import pytest
from fastapi.testclient import TestClient
from src.serving.serve import app

client = TestClient(app)

# Tuples of (question, is_in_domain)
EVALUATION_QUESTIONS = [
    ("Why did I get charged an overdraft fee when I had money?", True),
    ("Are there issues with credit reporting agencies putting wrong info?", True),
    ("How do I bake a chocolate cake?", False), # Out of domain
    ("My mortgage loan was transferred and the new company lost my payment.", True),
    ("What are the best movies to watch in 2024?", False), # Out of domain
    ("Someone opened a credit card in my name, identity theft.", True),
    ("Why are debt collectors calling me for a debt I don't owe?", True),
    ("What is the recipe for scrambled eggs?", False), # Out of domain
    ("The bank closed my account without warning and kept my money.", True),
    ("Why was my loan application denied despite good credit?", True)
]

@pytest.mark.parametrize("question, is_domain", EVALUATION_QUESTIONS)
def test_ask_complaints_eval(question, is_domain):
    response = client.post("/ask-complaints", json={"question": question})
    assert response.status_code == 200
    data = response.json()
    
    assert "answer" in data
    assert "cited_evidence_ids" in data
    assert "evidence_sufficiency_note" in data
    
    if not is_domain:
        # Out-of-domain queries should trigger a refusal or have zero citations
        answer_lower = data["answer"].lower()
        refusal_keywords = ["cannot answer", "cannot provide", "do not have", "cannot find"]
        assert any(k in answer_lower for k in refusal_keywords) or len(data["cited_evidence_ids"]) == 0
    else:
        # Domain queries should successfully return a structured response
        assert isinstance(data["cited_evidence_ids"], list)
        assert isinstance(data["answer"], str)
