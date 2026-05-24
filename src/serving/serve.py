import os
import sys

# Crucial: This must be set BEFORE any libraries like faiss, xgboost, or torch are imported
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import pandas as pd
import uvicorn
import time

from contextlib import asynccontextmanager
from src.rag.retrieval import get_retriever
from src.rag.generator import generate_answer

from src.serving.schemas import (
    CustomerFeatures,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    AskComplaintRequest,
    AskComplaintResponse,
    CustomerIntelRequest,
    CustomerIntelResponse,
    MetricsResponse
)
from src.data_pipeline.features import add_derived_features
from src.serving.model_loader import load_model

load_dotenv()

# Global references
try:
    ml_pipeline, model_ver = load_model()
except RuntimeError as e:
    ml_pipeline = None
    model_ver = f"failed_to_load: {str(e)}"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Model is already loaded globally to avoid macOS asyncio thread deadlocks
    yield
    # We do NOT set ml_pipeline = None on shutdown to prevent segfaults during sequential Pytest runs.


app = FastAPI(
    title="Customer Intelligence Platform API",
    description="API for scoring term-deposit subscriptions and RAG complaints",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

metrics_store = {
    "total_requests": 0,
    "total_errors": 0,
    "total_latency_ms": 0.0,
    "prediction_distribution": {0: 0, 1: 0},
    "rag_requests": 0,
    "rag_refusals": 0,
}

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        metrics_store["total_requests"] += 1
        if response.status_code >= 400:
            metrics_store["total_errors"] += 1
        return response
    except Exception as e:
        metrics_store["total_requests"] += 1
        metrics_store["total_errors"] += 1
        raise e
    finally:
        latency_ms = (time.time() - start_time) * 1000
        metrics_store["total_latency_ms"] += latency_ms

@app.get("/health")
def health_check() -> Dict[str, Any]:
    return {
        "status": "ok",
        "ml_model_version": model_ver,
        "vector_index_version": "faiss_v1"
    }

@app.post("/predict", response_model=PredictResponse)
def predict(features: CustomerFeatures):
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="ML Model is not loaded")
        
    df = pd.DataFrame([features.model_dump(by_alias=True)])
    df = add_derived_features(df)
    
    try:
        prob = float(ml_pipeline.predict_proba(df)[:, 1][0])
        threshold = 0.5
        prediction = 1 if prob >= threshold else 0
        
        metrics_store["prediction_distribution"][prediction] += 1
        
        return PredictResponse(
            prediction=prediction,
            probability=prob,
            threshold_decision=f">={threshold}",
            model_version=model_ver
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/batch-score", response_model=BatchPredictResponse)
def batch_score(request: BatchPredictRequest):
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="ML Model is not loaded")
        
    df = pd.DataFrame([f.model_dump(by_alias=True) for f in request.features])
    df = add_derived_features(df)
    
    try:
        preds = ml_pipeline.predict(df)
        
        counts = {
            0: int((preds == 0).sum()),
            1: int((preds == 1).sum())
        }
        
        return BatchPredictResponse(
            total_scored=len(preds),
            conversion_counts=counts,
            model_version=model_ver
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ask-complaints", response_model=AskComplaintResponse)
def ask_complaints(request: AskComplaintRequest):
    try:
        retriever = get_retriever()
        chunks, is_weak = retriever.retrieve(
            query=request.question,
            top_k=3,
            filter_product=request.filter_product
        )
        
        answer, cited_ids, note, prompt_ver = generate_answer(
            question=request.question,
            retrieved_chunks=chunks,
            is_weak=is_weak
        )
        
        metrics_store["rag_requests"] += 1
        refusal_keywords = ["cannot answer", "cannot provide", "do not have", "cannot find"]
        if any(k in answer.lower() for k in refusal_keywords) or len(cited_ids) == 0:
            metrics_store["rag_refusals"] += 1
        
        return AskComplaintResponse(
            answer=answer,
            cited_evidence_ids=cited_ids,
            evidence_sufficiency_note=note,
            prompt_version=prompt_ver
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics", response_model=MetricsResponse)
def get_metrics():
    total = metrics_store["total_requests"]
    error_rate = metrics_store["total_errors"] / total if total > 0 else 0.0
    avg_latency = metrics_store["total_latency_ms"] / total if total > 0 else 0.0
    rag_total = metrics_store["rag_requests"]
    refusal_rate = metrics_store["rag_refusals"] / rag_total if rag_total > 0 else 0.0
    
    return MetricsResponse(
        total_requests=total,
        error_rate=error_rate,
        avg_latency_ms=avg_latency,
        prediction_distribution=metrics_store["prediction_distribution"],
        rag_refusal_rate=refusal_rate
    )

@app.post("/customer-intel", response_model=CustomerIntelResponse)
def customer_intel(request: CustomerIntelRequest):
    if ml_pipeline is None:
        raise HTTPException(status_code=503, detail="ML Model is not loaded")
        
    df = pd.DataFrame([request.features.model_dump(by_alias=True)])
    df = add_derived_features(df)
    
    try:
        prob = float(ml_pipeline.predict_proba(df)[:, 1][0])
        prediction = 1 if prob >= 0.5 else 0
        metrics_store["prediction_distribution"][prediction] += 1
        
        query = "What are the most common complaint themes?"
        if request.product_filter:
            query = f"What are the most common complaint themes for {request.product_filter}?"
            
        retriever = get_retriever()
        chunks, is_weak = retriever.retrieve(
            query=query,
            top_k=3,
            filter_product=request.product_filter
        )
        answer, cited_ids, note, prompt_ver = generate_answer(
            question=query,
            retrieved_chunks=chunks,
            is_weak=is_weak
        )
        
        metrics_store["rag_requests"] += 1
        refusal_keywords = ["cannot answer", "cannot provide", "do not have", "cannot find"]
        if any(k in answer.lower() for k in refusal_keywords) or len(cited_ids) == 0:
            metrics_store["rag_refusals"] += 1
            
        return CustomerIntelResponse(
            prediction=prediction,
            probability=prob,
            complaint_themes_answer=answer,
            cited_evidence_ids=cited_ids
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.serving.serve:app", host="0.0.0.0", port=8000, reload=True)
