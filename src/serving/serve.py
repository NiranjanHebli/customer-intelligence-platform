import os
import sys

# Crucial: This must be set BEFORE any libraries like faiss, xgboost, or torch are imported
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['OMP_NUM_THREADS'] = '1'

from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
import uvicorn

from contextlib import asynccontextmanager
from src.rag.retrieval import get_retriever
from src.rag.generator import generate_answer

from src.serving.schemas import (
    CustomerFeatures,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    AskComplaintRequest,
    AskComplaintResponse
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
        
        return AskComplaintResponse(
            answer=answer,
            cited_evidence_ids=cited_ids,
            evidence_sufficiency_note=note,
            prompt_version=prompt_ver
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("src.serving.serve:app", host="0.0.0.0", port=8000, reload=True)
