from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import pandas as pd
import uvicorn
from contextlib import asynccontextmanager

from src.serving.schemas import (
    CustomerFeatures,
    PredictResponse,
    BatchPredictRequest,
    BatchPredictResponse
)
from src.serving.model_loader import load_model

# Global references
ml_pipeline = None
model_ver = "unknown"

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model on startup
    global ml_pipeline, model_ver
    try:
        ml_pipeline, model_ver = load_model()
    except RuntimeError as e:
        # We can either fail to start, or start and let endpoints fail.
        # Starting with endpoints failing is often better for liveness probes.
        ml_pipeline = None
        model_ver = f"failed_to_load: {str(e)}"
    yield
    # Clean up on shutdown
    ml_pipeline = None


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
        "vector_index_version": "not_implemented_yet" # Will be updated in Step 4
    }

from src.data_pipeline.features import add_derived_features

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

if __name__ == "__main__":
    uvicorn.run("src.serving.serve:app", host="0.0.0.0", port=8000, reload=True)
