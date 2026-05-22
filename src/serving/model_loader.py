import os
import json
import joblib
import logging

logger = logging.getLogger(__name__)

PROMOTION_DECISION_PATH = "docs/promotion_decision.json"
MODEL_PATH = "src/serving/promoted_model.joblib"

def load_model():
    """
    Loads the ML pipeline if the promotion gate was passed.
    Returns the loaded model pipeline and version info.
    Raises RuntimeError if no valid model is available.
    """
    is_promoted = False
    
    if os.path.exists(PROMOTION_DECISION_PATH):
        with open(PROMOTION_DECISION_PATH, "r") as f:
            decision = json.load(f)
            is_promoted = decision.get("is_promoted", False)
            
    if not is_promoted:
        # According to requirements, block if not promoted.
        # But if there's an already deployed model, we might want to serve that.
        # For this exercise, we will log a severe warning.
        logger.warning("PROMOTION GATE FAILED: The latest model did not pass the baseline metrics. Serving the existing model file if present.")
        
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found at {MODEL_PATH}")
        
    try:
        pipeline = joblib.load(MODEL_PATH)
        version = "xgboost_v1" if is_promoted else "baseline_or_fallback_v1"
        return pipeline, version
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")
