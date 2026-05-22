import os
import json
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.data_pipeline.features import add_derived_features, build_preprocessor
from src.training.evaluate import evaluate_pipeline, run_promotion_gate

# Configurations
DATA_PATH = "data/processed/bank_marketing_validated.csv"
MODEL_OUTPUT_DIR = "models"
PROMOTION_DECISION_PATH = "docs/promotion_decision.json"
EXPERIMENT_NAME = "Meridian_Bank_Marketing"

CATEGORICAL_COLS = [
    'job', 'marital', 'education', 'default', 'housing', 'loan', 
    'contact', 'month', 'day_of_week', 'poutcome'
]
NUMERICAL_COLS = [
    'age', 'duration', 'campaign', 'pdays', 'previous', 'emp.var.rate', 
    'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed',
    'pdays_contacted', 'has_previous_contact'
]

def train_and_evaluate():
    #  Load validated dataset
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Validated data not found at {DATA_PATH}. Please run validation first.")
        
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    #  Feature Engineering
    print("Engineering features...")
    df_feat = add_derived_features(df)
    
    # Separate features and target
    X = df_feat.drop(columns=["y"])
    y = (df_feat["y"] == "yes").astype(int)
    
    #  Stratified Train-Test Split (validation split)
    print("Splitting data into train and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    #  Set up MLflow
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Build the Preprocessor
    preprocessor = build_preprocessor(CATEGORICAL_COLS, NUMERICAL_COLS)
    

    # Baseline Model: Logistic Regression
    print("\n--- Training Baseline Model (Logistic Regression) ---")
    baseline_clf = LogisticRegression(max_iter=1000, random_state=42)
    baseline_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', baseline_clf)
    ])
    
    baseline_metrics = {}
    with mlflow.start_run(run_name="baseline_model") as baseline_run:
        # Fit pipeline
        baseline_pipeline.fit(X_train, y_train)
        
        # Evaluate
        baseline_metrics, cm_path, cal_path = evaluate_pipeline(
            baseline_pipeline, X_val, y_val, output_dir="metrics/baseline"
        )
        
        # Log params, metrics, models, and artifacts
        mlflow.log_params({
            "model_type": "LogisticRegression",
            "max_iter": 1000,
            "random_state": 42
        })
        mlflow.log_metrics(baseline_metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(cal_path)
        mlflow.sklearn.log_model(baseline_pipeline, "baseline_pipeline")
        
        print("Baseline metrics logged to MLflow:")
        print(json.dumps(baseline_metrics, indent=2))
        
    # Improved Model: XGBoost Classifier
    print("\n--- Training Improved Model (XGBoost) ---")
    improved_clf = XGBClassifier(
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        max_depth=5,
        n_estimators=100
    )
    improved_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', improved_clf)
    ])
    
    improved_metrics = {}
    with mlflow.start_run(run_name="improved_model") as improved_run:
        # Fit pipeline
        improved_pipeline.fit(X_train, y_train)
        
        # Evaluate
        improved_metrics, cm_path, cal_path = evaluate_pipeline(
            improved_pipeline, X_val, y_val, output_dir="metrics/improved"
        )
        
        # Log params, metrics, models, and artifacts
        mlflow.log_params({
            "model_type": "XGBClassifier",
            "max_depth": 5,
            "n_estimators": 100,
            "random_state": 42
        })
        mlflow.log_metrics(improved_metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(cal_path)
        mlflow.sklearn.log_model(improved_pipeline, "improved_pipeline")
        
        print("Improved model metrics logged to MLflow:")
        print(json.dumps(improved_metrics, indent=2))

    # Promotion Gate Comparison
    print("\n--- Running Relative Promotion Gate ---")
    decision = run_promotion_gate(baseline_metrics, improved_metrics)
    
    os.makedirs(os.path.dirname(PROMOTION_DECISION_PATH), exist_ok=True)
    with open(PROMOTION_DECISION_PATH, "w") as f:
        json.dump(decision, f, indent=2)
    print(f"Saved promotion decision to {PROMOTION_DECISION_PATH}.")
    print(decision["reason"])
    
    # Save decision properties to MLflow active run if possible or log as run tags/artifacts
    with mlflow.start_run(run_name="promotion_gate") as gate_run:
        mlflow.log_metrics({
            "pr_auc_diff": improved_metrics["pr_auc"] - baseline_metrics["pr_auc"],
            "f1_diff": improved_metrics["f1_score"] - baseline_metrics["f1_score"]
        })
        mlflow.set_tags({
            "is_promoted": str(decision["is_promoted"]),
            "reason": decision["reason"]
        })
        mlflow.log_artifact(PROMOTION_DECISION_PATH)

    #  Persist the promoted model for serving
    if decision["is_promoted"]:
        os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
        promoted_model_path = os.path.join(MODEL_OUTPUT_DIR, "promoted_model.joblib")
        joblib.dump(improved_pipeline, promoted_model_path)
        print(f"Successfully promoted and saved model to {promoted_model_path}.")
    else:
        print("Improved model did not meet promotion criteria. Sticking to baseline (no promoted model updated).")

if __name__ == "__main__":
    train_and_evaluate()
