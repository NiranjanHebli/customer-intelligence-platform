import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve, CalibrationDisplay

def evaluate_pipeline(pipeline, X, y, output_dir: str):
    """
    Evaluates a scikit-learn pipeline, computes metrics, and saves evaluation plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Predict probabilities and class labels
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = pipeline.predict(X)
    
    # Calculate metrics
    roc_auc = float(roc_auc_score(y, y_prob))
    pr_auc = float(average_precision_score(y, y_prob))
    f1 = float(f1_score(y, y_pred))
    
    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1_score": f1
    }
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    disp_cm = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_cm.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(cm_path)
    plt.close()
    
    # Plot Calibration Curve
    prob_true, prob_pred = calibration_curve(y, y_prob, n_bins=10)
    disp_cal = CalibrationDisplay(prob_true, prob_pred, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp_cal.plot(ax=ax)
    ax.set_title("Calibration Curve")
    cal_path = os.path.join(output_dir, "calibration_curve.png")
    plt.tight_layout()
    plt.savefig(cal_path)
    plt.close()
    
    return metrics, cm_path, cal_path


def run_promotion_gate(
    baseline_metrics: dict, 
    improved_metrics: dict, 
    min_pr_auc_gain: float = 0.02, 
    max_f1_drop: float = 0.02
) -> dict:
    """
    Executes the relative promotion gate logic comparing the improved model against the baseline.
    """
    pr_auc_diff = improved_metrics["pr_auc"] - baseline_metrics["pr_auc"]
    f1_diff = improved_metrics["f1_score"] - baseline_metrics["f1_score"]
    
    pr_auc_passed = pr_auc_diff >= min_pr_auc_gain
    f1_passed = f1_diff >= -max_f1_drop  # F1 drop must be less than max_f1_drop (i.e. diff >= -0.02)
    
    is_promoted = pr_auc_passed and f1_passed
    
    decision = {
        "is_promoted": is_promoted,
        "baseline_metrics": baseline_metrics,
        "improved_metrics": improved_metrics,
        "checks": {
            "pr_auc_gain_check": {
                "baseline": baseline_metrics["pr_auc"],
                "improved": improved_metrics["pr_auc"],
                "difference": pr_auc_diff,
                "required_gain": min_pr_auc_gain,
                "passed": bool(pr_auc_passed)
            },
            "f1_drop_check": {
                "baseline": baseline_metrics["f1_score"],
                "improved": improved_metrics["f1_score"],
                "difference": f1_diff,
                "allowed_drop": max_f1_drop,
                "passed": bool(f1_passed)
            }
        },
        "reason": (
            "Model promoted. Improved model beat baseline PR-AUC by >= 2% and preserved F1-score."
            if is_promoted else
            f"Model promotion rejected. Reasons: PR-AUC passed: {pr_auc_passed}, F1-score preserved: {f1_passed}."
        )
    }
    
    return decision
