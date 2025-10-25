"""
Custom metrics computation module.

Provides functions to calculate various performance metrics
for binary classification tasks.
"""

from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss


def compute_metrics(y_true, y_pred_proba):
    """
    Compute multiple performance metrics for binary classification.
    Args:
        y_true (array-like): True binary labels
        y_pred_proba (array-like): Predicted probabilities for positive class
    Returns:
        dict: Dictionary containing metric names and values
            - roc_auc: Area under ROC curve
            - pr_auc: Area under precision-recall curve
            - brier: Brier score (lower is better)
            - log_loss: Log loss (lower is better)
    """
    return {
        "roc_auc": roc_auc_score(y_true, y_pred_proba),
        "pr_auc": average_precision_score(y_true, y_pred_proba),
        "brier": brier_score_loss(y_true, y_pred_proba),
        "log_loss": log_loss(y_true, y_pred_proba),
    }