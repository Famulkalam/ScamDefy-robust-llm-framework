from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Any
import numpy as np

def compute_standard_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> Dict[str, float]:
    """
    Computes standard classification metrics.
    """
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    acc = accuracy_score(y_true, y_pred)
    
    metrics = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    
    if y_probs is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_probs)
        except ValueError:
            metrics["roc_auc"] = 0.0 # Handle case where only one class is present
            
    return metrics

def compute_imbalance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes metrics relevant for imbalanced datasets.
    """
    # False Positive Rate: FP / (TN + FP)
    # Predicted 1, Actual 0
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    
    fpr = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return {
        "fpr": fpr,
        "false_positives": int(fp),
        "true_negatives": int(tn)
    }
