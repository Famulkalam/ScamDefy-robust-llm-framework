import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve, f1_score
from typing import Tuple, Dict

def compute_ece(y_true: np.ndarray, y_probs: np.ndarray, n_bins: int = 10) -> float:
    """
    Computes Expected Calibration Error (ECE).
    """
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=n_bins, strategy='uniform')
    
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        # Find samples in this bin
        bin_mask = (y_probs > bin_edges[i]) & (y_probs <= bin_edges[i+1])
        n_samples = np.sum(bin_mask)
        if n_samples > 0:
            avg_conf = np.mean(y_probs[bin_mask])
            avg_acc = np.mean(y_true[bin_mask])
            ece += np.abs(avg_acc - avg_conf) * (n_samples / len(y_probs))
            
    return ece

def plot_reliability_diagram(y_true: np.ndarray, y_probs: np.ndarray, output_path: str):
    """
    Generates and saves a reliability diagram.
    """
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='.', label='Model')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()

def find_optimal_threshold(y_true: np.ndarray, y_probs: np.ndarray, policy: str = "f1") -> Tuple[float, Dict[str, float]]:
    """
    Finds optimal threshold based on policy.
    policy: 'f1' (maximize F1), 'conservative' (max precision), 'aggressive' (max recall)
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    
    best_thresh = 0.5
    best_score = 0.0
    
    if policy == "f1":
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        idx = np.argmax(f1_scores)
        best_thresh = thresholds[idx]
        best_score = f1_scores[idx]
        
    elif policy == "conservative":
        # Target: Precision >= 0.98 (Minimize False Positives)
        for i, prec in enumerate(precisions):
            if prec >= 0.98:
                best_thresh = thresholds[i]
                break # First threshold that meets precision requirement
        
    elif policy == "aggressive":
        # Target: Recall >= 0.99 (Catch all fraud)
        # Iterate backwards
        for i in range(len(recalls)-1, -1, -1):
            if recalls[i] >= 0.99:
                best_thresh = thresholds[i]
                best_score = recalls[i]
                break

    return float(best_thresh), {"score": float(best_score)}
