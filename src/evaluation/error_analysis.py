import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple

class ErrorAnalyzer:
    """
    Analyzes model mistakes to provide actionable insights for fraud analysts.
    """
    def __init__(self, texts: List[str], y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray):
        self.df = pd.DataFrame({
            'text': texts,
            'true_label': y_true,
            'pred_label': y_pred,
            'prob_scam': y_probs
        })
        
        self.false_positives = self.df[(self.df['true_label'] == 0) & (self.df['pred_label'] == 1)]
        self.false_negatives = self.df[(self.df['true_label'] == 1) & (self.df['pred_label'] == 0)]
        
    def get_top_errors(self, n: int = 20) -> Dict[str, pd.DataFrame]:
        """
        Returns the top 'n' most confident mistakes.
        FP: Highest probability of being scam (but is legit).
        FN: Lowest probability of being scam (but is scam).
        """
        top_fp = self.false_positives.sort_values(by='prob_scam', ascending=False).head(n)
        top_fn = self.false_negatives.sort_values(by='prob_scam', ascending=True).head(n)
        
        return {
            "top_false_positives": top_fp,
            "top_false_negatives": top_fn
        }

    def analyze_token_frequency(self, error_type: str = "fp", top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds common words in misclassified samples.
        """
        if error_type == "fp":
            texts = self.false_positives['text'].tolist()
        else:
            texts = self.false_negatives['text'].tolist()
            
        all_words = " ".join(texts).lower().split()
        # Basic filtering of extremely common stop words could be added here
        # For now, raw frequency is often insightful (e.g., 'verify' in legit emails)
        return Counter(all_words).most_common(top_n)

    def get_confidence_distribution(self) -> Dict[str, Dict[str, float]]:
        """
        Returns stats about confidence scores for errors.
        """
        return {
            "false_positives": {
                "mean_conf": float(self.false_positives['prob_scam'].mean()) if len(self.false_positives) > 0 else 0.0,
                "std_conf": float(self.false_positives['prob_scam'].std()) if len(self.false_positives) > 0 else 0.0
            },
            "false_negatives": {
                "mean_conf": float(self.false_negatives['prob_scam'].mean()) if len(self.false_negatives) > 0 else 0.0,
                "std_conf": float(self.false_negatives['prob_scam'].std()) if len(self.false_negatives) > 0 else 0.0
            }
        }
