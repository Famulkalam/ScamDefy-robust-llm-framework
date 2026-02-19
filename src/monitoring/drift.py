import pandas as pd
import numpy as np
from typing import Dict, List
import copy

class DriftSimulator:
    """
    Simulates concept drift and data drift to stress-test the model.
    """
    def __init__(self, test_df: pd.DataFrame):
        self.original_df = test_df.copy()
        
    def simulate_vocabulary_shift(self, new_lexicon: Dict[str, str], ratio: float = 0.3) -> pd.DataFrame:
        """
        Replaces common words with new 'slang' or unseen scam terms.
        E.g., "bank" -> "banc", "verify" -> "confirm_now"
        impact: Simulate evolution of scam tactics.
        """
        drifted_df = self.original_df.copy()
        
        def replace_words(text):
            for old, new in new_lexicon.items():
                # Naive replacement for simulation speed
                if old in text and np.random.random() < ratio:
                    text = text.replace(old, new)
            return text
            
        drifted_df['text'] = drifted_df['text'].apply(replace_words)
        drifted_df['source'] = 'drift_vocabulary'
        return drifted_df

    def simulate_prior_shift(self, scam_ratio: float = 0.2) -> pd.DataFrame:
        """
        Changes the class balance drastically.
        Simulates a targeted attack wave (scam spike).
        """
        scams = self.original_df[self.original_df['label'] == 1]
        legit = self.original_df[self.original_df['label'] == 0]
        
        if len(scams) == 0: return self.original_df
        
        # Calculate needed sizes
        # n_scam / (n_scam + n_legit) = ratio
        # Let's fix n_legit and upsample n_scam
        n_legit = len(legit)
        n_scam_needed = int(n_legit * scam_ratio / (1 - scam_ratio))
        
        scams_upsampled = scams.sample(n=n_scam_needed, replace=True, random_state=42)
        
        return pd.concat([legit, scams_upsampled]).sample(frac=1).reset_index(drop=True)

if __name__ == "__main__":
    # Example usage
    pass
