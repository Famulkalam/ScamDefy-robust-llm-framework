import json
import os
import time
import yaml
from typing import Dict, Any

class ExperimentTracker:
    """
    Lightweight experiment tracking system.
    Logs metrics, hyperparameters, and artifacts to a JSON file.
    Compatible structure for future MLflow integration.
    """
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        self.experiment_name = experiment_name
        self.config = config
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.run_id = f"{experiment_name}_{self.timestamp}"
        
        # Setup directories
        self.output_dir = os.path.join(config.get('project', {}).get('output_dir', 'experiments'), self.run_id)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save config snapshot
        self.save_config()
        
        # Initialize metrics log
        self.metrics_file = os.path.join(self.output_dir, "metrics.json")
        self.metrics_history = []
        
    def save_config(self):
        """Saves the configuration state for reproducibility."""
        config_path = os.path.join(self.output_dir, "config_snapshot.yaml")
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f)
            
    def log_metric(self, key: str, value: float, step: int = None):
        """Logs a single metric."""
        entry = {
            "key": key,
            "value": value,
            "step": step,
            "timestamp": time.time()
        }
        self.metrics_history.append(entry)
        self._flush()
        
    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """Logs multiple metrics at once."""
        for k, v in metrics.items():
            self.log_metric(k, v, step)
            
    def _flush(self):
        """Writes metrics to disk."""
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics_history, f, indent=2)
            
    def get_artifact_path(self, filename: str) -> str:
        """Returns a path to save an artifact (model, plot, etc.)."""
        return os.path.join(self.output_dir, filename)

    def log_artifact(self, source_path: str):
        """
        In a real MLflow setup, this would upload the file.
        Here, we just ensure it's in the output directory (already handled by get_artifact_path usage).
        """
        pass
