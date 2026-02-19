import pandas as pd
from typing import Dict, Any, List
import json
import os

class RobustnessReporter:
    """
    Generates an executive summary of model robustness.
    """
    def __init__(self, experiment_dir: str):
        self.experiment_dir = experiment_dir
        self.report_path = os.path.join(experiment_dir, "robustness_report.md")
        
    def generate_report(self, metrics: Dict[str, float], adv_results: Dict[str, float], drift_results: Dict[str, float] = None):
        """
        Synthesizes metrics into a markdown report.
        """
        report = []
        report.append("# ScamDefy Model Robustness Report")
        report.append(f"**Experiment**: {os.path.basename(self.experiment_dir)}")
        report.append("")
        
        report.append("## 1. Core Performance")
        report.append("| Metric | Value |")
        report.append("| :--- | :--- |")
        for k, v in metrics.items():
            report.append(f"| {k} | {v:.4f} |")
            
        report.append("")
        report.append("## 2. Adversarial Resilience")
        report.append("Performance drop under attack conditions:")
        report.append("| Attack Type | Accuracy | Drop |")
        report.append("| :--- | :--- | :--- |")
        
        clean_acc = metrics.get('accuracy', 0.0)
        for attack, acc in adv_results.items():
            drop = clean_acc - acc
            report.append(f"| {attack} | {acc:.4f} | -{drop:.4f} |")
            
        if drift_results:
            report.append("")
            report.append("## 3. Drift Simulation")
            report.append("Performance under distribution shift:")
            report.append("| Scenario | Accuracy | Drop |")
            report.append("| :--- | :--- | :--- |")
            for scenario, acc in drift_results.items():
                drop = clean_acc - acc
                report.append(f"| {scenario} | {acc:.4f} | -{drop:.4f} |")

        report.append("")
        report.append("## 4. Conclusion")
        if clean_acc > 0.95 and all(acc > 0.85 for acc in adv_results.values()):
            report.append("**STATUS: PASSED**. Model shows high resilience against tested attacks.")
        else:
            report.append("**STATUS: WARNING**. Model shows significant vulnerability to specific attacks.")
            
        with open(self.report_path, 'w') as f:
            f.write("\n".join(report))
            
        print(f"Report generated at: {self.report_path}")

if __name__ == "__main__":
    pass
