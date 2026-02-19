import os
import shutil
import yaml
import json
from datetime import datetime

def freeze_version(version_tag: str = "v1.0"):
    print("=" * 50)
    print(f"STEP 5: FREEZE VERSION {version_tag}")
    print("=" * 50)
    
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    base_dir = config['project']['output_dir']
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    latest_run = max(runs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    run_dir = os.path.join(base_dir, latest_run)
    
    # Define Release Dir
    release_dir = os.path.join("models", f"release_{version_tag}")
    os.makedirs(release_dir, exist_ok=True)
    
    print(f"Source: {run_dir}")
    print(f"Target: {release_dir}")
    
    # 1. Copy Model (HuggingFace)
    src_model = os.path.join(run_dir, "final_model")
    dst_model = os.path.join(release_dir, "model_hf")
    if os.path.exists(src_model):
        shutil.copytree(src_model, dst_model, dirs_exist_ok=True)
        print("[+] Copied PyTorch Model")
    else:
        print("[-] WARNING: PyTorch model not found!")

    # 2. Copy ONNX
    src_onnx = os.path.join(run_dir, "model.onnx")
    if os.path.exists(src_onnx):
        shutil.copy2(src_onnx, os.path.join(release_dir, "model.onnx"))
        print("[+] Copied ONNX Model")
    else:
        print("[-] WARNING: ONNX model not found!")
        
    # 3. Copy Config
    shutil.copy2("config/default.yaml", os.path.join(release_dir, "config.yaml"))
    print("[+] Archived Configuration")
    
    # 4. Copy Reports
    src_report = os.path.join(run_dir, "validation_report.json")
    if os.path.exists(src_report):
        shutil.copy2(src_report, os.path.join(release_dir, "validation_metrics.json"))
        print("[+] Archived Validation Report")
        
    src_exec_report = os.path.join(run_dir, "Executive_Fraud_Report.md")
    if os.path.exists(src_exec_report):
        shutil.copy2(src_exec_report, os.path.join(release_dir, "Executive_Summary.md"))
        print("[+] Archived Executive Summary")

    # 5. Create Manifest
    manifest = {
        "version": version_tag,
        "date": datetime.now().isoformat(),
        "source_run": latest_run,
        "components": os.listdir(release_dir)
    }
    
    with open(os.path.join(release_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
        
    print(f"\nSUCCESS: Release {version_tag} frozen at {release_dir}")

if __name__ == "__main__":
    freeze_version()
