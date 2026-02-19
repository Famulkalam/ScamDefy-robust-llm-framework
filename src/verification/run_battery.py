import torch
import onnx
import onnxruntime as ort
import os
import yaml
import numpy as np
import pandas as pd
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.evaluation.metrics import compute_standard_metrics, compute_imbalance_metrics
from src.adversarial.generator import AdversarialGenerator
from src.adversarial.attacks import AdversarialAttacks

def run_full_validation():
    print("=" * 50)
    print("STEP 2: FULL VALIDATION BATTERY")
    print("=" * 50)
    
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Locate Model
    base_dir = config['project']['output_dir']
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    latest_run = max(runs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    run_dir = os.path.join(base_dir, latest_run)
    
    if os.path.exists(os.path.join(run_dir, "final_model")):
         model_path = os.path.join(run_dir, "final_model")
    else:
         checkpoints = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")]
         if checkpoints:
             latest_ckpt = max(checkpoints, key=lambda x: int(x.split('-')[1]))
             model_path = os.path.join(run_dir, latest_ckpt)
         else:
             print("Model not found")
             return

    onnx_path = os.path.join(base_dir, latest_run, "model.onnx")
    
    print(f"Validating Model: {model_path}")
    
    # Load PyTorch Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_path)
    pt_model.eval()
    pt_model.to("cpu")
    
    # Load ONNX Session
    ort_session = ort.InferenceSession(onnx_path)
    
    # --- A. Final Evaluation Report (Data Metrics) ---
    print("\n[A] Generating Final Metrics Table...")
    processed_dir = config['data']['processed_path']
    test_balanced = pd.read_parquet(os.path.join(processed_dir, "test_balanced.parquet"))
    test_imbalanced = pd.read_parquet(os.path.join(processed_dir, "test_imbalanced.parquet"))
    
    def predict(texts):
        inputs = tokenizer(texts, return_tensors="pt", max_length=128, padding=True, truncation=True)
        with torch.no_grad():
            logits = pt_model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            preds = torch.argmax(logits, dim=1).numpy()
        return preds, probs
        
    # Clean F1 (Balanced)
    preds_bal, probs_bal = predict(test_balanced['text'].tolist())
    metrics_bal = compute_standard_metrics(test_balanced['label'].to_numpy(), preds_bal)
    
    # 99:1 Imbalanced Metrics
    preds_imb, probs_imb = predict(test_imbalanced['text'].tolist())
    metrics_imb = compute_standard_metrics(test_imbalanced['label'].to_numpy(), preds_imb)
    metrics_imb_extra = compute_imbalance_metrics(test_imbalanced['label'].to_numpy(), preds_imb)
    
    print(f"Clean F1 (Balanced): {metrics_bal['f1']:.4f}")
    print(f"Clean F1 (99:1 Imb): {metrics_imb['f1']:.4f}")
    print(f"FPR (99:1):          {metrics_imb_extra['fpr']:.4f}")
    
    # --- B. Zero-Day Evasion Test ---
    print("\n[B] Zero-Day Evasion Test (Bank Variant)")
    zero_day_text = "Urgent: Bank fraud alert. A transaction of £2,400 query. Click https://bank-secure-verify.com/login to cancel."
    
    # PyTorch
    inputs = tokenizer(zero_day_text, return_tensors="pt", max_length=128, truncation=True, padding='max_length')
    with torch.no_grad():
        pt_prob = torch.softmax(pt_model(**inputs).logits, dim=1)[0, 1].item()
        
    # ONNX
    onnx_inputs = {
        'input_ids': inputs['input_ids'].numpy(),
        'attention_mask': inputs['attention_mask'].numpy()
    }
    onnx_prob = ort_session.run(None, onnx_inputs)[0][0, 1]
    # Apply softmax if logits returned (depends on export, usually logits)
    def softmax(x): return np.exp(x) / np.sum(np.exp(x))
    onnx_prob = softmax(ort_session.run(None, onnx_inputs)[0][0])[1]

    print(f"Input: {zero_day_text}")
    print(f"PyTorch Prob (Scam): {pt_prob:.4f}")
    print(f"ONNX    Prob (Scam): {onnx_prob:.4f}")
    if abs(pt_prob - onnx_prob) > 1e-3:
        print("WARNING: Divergence between PyTorch and ONNX!")
    
    # --- C. False Positive Trap ---
    print("\n[C] False Positive Trap")
    legit_urgent = "Hi honey, I need you to transfer the rent money today, it's urgent! Love you."
    inputs = tokenizer(legit_urgent, return_tensors="pt", max_length=128, truncation=True)
    with torch.no_grad():
        fp_prob = torch.softmax(pt_model(**inputs).logits, dim=1)[0, 1].item()
    print(f"Input: {legit_urgent}")
    print(f"Prob (Scam): {fp_prob:.4f}")
    print(f"Decision: {'SCAM (FP)' if fp_prob > 0.5 else 'LEGIT (Pass)'}")
    
    # --- D. Combined Adversarial Escalation (L0-L4) ---
    print("\n[D] Adversarial Escalation (L0-L4)")
    base_text = "Your account has been compromised. Verify now."
    attacks = {
        "L0 (Clean)": base_text,
        "L1 (Homoglyph)": AdversarialAttacks.insert_homoglyphs(base_text, 0.3),
        "L2 (Leetspeak)": AdversarialAttacks.perturb_characters(base_text, 0.3),
        "L3 (Combined)": AdversarialAttacks.perturb_characters(AdversarialAttacks.insert_homoglyphs(base_text, 0.3), 0.3),
        "L5 (Extreme)": AdversarialAttacks.inject_filler(
            AdversarialAttacks.mask_urls(
                AdversarialAttacks.random_casing(
                   AdversarialAttacks.perturb_characters(AdversarialAttacks.insert_homoglyphs(base_text, 0.3), 0.3),
                   0.4
                ), 
                0.5
            ), 
            0.4
        )
    }
    
    results = []
    for level, text in attacks.items():
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        with torch.no_grad():
            prob = torch.softmax(pt_model(**inputs).logits, dim=1)[0, 1].item()
        results.append({"Level": level, "Text": text, "Prob": prob})
        print(f"{level}: {prob:.4f} | '{text}'")
        
    # Save Report
    report = {
        "metrics_balanced": metrics_bal,
        "metrics_imbalanced": {**metrics_imb, **metrics_imb_extra},
        "zero_day_test": {"text": zero_day_text, "pt_prob": pt_prob, "onnx_prob": float(onnx_prob)},
        "fp_trap": {"text": legit_urgent, "prob": fp_prob},
        "adversarial_ladder": results
    }
    
    with open(os.path.join(base_dir, latest_run, "validation_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nValidation Report Saved: {os.path.join(base_dir, latest_run, 'validation_report.json')}")

if __name__ == "__main__":
    run_full_validation()
