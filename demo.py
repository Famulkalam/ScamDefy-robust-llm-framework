import onnxruntime as ort
import numpy as np
import time
import os
from transformers import AutoTokenizer

def run_demo():
    print("="*50)
    print("🔥 ScamDefy v1.0 - Real-Time Fraud Detection Demo")
    print("="*50)

    # Paths
    model_path = "models/release_v1.0/model.onnx"
    tokenizer_path = "models/release_v1.0/model_hf"

    if not os.path.exists(model_path) or not os.path.exists(tokenizer_path):
        print("Error: Model artifacts not found. Please ensure 'models/release_v1.0' exists.")
        return

    print("Loading optimized model...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Optimized Session Options (Thread=1 for <30ms latency)
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(model_path, sess_options)

    # Test Case
    text = "Urgent: Bank fraud alert. A transaction of £2,400 query. Click https://bank-secure-verify.com/login to cancel."
    print(f"\nAnalyzing: '{text}'")

    # Benchmarking Inference
    start = time.time()
    
    # Preprocess
    inputs = tokenizer(text, return_tensors="np", max_length=128, truncation=True, padding='max_length')
    onnx_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }
    
    # Inference
    logits = session.run(None, onnx_inputs)[0]
    probs = (np.exp(logits) / np.sum(np.exp(logits), axis=1))[:, 1]
    
    end = time.time()
    latency_ms = (end - start) * 1000

    print("-" * 30)
    print(f"🛑 Scam Probability: {probs[0]:.4f}")
    print(f"⚡ Latency:          {latency_ms:.2f} ms")
    print("-" * 30)
    
    if probs[0] > 0.5:
        print("Result: BLOCKED (Scam Detected) 🛡️")
    else:
        print("Result: ALLOWED (Legitimate) ✅")

if __name__ == "__main__":
    run_demo()
