import time
import torch
import onnxruntime as ort
import numpy as np
import psutil
import os
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def benchmark_inference():
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    # Find Model (same logic as export)
    base_dir = config['project']['output_dir']
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    latest_run = max(runs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    run_dir = os.path.join(base_dir, latest_run)
    
    if os.path.exists(os.path.join(run_dir, "final_model")):
         model_dir = os.path.join(run_dir, "final_model")
    else:
         checkpoints = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")]
         if checkpoints:
             latest_ckpt = max(checkpoints, key=lambda x: int(x.split('-')[1]))
             model_dir = os.path.join(run_dir, latest_ckpt)
         else:
             print("Model not found")
             return

    onnx_path = os.path.join(base_dir, latest_run, "model.onnx")
    
    # Load PyTorch
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pt_model = AutoModelForSequenceClassification.from_pretrained(model_dir).to("cpu")
    pt_model.eval()
    
    # Load ONNX
    ort_session = ort.InferenceSession(onnx_path)
    
    # Data
    text = "URGENT request: verify your bank account immediately."
    inputs = tokenizer(text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    pt_inputs = inputs
    onnx_inputs = {
        'input_ids': inputs['input_ids'].numpy(), 
        'attention_mask': inputs['attention_mask'].numpy()
    }
    
    n_iters = 100
    
    # --- PyTorch Benchmark ---
    latencies_pt = []
    with torch.no_grad():
        for _ in range(n_iters):
            start = time.perf_counter()
            _ = pt_model(**pt_inputs)
            latencies_pt.append(time.perf_counter() - start)
            
    # --- ONNX Benchmark ---
    latencies_onnx = []
    for _ in range(n_iters):
        start = time.perf_counter()
        _ = ort_session.run(None, onnx_inputs)
        latencies_onnx.append(time.perf_counter() - start)
        
    # Metrics
    def get_stats(latencies):
        return {
            "p50": np.percentile(latencies, 50) * 1000,
            "p95": np.percentile(latencies, 95) * 1000,
            "p99": np.percentile(latencies, 99) * 1000,
            "throughput": 1.0 / np.mean(latencies)
        }
        
    stats_pt = get_stats(latencies_pt)
    stats_onnx = get_stats(latencies_onnx)
    
    print("-" * 30)
    print("Deployment Benchmark Report (CPU, Batch=1)")
    print("-" * 30)
    print(f"PyTorch P95 Latency: {stats_pt['p95']:.2f} ms")
    print(f"ONNX    P95 Latency: {stats_onnx['p95']:.2f} ms")
    print(f"Speedup: {stats_pt['p95'] / stats_onnx['p95']:.2f}x")
    print("-" * 30)
    
    return stats_pt, stats_onnx

if __name__ == "__main__":
    benchmark_inference()
