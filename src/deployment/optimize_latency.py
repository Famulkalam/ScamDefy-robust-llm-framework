import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import os
import yaml
import time
import numpy as np
from transformers import AutoTokenizer

def optimize_latency():
    print("=" * 50)
    print("PHASE 5: LATENCY OPTIMIZATION")
    print("=" * 50)
    
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    base_dir = config['project']['output_dir']
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    latest_run = max(runs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    run_dir = os.path.join(base_dir, latest_run)
    input_model_path = os.path.join(run_dir, "model.onnx")
    output_model_path = os.path.join(run_dir, "model_quantized.onnx")
    
    if not os.path.exists(input_model_path):
        print("Error: ONNX model not found. Run export first.")
        return

    print(f"Optimizing: {input_model_path}")
    
    # 1. Dynamic Quantization (Int8)
    print("Applying Dynamic Quantization (Int8)...")
    try:
        quantize_dynamic(
            input_model_path,
            output_model_path,
            weight_type=QuantType.QUInt8
        )
        print(f"Quantized Model Saved: {output_model_path}")
        use_quant = True
    except Exception as e:
        print(f"Quantization Failed: {e}")
        print("Proceeding with Base Model optimization (Thread Tuning)")
        use_quant = False
    
    # 2. Benchmark Comparison
    print("\nBenchmarking Optimization...")
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(run_dir, "final_model"))
    text = "Urgent: Verify your account now at http://bit.ly/secure"
    # Ensure standard length
    inputs = tokenizer(text, return_tensors="np", max_length=128, truncation=True, padding='max_length')
    onnx_inputs = {
        'input_ids': inputs['input_ids'].astype(np.int64),
        'attention_mask': inputs['attention_mask'].astype(np.int64)
    }
    
    # Session Options for Speed
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1  # Often faster for single batch on CPU
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    def benchmark(session, name, n_iters=100):
        # Warmup
        for _ in range(10): session.run(None, onnx_inputs)
        
        start = time.time()
        for _ in range(n_iters):
            session.run(None, onnx_inputs)
        end = time.time()
        avg_lat = (end - start) / n_iters * 1000
        print(f"{name}: {avg_lat:.2f} ms")
        return avg_lat

    # Original
    session_base = ort.InferenceSession(input_model_path, sess_options)
    base_lat = benchmark(session_base, "Original ONNX (Thread=1, Opt=ALL)")

    if use_quant:
        # Quantized
        session_quant = ort.InferenceSession(output_model_path, sess_options)
        quant_lat = benchmark(session_quant, "Quantized ONNX")
        final_lat = quant_lat
        improvement = (base_lat - quant_lat) / base_lat * 100
        print(f"\nImprovement: {improvement:.1f}%")
    else:
        final_lat = base_lat
    
    print(f"Final Latency: {final_lat:.2f} ms")
    
    if final_lat < 150:
        print("🎉 GOAL ACHIEVED: Latency < 150ms")
    else:
        print("⚠️ GOAL MISSED: Consider reducing sequence length or pruning.")

if __name__ == "__main__":
    optimize_latency()
