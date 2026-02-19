import torch
import os
import onnx
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def export_model_to_onnx():
    # 1. Load Config
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    model_path = os.path.join(config['project']['output_dir'], "distilroberta_finetune_latest/final_model")
    # Fallback to searching if exact path varies by run ID, but for now assume latest symlink or known path
    # In a real run, we'd pass the specific run ID. For this script, let's assume valid path provided or find latest.
    if not os.path.exists(model_path):
        # Attempt to find the latest run directory
        base_dir = config['project']['output_dir']
        runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        if not runs:
            print("No trained model found.")
            return
        latest_run = max(runs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
        run_dir = os.path.join(base_dir, latest_run)
        
        # Check for final_model or checkpoints
        if os.path.exists(os.path.join(run_dir, "final_model")):
             model_path = os.path.join(run_dir, "final_model")
        else:
             # Find checkpoints
             checkpoints = [d for d in os.listdir(run_dir) if d.startswith("checkpoint-")]
             if checkpoints:
                 latest_ckpt = max(checkpoints, key=lambda x: int(x.split('-')[1]))
                 model_path = os.path.join(run_dir, latest_ckpt)
             else:
                 print(f"No model found in {run_dir}")
                 return
    
    print(f"Exporting model from: {model_path}")
    
    # 2. Load PyTorch Model
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    model.to("cpu")
    
    # 3. Create Dummy Input
    dummy_text = "This is a sample scam message for export."
    inputs = tokenizer(dummy_text, return_tensors="pt", max_length=128, padding="max_length", truncation=True)
    
    # 4. Export
    output_onnx_path = os.path.join(os.path.dirname(model_path), "model.onnx")
    
    # Dynamic axes for variable batch size
    dynamic_axes = {
        'input_ids': {0: 'batch_size'},
        'attention_mask': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
    
    torch.onnx.export(
        model,
        (inputs['input_ids'], inputs['attention_mask']),
        output_onnx_path,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes=dynamic_axes,
        opset_version=14
    )
    
    # 5. Verify
    onnx_model = onnx.load(output_onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX model exported and verified at: {output_onnx_path}")
    
    # Save tokenizer for ONNX runtime usage if needed (usually just vocab files)
    tokenizer.save_pretrained(os.path.dirname(output_onnx_path))

if __name__ == "__main__":
    export_model_to_onnx()
