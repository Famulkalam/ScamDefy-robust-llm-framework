import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import yaml

from src.experiment.tracking import ExperimentTracker
from src.adversarial.generator import AdversarialGenerator

# --- Custom Dataset ---
class FraudDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# --- Metrics Computing ---
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# --- Main Training Loop ---
def train():
    # 1. Load Config
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # MPS support on Mac is optional but good
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # 2. Setup Tracking
    tracker = ExperimentTracker("distilroberta_finetune", config)
    
    # 3. Load Data
    processed_dir = config['data']['processed_path']
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "validation.parquet"))
    
    # 4. Adversarial Augmentation (On-the-fly or Pre-training)
    # We will augment the training set here directly for simplicity in the loop
    if config['training'].get('adversarial_mix_ratio', 0) > 0:
        print("Augmenting training data with adversarial examples...")
        adv_gen = AdversarialGenerator() # Uses Mock/OpenAI based on env
        # Augment only scam class (label=1)
        # Note: In a real large-scale run, we'd pre-generate this. 
        # Tokenisation is done in memory and cached.
        if 'adversarial' not in train_df['source'].values:
             train_df = adv_gen.augment_dataset(train_df, ratio=config['training']['adversarial_mix_ratio'])
             print(f"New training size after augmentation: {len(train_df)}")

    # 5. Tokenizer
    model_name = config['training']['model_name'] # distilroberta-base
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    train_dataset = FraudDataset(train_df['text'].to_numpy(), train_df['label'].to_numpy(), tokenizer)
    val_dataset = FraudDataset(val_df['text'].to_numpy(), val_df['label'].to_numpy(), tokenizer)
    
    # 6. Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # 7. Training Arguments
    training_args = TrainingArguments(
        output_dir=tracker.output_dir,
        num_train_epochs=config['training']['epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['batch_size'],
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(tracker.output_dir, 'logs'),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        seed=config['reproducibility']['random_seed']
    )
    
    # 8. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config['training']['early_stopping_patience'])]
    )
    
    # 9. Train
    print("Starting training...")
    train_result = trainer.train()
    
    # 10. Log Metrics
    tracker.log_metrics(train_result.metrics)
    
    # 11. Save Final Model
    final_model_path = tracker.get_artifact_path("final_model")
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Model saved to {final_model_path}")
    
    # 12. Evaluate
    eval_metrics = trainer.evaluate()
    tracker.log_metrics(eval_metrics)
    print("Evaluation Results:", eval_metrics)

if __name__ == "__main__":
    train()
