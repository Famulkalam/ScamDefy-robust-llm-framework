import pandas as pd
import os
import yaml

def audit_data_leakage():
    print("-" * 40)
    print("STEP 3: SANITY AUDIT - DATA LEAKAGE CHECK")
    print("-" * 40)
    
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    processed_dir = config['data']['processed_path']
    try:
        train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
        val_df = pd.read_parquet(os.path.join(processed_dir, "validation.parquet"))
        test_df = pd.read_parquet(os.path.join(processed_dir, "test_balanced.parquet"))
        print(f"Loaded Data: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    except FileNotFoundError:
        print("Error: Could not find processed data files. Run pipeline first.")
        return

    # 1. Exact String Overlap
    train_texts = set(train_df['text'].unique())
    val_texts = set(val_df['text'].unique())
    test_texts = set(test_df['text'].unique())
    
    train_val_overlap = train_texts.intersection(val_texts)
    train_test_overlap = train_texts.intersection(test_texts)
    val_test_overlap = val_texts.intersection(test_texts)
    
    print(f"\n[Leakage] Train-Val Overlap: {len(train_val_overlap)} samples")
    print(f"[Leakage] Train-Test Overlap: {len(train_test_overlap)} samples")
    print(f"[Leakage] Val-Test Overlap:   {len(val_test_overlap)} samples")
    
    if len(train_test_overlap) > 0:
        print("CRITICAL WARNING: Test data exists in Training set!")
        print(f"Example Leak: {list(train_test_overlap)[0]}")
    else:
        print("PASSED: No exact string overlap found between Train and Test.")

    # 2. Duplicate Check
    print("\n[Duplicates] Checking intra-split duplicates...")
    print(f"Train Duplicates: {len(train_df) - len(train_texts)}")
    print(f"Val Duplicates:   {len(val_df) - len(val_texts)}")
    print(f"Test Duplicates:  {len(test_df) - len(test_texts)}")

    # 3. Adversarial Leakage Check
    # Ensure adversarial examples in test aren't just copies of training data
    if 'source' in train_df.columns:
        adv_train = train_df[train_df['source'].astype(str).str.contains("adversarial")]
        if not adv_train.empty:
            adv_train_texts = set(adv_train['text'])
            adv_leak = adv_train_texts.intersection(test_texts)
            print(f"\n[Adversarial] Augmented Train samples found in Test: {len(adv_leak)}")
        else:
            print("\n[Adversarial] No offline adversarial samples in train set (likely on-the-fly).")

    print("-" * 40)

if __name__ == "__main__":
    audit_data_leakage()
