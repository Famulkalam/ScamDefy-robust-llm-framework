import pandas as pd
import os
from sklearn.model_selection import train_test_split
from .cleaning import clean_text

def load_uci(path):
    """Loads UCI SMS Spam Collection."""
    df = pd.read_csv(path, sep='\t', names=["label_str", "text"], header=None)
    df['label'] = df['label_str'].map({'ham': 0, 'spam': 1})
    df['source'] = 'uci_sms'
    return df[['text', 'label', 'source']]

def load_csv_dataset(path, source_name, label_col='label', text_col='text'):
    """Generic CSV loader."""
    df = pd.read_csv(path)
    # Ensure columns exist
    if label_col not in df.columns or text_col not in df.columns:
        print(f"Warning: {source_name} invalid format. Skipping.")
        return pd.DataFrame()
    
    df = df.rename(columns={label_col: 'label', text_col: 'text'})
    df['source'] = source_name
    return df[['text', 'label', 'source']]

def create_imbalanced_test_set(test_df, fraud_ratio=0.01):
    """
    Resamples the test set to match a specific fraud prevalence (e.g., 1%).
    """
    legit = test_df[test_df['label'] == 0]
    scam = test_df[test_df['label'] == 1]
    
    n_scam = int(len(legit) * fraud_ratio / (1 - fraud_ratio))
    
    # If we don't have enough scams, we limit legitimate instead
    if n_scam > len(scam):
         n_scam = len(scam)
         n_legit = int(n_scam * (1 - fraud_ratio) / fraud_ratio)
         legit = legit.sample(n=n_legit, random_state=42)
    else:
        scam = scam.sample(n=n_scam, random_state=42)
        
    return pd.concat([legit, scam]).sample(frac=1, random_state=42).reset_index(drop=True)

def run_pipeline():
    raw_dir = "data/raw"
    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)
    
    # 1. Load Data
    print("Loading datasets...")
    # Adjust paths based on download.py output
    df_uci = load_uci(os.path.join(raw_dir, "SMSSpamCollection"))
    df_enron = load_csv_dataset(os.path.join(raw_dir, "enron_subset.csv"), "enron")
    df_phish = load_csv_dataset(os.path.join(raw_dir, "phishing_emails.csv"), "phishing_kaggle")
    
    # 2. Combine
    df = pd.concat([df_uci, df_enron, df_phish], ignore_index=True)
    print(f"Total raw samples: {len(df)}")
    
    # 3. Clean
    print("Cleaning text...")
    df['clean_text'] = df['text'].apply(clean_text)
    df = df.drop_duplicates(subset=['clean_text'])
    print(f"Samples after deduplication: {len(df)}")
    
    # 4. Split
    # Balanced Train/Validation split first
    train_val, test = train_test_split(df, test_size=0.15, stratify=df['label'], random_state=42)
    train, val = train_test_split(train_val, test_size=0.15, stratify=train_val['label'], random_state=42)
    
    # 5. Create Imbalanced Test Set (Production Simulation)
    test_imbalanced = create_imbalanced_test_set(test, fraud_ratio=0.01)
    
    # 6. Save
    train.to_parquet(os.path.join(processed_dir, "train.parquet"), index=False)
    val.to_parquet(os.path.join(processed_dir, "validation.parquet"), index=False)
    test.to_parquet(os.path.join(processed_dir, "test_balanced.parquet"), index=False)
    test_imbalanced.to_parquet(os.path.join(processed_dir, "test_imbalanced.parquet"), index=False)
    
    print("Data pipeline completed details:")
    print(f"Train: {len(train)}")
    print(f"Val: {len(val)}")
    print(f"Test (Balanced): {len(test)}")
    print(f"Test (Imbalanced 1%): {len(test_imbalanced)}")

if __name__ == "__main__":
    run_pipeline()
