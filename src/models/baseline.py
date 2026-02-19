import os
import pandas as pd
import joblib
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
from xgboost import XGBClassifier
from src.experiment.tracking import ExperimentTracker

def train_baseline():
    # 1. Load Config
    with open("config/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    tracker = ExperimentTracker("baseline_xgboost", config)
    
    # 2. Load Data
    processed_dir = config['data']['processed_path']
    train_df = pd.read_parquet(os.path.join(processed_dir, "train.parquet"))
    val_df = pd.read_parquet(os.path.join(processed_dir, "validation.parquet"))
    test_df = pd.read_parquet(os.path.join(processed_dir, "test_balanced.parquet"))

    print(f"Training Baseline on {len(train_df)} samples...")

    # 3. Vectorization (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train = vectorizer.fit_transform(train_df['clean_text'])
    X_val = vectorizer.transform(val_df['clean_text'])
    X_test = vectorizer.transform(test_df['clean_text'])
    
    y_train = train_df['label']
    y_val = val_df['label']
    y_test = test_df['label']

    # 4. Train XGBoost
    model = XGBClassifier(
        n_estimators=100, 
        random_state=config['reproducibility']['random_seed'],
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 5. Evaluate
    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    val_f1 = f1_score(y_val, val_preds)
    
    print("Validation Results:")
    print(classification_report(y_val, val_preds))
    
    tracker.log_metrics({
        "accuracy": val_acc,
        "f1": val_f1
    })

    # 6. Save Model
    model_path = tracker.get_artifact_path("xgboost_model.joblib")
    vectorizer_path = tracker.get_artifact_path("tfidf_vectorizer.joblib")
    
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    print(f"Baseline model saved to {model_path}")

if __name__ == "__main__":
    train_baseline()
