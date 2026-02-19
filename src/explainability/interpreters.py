import numpy as np
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class ScamExplainer:
    """
    Wrapper for LIME and SHAP to provide governance-friendly explanations.
    """
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if torch.backends.mps.is_available(): os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"; self.device = "mps"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()
        
    def predict_proba(self, texts):
        # LIME expects a function that takes a list of strings and returns probabilities
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = F.softmax(logits, dim=1)
            
        return probs.cpu().numpy()

    def explain_lime(self, text: str, num_features: int = 10):
        """
        Geneates a LIME explanation for a single text instance.
        """
        explainer = LimeTextExplainer(class_names=["Legitimate", "Scam"])
        exp = explainer.explain_instance(
            text, 
            self.predict_proba, 
            num_features=num_features,
            num_samples=500 # Lower for speed in demo
        )
        return exp

    def explain_shap(self, texts: list):
        """
        Generates SHAP values (PartitionExplainer is best for text).
        Note: SHAP is computationally expensive.
        """
        # Define a prediction function for SHAP
        def f(x):
            tv = torch.tensor([self.tokenizer.encode(v, padding='max_length', max_length=128, truncation=True) for v in x]).to(self.device)
            attention_mask = (tv!=0).type(torch.int64).to(self.device)
            outputs = self.model(tv, attention_mask=attention_mask)[0].detach().cpu().numpy()
            scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
            val = sp.special.logit(scores[:,1]) # use one vs rest logit units
            return val
            
        # Implementation of full SHAP pipeline requires more setup, 
        # often easier to just use the Transformers pipeline + SHAP
        pass

if __name__ == '__main__':
    pass
