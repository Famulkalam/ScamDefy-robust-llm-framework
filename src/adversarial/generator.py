from typing import List, Optional
import random
import pandas as pd
from .attacks import AdversarialAttacks
from .llm_provider import BaseLLMProvider, MockLLMProvider

class AdversarialGenerator:
    """
    Orchestrates the generation of adversarial examples.
    Combines rule-based attacks (Character/Homoglyph) with LLM-based Paraphrasing.
    """
    def __init__(self, llm_provider: Optional[BaseLLMProvider] = None):
        self.llm_provider = llm_provider or MockLLMProvider()

    def generate_variants(self, text: str, attack_types: List[str] = None) -> List[str]:
        """
        Generates variants of the input text using specified attack types.
        """
        if attack_types is None:
            attack_types = ["char", "homoglyph", "llm"]
            
        variants = []
        
        # 1. Rule-based: Cheap and fast
        if "char" in attack_types:
            variants.append(AdversarialAttacks.perturb_characters(text, probability=0.2))
        
        if "homoglyph" in attack_types:
            variants.append(AdversarialAttacks.insert_homoglyphs(text, probability=0.2))
            
        # 2. LLM-based: Expensive (Paraphrase)
        # Only call if explicitly requested and useful
        if "llm" in attack_types:
            # We request 1 variant for efficiency unless configured otherwise
            llm_variants = self.llm_provider.generate_paraphrase(text, n_variants=1)
            variants.extend(llm_variants)
            
        if "combined_extreme" in attack_types:
            # The "Kitchen Sink" attack
            v = text
            v = AdversarialAttacks.random_casing(v, probability=0.4)
            v = AdversarialAttacks.mask_urls(v, probability=0.5)
            v = AdversarialAttacks.perturb_characters(v, probability=0.3)
            v = AdversarialAttacks.insert_homoglyphs(v, probability=0.3)
            v = AdversarialAttacks.inject_filler(v, probability=0.4)
            variants.append(v)

        # Deduplicate and remove identity
        unique_variants = set(variants)
        if text in unique_variants:
            unique_variants.remove(text)
            
        return list(unique_variants)

    def augment_dataset(self, df: pd.DataFrame, target_label: int = 1, ratio: float = 0.3) -> pd.DataFrame:
        """
        Augments the dataset by generating adversarial examples for a specific class.
        ratio: Fraction of the original class count to generate.
        """
        target_df = df[df['label'] == target_label]
        n_augment = int(len(target_df) * ratio)
        
        # Sample to augment
        samples = target_df.sample(n=n_augment, replace=True, random_state=42)
        
        new_rows = []
        for _, row in samples.iterrows():
            # Randomly choose attack mix
            attacks = []
            param = random.random()
            
            # v1.1 Hardening Mix Strategy
            if param < 0.4:
                attacks.append("combined_extreme") # 40% Extreme
            elif param < 0.7:
                attacks.append("char") # 30% Simple Leetspeak
            elif param < 0.9:
                attacks.append("homoglyph") # 20% Homoglyph
            else:
                attacks.append("llm") # 10% LLM (Expensive)
            
            variants = self.generate_variants(row['text'], attack_types=attacks)
            for v in variants:
                new_rows.append({
                    "text": v,
                    "label": target_label, 
                    "source": f"adversarial_{row.get('source', 'unknown')}"
                })
                
        augmented_df = pd.DataFrame(new_rows)
        return pd.concat([df, augmented_df], ignore_index=True)
