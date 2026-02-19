from abc import ABC, abstractmethod
import os
import openai
from typing import List

class BaseLLMProvider(ABC):
    """
    Abstract interface for LLM Red Team providers.
    Ensures we can swap OpenAI for Anthropic or Local (Ollama) easily.
    """
    @abstractmethod
    def generate_paraphrase(self, text: str, n_variants: int = 1) -> List[str]:
        """
        Generates 'n_variants' paraphrases of the input 'text'.
        The goal is to preserve the scam intent but change the wording to evade filters.
        """
        pass

class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: str = None):
        self.model_name = model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("Warning: OPENAI_API_KEY not set. Paraphrasing will be mocked.")
            
    def generate_paraphrase(self, text: str, n_variants: int = 1) -> List[str]:
        if not self.api_key:
             return [text] * n_variants # Mock behavior

        try:
            client = openai.OpenAI(api_key=self.api_key)
            prompt = (
                f"Rewrite the following scam message {n_variants} times to preserve its malicious intent "
                f"but avoid common spam keywords. Make it sound urgent and realistic.\n\n"
                f"Original: {text}\n\nVariants:"
            )
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.8
            )
            
            content = response.choices[0].message.content
            # Basic parsing assumption: variants separated by newlines or numbers
            variants = [v.strip() for v in content.split('\n') if v.strip()]
            return variants[:n_variants]
            
        except Exception as e:
            print(f"Error calling OpenAI: {e}")
            return [text] * n_variants

class MockLLMProvider(BaseLLMProvider):
     def generate_paraphrase(self, text: str, n_variants: int = 1) -> List[str]:
         return [f"Adversarial Variant: {text}" for _ in range(n_variants)]
