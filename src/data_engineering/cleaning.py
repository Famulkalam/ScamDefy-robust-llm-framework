import re
import html

def clean_text(text: str) -> str:
    """
    Standard text normalization pipeline.
    - Decodes HTML entities
    - Removes excessive whitespace
    - Keeps casing/punctuation mostly intact (vital for scam classification)
    """
    if not isinstance(text, str):
        return ""
        
    # 1. Decode HTML (common in phishing emails)
    text = html.unescape(text)
    
    # 2. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Note: We are NOT removing special characters (e.g., '@', '$') drastically, 
    # as 'b@nk' is a key adversarial feature we want to learn or detect.
    
    return text
