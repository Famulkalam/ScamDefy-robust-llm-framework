import random
import re

class AdversarialAttacks:
    """
    Collection of scam-specific adversarial attacks.
    Methods are static to be used in isolation or composed.
    """
    
    # Common leetspeak mappings used in scams
    LEET_MAP = {
        'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '+',
        'A': '@', 'E': '3', 'I': '1', 'O': '0', 'S': '$', 'T': '+'
    }
    
    # Common homoglyphs (look-alike characters)
    HOMOGLYPH_MAP = {
        'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', # Cyrillic
        'y': 'у', 'x': 'х', 'T': 'Т', 'H': 'Н', 'P': 'Р'
    }

    @staticmethod
    def perturb_characters(text: str, probability: float = 0.15) -> str:
        """
        Randomly replaces characters with leetspeak equivalents.
        Simulates 'b@nk', 'acc0unt', 'v3rify'.
        """
        chars = list(text)
        for i, char in enumerate(chars):
            if char in AdversarialAttacks.LEET_MAP and random.random() < probability:
                chars[i] = AdversarialAttacks.LEET_MAP[char]
        return "".join(chars)

    @staticmethod
    def insert_homoglyphs(text: str, probability: float = 0.1) -> str:
        """
        Replaces characters with visual look-alikes (homoglyphs).
        Effective against exact keyword matching.
        """
        chars = list(text)
        for i, char in enumerate(chars):
            if char in AdversarialAttacks.HOMOGLYPH_MAP and random.random() < probability:
                chars[i] = AdversarialAttacks.HOMOGLYPH_MAP[char]
        return "".join(chars)

    @staticmethod
    def inject_noise(text: str, probability: float = 0.05) -> str:
        """
        Randomly inserts invisible characters or spacing.
        e.g., "Verify" -> "V e r i f y" or "Verify\u200b"
        """
        # Simple spacing injection for now
        if random.random() < probability:
            return " ".join(list(text))
        return text

    @staticmethod
    def mask_urls(text: str, probability: float = 0.5) -> str:
        """
        Replaces URLs with suspicious or shortened versions.
        """
        url_pattern = r'https?://\S+|www\.\S+'
        suspicious_urls = [
            "http://bit.ly/secure-login", 
            "http://tinyurl.com/verify-now",
            "https://secure-bank-check.com",
            "http://192.168.1.1/login"
        ]
        
        def replace_url(match):
            return random.choice(suspicious_urls) if random.random() < probability else match.group(0)
            
        return re.sub(url_pattern, replace_url, text)

    @staticmethod
    def random_casing(text: str, probability: float = 0.3) -> str:
        """
        Randomizes casing to simulate urgency or filter evasion.
        e.g., "URGENT", "UrGeNt", "verify NOW"
        """
        chars = list(text)
        for i, char in enumerate(chars):
            if random.random() < probability:
                chars[i] = char.upper() if random.random() < 0.5 else char.lower()
        return "".join(chars)

    @staticmethod
    def inject_filler(text: str, probability: float = 0.3) -> str:
        """
        Injects random filler text to confuse models.
        """
        fillers = [
            " [Loading...]", " --Ignore below--", " (SecureMsg)", 
            "\n\nSent from my iPhone", " Ref: 99382"
        ]
        if random.random() < probability:
            return text + random.choice(fillers)
        return text
