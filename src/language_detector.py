"""
Language Detection module.
Detects language of text content with support for Polish and English.
"""

from typing import List, Optional, Tuple
from collections import Counter

from langdetect import detect, detect_langs, LangDetectException
from langdetect.lang_detect_exception import ErrorCode

from .config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE


def detect_language(text: str) -> Tuple[str, float]:
    """
    Detect the language of a given text.
    
    Args:
        text: Text to analyze
        
    Returns:
        Tuple of (language_code, confidence)
        Language code is 'pl', 'en', or 'other'
    """
    if not text or len(text.strip()) < 3:
        return DEFAULT_LANGUAGE, 0.0
    
    try:
        # Get language probabilities
        lang_probs = detect_langs(text)
        
        if not lang_probs:
            return DEFAULT_LANGUAGE, 0.0
        
        # Get the most probable language
        top_lang = lang_probs[0]
        lang_code = top_lang.lang
        confidence = top_lang.prob
        
        # Map to supported languages
        if lang_code in SUPPORTED_LANGUAGES:
            return lang_code, confidence
        else:
            return "other", confidence
            
    except LangDetectException as e:
        # Handle edge cases (too short, no features, etc.)
        return DEFAULT_LANGUAGE, 0.0


def batch_detect(texts: List[str]) -> List[Tuple[str, float]]:
    """
    Detect languages for a batch of texts.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        List of (language_code, confidence) tuples
    """
    return [detect_language(text) for text in texts]


def get_dominant_language(texts: List[str]) -> str:
    """
    Determine the dominant language in a collection of texts.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        Most common language code ('pl', 'en', or 'other')
    """
    if not texts:
        return DEFAULT_LANGUAGE
    
    detected = batch_detect(texts)
    languages = [lang for lang, conf in detected if conf > 0.5]
    
    if not languages:
        return DEFAULT_LANGUAGE
    
    # Count occurrences and return most common
    counter = Counter(languages)
    most_common = counter.most_common(1)[0][0]
    
    return most_common


def categorize_by_language(texts: List[str]) -> dict:
    """
    Categorize texts by detected language.
    
    Args:
        texts: List of texts to categorize
        
    Returns:
        Dictionary with language codes as keys and lists of (index, text) as values
    """
    categories = {"pl": [], "en": [], "other": []}
    
    for idx, text in enumerate(texts):
        lang, confidence = detect_language(text)
        categories[lang].append({
            "index": idx,
            "text": text,
            "confidence": confidence
        })
    
    return categories


def get_language_statistics(texts: List[str]) -> dict:
    """
    Get language distribution statistics for a collection of texts.
    
    Args:
        texts: List of texts to analyze
        
    Returns:
        Dictionary with language statistics
    """
    if not texts:
        return {
            "total": 0,
            "distribution": {},
            "percentages": {},
            "dominant": DEFAULT_LANGUAGE
        }
    
    detected = batch_detect(texts)
    languages = [lang for lang, conf in detected]
    
    counter = Counter(languages)
    total = len(texts)
    
    distribution = dict(counter)
    percentages = {lang: (count / total) * 100 for lang, count in counter.items()}
    
    return {
        "total": total,
        "distribution": distribution,
        "percentages": percentages,
        "dominant": counter.most_common(1)[0][0] if counter else DEFAULT_LANGUAGE
    }
