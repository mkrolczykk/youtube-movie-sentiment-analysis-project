"""
Text Preprocessing module.
Provides language-aware text cleaning, tokenization, and normalization.
"""

import re
import string
from typing import List, Optional, Set
import unicodedata

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from .config import LANGUAGE_NAMES, SPACY_MODELS, SUPPORTED_LANGUAGES

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """
    Language-aware text preprocessing pipeline.
    
    Supports Polish and English with language-specific processing.
    """
    
    # Emoji pattern
    EMOJI_PATTERN = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE
    )
    
    # URL pattern
    URL_PATTERN = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    
    # Mention pattern (@username)
    MENTION_PATTERN = re.compile(r'@[\w]+')
    
    # Hashtag pattern
    HASHTAG_PATTERN = re.compile(r'#[\w]+')
    
    # Multiple spaces/newlines
    WHITESPACE_PATTERN = re.compile(r'\s+')
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the preprocessor.
        
        Args:
            use_spacy: Whether to load spaCy models for lemmatization
        """
        self.use_spacy = use_spacy
        self._spacy_models = {}
        self._stopwords = {}
        
        # Load stopwords for supported languages
        self._load_stopwords()
        
        # Lazy load spaCy models
        if use_spacy:
            self._load_spacy_models()
    
    def _load_stopwords(self):
        """Load stopwords for all supported languages."""
        for lang_code in SUPPORTED_LANGUAGES:
            lang_name = LANGUAGE_NAMES.get(lang_code, "english")
            try:
                self._stopwords[lang_code] = set(stopwords.words(lang_name))
            except OSError:
                # Fallback to English if language not available
                self._stopwords[lang_code] = set(stopwords.words("english"))
        
        # Add common Polish stopwords that might be missing
        polish_extras = {
            "też", "oraz", "ale", "czy", "jak", "tak", "nie", "się", "jest",
            "może", "więc", "tylko", "jeszcze", "już", "bardzo", "gdzie",
            "kiedy", "kto", "co", "ten", "ta", "to", "być", "mieć"
        }
        self._stopwords.get("pl", set()).update(polish_extras)
    
    def _load_spacy_models(self):
        """Lazy load spaCy models."""
        import spacy
        
        for lang_code, model_name in SPACY_MODELS.items():
            try:
                self._spacy_models[lang_code] = spacy.load(model_name)
            except OSError:
                # Model not installed, will be handled gracefully
                self._spacy_models[lang_code] = None
    
    def clean_text(
        self,
        text: str,
        remove_emojis: bool = True,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags: bool = False,
        lowercase: bool = True,
        remove_punctuation: bool = False
    ) -> str:
        """
        Clean text by removing unwanted elements.
        
        Args:
            text: Input text
            remove_emojis: Remove emoji characters
            remove_urls: Remove URLs
            remove_mentions: Remove @mentions
            remove_hashtags: Remove #hashtags
            lowercase: Convert to lowercase
            remove_punctuation: Remove punctuation marks
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Normalize unicode
        text = unicodedata.normalize("NFKC", text)
        
        # Remove URLs
        if remove_urls:
            text = self.URL_PATTERN.sub(" ", text)
        
        # Remove mentions
        if remove_mentions:
            text = self.MENTION_PATTERN.sub(" ", text)
        
        # Remove hashtags (or keep just the word)
        if remove_hashtags:
            text = self.HASHTAG_PATTERN.sub(" ", text)
        else:
            text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove emojis
        if remove_emojis:
            text = self.EMOJI_PATTERN.sub(" ", text)
        
        # Convert to lowercase
        if lowercase:
            text = text.lower()
        
        # Remove punctuation
        if remove_punctuation:
            text = text.translate(str.maketrans("", "", string.punctuation))
        
        # Normalize whitespace
        text = self.WHITESPACE_PATTERN.sub(" ", text)
        
        return text.strip()
    
    def tokenize(self, text: str, language: str = "en") -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            language: Language code ('pl' or 'en')
            
        Returns:
            List of tokens
        """
        if not text:
            return []
        
        # Use NLTK tokenizer
        try:
            if language == "pl":
                # Polish-aware tokenization
                tokens = word_tokenize(text, language="polish")
            else:
                tokens = word_tokenize(text, language="english")
        except Exception:
            # Fallback to simple split
            tokens = text.split()
        
        return tokens
    
    def remove_stopwords(
        self, 
        tokens: List[str], 
        language: str = "en",
        custom_stopwords: Optional[Set[str]] = None
    ) -> List[str]:
        """
        Remove stopwords from token list.
        
        Args:
            tokens: List of tokens
            language: Language code
            custom_stopwords: Additional stopwords to remove
            
        Returns:
            Filtered token list
        """
        stop_set = self._stopwords.get(language, self._stopwords.get("en", set()))
        
        if custom_stopwords:
            stop_set = stop_set | custom_stopwords
        
        return [token for token in tokens if token.lower() not in stop_set]
    
    def lemmatize(self, text: str, language: str = "en") -> str:
        """
        Lemmatize text using spaCy.
        
        Args:
            text: Input text
            language: Language code
            
        Returns:
            Lemmatized text
        """
        if not self.use_spacy:
            return text
        
        nlp = self._spacy_models.get(language)
        if nlp is None:
            return text
        
        doc = nlp(text)
        lemmas = [token.lemma_ for token in doc if not token.is_space]
        
        return " ".join(lemmas)
    
    def preprocess(
        self,
        text: str,
        language: str = "en",
        clean: bool = True,
        tokenize: bool = True,
        remove_stops: bool = True,
        lemmatize: bool = False
    ) -> str:
        """
        Full preprocessing pipeline.
        
        Args:
            text: Input text
            language: Language code
            clean: Apply text cleaning
            tokenize: Tokenize text
            remove_stops: Remove stopwords
            lemmatize: Apply lemmatization
            
        Returns:
            Preprocessed text
        """
        if not text:
            return ""
        
        result = text
        
        if clean:
            result = self.clean_text(result)
        
        if lemmatize:
            result = self.lemmatize(result, language)
        
        if tokenize:
            tokens = self.tokenize(result, language)
            
            if remove_stops:
                tokens = self.remove_stopwords(tokens, language)
            
            result = " ".join(tokens)
        
        return result
    
    def batch_preprocess(
        self,
        texts: List[str],
        languages: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """
        Preprocess a batch of texts.
        
        Args:
            texts: List of input texts
            languages: List of language codes (same length as texts)
            **kwargs: Additional arguments passed to preprocess()
            
        Returns:
            List of preprocessed texts
        """
        if languages is None:
            languages = ["en"] * len(texts)
        
        return [
            self.preprocess(text, lang, **kwargs)
            for text, lang in zip(texts, languages)
        ]
    
    def extract_emojis(self, text: str) -> List[str]:
        """
        Extract emojis from text.
        
        Args:
            text: Input text
            
        Returns:
            List of emojis found
        """
        return self.EMOJI_PATTERN.findall(text)
    
    def get_text_stats(self, text: str) -> dict:
        """
        Get basic statistics about the text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with text statistics
        """
        words = text.split()
        chars = len(text)
        
        return {
            "char_count": chars,
            "word_count": len(words),
            "avg_word_length": sum(len(w) for w in words) / len(words) if words else 0,
            "sentence_count": len(re.findall(r'[.!?]+', text)) or 1,
            "has_urls": bool(self.URL_PATTERN.search(text)),
            "has_emojis": bool(self.EMOJI_PATTERN.search(text)),
            "has_mentions": bool(self.MENTION_PATTERN.search(text)),
        }
