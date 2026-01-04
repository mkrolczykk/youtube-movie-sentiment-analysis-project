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
        
        # Add comprehensive Polish stopwords (function words, prepositions, conjunctions, particles)
        # These are words that don't carry semantic meaning in text analysis
        polish_extras = {
            # Prepositions (przyimki)
            "w", "we", "na", "do", "z", "ze", "za", "przed", "po", "od", "bez", "dla", 
            "przez", "przy", "o", "u", "nad", "pod", "między", "pomiędzy", "wśród",
            "oprócz", "około", "koło", "obok", "wobec", "według", "dzięki", "mimo",
            "podczas", "poprzez", "wzdłuż", "naprzeciw", "względem", "ponad", "spod",
            
            # Conjunctions (spójniki)
            "i", "a", "ale", "lub", "albo", "czy", "że", "żeby", "bo", "ponieważ",
            "gdyż", "więc", "oraz", "ani", "jednak", "lecz", "zatem", "dlatego",
            "choć", "chociaż", "jeśli", "jeżeli", "gdy", "kiedy", "dopóki", "zanim",
            
            # Pronouns (zaimki)
            "ja", "ty", "on", "ona", "ono", "my", "wy", "oni", "one", "się",
            "mnie", "mi", "mną", "ciebie", "ci", "cię", "tobą", "jego", "jej", "jemu",
            "go", "mu", "ją", "nią", "nim", "nas", "nam", "nami", "was", "wam", "wami",
            "ich", "im", "nimi", "siebie", "sobie", "sobą",
            "ten", "ta", "to", "te", "ci", "tamten", "tamta", "tamto",
            "który", "która", "które", "którzy", "którego", "której", "których",
            "co", "kto", "czego", "kogo", "czemu", "komu", "czym", "kim",
            "jaki", "jaka", "jakie", "jacyś", "jakiś", "jakieś",
            "każdy", "każda", "każde", "wszyscy", "wszystko", "wszystkie",
            "sam", "sama", "samo", "sami", "same",
            "nikt", "nic", "żaden", "żadna", "żadne",
            "inny", "inna", "inne", "inni",
            "swój", "swoja", "swoje", "swojego", "swojej", "swoich",
            "mój", "moja", "moje", "mojego", "mojej", "moich",
            "twój", "twoja", "twoje", "twojego", "twojej", "twoich",
            "nasz", "nasza", "nasze", "naszego", "naszej", "naszych",
            "wasz", "wasza", "wasze", "waszego", "waszej", "waszych",
            
            # Particles and adverbs (partykuły i przysłówki)
            "tak", "nie", "też", "tylko", "już", "jeszcze", "bardzo", "bardziej",
            "najbardziej", "może", "chyba", "pewnie", "właśnie", "przecież", "jednak",
            "nawet", "prawie", "raczej", "całkiem", "dosyć", "dość", "trochę",
            "zawsze", "nigdy", "często", "rzadko", "czasem", "czasami",
            "teraz", "wtedy", "potem", "przedtem", "później", "wcześniej",
            "tutaj", "tam", "tu", "gdzie", "gdzieś", "nigdzie", "wszędzie",
            "jak", "jakby", "jakoś", "tak", "inaczej",
            "dlaczego", "czemu", "dlatego", "po co",
            "ile", "tyle", "wiele", "mało", "dużo", "kilka", "parę",
            
            # Auxiliary verbs and common verbs (czasowniki posiłkowe)
            "być", "jest", "są", "był", "była", "było", "byli", "były",
            "będzie", "będą", "będę", "będziesz", "będziemy", "będziecie",
            "mieć", "ma", "mam", "masz", "mamy", "macie", "mają",
            "miał", "miała", "miało", "mieli", "miały",
            "można", "trzeba", "należy", "warto", "wolno",
            "zostać", "stać", "robić", "zrobić", "mówić", "powiedzieć",
            
            # Other common function words
            "to", "tego", "temu", "tym", "ta", "tej", "tą",
            "jako", "czyli", "np", "itd", "itp", "etc",
            "pan", "pani", "państwo", "panie", "panowie",
            "no", "ot", "hej", "cześć", "ok", "okej",
            "xd", "xdd", "haha", "lol", "omg",  # Common internet expressions
        }
        self._stopwords.get("pl", set()).update(polish_extras)
        
        # Also add some common English function words that might be mixed in Polish comments
        english_extras = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
            "my", "your", "his", "its", "our", "their",
            "this", "that", "these", "those",
            "and", "or", "but", "if", "then", "so", "as", "of", "to", "for",
            "with", "on", "in", "at", "by", "from",
            "lol", "xd", "omg", "wtf", "btw", "idk",  # Internet slang
        }
        self._stopwords.get("en", set()).update(english_extras)
    
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
        
        # Filter tokens: remove stopwords, very short words, and non-alphabetic tokens
        filtered = []
        for token in tokens:
            token_lower = token.lower()
            # Skip stopwords
            if token_lower in stop_set:
                continue
            # Skip very short words (1-2 chars) unless they're meaningful
            if len(token) < 3:
                continue
            # Skip pure numbers
            if token.isdigit():
                continue
            # Keep tokens with at least some letters
            if any(c.isalpha() for c in token):
                filtered.append(token)
        
        return filtered
    
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
