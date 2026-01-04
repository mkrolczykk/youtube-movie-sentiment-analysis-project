"""
Configuration module for YouTube Sentiment Analysis application.
Contains constants, model paths, and application settings.
"""

from typing import Dict, List

# =============================================================================
# API Configuration
# =============================================================================
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
MAX_COMMENTS_PER_REQUEST = 100  # YouTube API limit

# =============================================================================
# Language Configuration
# =============================================================================
SUPPORTED_LANGUAGES = ["pl", "en"]
DEFAULT_LANGUAGE = "en"

# Language-specific spaCy models
SPACY_MODELS = {
    "pl": "pl_core_news_md",
    "en": "en_core_web_md",
}

# Language-specific stopwords - will be loaded from NLTK
LANGUAGE_NAMES = {
    "pl": "polish",
    "en": "english",
}

# =============================================================================
# Sentiment Analysis Configuration
# =============================================================================
# Multilingual BERT model for sentiment analysis (supports Polish and English)
SENTIMENT_MODEL = "nlptown/bert-base-multilingual-uncased-sentiment"

# Emotion detection model
EMOTION_MODEL = "j-hartmann/emotion-english-distilroberta-base"

# Sentiment thresholds
SENTIMENT_THRESHOLDS = {
    "positive": 0.33,   # score > 0.33 is positive
    "negative": -0.33,  # score < -0.33 is negative
    # Between -0.33 and 0.33 is neutral
}

# =============================================================================
# Text Analytics Configuration
# =============================================================================
# Word cloud settings
WORDCLOUD_CONFIG = {
    "width": 800,
    "height": 400,
    "background_color": "white",
    "max_words": 100,
    "colormap": "viridis",
}

# Topic modeling settings
TOPIC_MODELING_CONFIG = {
    "n_topics": 5,
    "n_top_words": 10,
    "max_iter": 20,
}

# N-gram settings
NGRAM_CONFIG = {
    "min_n": 2,
    "max_n": 3,
    "top_k": 20,
}

# =============================================================================
# Dashboard Configuration
# =============================================================================
# Color schemes for visualizations
COLORS = {
    "positive": "#00C853",      # Green
    "negative": "#FF1744",      # Red
    "neutral": "#FFC107",       # Amber
    "primary": "#6C63FF",       # Purple
    "secondary": "#2196F3",     # Blue
    "background": "#0E1117",    # Dark
    "surface": "#1E1E1E",       # Dark surface
    "text": "#FAFAFA",          # Light text
}

# Plotly color sequence
PLOTLY_COLORS = [
    "#6C63FF",  # Purple
    "#00C853",  # Green
    "#FF1744",  # Red
    "#FFC107",  # Amber
    "#2196F3",  # Blue
    "#E91E63",  # Pink
    "#00BCD4",  # Cyan
    "#FF9800",  # Orange
]

# Dashboard layout
DASHBOARD_CONFIG = {
    "page_title": "YouTube Sentiment Analyzer",
    "page_icon": "🎬",
    "layout": "wide",
    "initial_sidebar_state": "expanded",
}

# Default analysis settings
DEFAULT_SETTINGS = {
    "num_comments": 100,
    "min_comments": 10,
    "max_comments": 500,
}

# =============================================================================
# Export Configuration
# =============================================================================
EXPORT_COLUMNS = [
    "comment_text",
    "author",
    "published_at",
    "like_count",
    "language",
    "sentiment_score",
    "sentiment_label",
    "sentiment_confidence",
]
