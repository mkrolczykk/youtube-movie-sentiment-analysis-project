"""
Text Analytics module.
Provides advanced text analysis including word clouds, n-grams, topic modeling, and more.
"""

from typing import Dict, List, Optional, Tuple
from collections import Counter
import io
import base64

import numpy as np
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

from .config import (
    WORDCLOUD_CONFIG,
    TOPIC_MODELING_CONFIG,
    NGRAM_CONFIG,
    COLORS,
    LANGUAGE_NAMES
)
from .text_preprocessor import TextPreprocessor


class TextAnalytics:
    """
    Advanced text analytics engine.
    
    Provides word clouds, n-gram analysis, topic modeling, and keyword extraction.
    """
    
    def __init__(self, preprocessor: Optional[TextPreprocessor] = None):
        """
        Initialize the analytics engine.
        
        Args:
            preprocessor: Optional TextPreprocessor instance
        """
        self.preprocessor = preprocessor or TextPreprocessor(use_spacy=False)
    
    def generate_wordcloud(
        self,
        texts: List[str],
        language: str = "en",
        colormap: str = "viridis",
        max_words: int = 150,
        background_color: str = "#0E1117",  # Dark theme background
        width: int = 1200,
        height: int = 600
    ) -> str:
        """
        Generate a word cloud from texts.
        
        Args:
            texts: List of texts
            language: Language for preprocessing
            colormap: Matplotlib colormap name
            max_words: Maximum number of words
            background_color: Background color (dark by default for theme)
            width: Image width
            height: Image height
            
        Returns:
            Base64 encoded PNG image
        """
        # Preprocess and join texts
        processed = self.preprocessor.batch_preprocess(
            texts,
            languages=[language] * len(texts),
            clean=True,
            tokenize=True,
            remove_stops=True
        )
        combined_text = " ".join(processed)
        
        if not combined_text.strip():
            # Return empty image if no text
            return self._create_empty_wordcloud(width, height)
        
        # Generate word cloud - optimized to fill entire space
        wordcloud = WordCloud(
            width=width,
            height=height,
            background_color=background_color,
            colormap=colormap,
            max_words=max_words,
            prefer_horizontal=0.7,
            min_font_size=8,
            max_font_size=None,  # Auto-calculate max font size
            relative_scaling=0.5,  # Balance between frequency and rank
            margin=2,  # Minimal margin to fill space
            mode="RGBA" if background_color is None else "RGB",
            contour_width=0,
        ).generate(combined_text)
        
        return self._wordcloud_to_base64(wordcloud)
    
    def generate_sentiment_wordclouds(
        self,
        texts: List[str],
        sentiments: List[str],
        language: str = "en"
    ) -> Dict[str, str]:
        """
        Generate separate word clouds for positive and negative sentiments.
        
        Args:
            texts: List of texts
            sentiments: List of sentiment labels
            language: Language code
            
        Returns:
            Dictionary with 'positive', 'negative', 'neutral' word cloud images
        """
        # Separate texts by sentiment
        positive_texts = [t for t, s in zip(texts, sentiments) if s == "positive"]
        negative_texts = [t for t, s in zip(texts, sentiments) if s == "negative"]
        neutral_texts = [t for t, s in zip(texts, sentiments) if s == "neutral"]
        
        result = {}
        
        # Generate positive word cloud (green tones)
        if positive_texts:
            result["positive"] = self.generate_wordcloud(
                positive_texts, language, colormap="YlGn"
            )
        
        # Generate negative word cloud (red tones)
        if negative_texts:
            result["negative"] = self.generate_wordcloud(
                negative_texts, language, colormap="OrRd"
            )
        
        # Generate neutral word cloud (blue tones)
        if neutral_texts:
            result["neutral"] = self.generate_wordcloud(
                neutral_texts, language, colormap="Blues"
            )
        
        return result
    
    def _wordcloud_to_base64(self, wordcloud: WordCloud) -> str:
        """Convert wordcloud to base64 encoded PNG."""
        img_buffer = io.BytesIO()
        wordcloud.to_image().save(img_buffer, format='PNG')
        img_buffer.seek(0)
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    def _create_empty_wordcloud(self, width: int, height: int) -> str:
        """Create an empty placeholder image."""
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                fontsize=20, color='gray')
        ax.axis('off')
        
        img_buffer = io.BytesIO()
        fig.savefig(img_buffer, format='PNG', bbox_inches='tight', facecolor='white')
        plt.close(fig)
        img_buffer.seek(0)
        return base64.b64encode(img_buffer.getvalue()).decode()
    
    def extract_ngrams(
        self,
        texts: List[str],
        n: int = 2,
        top_k: int = 20,
        language: str = "en"
    ) -> List[Tuple[str, int]]:
        """
        Extract most common n-grams from texts.
        
        Args:
            texts: List of texts
            n: N-gram size (2 for bigrams, 3 for trigrams)
            top_k: Number of top n-grams to return
            language: Language for preprocessing
            
        Returns:
            List of (n-gram, count) tuples
        """
        # Preprocess texts
        processed = self.preprocessor.batch_preprocess(
            texts,
            languages=[language] * len(texts),
            clean=True,
            tokenize=True,
            remove_stops=True
        )
        
        # Filter out empty strings
        processed = [p for p in processed if p.strip()]
        
        if not processed:
            return []
        
        # Use CountVectorizer for n-gram extraction
        # Note: stopwords already removed during preprocessing
        vectorizer = CountVectorizer(
            ngram_range=(n, n),
            max_features=top_k * 2,  # Get more to filter
            token_pattern=r'(?u)\b\w+\b'  # Include single-character tokens
        )
        
        try:
            ngram_matrix = vectorizer.fit_transform(processed)
            ngram_counts = ngram_matrix.sum(axis=0).A1
            ngram_names = vectorizer.get_feature_names_out()
            
            # Sort by count
            sorted_indices = ngram_counts.argsort()[::-1][:top_k]
            
            return [
                (ngram_names[i], int(ngram_counts[i]))
                for i in sorted_indices
                if ngram_counts[i] > 0
            ]
        except ValueError as e:
            # Not enough data or vocabulary is empty
            return []
        except Exception as e:
            print(f"N-gram extraction error: {e}")
            return []
    
    def extract_keywords(
        self,
        texts: List[str],
        top_k: int = 20,
        language: str = "en"
    ) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF.
        
        Args:
            texts: List of texts
            top_k: Number of top keywords to return
            language: Language code
            
        Returns:
            List of (keyword, tfidf_score) tuples
        """
        # Preprocess texts
        processed = self.preprocessor.batch_preprocess(
            texts,
            languages=[language] * len(texts),
            clean=True,
            tokenize=True,
            remove_stops=True
        )
        
        # Filter out empty strings
        processed = [p for p in processed if p.strip()]
        
        if not processed:
            return []
        
        # TF-IDF extraction
        # Note: stopwords already removed during preprocessing
        vectorizer = TfidfVectorizer(
            max_features=top_k * 2,
            ngram_range=(1, 2),
            token_pattern=r'(?u)\b\w+\b'
        )
        
        try:
            tfidf_matrix = vectorizer.fit_transform(processed)
            feature_names = vectorizer.get_feature_names_out()
            
            # Average TF-IDF scores across documents
            avg_scores = tfidf_matrix.mean(axis=0).A1
            
            # Sort by score
            sorted_indices = avg_scores.argsort()[::-1][:top_k]
            
            return [
                (feature_names[i], float(avg_scores[i]))
                for i in sorted_indices
                if avg_scores[i] > 0
            ]
        except ValueError as e:
            # Not enough data or vocabulary is empty
            return []
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            return []
    
    def extract_topics(
        self,
        texts: List[str],
        n_topics: int = 5,
        n_top_words: int = 10,
        language: str = "en"
    ) -> List[Dict]:
        """
        Extract topics using Latent Dirichlet Allocation (LDA).
        
        Args:
            texts: List of texts
            n_topics: Number of topics to extract
            n_top_words: Number of top words per topic
            language: Language code
            
        Returns:
            List of topic dictionaries with words and weights
        """
        if len(texts) < n_topics:
            return []
        
        # Preprocess texts
        processed = self.preprocessor.batch_preprocess(
            texts,
            languages=[language] * len(texts),
            clean=True,
            tokenize=True,
            remove_stops=True
        )
        
        # Filter out empty strings
        processed = [p for p in processed if p.strip()]
        
        if len(processed) < n_topics:
            return []
        
        # Vectorize
        # Note: stopwords already removed during preprocessing
        vectorizer = CountVectorizer(
            max_features=1000,
            max_df=0.95,
            min_df=1,  # Reduced from 2 to work with smaller datasets
            token_pattern=r'(?u)\b\w+\b'
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform(processed)
            
            # Check if we have enough features
            if doc_term_matrix.shape[1] < n_topics:
                return []
            
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA model
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=TOPIC_MODELING_CONFIG["max_iter"],
                random_state=42
            )
            lda.fit(doc_term_matrix)
            
            # Extract topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[:-n_top_words-1:-1]
                top_words = [
                    {"word": feature_names[i], "weight": float(topic[i])}
                    for i in top_indices
                ]
                topics.append({
                    "topic_id": topic_idx,
                    "words": top_words,
                    "label": f"Topic {topic_idx + 1}"
                })
            
            return topics
            
        except ValueError as e:
            # Not enough data
            return []
        except Exception as e:
            print(f"Topic extraction error: {e}")
            return []
    
    def get_document_topics(
        self,
        texts: List[str],
        n_topics: int = 5,
        language: str = "en"
    ) -> Tuple[List[List[float]], List[Dict]]:
        """
        Get topic distribution for each document.
        
        Args:
            texts: List of texts
            n_topics: Number of topics
            language: Language code
            
        Returns:
            Tuple of (document_topic_distributions, topics)
        """
        if len(texts) < n_topics:
            return [], []
        
        # Preprocess
        processed = self.preprocessor.batch_preprocess(
            texts,
            languages=[language] * len(texts),
            clean=True,
            tokenize=True,
            remove_stops=True
        )
        
        # Filter out empty strings
        processed = [p for p in processed if p.strip()]
        
        if len(processed) < n_topics:
            return [], []
        
        # Vectorize
        vectorizer = CountVectorizer(
            max_features=1000,
            max_df=0.95,
            min_df=1,
            token_pattern=r'(?u)\b\w+\b'
        )
        
        try:
            doc_term_matrix = vectorizer.fit_transform(processed)
            
            if doc_term_matrix.shape[1] < n_topics:
                return [], []
            
            feature_names = vectorizer.get_feature_names_out()
            
            # LDA
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=20,
                random_state=42
            )
            doc_topics = lda.fit_transform(doc_term_matrix)
            
            # Get topics
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[:-6:-1]
                top_words = [feature_names[i] for i in top_indices]
                topics.append({
                    "topic_id": topic_idx,
                    "top_words": top_words,
                    "label": ", ".join(top_words[:3])
                })
            
            return doc_topics.tolist(), topics
            
        except Exception as e:
            print(f"Document topics error: {e}")
            return [], []
    
    def get_word_frequency(
        self,
        texts: List[str],
        top_k: int = 30,
        language: str = "en"
    ) -> List[Tuple[str, int]]:
        """
        Get word frequency distribution.
        
        Args:
            texts: List of texts
            top_k: Number of top words
            language: Language code
            
        Returns:
            List of (word, count) tuples
        """
        # Preprocess and tokenize
        all_words = []
        for text in texts:
            processed = self.preprocessor.preprocess(
                text, language, clean=True, tokenize=True, remove_stops=True
            )
            all_words.extend(processed.split())
        
        # Count words
        word_counts = Counter(all_words)
        
        return word_counts.most_common(top_k)
    
    def analyze_comment_lengths(self, texts: List[str]) -> Dict:
        """
        Analyze comment length distribution.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with length statistics
        """
        lengths = [len(text.split()) for text in texts]
        char_lengths = [len(text) for text in texts]
        
        return {
            "word_count": {
                "mean": float(np.mean(lengths)),
                "std": float(np.std(lengths)),
                "min": int(np.min(lengths)),
                "max": int(np.max(lengths)),
                "median": float(np.median(lengths))
            },
            "char_count": {
                "mean": float(np.mean(char_lengths)),
                "std": float(np.std(char_lengths)),
                "min": int(np.min(char_lengths)),
                "max": int(np.max(char_lengths)),
                "median": float(np.median(char_lengths))
            },
            "distribution": {
                "short": sum(1 for l in lengths if l < 10),
                "medium": sum(1 for l in lengths if 10 <= l < 50),
                "long": sum(1 for l in lengths if l >= 50)
            }
        }
    
    def get_emoji_analysis(self, texts: List[str]) -> Dict:
        """
        Analyze emoji usage in texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Dictionary with emoji statistics
        """
        all_emojis = []
        texts_with_emojis = 0
        
        for text in texts:
            emojis = self.preprocessor.extract_emojis(text)
            if emojis:
                texts_with_emojis += 1
                all_emojis.extend(emojis)
        
        emoji_counts = Counter(all_emojis)
        
        return {
            "total_emojis": len(all_emojis),
            "unique_emojis": len(emoji_counts),
            "texts_with_emojis": texts_with_emojis,
            "emoji_percentage": (texts_with_emojis / len(texts)) * 100 if texts else 0,
            "top_emojis": emoji_counts.most_common(10)
        }
