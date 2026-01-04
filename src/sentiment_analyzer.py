"""
Sentiment Analysis module.
Provides advanced sentiment analysis using transformer models with multilingual support.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import numpy as np

from .config import SENTIMENT_MODEL, SENTIMENT_THRESHOLDS


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    score: float           # -1 to 1 scale
    label: str             # 'positive', 'negative', 'neutral'
    confidence: float      # 0 to 1
    raw_scores: Dict[str, float]  # Original model scores


class SentimentAnalyzer:
    """
    Advanced sentiment analyzer using transformer-based models.
    
    Uses multilingual BERT model that supports both Polish and English.
    The model predicts star ratings (1-5) which are converted to sentiment scores.
    """
    
    # Star rating to sentiment score mapping
    # 1-2 stars = negative, 3 = neutral, 4-5 = positive
    STAR_TO_SCORE = {
        1: -1.0,
        2: -0.5,
        3: 0.0,
        4: 0.5,
        5: 1.0
    }
    
    def __init__(self, model_name: str = SENTIMENT_MODEL, device: Optional[str] = None):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model name/path
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.model_name = model_name
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self._model = None
        self._tokenizer = None
        self._pipeline = None
        
        # Suppress transformers warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
    
    @property
    def model(self):
        """Lazy load the model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load the tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    @property
    def sentiment_pipeline(self):
        """Lazy load the pipeline."""
        if self._pipeline is None:
            self._pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                truncation=True,
                max_length=512
            )
        return self._pipeline
    
    def _load_model(self):
        """Load model and tokenizer."""
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()
    
    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult with score, label, and confidence
        """
        if not text or len(text.strip()) == 0:
            return SentimentResult(
                score=0.0,
                label="neutral",
                confidence=0.0,
                raw_scores={}
            )
        
        # Truncate very long texts
        text = text[:512]
        
        try:
            # Use the pipeline for prediction
            result = self.sentiment_pipeline(text)[0]
            
            # Parse the result - model returns labels like "1 star", "2 stars", etc.
            label = result["label"]
            confidence = result["score"]
            
            # Extract star rating from label
            star_rating = int(label.split()[0])
            
            # Convert to sentiment score
            sentiment_score = self.STAR_TO_SCORE.get(star_rating, 0.0)
            
            # Determine sentiment label based on thresholds
            if sentiment_score > SENTIMENT_THRESHOLDS["positive"]:
                sentiment_label = "positive"
            elif sentiment_score < SENTIMENT_THRESHOLDS["negative"]:
                sentiment_label = "negative"
            else:
                sentiment_label = "neutral"
            
            return SentimentResult(
                score=sentiment_score,
                label=sentiment_label,
                confidence=confidence,
                raw_scores={"star_rating": star_rating, "model_confidence": confidence}
            )
            
        except Exception as e:
            # Return neutral on error
            return SentimentResult(
                score=0.0,
                label="neutral",
                confidence=0.0,
                raw_scores={"error": str(e)}
            )
    
    def analyze_batch(
        self, 
        texts: List[str], 
        batch_size: int = 16,
        show_progress: bool = False
    ) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts.
        
        Args:
            texts: List of texts to analyze
            batch_size: Number of texts to process at once
            show_progress: Whether to show progress (for long batches)
            
        Returns:
            List of SentimentResult objects
        """
        results = []
        
        # Filter empty texts
        valid_indices = []
        valid_texts = []
        for i, text in enumerate(texts):
            if text and len(text.strip()) > 0:
                valid_indices.append(i)
                valid_texts.append(text[:512])  # Truncate
        
        # Process in batches
        for i in range(0, len(valid_texts), batch_size):
            batch = valid_texts[i:i + batch_size]
            
            try:
                batch_results = self.sentiment_pipeline(batch)
                
                for result in batch_results:
                    label = result["label"]
                    confidence = result["score"]
                    star_rating = int(label.split()[0])
                    sentiment_score = self.STAR_TO_SCORE.get(star_rating, 0.0)
                    
                    if sentiment_score > SENTIMENT_THRESHOLDS["positive"]:
                        sentiment_label = "positive"
                    elif sentiment_score < SENTIMENT_THRESHOLDS["negative"]:
                        sentiment_label = "negative"
                    else:
                        sentiment_label = "neutral"
                    
                    results.append(SentimentResult(
                        score=sentiment_score,
                        label=sentiment_label,
                        confidence=confidence,
                        raw_scores={"star_rating": star_rating}
                    ))
                    
            except Exception:
                # Fill with neutral results on error
                for _ in batch:
                    results.append(SentimentResult(
                        score=0.0, label="neutral", confidence=0.0, raw_scores={}
                    ))
        
        # Reconstruct full results list with empty texts
        full_results = []
        result_idx = 0
        for i in range(len(texts)):
            if i in valid_indices:
                full_results.append(results[result_idx])
                result_idx += 1
            else:
                full_results.append(SentimentResult(
                    score=0.0, label="neutral", confidence=0.0, raw_scores={}
                ))
        
        return full_results
    
    def get_sentiment_statistics(self, results: List[SentimentResult]) -> Dict:
        """
        Calculate aggregate statistics from sentiment results.
        
        Args:
            results: List of SentimentResult objects
            
        Returns:
            Dictionary with aggregate statistics
        """
        if not results:
            return {
                "count": 0,
                "mean_score": 0.0,
                "std_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "positive_pct": 0.0,
                "negative_pct": 0.0,
                "neutral_pct": 0.0,
                "mean_confidence": 0.0
            }
        
        scores = [r.score for r in results]
        labels = [r.label for r in results]
        confidences = [r.confidence for r in results]
        
        positive_count = labels.count("positive")
        negative_count = labels.count("negative")
        neutral_count = labels.count("neutral")
        total = len(results)
        
        return {
            "count": total,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
            "positive_pct": (positive_count / total) * 100,
            "negative_pct": (negative_count / total) * 100,
            "neutral_pct": (neutral_count / total) * 100,
            "mean_confidence": float(np.mean(confidences))
        }


def create_analyzer(model_name: Optional[str] = None) -> SentimentAnalyzer:
    """
    Factory function to create a sentiment analyzer.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        Configured SentimentAnalyzer instance
    """
    return SentimentAnalyzer(model_name=model_name or SENTIMENT_MODEL)
