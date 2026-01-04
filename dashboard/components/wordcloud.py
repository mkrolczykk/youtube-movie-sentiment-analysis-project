"""
Word Cloud component for the dashboard.
Handles word cloud generation and display.
"""

import streamlit as st
import base64
from typing import Dict, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.text_analytics import TextAnalytics


def render_wordcloud_section(
    texts: list,
    sentiments: list,
    language: str = "en",
    analytics: Optional[TextAnalytics] = None
):
    """
    Render the word cloud section with tabs for different sentiments.
    
    Args:
        texts: List of comment texts
        sentiments: List of sentiment labels
        language: Primary language code
        analytics: TextAnalytics instance
    """
    if analytics is None:
        analytics = TextAnalytics()
    
    st.markdown("### ☁️ Word Clouds")
    
    # Generate word clouds for each sentiment
    with st.spinner("Generating word clouds..."):
        wordclouds = analytics.generate_sentiment_wordclouds(texts, sentiments, language)
    
    # Create tabs for different word clouds
    tab_all, tab_pos, tab_neg, tab_neu = st.tabs([
        "📊 All Comments",
        "😊 Positive",
        "😞 Negative",
        "😐 Neutral"
    ])
    
    with tab_all:
        # Generate overall word cloud
        all_wordcloud = analytics.generate_wordcloud(texts, language, colormap="viridis")
        display_wordcloud(all_wordcloud, "All Comments Word Cloud")
    
    with tab_pos:
        if "positive" in wordclouds:
            display_wordcloud(wordclouds["positive"], "Positive Comments Word Cloud")
        else:
            st.info("No positive comments to display")
    
    with tab_neg:
        if "negative" in wordclouds:
            display_wordcloud(wordclouds["negative"], "Negative Comments Word Cloud")
        else:
            st.info("No negative comments to display")
    
    with tab_neu:
        if "neutral" in wordclouds:
            display_wordcloud(wordclouds["neutral"], "Neutral Comments Word Cloud")
        else:
            st.info("No neutral comments to display")


def display_wordcloud(base64_image: str, caption: str = ""):
    """
    Display a base64 encoded word cloud image.
    
    Args:
        base64_image: Base64 encoded PNG image
        caption: Optional caption
    """
    st.markdown(
        f"""
        <div style="
            background: transparent;
            border-radius: 12px;
            padding: 0;
            margin: 0.5rem 0;
        ">
            <img src="data:image/png;base64,{base64_image}" 
                 style="width: 100%; height: auto; border-radius: 8px; display: block;">
        </div>
        """,
        unsafe_allow_html=True
    )
    if caption:
        st.caption(caption)


def render_comparison_wordclouds(
    texts_pl: list,
    texts_en: list,
    analytics: Optional[TextAnalytics] = None
):
    """
    Render side-by-side word clouds comparing Polish and English comments.
    
    Args:
        texts_pl: Polish comment texts
        texts_en: English comment texts
        analytics: TextAnalytics instance
    """
    if analytics is None:
        analytics = TextAnalytics()
    
    st.markdown("### 🌍 Language Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 🇵🇱 Polish Comments")
        if texts_pl:
            wc = analytics.generate_wordcloud(texts_pl, "pl", colormap="Reds")
            display_wordcloud(wc)
        else:
            st.info("No Polish comments")
    
    with col2:
        st.markdown("#### 🇬🇧 English Comments")
        if texts_en:
            wc = analytics.generate_wordcloud(texts_en, "en", colormap="Blues")
            display_wordcloud(wc)
        else:
            st.info("No English comments")
