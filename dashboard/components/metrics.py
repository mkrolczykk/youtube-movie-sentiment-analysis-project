"""
Metrics component for the dashboard.
Displays KPI cards and summary statistics.
"""

import streamlit as st
from typing import Dict, Optional
import pandas as pd


def render_video_header(video_metadata: Dict):
    """
    Render the video information header.
    
    Args:
        video_metadata: Video metadata dictionary
    """
    col1, col2 = st.columns([1, 3])
    
    with col1:
        if video_metadata.get("thumbnail_url"):
            st.image(video_metadata["thumbnail_url"], use_container_width=True)
    
    with col2:
        st.markdown(f"### {video_metadata.get('title', 'Unknown Video')}")
        st.caption(f"📺 {video_metadata.get('channel_title', 'Unknown Channel')}")
        
        # Stats row
        stats_cols = st.columns(3)
        with stats_cols[0]:
            views = video_metadata.get('view_count', 0)
            st.metric("Views", format_number(views))
        with stats_cols[1]:
            likes = video_metadata.get('like_count', 0)
            st.metric("Likes", format_number(likes))
        with stats_cols[2]:
            comments = video_metadata.get('comment_count', 0)
            st.metric("Comments", format_number(comments))


def render_sentiment_metrics(sentiment_stats: Dict, language_stats: Dict):
    """
    Render the main sentiment metrics cards.
    
    Args:
        sentiment_stats: Sentiment statistics dictionary
        language_stats: Language distribution statistics
    """
    # Main metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            "Total Analyzed Comments",
            str(sentiment_stats.get("count", 0)),
            icon="💬",
            color="#6C63FF"
        )
    
    with col2:
        avg_score = sentiment_stats.get("mean_score", 0)
        sentiment_emoji = "😊" if avg_score > 0.2 else "😐" if avg_score > -0.2 else "😞"
        render_metric_card(
            "Avg. Sentiment",
            f"{avg_score:.2f}",
            icon=sentiment_emoji,
            color=get_sentiment_color(avg_score)
        )
    
    with col3:
        pos_pct = sentiment_stats.get("positive_pct", 0)
        render_metric_card(
            "Positive",
            f"{pos_pct:.1f}%",
            icon="👍",
            color="#00C853"
        )
    
    with col4:
        neg_pct = sentiment_stats.get("negative_pct", 0)
        render_metric_card(
            "Negative",
            f"{neg_pct:.1f}%",
            icon="👎",
            color="#FF1744"
        )
    
    # Secondary metrics row
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        neutral_pct = sentiment_stats.get("neutral_pct", 0)
        render_metric_card(
            "Neutral",
            f"{neutral_pct:.1f}%",
            icon="😐",
            color="#FFC107"
        )
    
    with col2:
        confidence = sentiment_stats.get("mean_confidence", 0) * 100
        render_metric_card(
            "Avg. Confidence",
            f"{confidence:.1f}%",
            icon="🎯",
            color="#2196F3"
        )
    
    with col3:
        dominant_lang = language_stats.get("dominant", "en")
        lang_emoji = "🇵🇱" if dominant_lang == "pl" else "🇬🇧" if dominant_lang == "en" else "🌍"
        render_metric_card(
            "Main Language",
            dominant_lang.upper(),
            icon=lang_emoji,
            color="#9C27B0"
        )
    
    with col4:
        std_score = sentiment_stats.get("std_score", 0)
        render_metric_card(
            "Sentiment Spread",
            f"±{std_score:.2f}",
            icon="📊",
            color="#00BCD4"
        )


def render_metric_card(title: str, value: str, icon: str = "", color: str = "#6C63FF"):
    """
    Render a styled metric card.
    
    Args:
        title: Card title
        value: Main value to display
        icon: Emoji icon
        color: Accent color
    """
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, {color}22, {color}11);
        border-left: 4px solid {color};
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    ">
        <div style="font-size: 1.5rem;">{icon}</div>
        <div style="font-size: 1.8rem; font-weight: bold; color: {color};">{value}</div>
        <div style="font-size: 0.85rem; color: #888;">{title}</div>
    </div>
    """, unsafe_allow_html=True)


def render_sentiment_gauge(score: float):
    """
    Render a sentiment gauge visualization.
    
    Args:
        score: Sentiment score (-1 to 1)
    """
    # Convert score to percentage position (0-100)
    position = int((score + 1) * 50)
    
    # Determine color based on score
    if score > 0.2:
        color = "#00C853"
        label = "Positive"
    elif score < -0.2:
        color = "#FF1744"
        label = "Negative"
    else:
        color = "#FFC107"
        label = "Neutral"
    
    st.markdown(f"""
    <div style="text-align: center; padding: 1rem;">
        <h4 style="margin-bottom: 0.5rem;">Overall Sentiment</h4>
        <div style="
            background: linear-gradient(to right, #FF1744, #FFC107, #00C853);
            height: 20px;
            border-radius: 10px;
            position: relative;
            margin: 1rem 0;
        ">
            <div style="
                position: absolute;
                left: {position}%;
                top: -5px;
                transform: translateX(-50%);
                width: 30px;
                height: 30px;
                background: white;
                border-radius: 50%;
                border: 4px solid {color};
                box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            "></div>
        </div>
        <div style="display: flex; justify-content: space-between; color: #888; font-size: 0.8rem;">
            <span>Negative (-1)</span>
            <span>Neutral (0)</span>
            <span>Positive (+1)</span>
        </div>
        <div style="margin-top: 1rem;">
            <span style="
                background: {color};
                color: white;
                padding: 0.5rem 1.5rem;
                border-radius: 20px;
                font-weight: bold;
            ">{label}: {score:.3f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_language_breakdown(language_stats: Dict):
    """
    Render language distribution breakdown.
    
    Args:
        language_stats: Language statistics dictionary
    """
    distribution = language_stats.get("distribution", {})
    percentages = language_stats.get("percentages", {})
    
    if not distribution:
        st.info("No language data available")
        return
    
    st.markdown("#### 🌍 Language Distribution")
    
    for lang, count in distribution.items():
        pct = percentages.get(lang, 0)
        emoji = {"pl": "🇵🇱", "en": "🇬🇧", "other": "🌐"}.get(lang, "🌐")
        label = {"pl": "Polish", "en": "English", "other": "Other"}.get(lang, lang)
        
        st.markdown(f"""
        <div style="margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.2rem;">
                <span>{emoji} {label}</span>
                <span>{count} ({pct:.1f}%)</span>
            </div>
            <div style="
                background: #333;
                border-radius: 4px;
                height: 8px;
                overflow: hidden;
            ">
                <div style="
                    width: {pct}%;
                    height: 100%;
                    background: linear-gradient(90deg, #6C63FF, #2196F3);
                "></div>
            </div>
        </div>
        """, unsafe_allow_html=True)


def format_number(num: int) -> str:
    """Format large numbers with K/M suffix."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)


def get_sentiment_color(score: float) -> str:
    """Get color based on sentiment score."""
    if score > 0.2:
        return "#00C853"
    elif score < -0.2:
        return "#FF1744"
    return "#FFC107"
