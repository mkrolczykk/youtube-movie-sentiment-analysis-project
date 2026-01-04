"""
Sidebar component for the dashboard.
Handles user inputs and configuration.
"""

import streamlit as st
from typing import Tuple, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.youtube_client import YouTubeClient
from src.config import DEFAULT_SETTINGS


def render_sidebar() -> Tuple[Optional[str], int, Optional[str], bool]:
    """
    Render the sidebar with input controls.
    
    Returns:
        Tuple of (video_url, num_comments, api_key, analyze_clicked)
    """
    with st.sidebar:
        # Header with logo
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #6C63FF; margin: 0;">🎬</h1>
            <h2 style="margin: 0.5rem 0;">YouTube Sentiment</h2>
            <p style="color: #888; font-size: 0.9rem;">Analyzer</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # API Key Input
        st.subheader("🔑 API Configuration")
        api_key = st.text_input(
            "YouTube API Key",
            type="password",
            help="Enter your YouTube Data API v3 key. Get one at console.cloud.google.com",
            placeholder="AIza..."
        )
        
        # Validate API key button
        if api_key and st.button("Validate Key", use_container_width=True):
            with st.spinner("Validating..."):
                client = YouTubeClient(api_key)
                is_valid, message = client.validate_api_key()
                if is_valid:
                    st.success("✅ API key is valid!")
                else:
                    st.error(f"❌ {message}")
        
        st.divider()
        
        # Video Input
        st.subheader("🎥 Video Selection")
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste any YouTube video URL"
        )
        
        # Validate URL and show video info
        if video_url and api_key:
            video_id = YouTubeClient.extract_video_id(video_url)
            if video_id:
                st.caption(f"Video ID: `{video_id}`")
            else:
                st.warning("⚠️ Invalid YouTube URL")
        
        st.divider()
        
        # Analysis Settings
        st.subheader("⚙️ Analysis Settings")
        
        num_comments = st.slider(
            "Number of Comments",
            min_value=DEFAULT_SETTINGS["min_comments"],
            max_value=DEFAULT_SETTINGS["max_comments"],
            value=DEFAULT_SETTINGS["num_comments"],
            step=10,
            help="More comments = more accurate analysis, but slower"
        )
        
        # Advanced options in expander
        with st.expander("🔧 Advanced Options"):
            include_replies = st.checkbox(
                "Include Replies",
                value=False,
                help="Also analyze reply comments"
            )
            
            language_filter = st.selectbox(
                "Language Filter",
                options=["All Languages", "Polish Only", "English Only"],
                index=0
            )
            
            st.session_state["include_replies"] = include_replies
            st.session_state["language_filter"] = language_filter
        
        st.divider()
        
        # Analyze Button
        analyze_clicked = st.button(
            "🚀 Analyze Comments",
            use_container_width=True,
            type="primary",
            disabled=not (video_url and api_key)
        )
        
        # Instructions
        if not api_key:
            st.info("👆 Enter your API key to start")
        elif not video_url:
            st.info("👆 Enter a YouTube URL")
        
        # Footer
        st.divider()
        st.markdown("""
        <div style="text-align: center; color: #666; font-size: 0.8rem;">
            <p>Made for <b>EDT</b> course</p>
            <p>Data Science, 2nd year MSc</p>
        </div>
        """, unsafe_allow_html=True)
    
    return video_url, num_comments, api_key, analyze_clicked


def render_analysis_history():
    """Render the analysis history in the sidebar."""
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []
    
    history = st.session_state["analysis_history"]
    
    if history:
        with st.sidebar:
            st.subheader("📜 Recent Analyses")
            for item in history[-5:]:  # Show last 5
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.caption(item["title"][:30] + "..." if len(item["title"]) > 30 else item["title"])
                with col2:
                    sentiment_icon = "😊" if item["avg_sentiment"] > 0.2 else "😐" if item["avg_sentiment"] > -0.2 else "😞"
                    st.caption(sentiment_icon)


def add_to_history(video_title: str, avg_sentiment: float, video_id: str):
    """Add an analysis to the history."""
    if "analysis_history" not in st.session_state:
        st.session_state["analysis_history"] = []
    
    st.session_state["analysis_history"].append({
        "title": video_title,
        "avg_sentiment": avg_sentiment,
        "video_id": video_id
    })
