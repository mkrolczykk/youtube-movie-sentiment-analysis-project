"""
YouTube Sentiment Analyzer - Main Streamlit Application

An advanced sentiment analysis dashboard for YouTube video comments
with multi-language support (Polish/English).
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DASHBOARD_CONFIG, COLORS
from src.youtube_client import YouTubeClient, YouTubeAPIError, CommentsDisabledError
from src.language_detector import detect_language, get_language_statistics, categorize_by_language
from src.sentiment_analyzer import SentimentAnalyzer
from src.text_preprocessor import TextPreprocessor
from src.text_analytics import TextAnalytics

from dashboard.components.sidebar import render_sidebar, add_to_history
from dashboard.components.metrics import (
    render_video_header,
    render_sentiment_metrics,
    render_sentiment_gauge,
    render_language_breakdown
)
from dashboard.components.charts import (
    create_sentiment_distribution_chart,
    create_sentiment_histogram,
    create_sentiment_by_language_chart,
    create_keyword_bar_chart,
    create_ngram_treemap,
    create_topic_scatter,
    create_comment_length_sentiment_scatter,
    create_topics_bar_chart,
    create_emoji_chart
)
from dashboard.components.wordcloud import render_wordcloud_section, render_comparison_wordclouds


# Page configuration
st.set_page_config(
    page_title=DASHBOARD_CONFIG["page_title"],
    page_icon=DASHBOARD_CONFIG["page_icon"],
    layout=DASHBOARD_CONFIG["layout"],
    initial_sidebar_state=DASHBOARD_CONFIG["initial_sidebar_state"]
)

# Load custom CSS
css_path = Path(__file__).parent / "styles" / "custom.css"
if css_path.exists():
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "analysis_complete" not in st.session_state:
        st.session_state["analysis_complete"] = False
    if "df" not in st.session_state:
        st.session_state["df"] = None
    if "video_metadata" not in st.session_state:
        st.session_state["video_metadata"] = None
    if "sentiment_stats" not in st.session_state:
        st.session_state["sentiment_stats"] = None
    if "language_stats" not in st.session_state:
        st.session_state["language_stats"] = None


@st.cache_resource
def load_sentiment_analyzer():
    """Load and cache the sentiment analyzer."""
    return SentimentAnalyzer()


@st.cache_resource
def load_text_analytics():
    """Load and cache text analytics."""
    return TextAnalytics()


def run_analysis(video_url: str, num_comments: int, api_key: str):
    """
    Run the full analysis pipeline.
    
    Args:
        video_url: YouTube video URL
        num_comments: Number of comments to fetch
        api_key: YouTube API key
    """
    # Initialize components
    client = YouTubeClient(api_key)
    analyzer = load_sentiment_analyzer()
    analytics = load_text_analytics()
    
    # Extract video ID
    video_id = YouTubeClient.extract_video_id(video_url)
    if not video_id:
        st.error("❌ Invalid YouTube URL. Please check and try again.")
        return
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        # Step 1: Fetch video metadata
        status_text.text("📹 Fetching video information...")
        progress_bar.progress(10)
        
        video_metadata = client.get_video_metadata(video_id)
        if not video_metadata:
            st.error("❌ Could not fetch video information. Check if the video exists.")
            return
        
        st.session_state["video_metadata"] = video_metadata
        
        # Step 2: Fetch comments
        status_text.text(f"💬 Fetching {num_comments} comments...")
        progress_bar.progress(25)
        
        include_replies = st.session_state.get("include_replies", False)
        comments = client.fetch_comments(video_id, num_comments, include_replies)
        
        if not comments:
            st.warning("⚠️ No comments found for this video.")
            return
        
        # Step 3: Detect languages
        status_text.text("🌍 Detecting languages...")
        progress_bar.progress(40)
        
        texts = [c["text"] for c in comments]
        languages = []
        for text in texts:
            lang, conf = detect_language(text)
            languages.append(lang)
        
        language_stats = get_language_statistics(texts)
        st.session_state["language_stats"] = language_stats
        
        # Apply language filter if set
        language_filter = st.session_state.get("language_filter", "All Languages")
        if language_filter == "Polish Only":
            filtered_indices = [i for i, lang in enumerate(languages) if lang == "pl"]
            comments = [comments[i] for i in filtered_indices]
            texts = [texts[i] for i in filtered_indices]
            languages = [languages[i] for i in filtered_indices]
        elif language_filter == "English Only":
            filtered_indices = [i for i, lang in enumerate(languages) if lang == "en"]
            comments = [comments[i] for i in filtered_indices]
            texts = [texts[i] for i in filtered_indices]
            languages = [languages[i] for i in filtered_indices]
        
        if not comments:
            st.warning("⚠️ No comments match the language filter.")
            return
        
        # Step 4: Sentiment analysis
        status_text.text("🧠 Analyzing sentiment (this may take a moment)...")
        progress_bar.progress(60)
        
        sentiment_results = analyzer.analyze_batch(texts)
        
        # Step 5: Build DataFrame
        status_text.text("📊 Processing results...")
        progress_bar.progress(80)
        
        df = pd.DataFrame({
            "text": texts,
            "author": [c["author"] for c in comments],
            "published_at": [c["published_at"] for c in comments],
            "like_count": [c["like_count"] for c in comments],
            "language": languages,
            "sentiment_score": [r.score for r in sentiment_results],
            "sentiment_label": [r.label for r in sentiment_results],
            "sentiment_confidence": [r.confidence for r in sentiment_results],
            "word_count": [len(t.split()) for t in texts]
        })
        
        # Calculate statistics
        sentiment_stats = analyzer.get_sentiment_statistics(sentiment_results)
        
        st.session_state["df"] = df
        st.session_state["sentiment_stats"] = sentiment_stats
        st.session_state["analysis_complete"] = True
        
        # Add to history
        add_to_history(
            video_metadata["title"],
            sentiment_stats["mean_score"],
            video_id
        )
        
        # Complete
        progress_bar.progress(100)
        status_text.text("✅ Analysis complete!")
        
    except CommentsDisabledError:
        st.error("❌ Comments are disabled for this video.")
    except YouTubeAPIError as e:
        st.error(f"❌ YouTube API Error: {e}")
    except Exception as e:
        st.error(f"❌ An error occurred: {e}")
        raise e


def render_results():
    """Render the analysis results."""
    df = st.session_state["df"]
    video_metadata = st.session_state["video_metadata"]
    sentiment_stats = st.session_state["sentiment_stats"]
    language_stats = st.session_state["language_stats"]
    analytics = load_text_analytics()
    
    # Video header
    render_video_header(video_metadata)
    
    st.divider()
    
    # Sentiment gauge
    render_sentiment_gauge(sentiment_stats["mean_score"])
    
    st.divider()
    
    # Metrics cards
    render_sentiment_metrics(sentiment_stats, language_stats)
    
    st.divider()
    
    # Charts section
    st.markdown("## 📊 Sentiment Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_sentiment_distribution_chart(sentiment_stats)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_sentiment_histogram(df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Language analysis
    col1, col2 = st.columns(2)
    
    with col1:
        fig = create_sentiment_by_language_chart(df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_comment_length_sentiment_scatter(df)
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Word clouds section
    render_wordcloud_section(
        df["text"].tolist(),
        df["sentiment_label"].tolist(),
        language_stats.get("dominant", "en"),
        analytics
    )
    
    st.divider()
    
    # Text analytics section
    st.markdown("## 📝 Text Analysis")
    
    # Determine dominant language for analysis
    dominant_lang = language_stats.get("dominant", "en")
    
    tab_keywords, tab_ngrams, tab_topics, tab_emoji = st.tabs([
        "🔑 Keywords",
        "📊 N-grams",
        "🎯 Topics",
        "😀 Emojis"
    ])
    
    with tab_keywords:
        keywords = analytics.extract_keywords(df["text"].tolist(), top_k=15, language=dominant_lang)
        col1, col2 = st.columns([2, 1])
        with col1:
            fig = create_keyword_bar_chart(keywords, "Top Keywords (TF-IDF)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("#### 📋 Keyword List")
            for word, score in keywords[:10]:
                st.markdown(f"- **{word}**: {score:.4f}")
    
    with tab_ngrams:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Bigrams")
            bigrams = analytics.extract_ngrams(df["text"].tolist(), n=2, top_k=15, language=dominant_lang)
            fig = create_ngram_treemap(bigrams, "Most Common Bigrams")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Trigrams")
            trigrams = analytics.extract_ngrams(df["text"].tolist(), n=3, top_k=15, language=dominant_lang)
            fig = create_ngram_treemap(trigrams, "Most Common Trigrams")
            st.plotly_chart(fig, use_container_width=True)
    
    with tab_topics:
        st.markdown("#### 🎯 Topic Modeling (LDA)")
        
        if len(df) >= 10:
            topics = analytics.extract_topics(
                df["text"].tolist(),
                n_topics=min(5, len(df) // 10),
                language=dominant_lang
            )
            
            if topics:
                fig = create_topics_bar_chart(topics)
                st.plotly_chart(fig, use_container_width=True)
                
                # Topic details
                st.markdown("#### Topic Details")
                for topic in topics:
                    with st.expander(f"Topic {topic['topic_id'] + 1}"):
                        words = [w["word"] for w in topic["words"]]
                        st.markdown(", ".join(words))
            else:
                st.info("Not enough data for topic modeling. Try fetching more comments.")
        else:
            st.info("Need at least 10 comments for topic modeling.")
    
    with tab_emoji:
        emoji_data = analytics.get_emoji_analysis(df["text"].tolist())
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = create_emoji_chart(emoji_data)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Emoji Statistics")
            st.metric("Total Emojis", emoji_data.get("total_emojis", 0))
            st.metric("Unique Emojis", emoji_data.get("unique_emojis", 0))
            st.metric("Comments with Emojis", f"{emoji_data.get('emoji_percentage', 0):.1f}%")
    
    st.divider()
    
    # Language comparison (if both PL and EN present)
    lang_distribution = language_stats.get("distribution", {})
    if lang_distribution.get("pl", 0) > 0 and lang_distribution.get("en", 0) > 0:
        texts_pl = df[df["language"] == "pl"]["text"].tolist()
        texts_en = df[df["language"] == "en"]["text"].tolist()
        render_comparison_wordclouds(texts_pl, texts_en, analytics)
        st.divider()
    
    # Comment explorer
    st.markdown("## 💬 Comment Explorer")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        sentiment_filter = st.multiselect(
            "Filter by Sentiment",
            options=["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"]
        )
    
    with col2:
        lang_filter = st.multiselect(
            "Filter by Language",
            options=list(lang_distribution.keys()),
            default=list(lang_distribution.keys())
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort by",
            options=["sentiment_score", "like_count", "word_count"],
            format_func=lambda x: {"sentiment_score": "Sentiment", "like_count": "Likes", "word_count": "Length"}[x]
        )
    
    # Filter and display
    filtered_df = df[
        (df["sentiment_label"].isin(sentiment_filter)) &
        (df["language"].isin(lang_filter))
    ].sort_values(sort_by, ascending=False)
    
    # Display styled dataframe
    st.dataframe(
        filtered_df[["text", "author", "sentiment_score", "sentiment_label", "language", "like_count"]].head(100),
        use_container_width=True,
        column_config={
            "text": st.column_config.TextColumn("Comment", width="large"),
            "author": st.column_config.TextColumn("Author", width="medium"),
            "sentiment_score": st.column_config.NumberColumn("Score", format="%.2f"),
            "sentiment_label": st.column_config.TextColumn("Sentiment"),
            "language": st.column_config.TextColumn("Lang"),
            "like_count": st.column_config.NumberColumn("Likes")
        }
    )
    
    # Export section
    st.divider()
    st.markdown("## 💾 Export Results")
    
    col1, col2 = st.columns(2)
    
    with col1:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Dataset (CSV)",
            data=csv,
            file_name=f"sentiment_analysis_{video_metadata.get('video_id', 'export')}.csv",
            mime="text/csv"
        )
    
    with col2:
        summary = {
            "video_title": video_metadata.get("title"),
            "video_id": video_metadata.get("video_id"),
            "total_comments": sentiment_stats.get("count"),
            "mean_sentiment": sentiment_stats.get("mean_score"),
            "positive_pct": sentiment_stats.get("positive_pct"),
            "negative_pct": sentiment_stats.get("negative_pct"),
            "neutral_pct": sentiment_stats.get("neutral_pct"),
            "dominant_language": language_stats.get("dominant")
        }
        summary_df = pd.DataFrame([summary])
        summary_csv = summary_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Summary (CSV)",
            data=summary_csv,
            file_name=f"summary_{video_metadata.get('video_id', 'export')}.csv",
            mime="text/csv"
        )


def render_welcome():
    """Render the welcome screen when no analysis is loaded."""
    st.markdown("""
    <div style="text-align: center; padding: 4rem 2rem;">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">🎬 YouTube Sentiment Analyzer</h1>
        <p style="font-size: 1.2rem; color: #888; max-width: 600px; margin: 0 auto;">
            Discover what viewers really think about any YouTube video using advanced 
            sentiment analysis powered by multilingual transformers.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    features = [
        ("🌍", "Multi-language", "Polish & English support"),
        ("🧠", "AI-Powered", "Transformer-based analysis"),
        ("📊", "Rich Analytics", "Charts, word clouds, topics"),
        ("💬", "Deep Insights", "Explore every comment"),
    ]
    
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div style="
                text-align: center;
                padding: 1.5rem;
                background: linear-gradient(135deg, #1a1a2e, #16213e);
                border-radius: 12px;
                border: 1px solid rgba(108, 99, 255, 0.3);
            ">
                <div style="font-size: 2.5rem;">{icon}</div>
                <h3 style="margin: 0.5rem 0;">{title}</h3>
                <p style="color: #888; font-size: 0.9rem; margin: 0;">{desc}</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Getting started
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, rgba(108, 99, 255, 0.1), rgba(33, 150, 243, 0.1));
        border-radius: 12px;
        padding: 2rem;
        border-left: 4px solid #6C63FF;
    ">
        <h3>🚀 Getting Started</h3>
        <ol style="color: #CCC; line-height: 2;">
            <li>Enter your <b>YouTube Data API v3 key</b> in the sidebar</li>
            <li>Paste any <b>YouTube video URL</b></li>
            <li>Choose the number of comments to analyze</li>
            <li>Click <b>Analyze Comments</b> and explore the results!</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
    <div style="text-align: center; padding: 3rem 0; color: #666;">
        <p>Made for <b>Eksploracja Danych Tekstowych</b> course</p>
        <p>Data Science • Master's Program • 2nd Year</p>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main application entry point."""
    # Initialize
    init_session_state()
    
    # Render sidebar and get inputs
    video_url, num_comments, api_key, analyze_clicked = render_sidebar()
    
    # Main content area
    if analyze_clicked:
        run_analysis(video_url, num_comments, api_key)
    
    # Show results or welcome screen
    if st.session_state["analysis_complete"] and st.session_state["df"] is not None:
        render_results()
    else:
        render_welcome()


if __name__ == "__main__":
    main()
