"""
Charts component for the dashboard.
Contains Plotly visualizations for sentiment analysis.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import COLORS, PLOTLY_COLORS


def create_sentiment_distribution_chart(sentiment_stats: Dict) -> go.Figure:
    """
    Create a donut chart showing sentiment distribution.
    
    Args:
        sentiment_stats: Sentiment statistics dictionary
        
    Returns:
        Plotly figure
    """
    labels = ["Positive", "Neutral", "Negative"]
    values = [
        sentiment_stats.get("positive_count", 0),
        sentiment_stats.get("neutral_count", 0),
        sentiment_stats.get("negative_count", 0)
    ]
    colors = [COLORS["positive"], COLORS["neutral"], COLORS["negative"]]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.6,
        marker_colors=colors,
        textinfo="label+percent",
        textposition="outside",
        pull=[0.02, 0, 0.02]
    )])
    
    fig.update_layout(
        title=dict(text="Sentiment Distribution", x=0.5, font=dict(size=16)),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=350,
        margin=dict(t=60, b=60, l=20, r=20),
        annotations=[dict(
            text=f"{sentiment_stats.get('count', 0)}<br>comments",
            x=0.5, y=0.5,
            font_size=16,
            showarrow=False
        )]
    )
    
    return fig


def create_sentiment_histogram(df: pd.DataFrame) -> go.Figure:
    """
    Create a histogram of sentiment scores.
    
    Args:
        df: DataFrame with sentiment_score column
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(go.Histogram(
        x=df["sentiment_score"],
        nbinsx=20,
        marker_color=COLORS["primary"],
        opacity=0.8,
        name="Sentiment Score"
    ))
    
    # Add mean line
    mean_score = df["sentiment_score"].mean()
    fig.add_vline(
        x=mean_score,
        line_dash="dash",
        line_color=COLORS["secondary"],
        annotation_text=f"Mean: {mean_score:.2f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=dict(text="Sentiment Score Distribution", x=0.5, font=dict(size=16)),
        xaxis_title="Sentiment Score",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=350,
        margin=dict(t=60, b=60, l=60, r=20),
        xaxis=dict(gridcolor="#333", zerolinecolor="#555"),
        yaxis=dict(gridcolor="#333"),
        bargap=0.05
    )
    
    return fig


def create_sentiment_by_language_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a grouped bar chart comparing sentiment by language.
    
    Args:
        df: DataFrame with language and sentiment_label columns
        
    Returns:
        Plotly figure
    """
    # Group by language and sentiment
    grouped = df.groupby(["language", "sentiment_label"]).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    sentiment_order = ["positive", "neutral", "negative"]
    colors = {"positive": COLORS["positive"], "neutral": COLORS["neutral"], "negative": COLORS["negative"]}
    
    for sentiment in sentiment_order:
        if sentiment in grouped.columns:
            fig.add_trace(go.Bar(
                name=sentiment.capitalize(),
                x=grouped.index,
                y=grouped[sentiment],
                marker_color=colors[sentiment]
            ))
    
    fig.update_layout(
        title=dict(text="Sentiment by Language", x=0.5, font=dict(size=16)),
        yaxis_title="Count",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=380,
        margin=dict(t=60, b=80, l=60, r=20),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
        xaxis=dict(gridcolor="#333", title=None),
        yaxis=dict(gridcolor="#333")
    )
    
    return fig


def create_keyword_bar_chart(keywords: List[Tuple[str, float]], title: str = "Top Keywords") -> go.Figure:
    """
    Create a horizontal bar chart of top keywords.
    
    Args:
        keywords: List of (word, score) tuples
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not keywords:
        return create_empty_chart("No keywords found")
    
    words = [k[0] for k in keywords[:15]]
    scores = [k[1] for k in keywords[:15]]
    
    # Reverse for horizontal bar chart
    words = words[::-1]
    scores = scores[::-1]
    
    fig = go.Figure(go.Bar(
        x=scores,
        y=words,
        orientation="h",
        marker=dict(
            color=scores,
            colorscale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
        ),
        text=[f"{s:.3f}" for s in scores],
        textposition="auto"
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        xaxis_title="TF-IDF Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=400,
        margin=dict(t=60, b=60, l=120, r=20),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333")
    )
    
    return fig


def create_ngram_treemap(ngrams: List[Tuple[str, int]], title: str = "N-gram Analysis") -> go.Figure:
    """
    Create a treemap visualization of n-grams.
    
    Args:
        ngrams: List of (ngram, count) tuples
        title: Chart title
        
    Returns:
        Plotly figure
    """
    if not ngrams:
        return create_empty_chart("No n-grams found")
    
    labels = [n[0] for n in ngrams[:20]]
    values = [n[1] for n in ngrams[:20]]
    
    fig = go.Figure(go.Treemap(
        labels=labels,
        parents=[""] * len(labels),
        values=values,
        textinfo="label+value",
        marker=dict(
            colors=values,
            colorscale="Viridis"
        )
    ))
    
    fig.update_layout(
        title=dict(text=title, x=0.5, font=dict(size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=400,
        margin=dict(t=60, b=20, l=20, r=20)
    )
    
    return fig


def create_topic_scatter(
    doc_topics: List[List[float]],
    topics: List[Dict],
    df: pd.DataFrame
) -> go.Figure:
    """
    Create a scatter plot of documents by topic.
    
    Args:
        doc_topics: Document-topic distributions
        topics: Topic information
        df: DataFrame with document data
        
    Returns:
        Plotly figure
    """
    if not doc_topics or not topics:
        return create_empty_chart("Not enough data for topic modeling")
    
    # Get dominant topic for each document
    doc_topics_array = np.array(doc_topics)
    dominant_topics = doc_topics_array.argmax(axis=1)
    
    # Use first two topic dimensions for scatter
    if doc_topics_array.shape[1] >= 2:
        x = doc_topics_array[:, 0]
        y = doc_topics_array[:, 1]
    else:
        x = doc_topics_array[:, 0]
        y = np.zeros_like(x)
    
    # Create labels for hover
    topic_labels = [topics[t]["label"] for t in dominant_topics]
    
    fig = go.Figure()
    
    # Add scatter for each topic
    for topic_idx in range(len(topics)):
        mask = dominant_topics == topic_idx
        fig.add_trace(go.Scatter(
            x=x[mask],
            y=y[mask],
            mode="markers",
            name=topics[topic_idx]["label"][:30],
            marker=dict(
                size=8,
                color=PLOTLY_COLORS[topic_idx % len(PLOTLY_COLORS)],
                opacity=0.7
            ),
            hovertemplate=f"Topic: {topics[topic_idx]['label']}<extra></extra>"
        ))
    
    fig.update_layout(
        title=dict(text="Topic Clusters", x=0.5, font=dict(size=16)),
        xaxis_title="Topic Dimension 1",
        yaxis_title="Topic Dimension 2",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=400,
        margin=dict(t=60, b=60, l=60, r=20),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.3,
            xanchor="center",
            x=0.5,
            font=dict(size=10)
        )
    )
    
    return fig


def create_comment_length_sentiment_scatter(df: pd.DataFrame) -> go.Figure:
    """
    Create a scatter plot of comment length vs sentiment.
    
    Args:
        df: DataFrame with word_count and sentiment_score columns
        
    Returns:
        Plotly figure
    """
    if "word_count" not in df.columns:
        df["word_count"] = df["text"].str.split().str.len()
    
    # Sample if too many points
    plot_df = df.sample(min(500, len(df))) if len(df) > 500 else df
    
    fig = go.Figure()
    
    # Color by sentiment
    colors = plot_df["sentiment_label"].map({
        "positive": COLORS["positive"],
        "neutral": COLORS["neutral"],
        "negative": COLORS["negative"]
    })
    
    fig.add_trace(go.Scatter(
        x=plot_df["word_count"],
        y=plot_df["sentiment_score"],
        mode="markers",
        marker=dict(
            color=colors,
            size=6,
            opacity=0.6
        ),
        hovertemplate="Words: %{x}<br>Sentiment: %{y:.2f}<extra></extra>"
    ))
    
    # Add trend line
    z = np.polyfit(plot_df["word_count"], plot_df["sentiment_score"], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_df["word_count"].min(), plot_df["word_count"].max(), 100)
    
    fig.add_trace(go.Scatter(
        x=x_line,
        y=p(x_line),
        mode="lines",
        name="Trend",
        line=dict(color=COLORS["secondary"], dash="dash", width=2)
    ))
    
    fig.update_layout(
        title=dict(text="Comment Length vs Sentiment", x=0.5, font=dict(size=16)),
        xaxis_title="Word Count",
        yaxis_title="Sentiment Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=350,
        margin=dict(t=60, b=60, l=60, r=20),
        showlegend=False,
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333")
    )
    
    return fig


def create_topics_bar_chart(topics: List[Dict]) -> go.Figure:
    """
    Create a bar chart showing topic words.
    
    Args:
        topics: List of topic dictionaries
        
    Returns:
        Plotly figure
    """
    if not topics:
        return create_empty_chart("No topics found")
    
    fig = make_subplots(
        rows=1, cols=len(topics),
        subplot_titles=[f"Topic {t['topic_id']+1}" for t in topics]
    )
    
    for i, topic in enumerate(topics):
        words = [w["word"] for w in topic["words"][:8]][::-1]
        weights = [w["weight"] for w in topic["words"][:8]][::-1]
        
        fig.add_trace(
            go.Bar(
                x=weights,
                y=words,
                orientation="h",
                marker_color=PLOTLY_COLORS[i % len(PLOTLY_COLORS)],
                showlegend=False
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title=dict(text="Topic Keywords", x=0.5, font=dict(size=16)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=400,
        margin=dict(t=80, b=40, l=100, r=20)
    )
    
    fig.update_xaxes(gridcolor="#333")
    fig.update_yaxes(gridcolor="#333")
    
    return fig


def create_empty_chart(message: str = "No data available") -> go.Figure:
    """Create an empty chart with a message."""
    fig = go.Figure()
    
    fig.add_annotation(
        x=0.5, y=0.5,
        text=message,
        showarrow=False,
        font=dict(size=16, color="#888"),
        xref="paper", yref="paper"
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    
    return fig


def create_emoji_chart(emoji_data: Dict) -> go.Figure:
    """
    Create a chart showing emoji usage.
    
    Args:
        emoji_data: Dictionary with emoji statistics
        
    Returns:
        Plotly figure
    """
    top_emojis = emoji_data.get("top_emojis", [])
    
    if not top_emojis:
        return create_empty_chart("No emojis found")
    
    emojis = [e[0] for e in top_emojis[:10]]
    counts = [e[1] for e in top_emojis[:10]]
    
    fig = go.Figure(go.Bar(
        x=emojis,
        y=counts,
        marker_color=PLOTLY_COLORS,
        text=counts,
        textposition="auto"
    ))
    
    fig.update_layout(
        title=dict(text="Top Emojis Used", x=0.5, font=dict(size=16)),
        xaxis_title="Emoji",
        yaxis_title="Count",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA", size=20),
        height=300,
        margin=dict(t=60, b=60, l=60, r=20),
        xaxis=dict(gridcolor="#333", tickfont=dict(size=24)),
        yaxis=dict(gridcolor="#333")
    )
    
    return fig


def create_sentiment_over_time_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing sentiment trends over time.
    
    Args:
        df: DataFrame with published_at and sentiment_score columns
        
    Returns:
        Plotly figure
    """
    if "published_at" not in df.columns or df["published_at"].isna().all():
        return create_empty_chart("No timestamp data available")
    
    # Sort by date
    df_sorted = df.sort_values("published_at")
    
    # Resample by day and calculate rolling average
    df_sorted = df_sorted.set_index("published_at")
    
    # Create daily aggregates
    daily_sentiment = df_sorted["sentiment_score"].resample("D").agg(["mean", "count"]).dropna()
    
    if len(daily_sentiment) < 2:
        # If less than 2 days, show raw data with rolling average
        df_sorted = df_sorted.reset_index()
        df_sorted["rolling_sentiment"] = df_sorted["sentiment_score"].rolling(window=min(10, len(df_sorted)), min_periods=1).mean()
        
        fig = go.Figure()
        
        # Scatter for individual comments
        fig.add_trace(go.Scatter(
            x=df_sorted["published_at"],
            y=df_sorted["sentiment_score"],
            mode="markers",
            marker=dict(
                size=6,
                color=df_sorted["sentiment_score"],
                colorscale=[[0, COLORS["negative"]], [0.5, COLORS["neutral"]], [1, COLORS["positive"]]],
                opacity=0.5
            ),
            name="Comments",
            hovertemplate="Date: %{x}<br>Sentiment: %{y:.2f}<extra></extra>"
        ))
        
        # Line for rolling average
        fig.add_trace(go.Scatter(
            x=df_sorted["published_at"],
            y=df_sorted["rolling_sentiment"],
            mode="lines",
            line=dict(color=COLORS["primary"], width=3),
            name="Rolling Average"
        ))
    else:
        fig = go.Figure()
        
        # Area chart for daily sentiment
        fig.add_trace(go.Scatter(
            x=daily_sentiment.index,
            y=daily_sentiment["mean"],
            mode="lines+markers",
            fill="tozeroy",
            line=dict(color=COLORS["primary"], width=2),
            marker=dict(size=8),
            name="Daily Avg Sentiment",
            hovertemplate="Date: %{x}<br>Avg Sentiment: %{y:.2f}<br>Comments: %{customdata}<extra></extra>",
            customdata=daily_sentiment["count"]
        ))
        
        # Add neutral line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    
    fig.update_layout(
        title=dict(text="Sentiment Trend Over Time", x=0.5, font=dict(size=16)),
        yaxis_title="Sentiment Score",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=380,
        margin=dict(t=60, b=80, l=60, r=20),
        xaxis=dict(gridcolor="#333", title=None),
        yaxis=dict(gridcolor="#333", range=[-1.1, 1.1]),
        showlegend=True,
        legend=dict(orientation="h", yanchor="top", y=-0.12, xanchor="center", x=0.5)
    )
    
    return fig


def create_engagement_sentiment_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing correlation between engagement (likes) and sentiment.
    
    Args:
        df: DataFrame with like_count and sentiment columns
        
    Returns:
        Plotly figure
    """
    if "like_count" not in df.columns:
        return create_empty_chart("No engagement data available")
    
    # Group by sentiment and calculate average likes
    engagement_by_sentiment = df.groupby("sentiment_label").agg({
        "like_count": ["mean", "sum", "count"]
    }).round(2)
    
    engagement_by_sentiment.columns = ["avg_likes", "total_likes", "count"]
    engagement_by_sentiment = engagement_by_sentiment.reset_index()
    
    # Sort by sentiment order
    order = {"positive": 0, "neutral": 1, "negative": 2}
    engagement_by_sentiment["sort_order"] = engagement_by_sentiment["sentiment_label"].map(order)
    engagement_by_sentiment = engagement_by_sentiment.sort_values("sort_order")
    
    colors_map = {"positive": COLORS["positive"], "neutral": COLORS["neutral"], "negative": COLORS["negative"]}
    colors = [colors_map.get(s, COLORS["primary"]) for s in engagement_by_sentiment["sentiment_label"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=engagement_by_sentiment["sentiment_label"].str.capitalize(),
        y=engagement_by_sentiment["avg_likes"],
        marker_color=colors,
        text=[f"{v:.1f}" for v in engagement_by_sentiment["avg_likes"]],
        textposition="auto",
        hovertemplate="<b>%{x}</b><br>Avg Likes: %{y:.1f}<br>Total: %{customdata[0]}<br>Comments: %{customdata[1]}<extra></extra>",
        customdata=list(zip(engagement_by_sentiment["total_likes"], engagement_by_sentiment["count"]))
    ))
    
    fig.update_layout(
        title=dict(text="Average Likes by Sentiment", x=0.5, font=dict(size=16)),
        xaxis_title=None,
        yaxis_title="Average Likes",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=300,
        margin=dict(t=60, b=40, l=60, r=20),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333")
    )
    
    return fig


def create_top_commenters_chart(df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """
    Create a chart showing top commenters by activity.
    
    Args:
        df: DataFrame with author column
        top_n: Number of top commenters to show
        
    Returns:
        Plotly figure
    """
    if "author" not in df.columns:
        return create_empty_chart("No author data available")
    
    # Count comments per author
    author_stats = df.groupby("author").agg({
        "text": "count",
        "sentiment_score": "mean",
        "like_count": "sum"
    }).rename(columns={"text": "comment_count", "sentiment_score": "avg_sentiment"})
    
    author_stats = author_stats.sort_values("comment_count", ascending=False).head(top_n)
    author_stats = author_stats.reset_index()
    
    # Reverse for horizontal bar
    author_stats = author_stats.iloc[::-1]
    
    # Color based on sentiment
    colors = [COLORS["positive"] if s > 0.1 else (COLORS["negative"] if s < -0.1 else COLORS["neutral"]) 
              for s in author_stats["avg_sentiment"]]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=author_stats["comment_count"],
        y=author_stats["author"],
        orientation="h",
        marker_color=colors,
        text=author_stats["comment_count"],
        textposition="auto",
        hovertemplate="<b>%{y}</b><br>Comments: %{x}<br>Avg Sentiment: %{customdata[0]:.2f}<br>Total Likes: %{customdata[1]}<extra></extra>",
        customdata=list(zip(author_stats["avg_sentiment"], author_stats["like_count"]))
    ))
    
    fig.update_layout(
        title=dict(text="Most Active Commenters", x=0.5, font=dict(size=16)),
        xaxis_title="Number of Comments",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=350,
        margin=dict(t=60, b=40, l=150, r=20),
        xaxis=dict(gridcolor="#333"),
        yaxis=dict(gridcolor="#333")
    )
    
    return fig


def create_sentiment_confidence_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a chart showing sentiment confidence distribution.
    
    Args:
        df: DataFrame with sentiment_confidence column
        
    Returns:
        Plotly figure
    """
    if "sentiment_confidence" not in df.columns:
        return create_empty_chart("No confidence data available")
    
    # Create histogram of confidence scores
    fig = go.Figure()
    
    for sentiment, color in [("positive", COLORS["positive"]), ("neutral", COLORS["neutral"]), ("negative", COLORS["negative"])]:
        subset = df[df["sentiment_label"] == sentiment]["sentiment_confidence"]
        if len(subset) > 0:
            fig.add_trace(go.Histogram(
                x=subset,
                name=sentiment.capitalize(),
                marker_color=color,
                opacity=0.7,
                nbinsx=20
            ))
    
    fig.update_layout(
        title=dict(text="Sentiment Confidence Distribution", x=0.5, font=dict(size=16)),
        yaxis_title="Count",
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=330,
        margin=dict(t=60, b=70, l=60, r=20),
        xaxis=dict(gridcolor="#333", title=None),
        yaxis=dict(gridcolor="#333"),
        legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5)
    )
    
    return fig
