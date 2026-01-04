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
        xaxis_title="Language",
        yaxis_title="Count",
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#FAFAFA"),
        height=350,
        margin=dict(t=60, b=60, l=60, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        xaxis=dict(gridcolor="#333"),
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
