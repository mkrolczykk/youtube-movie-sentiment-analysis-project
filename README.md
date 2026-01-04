# YouTube Movie Sentiment Analysis

Sentiment analysis application for YouTube video comments with multi-language support (Polish/English), featuring an interactive Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![Transformers](https://img.shields.io/badge/Transformers-4.35%2B-orange)

## Project Overview

This application was created for the **Eksploracja Danych Tekstowych (Text Data Mining)** course as part of the Data Science Master's program.

### Features

- 🌍 **Multi-language Support**: Automatic detection and analysis of Polish and English comments
- 🧠 **AI-Powered Sentiment Analysis**: Uses multilingual BERT transformer model
- 📊 **Rich Visualizations**: Interactive Plotly charts, word clouds, topic modeling
- 💬 **Comment Explorer**: Filter, sort, and explore individual comments
- 📈 **Advanced Analytics**: TF-IDF keywords, n-grams, LDA topics, emoji analysis
- 💾 **Export Functionality**: Download results as CSV

## Quick Start

### Prerequisites

- Python 3.9 or higher
- YouTube Data API v3 key ([Get one here](https://console.cloud.google.com/))

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd youtube-movie-sentiment-analysis-project
   ```

2. **Create and activate virtual environment**

   ```bash
   python -m venv .venv
   
   # Windows
   .\.venv\Scripts\Activate.ps1
   
   # Linux/macOS
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy language models**

   ```bash
   python -m spacy download en_core_web_md
   python -m spacy download pl_core_news_md
   ```

5. **Run the application**

   ```bash
   streamlit run dashboard/app.py
   ```

6. **Open in browser**
   Navigate to `http://localhost:8501`

## YouTube API Key Setup

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project
3. Enable "YouTube Data API v3"
4. Create credentials → API Key
5. Enter the key in the application sidebar

## Project Structure

```
youtube-movie-sentiment-analysis-project/
├── src/                          # Core modules
│   ├── config.py                 # Configuration & constants
│   ├── youtube_client.py         # YouTube API integration
│   ├── language_detector.py      # Language detection
│   ├── text_preprocessor.py      # Text cleaning & tokenization
│   ├── sentiment_analyzer.py     # Transformer-based sentiment analysis
│   └── text_analytics.py         # Word clouds, topics, n-grams
├── dashboard/                    # Streamlit application
│   ├── app.py                    # Main application
│   ├── components/               # UI components
│   │   ├── sidebar.py            # Input controls
│   │   ├── metrics.py            # KPI cards
│   │   ├── charts.py             # Plotly visualizations
│   │   └── wordcloud.py          # Word cloud display
│   └── styles/
│       └── custom.css            # Custom styling
├── requirements.txt
├── .env.example
└── README.md
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Web Framework | Streamlit |
| Sentiment Analysis | HuggingFace Transformers (multilingual BERT) |
| NLP | spaCy, NLTK |
| Visualization | Plotly, Matplotlib, WordCloud |
| Topic Modeling | scikit-learn (LDA) |
| API | Google YouTube Data API v3 |

## Dashboard Sections

### 1. Sentiment Overview

- Overall sentiment gauge
- Distribution pie chart
- Language breakdown
- Key metrics cards

### 2. Visualizations

- Sentiment histogram
- Sentiment by language comparison
- Comment length vs sentiment scatter

### 3. Word Clouds

- All comments
- Positive comments (green)
- Negative comments (red)
- Neutral comments (blue)
- Polish vs English comparison

### 4. Text Analytics

- **Keywords**: TF-IDF based extraction
- **N-grams**: Bigram and trigram analysis
- **Topics**: LDA topic modeling
- **Emojis**: Emoji usage statistics

### 5. Comment Explorer

- Filterable data table
- Sort by sentiment, likes, length
- Filter by sentiment and language

## Usage Example

1. Enter your YouTube API key in the sidebar
2. Paste a YouTube video URL (e.g., movie trailer)
3. Select number of comments (10-500)
4. Click "Analyze Comments"
5. Explore the results across different tabs

## Language Support

The application automatically detects whether comments are in Polish or English and applies appropriate:

- Stopword removal
- Lemmatization (via spaCy models)
- Tokenization

The multilingual BERT model handles sentiment analysis for both languages seamlessly.

## Limitations

- YouTube API has daily quota limits (~10,000 units/day)
- Comments must be enabled on the video
- First run downloads ~500MB of transformer models
- Topic modeling requires at least 10 comments

## License

This project was created for educational purposes as part of the Data Science Master's program.
