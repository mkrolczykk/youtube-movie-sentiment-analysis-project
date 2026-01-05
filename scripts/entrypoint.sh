#!/bin/bash
# Entrypoint script for YouTube Sentiment Analysis container
# Handles startup configuration and graceful shutdown

set -e

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Trap SIGTERM and SIGINT for graceful shutdown
shutdown() {
    log_info "Received shutdown signal, stopping gracefully..."
    kill -TERM "$child" 2>/dev/null
    wait "$child"
    exit 0
}

trap shutdown SIGTERM SIGINT

# Print startup banner
echo "=============================================="
echo "  YouTube Sentiment Analysis Dashboard"
echo "  Environment: ${ENV:-development}"
echo "=============================================="

# Check environment variables
log_info "Checking environment configuration..."

if [ -n "$YOUTUBE_API_KEY" ]; then
    log_info "YouTube API key is configured"
else
    log_warn "YOUTUBE_API_KEY not set - user will need to provide it in the UI"
fi

# Verify Python environment
log_info "Verifying Python environment..."
python --version

# Check if required packages are installed
log_info "Checking required packages..."
python -c "import streamlit; import transformers; import spacy" 2>/dev/null || {
    log_error "Required Python packages are missing!"
    exit 1
}

# Verify spaCy models
log_info "Verifying spaCy models..."
python -c "import spacy; spacy.load('en_core_web_md'); spacy.load('pl_core_news_md')" 2>/dev/null || {
    log_warn "spaCy models not found, attempting to download..."
    python -m spacy download en_core_web_md
    python -m spacy download pl_core_news_md
}

# Create cache directories if they don't exist
mkdir -p /app/.cache/huggingface 2>/dev/null || true

# Start the application
log_info "Starting Streamlit application..."
log_info "Server will be available at http://0.0.0.0:${STREAMLIT_SERVER_PORT:-8501}"

# Execute the main command
exec "$@" &
child=$!
wait "$child"
