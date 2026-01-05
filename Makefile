# Makefile for YouTube Sentiment Analysis Docker Operations
# Usage: make [target]

.PHONY: help build dev prod stop clean logs shell test

# Default target
help:
	@echo "YouTube Sentiment Analysis - Docker Commands"
	@echo "============================================="
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  build       Build Docker image"
	@echo "  dev         Run in development mode (with hot reload)"
	@echo "  prod        Run in production mode"
	@echo "  stop        Stop all containers"
	@echo "  clean       Stop containers and remove images"
	@echo "  logs        View container logs"
	@echo "  shell       Open shell in running container"
	@echo "  test        Test the Docker build"
	@echo ""

# Build the Docker image
build:
	docker compose build

# Run in development mode
dev:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build

# Run in development mode (detached)
dev-d:
	docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build -d

# Run in production mode
prod:
	docker compose -f docker-compose.yml -f docker-compose.prod.yml up --build -d

# Stop all containers
stop:
	docker compose down

# Clean up containers, images, and volumes
clean:
	docker compose down -v --rmi local
	docker system prune -f

# View logs
logs:
	docker compose logs -f

# Open shell in running container
shell:
	docker compose exec sentiment-app /bin/bash

# Test build
test:
	docker build -t youtube-sentiment-analysis:test .
	docker run --rm youtube-sentiment-analysis:test python -c "import streamlit; import transformers; print('Build test: OK')"
	@echo "Build test passed!"

# Health check
health:
	curl -f http://localhost:8501/_stcore/health || exit 1
