# IMDb Sentiment Analyzer - Makefile

.PHONY: help install train test run docker-build docker-run docker-compose-up clean lint format check-deps monitor deploy

# Default target
help:
	@echo "IMDb Sentiment Analyzer - Available Commands:"
	@echo ""
	@echo "Setup & Installation:"
	@echo "  install          Install Python dependencies"
	@echo "  check-deps       Check if all dependencies are installed"
	@echo ""
	@echo "Model Training:"
	@echo "  train            Train the sentiment analysis model"
	@echo "  train-compare    Train and compare multiple models"
	@echo "  train-tune       Train with hyperparameter tuning"
	@echo ""
	@echo "Testing & Quality:"
	@echo "  test             Run test suite"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black"
	@echo ""
	@echo "Development:"
	@echo "  run              Run the Streamlit app locally"
	@echo "  run-dev          Run in development mode with auto-reload"
	@echo ""
	@echo "Docker Deployment:"
	@echo "  docker-build     Build Docker image"
	@echo "  docker-run       Run Docker container"
	@echo "  docker-compose-up Start with Docker Compose"
	@echo "  docker-compose-down Stop Docker Compose services"
	@echo ""
	@echo "Production Deployment:"
	@echo "  deploy-local     Deploy locally"
	@echo "  deploy-docker    Deploy with Docker"
	@echo "  deploy-compose   Deploy with Docker Compose"
	@echo ""
	@echo "Monitoring & Maintenance:"
	@echo "  monitor          Start monitoring"
	@echo "  health-check     Check application health"
	@echo "  logs             View application logs"
	@echo ""
	@echo "Cleanup:"
	@echo "  clean            Clean temporary files"
	@echo "  clean-all        Clean everything including models"

# Setup & Installation
install:
	@echo "Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

check-deps:
	@echo "Checking dependencies..."
	python -c "import streamlit, sklearn, pandas, numpy, joblib, plotly; print('All dependencies available!')"

# Model Training
train:
	@echo "Training sentiment analysis model..."
	python train_model.py

train-compare:
	@echo "Training and comparing multiple models..."
	python train_model.py --compare-models

train-tune:
	@echo "Training with hyperparameter tuning..."
	python train_model.py --compare-models --tune-hyperparams

# Testing & Quality
test:
	@echo "Running test suite..."
	python -m pytest test_app.py -v

lint:
	@echo "Running code linting..."
	python -m flake8 app.py train_model.py utils.py --max-line-length=100 --ignore=E203,W503

format:
	@echo "Formatting code..."
	python -m black app.py train_model.py utils.py config.py deploy.py monitor.py --line-length=100

# Development
run:
	@echo "Starting Streamlit app..."
	streamlit run app.py

run-dev:
	@echo "Starting Streamlit app in development mode..."
	streamlit run app.py --server.runOnSave=true

# Docker Deployment
docker-build:
	@echo "Building Docker image..."
	docker build -t sentiment-analyzer .

docker-run: docker-build
	@echo "Running Docker container..."
	docker run -p 8501:8501 -v $$(pwd)/model:/app/model sentiment-analyzer

docker-compose-up:
	@echo "Starting services with Docker Compose..."
	docker-compose up -d

docker-compose-down:
	@echo "Stopping Docker Compose services..."
	docker-compose down

# Production Deployment
deploy-local:
	@echo "Deploying locally..."
	python deploy.py --mode local

deploy-docker:
	@echo "Deploying with Docker..."
	python deploy.py --mode docker

deploy-compose:
	@echo "Deploying with Docker Compose..."
	python deploy.py --mode docker-compose

# Monitoring & Maintenance
monitor:
	@echo "Starting application monitoring..."
	python monitor.py

health-check:
	@echo "Performing health check..."
	python deploy.py --health-check

logs:
	@echo "Viewing application logs..."
	@if [ -f "training.log" ]; then echo "=== Training Logs ==="; tail -n 50 training.log; fi
	@if [ -f "monitoring.log" ]; then echo "=== Monitoring Logs ==="; tail -n 50 monitoring.log; fi

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.log" -delete
	find . -type f -name "*.tmp" -delete
	rm -f deployment_info.json
	rm -f metrics.json
	rm -f monitoring_report.txt

clean-all: clean
	@echo "Cleaning all generated files including models..."
	rm -rf model/
	rm -rf plots/
	rm -rf .pytest_cache/

# Development helpers
setup-dev: install
	@echo "Setting up development environment..."
	pip install black flake8 pytest
	@echo "Development environment ready!"

quick-start: install train run

# Production setup
setup-prod: install train test
	@echo "Production setup complete!"
	@echo "Run 'make deploy-docker' to deploy with Docker"

# Backup
backup:
	@echo "Creating backup..."
	tar -czf backup_$$(date +%Y%m%d_%H%M%S).tar.gz \
		app.py train_model.py utils.py config.py \
		requirements.txt Dockerfile docker-compose.yml \
		model/ .streamlit/ README.md

# Show project status
status:
	@echo "=== Project Status ==="
	@echo "Model files:"
	@ls -la model/ 2>/dev/null || echo "  No model files found"
	@echo ""
	@echo "Recent logs:"
	@tail -n 5 training.log 2>/dev/null || echo "  No training logs"
	@echo ""
	@echo "Docker images:"
	@docker images | grep sentiment-analyzer 2>/dev/null || echo "  No Docker images found"