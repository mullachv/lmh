# Hallucination via Manifold Distance Project
# Clean Makefile for unsupervised hallucination detection

.PHONY: help install setup clean unsupervised-manifold consistency-scoring cross-model-agreement unsupervised-pipeline

# Default target
help:
	@echo "Unsupervised Hallucination Detection - Available Commands:"
	@echo ""
	@echo "SETUP:"
	@echo "  setup       - Create virtual environment and install dependencies"
	@echo "  install     - Install/update all dependencies"
	@echo ""
	@echo "UNSUPERVISED METHODS:"
	@echo "  unsupervised-manifold - Run density-based manifold detection"
	@echo "  consistency-scoring   - Run consistency-based scoring"
	@echo "  cross-model-agreement - Run cross-model agreement detection"
	@echo "  unsupervised-pipeline - Run complete unsupervised pipeline"
	@echo ""
	@echo "UTILITIES:"
	@echo "  clean       - Clean generated files and cache"
	@echo "  help        - Show this help message"

# =============================================================================
# SETUP & INSTALLATION
# =============================================================================

# Setup virtual environment and install dependencies
setup:
	python -m venv .venv
	@echo "Virtual environment created. Run 'make install' to install dependencies"

# Install dependencies
install:
	source .venv/bin/activate && pip install --index-url https://pypi.org/simple/ -r requirements.txt

# =============================================================================
# UNSUPERVISED HALLUCINATION DETECTION
# =============================================================================

# Run density-based manifold detection
unsupervised-manifold:
	@echo "Running unsupervised manifold detection..."
	@source .venv/bin/activate && cd src && python unsupervised_manifold_detection.py

# Run consistency-based scoring
consistency-scoring:
	@echo "Running consistency-based scoring..."
	@source .venv/bin/activate && cd src && python consistency_scoring.py

# Run cross-model agreement detection
cross-model-agreement:
	@echo "Running cross-model agreement detection..."
	@source .venv/bin/activate && cd src && python cross_model_agreement.py

# Run complete unsupervised pipeline
unsupervised-pipeline:
	@echo "Running complete unsupervised hallucination detection pipeline..."
	@source .venv/bin/activate && cd src && python unsupervised_hallucination_pipeline.py

# =============================================================================
# TESTING
# =============================================================================

# Run all tests
test:
	@echo "Running all tests..."
	@source .venv/bin/activate && python -m pytest tests/ -v

# Run only unit tests
test-unit:
	@echo "Running unit tests..."
	@source .venv/bin/activate && python -m pytest tests/ -v -m "not slow and not integration"

# Run only integration tests
test-integration:
	@echo "Running integration tests..."
	@source .venv/bin/activate && python -m pytest tests/ -v -m "integration"

# Run tests with coverage
test-coverage:
	@echo "Running tests with coverage..."
	@source .venv/bin/activate && python -m pytest tests/ --cov=src --cov-report=html --cov-report=term

# Run specific test file
test-manifold:
	@echo "Running manifold detection tests..."
	@source .venv/bin/activate && python -m pytest tests/test_manifold_detection.py -v

# =============================================================================
# UTILITIES
# =============================================================================

# Clean generated files
clean:
	rm -f data/*.json data/*.png
	rm -rf __pycache__ src/__pycache__ notebooks/__pycache__
	rm -rf htmlcov/ .coverage
	find . -name "*.pyc" -delete
	@echo "Cleaned generated files and cache"