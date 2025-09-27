# Hallucination via Manifold Distance Project
# Makefile for common tasks

.PHONY: help install setup test run-analysis multi-model clustering setup-llama run-notebook clean

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install all dependencies"
	@echo "  setup       - Create virtual environment and install deps"
	@echo "  test        - Run embedding analysis test (10 prompts)"
	@echo "  run-analysis - Run full embedding analysis (10 prompts)"
	@echo "  multi-model - Run multi-model comparison (100+ prompts)"
	@echo "  clustering  - Run detailed clustering analysis"
	@echo "  setup-llama - Setup LLaMA access instructions"
	@echo "  run-notebook - Start Jupyter notebook server"
	@echo "  clean       - Clean generated files"

# Install dependencies
install:
	source .venv/bin/activate && pip install --index-url https://pypi.org/simple/ -r requirements.txt

# Setup virtual environment and install dependencies
setup:
	python -m venv .venv
	@echo "Virtual environment created. Run 'make install' to install dependencies"

# Run the embedding analysis test script
test:
	source .venv/bin/activate && cd src && python embedding_analysis.py

# Run the full embedding analysis with visualizations
run-analysis:
	source .venv/bin/activate && cd src && python embedding_analysis.py

# Run multi-model comparison analysis
multi-model:
	source .venv/bin/activate && cd src && python multi_model_analysis.py

# Run detailed clustering analysis
clustering:
	source .venv/bin/activate && cd src && python detailed_clustering_analysis.py

# Setup LLaMA access
setup-llama:
	source .venv/bin/activate && cd src && python setup_llama_access.py

# Start Jupyter notebook server
run-notebook:
	source .venv/bin/activate && jupyter notebook notebooks/

# Clean generated files
clean:
	rm -f data/*.json data/*.png
	rm -rf __pycache__ src/__pycache__ notebooks/__pycache__
	find . -name "*.pyc" -delete

# Quick development cycle: clean, test, and show results
dev: clean test
	@echo "Analysis complete. Check data/ directory for results."
