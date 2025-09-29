# Hallucination via Manifold Distance Project
# Makefile for common tasks

.PHONY: help install setup test run-analysis multi-model clustering setup-llama run-notebook clean llama-analysis check-models

# Default target
help:
	@echo "Available targets:"
	@echo "  install     - Install all dependencies"
	@echo "  setup       - Create virtual environment and install deps"
	@echo "  test        - Run embedding analysis test (10 prompts)"
	@echo "  run-analysis - Run full embedding analysis (10 prompts)"
	@echo "  multi-model - Run multi-model comparison (100+ prompts)"
	@echo "  clustering  - Run detailed clustering analysis"
	@echo "  llama-analysis - Run LLaMA embedding analysis (50 prompts)"
	@echo "  check-models - Check which models are downloaded"
	@echo "  model-dims - Show model embedding dimensions and performance"
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

# Run multi-model comparison analysis (only if not already done)
multi-model:
	@if [ ! -f data/multi_model_analysis.json ]; then \
		echo "Running multi-model analysis..."; \
		source .venv/bin/activate && cd src && python multi_model_analysis.py; \
	else \
		echo "Multi-model analysis already completed. Results in data/multi_model_analysis.json"; \
	fi

# Run detailed clustering analysis (only if not already done)
clustering:
	@if [ ! -f data/detailed_clustering_analysis.json ]; then \
		echo "Running detailed clustering analysis..."; \
		source .venv/bin/activate && cd src && python detailed_clustering_analysis.py; \
	else \
		echo "Detailed clustering analysis already completed. Results in data/detailed_clustering_analysis.json"; \
	fi

# Run LLaMA analysis (only if not already done)
llama-analysis:
	@if [ ! -f data/llama_simple_analysis.json ]; then \
		echo "Running LLaMA analysis..."; \
		source .venv/bin/activate && cd src && python llama_simple_analysis.py; \
	else \
		echo "LLaMA analysis already completed. Results in data/llama_simple_analysis.json"; \
	fi

# Check which models are downloaded
check-models:
	@echo "Checking downloaded models..."
	@echo "LLaMA-2-7b-hf: $$(if [ -d ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf ]; then echo "✅ Downloaded ($$(du -sh ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b-hf | cut -f1))"; else echo "❌ Not downloaded"; fi)"
	@echo "Other models: $$(ls ~/.cache/huggingface/hub/ | grep -E "(bert|roberta|distil)" | wc -l) models downloaded"

# Show model embedding dimensions and performance
model-dims:
	source .venv/bin/activate && cd src && python model_dimensions.py

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
