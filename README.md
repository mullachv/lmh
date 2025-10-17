# Unsupervised Hallucination Detection

A research project implementing unsupervised methods to detect hallucinations in large language models using embedding space analysis and manifold detection.

## Overview

This project explores multiple approaches to detect hallucinations in LLM outputs without requiring human-curated "reality" labels. The core hypothesis is that hallucinated content will exhibit different geometric properties in embedding space compared to factual content.

## Key Features

- **Unsupervised Detection**: No human labels required, eliminating selection bias
- **Multiple Detection Methods**: Density-based clustering, consistency scoring, cross-model agreement
- **Multi-Model Support**: Works with RoBERTa, LLaMA-2, BERT, and other transformer models
- **Comprehensive Analysis**: UMAP visualization, clustering analysis, distance metrics

## Detection Methods

### 1. Density-Based Manifold Detection
Uses HDBSCAN clustering to discover natural manifolds in embedding space. Content that falls far from these manifolds is considered more likely to be hallucinated.

### 2. Consistency-Based Scoring
Measures variance in multiple responses to the same prompt. Inconsistent responses indicate potential hallucination.

### 3. Cross-Model Agreement
Compares embeddings across different models. High disagreement suggests hallucination.

### 4. Combined Scoring
Weighted combination of all methods for robust detection.

## Installation

```bash
# Clone the repository
git clone https://github.com/mullachv/lmh.git
cd lmh

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```bash
# Run the complete unsupervised pipeline
make unsupervised-pipeline

# Run individual analyses
make llama-analysis          # LLaMA embedding analysis
make multi-model            # Multi-model comparison
make clustering             # DBSCAN clustering analysis

# Run tests
make test

# Show all available commands
make help
```

## Usage Examples

### Basic Pipeline
```python
from src.unsupervised_hallucination_pipeline import UnsupervisedHallucinationPipeline
import numpy as np

# Load your embeddings and prompts
embeddings = np.load('your_embeddings.npy')
prompts = ['What is the capital of France?', 'How do you make an invisibility potion?']

# Initialize pipeline
pipeline = UnsupervisedHallucinationPipeline()

# Run full analysis
results = pipeline.run_full_pipeline(embeddings, prompts)

# Get hallucination scores
for score in results['combined_scores']['individual_scores'][:5]:
    print(f"Score {score['combined_score']:.3f}: {score['prompt']}")
```

### Individual Methods
```python
from src.unsupervised_manifold_detection import UnsupervisedManifoldDetector

# Density-based detection
detector = UnsupervisedManifoldDetector(min_cluster_size=3)
stats = detector.fit(embeddings, prompts)
results = detector.score_prompts(prompts, embeddings)
```

## Model Performance

| Model | Dimensions | Silhouette Score | Best For |
|-------|------------|------------------|----------|
| **RoBERTa-base** | 768D | **0.120** | **Clustering** |
| **LLaMA-2-7b** | 4096D | **0.105** | **Separation** |
| BERT-base | 768D | 0.083 | General |
| DistilBERT | 768D | 0.083 | General |

## Project Structure

```
lmh/
├── src/                           # Core detection algorithms
│   ├── unsupervised_hallucination_pipeline.py  # Main pipeline
│   ├── unsupervised_manifold_detection.py      # Density-based clustering
│   ├── consistency_scoring.py                  # Consistency analysis
│   ├── cross_model_agreement.py                # Cross-model comparison
│   └── hallucination_scoring.py                # Distance-based scoring
├── data/                          # Analysis results and datasets
│   ├── diverse_prompts.json       # 130+ diverse prompts
│   ├── llama_simple_analysis.json # LLaMA analysis results
│   └── unsupervised_hallucination_analysis.json # Pipeline results
├── tests/                         # Test suite
├── notebooks/                     # Jupyter notebooks
├── requirements.txt               # Dependencies
├── Makefile                       # Build automation
└── daily_log.md                   # Research progress log
```

## Key Results

The unsupervised pipeline successfully identifies hallucination candidates:

**Top Hallucination Candidates:**
1. "Explain the Borsuk-Ulam theorem" (0.261) - Complex academic content
2. "Translate 'cat' into Mandarin" (0.258) - Language translation task  
3. "Who wrote Romeo and Juliet?" (0.232) - Factual question
4. "How many angels can dance on the head of a pin?" (0.213) - Philosophical/nonsensical
5. "What did Napoleon think of Bitcoin?" (0.201) - Anachronistic question

**Method Performance:**
- **Density-Based**: Best discrimination (std dev: 0.115)
- **Combined Approach**: Balanced scoring (std dev: 0.066)
- **Consistency-Based**: Limited discrimination (std dev: 0.002)

## Research Background

This project addresses the fundamental challenge of detecting hallucinations in LLM outputs. Traditional approaches require human-curated "reality" datasets, which introduce selection bias and circular reasoning. Our unsupervised methods eliminate these issues by using only the geometric properties of embeddings in high-dimensional space.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `make test`
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this work in your research, please cite:

```bibtex
@software{lmh_hallucination_detection,
  title={Unsupervised Hallucination Detection via Manifold Analysis},
  author={vsm},
  year={2025},
  url={https://github.com/mullachv/lmh}
}
```