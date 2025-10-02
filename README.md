# Hallucination via Manifold Distance

## Project Status: COMPLETED ✅

**Original Goal**: Quantify hallucination likelihood by measuring distance to learned manifolds in embedding space.

**Achievement**: Successfully implemented and tested multiple approaches, discovered fundamental flaws in reality-based methods, and identified better unsupervised approaches.

## What We Accomplished

### ✅ **Multi-Model Embedding Analysis**
- Tested 6 different embedding models (RoBERTa, LLaMA, BERT, etc.)
- **Best Performer**: RoBERTa-base (0.120 silhouette score)
- **Key Insight**: Non-normalized embeddings perform better for clustering

### ✅ **Comprehensive Clustering Analysis**
- UMAP visualization of embedding spaces
- DBSCAN clustering with silhouette scoring
- Distance-based analysis across multiple models

### ✅ **Reality Manifold Approach (ABANDONED)**
- **Problem Discovered**: Requires manual curation of "reality" prompts
- **Fundamental Flaws**: Selection bias, circular reasoning, cultural bias
- **Status**: Archived due to unscientific approach

### ✅ **Technical Infrastructure**
- Clean, automated pipeline with Makefile
- 130 diverse prompts across multiple categories
- Comprehensive analysis and visualization tools

## Key Scientific Discoveries

1. **Normalization Impact**: Non-normalized embeddings (RoBERTa, LLaMA) perform better for clustering than normalized ones (Sentence Transformers)

2. **Model Performance Ranking**:
   - **RoBERTa-base**: Best clustering (0.120 silhouette)
   - **LLaMA-2-7b**: Best separation (4096D, 0.342 avg distance)
   - **BERT/DistilBERT**: Moderate performance (0.083 silhouette)

3. **Reality Manifold Flaw**: Distance-based hallucination detection using human-curated "reality" prompts is fundamentally flawed due to selection bias and circular reasoning.

## Current Repository Structure

```
lmh/
├── src/                    # Core analysis scripts
│   ├── embedding_analysis.py
│   ├── multi_model_analysis.py
│   ├── llama_simple_analysis.py
│   ├── dbscan_analysis.py
│   ├── detailed_clustering_analysis.py
│   └── hallucination_scoring.py
├── data/                   # Analysis results
│   ├── diverse_prompts.json
│   ├── multi_model_analysis.json
│   ├── llama_simple_analysis.json
│   └── detailed_clustering_analysis.json
├── archive/               # Archived flawed approaches
├── notebooks/             # Jupyter notebooks
├── requirements.txt       # Dependencies
└── Makefile              # Build automation
```

## Next Steps (Future Research)

The project identified that **unsupervised approaches** are needed:

1. **Density-Based Clustering**: Use DBSCAN/HDBSCAN to find natural manifolds
2. **Consistency-Based Scoring**: Measure response variance within same prompt
3. **Cross-Model Agreement**: Use model disagreement as hallucination signal
4. **Temporal Consistency**: Track embedding changes over time

## Usage

```bash
# Setup
make setup
make install

# Clean up
make clean

# Show help
make help
```

## Research Timeline

- **Sept 27-30, 2025**: Multi-model analysis, clustering, LLaMA integration
- **Current**: Repository cleanup and documentation
- **Future**: Implement unsupervised hallucination detection approaches

---

**Status**: Research phase completed. Ready for next iteration with unsupervised approaches.