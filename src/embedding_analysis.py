#!/usr/bin/env python3
"""
Embedding Analysis for Manifold Distance Project
Supports multiple embedding models including LLaMA alternatives
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import umap
import os
from tqdm import tqdm

class EmbeddingAnalyzer:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        """
        Initialize with a sentence transformer model
        
        Available models:
        - "all-MiniLM-L6-v2": Fast, good quality (384 dim)
        - "all-mpnet-base-v2": Higher quality (768 dim)
        - "all-distilroberta-v1": Good balance (768 dim)
        - "paraphrase-multilingual-MiniLM-L12-v2": Multilingual (384 dim)
        """
        self.model_name = model_name
        print(f"Loading model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print(f"Model loaded. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def extract_embeddings(self, texts):
        """Extract embeddings for a list of texts"""
        print("Extracting embeddings...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def analyze_embeddings(self, embeddings, prompts):
        """Analyze the embedding space"""
        print(f"Embedding shape: {embeddings.shape}")
        
        # Basic statistics
        print(f"\nEmbedding Statistics:")
        print(f"  Mean: {np.mean(embeddings):.4f}")
        print(f"  Std: {np.std(embeddings):.4f}")
        print(f"  Min: {np.min(embeddings):.4f}")
        print(f"  Max: {np.max(embeddings):.4f}")
        
        # Cosine distances
        cosine_dist = cosine_distances(embeddings)
        print(f"\nCosine Distance Matrix:")
        print(cosine_dist)
        
        return cosine_dist
    
    def visualize_embeddings(self, embeddings, prompts, cosine_dist):
        """Create visualizations of the embedding space"""
        
        # UMAP 2D projection
        print("Creating UMAP 2D projection...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(prompts)-1))
        embedding_2d = reducer.fit_transform(embeddings)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Cosine distance heatmap
        im = axes[0, 0].imshow(cosine_dist, cmap='viridis')
        axes[0, 0].set_title('Cosine Distance Matrix')
        axes[0, 0].set_xlabel('Prompt Index')
        axes[0, 0].set_ylabel('Prompt Index')
        plt.colorbar(im, ax=axes[0, 0])
        
        # 2. UMAP 2D scatter plot
        scatter = axes[0, 1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=range(len(prompts)), cmap='tab10', s=100)
        for i, (x, y) in enumerate(embedding_2d):
            axes[0, 1].annotate(f"{i+1}", (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        axes[0, 1].set_title('2D UMAP Projection')
        axes[0, 1].set_xlabel('UMAP Dimension 1')
        axes[0, 1].set_ylabel('UMAP Dimension 2')
        
        # 3. K-means clustering
        n_clusters = min(3, len(prompts))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(embeddings)
        
        scatter = axes[1, 0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=cluster_labels, cmap='tab10', s=100)
        for i, (x, y) in enumerate(embedding_2d):
            axes[1, 0].annotate(f"{i+1}", (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        axes[1, 0].set_title(f'K-means Clustering (k={n_clusters})')
        axes[1, 0].set_xlabel('UMAP Dimension 1')
        axes[1, 0].set_ylabel('UMAP Dimension 2')
        
        # 4. GMM clustering
        gmm = GaussianMixture(n_components=n_clusters, random_state=42)
        gmm_labels = gmm.fit_predict(embeddings)
        gmm_probs = gmm.predict_proba(embeddings)
        entropy = -np.sum(gmm_probs * np.log(gmm_probs + 1e-10), axis=1)
        
        scatter = axes[1, 1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                   c=entropy, cmap='viridis', s=100)
        for i, (x, y) in enumerate(embedding_2d):
            axes[1, 1].annotate(f"{i+1}", (x, y), xytext=(5, 5), 
                              textcoords='offset points', fontsize=8)
        axes[1, 1].set_title('GMM Uncertainty (Entropy)')
        axes[1, 1].set_xlabel('UMAP Dimension 1')
        axes[1, 1].set_ylabel('UMAP Dimension 2')
        plt.colorbar(scatter, ax=axes[1, 1], label='Entropy')
        
        plt.tight_layout()
        plt.savefig('../data/embedding_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return embedding_2d, cluster_labels, gmm_labels, entropy
    
    def print_analysis(self, prompts, cluster_labels, gmm_labels, entropy):
        """Print detailed analysis results"""
        print(f"\n{'='*60}")
        print("DETAILED ANALYSIS RESULTS")
        print(f"{'='*60}")
        
        # K-means clusters
        print(f"\nK-means Clustering Results:")
        for cluster_id in range(max(cluster_labels) + 1):
            cluster_prompts = [prompts[i] for i in range(len(prompts)) if cluster_labels[i] == cluster_id]
            print(f"  Cluster {cluster_id}: {cluster_prompts}")
        
        # GMM clusters
        print(f"\nGMM Clustering Results:")
        for cluster_id in range(max(gmm_labels) + 1):
            cluster_prompts = [prompts[i] for i in range(len(prompts)) if gmm_labels[i] == cluster_id]
            print(f"  Component {cluster_id}: {cluster_prompts}")
        
        # Uncertainty analysis
        print(f"\nUncertainty Analysis (Entropy - Higher = Less Certain):")
        uncertainty_data = [(prompts[i], entropy[i]) for i in range(len(prompts))]
        uncertainty_data.sort(key=lambda x: x[1], reverse=True)
        
        for prompt, uncertainty in uncertainty_data:
            print(f"  {uncertainty:.3f}: {prompt}")
        
        # Identify potentially problematic prompts
        high_uncertainty_threshold = np.percentile(entropy, 75)
        problematic_prompts = [prompts[i] for i in range(len(prompts)) if entropy[i] > high_uncertainty_threshold]
        
        print(f"\nPotentially Problematic Prompts (High Uncertainty):")
        for prompt in problematic_prompts:
            print(f"  - {prompt}")
    
    def save_results(self, prompts, embeddings, cosine_dist, embedding_2d, 
                    cluster_labels, gmm_labels, entropy):
        """Save all results to JSON file"""
        results = {
            'model_name': self.model_name,
            'prompts': prompts,
            'embeddings': embeddings.tolist(),
            'cosine_distances': cosine_dist.tolist(),
            'embedding_2d': embedding_2d.tolist(),
            'kmeans_clusters': cluster_labels.tolist(),
            'gmm_clusters': gmm_labels.tolist(),
            'entropy': entropy.tolist(),
            'embedding_shape': embeddings.shape
        }
        
        os.makedirs('../data', exist_ok=True)
        output_file = f'../data/embedding_analysis_{self.model_name.replace("/", "_")}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")

def main():
    # Load sample prompts
    with open('../data/sample_prompts.json', 'r') as f:
        prompts = json.load(f)
    
    print(f"Loaded {len(prompts)} sample prompts:")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}")
    
    # Initialize analyzer with a good embedding model
    # You can change this to experiment with different models
    analyzer = EmbeddingAnalyzer("all-MiniLM-L6-v2")
    
    # Extract embeddings
    embeddings = analyzer.extract_embeddings(prompts)
    
    # Analyze embeddings
    cosine_dist = analyzer.analyze_embeddings(embeddings, prompts)
    
    # Visualize and cluster
    embedding_2d, cluster_labels, gmm_labels, entropy = analyzer.visualize_embeddings(
        embeddings, prompts, cosine_dist)
    
    # Print detailed analysis
    analyzer.print_analysis(prompts, cluster_labels, gmm_labels, entropy)
    
    # Save results
    analyzer.save_results(prompts, embeddings, cosine_dist, embedding_2d,
                         cluster_labels, gmm_labels, entropy)

if __name__ == "__main__":
    main()
