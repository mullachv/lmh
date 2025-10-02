"""
Unsupervised Manifold Detection for Hallucination Detection

This module implements density-based clustering to discover natural manifolds
in embedding space without requiring human-curated "reality" labels.

Key Methods:
1. HDBSCAN clustering to find natural dense regions
2. Distance to nearest high-density region as hallucination score
3. No human labels required - fully unsupervised
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
import hdbscan
from typing import List, Dict, Tuple, Optional
import seaborn as sns


class UnsupervisedManifoldDetector:
    """
    Unsupervised hallucination detection using density-based clustering.
    
    This approach discovers natural manifolds in embedding space without
    requiring human-curated "reality" labels, avoiding selection bias.
    """
    
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        """
        Initialize the unsupervised manifold detector.
        
        Args:
            min_cluster_size: Minimum size for HDBSCAN clusters
            min_samples: Minimum samples for HDBSCAN core points
        """
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.clusterer = None
        self.embeddings = None
        self.prompts = None
        self.cluster_labels = None
        self.manifold_centers = None
        
    def fit(self, embeddings: np.ndarray, prompts: List[str]) -> Dict:
        """
        Fit the manifold detector to discover natural clusters.
        
        Args:
            embeddings: Array of embeddings (n_samples, n_features)
            prompts: List of corresponding prompts
            
        Returns:
            Dictionary with clustering results and statistics
        """
        self.embeddings = embeddings
        self.prompts = prompts
        
        # Apply HDBSCAN clustering
        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric='euclidean'
        )
        
        self.cluster_labels = self.clusterer.fit_predict(embeddings)
        
        # Calculate manifold centers for each cluster
        self._calculate_manifold_centers()
        
        # Compute clustering statistics
        stats = self._compute_clustering_stats()
        
        return stats
    
    def _calculate_manifold_centers(self):
        """Calculate the center of each discovered manifold."""
        unique_labels = np.unique(self.cluster_labels)
        self.manifold_centers = {}
        
        for label in unique_labels:
            if label == -1:  # Skip noise points
                continue
                
            cluster_mask = self.cluster_labels == label
            cluster_embeddings = self.embeddings[cluster_mask]
            
            # Use mean as manifold center
            self.manifold_centers[label] = np.mean(cluster_embeddings, axis=0)
    
    def _compute_clustering_stats(self) -> Dict:
        """Compute comprehensive clustering statistics."""
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.cluster_labels == -1)
        
        # Calculate silhouette score (excluding noise points)
        non_noise_mask = self.cluster_labels != -1
        if n_clusters > 1 and np.sum(non_noise_mask) > 1:
            silhouette = silhouette_score(
                self.embeddings[non_noise_mask],
                self.cluster_labels[non_noise_mask]
            )
        else:
            silhouette = 0.0
        
        # Cluster size distribution
        cluster_sizes = {}
        for label in unique_labels:
            if label != -1:
                cluster_sizes[f"cluster_{label}"] = np.sum(self.cluster_labels == label)
        
        return {
            "n_clusters": n_clusters,
            "n_noise_points": int(n_noise),
            "noise_percentage": float(n_noise / len(self.cluster_labels) * 100),
            "silhouette_score": float(silhouette),
            "cluster_sizes": cluster_sizes,
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples
        }
    
    def compute_manifold_distances(self, new_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute distance from new embeddings to nearest manifold.
        
        Args:
            new_embeddings: Array of new embeddings to score
            
        Returns:
            Array of distances to nearest manifold
        """
        if self.manifold_centers is None:
            raise ValueError("Must fit the detector before computing distances")
        
        distances = []
        
        for embedding in new_embeddings:
            min_distance = float('inf')
            
            # Find distance to nearest manifold center
            for label, center in self.manifold_centers.items():
                # Use cosine distance
                cosine_sim = np.dot(embedding, center) / (
                    np.linalg.norm(embedding) * np.linalg.norm(center)
                )
                cosine_distance = 1 - cosine_sim
                min_distance = min(min_distance, cosine_distance)
            
            distances.append(min_distance)
        
        return np.array(distances)
    
    def score_prompts(self, prompts: List[str], embeddings: np.ndarray) -> Dict:
        """
        Score a list of prompts for hallucination likelihood.
        
        Args:
            prompts: List of prompts to score
            embeddings: Corresponding embeddings
            
        Returns:
            Dictionary with scores and rankings
        """
        distances = self.compute_manifold_distances(embeddings)
        
        # Create results with rankings
        results = []
        for i, (prompt, distance) in enumerate(zip(prompts, distances)):
            results.append({
                "prompt": prompt,
                "manifold_distance": float(distance),
                "rank": i
            })
        
        # Sort by distance (higher = more likely hallucination)
        results.sort(key=lambda x: x["manifold_distance"], reverse=True)
        
        # Update rankings
        for i, result in enumerate(results):
            result["rank"] = i + 1
        
        return {
            "scores": results,
            "mean_distance": float(np.mean(distances)),
            "std_distance": float(np.std(distances)),
            "min_distance": float(np.min(distances)),
            "max_distance": float(np.max(distances))
        }
    
    def visualize_manifolds(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of discovered manifolds.
        
        Args:
            save_path: Optional path to save the plot
        """
        if self.embeddings is None or self.cluster_labels is None:
            raise ValueError("Must fit the detector before visualization")
        
        # Use UMAP for 2D visualization
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(self.embeddings)
        except ImportError:
            # Fallback to PCA if UMAP not available
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
            embedding_2d = reducer.fit_transform(self.embeddings)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot clusters
        unique_labels = np.unique(self.cluster_labels)
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            if label == -1:
                # Plot noise points
                mask = self.cluster_labels == label
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                           c='black', marker='x', s=50, alpha=0.6, label='Noise')
            else:
                # Plot cluster
                mask = self.cluster_labels == label
                plt.scatter(embedding_2d[mask, 0], embedding_2d[mask, 1], 
                           c=[colors[i]], label=f'Cluster {label}', s=50, alpha=0.7)
        
        plt.title('Unsupervised Manifold Discovery\n(HDBSCAN Clustering)', fontsize=14)
        plt.xlabel('UMAP Dimension 1', fontsize=12)
        plt.ylabel('UMAP Dimension 2', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()


def load_embeddings_from_analysis(file_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load embeddings and prompts from existing analysis files.
    
    Args:
        file_path: Path to JSON analysis file
        
    Returns:
        Tuple of (embeddings, prompts)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    embeddings = np.array(data['embeddings'])
    prompts = data['prompts']
    
    return embeddings, prompts


def main():
    """Example usage of the unsupervised manifold detector."""
    print("Unsupervised Manifold Detection for Hallucination Detection")
    print("=" * 60)
    
    # Load existing LLaMA analysis
    try:
        embeddings, prompts = load_embeddings_from_analysis('../data/llama_simple_analysis.json')
        print(f"Loaded {len(embeddings)} embeddings from LLaMA analysis")
    except FileNotFoundError:
        print("LLaMA analysis not found. Please run 'make llama-analysis' first.")
        return
    
    # Initialize detector
    detector = UnsupervisedManifoldDetector(min_cluster_size=3, min_samples=2)
    
    # Fit to discover manifolds
    print("\nDiscovering natural manifolds...")
    stats = detector.fit(embeddings, prompts)
    
    # Print results
    print(f"\nClustering Results:")
    print(f"  Clusters found: {stats['n_clusters']}")
    print(f"  Noise points: {stats['n_noise_points']} ({stats['noise_percentage']:.1f}%)")
    print(f"  Silhouette score: {stats['silhouette_score']:.3f}")
    
    if stats['cluster_sizes']:
        print(f"  Cluster sizes: {stats['cluster_sizes']}")
    
    # Score all prompts
    print("\nScoring prompts for hallucination likelihood...")
    results = detector.score_prompts(prompts, embeddings)
    
    print(f"\nDistance Statistics:")
    print(f"  Mean distance: {results['mean_distance']:.3f}")
    print(f"  Std distance: {results['std_distance']:.3f}")
    print(f"  Range: {results['min_distance']:.3f} - {results['max_distance']:.3f}")
    
    # Show top 5 most likely hallucinations
    print(f"\nTop 5 Most Likely Hallucinations:")
    for i, result in enumerate(results['scores'][:5]):
        print(f"  {i+1}. Distance {result['manifold_distance']:.3f}: {result['prompt'][:60]}...")
    
    # Show top 5 least likely hallucinations
    print(f"\nTop 5 Least Likely Hallucinations:")
    for i, result in enumerate(results['scores'][-5:]):
        print(f"  {i+1}. Distance {result['manifold_distance']:.3f}: {result['prompt'][:60]}...")
    
    # Create visualization
    print(f"\nCreating manifold visualization...")
    detector.visualize_manifolds('../data/unsupervised_manifolds.png')
    
    # Save results
    output_data = {
        "clustering_stats": stats,
        "scoring_results": results,
        "method": "unsupervised_manifold_detection",
        "parameters": {
            "min_cluster_size": detector.min_cluster_size,
            "min_samples": detector.min_samples
        }
    }
    
    with open('../data/unsupervised_manifold_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to data/unsupervised_manifold_analysis.json")


if __name__ == "__main__":
    main()
