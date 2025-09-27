#!/usr/bin/env python3
"""
Detailed Clustering Analysis for Manifold Distance Project
Uses t-SNE, multiple clustering methods, and validation metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_distances
import umap
# import seaborn as sns  # Not needed for this analysis
from sentence_transformers import SentenceTransformer
import os

def load_data():
    """Load the existing embedding data"""
    with open('../data/embedding_analysis_all-MiniLM-L6-v2.json', 'r') as f:
        data = json.load(f)
    
    prompts = data['prompts']
    embeddings = np.array(data['embeddings'])
    
    return prompts, embeddings

def analyze_clustering_quality(embeddings, labels, method_name):
    """Analyze the quality of clustering"""
    if len(set(labels)) < 2:
        return None, None
    
    silhouette = silhouette_score(embeddings, labels)
    calinski = calinski_harabasz_score(embeddings, labels)
    
    print(f"\n{method_name} Clustering Quality:")
    print(f"  Silhouette Score: {silhouette:.3f} (higher is better, range: -1 to 1)")
    print(f"  Calinski-Harabasz Score: {calinski:.3f} (higher is better)")
    
    return silhouette, calinski

def detailed_clustering_analysis():
    """Perform comprehensive clustering analysis"""
    prompts, embeddings = load_data()
    
    print(f"Analyzing {len(prompts)} prompts with {embeddings.shape[1]}-dimensional embeddings")
    print(f"Embedding statistics: mean={np.mean(embeddings):.4f}, std={np.std(embeddings):.4f}")
    
    # 1. Dimensionality Reduction with multiple methods
    print("\n" + "="*60)
    print("DIMENSIONALITY REDUCTION ANALYSIS")
    print("="*60)
    
    # UMAP
    print("Computing UMAP...")
    umap_reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(5, len(prompts)-1))
    embeddings_umap = umap_reducer.fit_transform(embeddings)
    
    # t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(5, len(prompts)-1))
    embeddings_tsne = tsne.fit_transform(embeddings)
    
    # 2. Multiple Clustering Methods
    print("\n" + "="*60)
    print("CLUSTERING ANALYSIS")
    print("="*60)
    
    clustering_results = {}
    
    # K-means with different k values
    print("\nK-means Clustering:")
    for k in range(2, min(6, len(prompts))):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        silhouette, calinski = analyze_clustering_quality(embeddings, labels, f"K-means (k={k})")
        clustering_results[f'kmeans_k{k}'] = {'labels': labels, 'silhouette': silhouette, 'calinski': calinski}
    
    # Gaussian Mixture Model
    print("\nGaussian Mixture Model:")
    for n_components in range(2, min(6, len(prompts))):
        gmm = GaussianMixture(n_components=n_components, random_state=42)
        labels = gmm.fit_predict(embeddings)
        silhouette, calinski = analyze_clustering_quality(embeddings, labels, f"GMM (n={n_components})")
        clustering_results[f'gmm_n{n_components}'] = {'labels': labels, 'silhouette': silhouette, 'calinski': calinski}
    
    # DBSCAN
    print("\nDBSCAN Clustering:")
    dbscan = DBSCAN(eps=0.5, min_samples=2)
    labels = dbscan.fit_predict(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"  Found {n_clusters} clusters, {list(labels).count(-1)} noise points")
    if n_clusters > 1:
        silhouette, calinski = analyze_clustering_quality(embeddings, labels, "DBSCAN")
        clustering_results['dbscan'] = {'labels': labels, 'silhouette': silhouette, 'calinski': calinski}
    
    # Agglomerative Clustering
    print("\nAgglomerative Clustering:")
    for n_clusters in range(2, min(6, len(prompts))):
        agg = AgglomerativeClustering(n_clusters=n_clusters)
        labels = agg.fit_predict(embeddings)
        silhouette, calinski = analyze_clustering_quality(embeddings, labels, f"Agglomerative (n={n_clusters})")
        clustering_results[f'agg_n{n_clusters}'] = {'labels': labels, 'silhouette': silhouette, 'calinski': calinski}
    
    # 3. Find best clustering method
    print("\n" + "="*60)
    print("BEST CLUSTERING METHOD")
    print("="*60)
    
    best_method = None
    best_silhouette = -1
    
    for method, results in clustering_results.items():
        if results['silhouette'] is not None and results['silhouette'] > best_silhouette:
            best_silhouette = results['silhouette']
            best_method = method
    
    if best_method:
        print(f"Best method: {best_method} (Silhouette: {best_silhouette:.3f})")
        best_labels = clustering_results[best_method]['labels']
    else:
        print("No clear clustering found")
        best_labels = None
    
    # 4. Visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # UMAP plots
    scatter = axes[0, 0].scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], 
                               c=range(len(prompts)), cmap='tab10', s=100)
    for i, (x, y) in enumerate(embeddings_umap):
        axes[0, 0].annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 0].set_title('UMAP: Original Embeddings')
    axes[0, 0].set_xlabel('UMAP 1')
    axes[0, 0].set_ylabel('UMAP 2')
    
    # t-SNE plots
    scatter = axes[0, 1].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                               c=range(len(prompts)), cmap='tab10', s=100)
    for i, (x, y) in enumerate(embeddings_tsne):
        axes[0, 1].annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    axes[0, 1].set_title('t-SNE: Original Embeddings')
    axes[0, 1].set_xlabel('t-SNE 1')
    axes[0, 1].set_ylabel('t-SNE 2')
    
    # Cosine distance heatmap
    cosine_dist = cosine_distances(embeddings)
    im = axes[0, 2].imshow(cosine_dist, cmap='viridis')
    axes[0, 2].set_title('Cosine Distance Matrix')
    axes[0, 2].set_xlabel('Prompt Index')
    axes[0, 2].set_ylabel('Prompt Index')
    plt.colorbar(im, ax=axes[0, 2])
    
    # Best clustering on UMAP
    if best_labels is not None:
        scatter = axes[1, 0].scatter(embeddings_umap[:, 0], embeddings_umap[:, 1], 
                                   c=best_labels, cmap='tab10', s=100)
        for i, (x, y) in enumerate(embeddings_umap):
            axes[1, 0].annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 0].set_title(f'UMAP: {best_method} Clustering')
        axes[1, 0].set_xlabel('UMAP 1')
        axes[1, 0].set_ylabel('UMAP 2')
    
    # Best clustering on t-SNE
    if best_labels is not None:
        scatter = axes[1, 1].scatter(embeddings_tsne[:, 0], embeddings_tsne[:, 1], 
                                   c=best_labels, cmap='tab10', s=100)
        for i, (x, y) in enumerate(embeddings_tsne):
            axes[1, 1].annotate(f"{i+1}", (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
        axes[1, 1].set_title(f't-SNE: {best_method} Clustering')
        axes[1, 1].set_xlabel('t-SNE 1')
        axes[1, 1].set_ylabel('t-SNE 2')
    
    # Distance distribution
    distances = cosine_dist[np.triu_indices_from(cosine_dist, k=1)]
    axes[1, 2].hist(distances, bins=20, alpha=0.7, edgecolor='black')
    axes[1, 2].set_title('Distribution of Cosine Distances')
    axes[1, 2].set_xlabel('Cosine Distance')
    axes[1, 2].set_ylabel('Frequency')
    axes[1, 2].axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.3f}')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig('../data/detailed_clustering_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 5. Detailed cluster analysis
    if best_labels is not None:
        print("\n" + "="*60)
        print("DETAILED CLUSTER ANALYSIS")
        print("="*60)
        
        n_clusters = len(set(best_labels))
        print(f"Number of clusters: {n_clusters}")
        
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, label in enumerate(best_labels) if label == cluster_id]
            cluster_prompts = [prompts[i] for i in cluster_indices]
            print(f"\nCluster {cluster_id} ({len(cluster_prompts)} prompts):")
            for i, prompt in enumerate(cluster_prompts):
                print(f"  {i+1}. {prompt}")
            
            # Calculate intra-cluster distances
            if len(cluster_indices) > 1:
                cluster_embeddings = embeddings[cluster_indices]
                intra_distances = cosine_distances(cluster_embeddings)
                intra_distances = intra_distances[np.triu_indices_from(intra_distances, k=1)]
                print(f"  Average intra-cluster distance: {np.mean(intra_distances):.3f}")
    
    # 6. Save detailed results
    results = {
        'prompts': prompts,
        'embeddings': embeddings.tolist(),
        'umap_2d': embeddings_umap.tolist(),
        'tsne_2d': embeddings_tsne.tolist(),
        'cosine_distances': cosine_dist.tolist(),
        'clustering_results': {k: {**v, 'labels': v['labels'].tolist() if isinstance(v['labels'], np.ndarray) else v['labels']} 
                              for k, v in clustering_results.items()},
        'best_method': best_method,
        'best_silhouette': best_silhouette
    }
    
    with open('../data/detailed_clustering_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to ../data/detailed_clustering_analysis.json")
    print(f"Visualization saved to ../data/detailed_clustering_analysis.png")

if __name__ == "__main__":
    detailed_clustering_analysis()
