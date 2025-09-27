#!/usr/bin/env python3
"""
DBSCAN Analysis for Manifold Detection
Focus on RoBERTa embeddings with DBSCAN clustering
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors
import umap
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

def load_roberta_embeddings():
    """Load or generate RoBERTa embeddings"""
    try:
        # Try to load existing results
        with open('../data/multi_model_analysis.json', 'r') as f:
            data = json.load(f)
        
        if 'roberta-base' in data:
            embeddings = np.array(data['roberta-base']['embeddings'])
            prompts = data['roberta-base'].get('prompts', [])
            print(f"Loaded existing RoBERTa embeddings: {embeddings.shape}")
            return embeddings, prompts
    except:
        pass
    
    # Generate new embeddings if not found
    print("Generating new RoBERTa embeddings...")
    with open('../data/diverse_prompts.json', 'r') as f:
        prompts = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModel.from_pretrained('roberta-base')
    model = model.to(device)
    model.eval()
    
    embeddings = []
    with torch.no_grad():
        for text in tqdm(prompts, desc="Extracting RoBERTa embeddings"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            last_hidden_states = outputs.last_hidden_state
            
            # Mean pooling
            masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
            mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            embeddings.append(mean_embedding.cpu().numpy())
    
    embeddings = np.vstack(embeddings)
    print(f"Generated RoBERTa embeddings: {embeddings.shape}")
    return embeddings, prompts

def find_optimal_eps(embeddings, min_samples=3):
    """Find optimal eps parameter for DBSCAN using k-distance graph"""
    print("Finding optimal eps parameter...")
    
    # Calculate k-distances
    nbrs = NearestNeighbors(n_neighbors=min_samples)
    nbrs.fit(embeddings)
    distances, indices = nbrs.kneighbors(embeddings)
    
    # Sort distances to the k-th nearest neighbor
    k_distances = np.sort(distances[:, min_samples-1])
    
    # Plot k-distance graph
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(k_distances)), k_distances)
    plt.title('K-Distance Graph for DBSCAN')
    plt.xlabel('Points sorted by distance')
    plt.ylabel(f'{min_samples}-th nearest neighbor distance')
    plt.grid(True)
    
    # Find elbow point (approximate)
    # Look for the point where the curve starts to flatten
    diffs = np.diff(k_distances)
    elbow_idx = np.argmax(diffs) + 1
    suggested_eps = k_distances[elbow_idx]
    
    plt.axhline(y=suggested_eps, color='r', linestyle='--', 
                label=f'Suggested eps: {suggested_eps:.3f}')
    plt.legend()
    plt.savefig('../data/dbscan_k_distance_graph.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Suggested eps: {suggested_eps:.3f}")
    return suggested_eps

def dbscan_analysis(embeddings, prompts, eps_range=None, min_samples=3):
    """Perform comprehensive DBSCAN analysis"""
    print(f"\n{'='*60}")
    print("DBSCAN CLUSTERING ANALYSIS")
    print(f"{'='*60}")
    
    if eps_range is None:
        # Find optimal eps
        optimal_eps = find_optimal_eps(embeddings, min_samples)
        eps_range = [optimal_eps * 0.5, optimal_eps, optimal_eps * 1.5, optimal_eps * 2.0]
    
    results = {}
    
    for eps in eps_range:
        print(f"\nDBSCAN with eps={eps:.3f}, min_samples={min_samples}")
        
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(embeddings)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        print(f"  Clustered points: {len(prompts) - n_noise}")
        
        if n_clusters > 1:
            # Calculate silhouette score (excluding noise points)
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette = silhouette_score(embeddings[non_noise_mask], labels[non_noise_mask])
                print(f"  Silhouette score: {silhouette:.3f}")
            else:
                silhouette = None
        else:
            silhouette = None
        
        results[eps] = {
            'labels': labels,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'silhouette': silhouette
        }
    
    return results

def visualize_dbscan_results(embeddings, prompts, results):
    """Visualize DBSCAN results"""
    print(f"\n{'='*60}")
    print("CREATING DBSCAN VISUALIZATIONS")
    print(f"{'='*60}")
    
    # UMAP projection
    n_neighbors = max(2, min(15, len(prompts)-1))
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    embedding_2d = reducer.fit_transform(embeddings)
    
    n_eps = len(results)
    fig, axes = plt.subplots(2, (n_eps + 1) // 2, figsize=(5 * ((n_eps + 1) // 2), 10))
    if n_eps == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (eps, result) in enumerate(results.items()):
        if i >= len(axes):
            break
            
        labels = result['labels']
        n_clusters = result['n_clusters']
        n_noise = result['n_noise']
        silhouette = result['silhouette']
        
        # Create scatter plot
        unique_labels = set(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black for noise points
                col = 'black'
                marker = 'x'
                size = 50
                alpha = 0.6
            else:
                marker = 'o'
                size = 30
                alpha = 0.7
            
            class_member_mask = (labels == k)
            xy = embedding_2d[class_member_mask]
            axes[i].scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, s=size, alpha=alpha)
        
        title = f'eps={eps:.3f}\nClusters: {n_clusters}, Noise: {n_noise}'
        if silhouette is not None:
            title += f'\nSilhouette: {silhouette:.3f}'
        axes[i].set_title(title)
        axes[i].set_xlabel('UMAP 1')
        axes[i].set_ylabel('UMAP 2')
    
    # Hide unused subplots
    for i in range(len(results), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('../data/dbscan_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return embedding_2d

def analyze_clusters(embeddings, prompts, labels, eps):
    """Analyze the content of each cluster"""
    print(f"\n{'='*60}")
    print(f"CLUSTER CONTENT ANALYSIS (eps={eps:.3f})")
    print(f"{'='*60}")
    
    unique_labels = set(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(labels).count(-1)
    
    print(f"Total clusters: {n_clusters}")
    print(f"Noise points: {n_noise}")
    
    # Analyze each cluster
    for cluster_id in sorted(unique_labels):
        if cluster_id == -1:
            continue
            
        cluster_indices = [i for i, label in enumerate(labels) if label == cluster_id]
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
    
    # Analyze noise points
    if n_noise > 0:
        noise_indices = [i for i, label in enumerate(labels) if label == -1]
        noise_prompts = [prompts[i] for i in noise_indices]
        
        print(f"\nNoise Points ({len(noise_prompts)} prompts):")
        for i, prompt in enumerate(noise_prompts):
            print(f"  {i+1}. {prompt}")
        
        # Calculate distances from noise points to nearest clusters
        if n_clusters > 0:
            print(f"\nNoise point analysis:")
            for i, noise_idx in enumerate(noise_indices):
                noise_embedding = embeddings[noise_idx:noise_idx+1]
                min_distance = float('inf')
                nearest_cluster = None
                
                for cluster_id in unique_labels:
                    if cluster_id == -1:
                        continue
                    cluster_indices = [j for j, label in enumerate(labels) if label == cluster_id]
                    cluster_embeddings = embeddings[cluster_indices]
                    
                    # Distance to cluster centroid
                    cluster_centroid = np.mean(cluster_embeddings, axis=0)
                    distance = cosine_distances(noise_embedding, cluster_centroid.reshape(1, -1))[0, 0]
                    
                    if distance < min_distance:
                        min_distance = distance
                        nearest_cluster = cluster_id
                
                print(f"  Noise {i+1} ('{prompts[noise_idx][:50]}...'): nearest cluster {nearest_cluster}, distance {min_distance:.3f}")

def main():
    # Load RoBERTa embeddings
    embeddings, prompts = load_roberta_embeddings()
    
    # Perform DBSCAN analysis
    results = dbscan_analysis(embeddings, prompts)
    
    # Visualize results
    embedding_2d = visualize_dbscan_results(embeddings, prompts, results)
    
    # Find best result
    best_eps = None
    best_score = -1
    
    for eps, result in results.items():
        if result['silhouette'] is not None and result['silhouette'] > best_score:
            best_score = result['silhouette']
            best_eps = eps
    
    if best_eps is not None:
        print(f"\nBest DBSCAN result: eps={best_eps:.3f}, silhouette={best_score:.3f}")
        
        # Analyze the best clustering
        best_labels = results[best_eps]['labels']
        analyze_clusters(embeddings, prompts, best_labels, best_eps)
        
        # Save results
        output = {
            'prompts': prompts,
            'embeddings': embeddings.tolist(),
            'embedding_2d': embedding_2d.tolist(),
            'dbscan_results': {str(eps): {**result, 'labels': result['labels'].tolist()} 
                              for eps, result in results.items()},
            'best_eps': best_eps,
            'best_silhouette': best_score
        }
        
        with open('../data/dbscan_analysis.json', 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to ../data/dbscan_analysis.json")
    else:
        print("\nNo meaningful clusters found with DBSCAN")

if __name__ == "__main__":
    main()
