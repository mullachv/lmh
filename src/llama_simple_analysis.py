#!/usr/bin/env python3
"""
Simple LLaMA Analysis - Fixed version without GMM issues
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import umap
import os

def extract_llama_embeddings(prompts, max_prompts=50):
    """Extract embeddings using LLaMA model"""
    print(f"Loading LLaMA model...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
    
    # Fix tokenizer padding issue
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded. Hidden size: {model.config.hidden_size}")
    
    # Limit prompts if specified
    if max_prompts:
        prompts = prompts[:max_prompts]
    
    embeddings = []
    
    print(f"Extracting embeddings for {len(prompts)} prompts...")
    with torch.no_grad():
        for text in tqdm(prompts, desc="Extracting LLaMA embeddings"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            last_hidden_states = outputs.last_hidden_state
            
            # Mean pooling
            masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
            mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            embeddings.append(mean_embedding.cpu().numpy())
    
    return np.vstack(embeddings), prompts

def analyze_llama_embeddings(embeddings, prompts):
    """Analyze LLaMA embeddings"""
    print(f"\n{'='*60}")
    print("LLAMA EMBEDDING ANALYSIS")
    print(f"{'='*60}")
    
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Embedding statistics:")
    print(f"  Mean: {np.mean(embeddings):.4f}")
    print(f"  Std: {np.std(embeddings):.4f}")
    print(f"  Min: {np.min(embeddings):.4f}")
    print(f"  Max: {np.max(embeddings):.4f}")
    
    # Cosine distances
    cosine_dist = cosine_distances(embeddings)
    avg_distance = np.mean(cosine_dist[np.triu_indices_from(cosine_dist, k=1)])
    print(f"  Average cosine distance: {avg_distance:.3f}")
    
    # K-means clustering
    print(f"\nK-means Clustering Analysis:")
    best_silhouette = -1
    best_k = 2
    
    for k in range(2, min(6, len(prompts)//2)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        if len(set(labels)) > 1:
            silhouette = silhouette_score(embeddings, labels)
            print(f"  K-means (k={k}): Silhouette = {silhouette:.3f}")
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
    
    print(f"  Best K-means: k={best_k}, Silhouette={best_silhouette:.3f}")
    
    # DBSCAN clustering
    print(f"\nDBSCAN Clustering Analysis:")
    dbscan = DBSCAN(eps=0.5, min_samples=3)
    dbscan_labels = dbscan.fit_predict(embeddings)
    n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
    n_noise = list(dbscan_labels).count(-1)
    
    print(f"  DBSCAN: {n_clusters_dbscan} clusters, {n_noise} noise points")
    
    if n_clusters_dbscan > 1:
        non_noise_mask = dbscan_labels != -1
        if np.sum(non_noise_mask) > 1:
            dbscan_silhouette = silhouette_score(embeddings[non_noise_mask], dbscan_labels[non_noise_mask])
            print(f"  DBSCAN Silhouette: {dbscan_silhouette:.3f}")
    
    return {
        'cosine_distances': cosine_dist,
        'avg_cosine_distance': avg_distance,
        'best_kmeans': {'k': best_k, 'silhouette': best_silhouette},
        'dbscan': {'n_clusters': n_clusters_dbscan, 'n_noise': n_noise, 'labels': dbscan_labels}
    }

def create_simple_visualization(embeddings, prompts, analysis_results):
    """Create simple visualization without GMM"""
    print(f"\n{'='*60}")
    print("CREATING LLaMA VISUALIZATIONS")
    print(f"{'='*60}")
    
    # UMAP projection
    n_neighbors = max(2, min(15, len(prompts)-1))
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors)
    embedding_2d = reducer.fit_transform(embeddings)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Cosine distance heatmap
    im = axes[0, 0].imshow(analysis_results['cosine_distances'], cmap='viridis')
    axes[0, 0].set_title('LLaMA: Cosine Distance Matrix')
    axes[0, 0].set_xlabel('Prompt Index')
    axes[0, 0].set_ylabel('Prompt Index')
    plt.colorbar(im, ax=axes[0, 0])
    
    # 2. UMAP scatter plot
    scatter = axes[0, 1].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                               c=range(len(prompts)), cmap='tab10', s=20, alpha=0.7)
    axes[0, 1].set_title('LLaMA: UMAP 2D Projection')
    axes[0, 1].set_xlabel('UMAP 1')
    axes[0, 1].set_ylabel('UMAP 2')
    
    # 3. Best K-means clustering
    kmeans = KMeans(n_clusters=analysis_results['best_kmeans']['k'], random_state=42)
    kmeans_labels = kmeans.fit_predict(embeddings)
    scatter = axes[1, 0].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                               c=kmeans_labels, cmap='tab10', s=20, alpha=0.7)
    axes[1, 0].set_title(f'LLaMA: K-means (k={analysis_results["best_kmeans"]["k"]})')
    axes[1, 0].set_xlabel('UMAP 1')
    axes[1, 0].set_ylabel('UMAP 2')
    
    # 4. Distance distribution
    distances = analysis_results['cosine_distances'][np.triu_indices_from(analysis_results['cosine_distances'], k=1)]
    axes[1, 1].hist(distances, bins=30, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('LLaMA: Cosine Distance Distribution')
    axes[1, 1].set_xlabel('Cosine Distance')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].axvline(np.mean(distances), color='red', linestyle='--', 
                      label=f'Mean: {np.mean(distances):.3f}')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('../data/llama_simple_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return embedding_2d, kmeans_labels

def compare_with_other_models(llama_results):
    """Compare LLaMA with other models"""
    print(f"\n{'='*60}")
    print("MODEL COMPARISON")
    print(f"{'='*60}")
    
    # Load previous results
    try:
        with open('../data/multi_model_analysis.json', 'r') as f:
            other_results = json.load(f)
        
        print(f"{'Model':<25} {'Silhouette':<12} {'Avg Distance':<12} {'Dimension':<10}")
        print("-" * 60)
        
        # LLaMA results
        print(f"{'LLaMA-2-7b-hf':<25} {llama_results['best_kmeans']['silhouette']:<12.3f} {llama_results['avg_cosine_distance']:<12.3f} {4096:<10}")
        
        # Other models
        for model_name, result in other_results.items():
            print(f"{model_name:<25} {result['best_silhouette']:<12.3f} {result['avg_cosine_distance']:<12.3f} {result['embedding_dim']:<10}")
        
        # Find best model
        all_models = [('LLaMA-2-7b-hf', llama_results['best_kmeans']['silhouette'])]
        for model_name, result in other_results.items():
            all_models.append((model_name, result['best_silhouette']))
        
        best_model = max(all_models, key=lambda x: x[1])
        print(f"\n🏆 Best Model: {best_model[0]} (Silhouette: {best_model[1]:.3f})")
        
    except FileNotFoundError:
        print("Previous model results not found. LLaMA results:")
        print(f"  Silhouette Score: {llama_results['best_kmeans']['silhouette']:.3f}")
        print(f"  Average Distance: {llama_results['avg_cosine_distance']:.3f}")

def main():
    # Load prompts
    with open('../data/diverse_prompts.json', 'r') as f:
        prompts = json.load(f)
    
    print(f"Loaded {len(prompts)} diverse prompts")
    
    # Extract LLaMA embeddings
    embeddings, prompts = extract_llama_embeddings(prompts, max_prompts=50)
    
    # Analyze embeddings
    analysis_results = analyze_llama_embeddings(embeddings, prompts)
    
    # Create visualizations
    embedding_2d, kmeans_labels = create_simple_visualization(embeddings, prompts, analysis_results)
    
    # Compare with other models
    compare_with_other_models(analysis_results)
    
    # Save results (convert numpy arrays to lists for JSON serialization)
    results = {
        'model_name': 'meta-llama/Llama-2-7b-hf',
        'prompts': prompts,
        'embeddings': embeddings.tolist(),
        'embedding_2d': embedding_2d.tolist(),
        'cosine_distances': analysis_results['cosine_distances'].tolist(),
        'kmeans_labels': kmeans_labels.tolist(),
        'dbscan_labels': analysis_results['dbscan']['labels'].tolist(),
        'analysis_results': {
            'cosine_distances': analysis_results['cosine_distances'].tolist(),
            'avg_cosine_distance': float(analysis_results['avg_cosine_distance']),
            'best_kmeans': analysis_results['best_kmeans'],
            'dbscan': {
                'n_clusters': analysis_results['dbscan']['n_clusters'],
                'n_noise': analysis_results['dbscan']['n_noise'],
                'labels': analysis_results['dbscan']['labels'].tolist()
            }
        },
        'embedding_shape': list(embeddings.shape)
    }
    
    os.makedirs('../data', exist_ok=True)
    with open('../data/llama_simple_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ LLaMA analysis complete!")
    print(f"Results saved to ../data/llama_simple_analysis.json")
    print(f"Visualization saved to ../data/llama_simple_analysis.png")

if __name__ == "__main__":
    main()
