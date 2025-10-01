#!/usr/bin/env python3
"""
Apply Hallucination Scoring to LLaMA Analysis

This script loads the existing LLaMA embeddings and applies hallucination scoring
using different distance metrics to identify potentially hallucinated responses.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from hallucination_scoring import (
    hallucination_score, 
    batch_hallucination_scores,
    analyze_hallucination_distribution,
    save_hallucination_analysis
)


def load_llama_analysis(file_path: str):
    """Load LLaMA analysis results."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    embeddings = np.array(data['embeddings'])
    prompts = data['prompts']
    
    return embeddings, prompts


def apply_hallucination_scoring(embeddings, prompts, model_name="meta-llama/Llama-2-7b-hf"):
    """Apply hallucination scoring with multiple methods."""
    
    print(f"Applying hallucination scoring to {len(embeddings)} embeddings from {model_name}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    
    # Test different distance methods
    methods = ["cosine", "euclidean", "faiss_cosine", "faiss_l2"]
    results = {}
    
    for method in methods:
        print(f"\n--- Testing {method} method ---")
        
        # Calculate scores for all embeddings
        scores = batch_hallucination_scores(embeddings, embeddings, method=method)
        
        # Analyze distribution
        stats = analyze_hallucination_distribution(scores)
        
        print(f"Mean score: {stats['mean']:.4f}")
        print(f"Std score:  {stats['std']:.4f}")
        print(f"Min score:  {stats['min']:.4f}")
        print(f"Max score:  {stats['max']:.4f}")
        print(f"Above 0.5 threshold: {stats['above_threshold']}/{len(scores)}")
        
        # Find most/least likely to be hallucinations
        sorted_indices = np.argsort(scores)
        
        print(f"\nTop 5 most likely to be hallucinations:")
        for i in range(5):
            idx = sorted_indices[-(i+1)]
            print(f"  {i+1}. Score: {scores[idx]:.4f} - {prompts[idx][:80]}...")
        
        print(f"\nTop 5 least likely to be hallucinations:")
        for i in range(5):
            idx = sorted_indices[i]
            print(f"  {i+1}. Score: {scores[idx]:.4f} - {prompts[idx][:80]}...")
        
        results[method] = {
            'scores': scores,
            'statistics': stats
        }
        
        # Save individual analysis
        output_file = f"../data/hallucination_analysis_{method}.json"
        save_hallucination_analysis(scores, prompts, model_name, method, output_file)
        print(f"Saved analysis to {output_file}")
    
    return results


def create_hallucination_visualization(results, output_file):
    """Create visualization comparing different hallucination scoring methods."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    methods = list(results.keys())
    
    for i, method in enumerate(methods):
        scores = results[method]['scores']
        stats = results[method]['statistics']
        
        # Histogram
        axes[i].hist(scores, bins=20, alpha=0.7, edgecolor='black')
        axes[i].axvline(stats['mean'], color='red', linestyle='--', label=f"Mean: {stats['mean']:.3f}")
        axes[i].axvline(stats['median'], color='orange', linestyle='--', label=f"Median: {stats['median']:.3f}")
        axes[i].axvline(0.5, color='green', linestyle='-', label="Threshold: 0.5")
        
        axes[i].set_title(f'{method.title()} Method\n'
                         f'Mean: {stats["mean"]:.3f}, Std: {stats["std"]:.3f}\n'
                         f'Above threshold: {stats["above_threshold"]}/{len(scores)}')
        axes[i].set_xlabel('Hallucination Score')
        axes[i].set_ylabel('Frequency')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {output_file}")
    plt.close()


def compare_methods(results):
    """Compare different scoring methods."""
    
    print("\n" + "="*60)
    print("METHOD COMPARISON")
    print("="*60)
    
    comparison_data = []
    
    for method, data in results.items():
        stats = data['statistics']
        comparison_data.append({
            'method': method,
            'mean': stats['mean'],
            'std': stats['std'],
            'above_threshold': stats['above_threshold'],
            'percentile_95': stats['percentile_95']
        })
    
    # Sort by mean score
    comparison_data.sort(key=lambda x: x['mean'])
    
    print(f"{'Method':<15} {'Mean':<8} {'Std':<8} {'Std/Mean':<10} {'Above 0.5':<10} {'P95':<8}")
    print("-" * 70)
    
    for data in comparison_data:
        std_mean_ratio = data['std'] / data['mean'] if data['mean'] > 0 else float('inf')
        print(f"{data['method']:<15} {data['mean']:<8.3f} {data['std']:<8.3f} "
              f"{std_mean_ratio:<10.3f} {data['above_threshold']:<10} {data['percentile_95']:<8.3f}")
    
    # Find best method (highest discrimination)
    best_method = max(comparison_data, key=lambda x: x['std'])
    print(f"\nBest discrimination method: {best_method['method']} (highest std: {best_method['std']:.3f})")
    
    return comparison_data


def main():
    """Main execution function."""
    
    print("Hallucination Scoring Analysis")
    print("=" * 50)
    
    # Load LLaMA analysis
    try:
        embeddings, prompts = load_llama_analysis("../data/llama_simple_analysis.json")
        print(f"Loaded {len(embeddings)} embeddings from LLaMA analysis")
    except FileNotFoundError:
        print("LLaMA analysis file not found. Please run 'make llama-analysis' first.")
        return
    
    # Apply hallucination scoring
    results = apply_hallucination_scoring(embeddings, prompts)
    
    # Create visualization
    create_hallucination_visualization(results, "../data/hallucination_comparison.png")
    
    # Compare methods
    comparison = compare_methods(results)
    
    # Save comprehensive results
    comprehensive_results = {
        "model": "meta-llama/Llama-2-7b-hf",
        "total_embeddings": len(embeddings),
        "embedding_dimension": embeddings.shape[1],
        "methods_comparison": comparison,
        "detailed_results": {method: {
            "statistics": data["statistics"],
            "top_hallucination_indices": np.argsort(data["scores"])[-10:].tolist(),
            "bottom_hallucination_indices": np.argsort(data["scores"])[:10].tolist()
        } for method, data in results.items()}
    }
    
    with open("../data/comprehensive_hallucination_analysis.json", 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    print(f"\nComprehensive analysis saved to ../data/comprehensive_hallucination_analysis.json")
    print("Analysis complete!")


if __name__ == "__main__":
    main()
