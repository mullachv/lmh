#!/usr/bin/env python3
"""
Analyze cosine distances from LLaMA embeddings
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_cosine_distances():
    """Analyze the cosine distance distribution"""
    
    # Load LLaMA results
    try:
        with open('../data/llama_simple_analysis.json', 'r') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print("The JSON file may be corrupted. Let's check the file size...")
        import os
        size = os.path.getsize('../data/llama_simple_analysis.json')
        print(f"File size: {size:,} bytes")
        return
    
    cosine_distances = np.array(data['cosine_distances'])
    prompts = data['prompts']
    
    print('=== COSINE DISTANCE ANALYSIS ===')
    print(f'Number of prompts: {len(prompts)}')
    print(f'Distance matrix shape: {cosine_distances.shape}')
    
    # Get upper triangle (unique pairs)
    upper_tri = np.triu_indices_from(cosine_distances, k=1)
    distances = cosine_distances[upper_tri]
    
    print(f'Total pairwise distances: {len(distances)}')
    print(f'\\nDistance Statistics:')
    print(f'  Mean: {np.mean(distances):.3f}')
    print(f'  Std:  {np.std(distances):.3f}')
    print(f'  Min:  {np.min(distances):.3f}')
    print(f'  Max:  {np.max(distances):.3f}')
    print(f'  Median: {np.median(distances):.3f}')
    
    # Show some example pairs
    print(f'\\n=== EXAMPLE PAIRWISE DISTANCES ===')
    for i in range(min(3, len(prompts))):
        for j in range(i+1, min(i+2, len(prompts))):
            dist = cosine_distances[i, j]
            print(f'Prompt {i} vs {j}: {dist:.3f}')
            print(f'  \"{prompts[i][:60]}...\"')
            print(f'  \"{prompts[j][:60]}...\"')
            print()
    
    # Create visualization
    plt.figure(figsize=(12, 5))
    
    # Histogram of distances
    plt.subplot(1, 2, 1)
    plt.hist(distances, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(distances), color='red', linestyle='--', 
                label=f'Mean: {np.mean(distances):.3f}')
    plt.xlabel('Cosine Distance')
    plt.ylabel('Frequency')
    plt.title('LLaMA: Cosine Distance Distribution')
    plt.legend()
    
    # Distance matrix heatmap (subset)
    plt.subplot(1, 2, 2)
    subset_size = min(20, len(prompts))
    subset_distances = cosine_distances[:subset_size, :subset_size]
    im = plt.imshow(subset_distances, cmap='viridis')
    plt.colorbar(im)
    plt.title(f'LLaMA: Cosine Distance Matrix (first {subset_size} prompts)')
    plt.xlabel('Prompt Index')
    plt.ylabel('Prompt Index')
    
    plt.tight_layout()
    plt.savefig('../data/cosine_distance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return distances

if __name__ == "__main__":
    analyze_cosine_distances()
