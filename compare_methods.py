#!/usr/bin/env python3
"""
Compare Unsupervised Methods vs Previous Distance-Based Approaches

This script compares the results from:
1. New unsupervised methods (density-based + consistency)
2. Previous distance-based hallucination scoring
3. LLaMA embedding analysis
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
import pandas as pd

def load_previous_results():
    """Load results from previous distance-based methods."""
    try:
        with open('data/llama_simple_analysis.json', 'r') as f:
            llama_data = json.load(f)
        
        # Extract previous hallucination scores if available
        previous_scores = []
        if 'hallucination_scores' in llama_data:
            for score_data in llama_data['hallucination_scores']:
                previous_scores.append({
                    'prompt': score_data['prompt'],
                    'score': score_data['score'],
                    'method': 'previous_distance'
                })
        
        return previous_scores, llama_data
    except FileNotFoundError:
        print("Previous results not found")
        return [], {}

def load_unsupervised_results():
    """Load results from new unsupervised methods."""
    try:
        with open('data/unsupervised_hallucination_analysis.json', 'r') as f:
            unsupervised_data = json.load(f)
        
        # Extract combined scores
        combined_scores = []
        if 'combined_scores' in unsupervised_data and 'individual_scores' in unsupervised_data['combined_scores']:
            for score_data in unsupervised_data['combined_scores']['individual_scores']:
                combined_scores.append({
                    'prompt': score_data['prompt'],
                    'score': score_data['combined_score'],
                    'method': 'unsupervised_combined'
                })
        
        # Extract individual method scores
        manifold_scores = []
        if 'individual_analyses' in unsupervised_data and 'manifold_analysis' in unsupervised_data['individual_analyses']:
            for score_data in unsupervised_data['individual_analyses']['manifold_analysis']['scoring_results']['scores']:
                manifold_scores.append({
                    'prompt': score_data['prompt'],
                    'score': score_data['manifold_distance'],
                    'method': 'density_based'
                })
        
        consistency_scores = []
        if 'individual_analyses' in unsupervised_data and 'consistency_analysis' in unsupervised_data['individual_analyses']:
            for score_data in unsupervised_data['individual_analyses']['consistency_analysis']['individual_results']:
                consistency_scores.append({
                    'prompt': score_data['prompt'],
                    'score': score_data['consistency_score'],
                    'method': 'consistency_based'
                })
        
        return combined_scores, manifold_scores, consistency_scores, unsupervised_data
    except FileNotFoundError:
        print("Unsupervised results not found")
        return [], [], [], {}

def create_comparison_analysis():
    """Create comprehensive comparison analysis."""
    print("Loading results from all methods...")
    
    # Load all results
    previous_scores, llama_data = load_previous_results()
    combined_scores, manifold_scores, consistency_scores, unsupervised_data = load_unsupervised_results()
    
    print(f"Loaded {len(previous_scores)} previous scores")
    print(f"Loaded {len(combined_scores)} unsupervised combined scores")
    print(f"Loaded {len(manifold_scores)} density-based scores")
    print(f"Loaded {len(consistency_scores)} consistency scores")
    
    # Create comparison DataFrame
    all_scores = previous_scores + combined_scores + manifold_scores + consistency_scores
    df = pd.DataFrame(all_scores)
    
    if df.empty:
        print("No data to compare")
        return
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Method Comparison: Unsupervised vs Previous Approaches', fontsize=16)
    
    # 1. Score distributions by method
    ax1 = axes[0, 0]
    methods = df['method'].unique()
    for method in methods:
        method_scores = df[df['method'] == method]['score']
        ax1.hist(method_scores, alpha=0.7, label=method, bins=20)
    ax1.set_xlabel('Hallucination Score')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Score Distributions by Method')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Method comparison boxplot
    ax2 = axes[0, 1]
    df.boxplot(column='score', by='method', ax=ax2)
    ax2.set_title('Score Ranges by Method')
    ax2.set_xlabel('Method')
    ax2.set_ylabel('Hallucination Score')
    ax2.grid(True, alpha=0.3)
    
    # 3. Top 10 most likely hallucinations comparison
    ax3 = axes[1, 0]
    if len(combined_scores) > 0:
        # Get top 10 from unsupervised method
        top_unsupervised = sorted(combined_scores, key=lambda x: x['score'], reverse=True)[:10]
        prompts = [score['prompt'][:30] + '...' for score in top_unsupervised]
        scores = [score['score'] for score in top_unsupervised]
        
        ax3.barh(range(len(prompts)), scores)
        ax3.set_yticks(range(len(prompts)))
        ax3.set_yticklabels(prompts, fontsize=8)
        ax3.set_xlabel('Hallucination Score')
        ax3.set_title('Top 10 Most Likely Hallucinations (Unsupervised)')
        ax3.grid(True, alpha=0.3)
    
    # 4. Method statistics
    ax4 = axes[1, 1]
    method_stats = df.groupby('method')['score'].agg(['mean', 'std', 'min', 'max']).reset_index()
    
    x_pos = range(len(method_stats))
    means = method_stats['mean']
    stds = method_stats['std']
    
    ax4.bar(x_pos, means, yerr=stds, capsize=5, alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(method_stats['method'], rotation=45)
    ax4.set_ylabel('Mean Score')
    ax4.set_title('Method Performance Comparison')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('data/method_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison
    print("\n" + "="*60)
    print("METHOD COMPARISON ANALYSIS")
    print("="*60)
    
    print("\nMethod Statistics:")
    for method in methods:
        method_data = df[df['method'] == method]['score']
        print(f"\n{method.upper()}:")
        print(f"  Mean: {method_data.mean():.4f}")
        print(f"  Std:  {method_data.std():.4f}")
        print(f"  Min:  {method_data.min():.4f}")
        print(f"  Max:  {method_data.max():.4f}")
        print(f"  Count: {len(method_data)}")
    
    # Top hallucinations comparison
    if len(combined_scores) > 0:
        print(f"\nTop 5 Most Likely Hallucinations (Unsupervised Combined):")
        top_5 = sorted(combined_scores, key=lambda x: x['score'], reverse=True)[:5]
        for i, score in enumerate(top_5, 1):
            print(f"  {i}. {score['score']:.3f}: {score['prompt'][:60]}...")
    
    # Method correlation analysis
    if len(previous_scores) > 0 and len(combined_scores) > 0:
        print(f"\nCorrelation Analysis:")
        # Create prompt-to-score mapping
        prev_scores_dict = {s['prompt']: s['score'] for s in previous_scores}
        unsup_scores_dict = {s['prompt']: s['score'] for s in combined_scores}
        
        # Find common prompts
        common_prompts = set(prev_scores_dict.keys()) & set(unsup_scores_dict.keys())
        if len(common_prompts) > 0:
            prev_common = [prev_scores_dict[p] for p in common_prompts]
            unsup_common = [unsup_scores_dict[p] for p in common_prompts]
            correlation = np.corrcoef(prev_common, unsup_common)[0, 1]
            print(f"  Correlation between previous and unsupervised: {correlation:.4f}")
            print(f"  Common prompts analyzed: {len(common_prompts)}")
    
    return df

if __name__ == "__main__":
    df = create_comparison_analysis()
    print(f"\nComparison complete! Results saved to data/method_comparison.png")






