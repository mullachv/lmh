#!/usr/bin/env python3
"""
Reality Manifold Hallucination Scoring

This module implements hallucination detection by measuring distance to a 
"reality manifold" - a learned representation of factual, well-grounded prompts.
"""

import json
import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_distances
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt


def classify_prompts_by_reality(prompts: List[str]) -> Dict[str, List[int]]:
    """
    Classify prompts into reality-based categories for manifold learning.
    
    Args:
        prompts: List of all prompts
        
    Returns:
        Dictionary with 'factual', 'hallucinatory', 'ambiguous' categories
    """
    factual_keywords = [
        'what is', 'how do you', 'who is', 'where is', 'when is', 'why is',
        'capital', 'president', 'president', 'moons', 'speed of light',
        'photosynthesis', 'bake', 'largest planet', 'painted', 'height',
        'machine learning', 'quantum computing', 'neural network', 'blockchain',
        'study', 'coffee', 'tie', 'garden', 'time machine'
    ]
    
    hallucinatory_keywords = [
        'unicorn', 'angels', 'invisibility potion', 'hobbits', 'color of sound',
        'taste of number', 'catch a cloud', 'four-dimensional', 'napoleon bitcoin',
        'borsuk-ulam', 'relativity', 'gradient descent', 'string theory'
    ]
    
    ambiguous_keywords = [
        'meaning of life', 'best way', 'organize', 'fold', 'divide by zero'
    ]
    
    classification = {
        'factual': [],
        'hallucinatory': [], 
        'ambiguous': []
    }
    
    for i, prompt in enumerate(prompts):
        prompt_lower = prompt.lower()
        
        # Check for hallucinatory indicators
        if any(keyword in prompt_lower for keyword in hallucinatory_keywords):
            classification['hallucinatory'].append(i)
        # Check for ambiguous indicators  
        elif any(keyword in prompt_lower for keyword in ambiguous_keywords):
            classification['ambiguous'].append(i)
        # Default to factual for well-formed questions
        elif any(keyword in prompt_lower for keyword in factual_keywords):
            classification['factual'].append(i)
        else:
            # Default classification based on question structure
            if prompt.endswith('?') and len(prompt.split()) > 3:
                classification['factual'].append(i)
            else:
                classification['ambiguous'].append(i)
    
    return classification


def create_reality_manifold(embeddings: np.ndarray, factual_indices: List[int]) -> np.ndarray:
    """
    Create a reality manifold from factual embeddings.
    
    Args:
        embeddings: All embeddings (2D array)
        factual_indices: Indices of factual prompts
        
    Returns:
        Reality manifold embeddings
    """
    if len(factual_indices) == 0:
        raise ValueError("No factual prompts found for reality manifold")
    
    reality_embeddings = embeddings[factual_indices]
    print(f"Created reality manifold with {len(reality_embeddings)} factual embeddings")
    
    return reality_embeddings


def reality_manifold_hallucination_score(
    embedding: np.ndarray,
    reality_manifold: np.ndarray,
    method: str = "cosine",
    top_k: int = 5
) -> float:
    """
    Calculate hallucination score based on distance to reality manifold.
    
    Args:
        embedding: Single embedding vector
        reality_manifold: Array of factual embeddings
        method: Distance metric
        top_k: Number of nearest neighbors
        
    Returns:
        Hallucination score (higher = more likely hallucination)
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    if reality_manifold.shape[0] == 0:
        return 1.0
    
    # Ensure float32 for FAISS
    embedding = embedding.astype(np.float32)
    reality_manifold = reality_manifold.astype(np.float32)
    
    if method == "cosine":
        distances = cosine_distances(embedding, reality_manifold)
        score = float(np.mean(np.partition(distances[0], min(top_k, len(distances[0])-1))[:top_k]))
    elif method == "faiss_cosine":
        # Normalize for cosine similarity
        faiss.normalize_L2(embedding)
        faiss.normalize_L2(reality_manifold)
        
        dim = embedding.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(reality_manifold)
        
        similarities, _ = index.search(embedding, min(top_k, reality_manifold.shape[0]))
        # Convert similarity to distance
        distances = 1 - similarities
        score = float(np.mean(distances[0]))
    else:
        raise ValueError(f"Unsupported method: {method}")
    
    return score


def batch_reality_manifold_scores(
    embeddings: np.ndarray,
    reality_manifold: np.ndarray,
    method: str = "cosine",
    top_k: int = 5
) -> np.ndarray:
    """
    Calculate reality manifold scores for all embeddings.
    """
    scores = []
    for i in range(embeddings.shape[0]):
        score = reality_manifold_hallucination_score(
            embeddings[i], reality_manifold, method, top_k
        )
        scores.append(score)
    
    return np.array(scores)


def analyze_reality_manifold_results(
    scores: np.ndarray,
    prompts: List[str],
    classification: Dict[str, List[int]]
) -> Dict:
    """
    Analyze results of reality manifold scoring.
    """
    results = {
        'total_prompts': len(scores),
        'factual_count': len(classification['factual']),
        'hallucinatory_count': len(classification['hallucinatory']),
        'ambiguous_count': len(classification['ambiguous']),
        'statistics': {
            'mean': float(np.mean(scores)),
            'std': float(np.std(scores)),
            'min': float(np.min(scores)),
            'max': float(np.max(scores)),
            'median': float(np.median(scores))
        }
    }
    
    # Analyze by category
    for category, indices in classification.items():
        if indices:
            category_scores = [scores[i] for i in indices]
            results[f'{category}_scores'] = {
                'mean': float(np.mean(category_scores)),
                'std': float(np.std(category_scores)),
                'min': float(np.min(category_scores)),
                'max': float(np.max(category_scores))
            }
    
    # Find top hallucination candidates
    sorted_indices = np.argsort(scores)
    results['top_hallucination_candidates'] = [
        {
            'index': int(idx),
            'score': float(scores[idx]),
            'prompt': prompts[idx]
        }
        for idx in sorted_indices[-10:]
    ]
    
    return results


def main():
    """Test the reality manifold approach."""
    print("Reality Manifold Hallucination Scoring")
    print("=" * 50)
    
    # Load LLaMA analysis
    with open('../data/llama_simple_analysis.json', 'r') as f:
        data = json.load(f)
    
    embeddings = np.array(data['embeddings'])
    prompts = data['prompts']
    
    print(f"Loaded {len(embeddings)} embeddings")
    
    # Classify prompts
    classification = classify_prompts_by_reality(prompts)
    
    print(f"\nPrompt Classification:")
    print(f"  Factual: {len(classification['factual'])}")
    print(f"  Hallucinatory: {len(classification['hallucinatory'])}")
    print(f"  Ambiguous: {len(classification['ambiguous'])}")
    
    # Create reality manifold
    reality_manifold = create_reality_manifold(embeddings, classification['factual'])
    
    # Calculate scores
    scores = batch_reality_manifold_scores(embeddings, reality_manifold, method="cosine")
    
    # Analyze results
    results = analyze_reality_manifold_results(scores, prompts, classification)
    
    print(f"\nReality Manifold Scoring Results:")
    print(f"  Mean score: {results['statistics']['mean']:.4f}")
    print(f"  Std score: {results['statistics']['std']:.4f}")
    
    print(f"\nBy Category:")
    for category in ['factual', 'hallucinatory', 'ambiguous']:
        if f'{category}_scores' in results:
            stats = results[f'{category}_scores']
            print(f"  {category.capitalize()}: mean={stats['mean']:.4f}, std={stats['std']:.4f}")
    
    print(f"\nTop 10 Hallucination Candidates:")
    for i, candidate in enumerate(results['top_hallucination_candidates'], 1):
        print(f"  {i:2d}. Score {candidate['score']:.4f}: {candidate['prompt']}")
    
    # Save results
    with open('../data/reality_manifold_analysis.json', 'w') as f:
        json.dump({
            'classification': classification,
            'scores': scores.tolist(),
            'results': results
        }, f, indent=2)
    
    print(f"\nResults saved to ../data/reality_manifold_analysis.json")


if __name__ == "__main__":
    main()
