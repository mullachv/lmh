#!/usr/bin/env python3
"""
Hallucination Scoring via Manifold Distance

This module implements distance-based hallucination scoring by measuring
how far an embedding is from learned low-dimensional manifolds in embedding space.
"""

import numpy as np
import faiss
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from typing import List, Tuple, Optional, Union
import json
import os


def hallucination_score(
    embedding: np.ndarray,
    manifold_embeddings: np.ndarray,
    method: str = "cosine",
    top_k: int = 5,
    normalize: bool = True
) -> float:
    """
    Calculate hallucination likelihood score based on distance to known manifold.
    
    Args:
        embedding: Single embedding vector (1D array)
        manifold_embeddings: Array of embeddings from the learned manifold (2D array)
        method: Distance metric ("cosine", "euclidean", "manhattan", "faiss_cosine", "faiss_l2")
        top_k: Number of nearest neighbors to consider
        normalize: Whether to normalize the final score to [0,1] range
        
    Returns:
        Hallucination score (higher = more likely to be hallucination)
    """
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    
    if manifold_embeddings.shape[0] == 0:
        return 1.0  # No manifold data available
    
    # Ensure embeddings are float32 for FAISS
    embedding = embedding.astype(np.float32)
    manifold_embeddings = manifold_embeddings.astype(np.float32)
    
    if method in ["faiss_cosine", "faiss_l2"]:
        return _faiss_distance_score(embedding, manifold_embeddings, method, top_k, normalize)
    else:
        return _sklearn_distance_score(embedding, manifold_embeddings, method, top_k, normalize)


def _faiss_distance_score(
    embedding: np.ndarray,
    manifold_embeddings: np.ndarray,
    method: str,
    top_k: int,
    normalize: bool
) -> float:
    """Calculate distance using FAISS for efficiency."""
    dim = embedding.shape[1]
    
    # Create FAISS index
    if method == "faiss_cosine":
        # For cosine similarity, we need to normalize embeddings first
        faiss.normalize_L2(embedding)
        faiss.normalize_L2(manifold_embeddings)
        index = faiss.IndexFlatIP(dim)  # Inner product for cosine similarity
    else:  # faiss_l2
        index = faiss.IndexFlatL2(dim)
    
    # Add manifold embeddings to index
    index.add(manifold_embeddings)
    
    # Search for nearest neighbors
    distances, indices = index.search(embedding, min(top_k, manifold_embeddings.shape[0]))
    
    # For cosine similarity, convert to distance (1 - similarity)
    if method == "faiss_cosine":
        distances = 1 - distances
    
    # Aggregate distances (mean of top-k)
    score = float(np.mean(distances[0]))
    
    if normalize:
        # Normalize based on method-specific ranges
        if method == "faiss_cosine":
            score = min(score, 1.0)  # Cosine distance is bounded by [0,2]
        else:
            # For L2, normalize by a reasonable upper bound
            max_distance = np.sqrt(dim) * 2  # Rough upper bound for normalized embeddings
            score = min(score / max_distance, 1.0)
    
    return score


def _sklearn_distance_score(
    embedding: np.ndarray,
    manifold_embeddings: np.ndarray,
    method: str,
    top_k: int,
    normalize: bool
) -> float:
    """Calculate distance using sklearn for smaller datasets."""
    if method == "cosine":
        distances = cosine_distances(embedding, manifold_embeddings)
    elif method == "euclidean":
        distances = euclidean_distances(embedding, manifold_embeddings)
    elif method == "manhattan":
        distances = np.sum(np.abs(embedding - manifold_embeddings), axis=1)
        distances = distances.reshape(1, -1)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Get top-k distances
    top_distances = np.partition(distances[0], min(top_k, len(distances[0])-1))[:top_k]
    
    # Aggregate (mean of top-k)
    score = float(np.mean(top_distances))
    
    if normalize:
        if method == "cosine":
            score = min(score, 1.0)  # Cosine distance is bounded by [0,2]
        elif method == "euclidean":
            # Normalize by embedding dimension
            max_distance = np.sqrt(embedding.shape[1]) * 2
            score = min(score / max_distance, 1.0)
        elif method == "manhattan":
            # Normalize by embedding dimension
            max_distance = embedding.shape[1] * 2
            score = min(score / max_distance, 1.0)
    
    return score


def batch_hallucination_scores(
    embeddings: np.ndarray,
    manifold_embeddings: np.ndarray,
    method: str = "cosine",
    top_k: int = 5,
    normalize: bool = True
) -> np.ndarray:
    """
    Calculate hallucination scores for a batch of embeddings.
    
    Args:
        embeddings: Array of embeddings to score (2D array)
        manifold_embeddings: Array of embeddings from the learned manifold (2D array)
        method: Distance metric
        top_k: Number of nearest neighbors to consider
        normalize: Whether to normalize the final scores
        
    Returns:
        Array of hallucination scores
    """
    scores = []
    for i in range(embeddings.shape[0]):
        score = hallucination_score(
            embeddings[i], manifold_embeddings, method, top_k, normalize
        )
        scores.append(score)
    
    return np.array(scores)


def analyze_hallucination_distribution(
    scores: np.ndarray,
    threshold: float = 0.5
) -> dict:
    """
    Analyze the distribution of hallucination scores.
    
    Args:
        scores: Array of hallucination scores
        threshold: Threshold for binary classification
        
    Returns:
        Dictionary with statistics
    """
    stats = {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "median": float(np.median(scores)),
        "percentile_25": float(np.percentile(scores, 25)),
        "percentile_75": float(np.percentile(scores, 75)),
        "percentile_90": float(np.percentile(scores, 90)),
        "percentile_95": float(np.percentile(scores, 95)),
        "above_threshold": int(np.sum(scores > threshold)),
        "below_threshold": int(np.sum(scores <= threshold)),
        "threshold": threshold
    }
    
    return stats


def save_hallucination_analysis(
    scores: np.ndarray,
    prompts: List[str],
    model_name: str,
    method: str,
    output_file: str
) -> None:
    """
    Save hallucination analysis results to JSON.
    
    Args:
        scores: Array of hallucination scores
        prompts: List of corresponding prompts
        model_name: Name of the model used
        method: Distance method used
        output_file: Path to output JSON file
    """
    analysis = {
        "model_name": model_name,
        "method": method,
        "total_prompts": len(prompts),
        "scores": scores.tolist(),
        "statistics": analyze_hallucination_distribution(scores),
        "prompts_with_scores": [
            {"prompt": prompt, "score": float(score)}
            for prompt, score in zip(prompts, scores)
        ]
    }
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(analysis, f, indent=2)


def load_manifold_embeddings(file_path: str) -> np.ndarray:
    """
    Load manifold embeddings from JSON file.
    
    Args:
        file_path: Path to JSON file containing embeddings
        
    Returns:
        Array of embeddings
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if "embeddings" in data:
        return np.array(data["embeddings"])
    elif "manifold_embeddings" in data:
        return np.array(data["manifold_embeddings"])
    else:
        raise ValueError(f"No embeddings found in {file_path}")


if __name__ == "__main__":
    # Example usage
    print("Hallucination Scoring Module")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    manifold_embeddings = np.random.randn(100, 384)  # 100 samples, 384D
    test_embedding = np.random.randn(384)
    
    # Test different methods
    methods = ["cosine", "euclidean", "faiss_cosine", "faiss_l2"]
    
    for method in methods:
        score = hallucination_score(test_embedding, manifold_embeddings, method=method)
        print(f"{method:15}: {score:.4f}")
    
    print("\nBatch scoring example:")
    test_embeddings = np.random.randn(5, 384)
    batch_scores = batch_hallucination_scores(test_embeddings, manifold_embeddings)
    print(f"Batch scores: {batch_scores}")
    
    # Analyze distribution
    stats = analyze_hallucination_distribution(batch_scores)
    print(f"\nStatistics: {stats}")




