"""
Pytest configuration and shared fixtures for hallucination detection tests.
"""

import pytest
import numpy as np
import tempfile
import os
from typing import List, Dict, Tuple


@pytest.fixture(scope="session")
def sample_embeddings():
    """Create sample embeddings for testing across all test modules."""
    np.random.seed(42)  # For reproducible tests
    
    # Create 3 distinct clusters
    cluster1 = np.random.normal([1, 1], 0.1, (10, 2))  # Cluster 1
    cluster2 = np.random.normal([-1, -1], 0.1, (10, 2))  # Cluster 2  
    cluster3 = np.random.normal([0, 2], 0.1, (10, 2))   # Cluster 3
    
    # Combine clusters
    embeddings = np.vstack([cluster1, cluster2, cluster3])
    
    # Add some noise points
    noise = np.random.uniform(-2, 2, (5, 2))
    embeddings = np.vstack([embeddings, noise])
    
    return embeddings


@pytest.fixture(scope="session")
def sample_prompts():
    """Create sample prompts for testing across all test modules."""
    return [
        "What is the capital of France?",
        "How do you make coffee?",
        "What is machine learning?",
        "Explain quantum computing",
        "Who wrote Romeo and Juliet?",
        "What is 2 + 2?",
        "How do you tie a tie?",
        "What is the meaning of life?",
        "Explain the theory of relativity",
        "How do you plant a garden?",
        "What does a unicorn eat for breakfast?",
        "How many angels can dance on the head of a pin?",
        "What is the taste of the number seven?",
        "How do you fold a four-dimensional cube?",
        "What is the sound of a triangle?",
        "How do you catch a cloud?",
        "What is the color of Tuesday?",
        "How do you build a time machine?",
        "What is the weight of a thought?",
        "How do you organize a closet?",
        "What is the speed of dark?",
        "How do you divide by zero?",
        "What is the smell of music?",
        "How do you count to infinity?",
        "What is the temperature of silence?"
    ]


@pytest.fixture
def temp_analysis_file(sample_embeddings, sample_prompts):
    """Create a temporary analysis file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        import json
        data = {
            'embeddings': sample_embeddings.tolist(),
            'prompts': sample_prompts
        }
        json.dump(data, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    try:
        os.unlink(temp_path)
    except FileNotFoundError:
        pass


@pytest.fixture
def mock_matplotlib():
    """Mock matplotlib to avoid GUI issues in tests."""
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    return matplotlib


@pytest.fixture
def high_dimensional_embeddings():
    """Create high-dimensional embeddings for testing."""
    np.random.seed(42)
    return np.random.normal(0, 1, (20, 50))  # 20 samples, 50 dimensions


@pytest.fixture
def low_dimensional_embeddings():
    """Create low-dimensional embeddings for testing."""
    np.random.seed(42)
    return np.random.normal(0, 1, (20, 2))  # 20 samples, 2 dimensions


@pytest.fixture
def detector_params():
    """Default detector parameters for testing."""
    return {
        "min_cluster_size": 3,
        "min_samples": 2
    }


# Test data generators
def generate_clustered_embeddings(n_clusters: int = 3, 
                                points_per_cluster: int = 10,
                                noise_points: int = 5,
                                dimensions: int = 2,
                                seed: int = 42) -> np.ndarray:
    """Generate embeddings with known cluster structure."""
    np.random.seed(seed)
    
    embeddings = []
    
    # Generate clusters
    for i in range(n_clusters):
        center = np.random.uniform(-2, 2, dimensions)
        cluster = np.random.normal(center, 0.1, (points_per_cluster, dimensions))
        embeddings.append(cluster)
    
    # Add noise points
    noise = np.random.uniform(-3, 3, (noise_points, dimensions))
    embeddings.append(noise)
    
    return np.vstack(embeddings)


def generate_random_embeddings(n_points: int = 20, 
                             dimensions: int = 2, 
                             seed: int = 42) -> np.ndarray:
    """Generate random embeddings with no cluster structure."""
    np.random.seed(seed)
    return np.random.normal(0, 1, (n_points, dimensions))


# Utility functions for tests
def assert_embeddings_equal(emb1: np.ndarray, emb2: np.ndarray, rtol: float = 1e-5):
    """Assert that two embedding arrays are equal within tolerance."""
    assert emb1.shape == emb2.shape, f"Shape mismatch: {emb1.shape} vs {emb2.shape}"
    assert np.allclose(emb1, emb2, rtol=rtol), "Embeddings not equal within tolerance"


def assert_valid_clustering_stats(stats: Dict):
    """Assert that clustering statistics are valid."""
    required_keys = ["n_clusters", "n_noise_points", "noise_percentage", "silhouette_score"]
    
    for key in required_keys:
        assert key in stats, f"Missing key: {key}"
    
    assert isinstance(stats["n_clusters"], int)
    assert stats["n_clusters"] >= 0
    assert isinstance(stats["n_noise_points"], int)
    assert stats["n_noise_points"] >= 0
    assert isinstance(stats["noise_percentage"], (int, float))
    assert 0 <= stats["noise_percentage"] <= 100
    assert isinstance(stats["silhouette_score"], (int, float))
    assert -1 <= stats["silhouette_score"] <= 1


def assert_valid_scoring_results(results: Dict, expected_n_prompts: int):
    """Assert that scoring results are valid."""
    required_keys = ["scores", "mean_distance", "std_distance", "min_distance", "max_distance"]
    
    for key in required_keys:
        assert key in results, f"Missing key: {key}"
    
    assert len(results["scores"]) == expected_n_prompts
    
    # Check that scores are sorted by distance (descending)
    distances = [s["manifold_distance"] for s in results["scores"]]
    assert distances == sorted(distances, reverse=True)
    
    # Check that rankings are correct
    for i, score in enumerate(results["scores"]):
        assert score["rank"] == i + 1
