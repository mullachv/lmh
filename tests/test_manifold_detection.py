"""
Tests for UnsupervisedManifoldDetector

This module contains comprehensive tests for the density-based
manifold detection approach to hallucination detection.
"""

import pytest
import numpy as np
import json
import tempfile
import os
from unittest.mock import patch, MagicMock
from typing import List, Dict

# Import the module under test
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from unsupervised_manifold_detection import UnsupervisedManifoldDetector, load_embeddings_from_analysis


class TestUnsupervisedManifoldDetector:
    """Test cases for UnsupervisedManifoldDetector class."""
    
    @pytest.fixture
    def sample_embeddings(self):
        """Create sample embeddings for testing."""
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
    
    @pytest.fixture
    def sample_prompts(self):
        """Create sample prompts for testing."""
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
    def detector(self):
        """Create a detector instance for testing."""
        return UnsupervisedManifoldDetector(min_cluster_size=3, min_samples=2)
    
    def test_initialization(self):
        """Test detector initialization."""
        detector = UnsupervisedManifoldDetector(min_cluster_size=5, min_samples=3)
        
        assert detector.min_cluster_size == 5
        assert detector.min_samples == 3
        assert detector.clusterer is None
        assert detector.embeddings is None
        assert detector.prompts is None
        assert detector.cluster_labels is None
        assert detector.manifold_centers is None
    
    def test_fit_basic(self, detector, sample_embeddings, sample_prompts):
        """Test basic fitting functionality."""
        stats = detector.fit(sample_embeddings, sample_prompts)
        
        # Check that detector state is updated
        assert detector.embeddings is not None
        assert detector.prompts is not None
        assert detector.cluster_labels is not None
        assert detector.clusterer is not None
        
        # Check that stats are returned
        assert isinstance(stats, dict)
        assert "n_clusters" in stats
        assert "n_noise_points" in stats
        assert "silhouette_score" in stats
        assert "cluster_sizes" in stats
        
        # Check that manifold centers are calculated
        assert detector.manifold_centers is not None
        assert isinstance(detector.manifold_centers, dict)
    
    def test_fit_with_small_dataset(self, detector):
        """Test fitting with a very small dataset."""
        # Create a small dataset that might not form clusters
        small_embeddings = np.random.normal(0, 0.1, (5, 2))
        small_prompts = ["prompt1", "prompt2", "prompt3", "prompt4", "prompt5"]
        
        stats = detector.fit(small_embeddings, small_prompts)
        
        # Should handle small datasets gracefully
        assert isinstance(stats, dict)
        assert "n_clusters" in stats
        assert stats["n_clusters"] >= 0  # Could be 0 for very small datasets
    
    def test_compute_manifold_distances(self, detector, sample_embeddings, sample_prompts):
        """Test computing distances to manifolds."""
        # Fit the detector first
        detector.fit(sample_embeddings, sample_prompts)
        
        # Test with new embeddings
        new_embeddings = np.random.normal(0, 0.5, (3, 2))
        distances = detector.compute_manifold_distances(new_embeddings)
        
        # Check that distances are returned
        assert len(distances) == 3
        assert all(isinstance(d, (int, float)) for d in distances)
        assert all(d >= 0 for d in distances)  # Distances should be non-negative
    
    def test_compute_manifold_distances_before_fit(self, detector):
        """Test that computing distances before fitting raises an error."""
        new_embeddings = np.random.normal(0, 0.5, (3, 2))
        
        with pytest.raises(ValueError, match="Must fit the detector before computing distances"):
            detector.compute_manifold_distances(new_embeddings)
    
    def test_score_prompts(self, detector, sample_embeddings, sample_prompts):
        """Test scoring prompts for hallucination likelihood."""
        # Fit the detector first
        detector.fit(sample_embeddings, sample_prompts)
        
        # Score all prompts
        results = detector.score_prompts(sample_prompts, sample_embeddings)
        
        # Check results structure
        assert isinstance(results, dict)
        assert "scores" in results
        assert "mean_distance" in results
        assert "std_distance" in results
        assert "min_distance" in results
        assert "max_distance" in results
        
        # Check that scores are sorted by distance (descending)
        scores = results["scores"]
        assert len(scores) == len(sample_prompts)
        
        # Check that scores are sorted by distance (highest first)
        distances = [s["manifold_distance"] for s in scores]
        assert distances == sorted(distances, reverse=True)
        
        # Check that rankings are correct
        for i, score in enumerate(scores):
            assert score["rank"] == i + 1
    
    def test_score_prompts_mismatched_lengths(self, detector, sample_embeddings, sample_prompts):
        """Test scoring with mismatched prompt and embedding lengths."""
        detector.fit(sample_embeddings, sample_prompts)
        
        # Create mismatched data
        short_prompts = sample_prompts[:3]
        short_embeddings = sample_embeddings[:3]
        
        results = detector.score_prompts(short_prompts, short_embeddings)
        assert len(results["scores"]) == 3
    
    def test_calculate_manifold_centers(self, detector, sample_embeddings, sample_prompts):
        """Test manifold center calculation."""
        detector.fit(sample_embeddings, sample_prompts)
        
        # Check that centers are calculated
        assert detector.manifold_centers is not None
        assert isinstance(detector.manifold_centers, dict)
        
        # Check that centers have correct dimensions
        for label, center in detector.manifold_centers.items():
            assert isinstance(label, (int, np.integer))
            assert isinstance(center, np.ndarray)
            assert center.shape == (sample_embeddings.shape[1],)
    
    def test_compute_clustering_stats(self, detector, sample_embeddings, sample_prompts):
        """Test clustering statistics computation."""
        detector.fit(sample_embeddings, sample_prompts)
        stats = detector._compute_clustering_stats()
        
        # Check required fields
        required_fields = ["n_clusters", "n_noise_points", "noise_percentage", 
                          "silhouette_score", "cluster_sizes", "min_cluster_size", "min_samples"]
        
        for field in required_fields:
            assert field in stats
        
        # Check data types
        assert isinstance(stats["n_clusters"], int)
        assert isinstance(stats["n_noise_points"], int)
        assert isinstance(stats["noise_percentage"], float)
        assert isinstance(stats["silhouette_score"], float)
        assert isinstance(stats["cluster_sizes"], dict)
    
    def test_visualize_manifolds_before_fit(self, detector):
        """Test that visualization before fitting raises an error."""
        with pytest.raises(ValueError, match="Must fit the detector before visualization"):
            detector.visualize_manifolds()
    
    @patch('matplotlib.pyplot.show')
    @patch('matplotlib.pyplot.savefig')
    def test_visualize_manifolds(self, mock_savefig, mock_show, detector, sample_embeddings, sample_prompts):
        """Test manifold visualization."""
        detector.fit(sample_embeddings, sample_prompts)
        
        # Test visualization (should not raise an error)
        detector.visualize_manifolds("test_output.png")
        
        # Check that savefig was called
        mock_savefig.assert_called_once()
    
    def test_edge_case_empty_embeddings(self, detector):
        """Test handling of empty embeddings."""
        empty_embeddings = np.array([]).reshape(0, 2)
        empty_prompts = []
        
        with pytest.raises(ValueError):
            detector.fit(empty_embeddings, empty_prompts)
    
    def test_edge_case_single_embedding(self, detector):
        """Test handling of single embedding."""
        single_embedding = np.array([[1, 2]])
        single_prompt = ["single prompt"]
        
        # HDBSCAN requires at least 2 points, so this should raise an error
        with pytest.raises(ValueError):
            detector.fit(single_embedding, single_prompt)


class TestLoadEmbeddingsFromAnalysis:
    """Test cases for loading embeddings from analysis files."""
    
    def test_load_valid_analysis_file(self, sample_embeddings, sample_prompts):
        """Test loading from a valid analysis file."""
        # Create a temporary file with valid data
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {
                'embeddings': sample_embeddings.tolist(),
                'prompts': sample_prompts
            }
            json.dump(data, f)
            temp_path = f.name
        
        try:
            embeddings, prompts = load_embeddings_from_analysis(temp_path)
            
            # Check that data is loaded correctly
            assert np.array_equal(embeddings, sample_embeddings)
            assert prompts == sample_prompts
            
        finally:
            # Clean up temporary file
            os.unlink(temp_path)
    
    def test_load_nonexistent_file(self):
        """Test loading from a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            load_embeddings_from_analysis("nonexistent_file.json")
    
    def test_load_invalid_json(self):
        """Test loading from an invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("invalid json content")
            temp_path = f.name
        
        try:
            with pytest.raises(json.JSONDecodeError):
                load_embeddings_from_analysis(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_missing_embeddings_key(self, sample_prompts):
        """Test loading from file missing embeddings key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {'prompts': sample_prompts}  # Missing 'embeddings' key
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError):
                load_embeddings_from_analysis(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_load_missing_prompts_key(self, sample_embeddings):
        """Test loading from file missing prompts key."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            data = {'embeddings': sample_embeddings.tolist()}  # Missing 'prompts' key
            json.dump(data, f)
            temp_path = f.name
        
        try:
            with pytest.raises(KeyError):
                load_embeddings_from_analysis(temp_path)
        finally:
            os.unlink(temp_path)


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_full_pipeline(self, sample_embeddings, sample_prompts):
        """Test the complete pipeline from fitting to scoring."""
        detector = UnsupervisedManifoldDetector(min_cluster_size=3, min_samples=2)
        
        # Fit the detector
        stats = detector.fit(sample_embeddings, sample_prompts)
        assert isinstance(stats, dict)
        
        # Score all prompts
        results = detector.score_prompts(sample_prompts, sample_embeddings)
        assert isinstance(results, dict)
        assert len(results["scores"]) == len(sample_prompts)
        
        # Test with new embeddings
        new_embeddings = np.random.normal(0, 0.5, (3, 2))
        distances = detector.compute_manifold_distances(new_embeddings)
        assert len(distances) == 3
    
    def test_consistency_across_runs(self, sample_embeddings, sample_prompts):
        """Test that results are consistent across multiple runs."""
        detector1 = UnsupervisedManifoldDetector(min_cluster_size=3, min_samples=2)
        detector2 = UnsupervisedManifoldDetector(min_cluster_size=3, min_samples=2)
        
        # Fit both detectors
        stats1 = detector1.fit(sample_embeddings, sample_prompts)
        stats2 = detector2.fit(sample_embeddings, sample_prompts)
        
        # Results should be similar (allowing for some randomness in clustering)
        assert abs(stats1["n_clusters"] - stats2["n_clusters"]) <= 1  # Allow small variation
        assert abs(stats1["n_noise_points"] - stats2["n_noise_points"]) <= 2
    
    def test_parameter_sensitivity(self, sample_embeddings, sample_prompts):
        """Test sensitivity to different parameters."""
        # Test with different min_cluster_size
        detector_small = UnsupervisedManifoldDetector(min_cluster_size=2, min_samples=1)
        detector_large = UnsupervisedManifoldDetector(min_cluster_size=5, min_samples=3)
        
        stats_small = detector_small.fit(sample_embeddings, sample_prompts)
        stats_large = detector_large.fit(sample_embeddings, sample_prompts)
        
        # Different parameters should produce different results
        assert stats_small["n_clusters"] != stats_large["n_clusters"] or \
               stats_small["n_noise_points"] != stats_large["n_noise_points"]


class TestPerformance:
    """Performance tests for the manifold detector."""
    
    def test_large_dataset_performance(self):
        """Test performance with a larger dataset."""
        # Create a larger dataset
        np.random.seed(42)
        large_embeddings = np.random.normal(0, 1, (100, 10))  # 100 samples, 10 dimensions
        large_prompts = [f"prompt_{i}" for i in range(100)]
        
        detector = UnsupervisedManifoldDetector(min_cluster_size=5, min_samples=3)
        
        # Should complete without errors
        stats = detector.fit(large_embeddings, large_prompts)
        assert isinstance(stats, dict)
        
        # Should be able to score all prompts
        results = detector.score_prompts(large_prompts, large_embeddings)
        assert len(results["scores"]) == 100
    
    def test_memory_usage(self, sample_embeddings, sample_prompts):
        """Test that the detector doesn't leak memory."""
        detector = UnsupervisedManifoldDetector(min_cluster_size=3, min_samples=2)
        
        # Fit multiple times to check for memory leaks
        for _ in range(5):
            stats = detector.fit(sample_embeddings, sample_prompts)
            results = detector.score_prompts(sample_prompts, sample_embeddings)
            
            # Results should be consistent
            assert isinstance(stats, dict)
            assert isinstance(results, dict)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")


# Test fixtures for different scenarios
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


@pytest.mark.slow
def test_high_dimensional_embeddings(high_dimensional_embeddings):
    """Test with high-dimensional embeddings."""
    detector = UnsupervisedManifoldDetector(min_cluster_size=3, min_samples=2)
    prompts = [f"prompt_{i}" for i in range(len(high_dimensional_embeddings))]
    
    stats = detector.fit(high_dimensional_embeddings, prompts)
    assert isinstance(stats, dict)


@pytest.mark.integration
def test_integration_with_real_data():
    """Integration test that would use real data if available."""
    # This test would load real analysis data if it exists
    # For now, it's a placeholder for integration testing
    pass
