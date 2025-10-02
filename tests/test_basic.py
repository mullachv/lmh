"""
Basic tests to verify the testing setup works correctly.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))


def test_numpy_import():
    """Test that numpy can be imported and used."""
    arr = np.array([1, 2, 3, 4, 5])
    assert len(arr) == 5
    assert arr[0] == 1
    assert arr[-1] == 5


def test_pytest_working():
    """Test that pytest is working correctly."""
    assert True


def test_import_manifold_detector():
    """Test that we can import the manifold detector."""
    try:
        from unsupervised_manifold_detection import UnsupervisedManifoldDetector
        detector = UnsupervisedManifoldDetector()
        assert detector is not None
    except ImportError as e:
        pytest.fail(f"Could not import UnsupervisedManifoldDetector: {e}")


def test_basic_manifold_detection():
    """Test basic functionality of the manifold detector."""
    from unsupervised_manifold_detection import UnsupervisedManifoldDetector
    
    # Create simple test data
    np.random.seed(42)
    embeddings = np.random.normal(0, 1, (10, 2))
    prompts = [f"prompt_{i}" for i in range(10)]
    
    # Initialize detector
    detector = UnsupervisedManifoldDetector(min_cluster_size=2, min_samples=1)
    
    # Fit the detector
    stats = detector.fit(embeddings, prompts)
    
    # Check that stats are returned
    assert isinstance(stats, dict)
    assert "n_clusters" in stats
    assert "n_noise_points" in stats
    
    # Check that detector state is updated
    assert detector.embeddings is not None
    assert detector.prompts is not None
    assert detector.cluster_labels is not None


def test_embeddings_shape():
    """Test that embeddings have the expected shape."""
    np.random.seed(42)
    embeddings = np.random.normal(0, 1, (5, 3))
    
    assert embeddings.shape == (5, 3)
    assert embeddings.ndim == 2


@pytest.mark.parametrize("n_samples,n_features", [
    (10, 2),
    (20, 5),
    (5, 10),
    (1, 1)
])
def test_embedding_dimensions(n_samples, n_features):
    """Test embeddings with different dimensions."""
    np.random.seed(42)
    embeddings = np.random.normal(0, 1, (n_samples, n_features))
    
    assert embeddings.shape == (n_samples, n_features)
    assert embeddings.size == n_samples * n_features
