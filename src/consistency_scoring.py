"""
Consistency-Based Hallucination Detection

This module implements hallucination detection by measuring the consistency
of multiple responses to the same prompt. High variance in embeddings
suggests potential hallucination.

Key Methods:
1. Generate multiple responses to same prompt
2. Compute embedding variance within responses
3. High variance = potential hallucination
4. No external labels required - self-supervised
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import entropy
import seaborn as sns


class ConsistencyScorer:
    """
    Hallucination detection based on response consistency.
    
    This approach measures how consistent multiple responses are to the same
    prompt. High variance suggests the model is uncertain or hallucinating.
    """
    
    def __init__(self, n_responses: int = 5, temperature: float = 0.7):
        """
        Initialize the consistency scorer.
        
        Args:
            n_responses: Number of responses to generate per prompt
            temperature: Sampling temperature for response generation
        """
        self.n_responses = n_responses
        self.temperature = temperature
        self.consistency_scores = {}
        
    def compute_response_variance(self, response_embeddings: np.ndarray) -> Dict:
        """
        Compute variance metrics for a set of response embeddings.
        
        Args:
            response_embeddings: Array of embeddings (n_responses, n_features)
            
        Returns:
            Dictionary with variance metrics
        """
        if len(response_embeddings) < 2:
            return {"variance": 0.0, "std": 0.0, "entropy": 0.0}
        
        # Compute pairwise cosine distances
        distances = cosine_distances(response_embeddings)
        
        # Remove diagonal (self-distances)
        mask = ~np.eye(distances.shape[0], dtype=bool)
        pairwise_distances = distances[mask]
        
        # Compute variance metrics
        mean_distance = np.mean(pairwise_distances)
        std_distance = np.std(pairwise_distances)
        max_distance = np.max(pairwise_distances)
        min_distance = np.min(pairwise_distances)
        
        # Compute embedding variance (across features)
        embedding_variance = np.var(response_embeddings, axis=0)
        mean_embedding_variance = np.mean(embedding_variance)
        
        # Compute entropy of embedding distribution
        embedding_entropy = entropy(embedding_variance + 1e-8)  # Add small epsilon
        
        return {
            "mean_pairwise_distance": float(mean_distance),
            "std_pairwise_distance": float(std_distance),
            "max_pairwise_distance": float(max_distance),
            "min_pairwise_distance": float(min_distance),
            "embedding_variance": float(mean_embedding_variance),
            "embedding_entropy": float(embedding_entropy),
            "consistency_score": float(1.0 - mean_distance),  # Higher = more consistent
            "n_responses": len(response_embeddings)
        }
    
    def score_prompt_consistency(self, prompt: str, response_embeddings: List[np.ndarray]) -> Dict:
        """
        Score consistency for a single prompt.
        
        Args:
            prompt: The input prompt
            response_embeddings: List of embeddings for different responses
            
        Returns:
            Dictionary with consistency metrics
        """
        if not response_embeddings:
            return {"error": "No response embeddings provided"}
        
        # Convert to numpy array
        embeddings_array = np.array(response_embeddings)
        
        # Compute variance metrics
        variance_metrics = self.compute_response_variance(embeddings_array)
        
        # Add prompt information
        result = {
            "prompt": prompt,
            "n_responses": len(response_embeddings),
            **variance_metrics
        }
        
        return result
    
    def batch_score_consistency(self, prompts: List[str], 
                               all_response_embeddings: List[List[np.ndarray]]) -> Dict:
        """
        Score consistency for multiple prompts.
        
        Args:
            prompts: List of prompts
            all_response_embeddings: List of response embedding lists for each prompt
            
        Returns:
            Dictionary with batch consistency results
        """
        results = []
        
        for prompt, response_embeddings in zip(prompts, all_response_embeddings):
            result = self.score_prompt_consistency(prompt, response_embeddings)
            results.append(result)
        
        # Compute batch statistics
        consistency_scores = [r.get("consistency_score", 0) for r in results if "consistency_score" in r]
        embedding_variances = [r.get("embedding_variance", 0) for r in results if "embedding_variance" in r]
        
        batch_stats = {
            "n_prompts": len(prompts),
            "mean_consistency": float(np.mean(consistency_scores)) if consistency_scores else 0.0,
            "std_consistency": float(np.std(consistency_scores)) if consistency_scores else 0.0,
            "mean_embedding_variance": float(np.mean(embedding_variances)) if embedding_variances else 0.0,
            "std_embedding_variance": float(np.std(embedding_variances)) if embedding_variances else 0.0
        }
        
        # Sort by consistency (lower = more likely hallucination)
        results.sort(key=lambda x: x.get("consistency_score", 0))
        
        return {
            "individual_results": results,
            "batch_statistics": batch_stats
        }
    
    def simulate_response_variance(self, base_embedding: np.ndarray, 
                                 noise_level: float = 0.1) -> List[np.ndarray]:
        """
        Simulate multiple responses by adding noise to base embedding.
        
        This is a simulation method for testing - in practice, you would
        generate actual responses from the model.
        
        Args:
            base_embedding: Base embedding to add noise to
            noise_level: Standard deviation of noise to add
            
        Returns:
            List of simulated response embeddings
        """
        response_embeddings = []
        
        for _ in range(self.n_responses):
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level, base_embedding.shape)
            noisy_embedding = base_embedding + noise
            
            # Normalize to unit sphere
            noisy_embedding = noisy_embedding / np.linalg.norm(noisy_embedding)
            response_embeddings.append(noisy_embedding)
        
        return response_embeddings
    
    def visualize_consistency_scores(self, results: List[Dict], 
                                   save_path: Optional[str] = None) -> None:
        """
        Create visualization of consistency scores.
        
        Args:
            results: List of consistency results
            save_path: Optional path to save the plot
        """
        if not results:
            print("No results to visualize")
            return
        
        # Extract data for plotting
        prompts = [r["prompt"][:30] + "..." if len(r["prompt"]) > 30 else r["prompt"] 
                  for r in results]
        consistency_scores = [r.get("consistency_score", 0) for r in results]
        embedding_variances = [r.get("embedding_variance", 0) for r in results]
        
        # Create subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot 1: Consistency scores
        ax1.bar(range(len(consistency_scores)), consistency_scores, alpha=0.7)
        ax1.set_title('Consistency Scores by Prompt', fontsize=14)
        ax1.set_xlabel('Prompt Index')
        ax1.set_ylabel('Consistency Score (Higher = More Consistent)')
        ax1.grid(True, alpha=0.3)
        
        # Add prompt labels on x-axis (rotated)
        ax1.set_xticks(range(len(prompts)))
        ax1.set_xticklabels(prompts, rotation=45, ha='right')
        
        # Plot 2: Embedding variances
        ax2.bar(range(len(embedding_variances)), embedding_variances, alpha=0.7, color='orange')
        ax2.set_title('Embedding Variance by Prompt', fontsize=14)
        ax2.set_xlabel('Prompt Index')
        ax2.set_ylabel('Embedding Variance (Higher = More Inconsistent)')
        ax2.grid(True, alpha=0.3)
        
        # Add prompt labels on x-axis (rotated)
        ax2.set_xticks(range(len(prompts)))
        ax2.set_xticklabels(prompts, rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def create_consistency_report(self, results: List[Dict]) -> str:
        """
        Create a text report of consistency analysis.
        
        Args:
            results: List of consistency results
            
        Returns:
            Formatted report string
        """
        if not results:
            return "No results to report"
        
        # Sort by consistency score
        sorted_results = sorted(results, key=lambda x: x.get("consistency_score", 0))
        
        report = []
        report.append("Consistency-Based Hallucination Detection Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        consistency_scores = [r.get("consistency_score", 0) for r in results]
        report.append(f"Total prompts analyzed: {len(results)}")
        report.append(f"Mean consistency score: {np.mean(consistency_scores):.3f}")
        report.append(f"Std consistency score: {np.std(consistency_scores):.3f}")
        report.append("")
        
        # Most inconsistent (likely hallucinations)
        report.append("Top 5 Most Inconsistent (Likely Hallucinations):")
        for i, result in enumerate(sorted_results[:5]):
            score = result.get("consistency_score", 0)
            prompt = result.get("prompt", "Unknown")[:60]
            report.append(f"  {i+1}. Score {score:.3f}: {prompt}...")
        report.append("")
        
        # Most consistent (likely reliable)
        report.append("Top 5 Most Consistent (Likely Reliable):")
        for i, result in enumerate(sorted_results[-5:]):
            score = result.get("consistency_score", 0)
            prompt = result.get("prompt", "Unknown")[:60]
            report.append(f"  {i+1}. Score {score:.3f}: {prompt}...")
        
        return "\n".join(report)


def simulate_consistency_analysis(embeddings: np.ndarray, prompts: List[str], 
                                noise_levels: List[float] = [0.05, 0.1, 0.2]) -> Dict:
    """
    Simulate consistency analysis with different noise levels.
    
    Args:
        embeddings: Base embeddings for prompts
        prompts: List of prompts
        noise_levels: List of noise levels to test
        
    Returns:
        Dictionary with simulation results
    """
    scorer = ConsistencyScorer(n_responses=5)
    all_results = {}
    
    for noise_level in noise_levels:
        print(f"Simulating with noise level: {noise_level}")
        
        batch_embeddings = []
        for embedding in embeddings:
            # Simulate multiple responses
            response_embeddings = scorer.simulate_response_variance(embedding, noise_level)
            batch_embeddings.append(response_embeddings)
        
        # Score consistency
        results = scorer.batch_score_consistency(prompts, batch_embeddings)
        all_results[f"noise_{noise_level}"] = results
    
    return all_results


def main():
    """Example usage of the consistency scorer."""
    print("Consistency-Based Hallucination Detection")
    print("=" * 50)
    
    # Load existing analysis
    try:
        with open('../data/llama_simple_analysis.json', 'r') as f:
            data = json.load(f)
        
        embeddings = np.array(data['embeddings'])
        prompts = data['prompts']
        print(f"Loaded {len(embeddings)} embeddings from LLaMA analysis")
    except FileNotFoundError:
        print("LLaMA analysis not found. Please run analysis first.")
        return
    
    # Simulate consistency analysis with different noise levels
    print("\nSimulating consistency analysis...")
    simulation_results = simulate_consistency_analysis(embeddings, prompts)
    
    # Analyze results
    for noise_level, results in simulation_results.items():
        print(f"\n--- Results for {noise_level} ---")
        batch_stats = results["batch_statistics"]
        print(f"Mean consistency: {batch_stats['mean_consistency']:.3f}")
        print(f"Mean embedding variance: {batch_stats['mean_embedding_variance']:.3f}")
    
    # Create detailed analysis for moderate noise
    print(f"\nDetailed analysis for noise level 0.1:")
    moderate_results = simulation_results["noise_0.1"]["individual_results"]
    
    # Create report
    scorer = ConsistencyScorer()
    report = scorer.create_consistency_report(moderate_results)
    print(report)
    
    # Save results
    output_data = {
        "simulation_results": simulation_results,
        "method": "consistency_based_scoring",
        "parameters": {
            "n_responses": 5,
            "noise_levels": [0.05, 0.1, 0.2]
        }
    }
    
    with open('../data/consistency_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to data/consistency_analysis.json")


if __name__ == "__main__":
    main()
