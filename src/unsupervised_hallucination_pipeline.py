"""
Unsupervised Hallucination Detection Pipeline

This module combines multiple unsupervised approaches for robust
hallucination detection without requiring human-curated labels.

Combined Approaches:
1. Density-based clustering (natural manifold discovery)
2. Consistency-based scoring (response variance)
3. Cross-model agreement (model disagreement)
4. Temporal consistency (embedding drift over time)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import os

# Import our custom modules
from unsupervised_manifold_detection import UnsupervisedManifoldDetector
from consistency_scoring import ConsistencyScorer
from cross_model_agreement import CrossModelAgreementDetector


class UnsupervisedHallucinationPipeline:
    """
    Unified pipeline for unsupervised hallucination detection.
    
    This pipeline combines multiple unsupervised approaches to provide
    robust hallucination detection without human-curated labels.
    """
    
    def __init__(self, 
                 manifold_params: Dict = None,
                 consistency_params: Dict = None,
                 agreement_params: Dict = None):
        """
        Initialize the unsupervised hallucination detection pipeline.
        
        Args:
            manifold_params: Parameters for manifold detection
            consistency_params: Parameters for consistency scoring
            agreement_params: Parameters for cross-model agreement
        """
        # Default parameters
        self.manifold_params = manifold_params or {
            "min_cluster_size": 3,
            "min_samples": 2
        }
        
        self.consistency_params = consistency_params or {
            "n_responses": 5,
            "temperature": 0.7
        }
        
        self.agreement_params = agreement_params or {}
        
        # Initialize detectors
        self.manifold_detector = UnsupervisedManifoldDetector(**self.manifold_params)
        self.consistency_scorer = ConsistencyScorer(**self.consistency_params)
        self.agreement_detector = None  # Will be initialized when models are loaded
        
        # Results storage
        self.results = {}
        self.combined_scores = {}
        
    def load_embeddings(self, embeddings: np.ndarray, prompts: List[str]) -> None:
        """
        Load embeddings and prompts for analysis.
        
        Args:
            embeddings: Array of embeddings (n_prompts, n_features)
            prompts: List of corresponding prompts
        """
        self.embeddings = embeddings
        self.prompts = prompts
        print(f"Loaded {len(embeddings)} embeddings and prompts")
    
    def load_multiple_models(self, model_data: Dict[str, Tuple[np.ndarray, List[str]]]) -> None:
        """
        Load embeddings from multiple models for agreement analysis.
        
        Args:
            model_data: Dictionary mapping model names to (embeddings, prompts) tuples
        """
        if len(model_data) < 2:
            print("Warning: Need at least 2 models for agreement analysis")
            return
        
        model_names = list(model_data.keys())
        self.agreement_detector = CrossModelAgreementDetector(model_names)
        
        for model_name, (embeddings, prompts) in model_data.items():
            self.agreement_detector.load_model_embeddings(model_name, embeddings)
        
        print(f"Loaded {len(model_data)} models for agreement analysis")
    
    def run_manifold_analysis(self) -> Dict:
        """
        Run density-based manifold analysis.
        
        Returns:
            Dictionary with manifold analysis results
        """
        print("Running density-based manifold analysis...")
        
        # Fit manifold detector
        stats = self.manifold_detector.fit(self.embeddings, self.prompts)
        
        # Score all prompts
        scoring_results = self.manifold_detector.score_prompts(self.prompts, self.embeddings)
        
        # Create visualization
        self.manifold_detector.visualize_manifolds('../data/unsupervised_manifolds.png')
        
        results = {
            "clustering_stats": stats,
            "scoring_results": scoring_results,
            "method": "density_based_manifolds"
        }
        
        self.results["manifold_analysis"] = results
        return results
    
    def run_consistency_analysis(self, noise_level: float = 0.1) -> Dict:
        """
        Run consistency-based analysis.
        
        Args:
            noise_level: Noise level for response simulation
            
        Returns:
            Dictionary with consistency analysis results
        """
        print("Running consistency-based analysis...")
        
        # Simulate multiple responses for each prompt
        batch_embeddings = []
        for embedding in self.embeddings:
            response_embeddings = self.consistency_scorer.simulate_response_variance(
                embedding, noise_level
            )
            batch_embeddings.append(response_embeddings)
        
        # Score consistency
        results = self.consistency_scorer.batch_score_consistency(
            self.prompts, batch_embeddings
        )
        
        # Create visualization
        self.consistency_scorer.visualize_consistency_scores(
            results["individual_results"], 
            '../data/consistency_scores.png'
        )
        
        self.results["consistency_analysis"] = results
        return results
    
    def run_agreement_analysis(self) -> Dict:
        """
        Run cross-model agreement analysis.
        
        Returns:
            Dictionary with agreement analysis results
        """
        if self.agreement_detector is None:
            print("Warning: No models loaded for agreement analysis")
            return {}
        
        print("Running cross-model agreement analysis...")
        
        # Compute global agreement
        global_results = self.agreement_detector.compute_global_agreement()
        
        # Compute prompt-level agreement
        prompt_scores = self.agreement_detector.compute_prompt_agreement_scores(self.prompts)
        
        # Create visualizations
        self.agreement_detector.visualize_agreement_matrix('../data/agreement_matrix.png')
        self.agreement_detector.visualize_prompt_agreement(
            prompt_scores, '../data/prompt_agreement.png'
        )
        
        results = {
            "global_agreement": global_results,
            "prompt_agreement_scores": prompt_scores
        }
        
        self.results["agreement_analysis"] = results
        return results
    
    def combine_scores(self, weights: Dict[str, float] = None) -> Dict:
        """
        Combine scores from all methods into a unified hallucination score.
        
        Args:
            weights: Dictionary of method weights (default: equal weights)
            
        Returns:
            Dictionary with combined scores and rankings
        """
        if not self.results:
            print("Warning: No analysis results available")
            return {}
        
        # Default equal weights
        if weights is None:
            weights = {
                "manifold": 0.4,
                "consistency": 0.3,
                "agreement": 0.3
            }
        
        print("Combining scores from all methods...")
        
        combined_scores = []
        
        for i, prompt in enumerate(self.prompts):
            scores = {"prompt": prompt}
            
            # Manifold distance score (higher = more likely hallucination)
            if "manifold_analysis" in self.results:
                manifold_scores = self.results["manifold_analysis"]["scoring_results"]["scores"]
                for score_data in manifold_scores:
                    if score_data["prompt"] == prompt:
                        scores["manifold_distance"] = score_data["manifold_distance"]
                        break
            
            # Consistency score (lower = more likely hallucination)
            if "consistency_analysis" in self.results:
                consistency_scores = self.results["consistency_analysis"]["individual_results"]
                for score_data in consistency_scores:
                    if score_data["prompt"] == prompt:
                        scores["consistency_score"] = score_data["consistency_score"]
                        break
            
            # Agreement score (lower = more likely hallucination)
            if "agreement_analysis" in self.results:
                agreement_scores = self.results["agreement_analysis"]["prompt_agreement_scores"]
                for score_data in agreement_scores:
                    if score_data["prompt"] == prompt:
                        scores["agreement_score"] = score_data["agreement_score"]
                        break
            
            # Normalize and combine scores
            normalized_scores = {}
            combined_score = 0.0
            total_weight = 0.0
            
            # Manifold score (higher distance = higher hallucination likelihood)
            if "manifold_distance" in scores:
                normalized_scores["manifold"] = scores["manifold_distance"]
                combined_score += weights["manifold"] * scores["manifold_distance"]
                total_weight += weights["manifold"]
            
            # Consistency score (lower consistency = higher hallucination likelihood)
            if "consistency_score" in scores:
                normalized_scores["consistency"] = 1.0 - scores["consistency_score"]  # Invert
                combined_score += weights["consistency"] * (1.0 - scores["consistency_score"])
                total_weight += weights["consistency"]
            
            # Agreement score (lower agreement = higher hallucination likelihood)
            if "agreement_score" in scores:
                normalized_scores["agreement"] = 1.0 - scores["agreement_score"]  # Invert
                combined_score += weights["agreement"] * (1.0 - scores["agreement_score"])
                total_weight += weights["agreement"]
            
            # Normalize by total weight
            if total_weight > 0:
                combined_score = combined_score / total_weight
            
            scores["normalized_scores"] = normalized_scores
            scores["combined_score"] = combined_score
            combined_scores.append(scores)
        
        # Sort by combined score (higher = more likely hallucination)
        combined_scores.sort(key=lambda x: x["combined_score"], reverse=True)
        
        # Add rankings
        for i, score in enumerate(combined_scores):
            score["rank"] = i + 1
        
        # Compute statistics
        combined_values = [s["combined_score"] for s in combined_scores]
        stats = {
            "mean_combined_score": float(np.mean(combined_values)),
            "std_combined_score": float(np.std(combined_values)),
            "min_combined_score": float(np.min(combined_values)),
            "max_combined_score": float(np.max(combined_values)),
            "weights_used": weights
        }
        
        self.combined_scores = {
            "individual_scores": combined_scores,
            "statistics": stats
        }
        
        return self.combined_scores
    
    def create_comprehensive_report(self) -> str:
        """
        Create a comprehensive report of all analyses.
        
        Returns:
            Formatted report string
        """
        report = []
        report.append("Unsupervised Hallucination Detection - Comprehensive Report")
        report.append("=" * 70)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Summary of methods used
        report.append("Methods Applied:")
        if "manifold_analysis" in self.results:
            report.append("  ✓ Density-based manifold detection")
        if "consistency_analysis" in self.results:
            report.append("  ✓ Consistency-based scoring")
        if "agreement_analysis" in self.results:
            report.append("  ✓ Cross-model agreement detection")
        report.append("")
        
        # Individual method results
        if "manifold_analysis" in self.results:
            report.append("Density-Based Manifold Analysis:")
            stats = self.results["manifold_analysis"]["clustering_stats"]
            report.append(f"  Clusters found: {stats['n_clusters']}")
            report.append(f"  Noise points: {stats['n_noise_points']} ({stats['noise_percentage']:.1f}%)")
            report.append(f"  Silhouette score: {stats['silhouette_score']:.3f}")
            report.append("")
        
        if "consistency_analysis" in self.results:
            report.append("Consistency-Based Analysis:")
            stats = self.results["consistency_analysis"]["batch_statistics"]
            report.append(f"  Mean consistency: {stats['mean_consistency']:.3f}")
            report.append(f"  Mean embedding variance: {stats['mean_embedding_variance']:.3f}")
            report.append("")
        
        if "agreement_analysis" in self.results:
            report.append("Cross-Model Agreement Analysis:")
            stats = self.results["agreement_analysis"]["global_agreement"]["global_statistics"]
            report.append(f"  Mean agreement: {stats['mean_agreement']:.3f}")
            report.append(f"  Model pairs: {stats['n_model_pairs']}")
            report.append("")
        
        # Combined results
        if self.combined_scores:
            report.append("Combined Hallucination Scores:")
            stats = self.combined_scores["statistics"]
            report.append(f"  Mean combined score: {stats['mean_combined_score']:.3f}")
            report.append(f"  Score range: {stats['min_combined_score']:.3f} - {stats['max_combined_score']:.3f}")
            report.append("")
            
            # Top 5 most likely hallucinations
            report.append("Top 5 Most Likely Hallucinations:")
            for i, score in enumerate(self.combined_scores["individual_scores"][:5]):
                combined = score["combined_score"]
                prompt = score["prompt"][:60]
                report.append(f"  {i+1}. Score {combined:.3f}: {prompt}...")
            report.append("")
            
            # Top 5 least likely hallucinations
            report.append("Top 5 Least Likely Hallucinations:")
            for i, score in enumerate(self.combined_scores["individual_scores"][-5:]):
                combined = score["combined_score"]
                prompt = score["prompt"][:60]
                report.append(f"  {i+1}. Score {combined:.3f}: {prompt}...")
        
        return "\n".join(report)
    
    def save_results(self, output_path: str = "../data/unsupervised_hallucination_analysis.json") -> None:
        """
        Save all results to a JSON file.
        
        Args:
            output_path: Path to save the results
        """
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "method": "unsupervised_hallucination_detection",
            "parameters": {
                "manifold_params": self.manifold_params,
                "consistency_params": self.consistency_params,
                "agreement_params": self.agreement_params
            },
            "individual_analyses": self.results,
            "combined_scores": self.combined_scores
        }
        
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_path}")
    
    def run_full_pipeline(self, 
                         embeddings: np.ndarray,
                         prompts: List[str],
                         model_data: Dict[str, Tuple[np.ndarray, List[str]]] = None,
                         weights: Dict[str, float] = None) -> Dict:
        """
        Run the complete unsupervised hallucination detection pipeline.
        
        Args:
            embeddings: Primary embeddings for analysis
            prompts: List of prompts
            model_data: Optional multi-model data for agreement analysis
            weights: Optional weights for combining scores
            
        Returns:
            Dictionary with all results
        """
        print("Starting Unsupervised Hallucination Detection Pipeline")
        print("=" * 60)
        
        # Load primary data
        self.load_embeddings(embeddings, prompts)
        
        # Load multi-model data if available
        if model_data:
            self.load_multiple_models(model_data)
        
        # Run all analyses
        manifold_results = self.run_manifold_analysis()
        consistency_results = self.run_consistency_analysis()
        
        if self.agreement_detector:
            agreement_results = self.run_agreement_analysis()
        else:
            print("Skipping agreement analysis (insufficient models)")
        
        # Combine scores
        combined_results = self.combine_scores(weights)
        
        # Create comprehensive report
        report = self.create_comprehensive_report()
        print("\n" + report)
        
        # Save results
        self.save_results()
        
        return {
            "manifold_analysis": manifold_results,
            "consistency_analysis": consistency_results,
            "agreement_analysis": self.results.get("agreement_analysis", {}),
            "combined_scores": combined_results,
            "report": report
        }


def main():
    """Example usage of the unsupervised hallucination detection pipeline."""
    print("Unsupervised Hallucination Detection Pipeline")
    print("=" * 50)
    
    # Load primary analysis data
    try:
        with open('../data/llama_simple_analysis.json', 'r') as f:
            data = json.load(f)
        
        embeddings = np.array(data['embeddings'])
        prompts = data['prompts']
        print(f"Loaded {len(embeddings)} embeddings from LLaMA analysis")
    except FileNotFoundError:
        print("LLaMA analysis not found. Please run analysis first.")
        return
    
    # Initialize pipeline
    pipeline = UnsupervisedHallucinationPipeline()
    
    # Run full pipeline
    results = pipeline.run_full_pipeline(embeddings, prompts)
    
    print("\nPipeline execution complete!")
    print("Check the data/ directory for visualizations and results.")


if __name__ == "__main__":
    main()
