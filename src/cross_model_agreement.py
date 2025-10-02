"""
Cross-Model Agreement Detection for Hallucination Detection

This module implements hallucination detection by measuring agreement
between different embedding models on the same prompts. High disagreement
suggests potential hallucination.

Key Methods:
1. Embed same prompts with multiple models
2. Compute inter-model embedding distances
3. High disagreement = potential hallucination
4. Leverages model diversity instead of human labels
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import entropy
import seaborn as sns
from itertools import combinations


class CrossModelAgreementDetector:
    """
    Hallucination detection based on cross-model agreement.
    
    This approach measures how much different embedding models agree on
    the same prompts. High disagreement suggests uncertainty or hallucination.
    """
    
    def __init__(self, model_names: List[str]):
        """
        Initialize the cross-model agreement detector.
        
        Args:
            model_names: List of model names to compare
        """
        self.model_names = model_names
        self.n_models = len(model_names)
        self.embeddings_by_model = {}
        self.agreement_scores = {}
        
    def load_model_embeddings(self, model_name: str, embeddings: np.ndarray) -> None:
        """
        Load embeddings for a specific model.
        
        Args:
            model_name: Name of the model
            embeddings: Array of embeddings (n_prompts, n_features)
        """
        self.embeddings_by_model[model_name] = embeddings
        print(f"Loaded {len(embeddings)} embeddings for {model_name}")
    
    def compute_pairwise_agreement(self, model1_name: str, model2_name: str) -> Dict:
        """
        Compute agreement between two models.
        
        Args:
            model1_name: Name of first model
            model2_name: Name of second model
            
        Returns:
            Dictionary with agreement metrics
        """
        if model1_name not in self.embeddings_by_model:
            raise ValueError(f"Model {model1_name} not loaded")
        if model2_name not in self.embeddings_by_model:
            raise ValueError(f"Model {model2_name} not loaded")
        
        embeddings1 = self.embeddings_by_model[model1_name]
        embeddings2 = self.embeddings_by_model[model2_name]
        
        if len(embeddings1) != len(embeddings2):
            raise ValueError("Models must have same number of embeddings")
        
        # Compute pairwise cosine distances
        distances = cosine_distances(embeddings1, embeddings2)
        
        # Get diagonal (same prompt, different models)
        diagonal_distances = np.diag(distances)
        
        # Compute agreement metrics
        mean_distance = np.mean(diagonal_distances)
        std_distance = np.std(diagonal_distances)
        max_distance = np.max(diagonal_distances)
        min_distance = np.min(diagonal_distances)
        
        # Agreement score (lower distance = higher agreement)
        agreement_score = 1.0 - mean_distance
        
        return {
            "model1": model1_name,
            "model2": model2_name,
            "mean_distance": float(mean_distance),
            "std_distance": float(std_distance),
            "max_distance": float(max_distance),
            "min_distance": float(min_distance),
            "agreement_score": float(agreement_score),
            "n_prompts": len(embeddings1)
        }
    
    def compute_global_agreement(self) -> Dict:
        """
        Compute agreement across all model pairs.
        
        Returns:
            Dictionary with global agreement metrics
        """
        if len(self.embeddings_by_model) < 2:
            raise ValueError("Need at least 2 models for agreement analysis")
        
        # Compute pairwise agreements
        pairwise_agreements = []
        model_pairs = []
        
        for model1, model2 in combinations(self.model_names, 2):
            if model1 in self.embeddings_by_model and model2 in self.embeddings_by_model:
                agreement = self.compute_pairwise_agreement(model1, model2)
                pairwise_agreements.append(agreement)
                model_pairs.append(f"{model1}_vs_{model2}")
        
        # Compute global statistics
        agreement_scores = [a["agreement_score"] for a in pairwise_agreements]
        mean_distances = [a["mean_distance"] for a in pairwise_agreements]
        
        global_stats = {
            "n_model_pairs": len(pairwise_agreements),
            "mean_agreement": float(np.mean(agreement_scores)),
            "std_agreement": float(np.std(agreement_scores)),
            "min_agreement": float(np.min(agreement_scores)),
            "max_agreement": float(np.max(agreement_scores)),
            "mean_distance": float(np.mean(mean_distances)),
            "std_distance": float(np.std(mean_distances))
        }
        
        return {
            "global_statistics": global_stats,
            "pairwise_agreements": pairwise_agreements,
            "model_pairs": model_pairs
        }
    
    def compute_prompt_agreement_scores(self, prompts: List[str]) -> List[Dict]:
        """
        Compute agreement scores for individual prompts.
        
        Args:
            prompts: List of prompts to score
            
        Returns:
            List of dictionaries with prompt agreement scores
        """
        if len(self.embeddings_by_model) < 2:
            raise ValueError("Need at least 2 models for agreement analysis")
        
        prompt_scores = []
        
        for i, prompt in enumerate(prompts):
            # Get embeddings for this prompt from all models
            prompt_embeddings = {}
            for model_name in self.model_names:
                if model_name in self.embeddings_by_model:
                    prompt_embeddings[model_name] = self.embeddings_by_model[model_name][i]
            
            # Compute pairwise distances for this prompt
            model_names = list(prompt_embeddings.keys())
            distances = []
            
            for model1, model2 in combinations(model_names, 2):
                emb1 = prompt_embeddings[model1]
                emb2 = prompt_embeddings[model2]
                distance = cosine_distances([emb1], [emb2])[0, 0]
                distances.append(distance)
            
            # Compute agreement metrics for this prompt
            mean_distance = np.mean(distances) if distances else 0.0
            std_distance = np.std(distances) if distances else 0.0
            max_distance = np.max(distances) if distances else 0.0
            agreement_score = 1.0 - mean_distance
            
            prompt_scores.append({
                "prompt": prompt,
                "agreement_score": float(agreement_score),
                "mean_distance": float(mean_distance),
                "std_distance": float(std_distance),
                "max_distance": float(max_distance),
                "n_model_pairs": len(distances)
            })
        
        # Sort by agreement score (lower = more disagreement)
        prompt_scores.sort(key=lambda x: x["agreement_score"])
        
        # Add rankings
        for i, score in enumerate(prompt_scores):
            score["rank"] = i + 1
        
        return prompt_scores
    
    def visualize_agreement_matrix(self, save_path: Optional[str] = None) -> None:
        """
        Create visualization of agreement matrix between models.
        
        Args:
            save_path: Optional path to save the plot
        """
        if len(self.embeddings_by_model) < 2:
            print("Need at least 2 models for visualization")
            return
        
        # Compute agreement matrix
        n_models = len(self.model_names)
        agreement_matrix = np.zeros((n_models, n_models))
        
        for i, model1 in enumerate(self.model_names):
            for j, model2 in enumerate(self.model_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0  # Perfect self-agreement
                elif model1 in self.embeddings_by_model and model2 in self.embeddings_by_model:
                    agreement = self.compute_pairwise_agreement(model1, model2)
                    agreement_matrix[i, j] = agreement["agreement_score"]
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(agreement_matrix, 
                    xticklabels=self.model_names,
                    yticklabels=self.model_names,
                    annot=True, 
                    fmt='.3f',
                    cmap='RdYlBu_r',
                    cbar_kws={'label': 'Agreement Score'})
        
        plt.title('Cross-Model Agreement Matrix', fontsize=14)
        plt.xlabel('Models')
        plt.ylabel('Models')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Agreement matrix saved to {save_path}")
        
        plt.show()
    
    def visualize_prompt_agreement(self, prompt_scores: List[Dict], 
                                   save_path: Optional[str] = None) -> None:
        """
        Create visualization of agreement scores by prompt.
        
        Args:
            prompt_scores: List of prompt agreement scores
            save_path: Optional path to save the plot
        """
        if not prompt_scores:
            print("No prompt scores to visualize")
            return
        
        # Extract data
        prompts = [p["prompt"][:30] + "..." if len(p["prompt"]) > 30 else p["prompt"] 
                  for p in prompt_scores]
        agreement_scores = [p["agreement_score"] for p in prompt_scores]
        
        # Create plot
        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(agreement_scores)), agreement_scores, alpha=0.7)
        
        # Color bars by agreement level
        for i, bar in enumerate(bars):
            if agreement_scores[i] > 0.8:
                bar.set_color('green')  # High agreement
            elif agreement_scores[i] > 0.6:
                bar.set_color('orange')  # Medium agreement
            else:
                bar.set_color('red')  # Low agreement
        
        plt.title('Cross-Model Agreement by Prompt', fontsize=14)
        plt.xlabel('Prompt Index')
        plt.ylabel('Agreement Score (Higher = More Agreement)')
        plt.grid(True, alpha=0.3)
        
        # Add prompt labels
        plt.xticks(range(len(prompts)), prompts, rotation=45, ha='right')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='High Agreement (>0.8)'),
            Patch(facecolor='orange', label='Medium Agreement (0.6-0.8)'),
            Patch(facecolor='red', label='Low Agreement (<0.6)')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prompt agreement visualization saved to {save_path}")
        
        plt.show()
    
    def create_agreement_report(self, prompt_scores: List[Dict]) -> str:
        """
        Create a text report of agreement analysis.
        
        Args:
            prompt_scores: List of prompt agreement scores
            
        Returns:
            Formatted report string
        """
        if not prompt_scores:
            return "No agreement scores to report"
        
        report = []
        report.append("Cross-Model Agreement Analysis Report")
        report.append("=" * 50)
        report.append("")
        
        # Summary statistics
        agreement_scores = [p["agreement_score"] for p in prompt_scores]
        report.append(f"Total prompts analyzed: {len(prompt_scores)}")
        report.append(f"Mean agreement score: {np.mean(agreement_scores):.3f}")
        report.append(f"Std agreement score: {np.std(agreement_scores):.3f}")
        report.append("")
        
        # Most disagreed (likely hallucinations)
        report.append("Top 5 Most Disagreed (Likely Hallucinations):")
        for i, score in enumerate(prompt_scores[:5]):
            agreement = score["agreement_score"]
            prompt = score["prompt"][:60]
            report.append(f"  {i+1}. Agreement {agreement:.3f}: {prompt}...")
        report.append("")
        
        # Most agreed (likely reliable)
        report.append("Top 5 Most Agreed (Likely Reliable):")
        for i, score in enumerate(prompt_scores[-5:]):
            agreement = score["agreement_score"]
            prompt = score["prompt"][:60]
            report.append(f"  {i+1}. Agreement {agreement:.3f}: {prompt}...")
        
        return "\n".join(report)


def load_multiple_model_analyses(model_files: Dict[str, str]) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Load embeddings from multiple model analysis files.
    
    Args:
        model_files: Dictionary mapping model names to file paths
        
    Returns:
        Dictionary mapping model names to (embeddings, prompts) tuples
    """
    results = {}
    
    for model_name, file_path in model_files.items():
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            embeddings = np.array(data['embeddings'])
            prompts = data['prompts']
            results[model_name] = (embeddings, prompts)
            print(f"Loaded {model_name}: {len(embeddings)} embeddings")
            
        except FileNotFoundError:
            print(f"Warning: Could not load {model_name} from {file_path}")
        except KeyError:
            print(f"Warning: Invalid format in {file_path}")
    
    return results


def main():
    """Example usage of the cross-model agreement detector."""
    print("Cross-Model Agreement Detection for Hallucination Detection")
    print("=" * 60)
    
    # Define model files to compare
    model_files = {
        "LLaMA-2-7b": "../data/llama_simple_analysis.json",
        "RoBERTa-base": "../data/multi_model_analysis.json",  # This would need to be extracted
        # Add more models as available
    }
    
    # Load model analyses
    print("Loading model analyses...")
    model_data = load_multiple_model_analyses(model_files)
    
    if len(model_data) < 2:
        print("Need at least 2 models for agreement analysis")
        print("Available models:", list(model_data.keys()))
        return
    
    # Initialize detector
    model_names = list(model_data.keys())
    detector = CrossModelAgreementDetector(model_names)
    
    # Load embeddings for each model
    for model_name, (embeddings, prompts) in model_data.items():
        detector.load_model_embeddings(model_name, embeddings)
    
    # Compute global agreement
    print("\nComputing global agreement...")
    global_results = detector.compute_global_agreement()
    
    print(f"Global Agreement Statistics:")
    stats = global_results["global_statistics"]
    print(f"  Model pairs: {stats['n_model_pairs']}")
    print(f"  Mean agreement: {stats['mean_agreement']:.3f}")
    print(f"  Std agreement: {stats['std_agreement']:.3f}")
    print(f"  Agreement range: {stats['min_agreement']:.3f} - {stats['max_agreement']:.3f}")
    
    # Show pairwise agreements
    print(f"\nPairwise Agreements:")
    for agreement in global_results["pairwise_agreements"]:
        print(f"  {agreement['model1']} vs {agreement['model2']}: {agreement['agreement_score']:.3f}")
    
    # Compute prompt-level agreement
    print(f"\nComputing prompt-level agreement...")
    prompts = model_data[model_names[0]][1]  # Use prompts from first model
    prompt_scores = detector.compute_prompt_agreement_scores(prompts)
    
    # Create report
    report = detector.create_agreement_report(prompt_scores)
    print(report)
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    detector.visualize_agreement_matrix('../data/cross_model_agreement_matrix.png')
    detector.visualize_prompt_agreement(prompt_scores, '../data/prompt_agreement_scores.png')
    
    # Save results
    output_data = {
        "global_agreement": global_results,
        "prompt_agreement_scores": prompt_scores,
        "method": "cross_model_agreement",
        "models_compared": model_names
    }
    
    with open('../data/cross_model_agreement_analysis.json', 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nAnalysis complete! Results saved to data/cross_model_agreement_analysis.json")


if __name__ == "__main__":
    main()
