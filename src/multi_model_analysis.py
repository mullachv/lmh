#!/usr/bin/env python3
"""
Multi-Model Embedding Analysis
Tests multiple embedding models to compare manifold structures
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_distances
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import umap
from tqdm import tqdm
import os

class MultiModelAnalyzer:
    def __init__(self):
        self.models = {
            # Sentence Transformers (fast, good quality)
            'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2'),
            'all-mpnet-base-v2': SentenceTransformer('all-mpnet-base-v2'),
            'all-distilroberta-v1': SentenceTransformer('all-distilroberta-v1'),
            
            # BERT variants
            'bert-base-uncased': None,  # Will load with transformers
            'roberta-base': None,       # Will load with transformers
            'distilbert-base-uncased': None,  # Will load with transformers
            
            # LLaMA alternatives (if accessible)
            'llama-2-7b': None,        # Will load with transformers if accessible
        }
        
        self.results = {}
    
    def load_transformers_model(self, model_name):
        """Load a model using transformers library"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            return tokenizer, model
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            return None, None
    
    def extract_embeddings_transformers(self, texts, tokenizer, model, device='cpu'):
        """Extract embeddings using transformers library"""
        embeddings = []
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            for text in tqdm(texts, desc=f"Extracting embeddings with {model.config.name_or_path}"):
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                outputs = model(**inputs)
                attention_mask = inputs['attention_mask']
                last_hidden_states = outputs.last_hidden_state
                
                # Mean pooling
                masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
                mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
                
                embeddings.append(mean_embedding.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def analyze_model(self, model_name, texts):
        """Analyze a single model"""
        print(f"\n{'='*60}")
        print(f"ANALYZING MODEL: {model_name}")
        print(f"{'='*60}")
        
        try:
            if model_name in ['all-MiniLM-L6-v2', 'all-mpnet-base-v2', 'all-distilroberta-v1']:
                # Use sentence transformers
                model = self.models[model_name]
                embeddings = model.encode(texts, show_progress_bar=True)
                print(f"Embedding dimension: {embeddings.shape[1]}")
                
            else:
                # Use transformers library
                tokenizer, model = self.load_transformers_model(model_name)
                if tokenizer is None or model is None:
                    print(f"Skipping {model_name} - failed to load")
                    return None
                
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                embeddings = self.extract_embeddings_transformers(texts, tokenizer, model, device)
                print(f"Embedding dimension: {embeddings.shape[1]}")
            
            # Basic statistics
            print(f"Embedding statistics: mean={np.mean(embeddings):.4f}, std={np.std(embeddings):.4f}")
            
            # Clustering analysis
            best_silhouette = -1
            best_k = 2
            
            for k in range(2, min(6, len(texts)//2)):
                kmeans = KMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(embeddings)
                
                if len(set(labels)) > 1:
                    silhouette = silhouette_score(embeddings, labels)
                    print(f"K-means (k={k}): Silhouette = {silhouette:.3f}")
                    
                    if silhouette > best_silhouette:
                        best_silhouette = silhouette
                        best_k = k
            
            # UMAP visualization
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(texts)-1))
            embedding_2d = reducer.fit_transform(embeddings)
            
            # Cosine distances
            cosine_dist = cosine_distances(embeddings)
            avg_distance = np.mean(cosine_dist[np.triu_indices_from(cosine_dist, k=1)])
            
            result = {
                'model_name': model_name,
                'embeddings': embeddings.tolist(),
                'embedding_2d': embedding_2d.tolist(),
                'cosine_distances': cosine_dist.tolist(),
                'best_silhouette': best_silhouette,
                'best_k': best_k,
                'avg_cosine_distance': float(avg_distance),
                'embedding_dim': embeddings.shape[1],
                'num_prompts': len(texts)
            }
            
            print(f"Best clustering: k={best_k}, Silhouette={best_silhouette:.3f}")
            print(f"Average cosine distance: {avg_distance:.3f}")
            
            return result
            
        except Exception as e:
            print(f"Error analyzing {model_name}: {e}")
            return None
    
    def compare_models(self, texts):
        """Compare all models"""
        print(f"Comparing {len(self.models)} models on {len(texts)} prompts")
        
        for model_name in self.models.keys():
            result = self.analyze_model(model_name, texts)
            if result:
                self.results[model_name] = result
        
        # Summary comparison
        print(f"\n{'='*60}")
        print("MODEL COMPARISON SUMMARY")
        print(f"{'='*60}")
        
        comparison_data = []
        for model_name, result in self.results.items():
            comparison_data.append({
                'model': model_name,
                'silhouette': result['best_silhouette'],
                'avg_distance': result['avg_cosine_distance'],
                'dimension': result['embedding_dim']
            })
        
        # Sort by silhouette score
        comparison_data.sort(key=lambda x: x['silhouette'], reverse=True)
        
        print(f"{'Model':<25} {'Silhouette':<12} {'Avg Distance':<12} {'Dimension':<10}")
        print("-" * 60)
        for data in comparison_data:
            print(f"{data['model']:<25} {data['silhouette']:<12.3f} {data['avg_distance']:<12.3f} {data['dimension']:<10}")
        
        return comparison_data
    
    def create_comparison_plots(self, texts):
        """Create comparison plots for all models"""
        n_models = len(self.results)
        if n_models == 0:
            return
        
        fig, axes = plt.subplots(2, (n_models + 1) // 2, figsize=(5 * ((n_models + 1) // 2), 10))
        if n_models == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, (model_name, result) in enumerate(self.results.items()):
            if i >= len(axes):
                break
                
            embedding_2d = np.array(result['embedding_2d'])
            
            # Plot UMAP projection
            scatter = axes[i].scatter(embedding_2d[:, 0], embedding_2d[:, 1], 
                                    c=range(len(texts)), cmap='tab10', s=20, alpha=0.7)
            axes[i].set_title(f'{model_name}\nSilhouette: {result["best_silhouette"]:.3f}')
            axes[i].set_xlabel('UMAP 1')
            axes[i].set_ylabel('UMAP 2')
        
        # Hide unused subplots
        for i in range(len(self.results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('../data/multi_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_results(self):
        """Save all results"""
        with open('../data/multi_model_analysis.json', 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\nResults saved to ../data/multi_model_analysis.json")

def main():
    # Load diverse prompts
    with open('../data/diverse_prompts.json', 'r') as f:
        prompts = json.load(f)
    
    print(f"Loaded {len(prompts)} diverse prompts")
    
    # Initialize analyzer
    analyzer = MultiModelAnalyzer()
    
    # Compare models
    comparison_data = analyzer.compare_models(prompts)
    
    # Create comparison plots
    analyzer.create_comparison_plots(prompts)
    
    # Save results
    analyzer.save_results()
    
    print(f"\nAnalysis complete! Check ../data/ for results and visualizations.")

if __name__ == "__main__":
    main()











