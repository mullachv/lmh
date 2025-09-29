#!/usr/bin/env python3
"""
Model Embedding Dimensions Reference
Quick utility to access model dimensions and performance metrics
"""

import json
import os

def load_model_dimensions():
    """Load model dimensions from the reference file"""
    file_path = '../data/model_embedding_dimensions.json'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found")
        return None
    
    with open(file_path, 'r') as f:
        return json.load(f)

def get_model_dimension(model_name):
    """Get embedding dimension for a specific model"""
    data = load_model_dimensions()
    if not data:
        return None
    
    return data['model_embedding_dimensions'].get(model_name, "Model not found")

def list_models_by_dimension(dimension):
    """List all models with a specific embedding dimension"""
    data = load_model_dimensions()
    if not data:
        return None
    
    return data['dimension_groups'].get(f"{dimension}D", [])

def get_performance_ranking():
    """Get performance rankings for all models"""
    data = load_model_dimensions()
    if not data:
        return None
    
    return data['performance_ranking']

def print_model_summary():
    """Print a comprehensive summary of all models"""
    data = load_model_dimensions()
    if not data:
        return
    
    print("=" * 80)
    print("MODEL EMBEDDING DIMENSIONS REFERENCE")
    print("=" * 80)
    
    # Dimensions by model
    print("\n📊 EMBEDDING DIMENSIONS:")
    print("-" * 50)
    for model, dim in data['model_embedding_dimensions'].items():
        print(f"{model:<35} {dim:>4}D")
    
    # Performance rankings
    print("\n🏆 CLUSTERING PERFORMANCE (Silhouette Score):")
    print("-" * 50)
    for model, score in sorted(data['performance_ranking']['clustering_silhouette'].items(), 
                              key=lambda x: x[1], reverse=True):
        print(f"{model:<35} {score:>6.3f}")
    
    print("\n📏 AVERAGE COSINE DISTANCE:")
    print("-" * 50)
    for model, distance in sorted(data['performance_ranking']['average_cosine_distance'].items(), 
                                 key=lambda x: x[1]):
        print(f"{model:<35} {distance:>6.3f}")
    
    # Dimension groups
    print("\n📋 MODELS BY DIMENSION:")
    print("-" * 50)
    for dim_group, models in data['dimension_groups'].items():
        print(f"\n{dim_group}:")
        for model in models:
            print(f"  • {model}")
    
    # Normalization info
    print("\n🔍 EMBEDDING NORMALIZATION:")
    print("-" * 50)
    if 'embedding_normalization' in data:
        normalized = data['embedding_normalization']['normalized_to_unit_sphere']
        not_normalized = data['embedding_normalization']['not_normalized']
        
        print("✅ Normalized to Unit Sphere:")
        for model, stats in normalized.items():
            print(f"  • {model} (norm: {stats['mean_norm']:.1f} ± {stats['std_norm']:.1f})")
        
        print("\n❌ Not Normalized:")
        for model, stats in not_normalized.items():
            print(f"  • {model} (norm: {stats['mean_norm']:.1f} ± {stats['std_norm']:.1f})")
    
    # Analysis notes
    print("\n💡 KEY INSIGHTS:")
    print("-" * 50)
    for key, value in data['metadata']['analysis_notes'].items():
        print(f"  {key.replace('_', ' ').title()}: {value}")
    
    print(f"\n📅 Last updated: {data['metadata']['last_updated']}")
    print(f"📊 Total models: {data['metadata']['total_models']}")
    print(f"🔢 Unique dimensions: {data['metadata']['unique_dimensions']}")

def main():
    """Main function for command-line usage"""
    import sys
    
    if len(sys.argv) == 1:
        # No arguments - print full summary
        print_model_summary()
    elif len(sys.argv) == 2:
        # One argument - could be model name or dimension
        arg = sys.argv[1]
        
        # Check if it's a dimension number
        if arg.isdigit():
            dimension = int(arg)
            models = list_models_by_dimension(dimension)
            if models:
                print(f"Models with {dimension}D embeddings:")
                for model in models:
                    print(f"  • {model}")
            else:
                print(f"No models found with {dimension}D embeddings")
        else:
            # Treat as model name
            dim = get_model_dimension(arg)
            if dim:
                print(f"{arg}: {dim}D")
            else:
                print(f"Model '{arg}' not found")
    else:
        print("Usage:")
        print("  python model_dimensions.py                    # Print full summary")
        print("  python model_dimensions.py <model_name>       # Get dimension for model")
        print("  python model_dimensions.py <dimension>        # List models with dimension")

if __name__ == "__main__":
    main()
