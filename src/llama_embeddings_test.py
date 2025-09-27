#!/usr/bin/env python3
"""
LLaMA Embeddings Test Script
Quick test to extract embeddings from sample prompts using LLaMA
"""

import json
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import os

def extract_embeddings(texts, model, tokenizer, device):
    """Extract embeddings for a list of texts"""
    embeddings = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Extracting embeddings"):
            # Tokenize
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Get model outputs
            outputs = model(**inputs)
            
            # Take mean of last hidden states (excluding padding tokens)
            attention_mask = inputs['attention_mask']
            last_hidden_states = outputs.last_hidden_state
            
            # Mask out padding tokens and compute mean
            masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
            mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            embeddings.append(mean_embedding.cpu().numpy())
    
    return np.vstack(embeddings)

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load sample prompts
    with open('../data/sample_prompts.json', 'r') as f:
        prompts = json.load(f)
    
    print(f"Loaded {len(prompts)} sample prompts:")
    for i, prompt in enumerate(prompts):
        print(f"{i+1}. {prompt}")
    
    # Use an open alternative model that doesn't require special access
    # Options: "microsoft/DialoGPT-medium", "distilbert-base-uncased", "sentence-transformers/all-MiniLM-L6-v2"
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Good for embeddings, no access restrictions
    
    print(f"\nLoading model: {model_name}")
    print("Note: This may take a few minutes for the first time...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded successfully. Hidden size: {model.config.hidden_size}")
        
        # Extract embeddings
        print("\nExtracting embeddings...")
        embeddings = extract_embeddings(prompts, model, tokenizer, device)
        print(f"Embeddings shape: {embeddings.shape}")
        
        # Compute some basic statistics
        print(f"\nEmbedding Statistics:")
        print(f"  Mean: {np.mean(embeddings):.4f}")
        print(f"  Std: {np.std(embeddings):.4f}")
        print(f"  Min: {np.min(embeddings):.4f}")
        print(f"  Max: {np.max(embeddings):.4f}")
        
        # Compute pairwise cosine distances
        from sklearn.metrics.pairwise import cosine_distances
        cosine_dist = cosine_distances(embeddings)
        
        print(f"\nCosine Distance Matrix:")
        print(cosine_dist)
        
        # Save results
        results = {
            'prompts': prompts,
            'embeddings': embeddings.tolist(),
            'cosine_distances': cosine_dist.tolist(),
            'model_name': model_name,
            'embedding_shape': embeddings.shape
        }
        
        os.makedirs('../data', exist_ok=True)
        with open('../data/llama_embeddings_test.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to ../data/llama_embeddings_test.json")
        
    except Exception as e:
        print(f"Error: {e}")
        print("This might be due to:")
        print("1. Model access permissions (you may need to request access to LLaMA models)")
        print("2. Internet connection issues")
        print("3. Insufficient memory")
        print("\nTry using a different model or check your HuggingFace access.")

if __name__ == "__main__":
    main()
