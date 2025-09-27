#!/usr/bin/env python3
"""
Test LLaMA access and provide alternatives
"""

import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import os

def test_llama_access():
    """Test if we can access LLaMA models"""
    print("Testing LLaMA model access...")
    
    # Try different LLaMA model names
    llama_models = [
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-13b-hf", 
        "meta-llama/Llama-2-70b-hf",
        "huggingface/CodeLlama-7b-hf",  # Alternative that might be accessible
        "microsoft/DialoGPT-large",     # Alternative GPT model
    ]
    
    accessible_models = []
    
    for model_name in llama_models:
        try:
            print(f"\nTrying {model_name}...")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            print(f"✅ SUCCESS: {model_name} is accessible!")
            accessible_models.append(model_name)
            
            # Test with a small sample
            test_text = "What is the capital of France?"
            inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                embedding_dim = outputs.last_hidden_state.shape[-1]
                print(f"   Embedding dimension: {embedding_dim}")
            
            return model_name, tokenizer, model
            
        except Exception as e:
            print(f"❌ FAILED: {model_name}")
            print(f"   Error: {str(e)[:100]}...")
    
    return None, None, None

def extract_llama_embeddings(prompts, model_name, tokenizer, model, max_prompts=50):
    """Extract embeddings using accessible model"""
    print(f"\nExtracting embeddings using {model_name}...")
    
    # Limit prompts for testing
    test_prompts = prompts[:max_prompts]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    embeddings = []
    
    with torch.no_grad():
        for text in tqdm(test_prompts, desc="Extracting embeddings"):
            inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = model(**inputs)
            attention_mask = inputs['attention_mask']
            last_hidden_states = outputs.last_hidden_state
            
            # Mean pooling
            masked_embeddings = last_hidden_states * attention_mask.unsqueeze(-1)
            mean_embedding = masked_embeddings.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
            
            embeddings.append(mean_embedding.cpu().numpy())
    
    return np.vstack(embeddings), test_prompts

def analyze_llama_embeddings(embeddings, prompts):
    """Quick analysis of LLaMA embeddings"""
    from sklearn.metrics.pairwise import cosine_distances
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    print(f"\nEmbedding analysis:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {np.mean(embeddings):.4f}")
    print(f"  Std: {np.std(embeddings):.4f}")
    
    # Cosine distances
    cosine_dist = cosine_distances(embeddings)
    avg_distance = np.mean(cosine_dist[np.triu_indices_from(cosine_dist, k=1)])
    print(f"  Average cosine distance: {avg_distance:.3f}")
    
    # Clustering
    best_silhouette = -1
    best_k = 2
    
    for k in range(2, min(6, len(prompts)//2)):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embeddings)
        
        if len(set(labels)) > 1:
            silhouette = silhouette_score(embeddings, labels)
            print(f"  K-means (k={k}): Silhouette = {silhouette:.3f}")
            
            if silhouette > best_silhouette:
                best_silhouette = silhouette
                best_k = k
    
    print(f"  Best clustering: k={best_k}, Silhouette={best_silhouette:.3f}")
    
    return {
        'embeddings': embeddings.tolist(),
        'prompts': prompts,
        'avg_cosine_distance': float(avg_distance),
        'best_silhouette': best_silhouette,
        'best_k': best_k,
        'embedding_dim': embeddings.shape[1]
    }

def main():
    # Load prompts
    with open('../data/diverse_prompts.json', 'r') as f:
        prompts = json.load(f)
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Test LLaMA access
    model_name, tokenizer, model = test_llama_access()
    
    if model_name:
        print(f"\n🎉 SUCCESS! Using {model_name}")
        
        # Extract embeddings
        embeddings, test_prompts = extract_llama_embeddings(prompts, model_name, tokenizer, model)
        
        # Analyze embeddings
        results = analyze_llama_embeddings(embeddings, test_prompts)
        results['model_name'] = model_name
        
        # Save results
        os.makedirs('../data', exist_ok=True)
        output_file = f'../data/llama_embeddings_{model_name.replace("/", "_")}.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        # Compare with other models
        print(f"\nComparison with other models:")
        print(f"  {model_name}: Silhouette = {results['best_silhouette']:.3f}")
        print(f"  RoBERTa-base: Silhouette = 0.120")
        print(f"  BERT-base: Silhouette = 0.083")
        
    else:
        print(f"\n❌ No LLaMA models accessible")
        print(f"\nTo get LLaMA access:")
        print(f"1. Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf")
        print(f"2. Request access and wait for approval")
        print(f"3. Get token from: https://huggingface.co/settings/tokens")
        print(f"4. Run: hf auth login")
        print(f"5. Try this script again")
        
        print(f"\nAlternative: Use the models we already tested:")
        print(f"- RoBERTa-base (best clustering: 0.120)")
        print(f"- BERT-base (clustering: 0.083)")
        print(f"- DistilBERT (clustering: 0.083)")

if __name__ == "__main__":
    main()
