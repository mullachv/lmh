#!/usr/bin/env python3
"""
Set HuggingFace token as environment variable and test LLaMA access
"""

import os
import sys
from huggingface_hub import login, whoami
from transformers import AutoTokenizer, AutoModel

def set_token_and_test():
    print("HuggingFace Token Setup")
    print("=" * 40)
    
    # Check if already logged in
    try:
        user = whoami()
        print(f"Already logged in as: {user['name']}")
        return True
    except:
        print("Not logged in")
    
    # Instructions
    print("\n📋 Instructions:")
    print("1. Visit: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Copy the token")
    print("4. Run this command with your token:")
    print("   export HUGGINGFACE_HUB_TOKEN='your_token_here'")
    print("   python set_token.py")
    print("\nOr set it directly in your shell:")
    print("   export HUGGINGFACE_HUB_TOKEN='hf_...'")
    
    # Check if token is set in environment
    token = os.getenv('HUGGINGFACE_HUB_TOKEN')
    if token:
        print(f"\n✅ Token found in environment")
        try:
            login(token=token)
            user = whoami()
            print(f"✅ Successfully logged in as: {user['name']}")
            return True
        except Exception as e:
            print(f"❌ Login failed: {e}")
            return False
    else:
        print(f"\n❌ No token found in environment")
        print("Please set HUGGINGFACE_HUB_TOKEN environment variable")
        return False

def test_llama_access():
    """Test LLaMA access after login"""
    print("\n🦙 Testing LLaMA Access...")
    
    try:
        print("Loading LLaMA-2-7b-hf...")
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
        model = AutoModel.from_pretrained("meta-llama/Llama-2-7b-hf")
        
        print("✅ SUCCESS! LLaMA-2-7b-hf is accessible!")
        print(f"Model config: {model.config.name_or_path}")
        print(f"Hidden size: {model.config.hidden_size}")
        
        # Test embedding extraction
        test_text = "What is the capital of France?"
        # Fix tokenizer padding issue
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        with model.no_grad():
            outputs = model(**inputs)
            embedding_dim = outputs.last_hidden_state.shape[-1]
            print(f"Embedding dimension: {embedding_dim}")
        
        return True
        
    except Exception as e:
        print(f"❌ LLaMA access failed: {e}")
        return False

def main():
    # Set token and login
    if not set_token_and_test():
        print("\n❌ Please set your HuggingFace token and try again")
        return
    
    # Test LLaMA access
    if test_llama_access():
        print("\n🎉 LLaMA is ready to use!")
        print("You can now run the full analysis with LLaMA embeddings")
    else:
        print("\n❌ LLaMA access still not working")
        print("Make sure you have:")
        print("1. Requested access to LLaMA models")
        print("2. Received approval")
        print("3. Set your HuggingFace token correctly")

if __name__ == "__main__":
    main()
