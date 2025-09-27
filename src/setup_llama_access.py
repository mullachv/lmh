#!/usr/bin/env python3
"""
Setup script for LLaMA access
"""

import subprocess
import sys
import os

def check_huggingface_login():
    """Check if user is logged into HuggingFace"""
    try:
        result = subprocess.run(['huggingface-cli', 'whoami'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f"Logged in as: {result.stdout.strip()}")
            return True
        else:
            print("Not logged into HuggingFace")
            return False
    except FileNotFoundError:
        print("huggingface-cli not found. Installing...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'huggingface_hub[cli]'])
        return False

def setup_llama_access():
    """Guide user through LLaMA access setup"""
    print("="*60)
    print("LLAMA ACCESS SETUP")
    print("="*60)
    
    print("\n1. First, you need to request access to LLaMA models:")
    print("   Visit: https://huggingface.co/meta-llama/Llama-2-7b-hf")
    print("   Click 'Request access' and fill out the form")
    print("   Wait for approval (usually takes a few hours to days)")
    
    print("\n2. Get a HuggingFace token:")
    print("   Visit: https://huggingface.co/settings/tokens")
    print("   Create a new token with 'Read' permissions")
    print("   Copy the token")
    
    print("\n3. Login to HuggingFace:")
    if not check_huggingface_login():
        print("   Run: huggingface-cli login")
        print("   Paste your token when prompted")
    
    print("\n4. Test LLaMA access:")
    print("   Once approved, you can test with:")
    print("   python test_llama_access.py")
    
    print("\n" + "="*60)
    print("ALTERNATIVE MODELS (No access required)")
    print("="*60)
    print("While waiting for LLaMA access, you can use these models:")
    print("- microsoft/DialoGPT-medium")
    print("- distilbert-base-uncased") 
    print("- roberta-base")
    print("- bert-base-uncased")
    print("- sentence-transformers/all-mpnet-base-v2")

if __name__ == "__main__":
    setup_llama_access()
