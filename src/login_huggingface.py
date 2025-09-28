#!/usr/bin/env python3
"""
Login to HuggingFace and test LLaMA access
"""

import os
from huggingface_hub import login, whoami

def main():
    print("HuggingFace Login Helper")
    print("=" * 40)
    
    # Check current status
    try:
        user = whoami()
        print(f"Already logged in as: {user['name']}")
        return True
    except:
        print("Not logged in")
    
    # Get token from user
    print("\nTo get your HuggingFace token:")
    print("1. Visit: https://huggingface.co/settings/tokens")
    print("2. Create a new token with 'Read' permissions")
    print("3. Copy the token")
    
    token = input("\nEnter your HuggingFace token: ").strip()
    
    if not token:
        print("No token provided. Exiting.")
        return False
    
    try:
        # Login with token
        login(token=token)
        print("✅ Successfully logged in!")
        
        # Verify login
        user = whoami()
        print(f"Logged in as: {user['name']}")
        return True
        
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Ready to test LLaMA access!")
        print("Run: python test_llama_access.py")
    else:
        print("\n❌ Please try again with a valid token")
