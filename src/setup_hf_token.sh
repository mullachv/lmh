#!/bin/bash
# HuggingFace Token Setup Script

echo "🔑 HuggingFace Token Setup"
echo "=========================="

# Check if token is already set
if [ ! -z "$HUGGINGFACE_HUB_TOKEN" ]; then
    echo "✅ Token is already set in current session"
    echo "Token starts with: ${HUGGINGFACE_HUB_TOKEN:0:10}..."
else
    echo "❌ No token found in current session"
fi

echo ""
echo "📋 To set up your HuggingFace token permanently:"
echo ""
echo "1. Get your token from: https://huggingface.co/settings/tokens"
echo "2. Run this command (replace YOUR_TOKEN with your actual token):"
echo ""
echo "   echo 'export HUGGINGFACE_HUB_TOKEN=\"hf_YOUR_TOKEN_HERE\"' >> ~/.zshrc"
echo ""
echo "3. Reload your shell:"
echo "   source ~/.zshrc"
echo ""
echo "4. Test the setup:"
echo "   cd /Users/vsm/work3/lmh && source .venv/bin/activate && cd src && python set_token.py"
echo ""

# Check if token is in .zshrc
if grep -q "HUGGINGFACE_HUB_TOKEN" ~/.zshrc; then
    echo "✅ Token is already stored in ~/.zshrc"
    echo "Current token in .zshrc:"
    grep "HUGGINGFACE_HUB_TOKEN" ~/.zshrc | head -1
else
    echo "❌ Token not found in ~/.zshrc"
    echo "You need to add it manually using the command above"
fi

echo ""
echo "🔍 Current shell profile files:"
echo "   ~/.zshrc: $(ls -la ~/.zshrc 2>/dev/null || echo 'not found')"
echo "   ~/.bash_profile: $(ls -la ~/.bash_profile 2>/dev/null || echo 'not found')"
echo "   ~/.profile: $(ls -la ~/.profile 2>/dev/null || echo 'not found')"
