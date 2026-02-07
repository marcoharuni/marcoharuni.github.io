#!/bin/bash
# One-time setup for Marco Haruni's website
# Run this once: bash setup.sh

echo "🚀 Setting up your blog converter..."
echo ""

# Install Python packages
echo "📦 Installing Python packages..."
sudo apt install -y python3-markdown python3-pygments

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Setup complete!"
    echo ""
    echo "🎉 You can now write blog posts in Markdown!"
    echo ""
    echo "Quick start:"
    echo "1. Create a .md file in blog/ folder"
    echo "2. Run: python3 convert_blog.py"
    echo "3. Push to GitHub"
    echo ""
    echo "📖 See MARKDOWN-GUIDE.md for full instructions"
else
    echo ""
    echo "⚠️ Could not install system packages."
    echo ""
    echo "Try manual installation:"
    echo "    sudo apt install python3-markdown python3-pygments"
    echo ""
    echo "Or use a virtual environment (see MARKDOWN-GUIDE.md)"
fi
