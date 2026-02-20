#!/bin/bash
# Setup script for Genome Sonics on WSL2/Ubuntu
# Run with: bash setup_wsl.sh

set -e

echo "🧬 Setting up Genome → Music & Art..."
echo ""

# Check if running in WSL
if grep -qi microsoft /proc/version 2>/dev/null; then
    echo "✓ Running in WSL2"
else
    echo "⚠ Warning: This script is designed for WSL2/Ubuntu"
fi

# Update package list
echo "📦 Updating package list..."
sudo apt update

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt install -y python3 python3-pip python3-venv fluidsynth fluid-soundfont-gm

# Create virtual environment
echo "🐍 Creating Python virtual environment..."
python3 -m venv venv

# Activate and install
echo "📥 Installing Python packages..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create output directory
mkdir -p output

echo ""
echo "✅ Setup complete!"
echo ""
echo "To run the app:"
echo "  1. Activate environment: source venv/bin/activate"
echo "  2. Start app: streamlit run app.py"
echo ""
echo "The app will open in your browser at http://localhost:8501"
