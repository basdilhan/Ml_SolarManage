#!/bin/bash
# Quick Start Script for Solar Wastage Prediction Dashboard
# Usage: bash setup.sh (on Mac/Linux) or run in PowerShell on Windows

echo "=================================="
echo "🚀 Solar Wastage Prediction Setup"
echo "=================================="

# Check Python version
echo ""
echo "📌 Checking Python version..."
python --version

# Create models directory
echo ""
echo "📁 Creating models directory..."
mkdir -p models
mkdir -p data/processed

# Install requirements
echo ""
echo "📦 Installing dependencies..."
pip install -r requirements.txt

# Check if training notebook has been run
if [ ! -f "models/model_metadata.json" ]; then
    echo ""
    echo "⚠️  WARNING: Models not found!"
    echo "   Please run: notebooks/03_model_training.ipynb first"
    echo ""
else
    echo ""
    echo "✅ Models found!"
    echo ""
    echo "🎉 Ready to launch dashboard!"
    echo ""
    echo "Run this command to start:"
    echo "  streamlit run app.py"
    echo ""
fi

echo "=================================="
