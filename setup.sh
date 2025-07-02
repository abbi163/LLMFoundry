#!/bin/bash

# Trae LLM Lab Setup Script
# This script sets up the development environment for the Trae LLM Lab project

set -e  # Exit on any error

echo "🚀 Setting up Trae LLM Lab environment..."

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
REQUIRED_VERSION="3.8"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Python $PYTHON_VERSION is installed, but Python $REQUIRED_VERSION or higher is required."
    exit 1
fi

echo "✅ Python $PYTHON_VERSION detected"

# Create conda environment for LLM development
if ! command -v conda &> /dev/null; then
    echo "⚠️  Conda not found. Creating Python virtual environment instead..."
    # Fallback to venv if conda is not available
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    else
        echo "📦 Virtual environment already exists"
    fi
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
else
    # Initialize conda for script usage
    if ! eval "$(conda shell.bash hook)" 2>/dev/null; then
        echo "⚠️  Could not initialize conda. Please run 'conda init' and restart your terminal."
        echo "   Alternatively, manually activate the environment after setup:"
        echo "   conda activate llm_dev"
        echo ""
        echo "   Continuing with conda environment creation..."
    fi
    
    # Check if conda environment exists
    if conda env list | grep -q "llm_dev"; then
        echo "📦 Conda environment 'llm_dev' already exists"
    else
        echo "📦 Creating conda environment 'llm_dev'..."
        conda create -n llm_dev python=3.9 -y
    fi
    
    # Try to activate conda environment
    echo "🔧 Activating conda environment 'llm_dev'..."
    if ! conda activate llm_dev 2>/dev/null; then
        echo "⚠️  Could not activate conda environment automatically."
        echo "   Please manually activate it after setup: conda activate llm_dev"
        # Set a flag to indicate manual activation needed
        MANUAL_ACTIVATION_NEEDED=true
    fi
fi

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
if command -v conda &> /dev/null && conda env list | grep -q "llm_dev"; then
    echo "📦 Installing conda packages..."
    conda install jupyterlab nodejs -c conda-forge -y
    echo "📚 Installing additional pip packages..."
    pip install -r requirements.txt
else
    pip install -r requirements.txt
fi

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data
mkdir -p logs
mkdir -p checkpoints
mkdir -p models

# Install Jupyter kernel
echo "🔬 Setting up Jupyter kernel..."
if command -v conda &> /dev/null && conda env list | grep -q "llm_dev"; then
    python -m ipykernel install --user --name=llm_dev --display-name="LLM Development"
else
    python -m ipykernel install --user --name=trae-llm-lab --display-name="Trae LLM Lab"
fi

echo ""
# Fix NumPy compatibility issues
echo "🔧 Checking NumPy compatibility..."
NUMPY_VERSION=$(python -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "not installed")
if [[ "$NUMPY_VERSION" == "not installed" ]]; then
    echo "⚠️  NumPy not found, will be installed with requirements"
elif [[ "$NUMPY_VERSION" =~ ^2\. ]]; then
    echo "⚠️  NumPy 2.x detected ($NUMPY_VERSION), downgrading for compatibility..."
    pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
    echo "✅ NumPy downgraded to compatible version"
else
    echo "✅ NumPy version $NUMPY_VERSION is compatible"
fi

# Verify JupyterLab installation
echo "🔍 Verifying JupyterLab installation..."
if python -c "import jupyterlab" 2>/dev/null; then
    echo "✅ JupyterLab successfully installed"
else
    echo "⚠️  JupyterLab installation may have issues"
fi

# Test core package imports
echo "🧪 Testing core package compatibility..."
if python -c "import torch; import transformers; print('✅ Core ML packages working')" 2>/dev/null; then
    echo "✅ All core packages are compatible"
else
    echo "⚠️  Some packages may have compatibility issues. Check the troubleshooting guide."
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "To get started:"
if command -v conda &> /dev/null && conda env list | grep -q "llm_dev"; then
    if [ "$MANUAL_ACTIVATION_NEEDED" = true ]; then
        echo "1. First run: conda init"
        echo "2. Restart your terminal"
        echo "3. Activate the conda environment: conda activate llm_dev"
        echo "4. Start Jupyter: jupyter lab"
    else
        echo "1. Activate the conda environment: conda activate llm_dev"
        echo "2. Start Jupyter: jupyter lab"
    fi
else
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Start Jupyter: jupyter lab"
fi
echo "3. Open notebooks/01_hf_inference_basics.ipynb to begin"
echo ""
echo "Happy learning! 🤖"