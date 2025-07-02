# Troubleshooting Guide

This guide helps resolve common issues when setting up and using the Trae LLM Lab.

## Conda Activation Issues

### Problem: "Run 'conda init' before 'conda activate'"

This error occurs when conda hasn't been properly initialized in your shell.

**Solution:**

1. **Initialize conda:**
   ```bash
   conda init
   ```

2. **Restart your terminal** or reload your shell configuration:
   ```bash
   # For bash
   source ~/.bashrc
   
   # For zsh
   source ~/.zshrc
   ```

3. **Try activating the environment again:**
   ```bash
   conda activate llm_dev
   ```

4. **If the environment doesn't exist, create it:**
   ```bash
   conda create -n llm_dev python=3.9 -y
   conda activate llm_dev
   ```

### Alternative: Manual Environment Setup

If conda continues to have issues, you can set up the environment manually:

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate llm_dev

# Install additional packages
pip install -r requirements.txt

# Setup Jupyter kernel
python -m ipykernel install --user --name=llm_dev --display-name="LLM Development"
```

## Python Virtual Environment Fallback

If conda is not available or continues to have issues:

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Setup Jupyter kernel
python -m ipykernel install --user --name=trae-llm-lab --display-name="Trae LLM Lab"
```

## JupyterLab Issues

### Problem: JupyterLab not starting

**Solutions:**

1. **Check if JupyterLab is installed:**
   ```bash
   jupyter lab --version
   ```

2. **Install JupyterLab if missing:**
   ```bash
   pip install jupyterlab
   # or
   conda install jupyterlab -c conda-forge
   ```

3. **Clear JupyterLab cache:**
   ```bash
   jupyter lab clean
   jupyter lab build
   ```

### Problem: Kernel not found in JupyterLab

**Solutions:**

1. **List available kernels:**
   ```bash
   jupyter kernelspec list
   ```

2. **Install the kernel:**
   ```bash
   python -m ipykernel install --user --name=llm_dev --display-name="LLM Development"
   ```

3. **Restart JupyterLab** and refresh the browser

## Memory Issues

### Problem: Out of Memory (OOM) errors

**Solutions:**

1. **Reduce batch size** in training configurations
2. **Enable gradient checkpointing:**
   ```python
   model.gradient_checkpointing_enable()
   ```
3. **Use mixed precision training:**
   ```python
   from accelerate import Accelerator
   accelerator = Accelerator(mixed_precision="fp16")
   ```

## Package Installation Issues

### Problem: NumPy 2.0 Compatibility Issues

**Error message:**
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.0.2 as it may crash.
```

**Solution:**
This project pins NumPy to `<2.0.0` for compatibility. The setup script automatically detects and fixes this issue:

1. **Run the setup script (recommended):**
   ```bash
   ./setup.sh
   ```
   The script will automatically detect NumPy 2.x and downgrade it to a compatible version.

2. **Or manually downgrade NumPy:**
   ```bash
   pip install "numpy>=1.24.0,<2.0.0" --force-reinstall
   ```

3. **For persistent issues, recreate the environment:**
   ```bash
   # Remove existing environment
   conda env remove -n llm_dev
   
   # Recreate with fixed dependencies
   ./setup.sh
   ```

### Problem: Package conflicts or installation failures

**Solutions:**

1. **Update pip:**
   ```bash
   pip install --upgrade pip
   ```

2. **Clear pip cache:**
   ```bash
   pip cache purge
   ```

3. **Install packages one by one** to identify problematic packages:
   ```bash
   pip install torch
   pip install transformers
   # etc.
   ```

4. **Use conda for core packages:**
   ```bash
   conda install pytorch transformers -c pytorch -c huggingface
   ```

## GPU Issues

### Problem: CUDA not detected

**Check CUDA availability:**
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
```

**Solutions:**

1. **Install CUDA-compatible PyTorch:**
   ```bash
   # Check https://pytorch.org for the right command
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **For Apple Silicon (MPS):**
   ```python
   import torch
   print(f"MPS available: {torch.backends.mps.is_available()}")
   ```

## Getting Help

If you continue to experience issues:

1. **Check the logs** in the `logs/` directory
2. **Review the documentation** in the `docs/` folder
3. **Search for similar issues** in the project repository
4. **Create a detailed issue report** with:
   - Your operating system
   - Python version
   - Conda version (if applicable)
   - Full error message
   - Steps to reproduce

## Common Environment Variables

Set these if you encounter specific issues:

```bash
# For CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# For MPS fallback (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# For HuggingFace cache
export HF_HOME="./cache/huggingface"

# For transformers cache
export TRANSFORMERS_CACHE="./cache/transformers"
```