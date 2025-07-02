# Environment Setup Guide

This guide explains the environment setup for the Trae LLM Lab project, including the new `llm_dev` conda environment and package requirements.

## Environment Options

### 1. Conda Environment (Recommended)

The project now uses a dedicated conda environment named `llm_dev` for all LLM-related work. This provides better dependency management and isolation.

**Benefits:**
- Better package management
- Easier dependency resolution
- GPU support (CUDA) integration
- Environment reproducibility
- Faster package installation

**Setup:**
```bash
# Automatic setup
chmod +x setup.sh
./setup.sh

# Manual setup
conda env create -f environment.yml
conda activate llm_dev
```

This will:
- Create the `llm_dev` conda environment (if conda is available)
- Install JupyterLab and Node.js via conda
- Install all Python dependencies
- Set up Jupyter kernel
- Verify installations

### 2. Python Virtual Environment (Fallback)

If conda is not available, the setup script automatically falls back to Python's built-in virtual environment.

**Setup:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Package Analysis

Based on the analysis of `notebooks/FineTuning/HuggingFace Inference-v1.ipynb`, the following packages and versions have been identified:

### Core Dependencies

| Package | Version | Purpose |
|---------|---------|----------|
| `torch` | >=2.0.0,<2.2.0 | Deep learning framework |
| `transformers` | >=4.30.0,<5.0.0 | HuggingFace transformers library |
| `huggingface-hub` | >=0.16.0,<1.0.0 | Model hub integration |
| `tokenizers` | >=0.13.0,<1.0.0 | Fast tokenization |
| `safetensors` | >=0.3.0 | Safe tensor serialization |
| `regex` | >=2023.6.3 | Regular expressions for tokenization |

### Models Used in Notebook

The notebook demonstrates usage of the following pre-trained models:

1. **DistilBERT** (`distilbert-base-uncased-finetuned-sst-2-english`)
   - Task: Sentiment analysis
   - Use case: Text classification

2. **GPT-2** (`gpt2`)
   - Task: Text generation
   - Use case: Creative writing, completion

3. **T5** (`t5-small`)
   - Task: Text-to-text generation
   - Use case: Translation, summarization

4. **BERT** (`bert-base-uncased`)
   - Task: Fill-mask
   - Use case: Masked language modeling

5. **XLM-RoBERTa** (`papluca/xlm-roberta-base-language-detection`)
   - Task: Language detection
   - Use case: Multilingual classification

### Additional Dependencies

| Package | Purpose |
|---------|----------|
| `jupyterlab` | Enhanced Jupyter interface |
| `ipykernel` | Jupyter kernel support |
| `nbformat` | Notebook format handling |
| `typing-extensions` | Type hints support |
| `packaging` | Package version handling |

## Launching JupyterLab

After setup, launch JupyterLab:

```bash
# Activate environment first
conda activate llm_dev  # or source venv/bin/activate

# Launch JupyterLab
jupyter lab
```

JupyterLab will open in your browser at `http://localhost:8888`

## Environment Variables

Set these environment variables for optimal performance:

```bash
# For CUDA support
export CUDA_VISIBLE_DEVICES=0

# For MPS (Apple Silicon)
export PYTORCH_ENABLE_MPS_FALLBACK=1

# For memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

## GPU Support

For GPU acceleration:

### CUDA Setup
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-enabled PyTorch (if needed)
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### MPS (Apple Silicon)
```bash
# Check MPS availability
python -c "import torch; print(torch.backends.mps.is_available())"
```

## Troubleshooting

### Common Issues

1. **Conda environment activation fails**
   ```bash
   # Initialize conda for your shell
   conda init bash  # or zsh, fish, etc.
   # Restart terminal
   ```

2. **Package conflicts**
   ```bash
   # Clean conda cache
   conda clean --all
   
   # Recreate environment
   conda env remove -n llm_dev
   conda env create -f environment.yml
   ```

3. **CUDA version mismatch**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install compatible PyTorch
   # Visit: https://pytorch.org/get-started/locally/
   ```

4. **Memory issues**
   ```bash
   # Set memory limits
   export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
   ```

### Performance Optimization

1. **Enable mixed precision**
   ```python
   # In your training code
   from torch.cuda.amp import autocast, GradScaler
   ```

2. **Use fast tokenizers**
   ```python
   # Always use fast tokenizers when available
   tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
   ```

3. **Optimize data loading**
   ```python
   # Use multiple workers for data loading
   dataloader = DataLoader(dataset, num_workers=4, pin_memory=True)
   ```

## Next Steps

After setting up the environment:

1. **Verify installation**
   ```bash
   python -c "import torch, transformers; print('Setup successful!')"
   ```

2. **Start with notebooks**
   ```bash
   jupyter lab notebooks/
   ```

3. **Run tests**
   ```bash
   python -m pytest tests/ -v
   ```

4. **Explore CLI tools**
   ```bash
   python scripts/run_inference.py --help
   ```