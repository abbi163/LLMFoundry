# Fine-Tuning Tutorial

This comprehensive tutorial will guide you through fine-tuning Large Language Models using Trae AI optimizations. You'll learn how to prepare datasets, configure training parameters, and optimize performance for your specific use case.

## Table of Contents

1. [Introduction](#introduction)
2. [Prerequisites](#prerequisites)
3. [Dataset Preparation](#dataset-preparation)
4. [Model Configuration](#model-configuration)
5. [Training Configuration](#training-configuration)
6. [Fine-Tuning Process](#fine-tuning-process)
7. [Monitoring and Evaluation](#monitoring-and-evaluation)
8. [Advanced Techniques](#advanced-techniques)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Introduction

Fine-tuning is the process of adapting a pre-trained language model to perform well on a specific task or domain. With Trae AI optimizations, you can achieve better performance while using resources more efficiently.

### What You'll Learn

- How to prepare and preprocess datasets for fine-tuning
- Configuring Trae-optimized training parameters
- Monitoring training progress and performance
- Evaluating fine-tuned models
- Advanced optimization techniques

### Use Cases Covered

- **Text Classification**: Sentiment analysis, topic categorization
- **Text Generation**: Creative writing, code generation
- **Question Answering**: Domain-specific Q&A systems
- **Conversation**: Chatbots and dialogue systems

## Prerequisites

### Hardware Requirements

- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Optimal**: 32GB+ RAM, NVIDIA GPU with 16GB+ VRAM

### Software Requirements

```bash
# Ensure you have the environment set up
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Verify installations
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### Knowledge Prerequisites

- Basic Python programming
- Understanding of machine learning concepts
- Familiarity with neural networks (helpful but not required)

## Dataset Preparation

### Supported Data Formats

Trae LLM Lab supports multiple data formats:

#### 1. JSON Format

```json
[
  {
    "text": "I love this product! It's amazing.",
    "label": "positive"
  },
  {
    "text": "This is terrible. I hate it.",
    "label": "negative"
  }
]
```

#### 2. CSV Format

```csv
text,label
"I love this product! It's amazing.",positive
"This is terrible. I hate it.",negative
```

#### 3. Conversation Format

```json
[
  {
    "messages": [
      {"role": "user", "content": "What is machine learning?"},
      {"role": "assistant", "content": "Machine learning is a subset of AI..."}
    ]
  }
]
```

### Data Preprocessing

Use the built-in preprocessing utilities:

```python
from trae_llm.dataset import DatasetLoader, DataPreprocessor

# Load your dataset
dataset = DatasetLoader.load_json("path/to/your/data.json")

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Clean and preprocess text
cleaned_data = []
for item in dataset:
    cleaned_text = preprocessor.clean_text(item["text"])
    cleaned_data.append({
        "text": cleaned_text,
        "label": item["label"]
    })

# Split into train/validation/test sets
train_data, val_data, test_data = preprocessor.split_dataset(
    cleaned_data, 
    train_ratio=0.8, 
    val_ratio=0.1
)

print(f"Training samples: {len(train_data)}")
print(f"Validation samples: {len(val_data)}")
print(f"Test samples: {len(test_data)}")
```

### Creating Sample Datasets

For experimentation, you can create sample datasets:

```python
from trae_llm.dataset import create_sample_dataset

# Create a sample classification dataset
sample_data = create_sample_dataset(
    task_type="classification",
    num_samples=1000,
    num_classes=3
)

# Save to file
import json
with open("data/sample_classification.json", "w") as f:
    json.dump(sample_data, f, indent=2)
```

## Model Configuration

### Choosing a Base Model

Select an appropriate pre-trained model based on your task:

```python
from trae_llm.config import ModelConfig

# For text classification
model_config = ModelConfig(
    model_name="distilbert-base-uncased",  # Lightweight and fast
    num_labels=3,  # Number of classes
    task_type="classification"
)

# For text generation
model_config = ModelConfig(
    model_name="microsoft/DialoGPT-medium",  # Good for dialogue
    task_type="generation"
)

# For question answering
model_config = ModelConfig(
    model_name="bert-base-uncased",
    task_type="question_answering"
)
```

### Model Size Considerations

| Model Size | Parameters | VRAM Required | Use Case |
|------------|------------|---------------|----------|
| Small | < 100M | 2-4GB | Experimentation, fast inference |
| Medium | 100M-1B | 4-8GB | Balanced performance |
| Large | 1B-7B | 8-16GB | High quality, slower inference |
| Very Large | 7B+ | 16GB+ | Best quality, research |

### Advanced Model Configuration

```python
model_config = ModelConfig(
    model_name="microsoft/DialoGPT-medium",
    torch_dtype="float16",  # Use FP16 for memory efficiency
    device_map="auto",  # Automatic device placement
    trust_remote_code=True,  # Allow custom model code
    use_auth_token=False,  # Set to True if using private models
    load_in_8bit=False,  # Enable 8-bit quantization
    load_in_4bit=False   # Enable 4-bit quantization
)
```

## Training Configuration

### Basic Training Setup

```python
from trae_llm.config import TrainingConfig

training_config = TrainingConfig(
    # Output settings
    output_dir="./results",
    overwrite_output_dir=True,
    
    # Training parameters
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-5,
    
    # Optimization
    gradient_checkpointing=True,
    fp16=True,  # Mixed precision training
    
    # Evaluation and logging
    evaluation_strategy="steps",
    eval_steps=500,
    logging_steps=100,
    save_steps=1000,
    
    # Early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False
)
```

### Memory-Optimized Configuration

For limited GPU memory:

```python
training_config = TrainingConfig(
    output_dir="./results",
    
    # Smaller batch sizes
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,  # Simulate larger batch size
    
    # Memory optimizations
    gradient_checkpointing=True,
    fp16=True,
    dataloader_pin_memory=False,  # Reduce memory usage
    
    # Reduce evaluation frequency
    eval_steps=1000,
    save_steps=2000,
    
    # Other settings
    num_train_epochs=3,
    learning_rate=5e-5
)
```

### Performance-Optimized Configuration

For maximum training speed:

```python
training_config = TrainingConfig(
    output_dir="./results",
    
    # Larger batch sizes
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    
    # Performance optimizations
    fp16=True,
    dataloader_pin_memory=True,
    dataloader_num_workers=4,
    group_by_length=True,  # Group similar length sequences
    
    # Frequent evaluation
    evaluation_strategy="steps",
    eval_steps=250,
    logging_steps=50,
    
    # Training parameters
    num_train_epochs=3,
    learning_rate=5e-5,
    warmup_steps=500
)
```

## Fine-Tuning Process

### Method 1: Using the CLI Script

The easiest way to start fine-tuning:

```bash
# Basic fine-tuning
python scripts/train_model.py \
    --model_path "distilbert-base-uncased" \
    --train_data "data/train.json" \
    --eval_data "data/eval.json" \
    --output_dir "./results" \
    --num_epochs 3 \
    --batch_size 8 \
    --learning_rate 5e-5

# With configuration files
python scripts/train_model.py \
    --model_config "configs/model_config.json" \
    --training_config "configs/training_config.json" \
    --train_data "data/train.json" \
    --eval_data "data/eval.json"
```

### Method 2: Using Python API

```python
from trae_llm.trainer import train_model
from trae_llm.config import ModelConfig, TrainingConfig
from trae_llm.dataset import DatasetLoader

# Load configurations
model_config = ModelConfig(
    model_name="distilbert-base-uncased",
    num_labels=2
)

training_config = TrainingConfig(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=5e-5
)

# Load datasets
train_data = DatasetLoader.load_json("data/train.json")
eval_data = DatasetLoader.load_json("data/eval.json")

# Start training
trainer = train_model(
    model_config=model_config,
    training_config=training_config,
    train_data=train_data,
    eval_data=eval_data
)

print("Training completed!")
```

### Method 3: Advanced Custom Training

```python
from trae_llm.trainer import TraeOptimizedTrainer, create_trainer
from trae_llm.dataset import TextDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Create datasets
train_dataset = TextDataset(
    data=train_data,
    tokenizer=tokenizer,
    max_length=128
)

eval_dataset = TextDataset(
    data=eval_data,
    tokenizer=tokenizer,
    max_length=128
)

# Create trainer
trainer = create_trainer(
    model_config=model_config,
    training_config=training_config,
    train_data=train_data,
    eval_data=eval_data
)

# Add custom callbacks
from trae_llm.trainer import TraePerformanceCallback, TraeMemoryCallback

trainer.add_callback(TraePerformanceCallback())
trainer.add_callback(TraeMemoryCallback())

# Start training
trainer.train()

# Save the model
trainer.save_model("./final_model")
tokenizer.save_pretrained("./final_model")
```

## Monitoring and Evaluation

### Training Metrics

Monitor key metrics during training:

```python
# Access training metrics
metrics = trainer.training_metrics

# Get latest metrics
latest = metrics.get_latest_metrics()
print(f"Current loss: {latest.get('train_loss', 'N/A')}")
print(f"Current accuracy: {latest.get('eval_accuracy', 'N/A')}")

# Get metric history
loss_history = metrics.get_metric_history("train_loss")
accuracy_history = metrics.get_metric_history("eval_accuracy")
```

### Performance Monitoring

```python
# Get performance statistics
for callback in trainer.callback_handler.callbacks:
    if isinstance(callback, TraePerformanceCallback):
        perf_stats = callback.get_performance_summary()
        print(f"Average step time: {perf_stats['avg_step_time']:.2f}s")
        print(f"Peak memory usage: {perf_stats['peak_memory_usage']:.2f}GB")
```

### Model Evaluation

```python
# Evaluate on test set
test_dataset = TextDataset(
    data=test_data,
    tokenizer=tokenizer,
    max_length=128
)

eval_results = trainer.evaluate(test_dataset)
print(f"Test accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"Test F1 score: {eval_results['eval_f1']:.4f}")
```

### Visualization

Use the evaluation notebook for comprehensive visualization:

```python
# In Jupyter notebook
from trae_llm.evaluation import EvaluationVisualizer

visualizer = EvaluationVisualizer()

# Plot training curves
visualizer.plot_training_curves(
    loss_history=loss_history,
    accuracy_history=accuracy_history
)

# Plot performance metrics
visualizer.plot_performance_metrics(trainer)
```

## Advanced Techniques

### Learning Rate Scheduling

```python
training_config = TrainingConfig(
    # ... other parameters ...
    
    # Learning rate scheduling
    lr_scheduler_type="cosine",
    warmup_steps=500,
    warmup_ratio=0.1,
    
    # Advanced optimization
    adam_epsilon=1e-8,
    weight_decay=0.01,
    max_grad_norm=1.0
)
```

### Custom Loss Functions

```python
class CustomTrainer(TraeOptimizedTrainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation with label smoothing."""
        labels = inputs.get("labels")
        outputs = model(**inputs)
        
        # Apply label smoothing
        if labels is not None:
            loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(outputs.logits.view(-1, self.model.config.num_labels), 
                           labels.view(-1))
        else:
            loss = outputs.loss
        
        return (loss, outputs) if return_outputs else loss
```

### Data Augmentation

```python
from trae_llm.dataset import DataPreprocessor

preprocessor = DataPreprocessor()

# Apply data augmentation
augmented_data = []
for item in train_data:
    # Original sample
    augmented_data.append(item)
    
    # Augmented samples
    augmented_text = preprocessor.augment_text(
        item["text"],
        methods=["synonym_replacement", "random_insertion"]
    )
    
    augmented_data.append({
        "text": augmented_text,
        "label": item["label"]
    })

print(f"Original samples: {len(train_data)}")
print(f"Augmented samples: {len(augmented_data)}")
```

### Model Quantization

```python
# Load model with quantization
model_config = ModelConfig(
    model_name="distilbert-base-uncased",
    load_in_8bit=True,  # 8-bit quantization
    device_map="auto"
)

# Or 4-bit quantization for even more memory savings
model_config = ModelConfig(
    model_name="distilbert-base-uncased",
    load_in_4bit=True,
    device_map="auto"
)
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Out of Memory (OOM) Errors

```python
# Solution: Reduce batch size and enable optimizations
training_config = TrainingConfig(
    per_device_train_batch_size=2,  # Reduce from 8
    gradient_accumulation_steps=4,  # Maintain effective batch size
    gradient_checkpointing=True,
    fp16=True,
    dataloader_pin_memory=False
)
```

#### 2. Slow Training Speed

```python
# Solution: Optimize for speed
training_config = TrainingConfig(
    per_device_train_batch_size=16,  # Increase if memory allows
    dataloader_num_workers=4,  # Parallel data loading
    group_by_length=True,  # Reduce padding
    fp16=True,  # Faster computation
    dataloader_pin_memory=True
)
```

#### 3. Poor Convergence

```python
# Solution: Adjust learning rate and add warmup
training_config = TrainingConfig(
    learning_rate=2e-5,  # Lower learning rate
    warmup_steps=500,  # Gradual warmup
    lr_scheduler_type="cosine",
    weight_decay=0.01,  # Regularization
    num_train_epochs=5  # More epochs
)
```

#### 4. Overfitting

```python
# Solution: Add regularization and early stopping
training_config = TrainingConfig(
    weight_decay=0.01,
    dropout_rate=0.1,
    early_stopping_patience=3,
    eval_steps=250,  # More frequent evaluation
    load_best_model_at_end=True
)
```

### Debugging Tips

1. **Start Small**: Begin with a small dataset and model
2. **Monitor Metrics**: Watch training and validation loss closely
3. **Check Data**: Verify your data preprocessing is correct
4. **Use Callbacks**: Implement custom callbacks for debugging
5. **Save Checkpoints**: Regular checkpointing prevents loss of progress

## Best Practices

### Data Preparation

1. **Quality over Quantity**: Clean, high-quality data is more valuable than large amounts of noisy data
2. **Balanced Datasets**: Ensure balanced representation across classes/categories
3. **Validation Strategy**: Use proper train/validation/test splits
4. **Data Augmentation**: Use augmentation techniques to increase dataset diversity

### Training Strategy

1. **Start with Pretrained Models**: Always start with a relevant pretrained model
2. **Learning Rate**: Use learning rate scheduling and warmup
3. **Batch Size**: Find the optimal batch size for your hardware
4. **Early Stopping**: Implement early stopping to prevent overfitting
5. **Regular Evaluation**: Evaluate frequently to monitor progress

### Resource Management

1. **Memory Optimization**: Use gradient checkpointing and mixed precision
2. **Efficient Data Loading**: Use multiple workers and pin memory
3. **Model Size**: Choose appropriate model size for your task and hardware
4. **Quantization**: Consider quantization for deployment

### Experiment Tracking

1. **Version Control**: Track code, data, and model versions
2. **Hyperparameter Logging**: Log all hyperparameters and results
3. **Reproducibility**: Set random seeds for reproducible results
4. **Documentation**: Document your experiments and findings

### Production Considerations

1. **Model Validation**: Thoroughly validate models before deployment
2. **Performance Testing**: Test inference speed and memory usage
3. **Monitoring**: Implement monitoring for deployed models
4. **Fallback Strategies**: Have fallback plans for model failures

## Next Steps

After completing this tutorial:

1. **Experiment with Different Models**: Try various pretrained models for your task
2. **Advanced Techniques**: Explore techniques like LoRA, QLoRA, or adapter tuning
3. **Custom Architectures**: Implement custom model architectures
4. **Deployment**: Learn about model deployment and serving
5. **RAG Systems**: Explore Retrieval-Augmented Generation for knowledge-intensive tasks

## Additional Resources

- [HuggingFace Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Lightning Documentation](https://pytorch-lightning.readthedocs.io/)
- [Weights & Biases for Experiment Tracking](https://wandb.ai/)
- [Papers with Code for Latest Research](https://paperswithcode.com/)

---

**Happy Fine-Tuning!** 🚀

For more advanced topics, check out the other notebooks in the `notebooks/` directory and explore the `trae_llm/` modules for deeper customization.