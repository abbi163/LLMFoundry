"""Configuration management for models and training.

This module provides configuration classes for managing model parameters,
training settings, and other hyperparameters in a structured way.
"""

import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for model parameters."""
    
    model_name: str = "gpt2"
    model_type: str = "causal_lm"
    vocab_size: Optional[int] = None
    max_position_embeddings: int = 1024
    hidden_size: int = 768
    num_attention_heads: int = 12
    num_hidden_layers: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    use_cache: bool = True
    torch_dtype: str = "float32"
    device_map: Optional[str] = "auto"
    
    # Trae-specific optimizations
    enable_trae_optimizations: bool = True
    use_gradient_checkpointing: bool = False
    use_flash_attention: bool = False
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training parameters
    learning_rate: float = 5e-5
    batch_size: int = 8
    num_epochs: int = 3
    max_steps: Optional[int] = None
    warmup_steps: int = 500
    weight_decay: float = 0.01
    adam_epsilon: float = 1e-8
    max_grad_norm: float = 1.0
    
    # Learning rate scheduling
    lr_scheduler_type: str = "linear"
    lr_scheduler_kwargs: Dict[str, Any] = None
    
    # Evaluation and logging
    eval_steps: int = 500
    logging_steps: int = 100
    save_steps: int = 1000
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    
    # Data parameters
    max_seq_length: int = 512
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    
    # Output directories
    output_dir: str = "./checkpoints"
    logging_dir: str = "./logs"
    
    # Trae-specific training optimizations
    use_trae_trainer: bool = True
    enable_mixed_precision: bool = True
    gradient_accumulation_steps: int = 1
    dataloader_drop_last: bool = True
    
    # Advanced options
    resume_from_checkpoint: Optional[str] = None
    ignore_data_skip: bool = False
    
    def __post_init__(self):
        """Post-initialization processing."""
        if self.lr_scheduler_kwargs is None:
            self.lr_scheduler_kwargs = {}
        
        # Ensure output directories exist
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.logging_dir).mkdir(parents=True, exist_ok=True)
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    def get_training_args(self) -> Dict[str, Any]:
        """Get training arguments for Transformers Trainer."""
        return {
            'output_dir': self.output_dir,
            'learning_rate': self.learning_rate,
            'per_device_train_batch_size': self.batch_size,
            'per_device_eval_batch_size': self.batch_size,
            'num_train_epochs': self.num_epochs,
            'max_steps': self.max_steps,
            'warmup_steps': self.warmup_steps,
            'weight_decay': self.weight_decay,
            'adam_epsilon': self.adam_epsilon,
            'max_grad_norm': self.max_grad_norm,
            'lr_scheduler_type': self.lr_scheduler_type,
            'eval_steps': self.eval_steps,
            'logging_steps': self.logging_steps,
            'save_steps': self.save_steps,
            'evaluation_strategy': self.eval_strategy,
            'save_strategy': self.save_strategy,
            'logging_dir': self.logging_dir,
            'fp16': self.enable_mixed_precision,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'dataloader_num_workers': self.dataloader_num_workers,
            'dataloader_pin_memory': self.dataloader_pin_memory,
            'dataloader_drop_last': self.dataloader_drop_last,
            'resume_from_checkpoint': self.resume_from_checkpoint,
            'ignore_data_skip': self.ignore_data_skip,
        }


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    
    # Retrieval settings
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
    
    # Embedding settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    
    # Vector database settings
    vector_db_type: str = "chromadb"  # or "faiss", "simple"
    vector_db_path: str = "./vector_db"
    collection_name: str = "documents"
    
    # Generation settings
    max_context_length: int = 2048
    max_new_tokens: int = 256
    temperature: float = 0.7
    do_sample: bool = True
    
    # Trae optimizations
    enable_context_compression: bool = True
    enable_response_caching: bool = True
    enable_adaptive_retrieval: bool = True
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        with open(path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> 'RAGConfig':
        """Load configuration from JSON file."""
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


def create_default_configs() -> tuple[ModelConfig, TrainingConfig, RAGConfig]:
    """Create default configurations for quick setup."""
    model_config = ModelConfig()
    training_config = TrainingConfig()
    rag_config = RAGConfig()
    
    return model_config, training_config, rag_config


def save_all_configs(model_config: ModelConfig, 
                    training_config: TrainingConfig,
                    rag_config: RAGConfig,
                    base_dir: str = "./configs") -> None:
    """Save all configurations to a directory."""
    config_dir = Path(base_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    
    model_config.save(config_dir / "model_config.json")
    training_config.save(config_dir / "training_config.json")
    rag_config.save(config_dir / "rag_config.json")
    
    print(f"All configurations saved to {config_dir}")


def load_all_configs(base_dir: str = "./configs") -> tuple[ModelConfig, TrainingConfig, RAGConfig]:
    """Load all configurations from a directory."""
    config_dir = Path(base_dir)
    
    model_config = ModelConfig.load(config_dir / "model_config.json")
    training_config = TrainingConfig.load(config_dir / "training_config.json")
    rag_config = RAGConfig.load(config_dir / "rag_config.json")
    
    return model_config, training_config, rag_config