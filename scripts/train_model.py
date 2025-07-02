#!/usr/bin/env python3
"""CLI script for training LLM models with Trae optimizations.

This script provides a command-line interface for fine-tuning language models
using the Trae LLM Lab training pipeline.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
from trae_llm.config import ModelConfig, TrainingConfig
from trae_llm.dataset import DatasetLoader, create_sample_dataset
from trae_llm.trainer import train_model

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LLM models with Trae optimizations",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="Model name or path"
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to model configuration JSON file"
    )
    
    # Training arguments
    parser.add_argument(
        "--training_config",
        type=str,
        help="Path to training configuration JSON file"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for model checkpoints"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )
    
    # Dataset arguments
    parser.add_argument(
        "--train_data",
        type=str,
        help="Path to training data file (JSON or CSV)"
    )
    
    parser.add_argument(
        "--eval_data",
        type=str,
        help="Path to evaluation data file (JSON or CSV)"
    )
    
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column in dataset"
    )
    
    parser.add_argument(
        "--label_column",
        type=str,
        help="Name of label column in dataset (for supervised learning)"
    )
    
    parser.add_argument(
        "--use_sample_data",
        action="store_true",
        help="Use sample data for testing"
    )
    
    parser.add_argument(
        "--sample_size",
        type=int,
        default=1000,
        help="Size of sample dataset"
    )
    
    # Optimization arguments
    parser.add_argument(
        "--enable_trae_optimizations",
        action="store_true",
        default=True,
        help="Enable Trae AI optimizations"
    )
    
    parser.add_argument(
        "--use_gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing"
    )
    
    parser.add_argument(
        "--enable_mixed_precision",
        action="store_true",
        default=True,
        help="Enable mixed precision training"
    )
    
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Gradient accumulation steps"
    )
    
    # Logging and evaluation
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=100,
        help="Log every N steps"
    )
    
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps"
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps"
    )
    
    # Other options
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        help="Path to checkpoint to resume from"
    )
    
    parser.add_argument(
        "--save_model",
        action="store_true",
        default=True,
        help="Save the trained model"
    )
    
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push model to Hugging Face Hub"
    )
    
    parser.add_argument(
        "--hub_model_id",
        type=str,
        help="Model ID for Hugging Face Hub"
    )
    
    return parser.parse_args()


def load_config_from_file(config_path: str, config_class):
    """Load configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return config_class(**config_dict)
    except Exception as e:
        logger.error(f"Failed to load config from {config_path}: {e}")
        return None


def create_model_config(args) -> ModelConfig:
    """Create model configuration from arguments."""
    if args.model_config:
        config = load_config_from_file(args.model_config, ModelConfig)
        if config:
            return config
    
    return ModelConfig(
        model_name=args.model_name,
        max_position_embeddings=args.max_seq_length,
        enable_trae_optimizations=args.enable_trae_optimizations,
        use_gradient_checkpointing=args.use_gradient_checkpointing
    )


def create_training_config(args) -> TrainingConfig:
    """Create training configuration from arguments."""
    if args.training_config:
        config = load_config_from_file(args.training_config, TrainingConfig)
        if config:
            return config
    
    return TrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_seq_length=args.max_seq_length,
        output_dir=args.output_dir,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        enable_mixed_precision=args.enable_mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        resume_from_checkpoint=args.resume_from_checkpoint
    )


def load_datasets(args, tokenizer):
    """Load training and evaluation datasets."""
    dataset_loader = DatasetLoader(tokenizer)
    
    if args.use_sample_data:
        logger.info(f"Creating sample dataset with {args.sample_size} samples")
        train_dataset = create_sample_dataset(
            tokenizer, 
            num_samples=args.sample_size,
            max_length=args.max_seq_length
        )
        eval_dataset = create_sample_dataset(
            tokenizer,
            num_samples=args.sample_size // 5,
            max_length=args.max_seq_length
        )
        return train_dataset, eval_dataset
    
    # Load training data
    if not args.train_data:
        raise ValueError("Either --train_data or --use_sample_data must be specified")
    
    logger.info(f"Loading training data from {args.train_data}")
    if args.train_data.endswith('.json'):
        train_dataset = dataset_loader.load_from_json(
            args.train_data,
            text_column=args.text_column,
            label_column=args.label_column,
            max_length=args.max_seq_length
        )
    elif args.train_data.endswith('.csv'):
        train_dataset = dataset_loader.load_from_csv(
            args.train_data,
            text_column=args.text_column,
            label_column=args.label_column,
            max_length=args.max_seq_length
        )
    else:
        raise ValueError("Training data must be JSON or CSV format")
    
    # Load evaluation data
    eval_dataset = None
    if args.eval_data:
        logger.info(f"Loading evaluation data from {args.eval_data}")
        if args.eval_data.endswith('.json'):
            eval_dataset = dataset_loader.load_from_json(
                args.eval_data,
                text_column=args.text_column,
                label_column=args.label_column,
                max_length=args.max_seq_length
            )
        elif args.eval_data.endswith('.csv'):
            eval_dataset = dataset_loader.load_from_csv(
                args.eval_data,
                text_column=args.text_column,
                label_column=args.label_column,
                max_length=args.max_seq_length
            )
    
    return train_dataset, eval_dataset


def main():
    """Main training function."""
    args = parse_arguments()
    
    logger.info("Starting Trae LLM training...")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        # Load model and tokenizer
        logger.info("Loading model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        
        # Ensure tokenizer has pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model for causal LM (most common case)
        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_name)
        except:
            # Fallback to AutoModel
            model = AutoModel.from_pretrained(args.model_name)
        
        # Create configurations
        model_config = create_model_config(args)
        training_config = create_training_config(args)
        
        logger.info(f"Model config: {model_config}")
        logger.info(f"Training config: {training_config}")
        
        # Load datasets
        train_dataset, eval_dataset = load_datasets(args, tokenizer)
        
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
        
        # Train model
        trainer = train_model(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model_config=model_config,
            training_config=training_config,
            save_model=args.save_model
        )
        
        # Push to hub if requested
        if args.push_to_hub:
            if not args.hub_model_id:
                raise ValueError("--hub_model_id must be specified when using --push_to_hub")
            
            logger.info(f"Pushing model to Hub: {args.hub_model_id}")
            trainer.push_to_hub(args.hub_model_id)
        
        logger.info("Training completed successfully!")
        
        # Print training statistics
        if hasattr(trainer, 'training_metrics') and trainer.training_metrics:
            final_metrics = trainer.training_metrics[-1]
            logger.info(f"Final training loss: {final_metrics.loss:.4f}")
            logger.info(f"Final learning rate: {final_metrics.learning_rate:.2e}")
            if final_metrics.throughput:
                logger.info(f"Average throughput: {final_metrics.throughput:.2f} samples/sec")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    main()