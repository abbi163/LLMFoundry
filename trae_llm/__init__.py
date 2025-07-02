"""Trae LLM Lab - Python modules for reusable LLM code.

This package provides utilities and components for working with Large Language Models,
including configuration management, dataset handling, training, inference, and RAG pipelines.
"""

__version__ = "0.1.0"
__author__ = "Trae LLM Lab"

from .config import ModelConfig, TrainingConfig
from .dataset import DatasetLoader, TextDataset
from .trainer import TraeOptimizedTrainer
from .inference import InferenceEngine
from .rag_pipeline import RAGPipeline

__all__ = [
    "ModelConfig",
    "TrainingConfig", 
    "DatasetLoader",
    "TextDataset",
    "TraeOptimizedTrainer",
    "InferenceEngine",
    "RAGPipeline"
]