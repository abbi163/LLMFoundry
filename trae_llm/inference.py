"""Inference utilities for LLM models.

This module provides classes and functions for running inference with trained models,
including text generation, classification, and batch processing capabilities.
"""

import torch
import torch.nn.functional as F
from transformers import (
    AutoModel, 
    AutoTokenizer, 
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList
)
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import time
import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass

from .config import ModelConfig

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for inference results."""
    text: str
    tokens: List[str]
    logits: Optional[torch.Tensor] = None
    probabilities: Optional[torch.Tensor] = None
    generation_time: Optional[float] = None
    token_count: Optional[int] = None
    tokens_per_second: Optional[float] = None


@dataclass
class ClassificationResult:
    """Container for classification results."""
    predicted_class: Union[str, int]
    confidence: float
    all_probabilities: Dict[Union[str, int], float]
    logits: Optional[torch.Tensor] = None
    inference_time: Optional[float] = None


class CustomStoppingCriteria(StoppingCriteria):
    """Custom stopping criteria for text generation."""
    
    def __init__(self, stop_tokens: List[str], tokenizer: AutoTokenizer):
        self.stop_tokens = stop_tokens
        self.tokenizer = tokenizer
        self.stop_token_ids = [tokenizer.encode(token, add_special_tokens=False) for token in stop_tokens]
    
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # Check if any stop token appears at the end of the sequence
        for stop_ids in self.stop_token_ids:
            if len(stop_ids) == 1:
                if input_ids[0, -1] == stop_ids[0]:
                    return True
            else:
                if input_ids[0, -len(stop_ids):].tolist() == stop_ids:
                    return True
        return False


class InferenceEngine:
    """Main inference engine for LLM models."""
    
    def __init__(self, 
                 model_path: str,
                 model_config: Optional[ModelConfig] = None,
                 device: Optional[str] = None,
                 load_in_8bit: bool = False,
                 load_in_4bit: bool = False):
        """
        Initialize InferenceEngine.
        
        Args:
            model_path: Path to model or model name
            model_config: Model configuration
            device: Device to load model on
            load_in_8bit: Whether to load model in 8-bit precision
            load_in_4bit: Whether to load model in 4-bit precision
        """
        self.model_path = model_path
        self.model_config = model_config or ModelConfig()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        
        # Load model and tokenizer
        self._load_model_and_tokenizer()
        
        # Setup generation config
        self.generation_config = self._create_generation_config()
        
        logger.info(f"InferenceEngine initialized with model: {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model type: {self.model_config.model_type}")
    
    def _load_model_and_tokenizer(self):
        """Load model and tokenizer."""
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            model_kwargs = {
                "torch_dtype": getattr(torch, self.model_config.torch_dtype),
                "device_map": self.model_config.device_map,
                "load_in_8bit": self.load_in_8bit,
                "load_in_4bit": self.load_in_4bit,
            }
            
            if self.model_config.model_type == "causal_lm":
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, **model_kwargs
                )
            elif self.model_config.model_type == "classification":
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_path, **model_kwargs
                )
            else:
                self.model = AutoModel.from_pretrained(
                    self.model_path, **model_kwargs
                )
            
            # Move to device if not using device_map
            if self.model_config.device_map is None:
                self.model = self.model.to(self.device)
            
            self.model.eval()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _create_generation_config(self) -> GenerationConfig:
        """Create generation configuration."""
        return GenerationConfig(
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=self.model_config.use_cache
        )
    
    def generate_text(self, 
                     prompt: str,
                     max_new_tokens: int = 256,
                     temperature: float = 0.7,
                     do_sample: bool = True,
                     top_p: float = 0.9,
                     top_k: int = 50,
                     repetition_penalty: float = 1.1,
                     stop_tokens: Optional[List[str]] = None,
                     return_full_text: bool = False) -> InferenceResult:
        """Generate text from a prompt."""
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.model_config.max_position_embeddings
        ).to(self.device)
        
        # Setup stopping criteria
        stopping_criteria = None
        if stop_tokens:
            stopping_criteria = StoppingCriteriaList([
                CustomStoppingCriteria(stop_tokens, self.tokenizer)
            ])
        
        # Update generation config
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=self.model_config.use_cache
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
                return_dict_in_generate=True,
                output_scores=True
            )
        
        # Decode output
        generated_tokens = outputs.sequences[0]
        if not return_full_text:
            # Remove input tokens
            generated_tokens = generated_tokens[inputs['input_ids'].shape[1]:]
        
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        generation_time = time.time() - start_time
        token_count = len(generated_tokens)
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0
        
        # Get token strings
        token_strings = [self.tokenizer.decode([token]) for token in generated_tokens]
        
        return InferenceResult(
            text=generated_text,
            tokens=token_strings,
            generation_time=generation_time,
            token_count=token_count,
            tokens_per_second=tokens_per_second
        )
    
    def classify_text(self, text: str, class_labels: Optional[List[str]] = None) -> ClassificationResult:
        """Classify text using the model."""
        
        if self.model_config.model_type != "classification":
            raise ValueError("Model is not configured for classification")
        
        start_time = time.time()
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_position_embeddings
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=-1)
        
        # Get predicted class
        predicted_idx = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0, predicted_idx].item()
        
        # Create probability dictionary
        if class_labels:
            all_probs = {label: prob.item() for label, prob in zip(class_labels, probabilities[0])}
            predicted_class = class_labels[predicted_idx]
        else:
            all_probs = {i: prob.item() for i, prob in enumerate(probabilities[0])}
            predicted_class = predicted_idx
        
        inference_time = time.time() - start_time
        
        return ClassificationResult(
            predicted_class=predicted_class,
            confidence=confidence,
            all_probabilities=all_probs,
            logits=logits,
            inference_time=inference_time
        )
    
    def batch_generate(self, 
                      prompts: List[str],
                      batch_size: int = 4,
                      **generation_kwargs) -> List[InferenceResult]:
        """Generate text for multiple prompts in batches."""
        
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # Process batch
            batch_results = []
            for prompt in batch_prompts:
                result = self.generate_text(prompt, **generation_kwargs)
                batch_results.append(result)
            
            results.extend(batch_results)
            
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(prompts) + batch_size - 1)//batch_size}")
        
        return results
    
    def get_embeddings(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """Get embeddings for text(s)."""
        
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.model_config.max_position_embeddings
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # Use last hidden state, mean pooling
                last_hidden_state = outputs.hidden_states[-1]
                embedding = torch.mean(last_hidden_state, dim=1).squeeze()
                embeddings.append(embedding)
        
        return torch.stack(embeddings) if len(embeddings) > 1 else embeddings[0]
    
    def calculate_perplexity(self, text: str) -> float:
        """Calculate perplexity of text."""
        
        if self.model_config.model_type != "causal_lm":
            raise ValueError("Perplexity calculation requires a causal language model")
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.model_config.max_position_embeddings
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            perplexity = torch.exp(loss)
        
        return perplexity.item()
    
    def save_results(self, results: List[Union[InferenceResult, ClassificationResult]], 
                    output_path: str):
        """Save inference results to file."""
        
        serializable_results = []
        
        for result in results:
            if isinstance(result, InferenceResult):
                serializable_results.append({
                    'type': 'generation',
                    'text': result.text,
                    'tokens': result.tokens,
                    'generation_time': result.generation_time,
                    'token_count': result.token_count,
                    'tokens_per_second': result.tokens_per_second
                })
            elif isinstance(result, ClassificationResult):
                serializable_results.append({
                    'type': 'classification',
                    'predicted_class': result.predicted_class,
                    'confidence': result.confidence,
                    'all_probabilities': result.all_probabilities,
                    'inference_time': result.inference_time
                })
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")


class BatchInferenceEngine:
    """Optimized engine for batch inference."""
    
    def __init__(self, inference_engine: InferenceEngine):
        self.engine = inference_engine
    
    def process_file(self, 
                    input_file: str,
                    output_file: str,
                    text_column: str = 'text',
                    batch_size: int = 8,
                    **generation_kwargs):
        """Process a file of texts for batch inference."""
        
        # Load data
        if input_file.endswith('.json'):
            with open(input_file, 'r') as f:
                data = json.load(f)
            texts = [item[text_column] for item in data]
        elif input_file.endswith('.csv'):
            import pandas as pd
            df = pd.read_csv(input_file)
            texts = df[text_column].tolist()
        else:
            raise ValueError("Unsupported file format. Use .json or .csv")
        
        # Process in batches
        results = self.engine.batch_generate(
            texts, 
            batch_size=batch_size, 
            **generation_kwargs
        )
        
        # Save results
        self.engine.save_results(results, output_file)
        
        logger.info(f"Processed {len(texts)} texts, results saved to {output_file}")


def create_inference_engine(model_path: str, 
                           model_config: Optional[ModelConfig] = None,
                           **kwargs) -> InferenceEngine:
    """Create an inference engine with default settings."""
    
    if model_config is None:
        model_config = ModelConfig()
    
    return InferenceEngine(
        model_path=model_path,
        model_config=model_config,
        **kwargs
    )


if __name__ == "__main__":
    # Example usage
    model_path = "gpt2"
    
    # Create inference engine
    engine = create_inference_engine(model_path)
    
    # Generate text
    prompt = "The future of artificial intelligence is"
    result = engine.generate_text(prompt, max_new_tokens=50)
    
    print(f"Prompt: {prompt}")
    print(f"Generated: {result.text}")
    print(f"Generation time: {result.generation_time:.2f}s")
    print(f"Tokens per second: {result.tokens_per_second:.2f}")