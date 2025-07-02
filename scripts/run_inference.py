#!/usr/bin/env python3
"""CLI script for running inference with trained LLM models.

This script provides a command-line interface for generating text,
classifying text, and batch processing using the Trae LLM Lab inference engine.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from trae_llm.config import ModelConfig
from trae_llm.inference import InferenceEngine, create_inference_engine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run inference with LLM models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model or model name"
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to model configuration JSON file"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        help="Device to run inference on (cuda/cpu)"
    )
    
    parser.add_argument(
        "--load_in_8bit",
        action="store_true",
        help="Load model in 8-bit precision"
    )
    
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load model in 4-bit precision"
    )
    
    # Task selection
    task_group = parser.add_mutually_exclusive_group(required=True)
    task_group.add_argument(
        "--generate",
        action="store_true",
        help="Generate text from prompts"
    )
    
    task_group.add_argument(
        "--classify",
        action="store_true",
        help="Classify text"
    )
    
    task_group.add_argument(
        "--batch_process",
        action="store_true",
        help="Process batch of texts from file"
    )
    
    task_group.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode"
    )
    
    # Input arguments
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation"
    )
    
    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file for batch processing (JSON or CSV)"
    )
    
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file for results"
    )
    
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column in input file"
    )
    
    # Generation parameters
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature"
    )
    
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help="Top-p (nucleus) sampling parameter"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-k sampling parameter"
    )
    
    parser.add_argument(
        "--repetition_penalty",
        type=float,
        default=1.1,
        help="Repetition penalty"
    )
    
    parser.add_argument(
        "--do_sample",
        action="store_true",
        default=True,
        help="Use sampling for generation"
    )
    
    parser.add_argument(
        "--stop_tokens",
        type=str,
        nargs="*",
        help="Stop tokens for generation"
    )
    
    parser.add_argument(
        "--return_full_text",
        action="store_true",
        help="Return full text including prompt"
    )
    
    # Classification parameters
    parser.add_argument(
        "--class_labels",
        type=str,
        nargs="*",
        help="Class labels for classification"
    )
    
    # Batch processing parameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing"
    )
    
    # Output options
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output with timing information"
    )
    
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save results to file"
    )
    
    return parser.parse_args()


def load_model_config(config_path: str) -> ModelConfig:
    """Load model configuration from file."""
    try:
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return ModelConfig(**config_dict)
    except Exception as e:
        logger.error(f"Failed to load model config from {config_path}: {e}")
        return ModelConfig()


def generate_text(engine: InferenceEngine, args):
    """Generate text from prompt."""
    if not args.prompt:
        raise ValueError("--prompt must be specified for text generation")
    
    logger.info(f"Generating text for prompt: {args.prompt}")
    
    result = engine.generate_text(
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=args.do_sample,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        stop_tokens=args.stop_tokens,
        return_full_text=args.return_full_text
    )
    
    print("\n" + "="*50)
    print("GENERATED TEXT:")
    print("="*50)
    print(result.text)
    print("="*50)
    
    if args.verbose:
        print(f"\nGeneration Statistics:")
        print(f"- Tokens generated: {result.token_count}")
        print(f"- Generation time: {result.generation_time:.2f}s")
        print(f"- Tokens per second: {result.tokens_per_second:.2f}")
    
    return [result]


def classify_text(engine: InferenceEngine, args):
    """Classify text."""
    if not args.prompt:
        raise ValueError("--prompt must be specified for text classification")
    
    logger.info(f"Classifying text: {args.prompt}")
    
    result = engine.classify_text(
        text=args.prompt,
        class_labels=args.class_labels
    )
    
    print("\n" + "="*50)
    print("CLASSIFICATION RESULT:")
    print("="*50)
    print(f"Predicted class: {result.predicted_class}")
    print(f"Confidence: {result.confidence:.4f}")
    
    if args.verbose:
        print(f"\nAll probabilities:")
        for label, prob in result.all_probabilities.items():
            print(f"  {label}: {prob:.4f}")
        print(f"\nInference time: {result.inference_time:.2f}s")
    
    print("="*50)
    
    return [result]


def batch_process(engine: InferenceEngine, args):
    """Process batch of texts from file."""
    if not args.input_file:
        raise ValueError("--input_file must be specified for batch processing")
    
    logger.info(f"Processing batch from file: {args.input_file}")
    
    # Load input data
    if args.input_file.endswith('.json'):
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = [item[args.text_column] for item in data]
        else:
            texts = data[args.text_column]
    
    elif args.input_file.endswith('.csv'):
        import pandas as pd
        df = pd.read_csv(args.input_file)
        texts = df[args.text_column].tolist()
    
    else:
        raise ValueError("Input file must be JSON or CSV format")
    
    logger.info(f"Processing {len(texts)} texts")
    
    # Process based on task
    if args.generate:
        results = engine.batch_generate(
            prompts=texts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            stop_tokens=args.stop_tokens,
            return_full_text=args.return_full_text
        )
        
        print(f"\nGenerated {len(results)} responses")
        
        if args.verbose:
            avg_time = sum(r.generation_time for r in results) / len(results)
            avg_tokens = sum(r.token_count for r in results) / len(results)
            print(f"Average generation time: {avg_time:.2f}s")
            print(f"Average tokens generated: {avg_tokens:.1f}")
    
    elif args.classify:
        results = []
        for text in texts:
            result = engine.classify_text(text, args.class_labels)
            results.append(result)
        
        print(f"\nClassified {len(results)} texts")
        
        if args.verbose:
            avg_time = sum(r.inference_time for r in results) / len(results)
            print(f"Average inference time: {avg_time:.2f}s")
    
    else:
        raise ValueError("Either --generate or --classify must be specified for batch processing")
    
    return results


def interactive_mode(engine: InferenceEngine, args):
    """Run in interactive mode."""
    print("\n" + "="*50)
    print("INTERACTIVE MODE")
    print("Type 'quit' or 'exit' to stop")
    print("="*50)
    
    results = []
    
    while True:
        try:
            prompt = input("\nEnter your prompt: ").strip()
            
            if prompt.lower() in ['quit', 'exit', 'q']:
                break
            
            if not prompt:
                continue
            
            if args.generate or not args.classify:
                result = engine.generate_text(
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    do_sample=args.do_sample,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    repetition_penalty=args.repetition_penalty,
                    stop_tokens=args.stop_tokens,
                    return_full_text=args.return_full_text
                )
                
                print(f"\nGenerated: {result.text}")
                
                if args.verbose:
                    print(f"Time: {result.generation_time:.2f}s, "
                          f"Tokens: {result.token_count}, "
                          f"Speed: {result.tokens_per_second:.2f} tok/s")
            
            elif args.classify:
                result = engine.classify_text(prompt, args.class_labels)
                
                print(f"\nClass: {result.predicted_class} "
                      f"(confidence: {result.confidence:.4f})")
                
                if args.verbose:
                    print(f"Time: {result.inference_time:.2f}s")
            
            results.append(result)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
    
    return results


def save_results_to_file(results: List[Any], output_file: str):
    """Save results to output file."""
    logger.info(f"Saving results to {output_file}")
    
    # Convert results to serializable format
    serializable_results = []
    
    for result in results:
        if hasattr(result, 'text'):  # InferenceResult
            serializable_results.append({
                'type': 'generation',
                'text': result.text,
                'tokens': result.tokens,
                'generation_time': result.generation_time,
                'token_count': result.token_count,
                'tokens_per_second': result.tokens_per_second
            })
        elif hasattr(result, 'predicted_class'):  # ClassificationResult
            serializable_results.append({
                'type': 'classification',
                'predicted_class': result.predicted_class,
                'confidence': result.confidence,
                'all_probabilities': result.all_probabilities,
                'inference_time': result.inference_time
            })
    
    with open(output_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"Saved {len(results)} results to {output_file}")


def main():
    """Main inference function."""
    args = parse_arguments()
    
    logger.info("Starting Trae LLM inference...")
    logger.info(f"Model: {args.model_path}")
    
    try:
        # Load model configuration
        model_config = None
        if args.model_config:
            model_config = load_model_config(args.model_config)
        
        # Create inference engine
        logger.info("Loading model...")
        engine = create_inference_engine(
            model_path=args.model_path,
            model_config=model_config,
            device=args.device,
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit
        )
        
        logger.info("Model loaded successfully!")
        
        # Run inference based on task
        results = []
        
        if args.interactive:
            results = interactive_mode(engine, args)
        elif args.batch_process:
            results = batch_process(engine, args)
        elif args.generate:
            results = generate_text(engine, args)
        elif args.classify:
            results = classify_text(engine, args)
        
        # Save results if requested
        if args.save_results or args.output_file:
            output_file = args.output_file or "inference_results.json"
            save_results_to_file(results, output_file)
        
        logger.info("Inference completed successfully!")
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        raise


if __name__ == "__main__":
    main()