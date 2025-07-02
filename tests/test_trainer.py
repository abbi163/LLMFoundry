#!/usr/bin/env python3
"""Unit tests for the Trae trainer module.

This module contains comprehensive tests for the TraeOptimizedTrainer
and related training utilities.
"""

import unittest
import tempfile
import shutil
import json
import torch
from pathlib import Path
import sys
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from trae_llm.config import ModelConfig, TrainingConfig
from trae_llm.dataset import TextDataset, create_sample_dataset
from trae_llm.trainer import (
    TraeOptimizedTrainer,
    TrainingMetrics,
    TraePerformanceCallback,
    TraeMemoryCallback,
    TraeEarlyStoppingCallback,
    create_trainer,
    train_model
)

class TestTrainingConfig(unittest.TestCase):
    """Test TrainingConfig functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.config = TrainingConfig(
            output_dir=self.temp_dir,
            num_train_epochs=2,
            per_device_train_batch_size=2,
            learning_rate=5e-5
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_config_creation(self):
        """Test TrainingConfig creation."""
        self.assertEqual(self.config.num_train_epochs, 2)
        self.assertEqual(self.config.per_device_train_batch_size, 2)
        self.assertEqual(self.config.learning_rate, 5e-5)
        self.assertTrue(self.config.gradient_checkpointing)
    
    def test_config_save_load(self):
        """Test saving and loading configuration."""
        config_path = Path(self.temp_dir) / "training_config.json"
        
        # Save config
        self.config.save(str(config_path))
        self.assertTrue(config_path.exists())
        
        # Load config
        loaded_config = TrainingConfig.load(str(config_path))
        self.assertEqual(loaded_config.num_train_epochs, self.config.num_train_epochs)
        self.assertEqual(loaded_config.learning_rate, self.config.learning_rate)
    
    def test_get_training_args(self):
        """Test getting training arguments."""
        args = self.config.get_training_args()
        
        self.assertEqual(args.num_train_epochs, 2)
        self.assertEqual(args.per_device_train_batch_size, 2)
        self.assertEqual(args.learning_rate, 5e-5)
        self.assertTrue(args.gradient_checkpointing)

class TestTrainingMetrics(unittest.TestCase):
    """Test TrainingMetrics functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.metrics = TrainingMetrics()
    
    def test_log_metric(self):
        """Test logging metrics."""
        self.metrics.log_metric("loss", 0.5, step=1)
        self.metrics.log_metric("accuracy", 0.8, step=1)
        
        self.assertIn("loss", self.metrics.metrics)
        self.assertIn("accuracy", self.metrics.metrics)
        self.assertEqual(len(self.metrics.metrics["loss"]), 1)
        self.assertEqual(self.metrics.metrics["loss"][0]["value"], 0.5)
    
    def test_get_latest_metrics(self):
        """Test getting latest metrics."""
        self.metrics.log_metric("loss", 0.5, step=1)
        self.metrics.log_metric("loss", 0.3, step=2)
        
        latest = self.metrics.get_latest_metrics()
        self.assertEqual(latest["loss"], 0.3)
    
    def test_get_metric_history(self):
        """Test getting metric history."""
        self.metrics.log_metric("loss", 0.5, step=1)
        self.metrics.log_metric("loss", 0.3, step=2)
        
        history = self.metrics.get_metric_history("loss")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["value"], 0.5)
        self.assertEqual(history[1]["value"], 0.3)
    
    def test_save_load_metrics(self):
        """Test saving and loading metrics."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            metrics_path = f.name
        
        try:
            # Log some metrics
            self.metrics.log_metric("loss", 0.5, step=1)
            self.metrics.log_metric("accuracy", 0.8, step=1)
            
            # Save metrics
            self.metrics.save(metrics_path)
            
            # Load metrics
            new_metrics = TrainingMetrics()
            new_metrics.load(metrics_path)
            
            self.assertEqual(new_metrics.get_latest_metrics()["loss"], 0.5)
            self.assertEqual(new_metrics.get_latest_metrics()["accuracy"], 0.8)
        
        finally:
            Path(metrics_path).unlink(missing_ok=True)

class TestTraeCallbacks(unittest.TestCase):
    """Test Trae callback functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_trainer = Mock()
        self.mock_trainer.state = Mock()
        self.mock_trainer.state.global_step = 10
        self.mock_trainer.state.epoch = 1
    
    def test_performance_callback(self):
        """Test TraePerformanceCallback."""
        callback = TraePerformanceCallback()
        
        # Test step begin
        callback.on_step_begin(None, self.mock_trainer, None)
        self.assertIsNotNone(callback.step_start_time)
        
        # Test step end
        import time
        time.sleep(0.01)  # Small delay
        callback.on_step_end(None, self.mock_trainer, None)
        
        self.assertGreater(len(callback.step_times), 0)
        self.assertGreater(callback.step_times[0], 0)
    
    @patch('torch.cuda.is_available', return_value=True)
    @patch('torch.cuda.memory_allocated', return_value=1024*1024*100)  # 100MB
    @patch('torch.cuda.max_memory_allocated', return_value=1024*1024*200)  # 200MB
    def test_memory_callback(self, mock_max_mem, mock_mem, mock_cuda):
        """Test TraeMemoryCallback."""
        callback = TraeMemoryCallback()
        
        # Test step end
        callback.on_step_end(None, self.mock_trainer, None)
        
        self.assertGreater(len(callback.memory_usage), 0)
        self.assertEqual(callback.memory_usage[0]['allocated_mb'], 100)
        self.assertEqual(callback.memory_usage[0]['max_allocated_mb'], 200)
    
    def test_early_stopping_callback(self):
        """Test TraeEarlyStoppingCallback."""
        callback = TraeEarlyStoppingCallback(
            metric="eval_loss",
            patience=2,
            min_delta=0.01
        )
        
        # Mock logs
        logs1 = {"eval_loss": 0.5}
        logs2 = {"eval_loss": 0.4}  # Improvement
        logs3 = {"eval_loss": 0.41}  # No significant improvement
        logs4 = {"eval_loss": 0.42}  # No improvement
        
        # Test evaluation steps
        callback.on_evaluate(None, self.mock_trainer, logs1)
        self.assertFalse(callback.should_stop)
        
        callback.on_evaluate(None, self.mock_trainer, logs2)
        self.assertFalse(callback.should_stop)
        
        callback.on_evaluate(None, self.mock_trainer, logs3)
        self.assertFalse(callback.should_stop)
        
        callback.on_evaluate(None, self.mock_trainer, logs4)
        self.assertTrue(callback.should_stop)

class TestTextDataset(unittest.TestCase):
    """Test TextDataset functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create sample data
        self.sample_data = [
            {"text": "This is a positive review", "label": "positive"},
            {"text": "This is a negative review", "label": "negative"},
            {"text": "This is a neutral review", "label": "neutral"}
        ]
        
        # Mock tokenizer
        self.mock_tokenizer = Mock()
        self.mock_tokenizer.return_value = {
            "input_ids": [1, 2, 3, 4, 5],
            "attention_mask": [1, 1, 1, 1, 1]
        }
        self.mock_tokenizer.pad_token_id = 0
    
    def test_dataset_creation(self):
        """Test TextDataset creation."""
        dataset = TextDataset(
            data=self.sample_data,
            tokenizer=self.mock_tokenizer,
            max_length=128
        )
        
        self.assertEqual(len(dataset), 3)
        self.assertEqual(dataset.max_length, 128)
    
    def test_dataset_getitem(self):
        """Test TextDataset __getitem__."""
        dataset = TextDataset(
            data=self.sample_data,
            tokenizer=self.mock_tokenizer,
            max_length=128
        )
        
        item = dataset[0]
        
        self.assertIn("input_ids", item)
        self.assertIn("attention_mask", item)
        self.assertIn("labels", item)
        
        # Check tokenizer was called
        self.mock_tokenizer.assert_called()

class TestTrainerIntegration(unittest.TestCase):
    """Test trainer integration functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create sample dataset
        self.sample_data = create_sample_dataset(
            task_type="classification",
            num_samples=10,
            num_classes=2
        )
        
        # Create configs
        self.model_config = ModelConfig(
            model_name="distilbert-base-uncased",
            num_labels=2
        )
        
        self.training_config = TrainingConfig(
            output_dir=self.temp_dir,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            learning_rate=5e-5,
            save_steps=5,
            eval_steps=5,
            logging_steps=1
        )
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('transformers.AutoTokenizer.from_pretrained')
    @patch('transformers.AutoModelForSequenceClassification.from_pretrained')
    def test_create_trainer(self, mock_model, mock_tokenizer):
        """Test create_trainer function."""
        # Mock tokenizer
        mock_tokenizer_instance = Mock()
        mock_tokenizer_instance.pad_token_id = 0
        mock_tokenizer_instance.return_value = {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1])
        }
        mock_tokenizer.return_value = mock_tokenizer_instance
        
        # Mock model
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        # Create trainer
        trainer = create_trainer(
            model_config=self.model_config,
            training_config=self.training_config,
            train_data=self.sample_data[:8],
            eval_data=self.sample_data[8:]
        )
        
        self.assertIsInstance(trainer, TraeOptimizedTrainer)
        
        # Check that model and tokenizer were loaded
        mock_model.assert_called_once()
        mock_tokenizer.assert_called_once()
    
    def test_training_config_validation(self):
        """Test training configuration validation."""
        # Test invalid learning rate
        with self.assertRaises(ValueError):
            TrainingConfig(
                output_dir=self.temp_dir,
                learning_rate=-1.0  # Invalid negative learning rate
            )
        
        # Test invalid batch size
        with self.assertRaises(ValueError):
            TrainingConfig(
                output_dir=self.temp_dir,
                per_device_train_batch_size=0  # Invalid zero batch size
            )
    
    def test_model_config_validation(self):
        """Test model configuration validation."""
        # Test valid config
        config = ModelConfig(
            model_name="bert-base-uncased",
            num_labels=3
        )
        self.assertEqual(config.model_name, "bert-base-uncased")
        self.assertEqual(config.num_labels, 3)
        
        # Test invalid num_labels
        with self.assertRaises(ValueError):
            ModelConfig(
                model_name="bert-base-uncased",
                num_labels=0  # Invalid zero labels
            )

class TestTraeOptimizedTrainer(unittest.TestCase):
    """Test TraeOptimizedTrainer specific functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Mock components
        self.mock_model = Mock()
        self.mock_tokenizer = Mock()
        self.mock_train_dataset = Mock()
        self.mock_eval_dataset = Mock()
        
        # Mock training args
        self.mock_args = Mock()
        self.mock_args.output_dir = self.temp_dir
        self.mock_args.gradient_checkpointing = True
        self.mock_args.fp16 = True
        self.mock_args.dataloader_pin_memory = True
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('transformers.Trainer.__init__')
    def test_trainer_initialization(self, mock_super_init):
        """Test TraeOptimizedTrainer initialization."""
        mock_super_init.return_value = None
        
        trainer = TraeOptimizedTrainer(
            model=self.mock_model,
            args=self.mock_args,
            train_dataset=self.mock_train_dataset,
            eval_dataset=self.mock_eval_dataset,
            tokenizer=self.mock_tokenizer
        )
        
        # Check that parent __init__ was called
        mock_super_init.assert_called_once()
        
        # Check Trae-specific attributes
        self.assertIsInstance(trainer.training_metrics, TrainingMetrics)
        self.assertTrue(len(trainer.callback_handler.callbacks) > 0)
    
    def test_compute_metrics(self):
        """Test compute_metrics method."""
        trainer = TraeOptimizedTrainer(
            model=self.mock_model,
            args=self.mock_args,
            train_dataset=self.mock_train_dataset,
            eval_dataset=self.mock_eval_dataset,
            tokenizer=self.mock_tokenizer
        )
        
        # Mock evaluation predictions
        mock_eval_pred = Mock()
        mock_eval_pred.predictions = torch.tensor([[0.8, 0.2], [0.3, 0.7]])
        mock_eval_pred.label_ids = torch.tensor([0, 1])
        
        metrics = trainer.compute_metrics(mock_eval_pred)
        
        self.assertIn("accuracy", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        
        # Check accuracy calculation
        self.assertEqual(metrics["accuracy"], 1.0)  # Both predictions correct

if __name__ == "__main__":
    # Set up test environment
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Run tests
    unittest.main(verbosity=2)