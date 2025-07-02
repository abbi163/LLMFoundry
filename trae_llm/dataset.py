"""Dataset loaders and preprocessing utilities.

This module provides classes and functions for loading, preprocessing,
and managing datasets for LLM training and inference.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import pandas as pd
from datasets import load_dataset, Dataset as HFDataset
import logging

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for text data."""
    
    def __init__(self, 
                 texts: List[str],
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 labels: Optional[List[str]] = None):
        """
        Initialize TextDataset.
        
        Args:
            texts: List of input texts
            tokenizer: Tokenizer for encoding texts
            max_length: Maximum sequence length
            labels: Optional labels for supervised learning
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.labels = labels
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze()
        }
        
        # Add labels if available
        if self.labels is not None:
            label = self.labels[idx]
            if isinstance(label, str):
                # For text classification, encode label
                label_encoding = self.tokenizer(
                    label,
                    truncation=True,
                    padding='max_length',
                    max_length=self.max_length,
                    return_tensors='pt'
                )
                item['labels'] = label_encoding['input_ids'].squeeze()
            else:
                # For numeric labels
                item['labels'] = torch.tensor(label, dtype=torch.long)
        
        return item


class ConversationDataset(Dataset):
    """Dataset for conversation/chat data."""
    
    def __init__(self,
                 conversations: List[List[Dict[str, str]]],
                 tokenizer: AutoTokenizer,
                 max_length: int = 512,
                 system_message: str = "You are a helpful assistant."):
        """
        Initialize ConversationDataset.
        
        Args:
            conversations: List of conversations, each conversation is a list of messages
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            system_message: System message to prepend
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.system_message = system_message
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def __len__(self) -> int:
        return len(self.conversations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        conversation = self.conversations[idx]
        
        # Format conversation
        formatted_text = self._format_conversation(conversation)
        
        # Tokenize
        encoding = self.tokenizer(
            formatted_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()  # For causal LM
        }
    
    def _format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format conversation into a single string."""
        formatted = f"System: {self.system_message}\n"
        
        for message in conversation:
            role = message.get('role', 'user')
            content = message.get('content', '')
            formatted += f"{role.capitalize()}: {content}\n"
        
        return formatted.strip()


class DatasetLoader:
    """Utility class for loading various dataset formats."""
    
    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer
    
    def load_from_json(self, 
                      file_path: str,
                      text_column: str = 'text',
                      label_column: Optional[str] = None,
                      max_length: int = 512) -> TextDataset:
        """Load dataset from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = [item[text_column] for item in data]
            labels = [item[label_column] for item in data] if label_column else None
        else:
            texts = data[text_column]
            labels = data[label_column] if label_column else None
        
        return TextDataset(texts, self.tokenizer, max_length, labels)
    
    def load_from_csv(self,
                     file_path: str,
                     text_column: str = 'text',
                     label_column: Optional[str] = None,
                     max_length: int = 512) -> TextDataset:
        """Load dataset from CSV file."""
        df = pd.read_csv(file_path)
        
        texts = df[text_column].tolist()
        labels = df[label_column].tolist() if label_column else None
        
        return TextDataset(texts, self.tokenizer, max_length, labels)
    
    def load_from_huggingface(self,
                             dataset_name: str,
                             split: str = 'train',
                             text_column: str = 'text',
                             label_column: Optional[str] = None,
                             max_length: int = 512,
                             max_samples: Optional[int] = None) -> TextDataset:
        """Load dataset from HuggingFace datasets."""
        dataset = load_dataset(dataset_name, split=split)
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))
        
        texts = dataset[text_column]
        labels = dataset[label_column] if label_column else None
        
        return TextDataset(texts, self.tokenizer, max_length, labels)
    
    def load_conversation_data(self,
                              file_path: str,
                              max_length: int = 512,
                              system_message: str = "You are a helpful assistant.") -> ConversationDataset:
        """Load conversation data from JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
        
        return ConversationDataset(
            conversations, 
            self.tokenizer, 
            max_length, 
            system_message
        )
    
    def create_dataloader(self,
                         dataset: Dataset,
                         batch_size: int = 8,
                         shuffle: bool = True,
                         num_workers: int = 4,
                         pin_memory: bool = True) -> DataLoader:
        """Create DataLoader from dataset."""
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn
        )
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for batching."""
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            collated[key] = torch.stack([item[key] for item in batch])
        
        return collated


class DataPreprocessor:
    """Utility class for data preprocessing."""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Basic text cleaning."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters (optional)
        # text = re.sub(r'[^\w\s]', '', text)
        
        return text.strip()
    
    @staticmethod
    def split_long_text(text: str, 
                       max_length: int = 512, 
                       overlap: int = 50) -> List[str]:
        """Split long text into chunks with overlap."""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), max_length - overlap):
            chunk = ' '.join(words[i:i + max_length])
            chunks.append(chunk)
            
            if i + max_length >= len(words):
                break
        
        return chunks
    
    @staticmethod
    def balance_dataset(texts: List[str], 
                       labels: List[Any],
                       max_samples_per_class: Optional[int] = None) -> tuple[List[str], List[Any]]:
        """Balance dataset by sampling equal numbers from each class."""
        df = pd.DataFrame({'text': texts, 'label': labels})
        
        if max_samples_per_class:
            df = df.groupby('label').apply(
                lambda x: x.sample(min(len(x), max_samples_per_class))
            ).reset_index(drop=True)
        else:
            # Sample minimum class size from all classes
            min_size = df['label'].value_counts().min()
            df = df.groupby('label').apply(
                lambda x: x.sample(min_size)
            ).reset_index(drop=True)
        
        return df['text'].tolist(), df['label'].tolist()
    
    @staticmethod
    def create_train_val_split(dataset: Dataset, 
                              val_ratio: float = 0.2,
                              random_seed: int = 42) -> tuple[Dataset, Dataset]:
        """Split dataset into train and validation sets."""
        torch.manual_seed(random_seed)
        
        dataset_size = len(dataset)
        val_size = int(dataset_size * val_ratio)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        return train_dataset, val_dataset


def create_sample_dataset(tokenizer: AutoTokenizer, 
                         num_samples: int = 100,
                         max_length: int = 512) -> TextDataset:
    """Create a sample dataset for testing."""
    import random
    
    # Sample texts
    sample_texts = [
        "This is a sample text for training.",
        "Machine learning is fascinating.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning models can generate human-like text.",
        "Transformers have revolutionized NLP."
    ]
    
    texts = [random.choice(sample_texts) for _ in range(num_samples)]
    
    return TextDataset(texts, tokenizer, max_length)


if __name__ == "__main__":
    # Example usage
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    loader = DatasetLoader(tokenizer)
    
    # Create sample dataset
    dataset = create_sample_dataset(tokenizer)
    dataloader = loader.create_dataloader(dataset, batch_size=4)
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Test one batch
    for batch in dataloader:
        print(f"Batch keys: {batch.keys()}")
        print(f"Input shape: {batch['input_ids'].shape}")
        break