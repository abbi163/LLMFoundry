# Data Directory

This directory contains sample datasets and data-related files for the Trae LLM Lab project.

## Directory Structure

```
data/
├── readme.md              <- This file
├── sample_datasets/       <- Sample training datasets
├── knowledge_base/        <- RAG knowledge base files
├── embeddings/           <- Cached embeddings
└── processed/            <- Processed datasets
```

## Sample Datasets

The following sample datasets are available for experimentation:

### 1. Text Classification
- **Format**: JSON/CSV
- **Use Case**: Sentiment analysis, topic classification
- **Structure**: `{"text": "sample text", "label": "category"}`

### 2. Text Generation
- **Format**: JSON/Text files
- **Use Case**: Language modeling, creative writing
- **Structure**: Plain text or `{"input": "prompt", "output": "completion"}`

### 3. Question Answering
- **Format**: JSON
- **Use Case**: Reading comprehension, FAQ systems
- **Structure**: `{"context": "passage", "question": "query", "answer": "response"}`

### 4. Conversation Data
- **Format**: JSON
- **Use Case**: Chatbot training, dialogue systems
- **Structure**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`

## Data Sources

You can obtain datasets from the following sources:

### Public Datasets
- **Hugging Face Datasets**: https://huggingface.co/datasets
- **Kaggle**: https://www.kaggle.com/datasets
- **Papers with Code**: https://paperswithcode.com/datasets
- **Google Dataset Search**: https://datasetsearch.research.google.com/

### Popular NLP Datasets
- **IMDB Movie Reviews**: Sentiment analysis
- **SQuAD**: Question answering
- **CoNLL-2003**: Named entity recognition
- **WMT**: Machine translation
- **Common Crawl**: Large-scale web text

## Data Preparation

### Loading Data

Use the `DatasetLoader` class from `trae_llm.dataset`:

```python
from trae_llm.dataset import DatasetLoader

# Load from JSON
dataset = DatasetLoader.load_json("data/sample_datasets/classification.json")

# Load from CSV
dataset = DatasetLoader.load_csv("data/sample_datasets/sentiment.csv")

# Load from Hugging Face
dataset = DatasetLoader.load_huggingface("imdb")
```

### Data Preprocessing

Use the `DataPreprocessor` class:

```python
from trae_llm.dataset import DataPreprocessor

preprocessor = DataPreprocessor()

# Clean text
cleaned_texts = preprocessor.clean_texts(raw_texts)

# Split dataset
train_data, val_data, test_data = preprocessor.split_dataset(
    dataset, train_ratio=0.8, val_ratio=0.1
)

# Balance dataset
balanced_data = preprocessor.balance_dataset(dataset, "label")
```

## Knowledge Base for RAG

For RAG (Retrieval-Augmented Generation) experiments:

### Document Formats
- **Text files** (`.txt`): Plain text documents
- **JSON files** (`.json`): Structured documents with metadata
- **PDF files** (`.pdf`): Academic papers, reports (requires `PyPDF2`)
- **Markdown files** (`.md`): Documentation, articles

### Example Knowledge Base Structure

```json
{
  "documents": [
    {
      "id": "doc_001",
      "title": "Introduction to Machine Learning",
      "content": "Machine learning is a subset of artificial intelligence...",
      "metadata": {
        "source": "textbook",
        "chapter": 1,
        "author": "John Doe"
      }
    }
  ]
}
```

### Adding Documents to RAG Pipeline

```python
from trae_llm.rag_pipeline import create_rag_pipeline

# Create RAG pipeline
rag_pipeline = create_rag_pipeline("microsoft/DialoGPT-medium")

# Add documents from directory
rag_pipeline.add_documents_from_files(["data/knowledge_base/docs.txt"])

# Add individual documents
from trae_llm.rag_pipeline import Document
doc = Document(
    id="custom_doc",
    content="Your document content here",
    metadata={"source": "manual"}
)
rag_pipeline.add_documents([doc])
```

## Data Privacy and Ethics

### Important Considerations
- **Personal Information**: Ensure datasets don't contain sensitive personal data
- **Bias**: Check for and mitigate biases in training data
- **Licensing**: Respect dataset licenses and usage terms
- **Data Quality**: Validate and clean data before training

### Best Practices
1. **Data Validation**: Always inspect your data before training
2. **Version Control**: Track dataset versions and changes
3. **Documentation**: Document data sources, preprocessing steps
4. **Backup**: Keep backups of important datasets
5. **Security**: Store sensitive data securely

## Creating Sample Data

You can create sample datasets using the provided utilities:

```python
from trae_llm.dataset import create_sample_dataset

# Create sample classification dataset
sample_data = create_sample_dataset(
    task_type="classification",
    num_samples=1000,
    num_classes=3
)

# Save to file
import json
with open("data/sample_datasets/sample_classification.json", "w") as f:
    json.dump(sample_data, f, indent=2)
```

## Troubleshooting

### Common Issues

1. **File Not Found**: Check file paths and permissions
2. **Memory Issues**: Use data streaming for large datasets
3. **Encoding Problems**: Specify encoding when loading text files
4. **Format Errors**: Validate JSON/CSV structure before loading

### Performance Tips

1. **Chunking**: Process large datasets in chunks
2. **Caching**: Cache preprocessed data to disk
3. **Parallel Processing**: Use multiprocessing for data loading
4. **Memory Management**: Clear unused variables and use generators

## Additional Resources

- [Hugging Face Datasets Documentation](https://huggingface.co/docs/datasets/)
- [Pandas Data Manipulation](https://pandas.pydata.org/docs/)
- [Data Science Best Practices](https://www.kaggle.com/learn/data-cleaning)
- [NLP Data Preprocessing Guide](https://towardsdatascience.com/nlp-data-preprocessing-a-complete-guide-6c4b0c8b2c8f)