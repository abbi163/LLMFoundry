"""Trae-based RAG (Retrieval-Augmented Generation) pipeline.

This module provides a complete RAG implementation with Trae AI optimizations,
including document processing, vector storage, retrieval, and generation.
"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM

from .config import RAGConfig, ModelConfig
from .inference import InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class Document:
    """Container for document data."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'embedding': self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Document':
        embedding = np.array(data['embedding']) if data.get('embedding') else None
        return cls(
            id=data['id'],
            content=data['content'],
            metadata=data['metadata'],
            embedding=embedding
        )


@dataclass
class RetrievalResult:
    """Container for retrieval results."""
    documents: List[Document]
    scores: List[float]
    query: str
    retrieval_time: float
    
    def get_context(self, max_length: int = 2048) -> str:
        """Combine retrieved documents into context string."""
        context_parts = []
        current_length = 0
        
        for doc in self.documents:
            doc_text = f"Document: {doc.content}\n"
            if current_length + len(doc_text) <= max_length:
                context_parts.append(doc_text)
                current_length += len(doc_text)
            else:
                break
        
        return "\n".join(context_parts)


@dataclass
class RAGResult:
    """Container for RAG pipeline results."""
    query: str
    retrieved_documents: List[Document]
    retrieval_scores: List[float]
    context: str
    generated_response: str
    total_time: float
    retrieval_time: float
    generation_time: float


class DocumentProcessor:
    """Utility class for processing documents."""
    
    @staticmethod
    def chunk_text(text: str, 
                  chunk_size: int = 512, 
                  overlap: int = 50,
                  separator: str = "\n") -> List[str]:
        """Split text into overlapping chunks."""
        
        # Split by separator first
        paragraphs = text.split(separator)
        
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            # If adding this paragraph exceeds chunk size, save current chunk
            if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                words = current_chunk.split()
                overlap_words = words[-overlap:] if len(words) > overlap else words
                current_chunk = " ".join(overlap_words) + " " + paragraph
            else:
                current_chunk += (separator if current_chunk else "") + paragraph
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def create_documents_from_text(text: str,
                                  source_id: str,
                                  chunk_size: int = 512,
                                  overlap: int = 50,
                                  metadata: Optional[Dict[str, Any]] = None) -> List[Document]:
        """Create documents from text by chunking."""
        
        chunks = DocumentProcessor.chunk_text(text, chunk_size, overlap)
        documents = []
        
        base_metadata = metadata or {}
        
        for i, chunk in enumerate(chunks):
            doc_id = f"{source_id}_chunk_{i}"
            doc_metadata = {
                **base_metadata,
                'source_id': source_id,
                'chunk_index': i,
                'chunk_count': len(chunks)
            }
            
            documents.append(Document(
                id=doc_id,
                content=chunk,
                metadata=doc_metadata
            ))
        
        return documents
    
    @staticmethod
    def load_documents_from_file(file_path: str,
                               chunk_size: int = 512,
                               overlap: int = 50) -> List[Document]:
        """Load documents from various file formats."""
        
        file_path = Path(file_path)
        
        if file_path.suffix == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return DocumentProcessor.create_documents_from_text(
                text, file_path.stem, chunk_size, overlap,
                {'file_path': str(file_path), 'file_type': 'txt'}
            )
        
        elif file_path.suffix == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            documents = []
            if isinstance(data, list):
                for i, item in enumerate(data):
                    if isinstance(item, dict) and 'content' in item:
                        doc_id = item.get('id', f"{file_path.stem}_{i}")
                        metadata = {k: v for k, v in item.items() if k not in ['id', 'content']}
                        metadata.update({'file_path': str(file_path), 'file_type': 'json'})
                        
                        documents.append(Document(
                            id=doc_id,
                            content=item['content'],
                            metadata=metadata
                        ))
            
            return documents
        
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")


class VectorStore:
    """Vector storage and retrieval using FAISS."""
    
    def __init__(self, 
                 embedding_dim: int,
                 index_type: str = "flat"):
        """
        Initialize VectorStore.
        
        Args:
            embedding_dim: Dimension of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
        """
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.documents: List[Document] = []
        self.index = None
        
        self._create_index()
    
    def _create_index(self):
        """Create FAISS index."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        elif self.index_type == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.embedding_dim, 32)
        else:
            raise ValueError(f"Unsupported index type: {self.index_type}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store."""
        if not documents:
            return
        
        # Ensure all documents have embeddings
        embeddings = []
        for doc in documents:
            if doc.embedding is None:
                raise ValueError(f"Document {doc.id} has no embedding")
            embeddings.append(doc.embedding)
        
        # Normalize embeddings for cosine similarity
        embeddings = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings)
        
        # Add to index
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings)
        
        self.index.add(embeddings)
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents to vector store")
    
    def search(self, 
              query_embedding: np.ndarray,
              top_k: int = 5,
              score_threshold: float = 0.0) -> Tuple[List[Document], List[float]]:
        """Search for similar documents."""
        if self.index.ntotal == 0:
            return [], []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, min(top_k, self.index.ntotal))
        
        # Filter by score threshold
        results = []
        result_scores = []
        
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and score >= score_threshold:
                results.append(self.documents[idx])
                result_scores.append(float(score))
        
        return results, result_scores
    
    def save(self, path: str):
        """Save vector store to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents
        documents_data = [doc.to_dict() for doc in self.documents]
        with open(path / "documents.json", 'w') as f:
            json.dump(documents_data, f, indent=2)
        
        # Save metadata
        metadata = {
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type,
            'num_documents': len(self.documents)
        }
        with open(path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str):
        """Load vector store from disk."""
        path = Path(path)
        
        # Load metadata
        with open(path / "metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.embedding_dim = metadata['embedding_dim']
        self.index_type = metadata['index_type']
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents
        with open(path / "documents.json", 'r') as f:
            documents_data = json.load(f)
        
        self.documents = [Document.from_dict(doc_data) for doc_data in documents_data]
        
        logger.info(f"Vector store loaded from {path}")


class RAGPipeline:
    """Complete RAG pipeline with Trae optimizations."""
    
    def __init__(self,
                 model_path: str,
                 config: Optional[RAGConfig] = None,
                 model_config: Optional[ModelConfig] = None):
        """
        Initialize RAG pipeline.
        
        Args:
            model_path: Path to the language model
            config: RAG configuration
            model_config: Model configuration
        """
        self.config = config or RAGConfig()
        self.model_config = model_config or ModelConfig()
        
        # Initialize components
        self._initialize_embedding_model()
        self._initialize_vector_store()
        self._initialize_generation_model(model_path)
        
        # Cache for responses (if enabled)
        self.response_cache = {} if self.config.enable_response_caching else None
        
        logger.info("RAG Pipeline initialized successfully")
    
    def _initialize_embedding_model(self):
        """Initialize embedding model."""
        self.embedding_model = SentenceTransformer(self.config.embedding_model)
        logger.info(f"Embedding model loaded: {self.config.embedding_model}")
    
    def _initialize_vector_store(self):
        """Initialize vector store."""
        self.vector_store = VectorStore(
            embedding_dim=self.config.embedding_dim,
            index_type="flat"  # Can be made configurable
        )
        logger.info("Vector store initialized")
    
    def _initialize_generation_model(self, model_path: str):
        """Initialize generation model."""
        self.inference_engine = InferenceEngine(
            model_path=model_path,
            model_config=self.model_config
        )
        logger.info(f"Generation model loaded: {model_path}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to the knowledge base."""
        # Generate embeddings for documents
        contents = [doc.content for doc in documents]
        embeddings = self.embedding_model.encode(contents, convert_to_numpy=True)
        
        # Add embeddings to documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        # Add to vector store
        self.vector_store.add_documents(documents)
        
        logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def add_documents_from_files(self, file_paths: List[str]):
        """Add documents from files."""
        all_documents = []
        
        for file_path in file_paths:
            try:
                documents = DocumentProcessor.load_documents_from_file(
                    file_path,
                    chunk_size=self.config.chunk_size,
                    overlap=self.config.chunk_overlap
                )
                all_documents.extend(documents)
                logger.info(f"Loaded {len(documents)} documents from {file_path}")
            except Exception as e:
                logger.error(f"Failed to load documents from {file_path}: {e}")
        
        if all_documents:
            self.add_documents(all_documents)
    
    def retrieve(self, query: str) -> RetrievalResult:
        """Retrieve relevant documents for a query."""
        import time
        start_time = time.time()
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)[0]
        
        # Search vector store
        documents, scores = self.vector_store.search(
            query_embedding,
            top_k=self.config.top_k,
            score_threshold=self.config.similarity_threshold
        )
        
        retrieval_time = time.time() - start_time
        
        return RetrievalResult(
            documents=documents,
            scores=scores,
            query=query,
            retrieval_time=retrieval_time
        )
    
    def generate_response(self, query: str, context: str) -> str:
        """Generate response using retrieved context."""
        # Create prompt with context
        prompt = self._create_prompt(query, context)
        
        # Generate response
        result = self.inference_engine.generate_text(
            prompt,
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=self.config.do_sample
        )
        
        return result.text
    
    def _create_prompt(self, query: str, context: str) -> str:
        """Create prompt for generation."""
        prompt = f"""Context:
{context}

Question: {query}

Answer based on the provided context:"""
        
        return prompt
    
    def query(self, question: str) -> RAGResult:
        """Complete RAG pipeline query."""
        import time
        start_time = time.time()
        
        # Check cache if enabled
        if self.response_cache is not None:
            cache_key = hashlib.md5(question.encode()).hexdigest()
            if cache_key in self.response_cache:
                logger.info("Returning cached response")
                return self.response_cache[cache_key]
        
        # Retrieve relevant documents
        retrieval_result = self.retrieve(question)
        
        # Get context from retrieved documents
        context = retrieval_result.get_context(self.config.max_context_length)
        
        # Generate response
        generation_start = time.time()
        response = self.generate_response(question, context)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        # Create result
        result = RAGResult(
            query=question,
            retrieved_documents=retrieval_result.documents,
            retrieval_scores=retrieval_result.scores,
            context=context,
            generated_response=response,
            total_time=total_time,
            retrieval_time=retrieval_result.retrieval_time,
            generation_time=generation_time
        )
        
        # Cache result if enabled
        if self.response_cache is not None:
            self.response_cache[cache_key] = result
        
        return result
    
    def save_knowledge_base(self, path: str):
        """Save the knowledge base to disk."""
        self.vector_store.save(path)
    
    def load_knowledge_base(self, path: str):
        """Load the knowledge base from disk."""
        self.vector_store.load(path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'num_documents': len(self.vector_store.documents),
            'embedding_dim': self.config.embedding_dim,
            'chunk_size': self.config.chunk_size,
            'top_k': self.config.top_k,
            'cache_size': len(self.response_cache) if self.response_cache else 0
        }


def create_rag_pipeline(model_path: str,
                       config: Optional[RAGConfig] = None,
                       model_config: Optional[ModelConfig] = None) -> RAGPipeline:
    """Create a RAG pipeline with default settings."""
    
    if config is None:
        config = RAGConfig()
    
    if model_config is None:
        model_config = ModelConfig()
    
    return RAGPipeline(
        model_path=model_path,
        config=config,
        model_config=model_config
    )


if __name__ == "__main__":
    # Example usage
    model_path = "gpt2"
    
    # Create RAG pipeline
    rag = create_rag_pipeline(model_path)
    
    # Add sample documents
    sample_docs = [
        Document(
            id="doc1",
            content="Artificial intelligence is transforming various industries.",
            metadata={"topic": "AI"}
        ),
        Document(
            id="doc2",
            content="Machine learning algorithms can learn from data.",
            metadata={"topic": "ML"}
        )
    ]
    
    rag.add_documents(sample_docs)
    
    # Query the pipeline
    result = rag.query("What is artificial intelligence?")
    
    print(f"Query: {result.query}")
    print(f"Response: {result.generated_response}")
    print(f"Retrieved {len(result.retrieved_documents)} documents")
    print(f"Total time: {result.total_time:.2f}s")