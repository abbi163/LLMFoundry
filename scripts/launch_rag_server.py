#!/usr/bin/env python3
"""CLI script for launching a RAG API server.

This script provides a FastAPI-based REST API server for the RAG pipeline,
allowing users to query the knowledge base via HTTP requests.
"""

import argparse
import logging
import sys
from pathlib import Path
import json
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from trae_llm.config import RAGConfig, ModelConfig
from trae_llm.rag_pipeline import RAGPipeline, create_rag_pipeline, Document, DocumentProcessor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None

# Pydantic models for API
class QueryRequest(BaseModel):
    """Request model for RAG queries."""
    question: str
    top_k: Optional[int] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = None

class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    question: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    retrieval_scores: List[float]
    total_time: float
    retrieval_time: float
    generation_time: float

class DocumentRequest(BaseModel):
    """Request model for adding documents."""
    content: str
    metadata: Optional[Dict[str, Any]] = None
    doc_id: Optional[str] = None

class DocumentResponse(BaseModel):
    """Response model for document operations."""
    message: str
    document_count: int

class StatsResponse(BaseModel):
    """Response model for pipeline statistics."""
    num_documents: int
    embedding_dim: int
    chunk_size: int
    top_k: int
    cache_size: int

# Create FastAPI app
app = FastAPI(
    title="Trae RAG API",
    description="REST API for Retrieval-Augmented Generation with Trae optimizations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline on startup."""
    logger.info("Starting RAG API server...")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Trae RAG API Server",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    return {
        "status": "healthy",
        "pipeline_loaded": True,
        "stats": rag_pipeline.get_stats()
    }

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    """Query the RAG pipeline."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Override config parameters if provided
        original_config = rag_pipeline.config
        if request.top_k is not None:
            rag_pipeline.config.top_k = request.top_k
        if request.max_new_tokens is not None:
            rag_pipeline.config.max_new_tokens = request.max_new_tokens
        if request.temperature is not None:
            rag_pipeline.config.temperature = request.temperature
        
        # Query the pipeline
        result = rag_pipeline.query(request.question)
        
        # Restore original config
        rag_pipeline.config = original_config
        
        # Convert documents to serializable format
        retrieved_docs = []
        for doc in result.retrieved_documents:
            retrieved_docs.append({
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata
            })
        
        return QueryResponse(
            question=result.query,
            answer=result.generated_response,
            retrieved_documents=retrieved_docs,
            retrieval_scores=result.retrieval_scores,
            total_time=result.total_time,
            retrieval_time=result.retrieval_time,
            generation_time=result.generation_time
        )
    
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents", response_model=DocumentResponse)
async def add_document(request: DocumentRequest):
    """Add a document to the knowledge base."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Create document
        doc_id = request.doc_id or f"doc_{len(rag_pipeline.vector_store.documents)}"
        metadata = request.metadata or {}
        
        # Create documents from text (with chunking)
        documents = DocumentProcessor.create_documents_from_text(
            text=request.content,
            source_id=doc_id,
            chunk_size=rag_pipeline.config.chunk_size,
            overlap=rag_pipeline.config.chunk_overlap,
            metadata=metadata
        )
        
        # Add to pipeline
        rag_pipeline.add_documents(documents)
        
        return DocumentResponse(
            message=f"Added {len(documents)} document chunks",
            document_count=len(rag_pipeline.vector_store.documents)
        )
    
    except Exception as e:
        logger.error(f"Failed to add document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/upload", response_model=DocumentResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and add a document file to the knowledge base."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Load documents from file
            documents = DocumentProcessor.load_documents_from_file(
                tmp_file_path,
                chunk_size=rag_pipeline.config.chunk_size,
                overlap=rag_pipeline.config.chunk_overlap
            )
            
            # Add to pipeline
            rag_pipeline.add_documents(documents)
            
            return DocumentResponse(
                message=f"Uploaded and processed {file.filename}: {len(documents)} chunks",
                document_count=len(rag_pipeline.vector_store.documents)
            )
        
        finally:
            # Clean up temporary file
            os.unlink(tmp_file_path)
    
    except Exception as e:
        logger.error(f"Failed to upload document: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/stats", response_model=StatsResponse)
async def get_stats():
    """Get pipeline statistics."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    stats = rag_pipeline.get_stats()
    return StatsResponse(**stats)

@app.post("/documents/save")
async def save_knowledge_base(path: str = "./knowledge_base"):
    """Save the knowledge base to disk."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        rag_pipeline.save_knowledge_base(path)
        return {"message": f"Knowledge base saved to {path}"}
    
    except Exception as e:
        logger.error(f"Failed to save knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents/load")
async def load_knowledge_base(path: str):
    """Load the knowledge base from disk."""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        rag_pipeline.load_knowledge_base(path)
        stats = rag_pipeline.get_stats()
        return {
            "message": f"Knowledge base loaded from {path}",
            "document_count": stats["num_documents"]
        }
    
    except Exception as e:
        logger.error(f"Failed to load knowledge base: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch RAG API server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the language model"
    )
    
    parser.add_argument(
        "--model_config",
        type=str,
        help="Path to model configuration JSON file"
    )
    
    parser.add_argument(
        "--rag_config",
        type=str,
        help="Path to RAG configuration JSON file"
    )
    
    # Server arguments
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # Knowledge base arguments
    parser.add_argument(
        "--knowledge_base_path",
        type=str,
        help="Path to existing knowledge base to load"
    )
    
    parser.add_argument(
        "--documents_dir",
        type=str,
        help="Directory containing documents to load on startup"
    )
    
    # RAG parameters
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=512,
        help="Document chunk size"
    )
    
    parser.add_argument(
        "--chunk_overlap",
        type=int,
        default=50,
        help="Document chunk overlap"
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="Number of documents to retrieve"
    )
    
    parser.add_argument(
        "--embedding_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name"
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

def initialize_rag_pipeline(args):
    """Initialize the RAG pipeline."""
    global rag_pipeline
    
    # Load configurations
    model_config = None
    if args.model_config:
        model_config = load_config_from_file(args.model_config, ModelConfig)
    
    rag_config = None
    if args.rag_config:
        rag_config = load_config_from_file(args.rag_config, RAGConfig)
    else:
        # Create config from arguments
        rag_config = RAGConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            top_k=args.top_k,
            embedding_model=args.embedding_model
        )
    
    # Create RAG pipeline
    logger.info("Initializing RAG pipeline...")
    rag_pipeline = create_rag_pipeline(
        model_path=args.model_path,
        config=rag_config,
        model_config=model_config
    )
    
    # Load existing knowledge base if specified
    if args.knowledge_base_path:
        logger.info(f"Loading knowledge base from {args.knowledge_base_path}")
        rag_pipeline.load_knowledge_base(args.knowledge_base_path)
    
    # Load documents from directory if specified
    if args.documents_dir:
        logger.info(f"Loading documents from {args.documents_dir}")
        documents_dir = Path(args.documents_dir)
        if documents_dir.exists():
            file_paths = []
            for ext in ['*.txt', '*.json']:
                file_paths.extend(documents_dir.glob(ext))
            
            if file_paths:
                rag_pipeline.add_documents_from_files([str(p) for p in file_paths])
                logger.info(f"Loaded documents from {len(file_paths)} files")
            else:
                logger.warning(f"No supported files found in {documents_dir}")
        else:
            logger.warning(f"Documents directory {documents_dir} does not exist")
    
    logger.info("RAG pipeline initialized successfully!")
    stats = rag_pipeline.get_stats()
    logger.info(f"Knowledge base contains {stats['num_documents']} documents")

def main():
    """Main function to launch the RAG server."""
    args = parse_arguments()
    
    logger.info("Starting Trae RAG API Server...")
    logger.info(f"Model: {args.model_path}")
    logger.info(f"Host: {args.host}:{args.port}")
    
    try:
        # Initialize RAG pipeline
        initialize_rag_pipeline(args)
        
        # Launch server
        logger.info(f"Launching server on {args.host}:{args.port}")
        uvicorn.run(
            "launch_rag_server:app",
            host=args.host,
            port=args.port,
            workers=args.workers,
            reload=args.reload,
            log_level="info"
        )
    
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    main()