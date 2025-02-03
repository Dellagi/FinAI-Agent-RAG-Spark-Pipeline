from typing import List, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import pickle
import json
from datetime import datetime
import asyncio
from rank_bm25 import BM25Okapi
import torch
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize
import nltk
from chromadb.api import Collection
import openai
import logging
from ..config import Config

@dataclass
class Document:
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    bm25_tokens: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "metadata": self.metadata,
            "embedding": self.embedding.tolist() if self.embedding is not None else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        return cls(
            content=data["content"],
            metadata=data["metadata"],
            embedding=np.array(data["embedding"]) if data["embedding"] else None
        )

class RAGSystem:
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize NLTK
        try:
            nltk.download('punkt', quiet=True)
        except Exception as e:
            logging.warning(f"Failed to download NLTK data: {e}")
        
        # Initialize cross-encoder for reranking
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
            self.model = AutoModel.from_pretrained('cross-encoder/ms-marco-MiniLM-L-6-v2')
        except Exception as e:
            logging.error(f"Failed to load reranking model: {e}")
            raise
        
        # Initialize BM25
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Document] = []
        
        # Cache for preprocessed documents
        self.processed_docs_cache = {}
        
    def _preprocess_text(self, text: str) -> List[str]:
        """Tokenize and preprocess text for BM25"""
        try:
            tokens = word_tokenize(text.lower())
            return tokens
        except Exception as e:
            logging.error(f"Error preprocessing text: {e}")
            return text.lower().split()
        
    def _compute_cross_encoder_score(self, query: str, doc: str) -> float:
        """Compute relevance score using cross-encoder"""
        try:
            inputs = self.tokenizer(
                [query, doc],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings[0].unsqueeze(0),
                    embeddings[1].unsqueeze(0)
                )
                
            return similarity.item()
        except Exception as e:
            logging.error(f"Error computing cross-encoder score: {e}")
            return 0.0
        
    async def _get_semantic_embedding(self, text: str) -> np.ndarray:
        """Get embedding using OpenAI's text-embedding-ada-002"""
        try:
            response = await openai.Embedding.create(
                input=text,
                model="text-embedding-ada-002",
                api_key=self.config.OPENAI_API_KEY
            )
            return np.array(response['data'][0]['embedding'])
        except Exception as e:
            logging.error(f"Error getting semantic embedding: {e}")
            return np.zeros(1536)  # Default embedding dimension for ada-002
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the RAG system"""
        try:
            # Preprocess documents for BM25
            tokenized_docs = []
            for doc in documents:
                if doc.content not in self.processed_docs_cache:
                    tokens = self._preprocess_text(doc.content)
                    self.processed_docs_cache[doc.content] = tokens
                    doc.bm25_tokens = tokens
                else:
                    doc.bm25_tokens = self.processed_docs_cache[doc.content]
                tokenized_docs.append(doc.bm25_tokens)
                
            # Initialize or update BM25
            self.bm25 = BM25Okapi(tokenized_docs)
            self.documents.extend(documents)
            
            logging.info(f"Added {len(documents)} documents to RAG system")
            
        except Exception as e:
            logging.error(f"Error adding documents to RAG system: {e}")
            raise
        
    async def retrieve(
        self,
        query: str,
        k: int = 5,
        min_relevance_score: float = 0.5,
        rerank: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search
        """
        try:
            # Tokenize query
            query_tokens = self._preprocess_text(query)
            
            # Get BM25 scores
            if not self.bm25:
                logging.warning("BM25 index not initialized")
                return []
                
            bm25_scores = self.bm25.get_scores(query_tokens)
            
            # Get query embedding
            query_embedding = await self._get_semantic_embedding(query)
            
            # Calculate semantic similarity scores
            semantic_scores = []
            for doc in self.documents:
                if doc.embedding is None:
                    doc.embedding = await self._get_semantic_embedding(doc.content)
                semantic_score = np.dot(query_embedding, doc.embedding)
                semantic_scores.append(semantic_score)
                
            # Combine scores with weighted average
            combined_scores = []
            for bm25_score, semantic_score in zip(bm25_scores, semantic_scores):
                # Normalize scores
                norm_bm25 = bm25_score / max(bm25_scores) if max(bm25_scores) > 0 else 0
                norm_semantic = semantic_score / max(semantic_scores) if max(semantic_scores) > 0 else 0
                
                # Weighted combination (adjust weights as needed)
                combined_score = 0.4 * norm_bm25 + 0.6 * norm_semantic
                combined_scores.append(combined_score)
                
            # Get top k documents
            top_k_indices = np.argsort(combined_scores)[-k:][::-1]
            
            results = []
            for idx in top_k_indices:
                if combined_scores[idx] >= min_relevance_score:
                    doc = self.documents[idx]
                    result = {
                        "content": doc.content,
                        "metadata": doc.metadata,
                        "relevance_score": combined_scores[idx]
                    }
                    
                    if rerank:
                        # Calculate cross-encoder score for reranking
                        cross_encoder_score = self._compute_cross_encoder_score(
                            query, doc.content
                        )
                        result["reranking_score"] = cross_encoder_score
                        
                    results.append(result)
                    
            if rerank:
                # Sort by reranking score
                results.sort(key=lambda x: x["reranking_score"], reverse=True)
                
            return results
            
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}")
            return []
        
    def save_index(self, file_path: str):
        """Save the RAG system state"""
        try:
            state = {
                "documents": [doc.to_dict() for doc in self.documents],
                "processed_docs_cache": self.processed_docs_cache
            }
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
            logging.info(f"RAG system state saved to {file_path}")
            
        except Exception as e:
            logging.error(f"Error saving RAG system state: {e}")
            raise
            
    def load_index(self, file_path: str):
        """Load the RAG system state"""
        try:
            with open(file_path, 'rb') as f:
                state = pickle.load(f)
                
            self.documents = [Document.from_dict(doc_dict) for doc_dict in state["documents"]]
            self.processed_docs_cache = state["processed_docs_cache"]
            
            # Rebuild BM25 index
            tokenized_docs = [doc.bm25_tokens for doc in self.documents]
            self.bm25 = BM25Okapi(tokenized_docs)
            
            logging.info(f"RAG system state loaded from {file_path}")
            
        except Exception as e:
            logging.error(f"Error loading RAG system state: {e}")
            raise

class ContextualRAGRetriever:
    def __init__(self, config: Config, chroma_collection: Collection):
        self.config = config
        self.collection = chroma_collection
        self.rag_system = RAGSystem(config)
        
    async def _prepare_market_context(self) -> str:
        """Prepare current market context"""
        try:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            market_hours = "Normal Trading Hours" if 9 <= datetime.now().hour <= 16 else "Extended Hours"
            
            context = f"""
            Market Context:
            - Current Time: {current_time}
            - Trading Session: {market_hours}
            """
            
            return context
            
        except Exception as e:
            logging.error(f"Error preparing market context: {e}")
            return f"Current time: {datetime.now().isoformat()}"
        
    async def retrieve_with_context(
        self,
        query: str,
        symbol: str = None,
        k: int = 5,
        time_window: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """
        Retrieve documents with market context
        """
        try:
            # Get current market context
            market_context = await self._prepare_market_context()
            
            # Enhance query with context
            enhanced_query = f"""
            Context: {market_context}
            Symbol: {symbol if symbol else 'N/A'}
            Query: {query}
            """
            
            # Apply time window filter if provided
            if time_window:
                start, end = time_window
                # Add time context to query
                enhanced_query += f"\nTime period: {start} to {end}"
                
            # Retrieve relevant documents
            results = await self.rag_system.retrieve(
                query=enhanced_query,
                k=k,
                rerank=True
            )
            
            # Organize results
            return {
                "query_context": {
                    "original_query": query,
                    "enhanced_query": enhanced_query,
                    "market_context": market_context,
                    "symbol": symbol,
                    "time_window": {
                        "start": start.isoformat() if time_window else None,
                        "end": end.isoformat() if time_window else None
                    }
                },
                "results": results
            }
            
        except Exception as e:
            logging.error(f"Error retrieving with context: {e}")
            return {
                "query_context": {"error": str(e)},
                "results": []
            }
        
    async def update_from_chroma(self):
        """Update RAG system with documents from ChromaDB"""
        try:
            # Get all documents from ChromaDB
            results = self.collection.get()
            
            documents = []
            for i, doc in enumerate(results['documents']):
                document = Document(
                    content=doc,
                    metadata=results['metadatas'][i],
                    embedding=np.array(results['embeddings'][i]) if results['embeddings'] else None
                )
                documents.append(document)
                
            # Update RAG system
            self.rag_system.add_documents(documents)
            
            logging.info(f"Updated RAG system with {len(documents)} documents from ChromaDB")
            
        except Exception as e:
            logging.error(f"Error updating from ChromaDB: {e}")
            raise