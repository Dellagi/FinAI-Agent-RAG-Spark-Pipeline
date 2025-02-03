### config.py

from dataclasses import dataclass
from typing import List, Dict, Any
import os
from dotenv import load_dotenv

@dataclass
class Config:
    OPENAI_API_KEY: str
    GOOGLE_API_KEY: str
    GOOGLE_CSE_ID: str
    KAFKA_BROKERS: List[str]
    REDIS_URL: str
    ALPACA_API_KEY: str
    ALPACA_SECRET_KEY: str
    CHROMA_PERSIST_DIR: str
    RAG_INDEX_DIR: str
    RAG_MIN_RELEVANCE_SCORE: float
    RAG_UPDATE_INTERVAL_HOURS: int
    RAG_MAX_RESULTS: int
    
    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls(
            OPENAI_API_KEY=os.getenv("OPENAI_API_KEY"),
            GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY"),
            GOOGLE_CSE_ID=os.getenv("GOOGLE_CSE_ID"),
            KAFKA_BROKERS=os.getenv("KAFKA_BROKERS", "localhost:9092").split(","),
            REDIS_URL=os.getenv("REDIS_URL", "redis://localhost:6379"),
            ALPACA_API_KEY=os.getenv("ALPACA_API_KEY"),
            ALPACA_SECRET_KEY=os.getenv("ALPACA_SECRET_KEY"),
            CHROMA_PERSIST_DIR=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db"),
            RAG_INDEX_DIR=os.getenv("RAG_INDEX_DIR", "./rag_indices"),
            RAG_MIN_RELEVANCE_SCORE=float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.5")),
            RAG_UPDATE_INTERVAL_HOURS=int(os.getenv("RAG_UPDATE_INTERVAL_HOURS", "4")),
            RAG_MAX_RESULTS=int(os.getenv("RAG_MAX_RESULTS", "5"))
        )