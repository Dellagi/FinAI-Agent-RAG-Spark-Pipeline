
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
            CHROMA_PERSIST_DIR=os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        )

