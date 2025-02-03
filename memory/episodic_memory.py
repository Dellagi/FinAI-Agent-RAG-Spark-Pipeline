
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
import chromadb
from chromadb.config import Settings
import openai
import numpy as np
from enum import Enum

class MemoryType(Enum):
    TRADING_DECISION = "trading_decision"
    MARKET_EVENT = "market_event"
    RESEARCH_INSIGHT = "research_insight"
    PERFORMANCE_METRIC = "performance_metric"

@dataclass
class Episode:
    content: str
    memory_type: MemoryType
    timestamp: datetime
    metadata: Dict[str, Any]
    importance: float = 1.0
    context: Optional[Dict[str, Any]] = None
    emotions: Optional[List[str]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "memory_type": self.memory_type.value,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "importance": self.importance,
            "context": self.context,
            "emotions": self.emotions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Episode':
        return cls(
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=data["metadata"],
            importance=data.get("importance", 1.0),
            context=data.get("context"),
            emotions=data.get("emotions")
        )

class EpisodicMemoryStore:
    def __init__(self, config: Config):
        self.config = config
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=config.CHROMA_PERSIST_DIR
        ))
        
        # Create collections for different types of memories
        self.collections = {}
        for memory_type in MemoryType:
            self.collections[memory_type] = self.chroma_client.create_collection(
                name=f"trading_memory_{memory_type.value}",
                metadata={"memory_type": memory_type.value}
            )
    
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding using OpenAI's text-embedding-ada-002"""
        response = await openai.Embedding.create(
            input=text,
            model="text-embedding-ada-002",
            api_key=self.config.OPENAI_API_KEY
        )
        return response['data'][0]['embedding']
    
    async def _analyze_importance(self, episode: Episode) -> float:
        """Analyze the importance of an episode using GPT-4"""
        try:
            prompt = f"""
            Analyze the importance of this trading-related memory (0.0 to 1.0):
            
            Type: {episode.memory_type.value}
            Content: {episode.content}
            Context: {json.dumps(episode.context) if episode.context else 'None'}
            
            Consider:
            1. Financial impact
            2. Long-term relevance
            3. Uniqueness of insight
            4. Market significance
            
            Return only a float between 0.0 and 1.0.
            """
            
            response = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a trading memory importance analyzer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                api_key=self.config.OPENAI_API_KEY
            )
            
            importance = float(response.choices[0].message.content.strip())
            return max(0.0, min(1.0, importance))  # Ensure value is between 0 and 1
            
        except Exception as e:
            print(f"Error analyzing importance: {str(e)}")
            return 0.5  # Default importance
    
    async def add_memory(self, episode: Episode):
        """Add new episodic memory"""
        # Analyze importance if not provided
        if episode.importance == 1.0:
            episode.importance = await self._analyze_importance(episode)
        
        # Get embedding for the content
        embedding = await self._get_embedding(episode.content)
        
        # Prepare metadata
        metadata = {
            **episode.metadata,
            "importance": episode.importance,
            "timestamp": episode.timestamp.isoformat()
        }
        if episode.context:
            metadata["context"] = json.dumps(episode.context)
        if episode.emotions:
            metadata["emotions"] = json.dumps(episode.emotions)
        
        # Add to appropriate collection
        collection = self.collections[episode.memory_type]
        collection.add(
            embeddings=[embedding],
            documents=[episode.content],
            metadatas=[metadata],
            ids=[f"{episode.memory_type.value}_{datetime.now().isoformat()}"]
        )
    
    async def search_memories(
        self,
        query: str,
        memory_type: Optional[MemoryType] = None,
        limit: int = 5,
        min_importance: float = 0.0,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[Episode]:
        """Search for relevant memories with filtering options"""
        # Get embedding for query
        query_embedding = await self._get_embedding(query)
        
        # Prepare where clauses for filtering
        where_clauses = {}
        if min_importance > 0:
            where_clauses["importance"] = {"$gte": min_importance}
        
        if time_range:
            start, end = time_range
            where_clauses["timestamp"] = {
                "$gte": start.isoformat(),
                "$lte": end.isoformat()
            }
        
        # Determine which collections to search
        collections_to_search = (
            [self.collections[memory_type]] if memory_type 
            else list(self.collections.values())
        )
        
        # Search each collection
        all_results = []
        for collection in collections_to_search:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clauses if where_clauses else None
            )
            
            # Convert results to Episodes
            for i in range(len(results['documents'][0])):
                metadata = results['metadatas'][0][i]
                context = (
                    json.loads(metadata["context"]) 
                    if "context" in metadata else None
                )
                emotions = (
                    json.loads(metadata["emotions"]) 
                    if "emotions" in metadata else None
                )
                
                episode = Episode(
                    content=results['documents'][0][i],
                    memory_type=MemoryType(collection.metadata["memory_type"]),
                    timestamp=datetime.fromisoformat(metadata["timestamp"]),
                    metadata={
                        k: v for k, v in metadata.items() 
                        if k not in ["importance", "timestamp", "context", "emotions"]
                    },
                    importance=metadata["importance"],
                    context=context,
                    emotions=emotions
                )
                all_results.append((episode, results['distances'][0][i]))
        
        # Sort by relevance (distance) and return episodes
        all_results.sort(key=lambda x: x[1])
        return [episode for episode, _ in all_results[:limit]]
    
    async def consolidate_memories(self, memory_type: Optional[MemoryType] = None):
        """Periodically consolidate memories to create summary insights"""
        collections_to_consolidate = (
            [self.collections[memory_type]] if memory_type 
            else list(self.collections.values())
        )
        
        for collection in collections_to_consolidate:
            # Get all memories from the collection
            results = collection.get()
            
            if not results['documents']:
                continue
            
            # Group memories by date for consolidation
            memories_by_date = {}
            for i, doc in enumerate(results['documents']):
                date = datetime.fromisoformat(
                    results['metadatas'][i]["timestamp"]
                ).date()
                if date not in memories_by_date:
                    memories_by_date[date] = []
                memories_by_date[date].append({
                    "content": doc,
                    "metadata": results['metadatas'][i]
                })
            
            # Consolidate memories for each date
            for date, memories in memories_by_date.items():
                if len(memories) < 2:  # Need at least 2 memories to consolidate
                    continue
                
                # Create consolidation prompt
                memory_texts = [
                    f"- {mem['content']} (Importance: {mem['metadata']['importance']})"
                    for mem in memories
                ]
                
                prompt = f"""
                Analyze these trading memories from {date} and create a consolidated insight:
                
                Memories:
                {chr(10).join(memory_texts)}
                
                Create a concise summary that captures the key patterns and insights.
                """
                
                try:
                    response = await openai.ChatCompletion.create(
                        model="gpt-4",
                        messages=[
                            {
                                "role": "system",
                                "content": "You are a trading memory consolidator."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.5,
                        api_key=self.config.OPENAI_API_KEY
                    )
                    
                    # Create consolidated memory
                    consolidated_episode = Episode(
                        content=response.choices[0].message.content,
                        memory_type=MemoryType(collection.metadata["memory_type"]),
                        timestamp=datetime.combine(date, datetime.min.time()),
                        metadata={
                            "type": "consolidated_memory",
                            "source_count": len(memories),
                            "date": date.isoformat()
                        },
                        importance=max(mem['metadata']['importance'] for mem in memories)
                    )
                    
                    # Store consolidated memory
                    await self.add_memory(consolidated_episode)
                    
                except Exception as e:
                    print(f"Error consolidating memories: {str(e)}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memories"""
        stats = {}
        for memory_type in MemoryType:
            collection = self.collections[memory_type]
            results = collection.get()
            
            # Calculate importance statistics
            importances = [
                metadata["importance"] 
                for metadata in results['metadatas']
            ]
            
            stats[memory_type.value] = {
                "count": len(results['documents']),
                "avg_importance": np.mean(importances) if importances else 0,
                "high_importance_count": sum(1 for imp in importances if imp >= 0.8)
            }
        
        return stats

