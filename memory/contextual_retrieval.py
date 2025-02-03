from typing import List, Dict, Any, Optional
from llama_index.core import (
    SimpleDirectoryReader,
    VectorStoreIndex, 
    Settings,
    Document,
    QueryBundle
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
import openai
import numpy as np
from datetime import datetime
import json
import Stemmer

class ContextualRetriever(BaseRetriever):
    """Custom retriever combining embedding and BM25 approaches"""
    
    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        bm25_retriever: BM25Retriever,
        top_k: int = 3
    ):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes using multiple approaches"""
        # Get results from both retrievers
        vector_nodes = self.vector_retriever.retrieve(query_bundle)
        bm25_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        # Combine results
        all_nodes = {}  # Use dict to deduplicate by node_id
        
        for node in vector_nodes:
            all_nodes[node.node.node_id] = node
            
        for node in bm25_nodes:
            if node.node.node_id in all_nodes:
                # Combine scores if node exists in both results
                all_nodes[node.node.node_id].score = (
                    0.7 * all_nodes[node.node.node_id].score + 
                    0.3 * node.score
                )
            else:
                all_nodes[node.node.node_id] = node
        
        # Sort by combined score
        sorted_nodes = sorted(
            all_nodes.values(),
            key=lambda x: x.score,
            reverse=True
        )
        
        return sorted_nodes[:self.top_k]

def create_contextual_nodes(trading_history: str, trading_decisions: List[Dict]) -> List[TextNode]:
    """Create nodes with context from the trading history"""
    nodes = []
    
    for decision in trading_decisions:
        # Create prompt to generate context
        prompt = f"""Here is a trading decision we want to situate within the overall trading history.
        
        <history>
        {trading_history}
        </history>

        <decision>
        {decision['content']}
        </decision>
        
        Please provide a brief context (2-3 sentences) that situates this trading decision within the overall history,
        focusing on patterns, relationships, and significance. Answer with just the context."""

        # Get context using GPT-4
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a trading context analyzer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )
        
        context = response.choices[0].message.content
        
        # Create node with enhanced context
        node = TextNode(
            text=decision['content'],
            metadata={
                **decision['metadata'],
                'context': context,
                'timestamp': decision['timestamp']
            }
        )
        nodes.append(node)
    
    return nodes

def create_embedding_retriever(nodes: List[TextNode], top_k: int = 3) -> VectorIndexRetriever:
    """Create vector retriever from nodes"""
    vector_index = VectorStoreIndex(nodes)
    return vector_index.as_retriever(similarity_top_k=top_k)

def create_bm25_retriever(nodes: List[TextNode], top_k: int = 3) -> BM25Retriever:
    """Create BM25 retriever from nodes"""
    return BM25Retriever.from_defaults(
        nodes=nodes,
        similarity_top_k=top_k,
        stemmer=Stemmer.Stemmer("english"),
        language="english"
    )
