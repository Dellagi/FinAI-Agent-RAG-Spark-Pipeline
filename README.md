# AI-Powered Trading System with Local LLM

A sophisticated trading system leveraging local LLM deployment through Ollama, distributed computing with Apache Spark, vector-based episodic memory using ChromaDB, and advanced RAG (Retrieval Augmented Generation) capabilities to make intelligent trading decisions.

## 🚀 Key Features

1. Local LLM Integration
   - Custom finance-tuned LLM model served through Ollama
   - Optimized for trading analysis and decision making
   - No dependency on external API services
   - Customizable model parameters and system prompts

2. Real-time Processing
   - Apache Spark distributed computing
   - Kafka stream processing
   - Real-time technical analysis
   - Multi-symbol monitoring

3. Advanced Memory System
   - ChromaDB vector store for episodic memory
   - Semantic search capabilities
   - Memory importance scoring
   - Automatic memory consolidation
   - Context-aware retrieval

4. Contextual RAG System
   - Hybrid retrieval combining BM25 and dense embeddings
   - Cross-encoder reranking for improved relevance
   - Market context-aware document retrieval
   - Automatic index maintenance and optimization
   - Time-window based filtering
   - Configurable relevance thresholds

5. Intelligent Analysis
   - Multi-source research aggregation
   - Technical indicator analysis
   - Market context integration
   - Sentiment analysis
   - Historical pattern recognition
   - Risk assessment

## 🛠️ Technical Stack

- **LLM**: Custom finance-tuned model via Ollama
- **Vector Store**: ChromaDB with DuckDB
- **RAG System**: Hybrid BM25 + Dense Retrieval
- **Reranking**: Cross-encoder (MiniLM)
- **Distributed Computing**: Apache Spark
- **Stream Processing**: Kafka
- **Caching**: Redis
- **Market Data**: Alpaca API
- **Embeddings**: OpenAI Ada 002
- **Text Processing**: NLTK, Transformers

## 🔄 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Market Data    │───▶│  Spark Streaming │───▶│  Technical     │
│  Stream (Alpaca)│    │  Processing      │    │  Analysis      │
└─────────────────┘    └──────────────────┘    └────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐          ▼
│  Research       │───▶│  Local LLM       │    ┌────────────────┐
│  Aggregation    │    │  & RAG System    │◀───│  Trading Agent │
└─────────────────┘    └──────────────────┘    └────────────────┘
                              ▲                        │
┌─────────────────┐    ┌──────┴───────────┐          ▼
│  ChromaDB       │◀──▶│  Memory System   │    ┌────────────────┐
│  Vector Store   │    │  & BM25 Index    │◀───│  Decision      │
└─────────────────┘    └──────────────────┘    │  Engine        │
                                               └────────────────┘
```

## 💡 Project status: In progress

## 💡 System Components

### RAG System
- **Hybrid Retrieval**: Combines BM25 keyword search with dense embeddings
- **Reranking**: Uses cross-encoder for improved result relevance
- **Context Integration**: Incorporates market conditions and temporal context
- **Automatic Maintenance**: Regular index updates and optimization
- **Configurable Parameters**: Adjustable relevance thresholds and result limits

### Memory Management
- **Vector Storage**: Efficient storage and retrieval of trading decisions
- **Contextual Search**: Market-aware memory retrieval
- **Importance Scoring**: Automatic scoring of trading decisions
- **Memory Consolidation**: Periodic synthesis of related memories
- **Temporal Awareness**: Time-based filtering and relevance decay

### Analysis Pipeline
- **Multi-source Research**: Aggregates data from various sources
- **Technical Analysis**: Real-time indicator calculation
- **Market Context**: Dynamic market condition assessment
- **Risk Analysis**: Comprehensive risk factor evaluation
- **Pattern Recognition**: Historical pattern matching and analysis

## ⚙️ Configuration

The system uses environment variables for configuration. Key RAG-related settings:

```env
RAG_INDEX_DIR=./rag_indices
RAG_MIN_RELEVANCE_SCORE=0.5
RAG_UPDATE_INTERVAL_HOURS=4
RAG_MAX_RESULTS=5
```

## 🤝 Contributing

Contributions are welcome! Please check the [Contributing Guide](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see the [LICENSE](LICENSE) file for details