
# AI-Powered Trading System with Local LLM

A sophisticated trading system leveraging local LLM deployment through Ollama, distributed computing with Apache Spark, and vector-based episodic memory using ChromaDB to make intelligent trading decisions.

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

## 🛠️ Technical Stack

- **LLM**: Custom finance-tuned model via Ollama
- **Vector Store**: ChromaDB with DuckDB
- **Distributed Computing**: Apache Spark
- **Stream Processing**: Kafka
- **Caching**: Redis
- **Market Data**: Alpaca API

## 🔄 Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐
│  Market Data    │───▶│  Spark Streaming │───▶│  Technical     │
│  Stream (Alpaca)│    │  Processing      │    │  Analysis      │
└─────────────────┘    └──────────────────┘    └────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐            ▼
│  Research       │───▶│  Local LLM       │    ┌────────────────┐
│  Aggregation    │    │                  │◀───│  Trading Agent │
└─────────────────┘    └──────────────────┘    └────────────────┘
                                                       │
┌─────────────────┐    ┌──────────────────┐            ▼
│  ChromaDB       │◀──▶│  Episodic Memory │◀───┐────────────────┐
│  Vector Store   │    │  Management      │    │  Decision      │
└─────────────────┘    └──────────────────┘    │  Engine        │
                                               └────────────────┘
```


## 🤝 Contributing

Contributions are welcome! Please check the [Contributing Guide](CONTRIBUTING.md) for guidelines.

## 📝 License

MIT License - see the [LICENSE](LICENSE) file for details.
