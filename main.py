### main.py

import asyncio
from pyspark.sql import SparkSession
from concurrent.futures import ThreadPoolExecutor
import signal
import sys
from typing import Set, Dict
import redis
from datetime import datetime, timedelta
import logging
import os
from .agent.trading_agent import TradingAgent
from .config import Config

class TradingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.running = True
        self.monitored_symbols: Set[str] = set()
        self.last_analysis: Dict[str, datetime] = {}
        self.last_rag_update = datetime.now()
        
        # Initialize Spark
        self.spark = SparkSession.builder \
            .appName("AdvancedTrader") \
            .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.2.0") \
            .config("spark.executor.memory", "8g") \
            .config("spark.driver.memory", "4g") \
            .config("spark.sql.adaptive.enabled", "true") \
            .config("spark.sql.shuffle.partitions", "100") \
            .getOrCreate()

        # Initialize Redis for rate limiting and caching
        self.redis = redis.from_url(config.REDIS_URL)
        
        # Initialize components
        self.trading_agent = TradingAgent(config, self.spark)
        
        # Set up signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        # Setup memory consolidation and RAG schedule
        self.last_memory_consolidation = datetime.now()
        
        # Ensure RAG index directory exists
        os.makedirs(config.RAG_INDEX_DIR, exist_ok=True)
        
    def shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        print("\nInitiating graceful shutdown...")
        self.running = False
        
        # Save RAG index before shutdown
        try:
            rag_index_path = os.path.join(
                self.config.RAG_INDEX_DIR,
                f"rag_index_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            )
            self.trading_agent.rag_retriever.rag_system.save_index(rag_index_path)
            logging.info(f"RAG index saved to {rag_index_path}")
        except Exception as e:
            logging.error(f"Error saving RAG index during shutdown: {e}")

    async def run_market_data_stream(self):
        """Run the market data streaming component"""
        while self.running:
            try:
                await self.trading_agent.market_stream.stream_market_data(
                    list(self.monitored_symbols)
                )
            except Exception as e:
                logging.error(f"Market data stream error: {str(e)}")
                await asyncio.sleep(5)  # Wait before reconnecting

    async def analyze_symbol(self, symbol: str):
        """Analyze a single symbol"""
        try:
            # Check rate limiting
            last_analysis = self.last_analysis.get(symbol)
            if last_analysis and datetime.now() - last_analysis < timedelta(minutes=5):
                return  # Skip if analyzed recently

            # Perform analysis
            analysis = await self.trading_agent.analyze_trading_opportunity(symbol)
            
            # Log the analysis
            logging.info(f"Analysis for {symbol}:\n{json.dumps(analysis, indent=2)}")
            
            # Update last analysis time
            self.last_analysis[symbol] = datetime.now()
            
            # Store analysis in Redis for API access
            self.redis.setex(
                f"analysis:{symbol}",
                3600,  # 1 hour expiration
                json.dumps(analysis)
            )

        except Exception as e:
            logging.error(f"Error analyzing {symbol}: {str(e)}")

    async def run_rag_maintenance(self):
        """Periodically update and maintain the RAG system"""
        while self.running:
            try:
                current_time = datetime.now()
                
                # Check if it's time for RAG update
                if current_time - self.last_rag_update > timedelta(
                    hours=self.config.RAG_UPDATE_INTERVAL_HOURS
                ):
                    # Update RAG system with latest memories
                    await self.trading_agent.rag_retriever.update_from_chroma()
                    
                    # Save RAG index with timestamp
                    rag_index_path = os.path.join(
                        self.config.RAG_INDEX_DIR,
                        f"rag_index_{current_time.strftime('%Y%m%d_%H%M')}.pkl"
                    )
                    self.trading_agent.rag_retriever.rag_system.save_index(rag_index_path)
                    
                    # Cleanup old index files
                    self._cleanup_old_rag_indices()
                    
                    self.last_rag_update = current_time
                    logging.info("RAG system updated and indexed")
                    
            except Exception as e:
                logging.error(f"Error in RAG maintenance: {str(e)}")
                
            await asyncio.sleep(3600)  # Check every hour
            
    def _cleanup_old_rag_indices(self):
        """Clean up old RAG index files, keeping only the last 5"""
        try:
            index_files = []
            for file in os.listdir(self.config.RAG_INDEX_DIR):
                if file.startswith("rag_index_") and file.endswith(".pkl"):
                    file_path = os.path.join(self.config.RAG_INDEX_DIR, file)
                    index_files.append((file_path, os.path.getmtime(file_path)))
            
            # Sort by modification time (newest first)
            index_files.sort(key=lambda x: x[1], reverse=True)
            
            # Remove all but the last 5 files
            for file_path, _ in index_files[5:]:
                try:
                    os.remove(file_path)
                    logging.info(f"Removed old RAG index: {file_path}")
                except Exception as e:
                    logging.error(f"Error removing old RAG index {file_path}: {e}")
                    
        except Exception as e:
            logging.error(f"Error cleaning up old RAG indices: {e}")

    async def run_analysis_loop(self):
        """Run continuous analysis of monitored symbols"""
        while self.running:
            analysis_tasks = [
                self.analyze_symbol(symbol)
                for symbol in self.monitored_symbols
            ]
            
            if analysis_tasks:
                await asyncio.gather(*analysis_tasks)
            
            # Run memory consolidation if needed
            if datetime.now() - self.last_memory_consolidation > timedelta(hours=1):
                await self.trading_agent.memory_store.consolidate_memories()
                self.last_memory_consolidation = datetime.now()
            
            await asyncio.sleep(60)  # Wait between analysis rounds

    def add_symbols(self, symbols: List[str]):
        """Add symbols to monitor"""
        self.monitored_symbols.update(symbols)
        logging.info(f"Added symbols for monitoring: {symbols}")

    def remove_symbols(self, symbols: List[str]):
        """Remove symbols from monitoring"""
        self.monitored_symbols.difference_update(symbols)
        logging.info(f"Removed symbols from monitoring: {symbols}")

    async def run(self):
        """Run the main trading pipeline"""
        try:
            logging.info("Starting trading pipeline...")
            
            # Load latest RAG index if available
            try:
                index_files = [
                    f for f in os.listdir(self.config.RAG_INDEX_DIR)
                    if f.startswith("rag_index_") and f.endswith(".pkl")
                ]
                if index_files:
                    latest_index = max(
                        index_files,
                        key=lambda x: os.path.getmtime(
                            os.path.join(self.config.RAG_INDEX_DIR, x)
                        )
                    )
                    self.trading_agent.rag_retriever.rag_system.load_index(
                        os.path.join(self.config.RAG_INDEX_DIR, latest_index)
                    )
                    logging.info(f"Loaded RAG index from {latest_index}")
            except Exception as e:
                logging.error(f"Error loading RAG index: {e}")
            
            # Start key components in parallel
            await asyncio.gather(
                self.run_market_data_stream(),
                self.run_analysis_loop(),
                self.run_rag_maintenance()
            )
            
        except Exception as e:
            logging.error(f"Pipeline error: {str(e)}")
            
        finally:
            # Cleanup
            logging.info("Shutting down pipeline...")
            self.spark.stop()
            self.redis.close()

if __name__ == "__main__":
    # Load configuration
    config = Config.from_env()
    
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run pipeline
    pipeline = TradingPipeline(config)
    
    # Add initial symbols to monitor
    initial_symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    pipeline.add_symbols(initial_symbols)
    
    # Run the pipeline
    asyncio.run(pipeline.run())