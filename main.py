
import asyncio
from pyspark.sql import SparkSession
from concurrent.futures import ThreadPoolExecutor
import signal
import sys
from typing import Set, Dict
import redis
from datetime import datetime, timedelta
import logging
from .agent.trading_agent import TradingAgent
class TradingPipeline:
    def __init__(self, config: Config):
        self.config = config
        self.running = True
        self.monitored_symbols: Set[str] = set()
        self.last_analysis: Dict[str, datetime] = {}
        
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
        
        # Setup memory consolidation schedule
        self.last_memory_consolidation = datetime.now()

    def shutdown(self, signum, frame):
        """Graceful shutdown handler"""
        print("\nInitiating graceful shutdown...")
        self.running = False

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
            
            # Start key components in parallel
            await asyncio.gather(
                self.run_market_data_stream(),
                self.run_analysis_loop()
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

