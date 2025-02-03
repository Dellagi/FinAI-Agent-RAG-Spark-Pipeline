
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import alpaca_trade_api as tradeapi
from kafka import KafkaProducer
import json
import threading
import asyncio
import websockets
import logging

class MarketDataStream:
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.producer = KafkaProducer(
            bootstrap_servers=config.KAFKA_BROKERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        self.alpaca = tradeapi.REST(
            config.ALPACA_API_KEY,
            config.ALPACA_SECRET_KEY,
            base_url='https://paper-api.alpaca.markets'
        )
        
    async def stream_market_data(self, symbols: List[str]):
        """Stream real-time market data from Alpaca"""
        ws_url = "wss://stream.data.alpaca.markets/v2/iex"
        
        async with websockets.connect(ws_url) as websocket:
            auth_data = {
                "action": "auth",
                "key": self.config.ALPACA_API_KEY,
                "secret": self.config.ALPACA_SECRET_KEY
            }
            await websocket.send(json.dumps(auth_data))
            
            # Subscribe to trade updates
            subscribe_message = {
                "action": "subscribe",
                "trades": symbols,
                "quotes": symbols,
                "bars": symbols
            }
            await websocket.send(json.dumps(subscribe_message))
            
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    # Send to Kafka for processing
                    self.producer.send('market_data', data)
                    
                except Exception as e:
                    logging.error(f"WebSocket error: {str(e)}")
                    break

