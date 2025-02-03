from typing import List, Dict, Any
import openai
import pandas as pd
import numpy as np
from datetime import datetime
import json
import logging
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from ..research.research_engine import ResearchEngine
from ..memory.episodic_memory import EpisodicMemoryStore, Episode, MemoryType
from ..streaming.market_data_streaming import MarketDataStream
from ..retrieval.rag import ContextualRAGRetriever
from ..config import Config

class TradingAgent:
    def __init__(self, config: Config, spark: SparkSession):
        self.config = config
        self.spark = spark
        self.research_engine = ResearchEngine(config)
        self.memory_store = EpisodicMemoryStore(config)
        self.market_stream = MarketDataStream(config, spark)
        
        # Initialize RAG system
        self.rag_retriever = ContextualRAGRetriever(
            config,
            self.memory_store.collections[MemoryType.TRADING_DECISION]
        )
        
    async def analyze_trading_opportunity(self, symbol: str) -> Dict[str, Any]:
        """Analyze trading opportunity for a given symbol"""
        try:
            # Update RAG system with latest memories
            await self.rag_retriever.update_from_chroma()
            
            # Gather research and context
            research = await self.research_engine.gather_research(
                query=f"{symbol} stock analysis financial report",
                sources=["news", "academic", "financial"]
            )
            
            # Get relevant memories with importance threshold and RAG
            memories = await self.memory_store.search_memories(
                query=f"Previous trading analysis and decisions for {symbol}",
                memory_type=MemoryType.TRADING_DECISION,
                min_importance=0.6
            )
            
            # Get additional context from RAG system
            rag_results = await self.rag_retriever.retrieve_with_context(
                query=f"Recent trading patterns and market context for {symbol}",
                symbol=symbol,
                k=self.config.RAG_MAX_RESULTS
            )
            
            # Process market data using Spark
            market_df = self.spark.sql(f"""
                SELECT *
                FROM market_data
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC
                LIMIT 1000
            """)
            
            # Calculate technical indicators
            indicators = self._calculate_technical_indicators(market_df)
            market_context = self._analyze_market_context(indicators)
            
            # Prepare context for LLM with RAG insights
            analysis_context = self._prepare_analysis_context(
                symbol=symbol,
                research=research,
                memories=memories,
                indicators=indicators,
                market_context=market_context,
                rag_results=rag_results
            )
            
            # Get trading decision from LLM
            decision = await self._get_trading_decision(analysis_context)
            
            # Create and store memory episode
            episode = Episode(
                content=json.dumps({
                    'symbol': symbol,
                    'analysis': analysis_context,
                    'decision': decision,
                    'rag_insights': rag_results
                }),
                memory_type=MemoryType.TRADING_DECISION,
                timestamp=datetime.now(),
                metadata={
                    'symbol': symbol,
                    'action': decision['recommendation']
                },
                context={
                    'market_conditions': market_context,
                    'technical_indicators': indicators
                },
                emotions=decision.get('sentiment', [])
            )
            
            await self.memory_store.add_memory(episode)
            
            return {
                'symbol': symbol,
                'research': research,
                'technical_analysis': indicators,
                'market_context': market_context,
                'rag_insights': rag_results,
                'decision': decision,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logging.error(f"Error analyzing trading opportunity for {symbol}: {str(e)}")
            raise
            
    def _calculate_technical_indicators(self, df) -> Dict[str, Any]:
        """Calculate technical indicators using Spark"""
        df.createOrReplaceTempView("price_data")
        
        indicators = self.spark.sql("""
            WITH price_data_calcs AS (
                SELECT
                    symbol,
                    timestamp,
                    close,
                    volume,
                    AVG(close) OVER (
                        ORDER BY timestamp
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ) as sma_20,
                    AVG(close) OVER (
                        ORDER BY timestamp
                        ROWS BETWEEN 50 PRECEDING AND CURRENT ROW
                    ) as sma_50,
                    STDDEV(close) OVER (
                        ORDER BY timestamp
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ) as stddev_20,
                    AVG(volume) OVER (
                        ORDER BY timestamp
                        ROWS BETWEEN 20 PRECEDING AND CURRENT ROW
                    ) as avg_volume_20
                FROM price_data
            )
            SELECT
                symbol,
                timestamp,
                close,
                volume,
                sma_20,
                sma_50,
                stddev_20,
                avg_volume_20,
                (close - sma_20) / stddev_20 as z_score,
                volume / avg_volume_20 as volume_ratio,
                CASE
                    WHEN close > sma_20 AND sma_20 > sma_50 THEN 'STRONG_UPTREND'
                    WHEN close > sma_20 THEN 'UPTREND'
                    WHEN close < sma_20 AND sma_20 < sma_50 THEN 'STRONG_DOWNTREND'
                    WHEN close < sma_20 THEN 'DOWNTREND'
                    ELSE 'NEUTRAL'
                END as trend
            FROM price_data_calcs
            ORDER BY timestamp DESC
            LIMIT 1
        """)
        
        return indicators.collect()[0].asDict()
    
    def _analyze_market_context(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market context from indicators"""
        context = {
            'trend_strength': 0.0,
            'volatility_level': 'LOW',
            'volume_signal': 'NORMAL',
            'price_momentum': 'NEUTRAL'
        }
        
        # Analyze trend strength
        trend_map = {
            'STRONG_UPTREND': 1.0,
            'UPTREND': 0.5,
            'NEUTRAL': 0.0,
            'DOWNTREND': -0.5,
            'STRONG_DOWNTREND': -1.0
        }
        context['trend_strength'] = trend_map[indicators['trend']]
        
        # Analyze volatility
        if abs(indicators['z_score']) > 2.0:
            context['volatility_level'] = 'HIGH'
        elif abs(indicators['z_score']) > 1.0:
            context['volatility_level'] = 'MEDIUM'
            
        # Analyze volume
        volume_ratio = indicators['volume_ratio']
        if volume_ratio > 2.0:
            context['volume_signal'] = 'VERY_HIGH'
        elif volume_ratio > 1.5:
            context['volume_signal'] = 'HIGH'
        elif volume_ratio < 0.5:
            context['volume_signal'] = 'LOW'
            
        # Analyze momentum
        if indicators['close'] > indicators['sma_20'] * 1.05:
            context['price_momentum'] = 'STRONG_POSITIVE'
        elif indicators['close'] < indicators['sma_20'] * 0.95:
            context['price_momentum'] = 'STRONG_NEGATIVE'
        elif indicators['close'] > indicators['sma_20']:
            context['price_momentum'] = 'POSITIVE'
        elif indicators['close'] < indicators['sma_20']:
            context['price_momentum'] = 'NEGATIVE'
        
        return context
        
    def _prepare_analysis_context(
        self,
        symbol: str,
        research: Dict,
        memories: List[Episode],
        indicators: Dict[str, Any],
        market_context: Dict[str, Any],
        rag_results: Dict[str, Any]
    ) -> str:
        """Prepare context for LLM analysis"""
        context = f"""
        Analysis for {symbol}:

        Technical Indicators:
        - Current Price: ${indicators['close']:.2f}
        - SMA20: ${indicators['sma_20']:.2f}
        - SMA50: ${indicators['sma_50']:.2f}
        - Z-Score: {indicators['z_score']:.2f}
        - Volume Ratio: {indicators['volume_ratio']:.2f}
        - Current Trend: {indicators['trend']}
        
        Market Context:
        - Trend Strength: {market_context['trend_strength']}
        - Volatility Level: {market_context['volatility_level']}
        - Volume Signal: {market_context['volume_signal']}
        - Price Momentum: {market_context['price_momentum']}
        
        RAG System Insights:
        {self._format_rag_results(rag_results)}
        
        Recent Research:
        News:
        {self._format_research_items(research.get('news', []))}
        
        Academic Analysis:
        {self._format_research_items(research.get('academic', []))}
        
        Financial Reports:
        {self._format_research_items(research.get('financial', []))}
        
        Relevant Trading History:
        {self._format_memories(memories)}
        
        Additional Context:
        - Current Market Hours: {datetime.now().strftime('%H:%M:%S')}
        - Market Conditions: {'Normal Trading Hours' if 9 <= datetime.now().hour <= 16 else 'Extended Hours'}
        """
        return context
    
    @staticmethod
    def _format_rag_results(rag_results: Dict[str, Any]) -> str:
        """Format RAG system results for context"""
        formatted = []
        
        for result in rag_results.get('results', []):
            relevance_score = result.get('relevance_score', 0)
            reranking_score = result.get('reranking_score', 0)
            
            # Only include highly relevant results
            if relevance_score >= 0.7 or reranking_score >= 0.7:
                formatted.append(f"""
                Insight:
                {result['content'][:300]}...
                Relevance: {relevance_score:.2f}
                Reranking Score: {reranking_score:.2f}
                """)
                
        if not formatted:
            return "No highly relevant historical insights found."
            
        return "\n".join(formatted)
        
    async def _get_trading_decision(self, context: str) -> Dict[str, Any]:
        """Get trading decision from LLM"""
        try:
            response = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": """You are an expert financial analyst and trader.
                    Analyze the provided information and make a trading decision. Consider:
                    1. Technical analysis and price action
                    2. Market context and conditions
                    3. Fundamental factors from research
                    4. Historical patterns and past decisions
                    5. Risk management and position sizing
                    
                    Provide a structured analysis with:
                    - Clear trading recommendation (BUY, SELL, HOLD)
                    - Confidence level (0-100%)
                    - Key reasons for the decision
                    - Risk factors to monitor
                    - Suggested position size and stop loss
                    - Market sentiment assessment"""},
                    {"role": "user", "content": context}
                ],
                temperature=0.7,
                api_key=self.config.OPENAI_API_KEY
            )
            
            decision_text = response.choices[0].message.content
            
            # Extract sentiment from decision
            sentiment_prompt = f"""
            Analyze the following trading decision and extract the key emotions and sentiment:
            {decision_text}
            
            Return a list of relevant emotions/sentiment (e.g., confident, cautious, uncertain, optimistic, etc.)
            """
            
            sentiment_response = await openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Extract emotions and sentiment from trading decisions."},
                    {"role": "user", "content": sentiment_prompt}
                ],
                temperature=0.5,
                api_key=self.config.OPENAI_API_KEY
            )
            
            return {
                'recommendation': decision_text,
                'confidence': response.choices[0].finish_reason == "stop",
                'sentiment': sentiment_response.choices[0].message.content.split(', ')
            }
            
        except Exception as e:
            logging.error(f"Error getting LLM decision: {str(e)}")
            return {
                'recommendation': "Error analyzing trading opportunity",
                'confidence': False,
                'sentiment': ['uncertain']
            }
            
    @staticmethod
    def _format_research_items(items: List[Dict]) -> str:
        """Format research items for context"""
        formatted = []
        for item in items[:3]:  # Limit to most relevant items
            if 'title' in item:
                formatted.append(f"- {item['title']}")
                if 'summary' in item:
                    formatted.append(f"  Summary: {item['summary'][:200]}...")
            elif 'content' in item:
                formatted.append(f"- {item['content'][:200]}...")
        return "\n".join(formatted)

    @staticmethod
    def _format_memories(memories: List[Episode]) -> str:
        """Format previous trading memories for context"""
        formatted = []
        for memory in memories:
            memory_data = json.loads(memory.content)
            formatted.append(f"""
            Previous Analysis ({memory.timestamp.strftime('%Y-%m-%d %H:%M')}):
            Symbol: {memory_data['symbol']}
            Decision: {memory_data['decision']['recommendation'][:200]}...
            Importance: {memory.importance:.2f}
            """)
        return "\n".join(formatted)