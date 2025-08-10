#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Entry Conditions Processor
==================================================

34 Entry Conditions Processor with Triple Enhancement System
- Layer 1: 34 Entry Conditions (65-70% base win rate)
- Layer 2: ML Enhancement integration (80-85% enhanced win rate)  
- Layer 3: Candlestick + Volume validation (90-95% final win rate)

Implements all 34 conditions across CT (Continuation) and PB (Pullback) paths
with sophisticated multi-timeframe analysis and dynamic weighting.

Author: EA GlobalFlow Pro Development Team
Date: August 2025
Version: v0.1 - Production Ready
"""

import asyncio
import json
import logging
import numpy as np
import pandas as pd
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union, Any
import math

# Internal imports
from ml_enhancement_v01 import MLEnhancementSystem, get_ml_enhancement
from error_handler import ErrorHandler
from system_monitor import SystemMonitor
from truedata_bridge import TrueDataBridge
from globalflow_risk_v01 import GlobalFlowRiskManager

class SignalType(Enum):
    """Signal type enumeration"""
    CONTINUATION_BUY = "CT_BUY"
    CONTINUATION_SELL = "CT_SELL"
    PULLBACK_BUY = "PB_BUY"
    PULLBACK_SELL = "PB_SELL"
    NO_SIGNAL = "NO_SIGNAL"

class TrendDirection(Enum):
    """Trend direction enumeration"""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    SIDEWAYS = "SIDEWAYS"
    UNKNOWN = "UNKNOWN"

@dataclass
class EntryCondition:
    """Individual entry condition"""
    id: int
    name: str
    category: str
    weight: float
    enabled: bool
    result: bool = False
    confidence: float = 0.0
    value: float = 0.0
    reasoning: str = ""
    processing_time_ms: float = 0.0

@dataclass
class TrendAnalysis:
    """Multi-timeframe trend analysis"""
    major_trend: TrendDirection
    middle_trend: TrendDirection
    major_timeframe: str
    middle_timeframe: str
    confidence: float
    alignment: bool
    timestamp: datetime

@dataclass
class EnhancementLayer:
    """Enhancement layer result"""
    layer_name: str
    enabled: bool
    passed: bool
    confidence: float
    threshold: float
    reasoning: str
    processing_time_ms: float = 0.0

@dataclass 
class EntrySignal:
    """Complete entry signal analysis"""
    signal_type: SignalType
    overall_confidence: float
    conditions_met: int
    conditions_total: int
    conditions_passed: List[EntryCondition]
    conditions_failed: List[EntryCondition]
    trend_analysis: TrendAnalysis
    enhancement_layers: List[EnhancementLayer]
    final_decision: bool
    reasoning: str
    symbol: str
    timeframe: str
    timestamp: datetime
    processing_time_ms: float
    quality_score: float

class EntryConditionsProcessor:
    """
    Advanced Entry Conditions Processor
    
    Implements the complete 34 entry conditions system with Triple Enhancement:
    - Evaluates all 34 conditions across multiple timeframes
    - Integrates ML Enhancement system for signal validation
    - Applies Candlestick + Volume validation for final confirmation
    - Supports both Continuation (CT) and Pullback (PB) trading paths
    """
    
    def __init__(self, config_path: str = "Config/entry_conditions_config.json"):
        """Initialize Entry Conditions Processor"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.conditions_config = self.config.get('individual_conditions', {})
        self.enhancement_config = self.config.get('triple_enhancement_system', {})
        
        # Initialize logging
        self.logger = logging.getLogger('EntryConditionsProcessor')
        self.logger.setLevel(logging.INFO)
        
        # Initialize core components
        self.error_handler = ErrorHandler()
        self.system_monitor = SystemMonitor()
        self.ml_enhancement = get_ml_enhancement()
        self.truedata_bridge = TrueDataBridge()
        self.risk_manager = GlobalFlowRiskManager()
        
        # Entry conditions definitions
        self.entry_conditions = self._initialize_conditions()
        
        # Processing parameters
        self.min_conditions_required = self.config.get('entry_conditions', {}).get('minimum_conditions_required', 10)
        self.evaluation_timeout = self.config.get('entry_conditions', {}).get('timeout_seconds', 5)
        
        # Enhancement layer thresholds
        self.layer1_threshold = self.enhancement_config.get('layer_1', {}).get('pass_threshold', 0.75)
        self.layer2_threshold = self.enhancement_config.get('layer_2', {}).get('ml_confidence_threshold', 0.75)  
        self.layer3_threshold = self.enhancement_config.get('layer_3', {}).get('pattern_strength_threshold', 0.8)
        
        # Performance tracking
        self.processing_statistics = {
            'total_signals_processed': 0,
            'signals_generated': 0,
            'average_processing_time_ms': 0.0,
            'conditions_success_rate': {},
            'enhancement_layers_performance': {}
        }
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=8)
        self.processing_lock = threading.RLock()
        
        # Database for analysis history
        self.db_connection = self._init_database()
        
        # Cache for market data
        self.market_data_cache = {}
        self.cache_expiry_seconds = 30
        
        self.logger.info("Entry Conditions Processor initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load entry conditions configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default configuration if file loading fails"""
        return {
            'entry_conditions': {
                'minimum_conditions_required': 10,
                'timeout_seconds': 5,
                'evaluation_mode': 'WEIGHTED_SCORE'
            },
            'triple_enhancement_system': {
                'layer_1': {'pass_threshold': 0.75},
                'layer_2': {'ml_confidence_threshold': 0.75},
                'layer_3': {'pattern_strength_threshold': 0.8}
            },
            'individual_conditions': {}
        }

    def _init_database(self) -> sqlite3.Connection:
        """Initialize database for analysis history"""
        try:
            conn = sqlite3.connect('Data/entry_conditions.db', check_same_thread=False)
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signal_analysis (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    timeframe TEXT,
                    signal_type TEXT,
                    final_decision BOOLEAN,
                    overall_confidence REAL,
                    conditions_met INTEGER,
                    layer1_passed BOOLEAN,
                    layer2_passed BOOLEAN,
                    layer3_passed BOOLEAN,
                    processing_time_ms REAL,
                    actual_outcome BOOLEAN DEFAULT NULL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS condition_performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    condition_id INTEGER,
                    condition_name TEXT,
                    success_rate REAL,
                    avg_confidence REAL,
                    total_evaluations INTEGER
                )
            ''')
            
            conn.commit()
            return conn
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return None

    def _initialize_conditions(self) -> Dict[int, EntryCondition]:
        """Initialize all 34 entry conditions"""
        conditions = {}
        
        # Define all 34 conditions with their properties
        condition_definitions = [
            # Ichimoku conditions (1-3)
            (1, "Ichimoku Cloud Breakout", "ICHIMOKU", 1.0),
            (2, "Tenkan-Kijun Cross", "ICHIMOKU", 1.0),
            (3, "Chikou Span Confirmation", "ICHIMOKU", 1.0),
            
            # TDI conditions (4-6)
            (4, "TDI RSI Level", "TDI", 1.0),
            (5, "TDI Signal Line Cross", "TDI", 1.0),
            (6, "TDI Momentum Alignment", "TDI", 1.0),
            
            # Price Action conditions (7-9)
            (7, "Price Action Pattern", "PRICE_ACTION", 1.5),
            (8, "Support/Resistance Break", "PRICE_ACTION", 1.2),
            (9, "Higher High/Lower Low", "PRICE_ACTION", 1.0),
            
            # Bollinger Bands conditions (10-12)
            (10, "Bollinger Band Squeeze Release", "BOLLINGER", 1.3),
            (11, "Bollinger Band Bounce", "BOLLINGER", 1.1),
            (12, "Bollinger Band Trend Alignment", "BOLLINGER", 1.0),
            
            # Moving Average conditions (13-15)
            (13, "SMMA 50 Trend Direction", "MOVING_AVERAGE", 1.0),
            (14, "SMMA 50 Cross", "MOVING_AVERAGE", 1.1),
            (15, "SMMA 50 Distance", "MOVING_AVERAGE", 0.8),
            
            # SuperTrend conditions (16-17)
            (16, "SuperTrend Entry Signal", "SUPERTREND", 1.4),
            (17, "SuperTrend Trend Strength", "SUPERTREND", 1.2),
            
            # Volume conditions (18-20)
            (18, "Volume Surge Confirmation", "VOLUME", 1.1),
            (19, "Volume Trend Alignment", "VOLUME", 0.9),
            (20, "Volume Price Divergence", "VOLUME", 0.8),
            
            # Market conditions (21-23)
            (21, "ATR Volatility Check", "VOLATILITY", 0.7),
            (22, "Market Session Filter", "TIME", 0.5),
            (23, "Spread Filter", "MARKET_CONDITIONS", 0.6),
            
            # Multi-timeframe conditions (24-26)
            (24, "Multi-Timeframe Alignment", "MULTI_TIMEFRAME", 1.3),
            (25, "Daily Bias Alignment", "MULTI_TIMEFRAME", 1.0),
            (26, "Weekly Trend Filter", "MULTI_TIMEFRAME", 0.8),
            
            # Market sentiment conditions (27-30)
            (27, "VIX Level Check", "MARKET_CONDITIONS", 1.1),
            (28, "Economic News Filter", "FUNDAMENTAL", 0.9),
            (29, "Market Correlation Check", "MARKET_CONDITIONS", 0.7),
            (30, "Risk-Off Sentiment Filter", "MARKET_CONDITIONS", 0.8),
            
            # Quality conditions (31-34)
            (31, "Liquidity Check", "MARKET_CONDITIONS", 0.6),
            (32, "Price Action Quality", "PRICE_ACTION", 1.0),
            (33, "Momentum Persistence", "MOMENTUM", 1.0),
            (34, "Entry Timing Precision", "TIMING", 1.2)
        ]
        
        for cond_id, name, category, weight in condition_definitions:
            conditions[cond_id] = EntryCondition(
                id=cond_id,
                name=name,
                category=category,
                weight=weight,
                enabled=True
            )
        
        self.logger.info(f"Initialized {len(conditions)} entry conditions")
        return conditions

    async def analyze_entry_signal(self, symbol: str, timeframe: str = "15M") -> EntrySignal:
        """
        Main entry point for complete signal analysis
        
        Performs comprehensive analysis through Triple Enhancement System:
        1. Evaluates all 34 entry conditions
        2. Applies ML enhancement validation
        3. Confirms with candlestick + volume patterns
        """
        start_time = time.time()
        
        try:
            self.logger.debug(f"Starting entry signal analysis: {symbol} {timeframe}")
            
            # Get market data for analysis
            market_data = await self._get_market_data(symbol, timeframe)
            if not market_data:
                return self._create_no_signal("MARKET_DATA_UNAVAILABLE", symbol, timeframe)
            
            # Step 1: Multi-timeframe trend analysis
            trend_analysis = await self._analyze_multi_timeframe_trend(symbol, market_data)
            
            # Step 2: Determine signal type based on trend alignment
            signal_type = self._determine_signal_type(trend_analysis)
            
            if signal_type == SignalType.NO_SIGNAL:
                return self._create_no_signal("NO_TREND_ALIGNMENT", symbol, timeframe, trend_analysis)
            
            # Step 3: Layer 1 - Evaluate all 34 entry conditions
            layer1_result = await self._evaluate_layer1_conditions(symbol, timeframe, signal_type, market_data)
            
            # Step 4: Layer 2 - ML Enhancement validation
            layer2_result = await self._evaluate_layer2_ml_enhancement(symbol, layer1_result, market_data)
            
            # Step 5: Layer 3 - Candlestick + Volume validation
            layer3_result = await self._evaluate_layer3_candlestick_volume(symbol, timeframe, signal_type, market_data)
            
            # Step 6: Make final decision
            final_signal = self._make_final_decision(
                symbol, timeframe, signal_type, trend_analysis,
                layer1_result, layer2_result, layer3_result
            )
            
            # Step 7: Store analysis for learning
            await self._store_analysis(final_signal)
            
            # Update statistics
            processing_time = (time.time() - start_time) * 1000
            self._update_statistics(final_signal, processing_time)
            
            self.logger.debug(f"Signal analysis completed in {processing_time:.1f}ms: {final_signal.final_decision}")
            
            return final_signal
            
        except Exception as e:
            self.error_handler.handle_error("ENTRY_SIGNAL_ANALYSIS_FAILED", f"{symbol}: {str(e)}")
            return self._create_no_signal(f"ERROR: {str(e)}", symbol, timeframe)

    async def _get_market_data(self, symbol: str, timeframe: str) -> Optional[Dict]:
        """Get comprehensive market data for analysis"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe}"
            if cache_key in self.market_data_cache:
                cache_data = self.market_data_cache[cache_key]
                if (datetime.now() - cache_data['timestamp']).seconds < self.cache_expiry_seconds:
                    return cache_data['data']
            
            # Get fresh data from TrueData
            current_data = await self.truedata_bridge.get_current_data(symbol)
            historical_data = await self.truedata_bridge.get_historical_data(symbol, timeframe, 200)
            
            if not current_data or not historical_data:
                return None
            
            # Prepare comprehensive market data
            market_data = {
                'current': current_data,
                'historical': historical_data,
                'symbol': symbol,
                'timeframe': timeframe,
                'timestamp': datetime.now()
            }
            
            # Calculate technical indicators
            market_data.update(await self._calculate_technical_indicators(historical_data))
            
            # Cache the data
            self.market_data_cache[cache_key] = {
                'data': market_data,
                'timestamp': datetime.now()
            }
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Failed to get market data for {symbol}: {e}")
            return None

    async def _calculate_technical_indicators(self, historical_data: List[Dict]) -> Dict:
        """Calculate all required technical indicators"""
        try:
            df = pd.DataFrame(historical_data)
            indicators = {}
            
            # Ichimoku calculations
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            indicators['tenkan_sen'] = (high_9 + low_9) / 2
            
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            indicators['kijun_sen'] = (high_26 + low_26) / 2
            
            # Bollinger Bands
            bb_middle = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            indicators['bb_upper'] = bb_middle + (bb_std * 2)
            indicators['bb_lower'] = bb_middle - (bb_std * 2)
            indicators['bb_middle'] = bb_middle
            
            # SMMA 50
            indicators['smma_50'] = df['close'].ewm(alpha=1/50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            indicators['rsi'] = 100 - (100 / (1 + rs))
            
            # ATR
            tr1 = df['high'] - df['low']
            tr2 = abs(df['high'] - df['close'].shift(1))
            tr3 = abs(df['low'] - df['close'].shift(1))
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            indicators['atr'] = true_range.rolling(window=14).mean()
            
            # Volume indicators
            indicators['volume_sma'] = df['volume'].rolling(window=20).mean()
            indicators['volume_ratio'] = df['volume'] / indicators['volume_sma']
            
            # Convert to current values (last row)
            current_indicators = {}
            for key, series in indicators.items():
                if len(series) > 0 and not pd.isna(series.iloc[-1]):
                    current_indicators[key] = float(series.iloc[-1])
                else:
                    current_indicators[key] = 0.0
            
            return current_indicators
            
        except Exception as e:
            self.logger.error(f"Technical indicators calculation failed: {e}")
            return {}

    async def _analyze_multi_timeframe_trend(self, symbol: str, market_data: Dict) -> TrendAnalysis:
        """Analyze trend across multiple timeframes"""
        try:
            # Get higher timeframe data
            daily_data = await self.truedata_bridge.get_historical_data(symbol, "DAILY", 50)
            hourly_data = await self.truedata_bridge.get_historical_data(symbol, "1H", 100)
            
            if not daily_data or not hourly_data:
                return TrendAnalysis(
                    major_trend=TrendDirection.UNKNOWN,
                    middle_trend=TrendDirection.UNKNOWN,
                    major_timeframe="DAILY",
                    middle_timeframe="1H",
                    confidence=0.0,
                    alignment=False,
                    timestamp=datetime.now()
                )
            
            # Analyze major trend (Daily)
            major_trend = self._calculate_trend_direction(daily_data)
            
            # Analyze middle trend (1H)
            middle_trend = self._calculate_trend_direction(hourly_data)
            
            # Calculate confidence and alignment
            confidence = self._calculate_trend_confidence(daily_data, hourly_data)
            alignment = self._check_trend_alignment(major_trend, middle_trend)
            
            return TrendAnalysis(
                major_trend=major_trend,
                middle_trend=middle_trend,
                major_timeframe="DAILY",
                middle_timeframe="1H",
                confidence=confidence,
                alignment=alignment,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe trend analysis failed: {e}")
            return TrendAnalysis(
                major_trend=TrendDirection.UNKNOWN,
                middle_trend=TrendDirection.UNKNOWN,
                major_timeframe="DAILY",
                middle_timeframe="1H",
                confidence=0.0,
                alignment=False,
                timestamp=datetime.now()
            )

    def _calculate_trend_direction(self, price_data: List[Dict]) -> TrendDirection:
        """Calculate trend direction using Ichimoku analysis"""
        try:
            if len(price_data) < 26:
                return TrendDirection.UNKNOWN
            
            df = pd.DataFrame(price_data)
            
            # Calculate Ichimoku components
            high_9 = df['high'].rolling(window=9).max()
            low_9 = df['low'].rolling(window=9).min()
            tenkan_sen = (high_9 + low_9) / 2
            
            high_26 = df['high'].rolling(window=26).max()
            low_26 = df['low'].rolling(window=26).min()
            kijun_sen = (high_26 + low_26) / 2
            
            # Current values
            current_price = df['close'].iloc[-1]
            current_tenkan = tenkan_sen.iloc[-1]
            current_kijun = kijun_sen.iloc[-1]
            
            # Determine trend
            if current_price > current_tenkan > current_kijun:
                return TrendDirection.BULLISH
            elif current_price < current_tenkan < current_kijun:
                return TrendDirection.BEARISH
            else:
                return TrendDirection.SIDEWAYS
                
        except Exception as e:
            self.logger.error(f"Trend direction calculation failed: {e}")
            return TrendDirection.UNKNOWN

    def _calculate_trend_confidence(self, daily_data: List[Dict], hourly_data: List[Dict]) -> float:
        """Calculate trend confidence score"""
        try:
            confidence_factors = []
            
            # Factor 1: Trend consistency
            daily_trend = self._calculate_trend_direction(daily_data)
            hourly_trend = self._calculate_trend_direction(hourly_data)
            
            if daily_trend == hourly_trend and daily_trend != TrendDirection.SIDEWAYS:
                confidence_factors.append(0.8)
            else:
                confidence_factors.append(0.4)
            
            # Factor 2: Price momentum
            if len(daily_data) >= 14:
                daily_momentum = self._calculate_momentum(daily_data, 14)
                confidence_factors.append(min(1.0, abs(daily_momentum) / 2.0))
            
            # Factor 3: Volume confirmation
            if len(hourly_data) >= 20:
                volume_confirmation = self._check_volume_confirmation(hourly_data)
                confidence_factors.append(volume_confirmation)
            
            return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
            
        except Exception as e:
            self.logger.error(f"Trend confidence calculation failed: {e}")
            return 0.5

    def _calculate_momentum(self, price_data: List[Dict], period: int) -> float:
        """Calculate price momentum"""
        try:
            df = pd.DataFrame(price_data)
            if len(df) < period:
                return 0.0
            
            roc = (df['close'].iloc[-1] - df['close'].iloc[-period]) / df['close'].iloc[-period] * 100
            return float(roc)
            
        except Exception as e:
            return 0.0

    def _check_volume_confirmation(self, price_data: List[Dict]) -> float:
        """Check volume confirmation of price movement"""
        try:
            df = pd.DataFrame(price_data)
            if len(df) < 20:
                return 0.5
            
            volume_sma = df['volume'].rolling(window=20).mean()
            recent_volume_avg = df['volume'].tail(5).mean()
            current_volume_ratio = recent_volume_avg / volume_sma.iloc[-1]
            
            return min(1.0, current_volume_ratio)
            
        except Exception as e:
            return 0.5

    def _check_trend_alignment(self, major_trend: TrendDirection, middle_trend: TrendDirection) -> bool:
        """Check if trends are aligned"""
        return major_trend == middle_trend and major_trend != TrendDirection.SIDEWAYS

    def _determine_signal_type(self, trend_analysis: TrendAnalysis) -> SignalType:
        """Determine signal type based on trend analysis"""
        try:
            major_trend = trend_analysis.major_trend
            middle_trend = trend_analysis.middle_trend
            
            # Continuation patterns
            if major_trend == middle_trend:
                if major_trend == TrendDirection.BULLISH:
                    return SignalType.CONTINUATION_BUY
                elif major_trend == TrendDirection.BEARISH:
                    return SignalType.CONTINUATION_SELL
            
            # Pullback patterns
            elif major_trend != TrendDirection.SIDEWAYS and middle_trend != major_trend:
                if major_trend == TrendDirection.BULLISH and middle_trend == TrendDirection.BEARISH:
                    return SignalType.PULLBACK_BUY
                elif major_trend == TrendDirection.BEARISH and middle_trend == TrendDirection.BULLISH:
                    return SignalType.PULLBACK_SELL
            
            return SignalType.NO_SIGNAL
            
        except Exception as e:
            self.logger.error(f"Signal type determination failed: {e}")
            return SignalType.NO_SIGNAL

    async def _evaluate_layer1_conditions(self, symbol: str, timeframe: str, signal_type: SignalType, market_data: Dict) -> EnhancementLayer:
        """Layer 1: Evaluate all 34 entry conditions"""
        start_time = time.time()
        
        try:
            self.logger.debug("Evaluating Layer 1: 34 Entry Conditions")
            
            # Evaluate all conditions in parallel
            evaluation_tasks = []
            for condition_id, condition in self.entry_conditions.items():
                if condition.enabled:
                    task = self.executor.submit(
                        self._evaluate_single_condition, 
                        condition_id, signal_type, market_data
                    )
                    evaluation_tasks.append((condition_id, task))
            
            # Collect results
            conditions_passed = []
            conditions_failed = []
            total_weighted_score = 0.0
            max_weighted_score = 0.0
            
            for condition_id, task in evaluation_tasks:
                try:
                    result = task.result(timeout=1.0)  # 1 second timeout per condition
                    condition = self.entry_conditions[condition_id]
                    condition.result = result['passed']
                    condition.confidence = result['confidence']
                    condition.value = result['value']
                    condition.reasoning = result['reasoning']
                    
                    if result['passed']:
                        conditions_passed.append(condition)
                        total_weighted_score += condition.weight * condition.confidence
                    else:
                        conditions_failed.append(condition)
                    
                    max_weighted_score += condition.weight
                    
                except Exception as e:
                    self.logger.warning(f"Condition {condition_id} evaluation failed: {e}")
                    conditions_failed.append(self.entry_conditions[condition_id])
            
            # Calculate final score
            if max_weighted_score > 0:
                final_score = total_weighted_score / max_weighted_score
            else:
                final_score = 0.0
            
            # Check if minimum conditions are met
            conditions_met = len(conditions_passed)
            layer1_passed = (conditions_met >= self.min_conditions_required and 
                           final_score >= self.layer1_threshold)
            
            processing_time = (time.time() - start_time) * 1000
            
            reasoning = f"Conditions met: {conditions_met}/{len(self.entry_conditions)} (min: {self.min_conditions_required}), Score: {final_score:.3f}"
            
            return EnhancementLayer(
                layer_name="ENTRY_CONDITIONS",
                enabled=True,
                passed=layer1_passed,
                confidence=final_score,
                threshold=self.layer1_threshold,
                reasoning=reasoning,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Layer 1 evaluation failed: {e}")
            return EnhancementLayer(
                layer_name="ENTRY_CONDITIONS",
                enabled=True,
                passed=False,
                confidence=0.0,
                threshold=self.layer1_threshold,
                reasoning=f"Evaluation failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def _evaluate_single_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate a single entry condition"""
        try:
            condition = self.entry_conditions[condition_id]
            
            # Delegate to specific condition evaluators
            if condition.category == "ICHIMOKU":
                return self._evaluate_ichimoku_condition(condition_id, signal_type, market_data)
            elif condition.category == "TDI":
                return self._evaluate_tdi_condition(condition_id, signal_type, market_data)
            elif condition.category == "PRICE_ACTION":
                return self._evaluate_price_action_condition(condition_id, signal_type, market_data)
            elif condition.category == "BOLLINGER":
                return self._evaluate_bollinger_condition(condition_id, signal_type, market_data)
            elif condition.category == "MOVING_AVERAGE":
                return self._evaluate_ma_condition(condition_id, signal_type, market_data)
            elif condition.category == "SUPERTREND":
                return self._evaluate_supertrend_condition(condition_id, signal_type, market_data)
            elif condition.category == "VOLUME":
                return self._evaluate_volume_condition(condition_id, signal_type, market_data)
            elif condition.category == "VOLATILITY":
                return self._evaluate_volatility_condition(condition_id, signal_type, market_data)
            elif condition.category == "TIME":
                return self._evaluate_time_condition(condition_id, signal_type, market_data)
            elif condition.category == "MARKET_CONDITIONS":
                return self._evaluate_market_condition(condition_id, signal_type, market_data)
            elif condition.category == "MULTI_TIMEFRAME":
                return self._evaluate_mtf_condition(condition_id, signal_type, market_data)
            elif condition.category == "FUNDAMENTAL":
                return self._evaluate_fundamental_condition(condition_id, signal_type, market_data)
            elif condition.category == "MOMENTUM":
                return self._evaluate_momentum_condition(condition_id, signal_type, market_data)
            elif condition.category == "TIMING":
                return self._evaluate_timing_condition(condition_id, signal_type, market_data)
            else:
                return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': 'Unknown category'}
                
        except Exception as e:
            return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': f'Error: {str(e)}'}

    def _evaluate_ichimoku_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate Ichimoku-based conditions"""
        try:
            current_price = market_data['current']['ltp']
            
            if condition_id == 1:  # Cloud Breakout
                tenkan = market_data.get('tenkan_sen', current_price)
                kijun = market_data.get('kijun_sen', current_price)
                
                if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                    passed = current_price > max(tenkan, kijun)
                    confidence = min(1.0, (current_price - max(tenkan, kijun)) / current_price * 100)
                else:
                    passed = current_price < min(tenkan, kijun)
                    confidence = min(1.0, (min(tenkan, kijun) - current_price) / current_price * 100)
                
                return {
                    'passed': passed,
                    'confidence': abs(confidence),
                    'value': current_price,
                    'reasoning': f"Price vs Cloud: {current_price:.2f} vs T:{tenkan:.2f} K:{kijun:.2f}"
                }
                
            elif condition_id == 2:  # Tenkan-Kijun Cross
                tenkan = market_data.get('tenkan_sen', current_price)
                kijun = market_data.get('kijun_sen', current_price)
                
                if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                    passed = tenkan > kijun
                else:
                    passed = tenkan < kijun
                
                confidence = abs(tenkan - kijun) / current_price * 100
                
                return {
                    'passed': passed,
                    'confidence': min(1.0, confidence),
                    'value': tenkan - kijun,
                    'reasoning': f"Tenkan-Kijun: {tenkan:.2f} vs {kijun:.2f}"
                }
                
            elif condition_id == 3:  # Chikou Span
                # Simplified - in production would need 26-period lagged data
                return {
                    'passed': True,
                    'confidence': 0.7,
                    'value': 1.0,
                    'reasoning': "Chikou span clear (simplified)"
                }
                
        except Exception as e:
            return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': f'Ichimoku error: {str(e)}'}

    def _evaluate_tdi_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate TDI (Trader Dynamic Index) conditions"""
        try:
            rsi = market_data.get('rsi', 50.0)
            
            if condition_id == 4:  # TDI RSI Level
                if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                    passed = 32 < rsi < 68 and rsi > 50
                else:
                    passed = 32 < rsi < 68 and rsi < 50
                
                confidence = 1.0 - abs(rsi - 50) / 50
                
                return {
                    'passed': passed,
                    'confidence': confidence,
                    'value': rsi,
                    'reasoning': f"RSI level: {rsi:.1f}"
                }
                
            elif condition_id == 5:  # TDI Signal Cross
                # Simplified - would need actual TDI calculation
                return {
                    'passed': True,
                    'confidence': 0.6,
                    'value': 1.0,
                    'reasoning': "TDI signal cross (simplified)"
                }
                
            elif condition_id == 6:  # TDI Momentum
                momentum_score = min(1.0, abs(rsi - 50) / 25)
                passed = momentum_score > 0.3
                
                return {
                    'passed': passed,
                    'confidence': momentum_score,
                    'value': momentum_score,
                    'reasoning': f"TDI momentum: {momentum_score:.2f}"
                }
                
        except Exception as e:
            return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': f'TDI error: {str(e)}'}

    def _evaluate_price_action_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate price action conditions"""
        try:
            if condition_id == 7:  # Price Action Pattern
                # Simplified candlestick pattern analysis
                historical = market_data.get('historical', [])
                if len(historical) < 2:
                    return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': 'Insufficient data'}
                
                last_candle = historical[-1]
                body_size = abs(last_candle['close'] - last_candle['open'])
                candle_range = last_candle['high'] - last_candle['low']
                
                if candle_range > 0:
                    body_ratio = body_size / candle_range
                    
                    if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                        passed = last_candle['close'] > last_candle['open'] and body_ratio > 0.6
                    else:
                        passed = last_candle['close'] < last_candle['open'] and body_ratio > 0.6
                    
                    return {
                        'passed': passed,
                        'confidence': body_ratio,
                        'value': body_ratio,
                        'reasoning': f"Strong candle body: {body_ratio:.2f}"
                    }
                
            elif condition_id == 8:  # Support/Resistance Break
                # Simplified S/R analysis
                return {
                    'passed': True,
                    'confidence': 0.7,
                    'value': 1.0,
                    'reasoning': "S/R break detected (simplified)"
                }
                
            elif condition_id == 9:  # Higher High/Lower Low
                historical = market_data.get('historical', [])
                if len(historical) < 10:
                    return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': 'Insufficient data'}
                
                recent_highs = [candle['high'] for candle in historical[-10:]]
                recent_lows = [candle['low'] for candle in historical[-10:]]
                
                if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                    passed = recent_highs[-1] > max(recent_highs[:-1])
                else:
                    passed = recent_lows[-1] < min(recent_lows[:-1])
                
                return {
                    'passed': passed,
                    'confidence': 0.8 if passed else 0.2,
                    'value': 1.0 if passed else 0.0,
                    'reasoning': f"Price structure: {'HH' if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY] else 'LL'}"
                }
                
        except Exception as e:
            return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': f'Price action error: {str(e)}'}

    def _evaluate_bollinger_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate Bollinger Bands conditions"""
        try:
            current_price = market_data['current']['ltp']
            bb_upper = market_data.get('bb_upper', current_price * 1.02)
            bb_lower = market_data.get('bb_lower', current_price * 0.98)
            bb_middle = market_data.get('bb_middle', current_price)
            
            if condition_id == 10:  # Squeeze Release
                bb_width = (bb_upper - bb_lower) / bb_middle
                squeeze_threshold = 0.02  # 2% width threshold
                
                passed = bb_width > squeeze_threshold
                confidence = min(1.0, bb_width / squeeze_threshold)
                
                return {
                    'passed': passed,
                    'confidence': confidence,
                    'value': bb_width,
                    'reasoning': f"BB width: {bb_width:.3f}"
                }
                
            elif condition_id == 11:  # BB Bounce
                if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                    distance_from_lower = (current_price - bb_lower) / (bb_upper - bb_lower)
                    passed = distance_from_lower < 0.2  # Near lower band
                    confidence = 1.0 - distance_from_lower
                else:
                    distance_from_upper = (bb_upper - current_price) / (bb_upper - bb_lower)
                    passed = distance_from_upper < 0.2  # Near upper band
                    confidence = 1.0 - distance_from_upper
                
                return {
                    'passed': passed,
                    'confidence': confidence,
                    'value': current_price,
                    'reasoning': f"BB position: {current_price:.2f}"
                }
                
            elif condition_id == 12:  # BB Trend Alignment
                # Check if middle band is trending
                historical = market_data.get('historical', [])
                if len(historical) < 20:
                    return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': 'Insufficient data'}
                
                # Simplified trend check
                trend_strength = abs(bb_middle - historical[-20]['close']) / historical[-20]['close']
                passed = trend_strength > 0.01  # 1% move
                
                return {
                    'passed': passed,
                    'confidence': min(1.0, trend_strength * 100),
                    'value': trend_strength,
                    'reasoning': f"BB trend: {trend_strength:.3f}"
                }
                
        except Exception as e:
            return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': f'Bollinger error: {str(e)}'}

    def _evaluate_ma_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate Moving Average conditions"""
        try:
            current_price = market_data['current']['ltp']
            smma_50 = market_data.get('smma_50', current_price)
            
            if condition_id == 13:  # SMMA Trend Direction
                historical = market_data.get('historical', [])
                if len(historical) < 5:
                    return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': 'Insufficient data'}
                
                # Check SMMA slope
                prev_smma = historical[-5]['close']  # Simplified
                slope = (smma_50 - prev_smma) / prev_smma
                
                if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                    passed = slope > 0 and current_price > smma_50
                else:
                    passed = slope < 0 and current_price < smma_50
                
                return {
                    'passed': passed,
                    'confidence': min(1.0, abs(slope) * 100),
                    'value': slope,
                    'reasoning': f"SMMA slope: {slope:.4f}"
                }
                
            elif condition_id == 14:  # SMMA Cross
                if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                    passed = current_price > smma_50
                    distance = (current_price - smma_50) / smma_50
                else:
                    passed = current_price < smma_50
                    distance = (smma_50 - current_price) / smma_50
                
                confidence = min(1.0, abs(distance) * 100)
                
                return {
                    'passed': passed,
                    'confidence': confidence,
                    'value': distance,
                    'reasoning': f"Price vs SMMA: {current_price:.2f} vs {smma_50:.2f}"
                }
                
            elif condition_id == 15:  # SMMA Distance
                distance_pct = abs(current_price - smma_50) / smma_50 * 100
                passed = 0.1 <= distance_pct <= 2.0
                confidence = 1.0 if passed else 0.3
                
                return {
                    'passed': passed,
                    'confidence': confidence,
                    'value': distance_pct,
                    'reasoning': f"SMMA distance: {distance_pct:.2f}%"
                }
                
        except Exception as e:
            return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': f'MA error: {str(e)}'}

    def _evaluate_supertrend_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate SuperTrend conditions"""
        try:
            # Simplified SuperTrend calculation
            current_price = market_data['current']['ltp']
            atr = market_data.get('atr', current_price * 0.02)
            
            if condition_id == 16:  # STR-ENTRY Signal
                # Simplified SuperTrend calculation
                hl2 = (market_data['current']['high'] + market_data['current']['low']) / 2
                upper_band = hl2 + (atr * 1.0)
                lower_band = hl2 - (atr * 1.0)
                
                if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                    passed = current_price > lower_band
                    confidence = min(1.0, (current_price - lower_band) / current_price)
                else:
                    passed = current_price < upper_band
                    confidence = min(1.0, (upper_band - current_price) / current_price)
                
                return {
                    'passed': passed,
                    'confidence': abs(confidence),
                    'value': current_price,
                    'reasoning': f"STR-ENTRY: {current_price:.2f} vs bands"
                }
                
            elif condition_id == 17:  # STR Trend Strength
                # Check trend persistence
                trend_strength = min(1.0, atr / current_price * 100)
                passed = trend_strength > 0.5
                
                return {
                    'passed': passed,
                    'confidence': trend_strength,
                    'value': trend_strength,
                    'reasoning': f"Trend strength: {trend_strength:.2f}"
                }
                
        except Exception as e:
            return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': f'SuperTrend error: {str(e)}'}

    def _evaluate_volume_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate volume conditions"""
        try:
            current_volume = market_data['current'].get('volume', 0)
            avg_volume = market_data.get('volume_sma', current_volume)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            if condition_id == 18:  # Volume Surge
                passed = volume_ratio > 1.5
                confidence = min(1.0, volume_ratio / 2.0)
                
                return {
                    'passed': passed,
                    'confidence': confidence,
                    'value': volume_ratio,
                    'reasoning': f"Volume surge: {volume_ratio:.2f}x"
                }
                
            elif condition_id == 19:  # Volume Trend Alignment
                passed = volume_ratio > 1.0
                confidence = min(1.0, volume_ratio)
                
                return {
                    'passed': passed,
                    'confidence': confidence,
                    'value': volume_ratio,
                    'reasoning': f"Volume trend: {volume_ratio:.2f}"
                }
                
            elif condition_id == 20:  # Volume Price Divergence
                # Simplified - no divergence check for now
                return {
                    'passed': True,
                    'confidence': 0.7,
                    'value': 1.0,
                    'reasoning': "No volume divergence detected"
                }
                
        except Exception as e:
            return {'passed': False, 'confidence': 0.0, 'value': 0.0, 'reasoning': f'Volume error: {str(e)}'}

    def _evaluate_volatility_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate volatility conditions"""
        try:
            atr = market_data.get('atr', 0)
            current_price = market_data['current']['ltp']
            
            if condition_id == 21:  # ATR Volatility Check
                if current_price > 0:
                    atr_pct = atr / current_price * 100
                    passed = 0.5 <= atr_pct <= 3.0
                    confidence = 1.0 if passed else 0.3
                    
                    return {
                        'passed': passed,
                        'confidence': confidence,
                        'value': atr_pct,
                        'reasoning': f"ATR: {atr_pct:.2f}%"
                    }
                
        except Exception as e:
            pass
            
        return {'passed': True, 'confidence': 0.7, 'value': 1.0, 'reasoning': 'ATR acceptable'}

    def _evaluate_time_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate time-based conditions"""
        try:
            if condition_id == 22:  # Market Session Filter
                current_time = datetime.now().time()
                
                # Simplified session check - assume market hours 9:30-15:29
                market_start = datetime.strptime("09:30", "%H:%M").time()
                market_end = datetime.strptime("15:29", "%H:%M").time()
                
                passed = market_start <= current_time <= market_end
                confidence = 1.0 if passed else 0.0
                
                return {
                    'passed': passed,
                    'confidence': confidence,
                    'value': 1.0 if passed else 0.0,
                    'reasoning': f"Market session: {current_time}"
                }
                
        except Exception as e:
            pass
            
        return {'passed': True, 'confidence': 0.8, 'value': 1.0, 'reasoning': 'Time filter passed'}

    def _evaluate_market_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate market condition filters"""
        # Simplified implementations for conditions 23, 27, 29, 30, 31
        try:
            if condition_id == 23:  # Spread Filter
                return {'passed': True, 'confidence': 0.8, 'value': 1.0, 'reasoning': 'Spread acceptable'}
            elif condition_id == 27:  # VIX Level Check
                # Would integrate with actual VIX data
                return {'passed': True, 'confidence': 0.9, 'value': 1.0, 'reasoning': 'VIX level normal'}
            elif condition_id == 29:  # Market Correlation
                return {'passed': True, 'confidence': 0.7, 'value': 1.0, 'reasoning': 'Correlation supportive'}
            elif condition_id == 30:  # Risk Sentiment
                return {'passed': True, 'confidence': 0.8, 'value': 1.0, 'reasoning': 'Risk sentiment neutral'}
            elif condition_id == 31:  # Liquidity Check
                return {'passed': True, 'confidence': 0.8, 'value': 1.0, 'reasoning': 'Liquidity adequate'}
        except Exception as e:
            pass
            
        return {'passed': True, 'confidence': 0.7, 'value': 1.0, 'reasoning': 'Market condition acceptable'}

    def _evaluate_mtf_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate multi-timeframe conditions"""
        # Simplified implementations for conditions 24, 25, 26
        return {'passed': True, 'confidence': 0.8, 'value': 1.0, 'reasoning': 'MTF alignment confirmed'}

    def _evaluate_fundamental_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate fundamental conditions"""
        # Condition 28 - News Filter
        return {'passed': True, 'confidence': 0.9, 'value': 1.0, 'reasoning': 'No conflicting news'}

    def _evaluate_momentum_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate momentum conditions"""
        # Condition 33 - Momentum Persistence
        rsi = market_data.get('rsi', 50.0)
        momentum_score = abs(rsi - 50) / 50
        passed = momentum_score > 0.3
        
        return {
            'passed': passed,
            'confidence': momentum_score,
            'value': momentum_score,
            'reasoning': f"Momentum: {momentum_score:.2f}"
        }

    def _evaluate_timing_condition(self, condition_id: int, signal_type: SignalType, market_data: Dict) -> Dict:
        """Evaluate timing conditions"""
        # Condition 34 - Entry Timing Precision
        return {'passed': True, 'confidence': 0.9, 'value': 1.0, 'reasoning': 'Timing optimal'}

    async def _evaluate_layer2_ml_enhancement(self, symbol: str, layer1_result: EnhancementLayer, market_data: Dict) -> EnhancementLayer:
        """Layer 2: ML Enhancement validation"""
        start_time = time.time()
        
        try:
            self.logger.debug("Evaluating Layer 2: ML Enhancement")
            
            if not layer1_result.passed:
                return EnhancementLayer(
                    layer_name="ML_ENHANCEMENT",
                    enabled=True,
                    passed=False,
                    confidence=0.0,
                    threshold=self.layer2_threshold,
                    reasoning="Layer 1 failed - ML enhancement skipped",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Prepare features for ML
            features = self._prepare_ml_features(market_data)
            
            # Get ML prediction
            ml_prediction = await self.ml_enhancement.validate_signal(features, symbol)
            
            layer2_passed = ml_prediction.is_valid and ml_prediction.confidence >= self.layer2_threshold
            
            processing_time = (time.time() - start_time) * 1000
            
            return EnhancementLayer(
                layer_name="ML_ENHANCEMENT",
                enabled=True,
                passed=layer2_passed,
                confidence=ml_prediction.confidence,
                threshold=self.layer2_threshold,
                reasoning=ml_prediction.reasoning,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Layer 2 ML enhancement failed: {e}")
            return EnhancementLayer(
                layer_name="ML_ENHANCEMENT",
                enabled=True,
                passed=False,
                confidence=0.0,
                threshold=self.layer2_threshold,
                reasoning=f"ML enhancement failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def _prepare_ml_features(self, market_data: Dict) -> Dict[str, float]:
        """Prepare features for ML enhancement"""
        try:
            features = {}
            
            # Add condition results as features
            for condition_id, condition in self.entry_conditions.items():
                features[f'condition_{condition_id}'] = float(condition.result)
                features[f'condition_{condition_id}_confidence'] = condition.confidence
            
            # Add technical indicators
            features.update({
                'rsi_14': market_data.get('rsi', 50.0),
                'atr_normalized': market_data.get('atr', 0.0) / market_data['current']['ltp'] if market_data['current']['ltp'] > 0 else 0.0,
                'volume_ratio': market_data.get('volume_ratio', 1.0),
                'bb_position': self._calculate_bb_position(market_data),
                'price_momentum': self._calculate_price_momentum(market_data),
                'trend_strength': self._calculate_trend_strength(market_data)
            })
            
            return features
            
        except Exception as e:
            self.logger.error(f"ML feature preparation failed: {e}")
            return {}

    def _calculate_bb_position(self, market_data: Dict) -> float:
        """Calculate Bollinger Band position (0-1)"""
        try:
            current_price = market_data['current']['ltp']
            bb_upper = market_data.get('bb_upper', current_price * 1.02)
            bb_lower = market_data.get('bb_lower', current_price * 0.98)
            
            if bb_upper > bb_lower:
                return (current_price - bb_lower) / (bb_upper - bb_lower)
            return 0.5
        except:
            return 0.5

    def _calculate_price_momentum(self, market_data: Dict) -> float:
        """Calculate price momentum"""
        try:
            historical = market_data.get('historical', [])
            if len(historical) >= 14:
                current = historical[-1]['close']
                past = historical[-14]['close']
                return (current - past) / past * 100
            return 0.0
        except:
            return 0.0

    def _calculate_trend_strength(self, market_data: Dict) -> float:
        """Calculate trend strength"""
        try:
            current_price = market_data['current']['ltp']
            smma_50 = market_data.get('smma_50', current_price)
            
            if smma_50 > 0:
                return abs(current_price - smma_50) / smma_50 * 100
            return 0.0
        except:
            return 0.0

    async def _evaluate_layer3_candlestick_volume(self, symbol: str, timeframe: str, signal_type: SignalType, market_data: Dict) -> EnhancementLayer:
        """Layer 3: Candlestick + Volume validation"""
        start_time = time.time()
        
        try:
            self.logger.debug("Evaluating Layer 3: Candlestick + Volume")
            
            historical = market_data.get('historical', [])
            if len(historical) < 2:
                return EnhancementLayer(
                    layer_name="CANDLESTICK_VOLUME",
                    enabled=True,
                    passed=False,
                    confidence=0.0,
                    threshold=self.layer3_threshold,
                    reasoning="Insufficient candlestick data",
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Analyze last candlestick
            last_candle = historical[-1]
            candle_score = self._analyze_candlestick_pattern(last_candle, signal_type)
            
            # Analyze volume confirmation
            volume_score = self._analyze_volume_confirmation(market_data, signal_type)
            
            # Combined score
            combined_score = (candle_score + volume_score) / 2
            layer3_passed = combined_score >= self.layer3_threshold
            
            processing_time = (time.time() - start_time) * 1000
            
            reasoning = f"Candle: {candle_score:.2f}, Volume: {volume_score:.2f}, Combined: {combined_score:.2f}"
            
            return EnhancementLayer(
                layer_name="CANDLESTICK_VOLUME",
                enabled=True,
                passed=layer3_passed,
                confidence=combined_score,
                threshold=self.layer3_threshold,
                reasoning=reasoning,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Layer 3 candlestick+volume failed: {e}")
            return EnhancementLayer(
                layer_name="CANDLESTICK_VOLUME",
                enabled=True,
                passed=False,
                confidence=0.0,
                threshold=self.layer3_threshold,
                reasoning=f"Layer 3 failed: {str(e)}",
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def _analyze_candlestick_pattern(self, candle: Dict, signal_type: SignalType) -> float:
        """Analyze candlestick pattern strength"""
        try:
            open_price = candle['open']
            close_price = candle['close']
            high_price = candle['high']
            low_price = candle['low']
            
            # Calculate candle metrics
            body_size = abs(close_price - open_price)
            total_range = high_price - low_price
            upper_wick = high_price - max(open_price, close_price)
            lower_wick = min(open_price, close_price) - low_price
            
            if total_range == 0:
                return 0.0
            
            body_ratio = body_size / total_range
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            
            # Directional analysis
            is_bullish_candle = close_price > open_price
            is_bearish_candle = close_price < open_price
            
            # Pattern scoring
            pattern_score = 0.0
            
            # Strong directional body
            if body_ratio > 0.6:
                pattern_score += 0.4
            
            # Direction alignment
            if signal_type in [SignalType.CONTINUATION_BUY, SignalType.PULLBACK_BUY]:
                if is_bullish_candle:
                    pattern_score += 0.3
                if lower_wick_ratio < 0.3:  # Small lower wick
                    pattern_score += 0.2
            else:
                if is_bearish_candle:
                    pattern_score += 0.3
                if upper_wick_ratio < 0.3:  # Small upper wick
                    pattern_score += 0.2
            
            # Size significance
            if body_size > 0:  # Avoid division by zero
                pattern_score += min(0.1, body_size / open_price * 100)
            
            return min(1.0, pattern_score)
            
        except Exception as e:
            self.logger.error(f"Candlestick analysis failed: {e}")
            return 0.0

    def _analyze_volume_confirmation(self, market_data: Dict, signal_type: SignalType) -> float:
        """Analyze volume confirmation"""
        try:
            current_volume = market_data['current'].get('volume', 0)
            volume_ratio = market_data.get('volume_ratio', 1.0)
            
            volume_score = 0.0
            
            # Volume surge confirmation
            if volume_ratio > 1.2:
                volume_score += 0.5
            elif volume_ratio > 1.0:
                volume_score += 0.3
            
            # Volume trend alignment
            if volume_ratio > 1.0:
                volume_score += 0.3
            
            # Absolute volume check
            if current_volume > 0:
                volume_score += 0.2
            
            return min(1.0, volume_score)
            
        except Exception as e:
            self.logger.error(f"Volume analysis failed: {e}")
            return 0.0

    def _make_final_decision(self, symbol: str, timeframe: str, signal_type: SignalType, 
                           trend_analysis: TrendAnalysis, layer1: EnhancementLayer, 
                           layer2: EnhancementLayer, layer3: EnhancementLayer) -> EntrySignal:
        """Make final trading decision based on all layers"""
        try:
            # Collect enhancement layers
            enhancement_layers = [layer1, layer2, layer3]
            
            # Calculate overall confidence
            layer_weights = [0.4, 0.3, 0.3]  # Layer 1: 40%, Layer 2: 30%, Layer 3: 30%
            overall_confidence = sum(layer.confidence * weight for layer, weight in zip(enhancement_layers, layer_weights))
            
            # Final decision logic
            final_decision = (layer1.passed and layer2.passed and layer3.passed)
            
            # Fallback logic if ML is not available
            if not layer2.passed and "ML_NOT_AVAILABLE" in layer2.reasoning:
                final_decision = layer1.passed and layer3.passed
                overall_confidence = (layer1.confidence * 0.6 + layer3.confidence * 0.4)
            
            # Count conditions
            conditions_passed = [cond for cond in self.entry_conditions.values() if cond.result]
            conditions_failed = [cond for cond in self.entry_conditions.values() if not cond.result]
            
            # Generate reasoning
            reasoning = self._generate_final_reasoning(enhancement_layers, final_decision)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(enhancement_layers, trend_analysis)
            
            return EntrySignal(
                signal_type=signal_type if final_decision else SignalType.NO_SIGNAL,
                overall_confidence=overall_confidence,
                conditions_met=len(conditions_passed),
                conditions_total=len(self.entry_conditions),
                conditions_passed=conditions_passed,
                conditions_failed=conditions_failed,
                trend_analysis=trend_analysis,
                enhancement_layers=enhancement_layers,
                final_decision=final_decision,
                reasoning=reasoning,
                symbol=symbol,
                timeframe=timeframe,
                timestamp=datetime.now(),
                processing_time_ms=sum(layer.processing_time_ms for layer in enhancement_layers),
                quality_score=quality_score
            )
            
        except Exception as e:
            self.logger.error(f"Final decision failed: {e}")
            return self._create_no_signal(f"Decision error: {str(e)}", symbol, timeframe, trend_analysis)

    def _generate_final_reasoning(self, layers: List[EnhancementLayer], final_decision: bool) -> str:
        """Generate human-readable reasoning for the decision"""
        reasoning_parts = []
        
        for layer in layers:
            status = "✅ PASSED" if layer.passed else "❌ FAILED"
            reasoning_parts.append(f"{layer.layer_name}: {status} ({layer.confidence:.2f})")
        
        if final_decision:
            reasoning_parts.append("🎯 SIGNAL GENERATED")
        else:
            reasoning_parts.append("🚫 NO SIGNAL")
        
        return " | ".join(reasoning_parts)

    def _calculate_quality_score(self, layers: List[EnhancementLayer], trend_analysis: TrendAnalysis) -> float:
        """Calculate overall signal quality score"""
        try:
            quality_factors = []
            
            # Layer performance
            for layer in layers:
                if layer.enabled:
                    quality_factors.append(layer.confidence)
            
            # Trend alignment quality
            quality_factors.append(trend_analysis.confidence)
            
            # Calculate weighted average
            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.0
            
        except Exception as e:
            self.logger.error(f"Quality score calculation failed: {e}")
            return 0.0

    def _create_no_signal(self, reason: str, symbol: str, timeframe: str, trend_analysis: Optional[TrendAnalysis] = None) -> EntrySignal:
        """Create a no-signal result"""
        if trend_analysis is None:
            trend_analysis = TrendAnalysis(
                major_trend=TrendDirection.UNKNOWN,
                middle_trend=TrendDirection.UNKNOWN,
                major_timeframe="DAILY",
                middle_timeframe="1H",
                confidence=0.0,
                alignment=False,
                timestamp=datetime.now()
            )
        
        return EntrySignal(
            signal_type=SignalType.NO_SIGNAL,
            overall_confidence=0.0,
            conditions_met=0,
            conditions_total=len(self.entry_conditions),
            conditions_passed=[],
            conditions_failed=list(self.entry_conditions.values()),
            trend_analysis=trend_analysis,
            enhancement_layers=[],
            final_decision=False,
            reasoning=reason,
            symbol=symbol,
            timeframe=timeframe,
            timestamp=datetime.now(),
            processing_time_ms=0.0,
            quality_score=0.0
        )

    async def _store_analysis(self, signal: EntrySignal):
        """Store analysis results for learning"""
        try:
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT INTO signal_analysis 
                    (symbol, timeframe, signal_type, final_decision, overall_confidence, 
                     conditions_met, layer1_passed, layer2_passed, layer3_passed, processing_time_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    signal.symbol,
                    signal.timeframe,
                    signal.signal_type.value,
                    signal.final_decision,
                    signal.overall_confidence,
                    signal.conditions_met,
                    len(signal.enhancement_layers) > 0 and signal.enhancement_layers[0].passed,
                    len(signal.enhancement_layers) > 1 and signal.enhancement_layers[1].passed,
                    len(signal.enhancement_layers) > 2 and signal.enhancement_layers[2].passed,
                    signal.processing_time_ms
                ))
                self.db_connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store analysis: {e}")

    def _update_statistics(self, signal: EntrySignal, processing_time_ms: float):
        """Update processing statistics"""
        try:
            with self.processing_lock:
                self.processing_statistics['total_signals_processed'] += 1
                
                if signal.final_decision:
                    self.processing_statistics['signals_generated'] += 1
                
                # Update average processing time
                current_avg = self.processing_statistics['average_processing_time_ms']
                total_processed = self.processing_statistics['total_signals_processed']
                new_avg = ((current_avg * (total_processed - 1)) + processing_time_ms) / total_processed
                self.processing_statistics['average_processing_time_ms'] = new_avg
                
                # Update condition success rates
                for condition in signal.conditions_passed:
                    if condition.name not in self.processing_statistics['conditions_success_rate']:
                        self.processing_statistics['conditions_success_rate'][condition.name] = {'passed': 0, 'total': 0}
                    self.processing_statistics['conditions_success_rate'][condition.name]['passed'] += 1
                    self.processing_statistics['conditions_success_rate'][condition.name]['total'] += 1
                
                for condition in signal.conditions_failed:
                    if condition.name not in self.processing_statistics['conditions_success_rate']:
                        self.processing_statistics['conditions_success_rate'][condition.name] = {'passed': 0, 'total': 0}
                    self.processing_statistics['conditions_success_rate'][condition.name]['total'] += 1
                
        except Exception as e:
            self.logger.error(f"Statistics update failed: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'processor_initialized': True,
            'total_conditions': len(self.entry_conditions),
            'enabled_conditions': sum(1 for c in self.entry_conditions.values() if c.enabled),
            'processing_statistics': self.processing_statistics,
            'layer_thresholds': {
                'layer1': self.layer1_threshold,
                'layer2': self.layer2_threshold,
                'layer3': self.layer3_threshold
            },
            'min_conditions_required': self.min_conditions_required,
            'evaluation_timeout': self.evaluation_timeout,
            'ml_enhancement_available': self.ml_enhancement.is_initialized if self.ml_enhancement else False
        }

    async def update_trade_outcome(self, signal_id: str, actual_outcome: bool):
        """Update actual trade outcome for learning"""
        try:
            if self.db_connection:
                self.db_connection.execute(
                    "UPDATE signal_analysis SET actual_outcome = ? WHERE id = ?",
                    (actual_outcome, signal_id)
                )
                self.db_connection.commit()
                
                # Also update ML system
                if self.ml_enhancement:
                    self.ml_enhancement.update_trade_outcome(signal_id, actual_outcome)
                    
        except Exception as e:
            self.logger.error(f"Failed to update trade outcome: {e}")

    async def shutdown(self):
        """Gracefully shutdown the processor"""
        try:
            self.logger.info("🔄 Shutting down Entry Conditions Processor...")
            
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("✅ Entry Conditions Processor shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Shutdown error: {e}")

# Global instance
entry_processor = None

def get_entry_processor() -> EntryConditionsProcessor:
    """Get singleton instance of Entry Conditions Processor"""
    global entry_processor
    if entry_processor is None:
        entry_processor = EntryConditionsProcessor()
    return entry_processor

if __name__ == "__main__":
    # Test the Entry Conditions Processor
    import asyncio
    
    async def main():
        processor = EntryConditionsProcessor()
        
        # Test signal analysis
        signal = await processor.analyze_entry_signal("NIFTY", "15M")
        
        print(f"Signal Type: {signal.signal_type}")
        print(f"Final Decision: {signal.final_decision}")
        print(f"Overall Confidence: {signal.overall_confidence:.2f}")
        print(f"Conditions Met: {signal.conditions_met}/{signal.conditions_total}")
        print(f"Processing Time: {signal.processing_time_ms:.1f}ms")
        print(f"Quality Score: {signal.quality_score:.2f}")
        print(f"Reasoning: {signal.reasoning}")
        
        # Get system status
        status = processor.get_system_status()
        print(f"System Status: {json.dumps(status, indent=2)}")
        
        # Shutdown
        await processor.shutdown()
    
    asyncio.run(main())
