#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Advanced Technical Analysis Engine
Created: 2024-08-05
Author: EA GlobalFlow Development Team
Description: Comprehensive Technical Analysis Engine

This module implements advanced technical analysis capabilities:
- Ichimoku Kinko Hyo complete system
- Ichi-Trader Dynamic Index (TDI)
- SMMA, Bollinger Bands, SuperTrend
- Multi-timeframe analysis
- Pattern recognition integration
- Signal strength calculation
- Confluence analysis
"""

import sys
import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
import sqlite3
import talib
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Config.error_handler import GlobalErrorHandler
    from Config.system_monitor import SystemMonitor
    from Config.security_manager import SecurityManager
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    
    class GlobalErrorHandler:
        @staticmethod
        def handle_error(error, context="", severity="medium"):
            print(f"ERROR [{severity}] {context}: {error}")
    
    class SystemMonitor:
        def __init__(self):
            pass
        def log_metric(self, *args, **kwargs):
            pass
    
    class SecurityManager:
        def __init__(self):
            pass

class IndicatorType(Enum):
    """Technical indicator types"""
    ICHIMOKU = "ichimoku"
    TDI = "tdi"
    SMMA = "smma"
    BOLLINGER = "bollinger"
    SUPERTREND = "supertrend"
    RSI = "rsi"
    MACD = "macd"
    STOCHASTIC = "stochastic"
    ADX = "adx"
    ATR = "atr"

class TimeFrame(Enum):
    """Time frame types"""
    M1 = "1M"
    M5 = "5M"
    M15 = "15M"
    M30 = "30M"
    H1 = "1H"
    H4 = "4H"
    D1 = "1D"
    W1 = "1W"

class SignalStrength(Enum):
    """Signal strength levels"""
    VERY_WEAK = 1
    WEAK = 2
    MODERATE = 3
    STRONG = 4
    VERY_STRONG = 5

class TrendDirection(Enum):
    """Trend direction"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    SIDEWAYS = "sideways"

@dataclass
class OHLCV:
    """OHLCV data structure"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    
    def __post_init__(self):
        """Validate OHLCV data"""
        if self.high < max(self.open, self.close):
            self.high = max(self.open, self.close)
        if self.low > min(self.open, self.close):
            self.low = min(self.open, self.close)

@dataclass
class IchimokuData:
    """Ichimoku Kinko Hyo indicators"""
    tenkan_sen: float
    kijun_sen: float
    senkou_span_a: float
    senkou_span_b: float
    chikou_span: float
    kumo_top: float
    kumo_bottom: float
    kumo_thickness: float
    kumo_color: str  # 'bullish' or 'bearish'
    price_vs_kumo: str  # 'above', 'below', 'inside'
    tk_cross: str  # 'bullish', 'bearish', 'none'
    kumo_twist: str  # 'bullish', 'bearish', 'none'

@dataclass
class TDIData:
    """Trader Dynamic Index indicators"""
    rsi: float
    rsi_ma: float
    signal_line: float
    volatility_band_high: float
    volatility_band_low: float
    market_base_line: float
    zone: str  # 'overbought', 'oversold', 'neutral'
    signal_cross: str  # 'bullish', 'bearish', 'none'
    strength: SignalStrength

@dataclass
class BollingerData:
    """Bollinger Bands indicators"""
    upper: float
    middle: float
    lower: float
    width: float
    percent_b: float
    squeeze: bool
    expansion: bool
    position: str  # 'above_upper', 'below_lower', 'normal'

@dataclass
class SuperTrendData:
    """SuperTrend indicators"""
    value: float
    direction: int  # 1 for bullish, -1 for bearish
    signal: str  # 'buy', 'sell', 'hold'
    strength: float

@dataclass
class TechnicalSignal:
    """Combined technical signal"""
    timestamp: datetime
    symbol: str
    timeframe: TimeFrame
    direction: str  # 'BUY', 'SELL', 'HOLD'
    strength: SignalStrength
    confidence: float
    indicators: Dict[str, Any]
    confluence_score: float
    risk_reward_ratio: float
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class TechnicalAnalyzer:
    """
    Advanced Technical Analysis Engine
    
    Provides comprehensive technical analysis using multiple indicators
    and timeframes with confluence analysis and signal generation.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize Technical Analyzer"""
        self.logger = logging.getLogger(__name__)
        self.error_handler = GlobalErrorHandler()
        self.monitor = SystemMonitor()
        self.security = SecurityManager()
        
        # Configuration
        self.config = self._load_config(config_path)
        self.enabled = self.config.get('enabled', True)
        
        # Database connections
        self.db_path = self.config.get('database_path', 'C:\\EA_GlobalFlow_Bridge\\Data\\technical_analysis.db')
        
        # Data storage
        self.ohlcv_data: Dict[str, Dict[TimeFrame, List[OHLCV]]] = {}
        self.indicators_cache: Dict[str, Dict[TimeFrame, Dict]] = {}
        
        # Performance tracking
        self.calculations_performed = 0
        self.signals_generated = 0
        self.last_update = datetime.now()
        
        # Threading
        self.lock = threading.Lock()
        
        # Initialize system
        self._init_database()
        self._start_background_tasks()
        
        self.logger.info("Technical Analyzer initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load technical analysis configuration"""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            
            # Default configuration
            return {
                'enabled': True,
                'indicators': {
                    'ichimoku': {
                        'enabled': True,
                        'tenkan_period': 9,
                        'kijun_period': 26,
                        'senkou_span_b_period': 52,
                        'displacement': 26
                    },
                    'tdi': {
                        'enabled': True,
                        'rsi_period': 13,
                        'rsi_ma_period': 2,
                        'signal_period': 7,
                        'volatility_band': 1.6185,
                        'overbought': 68,
                        'oversold': 32
                    },
                    'smma': {
                        'enabled': True,
                        'period': 50
                    },
                    'bollinger': {
                        'enabled': True,
                        'period': 20,
                        'deviation': 2.0,
                        'squeeze_threshold': 0.1
                    },
                    'supertrend': {
                        'enabled': True,
                        'atr_period': 20,
                        'multiplier': 1.0
                    },
                    'rsi': {
                        'enabled': True,
                        'period': 14,
                        'overbought': 70,
                        'oversold': 30
                    },
                    'macd': {
                        'enabled': True,
                        'fast_period': 12,
                        'slow_period': 26,
                        'signal_period': 9
                    },
                    'atr': {
                        'enabled': True,
                        'period': 14
                    },
                    'adx': {
                        'enabled': True,
                        'period': 14,
                        'threshold': 25
                    }
                },
                'timeframes': {
                    'primary': ['5M', '15M', '1H'],
                    'confirmation': ['30M', '4H', '1D'],
                    'max_bars': 500
                },
                'signals': {
                    'min_confluence_score': 0.6,
                    'min_strength': 3,
                    'risk_reward_min': 1.5
                },
                'performance': {
                    'cache_duration_minutes': 5,
                    'max_cache_size': 1000
                }
            }
        except Exception as e:
            self.error_handler.handle_error(e, "Technical analysis config loading", "medium")
            return {}
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS ohlcv_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        open REAL,
                        high REAL,
                        low REAL,
                        close REAL,
                        volume REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timeframe, timestamp)
                    );
                    
                    CREATE TABLE IF NOT EXISTS technical_signals (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        direction TEXT NOT NULL,
                        strength INTEGER,
                        confidence REAL,
                        confluence_score REAL,
                        entry_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        risk_reward_ratio REAL,
                        indicators TEXT,
                        metadata TEXT,
                        outcome TEXT,
                        actual_return REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS indicator_values (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        timeframe TEXT NOT NULL,
                        timestamp DATETIME NOT NULL,
                        indicator_type TEXT NOT NULL,
                        values TEXT NOT NULL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, timeframe, timestamp, indicator_type)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_ohlcv_symbol_timeframe 
                    ON ohlcv_data(symbol, timeframe, timestamp);
                    
                    CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp 
                    ON technical_signals(symbol, timestamp);
                    
                    CREATE INDEX IF NOT EXISTS idx_indicators_symbol_type 
                    ON indicator_values(symbol, indicator_type, timestamp);
                """)
            
            self.logger.info("Technical analysis database initialized successfully")
        except Exception as e:
            self.error_handler.handle_error(e, "Technical analysis database initialization", "high")
    
    def add_ohlcv_data(self, symbol: str, timeframe: TimeFrame, data: List[OHLCV]):
        """Add OHLCV data for analysis"""
        try:
            with self.lock:
                if symbol not in self.ohlcv_data:
                    self.ohlcv_data[symbol] = {}
                
                if timeframe not in self.ohlcv_data[symbol]:
                    self.ohlcv_data[symbol][timeframe] = []
                
                # Add new data and sort by timestamp
                self.ohlcv_data[symbol][timeframe].extend(data)
                self.ohlcv_data[symbol][timeframe].sort(key=lambda x: x.timestamp)
                
                # Keep only the last N bars
                max_bars = self.config.get('timeframes', {}).get('max_bars', 500)
                if len(self.ohlcv_data[symbol][timeframe]) > max_bars:
                    self.ohlcv_data[symbol][timeframe] = self.ohlcv_data[symbol][timeframe][-max_bars:]
                
                # Clear cache for this symbol/timeframe
                if symbol in self.indicators_cache and timeframe in self.indicators_cache[symbol]:
                    del self.indicators_cache[symbol][timeframe]
                
                # Store to database
                self._store_ohlcv_data(symbol, timeframe, data)
                
                self.logger.debug(f"Added {len(data)} OHLCV bars for {symbol} {timeframe.value}")
                
        except Exception as e:
            self.error_handler.handle_error(e, f"OHLCV data addition {symbol}", "medium")
    
    def calculate_all_indicators(self, symbol: str, timeframe: TimeFrame) -> Optional[Dict]:
        """Calculate all technical indicators for symbol/timeframe"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{timeframe.value}"
            if (symbol in self.indicators_cache and 
                timeframe in self.indicators_cache[symbol]):
                
                cached_data = self.indicators_cache[symbol][timeframe]
                cache_time = cached_data.get('timestamp', datetime.min)
                cache_duration = self.config.get('performance', {}).get('cache_duration_minutes', 5)
                
                if datetime.now() - cache_time < timedelta(minutes=cache_duration):
                    return cached_data['indicators']
            
            # Get OHLCV data
            if symbol not in self.ohlcv_data or timeframe not in self.ohlcv_data[symbol]:
                return None
            
            ohlcv_list = self.ohlcv_data[symbol][timeframe]
            if len(ohlcv_list) < 52:  # Minimum for Ichimoku
                return None
            
            # Convert to pandas DataFrame for easier calculation
            df = pd.DataFrame([
                {
                    'timestamp': bar.timestamp,
                    'open': bar.open,
                    'high': bar.high,
                    'low': bar.low,
                    'close': bar.close,
                    'volume': bar.volume
                }
                for bar in ohlcv_list
            ])
            
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            # Calculate all indicators
            indicators = {}
            
            # Ichimoku
            if self.config.get('indicators', {}).get('ichimoku', {}).get('enabled', True):
                indicators['ichimoku'] = self._calculate_ichimoku(df)
            
            # TDI
            if self.config.get('indicators', {}).get('tdi', {}).get('enabled', True):
                indicators['tdi'] = self._calculate_tdi(df)
            
            # SMMA
            if self.config.get('indicators', {}).get('smma', {}).get('enabled', True):
                indicators['smma'] = self._calculate_smma(df)
            
            # Bollinger Bands
            if self.config.get('indicators', {}).get('bollinger', {}).get('enabled', True):
                indicators['bollinger'] = self._calculate_bollinger(df)
            
            # SuperTrend
            if self.config.get('indicators', {}).get('supertrend', {}).get('enabled', True):
                indicators['supertrend'] = self._calculate_supertrend(df)
            
            # RSI
            if self.config.get('indicators', {}).get('rsi', {}).get('enabled', True):
                indicators['rsi'] = self._calculate_rsi(df)
            
            # MACD
            if self.config.get('indicators', {}).get('macd', {}).get('enabled', True):
                indicators['macd'] = self._calculate_macd(df)
            
            # ATR
            if self.config.get('indicators', {}).get('atr', {}).get('enabled', True):
                indicators['atr'] = self._calculate_atr(df)
            
            # ADX
            if self.config.get('indicators', {}).get('adx', {}).get('enabled', True):
                indicators['adx'] = self._calculate_adx(df)
            
            # Cache results
            if symbol not in self.indicators_cache:
                self.indicators_cache[symbol] = {}
            
            self.indicators_cache[symbol][timeframe] = {
                'timestamp': datetime.now(),
                'indicators': indicators
            }
            
            # Store to database
            self._store_indicator_values(symbol, timeframe, indicators)
            
            self.calculations_performed += 1
            return indicators
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Indicator calculation {symbol} {timeframe.value}", "medium")
            return None
    
    def _calculate_ichimoku(self, df: pd.DataFrame) -> IchimokuData:
        """Calculate Ichimoku Kinko Hyo indicators"""
        try:
            config = self.config.get('indicators', {}).get('ichimoku', {})
            tenkan_period = config.get('tenkan_period', 9)
            kijun_period = config.get('kijun_period', 26)
            senkou_b_period = config.get('senkou_span_b_period', 52)
            displacement = config.get('displacement', 26)
            
            # Tenkan-sen (Conversion Line)
            tenkan_high = df['high'].rolling(window=tenkan_period).max()
            tenkan_low = df['low'].rolling(window=tenkan_period).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = df['high'].rolling(window=kijun_period).max()
            kijun_low = df['low'].rolling(window=kijun_period).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(displacement)
            
            # Senkou Span B (Leading Span B)
            senkou_high = df['high'].rolling(window=senkou_b_period).max()
            senkou_low = df['low'].rolling(window=senkou_b_period).min()
            senkou_span_b = ((senkou_high + senkou_low) / 2).shift(displacement)
            
            # Chikou Span (Lagging Span)
            chikou_span = df['close'].shift(-displacement)
            
            # Current values (last row)
            current_idx = len(df) - 1
            current_price = df['close'].iloc[current_idx]
            
            tenkan_val = tenkan_sen.iloc[current_idx] if not pd.isna(tenkan_sen.iloc[current_idx]) else 0
            kijun_val = kijun_sen.iloc[current_idx] if not pd.isna(kijun_sen.iloc[current_idx]) else 0
            
            # For current analysis, use present values of Senkou spans
            senkou_a_val = senkou_span_a.iloc[current_idx - displacement] if current_idx >= displacement else 0
            senkou_b_val = senkou_span_b.iloc[current_idx - displacement] if current_idx >= displacement else 0
            
            chikou_val = chikou_span.iloc[current_idx] if not pd.isna(chikou_span.iloc[current_idx]) else 0
            
            # Kumo (Cloud) analysis
            kumo_top = max(senkou_a_val, senkou_b_val)
            kumo_bottom = min(senkou_a_val, senkou_b_val)
            kumo_thickness = kumo_top - kumo_bottom
            kumo_color = 'bullish' if senkou_a_val > senkou_b_val else 'bearish'
            
            # Price vs Kumo
            if current_price > kumo_top:
                price_vs_kumo = 'above'
            elif current_price < kumo_bottom:
                price_vs_kumo = 'below'
            else:
                price_vs_kumo = 'inside'
            
            # TK Cross
            tk_cross = 'none'
            if len(tenkan_sen) > 1 and len(kijun_sen) > 1:
                prev_tk_diff = tenkan_sen.iloc[current_idx - 1] - kijun_sen.iloc[current_idx - 1]
                curr_tk_diff = tenkan_val - kijun_val
                
                if prev_tk_diff <= 0 and curr_tk_diff > 0:
                    tk_cross = 'bullish'
                elif prev_tk_diff >= 0 and curr_tk_diff < 0:
                    tk_cross = 'bearish'
            
            # Kumo Twist
            kumo_twist = 'none'
            if current_idx >= displacement + 1:
                prev_a = senkou_span_a.iloc[current_idx - displacement - 1]
                prev_b = senkou_span_b.iloc[current_idx - displacement - 1]
                
                if not pd.isna(prev_a) and not pd.isna(prev_b):
                    if prev_a <= prev_b and senkou_a_val > senkou_b_val:
                        kumo_twist = 'bullish'
                    elif prev_a >= prev_b and senkou_a_val < senkou_b_val:
                        kumo_twist = 'bearish'
            
            return IchimokuData(
                tenkan_sen=tenkan_val,
                kijun_sen=kijun_val,
                senkou_span_a=senkou_a_val,
                senkou_span_b=senkou_b_val,
                chikou_span=chikou_val,
                kumo_top=kumo_top,
                kumo_bottom=kumo_bottom,
                kumo_thickness=kumo_thickness,
                kumo_color=kumo_color,
                price_vs_kumo=price_vs_kumo,
                tk_cross=tk_cross,
                kumo_twist=kumo_twist
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, "Ichimoku calculation", "medium")
            return IchimokuData(0, 0, 0, 0, 0, 0, 0, 0, 'bearish', 'inside', 'none', 'none')
    
    def _calculate_tdi(self, df: pd.DataFrame) -> TDIData:
        """Calculate Trader Dynamic Index (TDI)"""
        try:
            config = self.config.get('indicators', {}).get('tdi', {})
            rsi_period = config.get('rsi_period', 13)
            rsi_ma_period = config.get('rsi_ma_period', 2)
            signal_period = config.get('signal_period', 7)
            volatility_band = config.get('volatility_band', 1.6185)
            overbought = config.get('overbought', 68)
            oversold = config.get('oversold', 32)
            
            # Calculate RSI
            rsi = talib.RSI(df['close'], timeperiod=rsi_period)
            
            # RSI Moving Average (Price Line)
            rsi_ma = talib.SMA(rsi, timeperiod=rsi_ma_period)
            
            # Signal Line
            signal_line = talib.SMA(rsi, timeperiod=signal_period)
            
            # Volatility Bands
            rsi_std = rsi.rolling(window=signal_period).std()
            volatility_band_high = 50 + (volatility_band * rsi_std)
            volatility_band_low = 50 - (volatility_band * rsi_std)
            
            # Market Base Line
            market_base_line = 50
            
            # Current values
            current_idx = len(df) - 1
            rsi_val = rsi.iloc[current_idx] if not pd.isna(rsi.iloc[current_idx]) else 50
            rsi_ma_val = rsi_ma.iloc[current_idx] if not pd.isna(rsi_ma.iloc[current_idx]) else 50
            signal_val = signal_line.iloc[current_idx] if not pd.isna(signal_line.iloc[current_idx]) else 50
            vb_high = volatility_band_high.iloc[current_idx] if not pd.isna(volatility_band_high.iloc[current_idx]) else overbought
            vb_low = volatility_band_low.iloc[current_idx] if not pd.isna(volatility_band_low.iloc[current_idx]) else oversold
            
            # Zone determination
            if rsi_val > vb_high:
                zone = 'overbought'
            elif rsi_val < vb_low:
                zone = 'oversold'
            else:
                zone = 'neutral'
            
            # Signal cross
            signal_cross = 'none'
            if len(rsi_ma) > 1 and len(signal_line) > 1:
                prev_diff = rsi_ma.iloc[current_idx - 1] - signal_line.iloc[current_idx - 1]
                curr_diff = rsi_ma_val - signal_val
                
                if prev_diff <= 0 and curr_diff > 0:
                    signal_cross = 'bullish'
                elif prev_diff >= 0 and curr_diff < 0:
                    signal_cross = 'bearish'
            
            # Strength calculation
            strength_score = abs(rsi_ma_val - signal_val) / 10  # Normalize
            if strength_score >= 4:
                strength = SignalStrength.VERY_STRONG
            elif strength_score >= 3:
                strength = SignalStrength.STRONG
            elif strength_score >= 2:
                strength = SignalStrength.MODERATE
            elif strength_score >= 1:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            return TDIData(
                rsi=rsi_val,
                rsi_ma=rsi_ma_val,
                signal_line=signal_val,
                volatility_band_high=vb_high,
                volatility_band_low=vb_low,
                market_base_line=market_base_line,
                zone=zone,
                signal_cross=signal_cross,
                strength=strength
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, "TDI calculation", "medium")
            return TDIData(50, 50, 50, 68, 32, 50, 'neutral', 'none', SignalStrength.WEAK)
    
    def _calculate_smma(self, df: pd.DataFrame) -> float:
        """Calculate Smoothed Moving Average (SMMA)"""
        try:
            period = self.config.get('indicators', {}).get('smma', {}).get('period', 50)
            
            # SMMA calculation
            closes = df['close'].values
            smma = np.zeros(len(closes))
            
            # Initialize with SMA
            smma[period-1] = np.mean(closes[:period])
            
            # Calculate SMMA
            for i in range(period, len(closes)):
                smma[i] = (smma[i-1] * (period - 1) + closes[i]) / period
            
            return smma[-1] if len(smma) > 0 else 0
            
        except Exception as e:
            self.error_handler.handle_error(e, "SMMA calculation", "low")
            return 0
    
    def _calculate_bollinger(self, df: pd.DataFrame) -> BollingerData:
        """Calculate Bollinger Bands"""
        try:
            config = self.config.get('indicators', {}).get('bollinger', {})
            period = config.get('period', 20)
            deviation = config.get('deviation', 2.0)
            squeeze_threshold = config.get('squeeze_threshold', 0.1)
            
            # Calculate Bollinger Bands
            middle = talib.SMA(df['close'], timeperiod=period)
            std = df['close'].rolling(window=period).std()
            upper = middle + (std * deviation)
            lower = middle - (std * deviation)
            
            # Current values
            current_idx = len(df) - 1
            current_price = df['close'].iloc[current_idx]
            
            upper_val = upper.iloc[current_idx] if not pd.isna(upper.iloc[current_idx]) else current_price * 1.02
            middle_val = middle.iloc[current_idx] if not pd.isna(middle.iloc[current_idx]) else current_price
            lower_val = lower.iloc[current_idx] if not pd.isna(lower.iloc[current_idx]) else current_price * 0.98
            
            # Width and %B
            width = (upper_val - lower_val) / middle_val if middle_val > 0 else 0
            percent_b = (current_price - lower_val) / (upper_val - lower_val) if upper_val > lower_val else 0.5
            
            # Squeeze detection
            squeeze = width < squeeze_threshold
            
            # Expansion detection (width increasing)
            expansion = False
            if len(upper) > 1 and len(lower) > 1 and len(middle) > 1:
                prev_width = (upper.iloc[current_idx - 1] - lower.iloc[current_idx - 1]) / middle.iloc[current_idx - 1]
                expansion = width > prev_width * 1.1  # 10% increase
            
            # Position relative to bands
            if current_price > upper_val:
                position = 'above_upper'
            elif current_price < lower_val:
                position = 'below_lower'
            else:
                position = 'normal'
            
            return BollingerData(
                upper=upper_val,
                middle=middle_val,
                lower=lower_val,
                width=width,
                percent_b=percent_b,
                squeeze=squeeze,
                expansion=expansion,
                position=position
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, "Bollinger calculation", "medium")
            price = df['close'].iloc[-1] if len(df) > 0 else 100
            return BollingerData(price * 1.02, price, price * 0.98, 0.02, 0.5, False, False, 'normal')
    
    def _calculate_supertrend(self, df: pd.DataFrame) -> SuperTrendData:
        """Calculate SuperTrend indicator"""
        try:
            config = self.config.get('indicators', {}).get('supertrend', {})
            atr_period = config.get('atr_period', 20)
            multiplier = config.get('multiplier', 1.0)
            
            # Calculate ATR
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
            
            # Calculate basic SuperTrend
            high_low_avg = (df['high'] + df['low']) / 2
            upper_band = high_low_avg + (multiplier * atr)
            lower_band = high_low_avg - (multiplier * atr)
            
            # Initialize SuperTrend arrays
            supertrend = np.zeros(len(df))
            direction = np.ones(len(df))  # 1 for bullish, -1 for bearish
            
            for i in range(1, len(df)):
                # Upper band calculation
                if pd.isna(upper_band.iloc[i]):
                    upper_band.iloc[i] = upper_band.iloc[i-1]
                else:
                    if upper_band.iloc[i] < upper_band.iloc[i-1] or df['close'].iloc[i-1] > upper_band.iloc[i-1]:
                        upper_band.iloc[i] = upper_band.iloc[i]
                    else:
                        upper_band.iloc[i] = upper_band.iloc[i-1]
                
                # Lower band calculation
                if pd.isna(lower_band.iloc[i]):
                    lower_band.iloc[i] = lower_band.iloc[i-1]
                else:
                    if lower_band.iloc[i] > lower_band.iloc[i-1] or df['close'].iloc[i-1] < lower_band.iloc[i-1]:
                        lower_band.iloc[i] = lower_band.iloc[i]
                    else:
                        lower_band.iloc[i] = lower_band.iloc[i-1]
                
                # SuperTrend calculation
                if supertrend[i-1] == upper_band.iloc[i-1] and df['close'].iloc[i] <= upper_band.iloc[i]:
                    supertrend[i] = upper_band.iloc[i]
                    direction[i] = -1
                elif supertrend[i-1] == upper_band.iloc[i-1] and df['close'].iloc[i] > upper_band.iloc[i]:
                    supertrend[i] = lower_band.iloc[i]
                    direction[i] = 1
                elif supertrend[i-1] == lower_band.iloc[i-1] and df['close'].iloc[i] >= lower_band.iloc[i]:
                    supertrend[i] = lower_band.iloc[i]
                    direction[i] = 1
                elif supertrend[i-1] == lower_band.iloc[i-1] and df['close'].iloc[i] < lower_band.iloc[i]:
                    supertrend[i] = upper_band.iloc[i]
                    direction[i] = -1
                else:
                    supertrend[i] = supertrend[i-1]
                    direction[i] = direction[i-1]
            
            # Current values
            current_idx = len(df) - 1
            current_value = supertrend[current_idx]
            current_direction = direction[current_idx]
            current_price = df['close'].iloc[current_idx]
            
            # Signal determination
            if current_direction == 1 and direction[current_idx - 1] == -1:
                signal = 'buy'
            elif current_direction == -1 and direction[current_idx - 1] == 1:
                signal = 'sell'
            else:
                signal = 'hold'
            
            # Strength calculation
            price_distance = abs(current_price - current_value) / current_price
            strength = min(1.0, price_distance * 10)  # Normalize to 0-1
            
            return SuperTrendData(
                value=current_value,
                direction=int(current_direction),
                signal=signal,
                strength=strength
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, "SuperTrend calculation", "medium")
            price = df['close'].iloc[-1] if len(df) > 0 else 100
            return SuperTrendData(price, 1, 'hold', 0.5)
    
    def _calculate_rsi(self, df: pd.DataFrame) -> Dict:
        """Calculate RSI indicator"""
        try:
            config = self.config.get('indicators', {}).get('rsi', {})
            period = config.get('period', 14)
            overbought = config.get('overbought', 70)
            oversold = config.get('oversold', 30)
            
            rsi = talib.RSI(df['close'], timeperiod=period)
            current_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
            
            # Determine zone
            if current_rsi > overbought:
                zone = 'overbought'
            elif current_rsi < oversold:
                zone = 'oversold'
            else:
                zone = 'neutral'
            
            # Divergence detection (simplified)
            divergence = 'none'
            if len(rsi) > 10:
                price_trend = np.polyfit(range(10), df['close'].iloc[-10:], 1)[0]
                rsi_trend = np.polyfit(range(10), rsi.iloc[-10:], 1)[0]
                
                if price_trend > 0 and rsi_trend < 0:
                    divergence = 'bearish'
                elif price_trend < 0 and rsi_trend > 0:
                    divergence = 'bullish'
            
            return {
                'value': current_rsi,
                'zone': zone,
                'divergence': divergence,
                'overbought_level': overbought,
                'oversold_level': oversold
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, "RSI calculation", "low")
            return {'value': 50, 'zone': 'neutral', 'divergence': 'none', 'overbought_level': 70, 'oversold_level': 30}
    
    def _calculate_macd(self, df: pd.DataFrame) -> Dict:
        """Calculate MACD indicator"""
        try:
            config = self.config.get('indicators', {}).get('macd', {})
            fast_period = config.get('fast_period', 12)
            slow_period = config.get('slow_period', 26)
            signal_period = config.get('signal_period', 9)
            
            macd, signal, histogram = talib.MACD(df['close'], 
                                              fastperiod=fast_period,
                                              slowperiod=slow_period, 
                                              signalperiod=signal_period)
            
            current_idx = len(df) - 1
            macd_val = macd.iloc[current_idx] if not pd.isna(macd.iloc[current_idx]) else 0
            signal_val = signal.iloc[current_idx] if not pd.isna(signal.iloc[current_idx]) else 0
            histogram_val = histogram.iloc[current_idx] if not pd.isna(histogram.iloc[current_idx]) else 0
            
            # Signal cross
            cross = 'none'
            if len(macd) > 1 and len(signal) > 1:
                prev_diff = macd.iloc[current_idx - 1] - signal.iloc[current_idx - 1]
                curr_diff = macd_val - signal_val
                
                if prev_diff <= 0 and curr_diff > 0:
                    cross = 'bullish'
                elif prev_diff >= 0 and curr_diff < 0:
                    cross = 'bearish'
            
            return {
                'macd': macd_val,
                'signal': signal_val,
                'histogram': histogram_val,
                'cross': cross
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, "MACD calculation", "low")
            return {'macd': 0, 'signal': 0, 'histogram': 0, 'cross': 'none'}
    
    def _calculate_atr(self, df: pd.DataFrame) -> float:
        """Calculate Average True Range (ATR)"""
        try:
            period = self.config.get('indicators', {}).get('atr', {}).get('period', 14)
            atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=period)
            return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0
        except Exception as e:
            self.error_handler.handle_error(e, "ATR calculation", "low")
            return 0
    
    def _calculate_adx(self, df: pd.DataFrame) -> Dict:
        """Calculate Average Directional Index (ADX)"""
        try:
            config = self.config.get('indicators', {}).get('adx', {})
            period = config.get('period', 14)
            threshold = config.get('threshold', 25)
            
            adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=period)
            plus_di = talib.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=period)
            minus_di = talib.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=period)
            
            current_idx = len(df) - 1
            adx_val = adx.iloc[current_idx] if not pd.isna(adx.iloc[current_idx]) else 25
            plus_di_val = plus_di.iloc[current_idx] if not pd.isna(plus_di.iloc[current_idx]) else 25
            minus_di_val = minus_di.iloc[current_idx] if not pd.isna(minus_di.iloc[current_idx]) else 25
            
            # Trend strength
            if adx_val > threshold:
                if plus_di_val > minus_di_val:
                    trend = 'bullish'
                else:
                    trend = 'bearish'
                strength = 'strong'
            else:
                trend = 'sideways'
                strength = 'weak'
            
            return {
                'adx': adx_val,
                'plus_di': plus_di_val,
                'minus_di': minus_di_val,
                'trend': trend,
                'strength': strength
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, "ADX calculation", "low")
            return {'adx': 25, 'plus_di': 25, 'minus_di': 25, 'trend': 'sideways', 'strength': 'weak'}
    
    def generate_signal(self, symbol: str, timeframe: TimeFrame) -> Optional[TechnicalSignal]:
        """Generate trading signal based on technical analysis"""
        try:
            # Get all indicators
            indicators = self.calculate_all_indicators(symbol, timeframe)
            if not indicators:
                return None
            
            # Get current price
            if (symbol not in self.ohlcv_data or 
                timeframe not in self.ohlcv_data[symbol] or
                len(self.ohlcv_data[symbol][timeframe]) == 0):
                return None
            
            current_bar = self.ohlcv_data[symbol][timeframe][-1]
            current_price = current_bar.close
            
            # Analyze signals from each indicator
            signals = self._analyze_indicator_signals(indicators, current_price)
            
            # Calculate confluence score
            confluence_score = self._calculate_confluence_score(signals)
            
            # Determine overall direction and strength
            direction, strength, confidence = self._determine_signal_direction(signals, confluence_score)
            
            # Calculate risk/reward parameters if we have a signal
            if direction in ['BUY', 'SELL']:
                entry_price, stop_loss, take_profit, rr_ratio = self._calculate_risk_reward(
                    current_price, direction, indicators
                )
            else:
                entry_price = current_price
                stop_loss = 0
                take_profit = 0
                rr_ratio = 0
            
            # Check minimum requirements
            min_confluence = self.config.get('signals', {}).get('min_confluence_score', 0.6)
            min_strength = self.config.get('signals', {}).get('min_strength', 3)
            min_rr = self.config.get('signals', {}).get('risk_reward_min', 1.5)
            
            if (confluence_score < min_confluence or 
                strength.value < min_strength or 
                (rr_ratio > 0 and rr_ratio < min_rr)):
                direction = 'HOLD'
            
            # Create technical signal
            signal = TechnicalSignal(
                timestamp=datetime.now(),
                symbol=symbol,
                timeframe=timeframe,
                direction=direction,
                strength=strength,
                confidence=confidence,
                indicators=indicators,
                confluence_score=confluence_score,
                risk_reward_ratio=rr_ratio,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'signals_analysis': signals,
                    'current_price': current_price,
                    'calculation_time': datetime.now().isoformat()
                }
            )
            
            # Store signal
            self._store_technical_signal(signal)
            
            if direction != 'HOLD':
                self.signals_generated += 1
                self.logger.info(f"Technical signal generated: {symbol} {timeframe.value} {direction} "
                               f"confidence: {confidence:.1%} confluence: {confluence_score:.1%}")
            
            return signal
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Signal generation {symbol} {timeframe.value}", "medium")
            return None
    
    def _analyze_indicator_signals(self, indicators: Dict, current_price: float) -> Dict:
        """Analyze signals from all indicators"""
        signals = {}
        
        try:
            # Ichimoku signals
            if 'ichimoku' in indicators:
                ich = indicators['ichimoku']
                signals['ichimoku'] = {
                    'direction': self._get_ichimoku_signal(ich, current_price),
                    'strength': self._get_ichimoku_strength(ich, current_price),
                    'weight': 2.0  # High weight for Ichimoku
                }
            
            # TDI signals
            if 'tdi' in indicators:
                tdi = indicators['tdi']
                signals['tdi'] = {
                    'direction': self._get_tdi_signal(tdi),
                    'strength': tdi.strength.value,
                    'weight': 1.5
                }
            
            # SuperTrend signals
            if 'supertrend' in indicators:
                st = indicators['supertrend']
                signals['supertrend'] = {
                    'direction': 'BUY' if st.direction == 1 else 'SELL',
                    'strength': 4 if st.signal in ['buy', 'sell'] else 2,
                    'weight': 1.8
                }
            
            # Bollinger signals
            if 'bollinger' in indicators:
                bb = indicators['bollinger']
                signals['bollinger'] = {
                    'direction': self._get_bollinger_signal(bb),
                    'strength': self._get_bollinger_strength(bb),
                    'weight': 1.2
                }
            
            # RSI signals
            if 'rsi' in indicators:
                rsi = indicators['rsi']
                signals['rsi'] = {
                    'direction': self._get_rsi_signal(rsi),
                    'strength': self._get_rsi_strength(rsi),
                    'weight': 1.0
                }
            
            # MACD signals
            if 'macd' in indicators:
                macd = indicators['macd']
                signals['macd'] = {
                    'direction': self._get_macd_signal(macd),
                    'strength': self._get_macd_strength(macd),
                    'weight': 1.3
                }
            
            return signals
            
        except Exception as e:
            self.error_handler.handle_error(e, "Indicator signals analysis", "medium")
            return {}
    
    def _get_ichimoku_signal(self, ich: IchimokuData, price: float) -> str:
        """Get Ichimoku signal direction"""
        bullish_signals = 0
        bearish_signals = 0
        
        # Price vs Kumo
        if ich.price_vs_kumo == 'above':
            bullish_signals += 2
        elif ich.price_vs_kumo == 'below':
            bearish_signals += 2
        
        # TK Cross
        if ich.tk_cross == 'bullish':
            bullish_signals += 2
        elif ich.tk_cross == 'bearish':
            bearish_signals += 2
        
        # Kumo color
        if ich.kumo_color == 'bullish':
            bullish_signals += 1
        else:
            bearish_signals += 1
        
        if bullish_signals > bearish_signals + 1:
            return 'BUY'
        elif bearish_signals > bullish_signals + 1:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _get_ichimoku_strength(self, ich: IchimokuData, price: float) -> int:
        """Get Ichimoku signal strength"""
        strength = 1
        
        if ich.tk_cross in ['bullish', 'bearish']:
            strength += 2
        if ich.kumo_twist in ['bullish', 'bearish']:
            strength += 1
        if ich.price_vs_kumo in ['above', 'below']:
            strength += 1
        
        return min(5, strength)
    
    def _get_tdi_signal(self, tdi: TDIData) -> str:
        """Get TDI signal direction"""
        if tdi.signal_cross == 'bullish' and tdi.rsi_ma > tdi.market_base_line:
            return 'BUY'
        elif tdi.signal_cross == 'bearish' and tdi.rsi_ma < tdi.market_base_line:
            return 'SELL'
        elif tdi.rsi_ma > tdi.signal_line and tdi.zone != 'overbought':
            return 'BUY'
        elif tdi.rsi_ma < tdi.signal_line and tdi.zone != 'oversold':
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _get_bollinger_signal(self, bb: BollingerData) -> str:
        """Get Bollinger Bands signal"""
        if bb.squeeze and bb.expansion:
            return 'BUY' if bb.percent_b > 0.5 else 'SELL'
        elif bb.position == 'above_upper':
            return 'SELL'
        elif bb.position == 'below_lower':
            return 'BUY'
        else:
            return 'NEUTRAL'
    
    def _get_bollinger_strength(self, bb: BollingerData) -> int:
        """Get Bollinger Bands signal strength"""
        if bb.squeeze and bb.expansion:
            return 4
        elif bb.position in ['above_upper', 'below_lower']:
            return 3
        else:
            return 2
    
    def _get_rsi_signal(self, rsi: Dict) -> str:
        """Get RSI signal"""
        if rsi['zone'] == 'oversold' and rsi['divergence'] == 'bullish':
            return 'BUY'
        elif rsi['zone'] == 'overbought' and rsi['divergence'] == 'bearish':
            return 'SELL'
        elif rsi['value'] < 40:
            return 'BUY'
        elif rsi['value'] > 60:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _get_rsi_strength(self, rsi: Dict) -> int:
        """Get RSI signal strength"""
        if rsi['divergence'] != 'none':
            return 4
        elif rsi['zone'] in ['overbought', 'oversold']:
            return 3
        else:
            return 2
    
    def _get_macd_signal(self, macd: Dict) -> str:
        """Get MACD signal"""
        if macd['cross'] == 'bullish' and macd['histogram'] > 0:
            return 'BUY'
        elif macd['cross'] == 'bearish' and macd['histogram'] < 0:
            return 'SELL'
        elif macd['macd'] > macd['signal'] and macd['histogram'] > 0:
            return 'BUY'
        elif macd['macd'] < macd['signal'] and macd['histogram'] < 0:
            return 'SELL'
        else:
            return 'NEUTRAL'
    
    def _get_macd_strength(self, macd: Dict) -> int:
        """Get MACD signal strength"""
        if macd['cross'] != 'none':
            return 4
        elif abs(macd['histogram']) > abs(macd['macd']) * 0.1:
            return 3
        else:
            return 2
    
    def _calculate_confluence_score(self, signals: Dict) -> float:
        """Calculate confluence score from all signals"""
        try:
            if not signals:
                return 0.0
            
            buy_weight = 0
            sell_weight = 0
            total_weight = 0
            
            for indicator, signal_data in signals.items():
                direction = signal_data['direction']
                strength = signal_data['strength']
                weight = signal_data['weight']
                
                adjusted_weight = weight * (strength / 5.0)  # Normalize strength
                total_weight += adjusted_weight
                
                if direction == 'BUY':
                    buy_weight += adjusted_weight
                elif direction == 'SELL':
                    sell_weight += adjusted_weight
            
            if total_weight == 0:
                return 0.0
            
            # Return the dominant direction's percentage
            max_weight = max(buy_weight, sell_weight)
            return max_weight / total_weight
            
        except Exception as e:
            self.error_handler.handle_error(e, "Confluence score calculation", "low")
            return 0.0
    
    def _determine_signal_direction(self, signals: Dict, confluence_score: float) -> Tuple[str, SignalStrength, float]:
        """Determine overall signal direction and strength"""
        try:
            if not signals or confluence_score < 0.5:
                return 'HOLD', SignalStrength.VERY_WEAK, 0.0
            
            buy_votes = 0
            sell_votes = 0
            total_strength = 0
            
            for signal_data in signals.values():
                direction = signal_data['direction']
                strength = signal_data['strength']
                weight = signal_data['weight']
                
                weighted_strength = strength * weight
                total_strength += weighted_strength
                
                if direction == 'BUY':
                    buy_votes += weighted_strength
                elif direction == 'SELL':
                    sell_votes += weighted_strength
            
            # Determine direction
            if buy_votes > sell_votes * 1.2:  # 20% margin required
                direction = 'BUY'
                confidence = buy_votes / (buy_votes + sell_votes)
            elif sell_votes > buy_votes * 1.2:
                direction = 'SELL'
                confidence = sell_votes / (buy_votes + sell_votes)
            else:
                direction = 'HOLD'
                confidence = 0.5
            
            # Determine strength
            avg_strength = total_strength / len(signals) if signals else 1
            
            if avg_strength >= 4:
                strength = SignalStrength.VERY_STRONG
            elif avg_strength >= 3.5:
                strength = SignalStrength.STRONG
            elif avg_strength >= 2.5:
                strength = SignalStrength.MODERATE
            elif avg_strength >= 1.5:
                strength = SignalStrength.WEAK
            else:
                strength = SignalStrength.VERY_WEAK
            
            return direction, strength, confidence
            
        except Exception as e:
            self.error_handler.handle_error(e, "Signal direction determination", "medium")
            return 'HOLD', SignalStrength.VERY_WEAK, 0.0
    
    def _calculate_risk_reward(self, current_price: float, direction: str, indicators: Dict) -> Tuple[float, float, float, float]:
        """Calculate risk/reward parameters"""
        try:
            entry_price = current_price
            atr = indicators.get('atr', current_price * 0.01)  # Default 1% ATR
            
            # Dynamic multipliers based on volatility
            stop_multiplier = 2.0
            target_multiplier = 3.0
            
            if direction == 'BUY':
                stop_loss = entry_price - (atr * stop_multiplier)
                take_profit = entry_price + (atr * target_multiplier)
            else:  # SELL
                stop_loss = entry_price + (atr * stop_multiplier)
                take_profit = entry_price - (atr * target_multiplier)
            
            # Calculate risk/reward ratio
            risk = abs(entry_price - stop_loss)
            reward = abs(take_profit - entry_price)
            rr_ratio = reward / risk if risk > 0 else 0
            
            return entry_price, stop_loss, take_profit, rr_ratio
            
        except Exception as e:
            self.error_handler.handle_error(e, "Risk/reward calculation", "low")
            return current_price, 0, 0, 0
    
    def _store_ohlcv_data(self, symbol: str, timeframe: TimeFrame, data: List[OHLCV]):
        """Store OHLCV data to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for bar in data:
                    cursor.execute("""
                        INSERT OR REPLACE INTO ohlcv_data (
                            symbol, timeframe, timestamp, open, high, low, close, volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        timeframe.value,
                        bar.timestamp.isoformat(),
                        bar.open,
                        bar.high,
                        bar.low,
                        bar.close,
                        bar.volume
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.error_handler.handle_error(e, "OHLCV data storage", "low")
    
    def _store_indicator_values(self, symbol: str, timeframe: TimeFrame, indicators: Dict):
        """Store indicator values to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for indicator_type, values in indicators.items():
                    cursor.execute("""
                        INSERT OR REPLACE INTO indicator_values (
                            symbol, timeframe, timestamp, indicator_type, values
                        ) VALUES (?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        timeframe.value,
                        datetime.now().isoformat(),
                        indicator_type,
                        json.dumps(values, default=str)
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.error_handler.handle_error(e, "Indicator values storage", "low")
    
    def _store_technical_signal(self, signal: TechnicalSignal):
        """Store technical signal to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO technical_signals (
                        symbol, timeframe, timestamp, direction, strength,
                        confidence, confluence_score, entry_price, stop_loss,
                        take_profit, risk_reward_ratio, indicators, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    signal.symbol,
                    signal.timeframe.value,
                    signal.timestamp.isoformat(),
                    signal.direction,
                    signal.strength.value,
                    signal.confidence,
                    signal.confluence_score,
                    signal.entry_price,
                    signal.stop_loss,
                    signal.take_profit,
                    signal.risk_reward_ratio,
                    json.dumps(signal.indicators, default=str),
                    json.dumps(signal.metadata, default=str)
                ))
                conn.commit()
                
        except Exception as e:
            self.error_handler.handle_error(e, "Technical signal storage", "medium")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Start cache cleanup
            threading.Thread(
                target=self._cache_cleanup_loop,
                daemon=True,
                name="Technical_Cache_Cleanup"
            ).start()
            
            self.logger.info("Technical analyzer background tasks started")
        except Exception as e:
            self.error_handler.handle_error(e, "Background tasks startup", "medium")
    
    def _cache_cleanup_loop(self):
        """Background cache cleanup loop"""
        while True:
            try:
                time.sleep(3600)  # 1 hour
                self._cleanup_cache()
            except Exception as e:
                self.error_handler.handle_error(e, "Cache cleanup loop", "low")
                time.sleep(300)  # 5 minute retry
    
    def _cleanup_cache(self):
        """Cleanup old cache entries"""
        try:
            max_cache_size = self.config.get('performance', {}).get('max_cache_size', 1000)
            cache_duration = timedelta(minutes=self.config.get('performance', {}).get('cache_duration_minutes', 5))
            current_time = datetime.now()
            
            with self.lock:
                # Cleanup indicators cache
                for symbol in list(self.indicators_cache.keys()):
                    for timeframe in list(self.indicators_cache[symbol].keys()):
                        cache_time = self.indicators_cache[symbol][timeframe].get('timestamp', datetime.min)
                        if current_time - cache_time > cache_duration:
                            del self.indicators_cache[symbol][timeframe]
                    
                    # Remove empty symbol entries
                    if not self.indicators_cache[symbol]:
                        del self.indicators_cache[symbol]
                
                # Cleanup OHLCV data if too large
                total_bars = sum(
                    len(timeframe_data)
                    for symbol_data in self.ohlcv_data.values()
                    for timeframe_data in symbol_data.values()
                )
                
                if total_bars > max_cache_size:
                    # Remove oldest data
                    for symbol in self.ohlcv_data:
                        for timeframe in self.ohlcv_data[symbol]:
                            max_bars = self.config.get('timeframes', {}).get('max_bars', 500)
                            if len(self.ohlcv_data[symbol][timeframe]) > max_bars:
                                self.ohlcv_data[symbol][timeframe] = self.ohlcv_data[symbol][timeframe][-max_bars:]
            
            self.logger.debug("Cache cleanup completed")
            
        except Exception as e:
            self.error_handler.handle_error(e, "Cache cleanup", "low")
    
    def get_system_status(self) -> Dict:
        """Get technical analyzer system status"""
        try:
            with self.lock:
                total_symbols = len(self.ohlcv_data)
                total_timeframes = sum(len(tf_data) for tf_data in self.ohlcv_data.values())
                cached_indicators = sum(len(tf_data) for tf_data in self.indicators_cache.values())
            
            return {
                'enabled': self.enabled,
                'calculations_performed': self.calculations_performed,
                'signals_generated': self.signals_generated,
                'last_update': self.last_update.isoformat(),
                'data_status': {
                    'symbols_tracked': total_symbols,
                    'timeframes_active': total_timeframes,
                    'cached_indicators': cached_indicators
                },
                'database_path': self.db_path,
                'indicators_enabled': {
                    indicator: config.get('enabled', True)
                    for indicator, config in self.config.get('indicators', {}).items()
                }
            }
        except Exception as e:
            self.error_handler.handle_error(e, "System status", "low")
            return {'error': str(e)}
    
    def get_indicator_summary(self, symbol: str, timeframe: TimeFrame) -> Optional[Dict]:
        """Get summary of all indicators for symbol/timeframe"""
        try:
            indicators = self.calculate_all_indicators(symbol, timeframe)
            if not indicators:
                return None
            
            summary = {
                'symbol': symbol,
                'timeframe': timeframe.value,
                'timestamp': datetime.now().isoformat(),
                'indicators': {}
            }
            
            # Summarize each indicator
            for indicator_name, indicator_data in indicators.items():
                if indicator_name == 'ichimoku':
                    summary['indicators']['ichimoku'] = {
                        'price_vs_kumo': indicator_data.price_vs_kumo,
                        'tk_cross': indicator_data.tk_cross,
                        'kumo_color': indicator_data.kumo_color,
                        'signal': self._get_ichimoku_signal(indicator_data, 0)
                    }
                elif indicator_name == 'tdi':
                    summary['indicators']['tdi'] = {
                        'zone': indicator_data.zone,
                        'signal_cross': indicator_data.signal_cross,
                        'strength': indicator_data.strength.name,
                        'signal': self._get_tdi_signal(indicator_data)
                    }
                elif indicator_name == 'bollinger':
                    summary['indicators']['bollinger'] = {
                        'position': indicator_data.position,
                        'squeeze': indicator_data.squeeze,
                        'expansion': indicator_data.expansion,
                        'signal': self._get_bollinger_signal(indicator_data)
                    }
                elif indicator_name == 'supertrend':
                    summary['indicators']['supertrend'] = {
                        'direction': 'bullish' if indicator_data.direction == 1 else 'bearish',
                        'signal': indicator_data.signal,
                        'strength': indicator_data.strength
                    }
                else:
                    # For other indicators, include key values
                    summary['indicators'][indicator_name] = indicator_data
            
            return summary
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Indicator summary {symbol} {timeframe.value}", "low")
            return None
    
    def update_signal_outcome(self, 
                            symbol: str, 
                            timeframe: TimeFrame,
                            timestamp: datetime, 
                            outcome: str, 
                            return_percent: float):
        """Update signal outcome for learning"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE technical_signals 
                    SET outcome = ?, actual_return = ?
                    WHERE symbol = ? AND timeframe = ? AND timestamp = ?
                """, (outcome, return_percent, symbol, timeframe.value, timestamp.isoformat()))
                
                if cursor.rowcount > 0:
                    conn.commit()
                    self.logger.info(f"Technical signal outcome updated: {symbol} {timeframe.value} {outcome} ({return_percent:.2%})")
                    
        except Exception as e:
            self.error_handler.handle_error(e, "Signal outcome update", "medium")
    
    def shutdown(self):
        """Shutdown technical analyzer"""
        try:
            self.logger.info("Shutting down Technical Analyzer...")
            
            with self.lock:
                self.ohlcv_data.clear()
                self.indicators_cache.clear()
            
            self.enabled = False
            self.logger.info("Technical Analyzer shutdown complete")
            
        except Exception as e:
            self.error_handler.handle_error(e, "Technical analyzer shutdown", "medium")

# Global instance
technical_analyzer = None

def get_technical_analyzer(config_path: str = None) -> TechnicalAnalyzer:
    """Get global technical analyzer instance"""
    global technical_analyzer
    if technical_analyzer is None:
        technical_analyzer = TechnicalAnalyzer(config_path)
    return technical_analyzer

if __name__ == "__main__":
    # Test the Technical Analyzer
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize analyzer
        analyzer = TechnicalAnalyzer()
        
        # Create test OHLCV data
        test_data = []
        base_price = 100.0
        
        for i in range(100):
            # Generate realistic OHLCV data
            open_price = base_price + np.random.normal(0, 0.5)
            close_price = open_price + np.random.normal(0, 0.8)
            high_price = max(open_price, close_price) + abs(np.random.normal(0, 0.3))
            low_price = min(open_price, close_price) - abs(np.random.normal(0, 0.3))
            volume = abs(np.random.normal(50000, 10000))
            
            test_data.append(OHLCV(
                timestamp=datetime.now() - timedelta(minutes=15 * (100 - i)),
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            ))
            
            base_price = close_price  # Use close as next base
        
        # Add test data
        analyzer.add_ohlcv_data('EURUSD', TimeFrame.M15, test_data)
        
        # Calculate indicators
        indicators = analyzer.calculate_all_indicators('EURUSD', TimeFrame.M15)
        if indicators:
            print("Indicators calculated successfully:")
            for name, data in indicators.items():
                print(f"  {name}: {type(data).__name__}")
        
        # Generate signal
        signal = analyzer.generate_signal('EURUSD', TimeFrame.M15)
        if signal:
            print(f"\nTechnical Signal Generated:")
            print(f"  Symbol: {signal.symbol}")
            print(f"  Direction: {signal.direction}")
            print(f"  Strength: {signal.strength.name}")
            print(f"  Confidence: {signal.confidence:.1%}")
            print(f"  Confluence Score: {signal.confluence_score:.1%}")
            print(f"  Risk/Reward: {signal.risk_reward_ratio:.2f}")
        else:
            print("No signal generated")
        
        # Get system status
        status = analyzer.get_system_status()
        print(f"\nSystem Status: {json.dumps(status, indent=2)}")
        
        print("Technical Analyzer test completed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'analyzer' in locals():
            analyzer.shutdown()