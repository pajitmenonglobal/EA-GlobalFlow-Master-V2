#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Non-F&O Market Scanner
Advanced market scanner for Forex, Commodities, and CFDs with opportunity ranking

Author: EA GlobalFlow Pro Team
Version: v0.1
Date: 2025
"""

import os
import sys
import json
import time
import logging
import threading
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import queue
import concurrent.futures
from collections import defaultdict
import heapq

class MarketType(Enum):
    FOREX = "FOREX"
    COMMODITY = "COMMODITY" 
    CFD = "CFD"
    CRYPTO = "CRYPTO"
    INDEX = "INDEX"

class OpportunityGrade(Enum):
    EXCELLENT = "EXCELLENT"  # 90%+ score
    GOOD = "GOOD"           # 75-90% score
    FAIR = "FAIR"           # 60-75% score
    POOR = "POOR"           # <60% score

class ScanFrequency(Enum):
    REALTIME = "REALTIME"   # Every tick
    HIGH = "HIGH"           # Every 5 seconds
    MEDIUM = "MEDIUM"       # Every 30 seconds
    LOW = "LOW"             # Every 2 minutes

@dataclass
class MarketInstrument:
    symbol: str
    name: str
    market_type: MarketType
    base_currency: str
    quote_currency: str
    pip_value: float
    contract_size: float
    margin_requirement: float
    trading_hours: Dict[str, str]
    enabled: bool = True
    min_lot_size: float = 0.01
    max_lot_size: float = 100.0
    spread_typical: float = 0.0
    commission: float = 0.0

@dataclass
class MarketData:
    symbol: str
    timestamp: datetime
    bid: float
    ask: float
    last: float
    volume: int
    high: float
    low: float
    open: float
    close: float
    spread: float
    volatility: float = 0.0
    momentum: float = 0.0
    trend_strength: float = 0.0

@dataclass
class TechnicalAnalysis:
    symbol: str
    timeframe: str
    timestamp: datetime
    
    # Trend indicators
    sma_20: float = 0.0
    sma_50: float = 0.0
    ema_12: float = 0.0
    ema_26: float = 0.0
    macd: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    
    # Momentum indicators
    rsi: float = 0.0
    stochastic_k: float = 0.0
    stochastic_d: float = 0.0
    williams_r: float = 0.0
    cci: float = 0.0
    
    # Volatility indicators
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_width: float = 0.0
    atr: float = 0.0
    
    # Volume indicators
    volume_sma: float = 0.0
    volume_ratio: float = 0.0
    
    # Support/Resistance
    support_1: float = 0.0
    support_2: float = 0.0
    resistance_1: float = 0.0
    resistance_2: float = 0.0
    
    # Pattern recognition
    candlestick_pattern: str = "NONE"
    chart_pattern: str = "NONE"
    
    # Overall scores
    trend_score: float = 0.0
    momentum_score: float = 0.0
    volatility_score: float = 0.0
    overall_score: float = 0.0

@dataclass
class ScanOpportunity:
    opportunity_id: str
    symbol: str
    market_type: MarketType
    signal_type: str  # BUY/SELL
    grade: OpportunityGrade
    confidence_score: float
    ml_score: float
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    position_size: float
    timeframe: str
    scan_timestamp: datetime
    
    # Analysis details
    technical_analysis: TechnicalAnalysis
    fundamental_factors: Dict[str, Any] = field(default_factory=dict)
    market_context: Dict[str, Any] = field(default_factory=dict)
    
    # Scoring breakdown
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    volatility_score: float = 0.0
    liquidity_score: float = 0.0
    timing_score: float = 0.0
    
    # Risk assessment
    risk_factors: List[str] = field(default_factory=list)
    risk_score: float = 0.0
    
    # Validity
    expires_at: datetime = None
    is_valid: bool = True

class MarketScanner:
    """
    Advanced Non-F&O Market Scanner for EA GlobalFlow Pro v0.1
    Scans Forex, Commodities, and CFDs for trading opportunities
    """
    
    def __init__(self, config_manager=None, fyers_bridge=None, 
                 truedata_bridge=None, error_handler=None):
        """Initialize market scanner"""
        self.config_manager = config_manager
        self.fyers_bridge = fyers_bridge
        self.truedata_bridge = truedata_bridge
        self.error_handler = error_handler
        self.logger = logging.getLogger('MarketScanner')
        
        # Configuration
        self.scanner_config = {}
        self.is_initialized = False
        self.is_scanning = False
        
        # Market instruments
        self.instruments = {}
        self.forex_pairs = {}
        self.commodities = {}
        self.cfds = {}
        
        # Market data
        self.market_data = {}
        self.technical_analysis = {}
        self.data_lock = threading.Lock()
        
        # Scanning
        self.scan_queue = queue.PriorityQueue()
        self.opportunities = {}
        self.opportunity_rankings = []
        self.max_opportunities = 100
        
        # Threading
        self.scan_threads = []
        self.data_threads = []
        self.analysis_threads = []
        self.num_scan_threads = 4
        self.num_analysis_threads = 2
        
        # Performance tracking
        self.scan_statistics = {
            'total_scans': 0,
            'opportunities_found': 0,
            'successful_signals': 0,
            'false_signals': 0,
            'scan_time_avg': 0.0,
            'last_scan_time': None
        }
        
        # Callbacks
        self.opportunity_callbacks = []
        self.market_data_callbacks = []
        
        # Cache
        self.analysis_cache = {}
        self.cache_expiry = 300  # 5 minutes
        
        # Economic calendar integration
        self.economic_events = {}
        self.news_sentiment = {}
        
    def initialize(self) -> bool:
        """
        Initialize market scanner
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing Market Scanner v0.1...")
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Initialize market instruments
            if not self._initialize_instruments():
                return False
            
            # Setup technical analysis
            if not self._setup_technical_analysis():
                return False
            
            # Start data collection threads
            self._start_data_collection()
            
            # Start scanning threads
            self._start_scanning_threads()
            
            self.is_initialized = True
            self.logger.info("✅ Market Scanner initialized successfully")
            self.logger.info(f"📊 Monitoring {len(self.instruments)} instruments")
            return True
            
        except Exception as e:
            self.logger.error(f"Market Scanner initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("market_scanner_init", e)
            return False
    
    def _load_config(self) -> bool:
        """Load scanner configuration"""
        try:
            if self.config_manager:
                self.scanner_config = self.config_manager.get_config('non_fno_scanner', {})
            else:
                # Default configuration
                self.scanner_config = {
                    'enabled': True,
                    'scan_frequency': 'MEDIUM',
                    'max_opportunities': 100,
                    'min_confidence_score': 0.7,
                    'max_risk_per_trade': 2.0,
                    'supported_markets': {
                        'forex': True,
                        'commodities': True,
                        'cfds': True,
                        'crypto': False,
                        'indices': True
                    },
                    'timeframes': ['M5', 'M15', 'M30', 'H1'],
                    'technical_indicators': {
                        'trend': ['SMA', 'EMA', 'MACD'],
                        'momentum': ['RSI', 'Stochastic', 'Williams%R'],
                        'volatility': ['BollingerBands', 'ATR'],
                        'volume': ['VolumeSMA', 'VolumeRatio']
                    },
                    'ml_enhancement': True,
                    'economic_calendar_integration': True,
                    'news_sentiment_analysis': True
                }
            
            # Update settings
            self.max_opportunities = self.scanner_config.get('max_opportunities', 100)
            
            self.logger.info("Scanner configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load scanner config: {e}")
            return False
    
    def _initialize_instruments(self) -> bool:
        """Initialize market instruments"""
        try:
            # Initialize Forex pairs
            if self.scanner_config.get('supported_markets', {}).get('forex', True):
                self._initialize_forex_pairs()
            
            # Initialize Commodities
            if self.scanner_config.get('supported_markets', {}).get('commodities', True):
                self._initialize_commodities()
            
            # Initialize CFDs
            if self.scanner_config.get('supported_markets', {}).get('cfds', True):
                self._initialize_cfds()
            
            # Initialize Indices
            if self.scanner_config.get('supported_markets', {}).get('indices', True):
                self._initialize_indices()
            
            self.logger.info(f"Initialized {len(self.instruments)} instruments across all markets")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize instruments: {e}")
            return False
    
    def _initialize_forex_pairs(self):
        """Initialize forex currency pairs"""
        try:
            forex_pairs = [
                # Major pairs
                ('EUR/USD', 'Euro/US Dollar', 0.0001, 100000, 3.33),
                ('GBP/USD', 'British Pound/US Dollar', 0.0001, 100000, 3.33),
                ('USD/JPY', 'US Dollar/Japanese Yen', 0.01, 100000, 3.33),
                ('USD/CHF', 'US Dollar/Swiss Franc', 0.0001, 100000, 3.33),
                ('AUD/USD', 'Australian Dollar/US Dollar', 0.0001, 100000, 3.33),
                ('USD/CAD', 'US Dollar/Canadian Dollar', 0.0001, 100000, 3.33),
                ('NZD/USD', 'New Zealand Dollar/US Dollar', 0.0001, 100000, 3.33),
                
                # Cross pairs
                ('EUR/GBP', 'Euro/British Pound', 0.0001, 100000, 3.33),
                ('EUR/JPY', 'Euro/Japanese Yen', 0.01, 100000, 3.33),
                ('GBP/JPY', 'British Pound/Japanese Yen', 0.01, 100000, 3.33),
                ('CHF/JPY', 'Swiss Franc/Japanese Yen', 0.01, 100000, 3.33),
                ('AUD/JPY', 'Australian Dollar/Japanese Yen', 0.01, 100000, 3.33),
                ('CAD/JPY', 'Canadian Dollar/Japanese Yen', 0.01, 100000, 3.33),
                
                # Exotic pairs (selective)
                ('USD/TRY', 'US Dollar/Turkish Lira', 0.0001, 100000, 5.0),
                ('EUR/TRY', 'Euro/Turkish Lira', 0.0001, 100000, 5.0),
                ('USD/ZAR', 'US Dollar/South African Rand', 0.0001, 100000, 5.0)
            ]
            
            for symbol, name, pip_value, contract_size, margin in forex_pairs:
                base_currency, quote_currency = symbol.split('/')
                
                instrument = MarketInstrument(
                    symbol=symbol,
                    name=name,
                    market_type=MarketType.FOREX,
                    base_currency=base_currency,
                    quote_currency=quote_currency,
                    pip_value=pip_value,
                    contract_size=contract_size,
                    margin_requirement=margin,
                    trading_hours={
                        'monday_open': '22:00:00',
                        'friday_close': '22:00:00',
                        'timezone': 'UTC'
                    },
                    enabled=True,
                    spread_typical=1.5  # pips
                )
                
                self.instruments[symbol] = instrument
                self.forex_pairs[symbol] = instrument
            
            self.logger.info(f"Initialized {len(forex_pairs)} forex pairs")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize forex pairs: {e}")
    
    def _initialize_commodities(self):
        """Initialize commodity instruments"""
        try:
            commodities = [
                # Energy
                ('CRUDE_OIL', 'Crude Oil WTI', 'USD', 'BARREL', 0.01, 1000, 5.0),
                ('BRENT_OIL', 'Brent Crude Oil', 'USD', 'BARREL', 0.01, 1000, 5.0),
                ('NATURAL_GAS', 'Natural Gas', 'USD', 'MMBTU', 0.001, 10000, 5.0),
                
                # Metals
                ('GOLD', 'Gold Spot', 'USD', 'OUNCE', 0.01, 100, 3.33),
                ('SILVER', 'Silver Spot', 'USD', 'OUNCE', 0.001, 5000, 3.33),
                ('PLATINUM', 'Platinum Spot', 'USD', 'OUNCE', 0.01, 100, 5.0),
                ('PALLADIUM', 'Palladium Spot', 'USD', 'OUNCE', 0.01, 100, 5.0),
                ('COPPER', 'Copper', 'USD', 'POUND', 0.0001, 25000, 5.0),
                
                # Agricultural (selective)
                ('WHEAT', 'Wheat', 'USD', 'BUSHEL', 0.25, 5000, 10.0),
                ('CORN', 'Corn', 'USD', 'BUSHEL', 0.25, 5000, 10.0),
                ('SOYBEANS', 'Soybeans', 'USD', 'BUSHEL', 0.25, 5000, 10.0)
            ]
            
            for symbol, name, quote_currency, unit, pip_value, contract_size, margin in commodities:
                instrument = MarketInstrument(
                    symbol=symbol,
                    name=name,
                    market_type=MarketType.COMMODITY,
                    base_currency=unit,
                    quote_currency=quote_currency,
                    pip_value=pip_value,
                    contract_size=contract_size,
                    margin_requirement=margin,
                    trading_hours={
                        'start': '23:00:00',
                        'end': '22:00:00',
                        'timezone': 'UTC',
                        'break_start': '21:00:00',
                        'break_end': '23:00:00'
                    },
                    enabled=True
                )
                
                self.instruments[symbol] = instrument
                self.commodities[symbol] = instrument
            
            self.logger.info(f"Initialized {len(commodities)} commodities")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize commodities: {e}")
    
    def _initialize_cfds(self):
        """Initialize CFD instruments"""
        try:
            cfds = [
                # Stock CFDs (major stocks)
                ('AAPL_CFD', 'Apple Inc CFD', 'USD', 'SHARE', 0.01, 1, 20.0),
                ('MSFT_CFD', 'Microsoft Corp CFD', 'USD', 'SHARE', 0.01, 1, 20.0),
                ('GOOGL_CFD', 'Alphabet Inc CFD', 'USD', 'SHARE', 0.01, 1, 20.0),
                ('AMZN_CFD', 'Amazon.com Inc CFD', 'USD', 'SHARE', 0.01, 1, 20.0),
                ('TSLA_CFD', 'Tesla Inc CFD', 'USD', 'SHARE', 0.01, 1, 20.0),
                
                # Crypto CFDs (if enabled)
                ('BTC_USD_CFD', 'Bitcoin CFD', 'USD', 'BTC', 0.01, 1, 50.0),
                ('ETH_USD_CFD', 'Ethereum CFD', 'USD', 'ETH', 0.01, 1, 50.0),
                ('LTC_USD_CFD', 'Litecoin CFD', 'USD', 'LTC', 0.01, 1, 50.0)
            ]
            
            for symbol, name, quote_currency, unit, pip_value, contract_size, margin in cfds:
                instrument = MarketInstrument(
                    symbol=symbol,
                    name=name,
                    market_type=MarketType.CFD,
                    base_currency=unit,
                    quote_currency=quote_currency,
                    pip_value=pip_value,
                    contract_size=contract_size,
                    margin_requirement=margin,
                    trading_hours={
                        'start': '00:00:00',
                        'end': '23:59:59',
                        'timezone': 'UTC'
                    },
                    enabled=True
                )
                
                self.instruments[symbol] = instrument
                self.cfds[symbol] = instrument
            
            self.logger.info(f"Initialized {len(cfds)} CFDs")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CFDs: {e}")
    
    def _initialize_indices(self):
        """Initialize index instruments"""
        try:
            indices = [
                # Major indices
                ('SPX500', 'S&P 500 Index', 'USD', 'INDEX', 0.01, 1, 5.0),
                ('NAS100', 'NASDAQ 100 Index', 'USD', 'INDEX', 0.01, 1, 5.0),
                ('DJI30', 'Dow Jones Industrial Average', 'USD', 'INDEX', 0.01, 1, 5.0),
                ('GER30', 'Germany 30 Index', 'EUR', 'INDEX', 0.01, 1, 5.0),
                ('UK100', 'FTSE 100 Index', 'GBP', 'INDEX', 0.01, 1, 5.0),
                ('JPN225', 'Nikkei 225 Index', 'JPY', 'INDEX', 0.01, 1, 5.0),
                ('AUS200', 'ASX 200 Index', 'AUD', 'INDEX', 0.01, 1, 5.0)
            ]
            
            for symbol, name, quote_currency, unit, pip_value, contract_size, margin in indices:
                instrument = MarketInstrument(
                    symbol=symbol,
                    name=name,
                    market_type=MarketType.INDEX,
                    base_currency=unit,
                    quote_currency=quote_currency,
                    pip_value=pip_value,
                    contract_size=contract_size,
                    margin_requirement=margin,
                    trading_hours={
                        'start': '01:00:00',
                        'end': '23:00:00',
                        'timezone': 'UTC'
                    },
                    enabled=True
                )
                
                self.instruments[symbol] = instrument
            
            self.logger.info(f"Initialized {len(indices)} indices")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize indices: {e}")
    
    def _setup_technical_analysis(self) -> bool:
        """Setup technical analysis framework"""
        try:
            # Initialize technical analysis for each timeframe
            timeframes = self.scanner_config.get('timeframes', ['M5', 'M15', 'M30', 'H1'])
            
            for symbol in self.instruments:
                self.technical_analysis[symbol] = {}
                for timeframe in timeframes:
                    self.technical_analysis[symbol][timeframe] = TechnicalAnalysis(
                        symbol=symbol,
                        timeframe=timeframe,
                        timestamp=datetime.now()
                    )
            
            self.logger.info("Technical analysis framework setup completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup technical analysis: {e}")
            return False
    
    def _start_data_collection(self):
        """Start data collection threads"""
        try:
            # Start market data collection thread
            data_thread = threading.Thread(target=self._market_data_collection_loop, daemon=True)
            data_thread.start()
            self.data_threads.append(data_thread)
            
            # Start technical analysis update thread
            analysis_thread = threading.Thread(target=self._technical_analysis_loop, daemon=True)
            analysis_thread.start()
            self.analysis_threads.append(analysis_thread)
            
            self.logger.info("Data collection threads started")
            
        except Exception as e:
            self.logger.error(f"Failed to start data collection: {e}")
    
    def _start_scanning_threads(self):
        """Start scanning threads"""
        try:
            self.is_scanning = True
            
            # Start multiple scanning threads for parallel processing
            for i in range(self.num_scan_threads):
                scan_thread = threading.Thread(target=self._scanning_loop, daemon=True)
                scan_thread.start()
                self.scan_threads.append(scan_thread)
            
            # Start opportunity ranking thread
            ranking_thread = threading.Thread(target=self._opportunity_ranking_loop, daemon=True)
            ranking_thread.start()
            self.scan_threads.append(ranking_thread)
            
            # Start scan coordination thread
            coordinator_thread = threading.Thread(target=self._scan_coordinator_loop, daemon=True)
            coordinator_thread.start()
            self.scan_threads.append(coordinator_thread)
            
            self.logger.info(f"Started {len(self.scan_threads)} scanning threads")
            
        except Exception as e:
            self.logger.error(f"Failed to start scanning threads: {e}")
    
    def _market_data_collection_loop(self):
        """Market data collection loop"""
        while self.is_scanning:
            try:
                # Collect market data for all instruments
                for symbol, instrument in self.instruments.items():
                    if instrument.enabled:
                        market_data = self._collect_market_data(symbol)
                        if market_data:
                            with self.data_lock:
                                self.market_data[symbol] = market_data
                            
                            # Notify callbacks
                            for callback in self.market_data_callbacks:
                                try:
                                    callback(market_data)
                                except Exception as e:
                                    self.logger.error(f"Market data callback error: {e}")
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                self.logger.error(f"Market data collection error: {e}")
                if self.error_handler:
                    self.error_handler.handle_error("market_data_collection", e)
                time.sleep(10)
    
    def _technical_analysis_loop(self):
        """Technical analysis update loop"""
        while self.is_scanning:
            try:
                # Update technical analysis for all instruments
                for symbol in self.instruments:
                    if symbol in self.market_data:
                        self._update_technical_analysis(symbol)
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Technical analysis error: {e}")
                if self.error_handler:
                    self.error_handler.handle_error("technical_analysis", e)
                time.sleep(60)
    
    def _scanning_loop(self):
        """Main scanning loop"""
        while self.is_scanning:
            try:
                # Get next instrument to scan from queue
                try:
                    priority, symbol = self.scan_queue.get(timeout=1)
                    
                    # Perform scan
                    opportunities = self._scan_instrument(symbol)
                    
                    # Process found opportunities
                    for opportunity in opportunities:
                        self._process_opportunity(opportunity)
                    
                    self.scan_queue.task_done()
                    
                except queue.Empty:
                    continue
                
            except Exception as e:
                self.logger.error(f"Scanning loop error: {e}")
                if self.error_handler:
                    self.error_handler.handle_error("scanning_loop", e)
                time.sleep(5)
    
    def _scan_coordinator_loop(self):
        """Scan coordination loop"""
        scan_frequency = self.scanner_config.get('scan_frequency', 'MEDIUM')
        
        # Determine scan interval
        intervals = {
            'REALTIME': 1,
            'HIGH': 5,
            'MEDIUM': 30,
            'LOW': 120
        }
        interval = intervals.get(scan_frequency, 30)
        
        while self.is_scanning:
            try:
                # Add all enabled instruments to scan queue
                for symbol, instrument in self.instruments.items():
                    if instrument.enabled:
                        # Priority based on market type and volatility
                        priority = self._calculate_scan_priority(symbol, instrument)
                        self.scan_queue.put((priority, symbol))
                
                self.scan_statistics['total_scans'] += 1
                self.scan_statistics['last_scan_time'] = datetime.now()
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Scan coordinator error: {e}")
                time.sleep(60)
    
    def _opportunity_ranking_loop(self):
        """Opportunity ranking and cleanup loop"""
        while self.is_scanning:
            try:
                # Rank opportunities
                self._rank_opportunities()
                
                # Clean up expired opportunities
                self._cleanup_expired_opportunities()
                
                # Update statistics
                self._update_scan_statistics()
                
                time.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Opportunity ranking error: {e}")
                time.sleep(120)
    
    def _collect_market_data(self, symbol: str) -> Optional[MarketData]:
        """Collect market data for symbol"""
        try:
            # This would interface with actual data feeds
            # For now, generating simulated data
            
            current_time = datetime.now()
            
            # Simulate market data
            base_price = 1.1000 if 'EUR/USD' in symbol else 100.0
            volatility = np.random.normal(0, 0.001)
            
            bid = base_price + volatility - 0.0002
            ask = base_price + volatility + 0.0002
            last = base_price + volatility
            
            market_data = MarketData(
                symbol=symbol,
                timestamp=current_time,
                bid=bid,
                ask=ask,
                last=last,
                volume=np.random.randint(1000, 10000),
                high=last + abs(volatility) * 2,
                low=last - abs(volatility) * 2,
                open=last - volatility * 0.5,
                close=last,
                spread=ask - bid,
                volatility=abs(volatility),
                momentum=np.random.uniform(-1, 1),
                trend_strength=np.random.uniform(0, 1)
            )
            
            return market_data
            
        except Exception as e:
            self.logger.error(f"Market data collection error for {symbol}: {e}")
            return None
    
    def _update_technical_analysis(self, symbol: str):
        """Update technical analysis for symbol"""
        try:
            if symbol not in self.market_data:
                return
            
            market_data = self.market_data[symbol]
            timeframes = self.scanner_config.get('timeframes', ['M5', 'M15', 'M30', 'H1'])
            
            for timeframe in timeframes:
                if symbol in self.technical_analysis and timeframe in self.technical_analysis[symbol]:
                    ta = self.technical_analysis[symbol][timeframe]
                    
                    # Update timestamp
                    ta.timestamp = datetime.now()
                    
                    # Calculate technical indicators
                    self._calculate_technical_indicators(ta, market_data, timeframe)
                    
                    # Calculate scores
                    ta.trend_score = self._calculate_trend_score(ta)
                    ta.momentum_score = self._calculate_momentum_score(ta)
                    ta.volatility_score = self._calculate_volatility_score(ta)
                    ta.overall_score = (ta.trend_score + ta.momentum_score + ta.volatility_score) / 3
            
        except Exception as e:
            self.logger.error(f"Technical analysis update error for {symbol}: {e}")
    
    def _calculate_technical_indicators(self, ta: TechnicalAnalysis, market_data: MarketData, timeframe: str):
        """Calculate technical indicators"""
        try:
            # This would use actual price history
            # For now, using simplified calculations based on current data
            
            price = market_data.last
            
            # Moving averages (simplified)
            ta.sma_20 = price * (1 + np.random.uniform(-0.01, 0.01))
            ta.sma_50 = price * (1 + np.random.uniform(-0.02, 0.02))
            ta.ema_12 = price * (1 + np.random.uniform(-0.005, 0.005))
            ta.ema_26 = price * (1 + np.random.uniform(-0.015, 0.015))
            
            # MACD
            ta.macd = ta.ema_12 - ta.ema_26
            ta.macd_signal = ta.macd * 0.9
            ta.macd_histogram = ta.macd - ta.macd_signal
            
            # RSI (simplified)
            ta.rsi = 30 + np.random.uniform(0, 40)
            
            # Stochastic
            ta.stochastic_k = np.random.uniform(20, 80)
            ta.stochastic_d = ta.stochastic_k * 0.9
            
            # Williams %R
            ta.williams_r = -np.random.uniform(20, 80)
            
            # CCI
            ta.cci = np.random.uniform(-200, 200)
            
            # Bollinger Bands
            bb_width = price * 0.02
            ta.bb_middle = ta.sma_20
            ta.bb_upper = ta.bb_middle + bb_width
            ta.bb_lower = ta.bb_middle - bb_width
            ta.bb_width = (ta.bb_upper - ta.bb_lower) / ta.bb_middle * 100
            
            # ATR
            ta.atr = price * np.random.uniform(0.005, 0.02)
            
            # Volume indicators
            ta.volume_sma = market_data.volume * (1 + np.random.uniform(-0.2, 0.2))
            ta.volume_ratio = market_data.volume / ta.volume_sma if ta.volume_sma > 0 else 1.0
            
            # Support/Resistance (simplified)
            ta.support_1 = price * 0.99
            ta.support_2 = price * 0.98
            ta.resistance_1 = price * 1.01
            ta.resistance_2 = price * 1.02
            
            # Pattern recognition (simplified)
            patterns = ['DOJI', 'HAMMER', 'ENGULFING', 'PINBAR', 'NONE']
            ta.candlestick_pattern = np.random.choice(patterns)
            
            chart_patterns = ['TRIANGLE', 'CHANNEL', 'BREAKOUT', 'NONE']
            ta.chart_pattern = np.random.choice(chart_patterns)
            
        except Exception as e:
            self.logger.error(f"Technical indicator calculation error: {e}")
    
    def _calculate_trend_score(self, ta: TechnicalAnalysis) -> float:
        """Calculate trend score"""
        try:
            score = 0.0
            
            # SMA alignment
            if ta.sma_20 > ta.sma_50:
                score += 25
            
            # Price vs SMA
            current_price = ta.bb_middle  # Using middle BB as proxy for current price
            if current_price > ta.sma_20:
                score += 25
            
            # MACD
            if ta.macd > ta.macd_signal:
                score += 25
            
            # MACD histogram
            if ta.macd_histogram > 0:
                score += 25
            
            return score
            
        except Exception as e:
            self.logger.error(f"Trend score calculation error: {e}")
            return 0.0
    
    def _calculate_momentum_score(self, ta: TechnicalAnalysis) -> float:
        """Calculate momentum score"""
        try:
            score = 0.0
            
            # RSI
            if 30 <= ta.rsi <= 70:
                score += 25
            elif ta.rsi > 70:
                score += 10  # Overbought
            elif ta.rsi < 30:
                score += 10  # Oversold
            
            # Stochastic
            if ta.stochastic_k > ta.stochastic_d:
                score += 25
            
            # Williams %R
            if -80 <= ta.williams_r <= -20:
                score += 25
            
            # Volume confirmation
            if ta.volume_ratio > 1.2:
                score += 25
            
            return score
            
        except Exception as e:
            self.logger.error(f"Momentum score calculation error: {e}")
            return 0.0
    
    def _calculate_volatility_score(self, ta: TechnicalAnalysis) -> float:
        """Calculate volatility score"""
        try:
            score = 0.0
            
            # Bollinger Band width
            if 1.0 <= ta.bb_width <= 3.0:
                score += 50  # Optimal volatility
            elif ta.bb_width < 1.0:
                score += 20  # Low volatility
            elif ta.bb_width > 3.0:
                score += 20  # High volatility
            
            # ATR relative to price
            current_price = ta.bb_middle
            atr_percent = (ta.atr / current_price) * 100 if current_price > 0 else 0
            if 0.5 <= atr_percent <= 2.0:
                score += 50
            
            return score
            
        except Exception as e:
            self.logger.error(f"Volatility score calculation error: {e}")
            return 0.0
    
    def _calculate_scan_priority(self, symbol: str, instrument: MarketInstrument) -> int:
        """Calculate scan priority for instrument"""
        try:
            priority = 50  # Base priority
            
            # Market type priority
            type_priorities = {
                MarketType.FOREX: 10,
                MarketType.COMMODITY: 20,
                MarketType.INDEX: 30,
                MarketType.CFD: 40
            }
            priority += type_priorities.get(instrument.market_type, 50)
            
            # Volatility priority
            if symbol in self.market_data:
                volatility = self.market_data[symbol].volatility
                if volatility > 0.01:
                    priority -= 10  # Higher priority for volatile instruments
            
            # Volume priority
            if symbol in self.market_data:
                volume_ratio = getattr(self.market_data[symbol], 'volume_ratio', 1.0)
                if volume_ratio > 1.5:
                    priority -= 10  # Higher priority for high volume
            
            return priority
            
        except Exception as e:
            self.logger.error(f"Priority calculation error: {e}")
            return 50
    
    def _scan_instrument(self, symbol: str) -> List[ScanOpportunity]:
        """Scan individual instrument for opportunities"""
        try:
            scan_start_time = time.time()
            opportunities = []
            
            if symbol not in self.market_data or symbol not in self.technical_analysis:
                return opportunities
            
            market_data = self.market_data[symbol]
            instrument = self.instruments[symbol]
            
            # Scan each timeframe
            timeframes = self.scanner_config.get('timeframes', ['M5', 'M15', 'M30', 'H1'])
            
            for timeframe in timeframes:
                if timeframe in self.technical_analysis[symbol]:
                    ta = self.technical_analysis[symbol][timeframe]
                    
                    # Check for buy opportunities
                    buy_opportunity = self._check_buy_opportunity(symbol, market_data, ta, timeframe)
                    if buy_opportunity:
                        opportunities.append(buy_opportunity)
                    
                    # Check for sell opportunities
                    sell_opportunity = self._check_sell_opportunity(symbol, market_data, ta, timeframe)
                    if sell_opportunity:
                        opportunities.append(sell_opportunity)
            
            # Update scan time statistics
            scan_time = time.time() - scan_start_time
            self._update_scan_time_stats(scan_time)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Instrument scan error for {symbol}: {e}")
            return []
    
    def _check_buy_opportunity(self, symbol: str, market_data: MarketData, 
                              ta: TechnicalAnalysis, timeframe: str) -> Optional[ScanOpportunity]:
        """Check for buy opportunity"""
        try:
            # Buy signal conditions
            buy_signals = 0
            total_signals = 0
            
            # Trend conditions
            total_signals += 1
            if ta.sma_20 > ta.sma_50 and market_data.last > ta.sma_20:
                buy_signals += 1
            
            # MACD conditions
            total_signals += 1
            if ta.macd > ta.macd_signal and ta.macd_histogram > 0:
                buy_signals += 1
            
            # RSI conditions
            total_signals += 1
            if 30 <= ta.rsi <= 70:
                buy_signals += 1
            
            # Stochastic conditions
            total_signals += 1
            if ta.stochastic_k > ta.stochastic_d and ta.stochastic_k < 80:
                buy_signals += 1
            
            # Volume confirmation
            total_signals += 1
            if ta.volume_ratio > 1.2:
                buy_signals += 1
            
            # Calculate confidence score
            confidence_score = buy_signals / total_signals if total_signals > 0 else 0
            
            # Minimum confidence threshold
            min_confidence = self.scanner_config.get('min_confidence_score', 0.7)
            if confidence_score < min_confidence:
                return None
            
            # Create opportunity
            opportunity_id = f"BUY_{symbol}_{timeframe}_{int(time.time())}"
            
            # Calculate entry parameters
            entry_price = market_data.ask
            stop_loss = ta.support_1
            take_profit = ta.resistance_1
            
            risk_reward_ratio = (take_profit - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0
            
            if risk_reward_ratio < 1.5:  # Minimum risk/reward
                return None
            
            # Determine grade
            grade = self._determine_opportunity_grade(confidence_score)
            
            opportunity = ScanOpportunity(
                opportunity_id=opportunity_id,
                symbol=symbol,
                market_type=self.instruments[symbol].market_type,
                signal_type='BUY',
                grade=grade,
                confidence_score=confidence_score,
                ml_score=0.0,  # Would be calculated by ML system
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                position_size=0.0,  # Would be calculated by risk manager
                timeframe=timeframe,
                scan_timestamp=datetime.now(),
                technical_analysis=ta,
                technical_score=ta.overall_score,
                expires_at=datetime.now() + timedelta(minutes=30)
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Buy opportunity check error: {e}")
            return None
    
    def _check_sell_opportunity(self, symbol: str, market_data: MarketData,
                               ta: TechnicalAnalysis, timeframe: str) -> Optional[ScanOpportunity]:
        """Check for sell opportunity"""
        try:
            # Sell signal conditions
            sell_signals = 0
            total_signals = 0
            
            # Trend conditions
            total_signals += 1
            if ta.sma_20 < ta.sma_50 and market_data.last < ta.sma_20:
                sell_signals += 1
            
            # MACD conditions
            total_signals += 1
            if ta.macd < ta.macd_signal and ta.macd_histogram < 0:
                sell_signals += 1
            
            # RSI conditions
            total_signals += 1
            if 30 <= ta.rsi <= 70:
                sell_signals += 1
            
            # Stochastic conditions
            total_signals += 1
            if ta.stochastic_k < ta.stochastic_d and ta.stochastic_k > 20:
                sell_signals += 1
            
            # Volume confirmation
            total_signals += 1
            if ta.volume_ratio > 1.2:
                sell_signals += 1
            
            # Calculate confidence score
            confidence_score = sell_signals / total_signals if total_signals > 0 else 0
            
            # Minimum confidence threshold
            min_confidence = self.scanner_config.get('min_confidence_score', 0.7)
            if confidence_score < min_confidence:
                return None
            
            # Create opportunity
            opportunity_id = f"SELL_{symbol}_{timeframe}_{int(time.time())}"
            
            # Calculate entry parameters
            entry_price = market_data.bid
            stop_loss = ta.resistance_1
            take_profit = ta.support_1
            
            risk_reward_ratio = (entry_price - take_profit) / (stop_loss - entry_price) if stop_loss > entry_price else 0
            
            if risk_reward_ratio < 1.5:  # Minimum risk/reward
                return None
            
            # Determine grade
            grade = self._determine_opportunity_grade(confidence_score)
            
            opportunity = ScanOpportunity(
                opportunity_id=opportunity_id,
                symbol=symbol,
                market_type=self.instruments[symbol].market_type,
                signal_type='SELL',
                grade=grade,
                confidence_score=confidence_score,
                ml_score=0.0,  # Would be calculated by ML system
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                position_size=0.0,  # Would be calculated by risk manager
                timeframe=timeframe,
                scan_timestamp=datetime.now(),
                technical_analysis=ta,
                technical_score=ta.overall_score,
                expires_at=datetime.now() + timedelta(minutes=30)
            )
            
            return opportunity
            
        except Exception as e:
            self.logger.error(f"Sell opportunity check error: {e}")
            return None
    
    def _determine_opportunity_grade(self, confidence_score: float) -> OpportunityGrade:
        """Determine opportunity grade based on confidence score"""
        if confidence_score >= 0.9:
            return OpportunityGrade.EXCELLENT
        elif confidence_score >= 0.75:
            return OpportunityGrade.GOOD
        elif confidence_score >= 0.6:
            return OpportunityGrade.FAIR
        else:
            return OpportunityGrade.POOR
    
    def _process_opportunity(self, opportunity: ScanOpportunity):
        """Process found opportunity"""
        try:
            # Store opportunity
            self.opportunities[opportunity.opportunity_id] = opportunity
            
            # Update statistics
            self.scan_statistics['opportunities_found'] += 1
            
            # Notify callbacks
            for callback in self.opportunity_callbacks:
                try:
                    callback(opportunity)
                except Exception as e:
                    self.logger.error(f"Opportunity callback error: {e}")
            
            self.logger.info(f"New {opportunity.grade.value} {opportunity.signal_type} opportunity found: "
                           f"{opportunity.symbol} ({opportunity.timeframe}) - "
                           f"Confidence: {opportunity.confidence_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"Opportunity processing error: {e}")
    
    def _rank_opportunities(self):
        """Rank all current opportunities"""
        try:
            current_time = datetime.now()
            valid_opportunities = []
            
            # Filter valid opportunities
            for opp_id, opportunity in self.opportunities.items():
                if opportunity.is_valid and opportunity.expires_at > current_time:
                    valid_opportunities.append(opportunity)
            
            # Sort by composite score
            valid_opportunities.sort(
                key=lambda x: (x.grade.value, x.confidence_score, x.risk_reward_ratio, x.technical_score),
                reverse=True
            )
            
            # Keep only top opportunities
            self.opportunity_rankings = valid_opportunities[:self.max_opportunities]
            
        except Exception as e:
            self.logger.error(f"Opportunity ranking error: {e}")
    
    def _cleanup_expired_opportunities(self):
        """Clean up expired opportunities"""
        try:
            current_time = datetime.now()
            expired_ids = []
            
            for opp_id, opportunity in self.opportunities.items():
                if opportunity.expires_at <= current_time:
                    expired_ids.append(opp_id)
            
            for opp_id in expired_ids:
                del self.opportunities[opp_id]
            
            if expired_ids:
                self.logger.debug(f"Cleaned up {len(expired_ids)} expired opportunities")
                
        except Exception as e:
            self.logger.error(f"Opportunity cleanup error: {e}")
    
    def _update_scan_statistics(self):
        """Update scan statistics"""
        try:
            self.scan_statistics['active_opportunities'] = len(self.opportunities)
            self.scan_statistics['ranked_opportunities'] = len(self.opportunity_rankings)
            
            # Calculate success rate (simplified)
            if self.scan_statistics['opportunities_found'] > 0:
                success_rate = (self.scan_statistics['successful_signals'] / 
                              self.scan_statistics['opportunities_found']) * 100
                self.scan_statistics['success_rate'] = success_rate
            
        except Exception as e:
            self.logger.error(f"Statistics update error: {e}")
    
    def _update_scan_time_stats(self, scan_time: float):
        """Update scan time statistics"""
        try:
            if self.scan_statistics['scan_time_avg'] == 0:
                self.scan_statistics['scan_time_avg'] = scan_time
            else:
                # Moving average
                self.scan_statistics['scan_time_avg'] = (
                    self.scan_statistics['scan_time_avg'] * 0.9 + scan_time * 0.1
                )
                
        except Exception as e:
            self.logger.error(f"Scan time statistics error: {e}")
    
    def get_top_opportunities(self, count: int = 20, market_type: MarketType = None,
                             signal_type: str = None) -> List[ScanOpportunity]:
        """Get top ranked opportunities"""
        try:
            opportunities = self.opportunity_rankings.copy()
            
            # Filter by market type
            if market_type:
                opportunities = [opp for opp in opportunities if opp.market_type == market_type]
            
            # Filter by signal type
            if signal_type:
                opportunities = [opp for opp in opportunities if opp.signal_type == signal_type]
            
            return opportunities[:count]
            
        except Exception as e:
            self.logger.error(f"Get top opportunities error: {e}")
            return []
    
    def get_opportunity_by_id(self, opportunity_id: str) -> Optional[ScanOpportunity]:
        """Get opportunity by ID"""
        return self.opportunities.get(opportunity_id)
    
    def get_scan_statistics(self) -> Dict[str, Any]:
        """Get scan statistics"""
        return self.scan_statistics.copy()
    
    def add_opportunity_callback(self, callback: Callable):
        """Add opportunity notification callback"""
        self.opportunity_callbacks.append(callback)
    
    def add_market_data_callback(self, callback: Callable):
        """Add market data update callback"""
        self.market_data_callbacks.append(callback)
    
    def should_scan(self) -> bool:
        """Check if scanner should be running"""
        return self.is_scanning and self.is_initialized
    
    def perform_scan(self):
        """Perform manual scan trigger"""
        try:
            if not self.should_scan():
                return
            
            # This method is called by the main bridge
            # Actual scanning happens in background threads
            pass
            
        except Exception as e:
            self.logger.error(f"Manual scan error: {e}")
    
    def is_healthy(self) -> bool:
        """Check if market scanner is healthy"""
        try:
            return (
                self.is_initialized and
                self.is_scanning and
                len(self.scan_threads) > 0 and
                len(self.data_threads) > 0 and
                datetime.now() - self.scan_statistics.get('last_scan_time', datetime.now()) < timedelta(minutes=5)
            )
        except:
            return False
    
    def stop(self):
        """Stop market scanner"""
        try:
            self.is_scanning = False
            self.logger.info("Market Scanner stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping market scanner: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create market scanner
    scanner = MarketScanner()
    
    # Initialize
    if scanner.initialize():
        print("✅ Market Scanner initialized successfully")
        
        # Let it run for a bit
        time.sleep(10)
        
        # Get top opportunities
        opportunities = scanner.get_top_opportunities(count=5)
        print(f"Found {len(opportunities)} top opportunities:")
        
        for opp in opportunities:
            print(f"  {opp.signal_type} {opp.symbol} ({opp.timeframe}) - "
                  f"Grade: {opp.grade.value}, Confidence: {opp.confidence_score:.2f}")
        
        # Get statistics
        stats = scanner.get_scan_statistics()
        print(f"Scan statistics: {stats}")
        
        # Stop
        scanner.stop()
    else:
        print("❌ Market Scanner initialization failed")