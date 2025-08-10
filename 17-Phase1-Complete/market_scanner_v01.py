#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Non-F&O Market Scanner
Created: 2024-08-05
Author: EA GlobalFlow Development Team
Description: Comprehensive Non-F&O Market Scanner

This module implements the advanced market scanner for non-F&O markets:
- Forex Markets (Currency Pairs)  
- International Commodities (Gold, Silver, Oil, etc.)
- CFDs (Stock indices, individual stocks)

The scanner:
- Scans entire symbol lists provided by broker
- Checks 34 entry conditions for each symbol
- Ranks opportunities by success probability
- Opens top N charts automatically with EA
- Provides single chart operation (no secondary charts)
- Integrates with account balance for dynamic lot sizing
"""

import sys
import os
import time
import logging
import threading
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import numpy as np
import pandas as pd
import sqlite3
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

# Add the parent directory to Python path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from Config.error_handler import GlobalErrorHandler
    from Config.system_monitor import SystemMonitor
    from Config.security_manager import SecurityManager
    from Config.entry_conditions_processor import EntryConditionsProcessor, MarketType, TechnicalIndicators, SignalDirection
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Creating minimal fallback classes...")
    
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
    
    # Minimal fallback classes
    class MarketType(Enum):
        FOREX = "forex"
        COMMODITY = "commodity"
        CFD = "cfd"
    
    class SignalDirection(Enum):
        BUY = "BUY"
        SELL = "SELL"
        NONE = "NONE"

class ScannerStatus(Enum):
    """Scanner status states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"

class OpportunityRank(Enum):
    """Opportunity ranking levels"""
    EXCELLENT = 5
    VERY_GOOD = 4
    GOOD = 3
    FAIR = 2
    POOR = 1

@dataclass
class MarketSymbol:
    """Market symbol information"""
    symbol: str
    name: str
    market_type: MarketType
    exchange: str
    currency: str
    pip_size: float
    min_lot_size: float
    max_lot_size: float
    lot_step: float
    margin_rate: float
    enabled: bool = True
    last_scanned: Optional[datetime] = None
    scan_priority: int = 1  # 1=high, 2=medium, 3=low

@dataclass
class MarketOpportunity:
    """Market opportunity result"""
    symbol: str
    market_type: MarketType
    signal_direction: SignalDirection
    confidence: float
    success_probability: float
    conditions_met: int
    total_conditions: int
    entry_price: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    potential_return: float
    volatility_score: float
    liquidity_score: float
    rank: OpportunityRank
    timestamp: datetime
    timeframe: str
    technical_data: Dict[str, float] = field(default_factory=dict)
    risk_metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ScannerMetrics:
    """Scanner performance metrics"""
    symbols_scanned: int = 0
    opportunities_found: int = 0
    charts_opened: int = 0
    scan_duration_seconds: float = 0.0
    last_scan_time: Optional[datetime] = None
    success_rate: float = 0.0
    avg_return: float = 0.0
    total_scans: int = 0

class NonFnOMarketScanner:
    """
    Comprehensive Non-F&O Market Scanner
    
    Scans forex, commodities, and CFD markets for trading opportunities
    using the 34 entry conditions system. Ranks opportunities and opens
    charts automatically based on success probability.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize Non-F&O Market Scanner"""
        self.logger = logging.getLogger(__name__)
        self.error_handler = GlobalErrorHandler()
        self.monitor = SystemMonitor()
        self.security = SecurityManager()
        
        # Configuration
        self.config = self._load_config(config_path)
        self.enabled = self.config.get('enabled', True)
        
        # Database connections
        self.db_path = self.config.get('database_path', 'C:\\EA_GlobalFlow_Bridge\\Data\\market_scanner.db')
        self.redis_client = None
        self._init_redis()
        
        # Entry conditions processor
        self.entry_processor = None
        self._init_entry_processor()
        
        # Market symbols
        self.symbols: Dict[str, MarketSymbol] = {}
        self.load_symbols()
        
        # Scanner state
        self.status = ScannerStatus.STOPPED
        self.is_scanning = False
        self.scan_thread = None
        self.stop_event = threading.Event()
        
        # Performance tracking
        self.metrics = ScannerMetrics()
        self.opportunities: List[MarketOpportunity] = []
        self.opened_charts: Dict[str, datetime] = {}
        
        # Threading
        self.lock = threading.Lock()
        self.max_workers = self.config.get('scanning', {}).get('max_workers', 4)
        
        # Account integration
        self.account_balance = 0.0
        self.max_risk_per_trade = 0.02  # 2% default
        
        # Initialize system
        self._init_database()
        self._start_background_tasks()
        
        self.logger.info(f"Non-F&O Market Scanner initialized with {len(self.symbols)} symbols")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load scanner configuration"""
        try:
            if config_path and os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    return json.load(f)
            
            # Default configuration
            return {
                'enabled': True,
                'scanning': {
                    'interval_seconds': 300,  # 5 minutes
                    'max_workers': 4,
                    'timeout_per_symbol': 10,
                    'max_charts_to_open': 20,
                    'min_success_probability': 0.75,
                    'scan_priority_markets': ['forex', 'commodity']
                },
                'markets': {
                    'forex': {
                        'enabled': True,
                        'major_pairs': True,
                        'minor_pairs': True,
                        'exotic_pairs': False,
                        'max_spread': 3.0  # pips
                    },
                    'commodity': {
                        'enabled': True,
                        'metals': True,
                        'energy': True,
                        'agricultural': False
                    },
                    'cfd': {
                        'enabled': True,
                        'indices': True,
                        'stocks': False  # Too many symbols
                    }
                },
                'risk_management': {
                    'max_risk_per_trade_percent': 2.0,
                    'max_total_risk_percent': 10.0,
                    'dynamic_lot_sizing': True,
                    'account_balance_integration': True
                },
                'ranking': {
                    'confidence_weight': 0.3,
                    'success_probability_weight': 0.25,
                    'risk_reward_weight': 0.2,
                    'volatility_weight': 0.15,
                    'liquidity_weight': 0.1
                },
                'chart_management': {
                    'auto_open_enabled': True,
                    'max_charts_per_market': 8,
                    'chart_close_on_signal_exit': True,
                    'chart_template': 'EA_GlobalFlow_Template'
                }
            }
        except Exception as e:
            self.error_handler.handle_error(e, "Scanner config loading", "medium")
            return {}
    
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=3,
                decode_responses=True,
                socket_timeout=5
            )
            self.redis_client.ping()
            self.logger.info("Redis connection established for market scanner")
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    def _init_entry_processor(self):
        """Initialize entry conditions processor"""
        try:
            # Import here to avoid circular imports
            from Config.entry_conditions_processor import get_entry_conditions_processor
            self.entry_processor = get_entry_conditions_processor()
            self.logger.info("Entry conditions processor initialized")
        except Exception as e:
            self.error_handler.handle_error(e, "Entry processor initialization", "high")
    
    def _init_database(self):
        """Initialize SQLite database"""
        try:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            with sqlite3.connect(self.db_path) as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS market_symbols (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT UNIQUE NOT NULL,
                        name TEXT,
                        market_type TEXT,
                        exchange TEXT,
                        currency TEXT,
                        pip_size REAL,
                        min_lot_size REAL,
                        max_lot_size REAL,
                        lot_step REAL,
                        margin_rate REAL,
                        enabled BOOLEAN DEFAULT 1,
                        scan_priority INTEGER DEFAULT 1,
                        last_scanned DATETIME,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS market_opportunities (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        market_type TEXT,
                        signal_direction TEXT,
                        confidence REAL,
                        success_probability REAL,
                        conditions_met INTEGER,
                        total_conditions INTEGER,
                        entry_price REAL,
                        stop_loss REAL,
                        take_profit REAL,
                        risk_reward_ratio REAL,
                        potential_return REAL,
                        volatility_score REAL,
                        liquidity_score REAL,
                        rank INTEGER,
                        timeframe TEXT,
                        technical_data TEXT,
                        risk_metrics TEXT,
                        metadata TEXT,
                        chart_opened BOOLEAN DEFAULT 0,
                        outcome TEXT,
                        actual_return REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS scanner_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        scan_timestamp DATETIME,
                        symbols_scanned INTEGER,
                        opportunities_found INTEGER,
                        charts_opened INTEGER,
                        scan_duration_seconds REAL,
                        success_rate REAL,
                        avg_return REAL,
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                    );
                    
                    CREATE TABLE IF NOT EXISTS opened_charts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        market_type TEXT,
                        opportunity_id INTEGER,
                        opened_at DATETIME,
                        closed_at DATETIME,
                        chart_template TEXT,
                        status TEXT DEFAULT 'active',
                        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (opportunity_id) REFERENCES market_opportunities (id)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_opportunities_timestamp 
                    ON market_opportunities(created_at);
                    
                    CREATE INDEX IF NOT EXISTS idx_opportunities_rank 
                    ON market_opportunities(rank, success_probability);
                    
                    CREATE INDEX IF NOT EXISTS idx_symbols_market_type 
                    ON market_symbols(market_type, enabled);
                """)
            
            self.logger.info("Market scanner database initialized successfully")
        except Exception as e:
            self.error_handler.handle_error(e, "Scanner database initialization", "high")
    
    def load_symbols(self):
        """Load market symbols configuration"""
        try:
            # Load forex symbols
            if self.config.get('markets', {}).get('forex', {}).get('enabled', True):
                self._load_forex_symbols()
            
            # Load commodity symbols
            if self.config.get('markets', {}).get('commodity', {}).get('enabled', True):
                self._load_commodity_symbols()
            
            # Load CFD symbols
            if self.config.get('markets', {}).get('cfd', {}).get('enabled', True):
                self._load_cfd_symbols()
            
            # Save symbols to database
            self._save_symbols_to_db()
            
            self.logger.info(f"Loaded {len(self.symbols)} market symbols")
            
        except Exception as e:
            self.error_handler.handle_error(e, "Symbol loading", "high")
    
    def _load_forex_symbols(self):
        """Load forex currency pairs"""
        try:
            forex_config = self.config.get('markets', {}).get('forex', {})
            
            # Major pairs
            if forex_config.get('major_pairs', True):
                major_pairs = [
                    ('EURUSD', 'Euro vs US Dollar'),
                    ('GBPUSD', 'British Pound vs US Dollar'),
                    ('USDJPY', 'US Dollar vs Japanese Yen'),
                    ('USDCHF', 'US Dollar vs Swiss Franc'),
                    ('AUDUSD', 'Australian Dollar vs US Dollar'),
                    ('USDCAD', 'US Dollar vs Canadian Dollar'),
                    ('NZDUSD', 'New Zealand Dollar vs US Dollar')
                ]
                
                for symbol, name in major_pairs:
                    self.symbols[symbol] = MarketSymbol(
                        symbol=symbol,
                        name=name,
                        market_type=MarketType.FOREX,
                        exchange='FOREX',
                        currency='USD',
                        pip_size=0.0001 if 'JPY' not in symbol else 0.01,
                        min_lot_size=0.01,
                        max_lot_size=100.0,
                        lot_step=0.01,
                        margin_rate=0.02,  # 50:1 leverage
                        scan_priority=1
                    )
            
            # Minor pairs
            if forex_config.get('minor_pairs', True):
                minor_pairs = [
                    ('EURGBP', 'Euro vs British Pound'),
                    ('EURJPY', 'Euro vs Japanese Yen'),
                    ('EURCHF', 'Euro vs Swiss Franc'),
                    ('EURAUD', 'Euro vs Australian Dollar'),
                    ('EURCAD', 'Euro vs Canadian Dollar'),
                    ('GBPJPY', 'British Pound vs Japanese Yen'),
                    ('GBPCHF', 'British Pound vs Swiss Franc'),
                    ('AUDJPY', 'Australian Dollar vs Japanese Yen'),
                    ('CADJPY', 'Canadian Dollar vs Japanese Yen'),
                    ('CHFJPY', 'Swiss Franc vs Japanese Yen')
                ]
                
                for symbol, name in minor_pairs:
                    self.symbols[symbol] = MarketSymbol(
                        symbol=symbol,
                        name=name,
                        market_type=MarketType.FOREX,
                        exchange='FOREX',
                        currency='USD',
                        pip_size=0.0001 if 'JPY' not in symbol else 0.01,
                        min_lot_size=0.01,
                        max_lot_size=100.0,
                        lot_step=0.01,
                        margin_rate=0.02,
                        scan_priority=2
                    )
            
            # Exotic pairs (if enabled)
            if forex_config.get('exotic_pairs', False):
                exotic_pairs = [
                    ('USDZAR', 'US Dollar vs South African Rand'),
                    ('USDTRY', 'US Dollar vs Turkish Lira'),
                    ('USDSEK', 'US Dollar vs Swedish Krona'),
                    ('USDNOK', 'US Dollar vs Norwegian Krone'),
                    ('USDPLN', 'US Dollar vs Polish Zloty')
                ]
                
                for symbol, name in exotic_pairs:
                    self.symbols[symbol] = MarketSymbol(
                        symbol=symbol,
                        name=name,
                        market_type=MarketType.FOREX,
                        exchange='FOREX',
                        currency='USD',
                        pip_size=0.0001,
                        min_lot_size=0.01,
                        max_lot_size=50.0,
                        lot_step=0.01,
                        margin_rate=0.05,  # 20:1 leverage
                        scan_priority=3
                    )
                    
        except Exception as e:
            self.error_handler.handle_error(e, "Forex symbols loading", "medium")
    
    def _load_commodity_symbols(self):
        """Load commodity symbols"""
        try:
            commodity_config = self.config.get('markets', {}).get('commodity', {})
            
            # Metals
            if commodity_config.get('metals', True):
                metals = [
                    ('XAUUSD', 'Gold vs US Dollar'),
                    ('XAGUSD', 'Silver vs US Dollar'),
                    ('XPTUSD', 'Platinum vs US Dollar'),
                    ('XPDUSD', 'Palladium vs US Dollar')
                ]
                
                for symbol, name in metals:
                    self.symbols[symbol] = MarketSymbol(
                        symbol=symbol,
                        name=name,
                        market_type=MarketType.COMMODITY,
                        exchange='COMEX',
                        currency='USD',
                        pip_size=0.01,
                        min_lot_size=0.01,
                        max_lot_size=100.0,
                        lot_step=0.01,
                        margin_rate=0.01,  # 100:1 leverage
                        scan_priority=1
                    )
            
            # Energy
            if commodity_config.get('energy', True):
                energy = [
                    ('USOIL', 'US Crude Oil'),
                    ('UKOIL', 'UK Brent Oil'),
                    ('NGAS', 'Natural Gas')
                ]
                
                for symbol, name in energy:
                    self.symbols[symbol] = MarketSymbol(
                        symbol=symbol,
                        name=name,
                        market_type=MarketType.COMMODITY,
                        exchange='NYMEX',
                        currency='USD',
                        pip_size=0.01,
                        min_lot_size=0.01,
                        max_lot_size=100.0,
                        lot_step=0.01,
                        margin_rate=0.02,  # 50:1 leverage
                        scan_priority=1
                    )
            
            # Agricultural (if enabled)
            if commodity_config.get('agricultural', False):
                agricultural = [
                    ('WHEAT', 'Wheat Futures'),
                    ('CORN', 'Corn Futures'),
                    ('SOYBN', 'Soybean Futures'),
                    ('SUGAR', 'Sugar Futures'),
                    ('COCOA', 'Cocoa Futures'),
                    ('COFFEE', 'Coffee Futures')
                ]
                
                for symbol, name in agricultural:
                    self.symbols[symbol] = MarketSymbol(
                        symbol=symbol,
                        name=name,
                        market_type=MarketType.COMMODITY,
                        exchange='CBOT',
                        currency='USD',
                        pip_size=0.25,
                        min_lot_size=0.1,
                        max_lot_size=50.0,
                        lot_step=0.1,
                        margin_rate=0.05,  # 20:1 leverage
                        scan_priority=3
                    )
                    
        except Exception as e:
            self.error_handler.handle_error(e, "Commodity symbols loading", "medium")
    
    def _load_cfd_symbols(self):
        """Load CFD symbols"""
        try:
            cfd_config = self.config.get('markets', {}).get('cfd', {})
            
            # Stock indices
            if cfd_config.get('indices', True):
                indices = [
                    ('US30', 'Dow Jones Industrial Average'),
                    ('SPX500', 'S&P 500'),
                    ('NAS100', 'NASDAQ 100'),
                    ('UK100', 'FTSE 100'),
                    ('GER30', 'DAX 30'),
                    ('FRA40', 'CAC 40'),
                    ('JPN225', 'Nikkei 225'),
                    ('AUS200', 'ASX 200'),
                    ('HK50', 'Hang Seng 50')
                ]
                
                for symbol, name in indices:
                    self.symbols[symbol] = MarketSymbol(
                        symbol=symbol,
                        name=name,
                        market_type=MarketType.CFD,
                        exchange='CFD',
                        currency='USD',
                        pip_size=0.1,
                        min_lot_size=0.01,
                        max_lot_size=100.0,
                        lot_step=0.01,
                        margin_rate=0.01,  # 100:1 leverage
                        scan_priority=1
                    )
            
            # Individual stocks (if enabled - usually too many)
            if cfd_config.get('stocks', False):
                # Major US stocks
                stocks = [
                    ('AAPL', 'Apple Inc.'),
                    ('MSFT', 'Microsoft Corporation'),
                    ('GOOGL', 'Alphabet Inc.'),
                    ('AMZN', 'Amazon.com Inc.'),
                    ('TSLA', 'Tesla Inc.'),
                    ('META', 'Meta Platforms Inc.'),
                    ('NFLX', 'Netflix Inc.'),
                    ('NVDA', 'NVIDIA Corporation')
                ]
                
                for symbol, name in stocks:
                    self.symbols[symbol] = MarketSymbol(
                        symbol=symbol,
                        name=name,
                        market_type=MarketType.CFD,
                        exchange='NASDAQ',
                        currency='USD',
                        pip_size=0.01,
                        min_lot_size=0.01,
                        max_lot_size=10.0,
                        lot_step=0.01,
                        margin_rate=0.05,  # 20:1 leverage
                        scan_priority=2
                    )
                    
        except Exception as e:
            self.error_handler.handle_error(e, "CFD symbols loading", "medium")
    
    def _save_symbols_to_db(self):
        """Save symbols to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for symbol_obj in self.symbols.values():
                    cursor.execute("""
                        INSERT OR REPLACE INTO market_symbols (
                            symbol, name, market_type, exchange, currency,
                            pip_size, min_lot_size, max_lot_size, lot_step,
                            margin_rate, enabled, scan_priority
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol_obj.symbol,
                        symbol_obj.name,
                        symbol_obj.market_type.value,
                        symbol_obj.exchange,
                        symbol_obj.currency,
                        symbol_obj.pip_size,
                        symbol_obj.min_lot_size,
                        symbol_obj.max_lot_size,
                        symbol_obj.lot_step,
                        symbol_obj.margin_rate,
                        symbol_obj.enabled,
                        symbol_obj.scan_priority
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.error_handler.handle_error(e, "Symbols database save", "medium")
    
    def start_scanning(self):
        """Start the market scanning process"""
        try:
            if self.is_scanning:
                self.logger.warning("Scanner is already running")
                return False
            
            if not self.enabled:
                self.logger.warning("Scanner is disabled")
                return False
            
            self.status = ScannerStatus.STARTING
            self.is_scanning = True
            self.stop_event.clear()
            
            # Start scanning thread
            self.scan_thread = threading.Thread(
                target=self._scanning_loop,
                daemon=True,
                name="Market_Scanner_Loop"
            )
            self.scan_thread.start()
            
            self.status = ScannerStatus.RUNNING
            self.logger.info("Market scanner started successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, "Scanner startup", "high")
            self.status = ScannerStatus.ERROR
            self.is_scanning = False
            return False
    
    def stop_scanning(self):
        """Stop the market scanning process"""
        try:
            if not self.is_scanning:
                self.logger.warning("Scanner is not running")
                return
            
            self.logger.info("Stopping market scanner...")
            self.stop_event.set()
            self.is_scanning = False
            
            # Wait for scan thread to finish
            if self.scan_thread and self.scan_thread.is_alive():
                self.scan_thread.join(timeout=30)
            
            self.status = ScannerStatus.STOPPED
            self.logger.info("Market scanner stopped successfully")
            
        except Exception as e:
            self.error_handler.handle_error(e, "Scanner shutdown", "medium")
            self.status = ScannerStatus.ERROR
    
    def _scanning_loop(self):
        """Main scanning loop"""
        while not self.stop_event.is_set() and self.is_scanning:
            try:
                scan_start_time = time.time()
                
                # Perform market scan
                opportunities = self._perform_market_scan()
                
                # Rank opportunities
                ranked_opportunities = self._rank_opportunities(opportunities)
                
                # Open charts for best opportunities
                charts_opened = self._open_charts_for_opportunities(ranked_opportunities)
                
                # Update metrics
                scan_duration = time.time() - scan_start_time
                self._update_scan_metrics(len(self.symbols), len(opportunities), charts_opened, scan_duration)
                
                # Log scan results
                self.logger.info(f"Market scan completed: {len(self.symbols)} symbols, "
                               f"{len(opportunities)} opportunities, {charts_opened} charts opened "
                               f"({scan_duration:.1f}s)")
                
                # Wait for next scan
                scan_interval = self.config.get('scanning', {}).get('interval_seconds', 300)
                self.stop_event.wait(scan_interval)
                
            except Exception as e:
                self.error_handler.handle_error(e, "Scanning loop", "high")
                self.status = ScannerStatus.ERROR
                self.stop_event.wait(60)  # Wait 1 minute before retry
    
    def _perform_market_scan(self) -> List[MarketOpportunity]:
        """Perform comprehensive market scan"""
        opportunities = []
        
        try:
            # Get enabled symbols sorted by priority
            enabled_symbols = [sym for sym in self.symbols.values() if sym.enabled]
            enabled_symbols.sort(key=lambda x: x.scan_priority)
            
            # Limit symbols based on configuration
            max_symbols = self.config.get('scanning', {}).get('max_symbols_per_scan', 100)
            symbols_to_scan = enabled_symbols[:max_symbols]
            
            # Use ThreadPoolExecutor for parallel scanning
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit scan tasks
                future_to_symbol = {
                    executor.submit(self._scan_single_symbol, symbol): symbol
                    for symbol in symbols_to_scan
                }
                
                # Collect results
                timeout = self.config.get('scanning', {}).get('timeout_per_symbol', 10)
                for future in as_completed(future_to_symbol, timeout=timeout * len(symbols_to_scan)):
                    symbol = future_to_symbol[future]
                    try:
                        opportunity = future.result(timeout=timeout)
                        if opportunity:
                            opportunities.append(opportunity)
                    except Exception as e:
                        self.error_handler.handle_error(e, f"Symbol scan {symbol.symbol}", "low")
            
            # Store opportunities
            self._store_opportunities(opportunities)
            
        except Exception as e:
            self.error_handler.handle_error(e, "Market scan execution", "high")
        
        return opportunities
    
    def _scan_single_symbol(self, symbol: MarketSymbol) -> Optional[MarketOpportunity]:
        """Scan a single symbol for opportunities"""
        try:
            # Get market data for symbol
            market_data = self._get_market_data(symbol.symbol)
            if not market_data:
                return None
            
            # Get technical indicators
            indicators = self._calculate_technical_indicators(symbol.symbol, market_data)
            if not indicators:
                return None
            
            # Process entry conditions if entry processor available
            entry_signal = None
            if self.entry_processor:
                timeframe_data = {
                    'D1': market_data.get('daily', {}),
                    '1H': market_data.get('hourly', {}),
                    '15M': market_data.get('m15', {})
                }
                
                entry_signal = self.entry_processor.process_entry_conditions(
                    symbol.symbol,
                    symbol.market_type,
                    timeframe_data,
                    indicators
                )
            
            if not entry_signal:
                return None
            
            # Calculate opportunity metrics
            opportunity = self._create_opportunity_from_signal(symbol, entry_signal, market_data, indicators)
            
            # Update symbol last scanned time
            symbol.last_scanned = datetime.now()
            
            return opportunity
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Single symbol scan {symbol.symbol}", "low")
            return None
    
    def _get_market_data(self, symbol: str) -> Optional[Dict]:
        """Get market data for symbol (mock implementation)"""
        try:
            # This would integrate with actual broker APIs
            # For now, return mock data structure
            
            # Mock current price based on symbol
            base_price = hash(symbol) % 1000 + 1000  # Generate consistent price
            
            return {
                'symbol': symbol,
                'bid': base_price - 0.0002,
                'ask': base_price + 0.0002,
                'price': base_price,
                'change': (hash(symbol + str(datetime.now().hour)) % 100 - 50) * 0.01,
                'change_percent': (hash(symbol + str(datetime.now().hour)) % 100 - 50) * 0.001,
                'volume': abs(hash(symbol + 'volume') % 1000000),
                'daily': {
                    'open': base_price * 0.999,
                    'high': base_price * 1.002,
                    'low': base_price * 0.998,
                    'close': base_price
                },
                'hourly': {
                    'open': base_price * 0.9995,
                    'high': base_price * 1.001,
                    'low': base_price * 0.9995,
                    'close': base_price
                },
                'm15': {
                    'open': base_price * 0.9998,
                    'high': base_price * 1.0005,
                    'low': base_price * 0.9998,
                    'close': base_price
                }
            }
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Market data retrieval {symbol}", "low")
            return None
    
    def _calculate_technical_indicators(self, symbol: str, market_data: Dict) -> Optional[TechnicalIndicators]:
        """Calculate technical indicators (mock implementation)"""
        try:
            price = market_data['price']
            
            # Mock technical indicators - in real implementation, would calculate from historical data
            return TechnicalIndicators(
                tenkan_sen=price * 1.001,
                kijun_sen=price * 0.999,
                senkou_span_a=price * 1.0005,
                senkou_span_b=price * 0.9995,
                kumo_top=price * 1.0005,
                kumo_bottom=price * 0.9995,
                tdi_rsi=50 + (hash(symbol) % 40 - 20),  # 30-70 range
                tdi_signal=50 + (hash(symbol + 'signal') % 30 - 15),  # 35-65 range
                tdi_market_base_line=50,
                smma_50=price * 0.998,
                bb_upper=price * 1.002,
                bb_middle=price,
                bb_lower=price * 0.998,
                bb_squeeze=hash(symbol) % 5 == 0,  # 20% chance
                bb_expansion=hash(symbol) % 4 == 0,  # 25% chance
                str_entry=price * 0.997,
                str_entry_signal=SignalDirection.BUY if hash(symbol) % 2 == 0 else SignalDirection.SELL,
                current_price=price,
                previous_close=price * (1 - market_data['change_percent']),
                volume=market_data['volume'],
                atr=price * 0.01,  # 1% ATR
                adx=25 + (hash(symbol + 'adx') % 50),  # 25-75 range
                rsi=50 + (hash(symbol + 'rsi') % 40 - 20)  # 30-70 range
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Technical indicators calculation {symbol}", "low")
            return None
    
    def _create_opportunity_from_signal(self, 
                                      symbol: MarketSymbol,
                                      entry_signal,
                                      market_data: Dict,
                                      indicators: TechnicalIndicators) -> MarketOpportunity:
        """Create market opportunity from entry signal"""
        try:
            # Calculate success probability based on confidence and conditions
            base_probability = entry_signal.confidence
            conditions_ratio = entry_signal.conditions_met / entry_signal.total_conditions
            success_probability = (base_probability * 0.7 + conditions_ratio * 0.3)
            
            # Calculate risk metrics
            price = market_data['price']
            atr_pct = indicators.atr / price
            
            # Dynamic stop loss and take profit based on ATR and volatility
            if entry_signal.signal_direction.value == 'BUY':
                stop_loss = price * (1 - atr_pct * 2)
                take_profit = price * (1 + atr_pct * 3)
            else:
                stop_loss = price * (1 + atr_pct * 2)
                take_profit = price * (1 - atr_pct * 3)
            
            risk_per_pip = abs(price - stop_loss)
            reward_per_pip = abs(take_profit - price)
            risk_reward_ratio = reward_per_pip / risk_per_pip if risk_per_pip > 0 else 0
            
            # Calculate potential return percentage
            potential_return = abs(take_profit - price) / price
            
            # Calculate volatility score (higher ATR = higher volatility)
            volatility_score = min(1.0, atr_pct * 50)  # Normalize ATR to 0-1 scale
            
            # Calculate liquidity score (mock - based on volume)
            avg_volume = market_data['volume']  # Would use historical average
            liquidity_score = min(1.0, market_data['volume'] / max(avg_volume, 1) * 0.5)
            
            # Determine rank based on multiple factors
            rank_score = (
                success_probability * 0.4 +
                min(risk_reward_ratio / 3, 1.0) * 0.3 +
                liquidity_score * 0.2 +
                (1 - volatility_score) * 0.1  # Lower volatility gets higher score
            )
            
            if rank_score >= 0.8:
                rank = OpportunityRank.EXCELLENT
            elif rank_score >= 0.7:
                rank = OpportunityRank.VERY_GOOD
            elif rank_score >= 0.6:
                rank = OpportunityRank.GOOD
            elif rank_score >= 0.5:
                rank = OpportunityRank.FAIR
            else:
                rank = OpportunityRank.POOR
            
            return MarketOpportunity(
                symbol=symbol.symbol,
                market_type=symbol.market_type,
                signal_direction=entry_signal.signal_direction,
                confidence=entry_signal.confidence,
                success_probability=success_probability,
                conditions_met=entry_signal.conditions_met,
                total_conditions=entry_signal.total_conditions,
                entry_price=price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                risk_reward_ratio=risk_reward_ratio,
                potential_return=potential_return,
                volatility_score=volatility_score,
                liquidity_score=liquidity_score,
                rank=rank,
                timestamp=datetime.now(),
                timeframe='15M',
                technical_data={
                    'atr': indicators.atr,
                    'rsi': indicators.rsi,
                    'adx': indicators.adx,
                    'volume': market_data['volume']
                },
                risk_metrics={
                    'atr_percent': atr_pct,
                    'risk_per_pip': risk_per_pip,
                    'reward_per_pip': reward_per_pip,
                    'rank_score': rank_score
                },
                metadata={
                    'path_type': entry_signal.path_type.value,
                    'major_trend': entry_signal.major_trend.value,
                    'middle_trend': entry_signal.middle_trend.value,
                    'market_data_timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Opportunity creation {symbol.symbol}", "medium")
            # Return a minimal opportunity
            return MarketOpportunity(
                symbol=symbol.symbol,
                market_type=symbol.market_type,
                signal_direction=SignalDirection.NONE,
                confidence=0.0,
                success_probability=0.0,
                conditions_met=0,
                total_conditions=34,
                entry_price=market_data.get('price', 0),
                stop_loss=0,
                take_profit=0,
                risk_reward_ratio=0,
                potential_return=0,
                volatility_score=0,
                liquidity_score=0,
                rank=OpportunityRank.POOR,
                timestamp=datetime.now(),
                timeframe='15M'
            )
    
    def _rank_opportunities(self, opportunities: List[MarketOpportunity]) -> List[MarketOpportunity]:
        """Rank opportunities by success probability and other factors"""
        try:
            # Filter opportunities by minimum success probability
            min_probability = self.config.get('scanning', {}).get('min_success_probability', 0.75)
            filtered_opportunities = [
                opp for opp in opportunities 
                if opp.success_probability >= min_probability and opp.rank.value >= 3
            ]
            
            # Sort by rank and success probability
            filtered_opportunities.sort(
                key=lambda x: (x.rank.value, x.success_probability, x.risk_reward_ratio),
                reverse=True
            )
            
            self.opportunities = filtered_opportunities
            return filtered_opportunities
            
        except Exception as e:
            self.error_handler.handle_error(e, "Opportunity ranking", "medium")
            return opportunities
    
    def _open_charts_for_opportunities(self, opportunities: List[MarketOpportunity]) -> int:
        """Open charts for best opportunities"""
        charts_opened = 0
        
        try:
            if not self.config.get('chart_management', {}).get('auto_open_enabled', True):
                return 0
            
            max_charts = self.config.get('scanning', {}).get('max_charts_to_open', 20)
            max_per_market = self.config.get('chart_management', {}).get('max_charts_per_market', 8)
            
            # Count existing charts by market type
            market_chart_counts = {}
            for market_type in [MarketType.FOREX, MarketType.COMMODITY, MarketType.CFD]:
                market_chart_counts[market_type] = sum(
                    1 for symbol in self.opened_charts.keys()
                    if self.symbols.get(symbol, MarketSymbol('', '', market_type, '', '', 0, 0, 0, 0, 0)).market_type == market_type
                )
            
            # Open charts for top opportunities
            for opportunity in opportunities[:max_charts]:
                if charts_opened >= max_charts:
                    break
                
                # Check market-specific limits
                market_count = market_chart_counts.get(opportunity.market_type, 0)
                if market_count >= max_per_market:
                    continue
                
                # Check if chart already open
                if opportunity.symbol in self.opened_charts:
                    continue
                
                # Calculate lot size based on account balance and risk
                lot_size = self._calculate_lot_size(opportunity)
                
                # Open chart (mock implementation)
                if self._open_chart(opportunity, lot_size):
                    charts_opened += 1
                    market_chart_counts[opportunity.market_type] += 1
                    self.opened_charts[opportunity.symbol] = datetime.now()
            
            return charts_opened
            
        except Exception as e:
            self.error_handler.handle_error(e, "Chart opening", "medium")
            return 0
    
    def _calculate_lot_size(self, opportunity: MarketOpportunity) -> float:
        """Calculate optimal lot size based on risk management"""
        try:
            if not self.config.get('risk_management', {}).get('dynamic_lot_sizing', True):
                return self.symbols[opportunity.symbol].min_lot_size
            
            # Get account balance (mock - would integrate with broker API)
            account_balance = self.account_balance or 10000  # Default $10k
            
            # Calculate risk amount
            risk_percent = self.config.get('risk_management', {}).get('max_risk_per_trade_percent', 2.0) / 100
            risk_amount = account_balance * risk_percent
            
            # Calculate position size based on stop loss distance
            symbol_obj = self.symbols[opportunity.symbol]
            price_diff = abs(opportunity.entry_price - opportunity.stop_loss)
            
            if price_diff <= 0:
                return symbol_obj.min_lot_size
            
            # For forex: risk_amount / (price_diff_in_pips * pip_value * lot_size) = lot_size
            # Simplified calculation
            pip_value = symbol_obj.pip_size * 10  # $10 per pip for standard lot
            pips_at_risk = price_diff / symbol_obj.pip_size
            
            if pips_at_risk <= 0:
                return symbol_obj.min_lot_size
            
            optimal_lot_size = risk_amount / (pips_at_risk * pip_value)
            
            # Apply constraints
            optimal_lot_size = max(symbol_obj.min_lot_size, optimal_lot_size)
            optimal_lot_size = min(symbol_obj.max_lot_size, optimal_lot_size)
            
            # Round to lot step
            optimal_lot_size = round(optimal_lot_size / symbol_obj.lot_step) * symbol_obj.lot_step
            
            return optimal_lot_size
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Lot size calculation {opportunity.symbol}", "low")
            return self.symbols[opportunity.symbol].min_lot_size
    
    def _open_chart(self, opportunity: MarketOpportunity, lot_size: float) -> bool:
        """Open chart for opportunity (mock implementation)"""
        try:
            # In real implementation, this would:
            # 1. Send command to MT5 to open chart
            # 2. Apply EA to the chart
            # 3. Set up trade parameters
            # 4. Configure chart template
            
            # Mock implementation - log the action
            self.logger.info(f"Opening chart: {opportunity.symbol} {opportunity.signal_direction.value} "
                           f"@ {opportunity.entry_price:.5f} (lot: {lot_size:.2f})")
            
            # Store chart info
            self._store_opened_chart(opportunity, lot_size)
            
            return True
            
        except Exception as e:
            self.error_handler.handle_error(e, f"Chart opening {opportunity.symbol}", "medium")
            return False
    
    def _store_opportunities(self, opportunities: List[MarketOpportunity]):
        """Store opportunities to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for opp in opportunities:
                    cursor.execute("""
                        INSERT INTO market_opportunities (
                            symbol, market_type, signal_direction, confidence,
                            success_probability, conditions_met, total_conditions,
                            entry_price, stop_loss, take_profit, risk_reward_ratio,
                            potential_return, volatility_score, liquidity_score,
                            rank, timeframe, technical_data, risk_metrics, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        opp.symbol,
                        opp.market_type.value,
                        opp.signal_direction.value,
                        opp.confidence,
                        opp.success_probability,
                        opp.conditions_met,
                        opp.total_conditions,
                        opp.entry_price,
                        opp.stop_loss,
                        opp.take_profit,
                        opp.risk_reward_ratio,
                        opp.potential_return,
                        opp.volatility_score,
                        opp.liquidity_score,
                        opp.rank.value,
                        opp.timeframe,
                        json.dumps(opp.technical_data),
                        json.dumps(opp.risk_metrics),
                        json.dumps(opp.metadata)
                    ))
                
                conn.commit()
                
        except Exception as e:
            self.error_handler.handle_error(e, "Opportunities storage", "medium")
    
    def _store_opened_chart(self, opportunity: MarketOpportunity, lot_size: float):
        """Store opened chart information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get opportunity ID
                cursor.execute("""
                    SELECT id FROM market_opportunities 
                    WHERE symbol = ? AND created_at >= ?
                    ORDER BY created_at DESC LIMIT 1
                """, (opportunity.symbol, (datetime.now() - timedelta(minutes=10)).isoformat()))
                
                result = cursor.fetchone()
                opportunity_id = result[0] if result else None
                
                cursor.execute("""
                    INSERT INTO opened_charts (
                        symbol, market_type, opportunity_id, opened_at, 
                        chart_template, status
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    opportunity.symbol,
                    opportunity.market_type.value,
                    opportunity_id,
                    datetime.now().isoformat(),
                    self.config.get('chart_management', {}).get('chart_template', 'EA_GlobalFlow_Template'),
                    'active'
                ))
                
                conn.commit()
                
        except Exception as e:
            self.error_handler.handle_error(e, "Chart storage", "low")
    
    def _update_scan_metrics(self, symbols_scanned: int, opportunities_found: int, charts_opened: int, duration: float):
        """Update scanning metrics"""
        try:
            self.metrics.symbols_scanned = symbols_scanned
            self.metrics.opportunities_found = opportunities_found
            self.metrics.charts_opened = charts_opened
            self.metrics.scan_duration_seconds = duration
            self.metrics.last_scan_time = datetime.now()
            self.metrics.total_scans += 1
            
            # Store metrics to database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO scanner_metrics (
                        scan_timestamp, symbols_scanned, opportunities_found,
                        charts_opened, scan_duration_seconds, success_rate, avg_return
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now().isoformat(),
                    symbols_scanned,
                    opportunities_found,
                    charts_opened,
                    duration,
                    self.metrics.success_rate,
                    self.metrics.avg_return
                ))
                conn.commit()
            
            # Update system monitor
            self.monitor.log_metric('scanner_symbols_scanned', symbols_scanned)
            self.monitor.log_metric('scanner_opportunities_found', opportunities_found)
            self.monitor.log_metric('scanner_charts_opened', charts_opened)
            self.monitor.log_metric('scanner_scan_duration', duration)
            
        except Exception as e:
            self.error_handler.handle_error(e, "Metrics update", "low")
    
    def _start_background_tasks(self):
        """Start background tasks"""
        try:
            # Start performance monitoring
            threading.Thread(
                target=self._performance_monitor_loop,
                daemon=True,
                name="Scanner_Performance_Monitor"
            ).start()
            
            # Start chart management
            threading.Thread(
                target=self._chart_management_loop,
                daemon=True,
                name="Scanner_Chart_Manager"
            ).start()
            
            self.logger.info("Scanner background tasks started")
        except Exception as e:
            self.error_handler.handle_error(e, "Background tasks startup", "medium")
    
    def _performance_monitor_loop(self):
        """Background performance monitoring loop"""
        while True:
            try:
                time.sleep(1800)  # 30 minutes
                self._update_performance_metrics()
            except Exception as e:
                self.error_handler.handle_error(e, "Performance monitoring loop", "low")
                time.sleep(300)  # 5 minute retry
    
    def _chart_management_loop(self):
        """Background chart management loop"""
        while True:
            try:
                time.sleep(600)  # 10 minutes
                self._cleanup_closed_charts()
            except Exception as e:
                self.error_handler.handle_error(e, "Chart management loop", "low")
                time.sleep(300)  # 5 minute retry
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate recent performance (last 24 hours)
                since_time = (datetime.now() - timedelta(hours=24)).isoformat()
                
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_opportunities,
                        AVG(CASE WHEN outcome = 'success' THEN 1.0 ELSE 0.0 END) as success_rate,
                        AVG(actual_return) as avg_return
                    FROM market_opportunities 
                    WHERE created_at > ? AND outcome IS NOT NULL
                """, (since_time,))
                
                result = cursor.fetchone()
                if result and result[0] > 0:
                    total_opportunities, success_rate, avg_return = result
                    
                    self.metrics.success_rate = success_rate or 0
                    self.metrics.avg_return = avg_return or 0
                    
                    # Log performance metrics
                    self.monitor.log_metric('scanner_success_rate_24h', self.metrics.success_rate)
                    self.monitor.log_metric('scanner_avg_return_24h', self.metrics.avg_return)
                    
                    self.logger.info(f"Scanner Performance 24h: {total_opportunities} opportunities, "
                                   f"{self.metrics.success_rate:.1%} success rate, "
                                   f"{self.metrics.avg_return:.2%} avg return")
                
        except Exception as e:
            self.error_handler.handle_error(e, "Performance metrics update", "low")
    
    def _cleanup_closed_charts(self):
        """Cleanup closed charts"""
        try:
            # Remove charts that are older than 24 hours or marked as closed
            cutoff_time = datetime.now() - timedelta(hours=24)
            
            charts_to_remove = []
            for symbol, opened_time in self.opened_charts.items():
                if opened_time < cutoff_time:
                    charts_to_remove.append(symbol)
            
            for symbol in charts_to_remove:
                del self.opened_charts[symbol]
                
            if charts_to_remove:
                self.logger.info(f"Cleaned up {len(charts_to_remove)} old charts")
                
        except Exception as e:
            self.error_handler.handle_error(e, "Chart cleanup", "low")
    
    def get_system_status(self) -> Dict:
        """Get scanner system status"""
        try:
            return {
                'enabled': self.enabled,
                'status': self.status.value,
                'is_scanning': self.is_scanning,
                'total_symbols': len(self.symbols),
                'enabled_symbols': len([s for s in self.symbols.values() if s.enabled]),
                'current_opportunities': len(self.opportunities),
                'opened_charts': len(self.opened_charts),
                'metrics': {
                    'symbols_scanned': self.metrics.symbols_scanned,
                    'opportunities_found': self.metrics.opportunities_found,
                    'charts_opened': self.metrics.charts_opened,
                    'scan_duration_seconds': self.metrics.scan_duration_seconds,
                    'last_scan_time': self.metrics.last_scan_time.isoformat() if self.metrics.last_scan_time else None,
                    'success_rate': self.metrics.success_rate,
                    'avg_return': self.metrics.avg_return,
                    'total_scans': self.metrics.total_scans
                },
                'redis_connected': self.redis_client is not None,
                'database_path': self.db_path
            }
        except Exception as e:
            self.error_handler.handle_error(e, "System status", "low")
            return {'error': str(e)}
    
    def get_current_opportunities(self, limit: int = 20) -> List[Dict]:
        """Get current market opportunities"""
        try:
            opportunities = []
            for opp in self.opportunities[:limit]:
                opportunities.append({
                    'symbol': opp.symbol,
                    'market_type': opp.market_type.value,
                    'signal_direction': opp.signal_direction.value,
                    'confidence': opp.confidence,
                    'success_probability': opp.success_probability,
                    'conditions_met': opp.conditions_met,
                    'entry_price': opp.entry_price,
                    'risk_reward_ratio': opp.risk_reward_ratio,
                    'potential_return': opp.potential_return,
                    'rank': opp.rank.value,
                    'timestamp': opp.timestamp.isoformat()
                })
            
            return opportunities
            
        except Exception as e:
            self.error_handler.handle_error(e, "Current opportunities", "low")
            return []
    
    def update_account_balance(self, balance: float):
        """Update account balance for lot size calculations"""
        try:
            self.account_balance = balance
            self.logger.info(f"Account balance updated: ${balance:,.2f}")
        except Exception as e:
            self.error_handler.handle_error(e, "Account balance update", "low")
    
    def shutdown(self):
        """Shutdown market scanner"""
        try:
            self.logger.info("Shutting down Market Scanner...")
            
            # Stop scanning
            self.stop_scanning()
            
            # Close Redis connection
            if self.redis_client:
                self.redis_client.close()
            
            self.enabled = False
            self.logger.info("Market Scanner shutdown complete")
            
        except Exception as e:
            self.error_handler.handle_error(e, "Scanner shutdown", "medium")

# Global instance
market_scanner = None

def get_market_scanner(config_path: str = None) -> NonFnOMarketScanner:
    """Get global market scanner instance"""
    global market_scanner
    if market_scanner is None:
        market_scanner = NonFnOMarketScanner(config_path)
    return market_scanner

if __name__ == "__main__":
    # Test the Market Scanner
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize scanner
        scanner = NonFnOMarketScanner()
        
        # Get system status
        status = scanner.get_system_status()
        print(f"Scanner Status: {json.dumps(status, indent=2)}")
        
        # Test single scan (without starting full scanning loop)
        print("\nTesting single market scan...")
        opportunities = scanner._perform_market_scan()
        
        print(f"Found {len(opportunities)} opportunities:")
        for opp in opportunities[:5]:  # Show top 5
            print(f"  {opp.symbol} {opp.signal_direction.value} - "
                  f"Confidence: {opp.confidence:.1%}, "
                  f"Success Prob: {opp.success_probability:.1%}, "
                  f"Rank: {opp.rank.name}")
        
        # Test opportunity ranking
        ranked = scanner._rank_opportunities(opportunities)
        print(f"\nRanked opportunities: {len(ranked)}")
        
        print("Market Scanner test completed successfully")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'scanner' in locals():
            scanner.shutdown()
