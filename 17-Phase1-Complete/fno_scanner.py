#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - F&O Market Scanner with Dynamic Charts
================================================================

This module scans Indian FnO markets (Futures, Index Options, Stock Options) 
using TrueData feed and implements the sophisticated Hybrid OI Logic to open 
secondary charts dynamically.

Key Features:
- Real-time FnO market scanning via TrueData API
- Hybrid OI Logic implementation (Directional + Independent Bias)
- Dynamic secondary chart management (Call + Put sides)
- ATM/OTM strike price analysis with 5-level scanning
- Integrated with Triple Enhancement System
- Multi-timeframe trend analysis (Daily/4H → 1H/30M → 15M/5M)

Author: EA GlobalFlow Pro Development Team
Date: August 2025
Version: v0.1 - Production Ready
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import requests
from websocket import WebSocketApp
import sqlite3
import redis
import win32pipe
import win32file

# Import internal modules
from truedata_bridge import TrueDataBridge
from option_chain_analyzer import OptionChainAnalyzer
from globalflow_risk_v01 import GlobalFlowRiskManager
from error_handler import ErrorHandler
from security_manager import SecurityManager
from system_monitor import SystemMonitor

@dataclass
class TrendAnalysis:
    """Container for multi-timeframe trend analysis"""
    major_trend: str  # "BULLISH", "BEARISH", "SIDEWAYS"
    middle_trend: str
    major_timeframe: str  # "DAILY" or "4H"
    middle_timeframe: str  # "1H" or "30M"
    continuation_signal: bool
    pullback_signal: bool
    confidence: float
    timestamp: datetime

@dataclass
class FnOAsset:
    """Container for FnO asset information"""
    symbol: str
    asset_type: str  # "FUTURES", "INDEX_OPTIONS", "STOCK_OPTIONS"
    underlying: str
    expiry_date: str
    strike_price: Optional[float] = None
    option_type: Optional[str] = None  # "CE" or "PE"
    lot_size: int = 1
    tick_size: float = 0.05
    current_price: float = 0.0
    volume: int = 0
    open_interest: int = 0
    last_update: Optional[datetime] = None

@dataclass
class SecondaryChart:
    """Container for secondary chart information"""
    chart_id: str
    symbol: str
    strike_price: float
    option_type: str  # "CE" or "PE"
    atm_price: float
    oi_bias_type: str  # "DIRECTIONAL" or "INDEPENDENT"
    open_time: datetime
    status: str  # "ACTIVE", "CLOSED", "ERROR"
    pnl: float = 0.0
    max_trades: int = 3
    current_trades: int = 0
    auto_close_triggers: List[str] = None

class FnOScanner:
    """
    Advanced F&O Market Scanner with Dynamic Chart Management
    
    This class implements the complete F&O scanning logic including:
    - Real-time market data processing via TrueData
    - Multi-timeframe trend analysis 
    - Hybrid OI Logic for ATM/OTM analysis
    - Dynamic secondary chart opening and management
    - Integration with Triple Enhancement System
    """
    
    def __init__(self, config_path: str = "Config/ea_config_v01.json"):
        """Initialize F&O Scanner with comprehensive configuration"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.fno_config = self.config.get('fno_settings', {})
        
        # Initialize logging
        self.logger = logging.getLogger('FnOScanner')
        self.logger.setLevel(logging.INFO)
        
        # Initialize core components
        self.error_handler = ErrorHandler()
        self.security_manager = SecurityManager()
        self.system_monitor = SystemMonitor()
        self.risk_manager = GlobalFlowRiskManager()
        
        # Initialize data bridges
        self.truedata_bridge = TrueDataBridge()
        self.option_chain_analyzer = OptionChainAnalyzer()
        
        # Scanner state management
        self.is_running = False
        self.scan_thread = None
        self.active_assets: Dict[str, FnOAsset] = {}
        self.active_charts: Dict[str, SecondaryChart] = {}
        self.scan_queue = queue.Queue()
        
        # Market timing
        self.market_hours = self._load_market_hours()
        self.trading_start = "09:30"
        self.trading_end = "15:29"
        
        # Performance metrics
        self.scan_metrics = {
            'total_scans': 0,
            'signals_generated': 0,
            'charts_opened': 0,
            'successful_trades': 0,
            'scan_speed_ms': 0,
            'last_scan_time': None
        }
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Database for persistence
        self.db_connection = self._init_database()
        
        # Redis for real-time data caching
        try:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
            self.redis_client.ping()
        except:
            self.redis_client = None
            self.logger.warning("Redis not available - using in-memory caching")
        
        self.logger.info("F&O Scanner initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Return default configuration if file loading fails"""
        return {
            'fno_settings': {
                'scan_interval_ms': 1000,
                'max_active_charts': 20,
                'max_charts_per_underlying': 4,
                'strike_levels_to_scan': 5,
                'min_oi_threshold': 100,
                'oi_bias_threshold': 30.0,
                'auto_close_time': "15:29",
                'enable_expiry_day_override': True,
                'enable_real_time_updates': True
            },
            'trading_hours': {
                'start': "09:30",
                'end': "15:29",
                'timezone': "Asia/Kolkata"
            },
            'risk_management': {
                'max_position_size': 100000,
                'max_daily_loss': 50000,
                'position_sizing_method': "VIX_BASED"
            }
        }

    def _load_market_hours(self) -> Dict:
        """Load market trading hours configuration"""
        try:
            with open("Config/trading_hours_v01.json", 'r') as f:
                hours_config = json.load(f)
            return hours_config.get('indian_fno', {
                'start': "09:30",
                'end': "15:29",
                'timezone': "Asia/Kolkata",
                'trading_days': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            })
        except:
            return {
                'start': "09:30",
                'end': "15:29",
                'timezone': "Asia/Kolkata",
                'trading_days': ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            }

    def _init_database(self) -> sqlite3.Connection:
        """Initialize SQLite database for persistence"""
        try:
            conn = sqlite3.connect('Data/fno_scanner.db', check_same_thread=False)
            
            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS scan_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    signal_type TEXT,
                    trend_analysis TEXT,
                    chart_opened BOOLEAN,
                    success BOOLEAN
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS active_charts (
                    chart_id TEXT PRIMARY KEY,
                    symbol TEXT,
                    strike_price REAL,
                    option_type TEXT,
                    open_time DATETIME,
                    close_time DATETIME,
                    status TEXT,
                    pnl REAL DEFAULT 0.0
                )
            ''')
            
            conn.commit()
            return conn
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return None

    async def start_scanner(self) -> bool:
        """Start the F&O market scanner"""
        try:
            if self.is_running:
                self.logger.warning("Scanner already running")
                return True
            
            # Validate market hours
            if not self._is_market_open():
                self.logger.info("Market is closed - scanner will wait")
                await self._wait_for_market_open()
            
            # Initialize data connections
            if not await self.truedata_bridge.connect():
                raise Exception("Failed to connect to TrueData")
            
            # Start scanner thread
            self.is_running = True
            self.scan_thread = threading.Thread(target=self._scanner_main_loop, daemon=True)
            self.scan_thread.start()
            
            self.logger.info("F&O Scanner started successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error("SCANNER_START_FAILED", str(e))
            return False

    async def stop_scanner(self) -> bool:
        """Stop the F&O market scanner"""
        try:
            self.is_running = False
            
            # Close all active charts
            await self._close_all_charts("SCANNER_SHUTDOWN")
            
            # Disconnect from data sources
            await self.truedata_bridge.disconnect()
            
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
            
            # Shutdown thread pool
            self.executor.shutdown(wait=True)
            
            self.logger.info("F&O Scanner stopped successfully")
            return True
            
        except Exception as e:
            self.error_handler.handle_error("SCANNER_STOP_FAILED", str(e))
            return False

    def _scanner_main_loop(self):
        """Main scanner loop - runs in separate thread"""
        self.logger.info("Scanner main loop started")
        
        while self.is_running:
            try:
                scan_start_time = time.time()
                
                # Check if market is open
                if not self._is_market_open():
                    time.sleep(60)  # Check every minute when market closed
                    continue
                
                # Get F&O assets to scan
                assets_to_scan = self._get_fno_assets_list()
                
                # Process assets in parallel
                scan_tasks = []
                for asset in assets_to_scan:
                    task = self.executor.submit(self._scan_asset, asset)
                    scan_tasks.append(task)
                
                # Wait for all scans to complete
                for task in scan_tasks:
                    try:
                        result = task.result(timeout=5.0)  # 5 second timeout per asset
                        if result:
                            self._process_scan_result(result)
                    except Exception as e:
                        self.logger.error(f"Asset scan failed: {e}")
                
                # Update performance metrics
                scan_duration_ms = (time.time() - scan_start_time) * 1000
                self._update_scan_metrics(scan_duration_ms)
                
                # Manage existing charts
                await self._manage_active_charts()
                
                # Dynamic sleep based on scan speed
                sleep_time = max(0.1, (self.fno_config.get('scan_interval_ms', 1000) / 1000) - (scan_duration_ms / 1000))
                time.sleep(sleep_time)
                
            except Exception as e:
                self.error_handler.handle_error("SCANNER_LOOP_ERROR", str(e))
                time.sleep(5)  # Error cooldown

    def _get_fno_assets_list(self) -> List[str]:
        """Get list of F&O assets to scan from TrueData"""
        try:
            # Get active F&O symbols from TrueData
            assets = self.truedata_bridge.get_fno_symbols()
            
            # Filter based on configuration
            filtered_assets = []
            for asset in assets:
                # Check if asset type is supported
                if self._is_supported_asset_type(asset):
                    # Check minimum liquidity requirements
                    if self._meets_liquidity_requirements(asset):
                        filtered_assets.append(asset)
            
            return filtered_assets[:100]  # Limit to prevent overload
            
        except Exception as e:
            self.logger.error(f"Failed to get F&O assets list: {e}")
            return []

    def _scan_asset(self, symbol: str) -> Optional[Dict]:
        """Scan individual F&O asset for entry conditions"""
        try:
            # Get asset basic info
            asset_info = self._get_asset_info(symbol)
            if not asset_info:
                return None
            
            # Only process Futures and Options
            if asset_info.asset_type not in ["FUTURES", "INDEX_OPTIONS", "STOCK_OPTIONS"]:
                return None
            
            # Step 1: Multi-timeframe trend analysis
            trend_analysis = self._analyze_multi_timeframe_trend(symbol)
            if not trend_analysis:
                return None
            
            # Step 2: Check for Continuation or Pullback path signals
            ct_signal = self._check_continuation_path(symbol, trend_analysis)
            pb_signal = self._check_pullback_path(symbol, trend_analysis)
            
            if not (ct_signal or pb_signal):
                return None
            
            # Step 3: Get current spot price and identify ATM
            current_price = self._get_current_price(symbol)
            if not current_price:
                return None
            
            # Step 4: For Options, perform Hybrid OI Analysis
            if asset_info.asset_type in ["INDEX_OPTIONS", "STOCK_OPTIONS"]:
                atm_strike = self._find_atm_strike(symbol, current_price)
                if not atm_strike:
                    return None
                
                # Hybrid OI Logic Analysis
                oi_analysis = await self.option_chain_analyzer.analyze_hybrid_oi_logic(
                    symbol, atm_strike, current_price
                )
                
                if not oi_analysis or not oi_analysis.get('should_open_charts'):
                    return None
                
                return {
                    'symbol': symbol,
                    'asset_info': asset_info,
                    'trend_analysis': trend_analysis,
                    'signal_type': 'CT' if ct_signal else 'PB',
                    'current_price': current_price,
                    'atm_strike': atm_strike,
                    'oi_analysis': oi_analysis,
                    'timestamp': datetime.now()
                }
            
            else:
                # For Futures, direct entry without secondary charts
                return {
                    'symbol': symbol,
                    'asset_info': asset_info,
                    'trend_analysis': trend_analysis,
                    'signal_type': 'CT' if ct_signal else 'PB',
                    'current_price': current_price,
                    'timestamp': datetime.now()
                }
                
        except Exception as e:
            self.logger.error(f"Asset scan failed for {symbol}: {e}")
            return None

    def _analyze_multi_timeframe_trend(self, symbol: str) -> Optional[TrendAnalysis]:
        """Perform multi-timeframe trend analysis"""
        try:
            # Get historical data for multiple timeframes
            daily_data = self.truedata_bridge.get_historical_data(symbol, "DAILY", 50)
            hourly_data = self.truedata_bridge.get_historical_data(symbol, "1H", 100)
            
            if not daily_data or not hourly_data:
                return None
            
            # Analyze major trend (Daily/4H)
            major_trend = self._calculate_trend_direction(daily_data)
            
            # Analyze middle trend (1H/30M)  
            middle_trend = self._calculate_trend_direction(hourly_data)
            
            # Check for continuation pattern
            continuation_signal = (major_trend == middle_trend and 
                                 major_trend != "SIDEWAYS")
            
            # Check for pullback pattern
            pullback_signal = (major_trend != "SIDEWAYS" and 
                             middle_trend != major_trend and
                             middle_trend != "SIDEWAYS")
            
            # Calculate confidence based on trend strength
            confidence = self._calculate_trend_confidence(daily_data, hourly_data)
            
            return TrendAnalysis(
                major_trend=major_trend,
                middle_trend=middle_trend,
                major_timeframe="DAILY",
                middle_timeframe="1H",
                continuation_signal=continuation_signal,
                pullback_signal=pullback_signal,
                confidence=confidence,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Trend analysis failed for {symbol}: {e}")
            return None

    def _calculate_trend_direction(self, price_data: List[Dict]) -> str:
        """Calculate trend direction using Ichimoku and price action"""
        try:
            if len(price_data) < 26:  # Minimum for Ichimoku calculation
                return "SIDEWAYS"
            
            # Convert to DataFrame for analysis
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
            
            # Trend determination
            if current_price > current_tenkan > current_kijun:
                return "BULLISH"
            elif current_price < current_tenkan < current_kijun:
                return "BEARISH"
            else:
                return "SIDEWAYS"
                
        except Exception as e:
            self.logger.error(f"Trend calculation failed: {e}")
            return "SIDEWAYS"

    def _calculate_trend_confidence(self, daily_data: List[Dict], hourly_data: List[Dict]) -> float:
        """Calculate trend confidence score (0.0 to 1.0)"""
        try:
            confidence_factors = []
            
            # Factor 1: Price momentum
            daily_momentum = self._calculate_momentum(daily_data, 14)
            hourly_momentum = self._calculate_momentum(hourly_data, 14)
            confidence_factors.append((abs(daily_momentum) + abs(hourly_momentum)) / 2)
            
            # Factor 2: Volume trend alignment
            volume_alignment = self._check_volume_alignment(daily_data)
            confidence_factors.append(volume_alignment)
            
            # Factor 3: Volatility stability
            volatility_score = self._calculate_volatility_score(hourly_data)
            confidence_factors.append(volatility_score)
            
            # Average all factors
            return min(1.0, sum(confidence_factors) / len(confidence_factors))
            
        except Exception as e:
            self.logger.error(f"Confidence calculation failed: {e}")
            return 0.5

    def _check_continuation_path(self, symbol: str, trend_analysis: TrendAnalysis) -> bool:
        """Check if Continuation path conditions are met"""
        try:
            # Continuation path requires trend alignment
            if not trend_analysis.continuation_signal:
                return False
            
            # Get current market data for additional checks
            current_data = self.truedata_bridge.get_current_data(symbol)
            if not current_data:
                return False
            
            # Additional continuation filters
            # 1. Price should be near trend direction
            # 2. Volume should support the move
            # 3. No major resistance/support nearby
            
            return self._validate_continuation_conditions(symbol, current_data, trend_analysis)
            
        except Exception as e:
            self.logger.error(f"Continuation path check failed for {symbol}: {e}")
            return False

    def _check_pullback_path(self, symbol: str, trend_analysis: TrendAnalysis) -> bool:
        """Check if Pullback path conditions are met"""
        try:
            # Pullback path requires trend divergence with potential reversal
            if not trend_analysis.pullback_signal:
                return False
            
            # Get current market data
            current_data = self.truedata_bridge.get_current_data(symbol)
            if not current_data:
                return False
            
            # Additional pullback filters
            # 1. Price should be retracing against major trend
            # 2. Look for reversal signals
            # 3. Support/resistance levels
            
            return self._validate_pullback_conditions(symbol, current_data, trend_analysis)
            
        except Exception as e:
            self.logger.error(f"Pullback path check failed for {symbol}: {e}")
            return False

    async def _process_scan_result(self, scan_result: Dict):
        """Process a positive scan result and potentially open charts"""
        try:
            symbol = scan_result['symbol']
            asset_info = scan_result['asset_info']
            
            # Check if we can open more charts
            if not self._can_open_new_charts(symbol):
                self.logger.info(f"Cannot open new charts for {symbol} - limits reached")
                return
            
            # For Options, open secondary charts
            if asset_info.asset_type in ["INDEX_OPTIONS", "STOCK_OPTIONS"]:
                await self._open_secondary_charts(scan_result)
            
            # For Futures, process direct entry
            else:
                await self._process_futures_signal(scan_result)
            
            # Update metrics
            self.scan_metrics['signals_generated'] += 1
            
            # Store scan result in database
            self._store_scan_result(scan_result)
            
        except Exception as e:
            self.error_handler.handle_error("SCAN_RESULT_PROCESSING_FAILED", str(e))

    async def _open_secondary_charts(self, scan_result: Dict):
        """Open secondary charts for Options based on Hybrid OI Logic"""
        try:
            symbol = scan_result['symbol']
            oi_analysis = scan_result['oi_analysis']
            atm_strike = scan_result['atm_strike']
            
            # Get chart configuration from OI analysis
            call_strike = oi_analysis['call_chart_strike']
            put_strike = oi_analysis['put_chart_strike']
            
            # Generate unique chart IDs
            call_chart_id = f"{symbol}_CALL_{call_strike}_{int(time.time())}"
            put_chart_id = f"{symbol}_PUT_{put_strike}_{int(time.time())}"
            
            # Create Call chart
            call_chart = SecondaryChart(
                chart_id=call_chart_id,
                symbol=f"{symbol}{call_strike}CE",
                strike_price=call_strike,
                option_type="CE",
                atm_price=atm_strike,
                oi_bias_type=oi_analysis['bias_type'],
                open_time=datetime.now(),
                status="ACTIVE",
                auto_close_triggers=["STOP_LOSS", "TAKE_PROFIT", "3:29_PM", "POSITION_LIMIT"]
            )
            
            # Create Put chart
            put_chart = SecondaryChart(
                chart_id=put_chart_id,
                symbol=f"{symbol}{put_strike}PE",
                strike_price=put_strike,
                option_type="PE",
                atm_price=atm_strike,
                oi_bias_type=oi_analysis['bias_type'],
                open_time=datetime.now(),
                status="ACTIVE",
                auto_close_triggers=["STOP_LOSS", "TAKE_PROFIT", "3:29_PM", "POSITION_LIMIT"]
            )
            
            # Add to active charts
            self.active_charts[call_chart_id] = call_chart
            self.active_charts[put_chart_id] = put_chart
            
            # Send to MT5 via Named Pipes
            await self._send_chart_open_command(call_chart)
            await self._send_chart_open_command(put_chart)
            
            # Update metrics
            self.scan_metrics['charts_opened'] += 2
            
            self.logger.info(f"Opened secondary charts for {symbol}: Call@{call_strike}, Put@{put_strike}")
            
        except Exception as e:
            self.error_handler.handle_error("SECONDARY_CHART_OPEN_FAILED", str(e))

    async def _send_chart_open_command(self, chart: SecondaryChart):
        """Send chart open command to MT5 via Named Pipes"""
        try:
            command = {
                'action': 'OPEN_SECONDARY_CHART',
                'chart_id': chart.chart_id,
                'symbol': chart.symbol,
                'strike_price': chart.strike_price,
                'option_type': chart.option_type,
                'oi_bias_type': chart.oi_bias_type,
                'visual_marker': 'CALL_CHART' if chart.option_type == 'CE' else 'PUT_CHART',
                'timestamp': chart.open_time.isoformat()
            }
            
            # Send via Named Pipe to MT5
            pipe_name = r'\\.\pipe\EA_GlobalFlow_Bridge'
            try:
                with open(pipe_name, 'w') as pipe:
                    json.dump(command, pipe)
                    
                self.logger.info(f"Chart open command sent: {chart.chart_id}")
                
            except Exception as pipe_error:
                self.logger.error(f"Named pipe communication failed: {pipe_error}")
                # Fallback: Store command for later retry
                self._store_pending_command(command)
                
        except Exception as e:
            self.logger.error(f"Failed to send chart open command: {e}")

    async def _manage_active_charts(self):
        """Manage all active secondary charts"""
        try:
            current_time = datetime.now()
            charts_to_close = []
            
            for chart_id, chart in self.active_charts.items():
                # Check auto-close conditions
                should_close, close_reason = self._should_close_chart(chart, current_time)
                
                if should_close:
                    charts_to_close.append((chart_id, close_reason))
                else:
                    # Update chart status
                    await self._update_chart_status(chart)
            
            # Close charts that meet close conditions
            for chart_id, reason in charts_to_close:
                await self._close_chart(chart_id, reason)
                
        except Exception as e:
            self.logger.error(f"Chart management failed: {e}")

    def _should_close_chart(self, chart: SecondaryChart, current_time: datetime) -> Tuple[bool, str]:
        """Check if chart should be auto-closed"""
        try:
            # Check time-based closure (3:29 PM)
            if current_time.strftime("%H:%M") >= "15:29":
                return True, "END_OF_TRADING_SESSION"
            
            # Check expiry day special handling
            if self._is_expiry_day(chart.symbol):
                if current_time.strftime("%H:%M") >= "15:25":  # Close 4 minutes early on expiry
                    return True, "EXPIRY_DAY_CLOSE"
            
            # Check position limits
            if chart.current_trades >= chart.max_trades:
                return True, "MAX_TRADES_REACHED"
            
            # Check if underlying has moved significantly (risk management)
            current_underlying_price = self._get_current_price(chart.symbol.split('CE')[0].split('PE')[0])
            if current_underlying_price:
                price_move_percent = abs(current_underlying_price - chart.atm_price) / chart.atm_price * 100
                if price_move_percent > 5.0:  # 5% move threshold
                    return True, "UNDERLYING_MOVED_SIGNIFICANTLY"
            
            # Check chart age (max 4 hours active)
            chart_age_hours = (current_time - chart.open_time).total_seconds() / 3600
            if chart_age_hours > 4:
                return True, "CHART_AGE_LIMIT"
            
            return False, ""
            
        except Exception as e:
            self.logger.error(f"Chart close check failed: {e}")
            return True, "ERROR_CHECK_FAILED"

    async def _close_chart(self, chart_id: str, reason: str):
        """Close a specific secondary chart"""
        try:
            if chart_id not in self.active_charts:
                return
            
            chart = self.active_charts[chart_id]
            chart.status = "CLOSED"
            
            # Send close command to MT5
            close_command = {
                'action': 'CLOSE_CHART',
                'chart_id': chart_id,
                'reason': reason,
                'timestamp': datetime.now().isoformat()
            }
            
            # Send via Named Pipe
            await self._send_mt5_command(close_command)
            
            # Update database
            if self.db_connection:
                self.db_connection.execute(
                    "UPDATE active_charts SET status = ?, close_time = ? WHERE chart_id = ?",
                    ("CLOSED", datetime.now(), chart_id)
                )
                self.db_connection.commit()
            
            # Remove from active charts
            del self.active_charts[chart_id]
            
            self.logger.info(f"Chart closed: {chart_id} - Reason: {reason}")
            
        except Exception as e:
            self.logger.error(f"Chart closure failed for {chart_id}: {e}")

    def _is_market_open(self) -> bool:
        """Check if Indian F&O market is currently open"""
        try:
            now = datetime.now()
            current_time = now.strftime("%H:%M")
            current_day = now.strftime("%A")
            
            # Check trading days
            if current_day not in self.market_hours.get('trading_days', []):
                return False
            
            # Check trading hours
            start_time = self.market_hours.get('start', '09:30')
            end_time = self.market_hours.get('end', '15:29')
            
            return start_time <= current_time <= end_time
            
        except Exception as e:
            self.logger.error(f"Market hours check failed: {e}")
            return False

    def _get_asset_info(self, symbol: str) -> Optional[FnOAsset]:
        """Get detailed asset information"""
        try:
            # Parse symbol to identify asset type
            asset_type = self._identify_asset_type(symbol)
            
            # Get basic info from TrueData
            asset_data = self.truedata_bridge.get_instrument_info(symbol)
            if not asset_data:
                return None
            
            return FnOAsset(
                symbol=symbol,
                asset_type=asset_type,
                underlying=asset_data.get('underlying', symbol),
                expiry_date=asset_data.get('expiry', ''),
                strike_price=asset_data.get('strike_price'),
                option_type=asset_data.get('option_type'),
                lot_size=asset_data.get('lot_size', 1),
                tick_size=asset_data.get('tick_size', 0.05),
                current_price=asset_data.get('ltp', 0.0),
                volume=asset_data.get('volume', 0),
                open_interest=asset_data.get('oi', 0),
                last_update=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Asset info retrieval failed for {symbol}: {e}")
            return None

    def _identify_asset_type(self, symbol: str) -> str:
        """Identify if symbol is Futures, Index Options, or Stock Options"""
        try:
            # Index Options (NIFTY, BANKNIFTY, etc.)
            index_prefixes = ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'SENSEX', 'BANKEX']
            for prefix in index_prefixes:
                if symbol.startswith(prefix) and ('CE' in symbol or 'PE' in symbol):
                    return "INDEX_OPTIONS"
            
            # Stock Options (contain CE or PE)
            if 'CE' in symbol or 'PE' in symbol:
                return "STOCK_OPTIONS"
            
            # Futures (no CE/PE suffix)
            return "FUTURES"
            
        except Exception as e:
            self.logger.error(f"Asset type identification failed for {symbol}: {e}")
            return "UNKNOWN"

    def _find_atm_strike(self, symbol: str, current_price: float) -> Optional[float]:
        """Find ATM (At-The-Money) strike price"""
        try:
            # Get strike interval for the underlying
            strike_interval = self._get_strike_interval(symbol)
            
            # Calculate nearest ATM strike
            atm_strike = round(current_price / strike_interval) * strike_interval
            
            return atm_strike
            
        except Exception as e:
            self.logger.error(f"ATM strike calculation failed for {symbol}: {e}")
            return None

    def _get_strike_interval(self, symbol: str) -> float:
        """Get strike price interval for symbol"""
        # NIFTY has 50 point intervals, others typically 100
        if symbol.startswith('NIFTY'):
            return 50.0
        else:
            return 100.0

    def get_scanner_status(self) -> Dict:
        """Get current scanner status and metrics"""
        return {
            'is_running': self.is_running,
            'market_open': self._is_market_open(),
            'active_charts_count': len(self.active_charts),
            'active_charts': {cid: {
                'symbol': chart.symbol,
                'strike': chart.strike_price,
                'type': chart.option_type,
                'status': chart.status,
                'open_time': chart.open_time.isoformat(),
                'trades': chart.current_trades
            } for cid, chart in self.active_charts.items()},
            'metrics': self.scan_metrics,
            'last_update': datetime.now().isoformat()
        }

    async def manual_scan_symbol(self, symbol: str) -> Dict:
        """Manually trigger scan for specific symbol"""
        try:
            self.logger.info(f"Manual scan requested for {symbol}")
            
            # Perform scan
            result = self._scan_asset(symbol)
            
            if result:
                await self._process_scan_result(result)
                return {
                    'success': True,
                    'symbol': symbol,
                    'signal_generated': True,
                    'result': result
                }
            else:
                return {
                    'success': True,
                    'symbol': symbol,
                    'signal_generated': False,
                    'message': 'No trading signal found'
                }
                
        except Exception as e:
            return {
                'success': False,
                'symbol': symbol,
                'error': str(e)
            }

    def _update_scan_metrics(self, scan_duration_ms: float):
        """Update scanner performance metrics"""
        self.scan_metrics['total_scans'] += 1
        self.scan_metrics['scan_speed_ms'] = scan_duration_ms
        self.scan_metrics['last_scan_time'] = datetime.now().isoformat()

    def _store_scan_result(self, scan_result: Dict):
        """Store scan result in database"""
        try:
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT INTO scan_history 
                    (symbol, signal_type, trend_analysis, chart_opened, success)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    scan_result['symbol'],
                    scan_result['signal_type'],
                    json.dumps(scan_result.get('trend_analysis', {}), default=str),
                    scan_result.get('asset_info', {}).get('asset_type') in ["INDEX_OPTIONS", "STOCK_OPTIONS"],
                    True
                ))
                self.db_connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store scan result: {e}")

    # Additional helper methods for completeness...
    def _is_supported_asset_type(self, asset: str) -> bool:
        """Check if asset type is supported"""
        return self._identify_asset_type(asset) in ["FUTURES", "INDEX_OPTIONS", "STOCK_OPTIONS"]

    def _meets_liquidity_requirements(self, asset: str) -> bool:
        """Check if asset meets minimum liquidity requirements"""
        try:
            asset_data = self.truedata_bridge.get_current_data(asset)
            if not asset_data:
                return False
            
            # Check minimum OI and volume
            min_oi = self.fno_config.get('min_oi_threshold', 100)
            return asset_data.get('oi', 0) >= min_oi
            
        except:
            return False

    def _can_open_new_charts(self, symbol: str) -> bool:
        """Check if new charts can be opened"""
        # Check global chart limit
        if len(self.active_charts) >= self.fno_config.get('max_active_charts', 20):
            return False
        
        # Check per-underlying limit
        underlying = symbol.split('CE')[0].split('PE')[0]
        underlying_charts = sum(1 for chart in self.active_charts.values() 
                              if underlying in chart.symbol)
        
        return underlying_charts < self.fno_config.get('max_charts_per_underlying', 4)

# Global instance for use by other modules
fno_scanner = None

def get_fno_scanner() -> FnOScanner:
    """Get singleton instance of FnO Scanner"""
    global fno_scanner
    if fno_scanner is None:
        fno_scanner = FnOScanner()
    return fno_scanner

if __name__ == "__main__":
    # Test the scanner
    import asyncio
    
    async def main():
        scanner = FnOScanner()
        
        # Start scanner
        await scanner.start_scanner()
        
        # Let it run for a while
        await asyncio.sleep(300)  # 5 minutes
        
        # Get status
        status = scanner.get_scanner_status()
        print(f"Scanner Status: {json.dumps(status, indent=2)}")
        
        # Stop scanner
        await scanner.stop_scanner()
    
    asyncio.run(main())
