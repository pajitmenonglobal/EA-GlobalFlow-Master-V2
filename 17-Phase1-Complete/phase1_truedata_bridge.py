#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - TrueData Bridge
Complete integration with TrueData feed for NSE/BSE market data

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
import requests
import websocket
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
import pandas as pd
from dataclasses import dataclass
from enum import Enum
import queue

class DataType(Enum):
    TICK = "TICK"
    OHLC = "OHLC"
    DEPTH = "DEPTH"
    OPTION_CHAIN = "OPTION_CHAIN"

class Exchange(Enum):
    NSE = "NSE"
    BSE = "BSE"
    NFO = "NFO"  # NSE F&O
    BFO = "BFO"  # BSE F&O

@dataclass
class TickData:
    symbol: str
    ltp: float
    volume: int
    open: float
    high: float
    low: float
    close: float
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: int = 0
    ask_qty: int = 0

@dataclass
class OptionData:
    symbol: str
    strike: float
    option_type: str  # CE/PE
    ltp: float
    volume: int
    oi: int
    oi_change: int
    iv: float
    delta: float
    gamma: float
    theta: float
    vega: float
    bid: float
    ask: float
    timestamp: datetime

class TruedataBridge:
    """
    TrueData Bridge for EA GlobalFlow Pro v0.1
    Handles real-time market data, F&O data, and option chains
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize TrueData bridge"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger('TruedataBridge')
        
        # Configuration
        self.username = ""
        self.password = ""
        self.api_key = ""
        self.base_url = "https://api.truedata.in"
        self.ws_url = "wss://ws.truedata.in"
        
        # Authentication
        self.access_token = ""
        self.is_authenticated = False
        self.is_initialized = False
        self.is_connected = False
        
        # WebSocket connection
        self.ws = None
        self.ws_thread = None
        self.is_ws_connected = False
        
        # Rate limiting
        self.last_request_time = 0
        self.requests_per_second = 5
        self.min_request_interval = 1.0 / self.requests_per_second
        
        # Data management
        self.tick_data_cache = {}
        self.option_data_cache = {}
        self.subscribed_symbols = set()
        self.data_callbacks = {}
        
        # Data queues
        self.tick_queue = queue.Queue(maxsize=1000)
        self.option_queue = queue.Queue(maxsize=1000)
        
        # Processing threads
        self.data_processor_thread = None
        self.is_processing = False
        
        # Connection monitoring
        self.connection_thread = None
        self.is_monitoring = False
        self.last_heartbeat = datetime.now()
        
        # Symbol mapping
        self.symbol_mapping = {}
        self.reverse_mapping = {}
        
    def initialize(self) -> bool:
        """
        Initialize TrueData bridge
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing TrueData Bridge v0.1...")
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Authenticate
            if not self._authenticate():
                return False
            
            # Load symbol mappings
            if not self._load_symbol_mappings():
                return False
            
            # Initialize WebSocket connection
            if not self._init_websocket():
                return False
            
            # Start data processing
            self._start_data_processing()
            
            # Start connection monitoring
            self._start_connection_monitoring()
            
            self.is_initialized = True
            self.logger.info("✅ TrueData Bridge initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"TrueData Bridge initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("truedata_init", e)
            return False
    
    def _load_config(self) -> bool:
        """Load TrueData configuration"""
        try:
            if self.config_manager:
                truedata_config = self.config_manager.get_config('api_integrations', {}).get('truedata', {})
                api_credentials = self.config_manager.get_config('api_credentials', {}).get('truedata', {})
            else:
                # Load from file
                config_file = os.path.join(os.path.dirname(__file__), 'Config', 'api_credentials_v01.json')
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        api_credentials = config.get('truedata', {})
                        truedata_config = {'enabled': True}
                else:
                    self.logger.error("API credentials file not found")
                    return False
            
            # Extract credentials
            self.username = api_credentials.get('username', '')
            self.password = api_credentials.get('password', '')
            self.api_key = api_credentials.get('api_key', '')
            
            # Validate credentials
            if not all([self.username, self.password, self.api_key]):
                self.logger.error("Missing TrueData API credentials")
                return False
            
            # Rate limiting config
            rate_limits = truedata_config.get('rate_limits', {})
            self.requests_per_second = rate_limits.get('data_requests_per_second', 5)
            self.min_request_interval = 1.0 / self.requests_per_second
            
            self.logger.info("TrueData configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load TrueData config: {e}")
            return False
    
    def _authenticate(self) -> bool:
        """Authenticate with TrueData API"""
        try:
            self.logger.info("Authenticating with TrueData API...")
            
            auth_data = {
                "username": self.username,
                "password": self.password,
                "api_key": self.api_key
            }
            
            response = requests.post(
                f"{self.base_url}/auth/login",
                json=auth_data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    self.access_token = result.get('access_token', '')
                    self.is_authenticated = True
                    self.is_connected = True
                    self.logger.info("✅ TrueData authentication successful")
                    return True
                else:
                    self.logger.error(f"TrueData authentication failed: {result}")
                    return False
            else:
                self.logger.error(f"TrueData authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"TrueData authentication error: {e}")
            return False
    
    def _load_symbol_mappings(self) -> bool:
        """Load symbol mappings for different exchanges"""
        try:
            # This would typically load from TrueData API or local file
            # For now, using basic mappings
            self.symbol_mapping = {
                'NIFTY': 'NSE:NIFTY50',
                'BANKNIFTY': 'NSE:BANKNIFTY',
                'SENSEX': 'BSE:SENSEX',
                'RELIANCE': 'NSE:RELIANCE'
            }
            
            # Create reverse mapping
            self.reverse_mapping = {v: k for k, v in self.symbol_mapping.items()}
            
            self.logger.info(f"Loaded {len(self.symbol_mapping)} symbol mappings")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load symbol mappings: {e}")
            return False
    
    def _init_websocket(self) -> bool:
        """Initialize WebSocket connection"""
        try:
            self.logger.info("Initializing WebSocket connection...")
            
            # WebSocket URL with auth token
            ws_url = f"{self.ws_url}?token={self.access_token}"
            
            # Create WebSocket connection
            self.ws = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_ws_open,
                on_message=self._on_ws_message,
                on_error=self._on_ws_error,
                on_close=self._on_ws_close
            )
            
            # Start WebSocket in separate thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
            self.ws_thread.start()
            
            # Wait for connection
            time.sleep(2)
            
            if self.is_ws_connected:
                self.logger.info("✅ WebSocket connected successfully")
                return True
            else:
                self.logger.error("❌ WebSocket connection failed")
                return False
                
        except Exception as e:
            self.logger.error(f"WebSocket initialization error: {e}")
            return False
    
    def _on_ws_open(self, ws):
        """WebSocket open callback"""
        self.logger.info("WebSocket connection opened")
        self.is_ws_connected = True
        self.last_heartbeat = datetime.now()
    
    def _on_ws_message(self, ws, message):
        """WebSocket message callback"""
        try:
            data = json.loads(message)
            message_type = data.get('type', '')
            
            if message_type == 'tick':
                self._process_tick_data(data)
            elif message_type == 'depth':
                self._process_depth_data(data)
            elif message_type == 'option':
                self._process_option_data(data)
            elif message_type == 'heartbeat':
                self.last_heartbeat = datetime.now()
            
        except Exception as e:
            self.logger.error(f"WebSocket message processing error: {e}")
    
    def _on_ws_error(self, ws, error):
        """WebSocket error callback"""
        self.logger.error(f"WebSocket error: {error}")
        self.is_ws_connected = False
    
    def _on_ws_close(self, ws, close_status_code, close_msg):
        """WebSocket close callback"""
        self.logger.warning(f"WebSocket connection closed: {close_status_code} - {close_msg}")
        self.is_ws_connected = False
    
    def _process_tick_data(self, data):
        """Process incoming tick data"""
        try:
            symbol = data.get('symbol', '')
            
            tick_data = TickData(
                symbol=symbol,
                ltp=data.get('ltp', 0.0),
                volume=data.get('volume', 0),
                open=data.get('open', 0.0),
                high=data.get('high', 0.0),
                low=data.get('low', 0.0),
                close=data.get('close', 0.0),
                timestamp=datetime.now(),
                bid=data.get('bid', 0.0),
                ask=data.get('ask', 0.0),
                bid_qty=data.get('bid_qty', 0),
                ask_qty=data.get('ask_qty', 0)
            )
            
            # Update cache
            self.tick_data_cache[symbol] = tick_data
            
            # Add to processing queue
            if not self.tick_queue.full():
                self.tick_queue.put(tick_data)
            
            # Call registered callbacks
            if symbol in self.data_callbacks:
                for callback in self.data_callbacks[symbol]:
                    try:
                        callback(tick_data)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
            
        except Exception as e:
            self.logger.error(f"Tick data processing error: {e}")
    
    def _process_depth_data(self, data):
        """Process market depth data"""
        try:
            symbol = data.get('symbol', '')
            self.logger.debug(f"Depth data received for {symbol}")
            # Process depth data as needed
            
        except Exception as e:
            self.logger.error(f"Depth data processing error: {e}")
    
    def _process_option_data(self, data):
        """Process option chain data"""
        try:
            symbol = data.get('symbol', '')
            
            option_data = OptionData(
                symbol=symbol,
                strike=data.get('strike', 0.0),
                option_type=data.get('option_type', ''),
                ltp=data.get('ltp', 0.0),
                volume=data.get('volume', 0),
                oi=data.get('oi', 0),
                oi_change=data.get('oi_change', 0),
                iv=data.get('iv', 0.0),
                delta=data.get('delta', 0.0),
                gamma=data.get('gamma', 0.0),
                theta=data.get('theta', 0.0),
                vega=data.get('vega', 0.0),
                bid=data.get('bid', 0.0),
                ask=data.get('ask', 0.0),
                timestamp=datetime.now()
            )
            
            # Update cache
            self.option_data_cache[symbol] = option_data
            
            # Add to processing queue
            if not self.option_queue.full():
                self.option_queue.put(option_data)
            
        except Exception as e:
            self.logger.error(f"Option data processing error: {e}")
    
    def _start_data_processing(self):
        """Start data processing thread"""
        try:
            self.is_processing = True
            self.data_processor_thread = threading.Thread(target=self._data_processor_loop, daemon=True)
            self.data_processor_thread.start()
            self.logger.info("Data processing thread started")
            
        except Exception as e:
            self.logger.error(f"Failed to start data processing: {e}")
    
    def _data_processor_loop(self):
        """Data processing loop"""
        while self.is_processing:
            try:
                # Process tick data
                try:
                    tick_data = self.tick_queue.get(timeout=1)
                    # Additional processing can be done here
                    self.tick_queue.task_done()
                except queue.Empty:
                    pass
                
                # Process option data
                try:
                    option_data = self.option_queue.get(timeout=1)
                    # Additional processing can be done here
                    self.option_queue.task_done()
                except queue.Empty:
                    pass
                
                time.sleep(0.1)  # Small delay
                
            except Exception as e:
                self.logger.error(f"Data processing error: {e}")
                time.sleep(1)
    
    def _start_connection_monitoring(self):
        """Start connection monitoring thread"""
        try:
            self.is_monitoring = True
            self.connection_thread = threading.Thread(target=self._connection_monitor_loop, daemon=True)
            self.connection_thread.start()
            self.logger.info("Connection monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start connection monitoring: {e}")
    
    def _connection_monitor_loop(self):
        """Connection monitoring loop"""
        while self.is_monitoring:
            try:
                # Check WebSocket connection
                if not self.is_ws_connected:
                    self.logger.warning("WebSocket disconnected, attempting reconnection...")
                    self._reconnect_websocket()
                
                # Check heartbeat
                time_since_heartbeat = datetime.now() - self.last_heartbeat
                if time_since_heartbeat.total_seconds() > 60:  # 1 minute timeout
                    self.logger.warning("No heartbeat received, connection may be lost")
                    self._reconnect_websocket()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Connection monitoring error: {e}")
                time.sleep(60)
    
    def _reconnect_websocket(self):
        """Reconnect WebSocket"""
        try:
            if self.ws:
                self.ws.close()
            
            time.sleep(5)  # Wait before reconnecting
            
            if self._init_websocket():
                # Re-subscribe to all symbols
                for symbol in self.subscribed_symbols.copy():
                    self._subscribe_symbol(symbol)
                
        except Exception as e:
            self.logger.error(f"WebSocket reconnection error: {e}")
    
    def _rate_limit_check(self):
        """Check and enforce rate limits"""
        try:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_request_interval:
                sleep_time = self.min_request_interval - time_since_last
                time.sleep(sleep_time)
            
            self.last_request_time = time.time()
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
    
    def subscribe_symbol(self, symbol: str, callback: Optional[Callable] = None) -> bool:
        """
        Subscribe to symbol for real-time data
        
        Args:
            symbol: Trading symbol
            callback: Optional callback function for data updates
            
        Returns:
            True if subscription successful
        """
        try:
            if not self.is_ws_connected:
                self.logger.error("WebSocket not connected")
                return False
            
            # Add callback if provided
            if callback:
                if symbol not in self.data_callbacks:
                    self.data_callbacks[symbol] = []
                self.data_callbacks[symbol].append(callback)
            
            # Subscribe via WebSocket
            if self._subscribe_symbol(symbol):
                self.subscribed_symbols.add(symbol)
                self.logger.info(f"✅ Subscribed to {symbol}")
                return True
            else:
                self.logger.error(f"❌ Failed to subscribe to {symbol}")
                return False
                
        except Exception as e:
            self.logger.error(f"Symbol subscription error: {e}")
            return False
    
    def _subscribe_symbol(self, symbol: str) -> bool:
        """Internal method to subscribe to symbol"""
        try:
            subscribe_msg = {
                "action": "subscribe",
                "symbol": symbol,
                "type": "tick"
            }
            
            self.ws.send(json.dumps(subscribe_msg))
            return True
            
        except Exception as e:
            self.logger.error(f"Symbol subscription error: {e}")
            return False
    
    def unsubscribe_symbol(self, symbol: str) -> bool:
        """Unsubscribe from symbol"""
        try:
            if not self.is_ws_connected:
                return False
            
            unsubscribe_msg = {
                "action": "unsubscribe",
                "symbol": symbol
            }
            
            self.ws.send(json.dumps(unsubscribe_msg))
            
            self.subscribed_symbols.discard(symbol)
            if symbol in self.data_callbacks:
                del self.data_callbacks[symbol]
            
            self.logger.info(f"✅ Unsubscribed from {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Symbol unsubscription error: {e}")
            return False
    
    def get_live_data(self, symbol: str) -> Optional[TickData]:
        """Get latest live data for symbol"""
        try:
            return self.tick_data_cache.get(symbol)
        except Exception as e:
            self.logger.error(f"Get live data error: {e}")
            return None
    
    def get_historical_data(self, symbol: str, interval: str, 
                           from_date: datetime, to_date: datetime) -> pd.DataFrame:
        """Get historical data"""
        try:
            if not self.is_authenticated:
                return pd.DataFrame()
            
            self._rate_limit_check()
            
            params = {
                "symbol": symbol,
                "interval": interval,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d")
            }
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(
                f"{self.base_url}/historical",
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    df = pd.DataFrame(data['data'])
                    if not df.empty:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        df.set_index('timestamp', inplace=True)
                    return df
                else:
                    self.logger.error(f"Historical data API error: {data}")
                    return pd.DataFrame()
            else:
                self.logger.error(f"Historical data request failed: {response.status_code}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Get historical data error: {e}")
            return pd.DataFrame()
    
    def get_option_chain(self, underlying: str, expiry: str) -> Dict[str, List[OptionData]]:
        """Get option chain data"""
        try:
            if not self.is_authenticated:
                return {}
            
            self._rate_limit_check()
            
            params = {
                "underlying": underlying,
                "expiry": expiry
            }
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(
                f"{self.base_url}/optionchain",
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    option_chain = {'CE': [], 'PE': []}
                    
                    for option in data['data']:
                        option_data = OptionData(
                            symbol=option['symbol'],
                            strike=option['strike'],
                            option_type=option['option_type'],
                            ltp=option['ltp'],
                            volume=option['volume'],
                            oi=option['oi'],
                            oi_change=option['oi_change'],
                            iv=option.get('iv', 0.0),
                            delta=option.get('delta', 0.0),
                            gamma=option.get('gamma', 0.0),
                            theta=option.get('theta', 0.0),
                            vega=option.get('vega', 0.0),
                            bid=option.get('bid', 0.0),
                            ask=option.get('ask', 0.0),
                            timestamp=datetime.now()
                        )
                        
                        option_chain[option['option_type']].append(option_data)
                    
                    return option_chain
                else:
                    self.logger.error(f"Option chain API error: {data}")
                    return {}
            else:
                self.logger.error(f"Option chain request failed: {response.status_code}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Get option chain error: {e}")
            return {}
    
    def get_fno_symbols(self, exchange: Exchange = Exchange.NFO) -> List[str]:
        """Get list of F&O symbols"""
        try:
            if not self.is_authenticated:
                return []
            
            self._rate_limit_check()
            
            params = {"exchange": exchange.value}
            headers = {"Authorization": f"Bearer {self.access_token}"}
            
            response = requests.get(
                f"{self.base_url}/symbols/fno",
                params=params,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get('success'):
                    return data['symbols']
                else:
                    self.logger.error(f"F&O symbols API error: {data}")
                    return []
            else:
                self.logger.error(f"F&O symbols request failed: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.error(f"Get F&O symbols error: {e}")
            return []
    
    def get_vix_data(self) -> Optional[float]:
        """Get India VIX data"""
        try:
            vix_data = self.get_live_data("NSE:INDIAVIX")
            if vix_data:
                return vix_data.ltp
            return None
            
        except Exception as e:
            self.logger.error(f"Get VIX data error: {e}")
            return None
    
    def maintain_connection(self):
        """Maintain API connection"""
        try:
            if not self.is_connected or not self.is_ws_connected:
                self.logger.info("Attempting to reconnect to TrueData...")
                if self._authenticate() and self._init_websocket():
                    self.logger.info("✅ Reconnected to TrueData successfully")
                else:
                    self.logger.error("❌ Failed to reconnect to TrueData")
            
        except Exception as e:
            self.logger.error(f"Connection maintenance error: {e}")
    
    def is_healthy(self) -> bool:
        """Check if TrueData bridge is healthy"""
        try:
            time_since_heartbeat = datetime.now() - self.last_heartbeat
            return (
                self.is_initialized and
                self.is_authenticated and
                self.is_connected and
                self.is_ws_connected and
                time_since_heartbeat.total_seconds() < 120  # 2 minutes
            )
        except:
            return False
    
    def stop(self):
        """Stop TrueData bridge"""
        try:
            self.is_monitoring = False
            self.is_processing = False
            
            if self.ws:
                self.ws.close()
            
            self.is_connected = False
            self.is_ws_connected = False
            self.logger.info("TrueData Bridge stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping TrueData bridge: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create TrueData bridge
    truedata_bridge = TruedataBridge()
    
    # Initialize
    if truedata_bridge.initialize():
        print("✅ TrueData Bridge initialized successfully")
        
        # Test symbol subscription
        def data_callback(tick_data):
            print(f"Received data: {tick_data.symbol} - LTP: {tick_data.ltp}")
        
        truedata_bridge.subscribe_symbol("NSE:NIFTY50", data_callback)
        
        # Test VIX data
        vix_value = truedata_bridge.get_vix_data()
        print(f"VIX: {vix_value}")
        
        # Test F&O symbols
        fno_symbols = truedata_bridge.get_fno_symbols()
        print(f"F&O symbols count: {len(fno_symbols)}")
        
        # Keep running for a bit
        time.sleep(10)
        
        # Stop
        truedata_bridge.stop()
    else:
        print("❌ TrueData Bridge initialization failed")