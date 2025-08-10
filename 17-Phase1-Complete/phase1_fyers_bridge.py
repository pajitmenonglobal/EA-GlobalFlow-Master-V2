#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Fyers API Bridge
Complete integration with Fyers API for trading and data

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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum

# Add Fyers API if available
try:
    from fyers_api import fyersModel
    FYERS_AVAILABLE = True
except ImportError:
    FYERS_AVAILABLE = False
    print("Warning: Fyers API not available. Install with: pip install fyers-api")

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class ProductType(Enum):
    INTRADAY = "INTRADAY"
    MARGIN = "MARGIN" 
    CNC = "CNC"

@dataclass
class OrderResponse:
    success: bool
    order_id: str
    message: str
    data: Dict[str, Any]

@dataclass
class MarketData:
    symbol: str
    ltp: float
    open: float
    high: float
    low: float
    close: float
    volume: int
    timestamp: datetime

class FyersBridge:
    """
    Fyers API Bridge for EA GlobalFlow Pro v0.1
    Handles authentication, trading, and market data
    """
    
    def __init__(self, config_manager=None, totp_login=None, error_handler=None):
        """Initialize Fyers bridge"""
        self.config_manager = config_manager
        self.totp_login = totp_login
        self.error_handler = error_handler
        self.logger = logging.getLogger('FyersBridge')
        
        # Configuration
        self.app_id = ""
        self.secret_key = ""
        self.redirect_url = ""
        self.client_id = ""
        self.access_token = ""
        
        # Fyers model
        self.fyers = None
        self.is_authenticated = False
        self.is_initialized = False
        self.is_connected = False
        
        # Rate limiting
        self.last_request_time = 0
        self.requests_per_second = 10
        self.min_request_interval = 1.0 / self.requests_per_second
        
        # Market data
        self.market_data_cache = {}
        self.last_data_update = {}
        
        # Connection monitoring
        self.connection_thread = None
        self.is_monitoring = False
        self.last_heartbeat = datetime.now()
        
        # Order tracking
        self.pending_orders = {}
        self.order_history = []
        
    def initialize(self) -> bool:
        """
        Initialize Fyers bridge
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing Fyers Bridge v0.1...")
            
            # Check if Fyers API is available
            if not FYERS_AVAILABLE:
                self.logger.error("Fyers API not available - please install fyers-api package")
                return False
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Authenticate
            if not self._authenticate():
                return False
            
            # Start connection monitoring
            self._start_connection_monitoring()
            
            self.is_initialized = True
            self.logger.info("✅ Fyers Bridge initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Fyers config: {e}")
            return False
    
    def _authenticate(self) -> bool:
        """Authenticate with Fyers API"""
        try:
            self.logger.info("Authenticating with Fyers API...")
            
            # Create Fyers session
            session = fyersModel.SessionModel(
                client_id=self.client_id,
                secret_key=self.secret_key,
                redirect_uri=self.redirect_url,
                response_type="code",
                grant_type="authorization_code"
            )
            
            # Generate auth URL
            auth_url = session.generate_authcode()
            self.logger.info(f"Auth URL generated: {auth_url}")
            
            # Use TOTP auto-login if available
            auth_code = None
            if self.totp_login:
                auth_code = self.totp_login.get_fyers_auth_code(auth_url)
            
            if not auth_code:
                self.logger.error("Failed to get auth code - manual intervention required")
                return False
            
            # Generate access token
            session.set_token(auth_code)
            response = session.generate_token()
            
            if response['s'] == 'ok':
                self.access_token = response['access_token']
                
                # Create Fyers model
                self.fyers = fyersModel.FyersModel(
                    client_id=self.client_id,
                    token=self.access_token
                )
                
                self.is_authenticated = True
                self.is_connected = True
                self.logger.info("✅ Fyers authentication successful")
                return True
            else:
                self.logger.error(f"Fyers authentication failed: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Fyers authentication error: {e}")
            return False
    
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
                # Check connection health
                if self.is_authenticated and self.fyers:
                    profile = self.get_profile()
                    if profile:
                        self.is_connected = True
                        self.last_heartbeat = datetime.now()
                    else:
                        self.is_connected = False
                        self.logger.warning("Fyers connection lost")
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Connection monitoring error: {e}")
                time.sleep(60)
    
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
    
    def place_order(self, symbol: str, side: OrderSide, quantity: int, 
                   order_type: OrderType = OrderType.MARKET, 
                   price: float = 0.0, trigger_price: float = 0.0,
                   product_type: ProductType = ProductType.INTRADAY,
                   stop_loss: float = 0.0, take_profit: float = 0.0) -> OrderResponse:
        """
        Place order through Fyers API
        
        Args:
            symbol: Trading symbol (e.g., 'NSE:NIFTY23DECFUT')
            side: BUY or SELL
            quantity: Number of shares/contracts
            order_type: Order type (MARKET, LIMIT, etc.)
            price: Limit price (for limit orders)
            trigger_price: Trigger price (for stop orders)
            product_type: Product type (INTRADAY, CNC, etc.)
            stop_loss: Stop loss price
            take_profit: Take profit price
            
        Returns:
            OrderResponse object
        """
        try:
            if not self.is_authenticated or not self.fyers:
                return OrderResponse(False, "", "Not authenticated", {})
            
            # Rate limiting
            self._rate_limit_check()
            
            # Prepare order data
            order_data = {
                "symbol": symbol,
                "qty": quantity,
                "type": 2 if order_type == OrderType.MARKET else 1,  # 1=Limit, 2=Market
                "side": 1 if side == OrderSide.BUY else -1,  # 1=Buy, -1=Sell
                "productType": self._get_product_type_code(product_type),
                "limitPrice": price if order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] else 0,
                "stopPrice": trigger_price if order_type in [OrderType.STOP, OrderType.STOP_LIMIT] else 0,
                "validity": "DAY",
                "discqty": 0,
                "offlineOrder": "False"
            }
            
            self.logger.info(f"Placing order: {symbol} {side.value} {quantity} @ {price}")
            
            # Place order
            response = self.fyers.place_order(order_data)
            
            if response['s'] == 'ok':
                order_id = response['id']
                self.pending_orders[order_id] = {
                    'symbol': symbol,
                    'side': side.value,
                    'quantity': quantity,
                    'price': price,
                    'timestamp': datetime.now()
                }
                
                # Place stop loss and take profit orders if specified
                if stop_loss > 0:
                    self._place_stop_loss_order(symbol, side, quantity, stop_loss, order_id)
                
                if take_profit > 0:
                    self._place_take_profit_order(symbol, side, quantity, take_profit, order_id)
                
                self.logger.info(f"✅ Order placed successfully: {order_id}")
                return OrderResponse(True, order_id, "Order placed successfully", response)
            else:
                error_msg = response.get('message', 'Unknown error')
                self.logger.error(f"❌ Order placement failed: {error_msg}")
                return OrderResponse(False, "", error_msg, response)
                
        except Exception as e:
            self.logger.error(f"Order placement error: {e}")
            if self.error_handler:
                self.error_handler.handle_error("fyers_place_order", e)
            return OrderResponse(False, "", str(e), {})
    
    def _get_product_type_code(self, product_type: ProductType) -> str:
        """Convert product type to Fyers code"""
        mapping = {
            ProductType.INTRADAY: "INTRADAY",
            ProductType.MARGIN: "MARGIN",
            ProductType.CNC: "CNC"
        }
        return mapping.get(product_type, "INTRADAY")
    
    def _place_stop_loss_order(self, symbol: str, original_side: OrderSide, 
                              quantity: int, stop_price: float, parent_order_id: str):
        """Place stop loss order"""
        try:
            # Opposite side for stop loss
            sl_side = OrderSide.SELL if original_side == OrderSide.BUY else OrderSide.BUY
            
            sl_response = self.place_order(
                symbol=symbol,
                side=sl_side,
                quantity=quantity,
                order_type=OrderType.STOP,
                trigger_price=stop_price,
                product_type=ProductType.INTRADAY
            )
            
            if sl_response.success:
                self.logger.info(f"Stop loss order placed: {sl_response.order_id}")
            else:
                self.logger.error(f"Failed to place stop loss: {sl_response.message}")
                
        except Exception as e:
            self.logger.error(f"Stop loss order error: {e}")
    
    def _place_take_profit_order(self, symbol: str, original_side: OrderSide,
                                quantity: int, tp_price: float, parent_order_id: str):
        """Place take profit order"""
        try:
            # Opposite side for take profit
            tp_side = OrderSide.SELL if original_side == OrderSide.BUY else OrderSide.BUY
            
            tp_response = self.place_order(
                symbol=symbol,
                side=tp_side,
                quantity=quantity,
                order_type=OrderType.LIMIT,
                price=tp_price,
                product_type=ProductType.INTRADAY
            )
            
            if tp_response.success:
                self.logger.info(f"Take profit order placed: {tp_response.order_id}")
            else:
                self.logger.error(f"Failed to place take profit: {tp_response.message}")
                
        except Exception as e:
            self.logger.error(f"Take profit order error: {e}")
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order by ID"""
        try:
            if not self.is_authenticated or not self.fyers:
                return False
            
            self._rate_limit_check()
            
            cancel_data = {"id": order_id}
            response = self.fyers.cancel_order(cancel_data)
            
            if response['s'] == 'ok':
                self.logger.info(f"✅ Order cancelled: {order_id}")
                if order_id in self.pending_orders:
                    del self.pending_orders[order_id]
                return True
            else:
                self.logger.error(f"❌ Order cancellation failed: {response}")
                return False
                
        except Exception as e:
            self.logger.error(f"Order cancellation error: {e}")
            return False
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get current positions"""
        try:
            if not self.is_authenticated or not self.fyers:
                return []
            
            self._rate_limit_check()
            
            response = self.fyers.positions()
            
            if response['s'] == 'ok':
                positions = response.get('netPositions', [])
                self.logger.debug(f"Retrieved {len(positions)} positions")
                return positions
            else:
                self.logger.error(f"Failed to get positions: {response}")
                return []
                
        except Exception as e:
            self.logger.error(f"Get positions error: {e}")
            return []
    
    def get_orders(self) -> List[Dict[str, Any]]:
        """Get order history"""
        try:
            if not self.is_authenticated or not self.fyers:
                return []
            
            self._rate_limit_check()
            
            response = self.fyers.orderbook()
            
            if response['s'] == 'ok':
                orders = response.get('orderBook', [])
                self.logger.debug(f"Retrieved {len(orders)} orders")
                return orders
            else:
                self.logger.error(f"Failed to get orders: {response}")
                return []
                
        except Exception as e:
            self.logger.error(f"Get orders error: {e}")
            return []
    
    def get_profile(self) -> Optional[Dict[str, Any]]:
        """Get user profile"""
        try:
            if not self.is_authenticated or not self.fyers:
                return None
            
            self._rate_limit_check()
            
            response = self.fyers.get_profile()
            
            if response['s'] == 'ok':
                return response['data']
            else:
                self.logger.error(f"Failed to get profile: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Get profile error: {e}")
            return None
    
    def get_funds(self) -> Optional[Dict[str, Any]]:
        """Get account funds information"""
        try:
            if not self.is_authenticated or not self.fyers:
                return None
            
            self._rate_limit_check()
            
            response = self.fyers.funds()
            
            if response['s'] == 'ok':
                return response['fund_limit']
            else:
                self.logger.error(f"Failed to get funds: {response}")
                return None
                
        except Exception as e:
            self.logger.error(f"Get funds error: {e}")
            return None
    
    def get_market_data(self, symbols: List[str]) -> Dict[str, MarketData]:
        """Get market data for symbols"""
        try:
            if not self.is_authenticated or not self.fyers:
                return {}
            
            self._rate_limit_check()
            
            # Format symbols for Fyers API
            symbol_string = ",".join(symbols)
            data = {"symbols": symbol_string}
            
            response = self.fyers.depth(data)
            
            market_data = {}
            if response['s'] == 'ok':
                for symbol_data in response['d'].values():
                    symbol = symbol_data['symbol']
                    
                    market_data[symbol] = MarketData(
                        symbol=symbol,
                        ltp=symbol_data.get('ltp', 0.0),
                        open=symbol_data.get('open_price', 0.0),
                        high=symbol_data.get('high_price', 0.0),
                        low=symbol_data.get('low_price', 0.0),
                        close=symbol_data.get('prev_close_price', 0.0),
                        volume=symbol_data.get('volume', 0),
                        timestamp=datetime.now()
                    )
                
                # Update cache
                self.market_data_cache.update(market_data)
                
            return market_data
            
        except Exception as e:
            self.logger.error(f"Get market data error: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, resolution: str, 
                           date_from: datetime, date_to: datetime) -> pd.DataFrame:
        """Get historical data"""
        try:
            if not self.is_authenticated or not self.fyers:
                return pd.DataFrame()
            
            self._rate_limit_check()
            
            data = {
                "symbol": symbol,
                "resolution": resolution,
                "date_format": "1",
                "range_from": date_from.strftime("%Y-%m-%d"),
                "range_to": date_to.strftime("%Y-%m-%d"),
                "cont_flag": "1"
            }
            
            response = self.fyers.history(data)
            
            if response['s'] == 'ok':
                candles = response['candles']
                df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                df.set_index('timestamp', inplace=True)
                return df
            else:
                self.logger.error(f"Failed to get historical data: {response}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Get historical data error: {e}")
            return pd.DataFrame()
    
    def get_option_chain(self, symbol: str, expiry: str) -> Dict[str, Any]:
        """Get option chain data"""
        try:
            if not self.is_authenticated or not self.fyers:
                return {}
            
            self._rate_limit_check()
            
            data = {
                "symbol": symbol,
                "expiry": expiry
            }
            
            response = self.fyers.optionchain(data)
            
            if response['s'] == 'ok':
                return response['data']
            else:
                self.logger.error(f"Failed to get option chain: {response}")
                return {}
                
        except Exception as e:
            self.logger.error(f"Get option chain error: {e}")
            return {}
    
    def maintain_connection(self):
        """Maintain API connection"""
        try:
            if not self.is_connected:
                self.logger.info("Attempting to reconnect to Fyers...")
                if self._authenticate():
                    self.logger.info("✅ Reconnected to Fyers successfully")
                else:
                    self.logger.error("❌ Failed to reconnect to Fyers")
            
        except Exception as e:
            self.logger.error(f"Connection maintenance error: {e}")
    
    def is_healthy(self) -> bool:
        """Check if Fyers bridge is healthy"""
        try:
            time_since_heartbeat = datetime.now() - self.last_heartbeat
            return (
                self.is_initialized and
                self.is_authenticated and
                self.is_connected and
                time_since_heartbeat.total_seconds() < 300  # 5 minutes
            )
        except:
            return False
    
    def stop(self):
        """Stop Fyers bridge"""
        try:
            self.is_monitoring = False
            self.is_connected = False
            self.logger.info("Fyers Bridge stopped")
        except Exception as e:
            self.logger.error(f"Error stopping Fyers bridge: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create Fyers bridge
    fyers_bridge = FyersBridge()
    
    # Initialize
    if fyers_bridge.initialize():
        print("✅ Fyers Bridge initialized successfully")
        
        # Test market data
        market_data = fyers_bridge.get_market_data(["NSE:NIFTY50-INDEX"])
        print(f"Market data: {market_data}")
        
        # Test profile
        profile = fyers_bridge.get_profile()
        print(f"Profile: {profile}")
        
        # Test funds
        funds = fyers_bridge.get_funds()
        print(f"Funds: {funds}")
        
        # Stop
        fyers_bridge.stop()
    else:
        print("❌ Fyers Bridge initialization failed")(f"Fyers Bridge initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("fyers_init", e)
            return False
    
    def _load_config(self) -> bool:
        """Load Fyers configuration"""
        try:
            if self.config_manager:
                fyers_config = self.config_manager.get_config('api_integrations', {}).get('fyers', {})
                api_credentials = self.config_manager.get_config('api_credentials', {}).get('fyers', {})
            else:
                # Load from file
                config_file = os.path.join(os.path.dirname(__file__), 'Config', 'api_credentials_v01.json')
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                        api_credentials = config.get('fyers', {})
                        fyers_config = {'enabled': True}
                else:
                    self.logger.error("API credentials file not found")
                    return False
            
            # Extract credentials
            self.app_id = api_credentials.get('app_id', '')
            self.secret_key = api_credentials.get('secret_key', '')
            self.redirect_url = api_credentials.get('redirect_url', 'http://127.0.0.1:8000/')
            self.client_id = api_credentials.get('client_id', '')
            
            # Validate credentials
            if not all([self.app_id, self.secret_key, self.client_id]):
                self.logger.error("Missing Fyers API credentials")
                return False
            
            # Rate limiting config
            rate_limits = fyers_config.get('rate_limits', {})
            self.requests_per_second = rate_limits.get('orders_per_second', 10)
            self.min_request_interval = 1.0 / self.requests_per_second
            
            self.logger.info("Fyers configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error