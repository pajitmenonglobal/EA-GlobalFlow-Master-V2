#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Main Risk Manager
Comprehensive risk management system with VIX-based position sizing

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
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from dataclasses import dataclass
from enum import Enum

class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

@dataclass
class Position:
    symbol: str
    side: str  # BUY/SELL
    size: float
    entry_price: float
    current_price: float
    stop_loss: float
    take_profit: float
    unrealized_pnl: float
    timestamp: datetime

@dataclass
class RiskMetrics:
    account_balance: float
    equity: float
    margin_used: float
    free_margin: float
    margin_level: float
    daily_pnl: float
    unrealized_pnl: float
    drawdown_current: float
    drawdown_max: float
    portfolio_heat: float
    vix_value: float
    correlation_risk: float

class GlobalFlowRisk:
    """
    Main Risk Manager for EA GlobalFlow Pro v0.1
    Handles position sizing, correlation monitoring, emergency protocols
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize risk manager"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger('GlobalFlowRisk')
        
        # Risk configuration
        self.risk_config = {}
        self.is_initialized = False
        self.is_running = False
        
        # Account information
        self.account_balance = 0.0
        self.account_equity = 0.0
        self.account_currency = "INR"
        
        # Position tracking
        self.open_positions = {}  # symbol -> Position
        self.position_correlations = {}
        self.position_lock = threading.Lock()
        
        # Risk metrics
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        self.daily_pnl = 0.0
        self.portfolio_heat = 0.0
        self.vix_value = 15.0  # Default VIX
        
        # Risk limits
        self.max_drawdown_limit = 10.0  # 10%
        self.daily_loss_limit = 5.0     # 5%
        self.max_portfolio_heat = 20.0  # 20%
        self.max_correlation = 0.7      # 70%
        self.max_risk_per_trade = 2.0   # 2%
        
        # Emergency protocols
        self.emergency_stop = False
        self.risk_level = RiskLevel.LOW
        
        # VIX-based multipliers
        self.vix_multipliers = {
            (0, 15): 1.0,      # Low volatility
            (15, 25): 0.8,     # Medium volatility
            (25, 35): 0.6,     # High volatility
            (35, 50): 0.4,     # Very high volatility
            (50, 100): 0.2     # Extreme volatility
        }
        
        # Monitoring thread
        self.monitor_thread = None
        self.last_update = datetime.now()
        
    def initialize(self) -> bool:
        """
        Initialize risk management system
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing GlobalFlow Risk Manager v0.1...")
            
            # Load risk configuration
            if not self._load_risk_config():
                return False
            
            # Initialize account information
            if not self._initialize_account_info():
                return False
            
            # Start risk monitoring
            self._start_risk_monitoring()
            
            self.is_initialized = True
            self.is_running = True
            self.logger.info("✅ Risk Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Risk Manager initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("risk_init", e)
            return False
    
    def _load_risk_config(self) -> bool:
        """Load risk management configuration"""
        try:
            if self.config_manager:
                self.risk_config = self.config_manager.get_config('risk_management', {})
            else:
                # Default configuration
                self.risk_config = {
                    'max_drawdown': 10.0,
                    'daily_loss_limit': 5.0,
                    'max_portfolio_heat': 20.0,
                    'max_correlation': 0.7,
                    'max_risk_per_trade': 2.0,
                    'vix_based_sizing': True,
                    'emergency_protocols': {
                        'enabled': True,
                        'auto_close_on_limit': True
                    }
                }
            
            # Update limits from config
            self.max_drawdown_limit = self.risk_config.get('max_drawdown', 10.0)
            self.daily_loss_limit = self.risk_config.get('daily_loss_limit', 5.0)
            self.max_portfolio_heat = self.risk_config.get('max_portfolio_heat', 20.0)
            self.max_correlation = self.risk_config.get('max_correlation', 0.7)
            self.max_risk_per_trade = self.risk_config.get('max_risk_per_trade', 2.0)
            
            self.logger.info("Risk configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load risk config: {e}")
            return False
    
    def _initialize_account_info(self) -> bool:
        """Initialize account information"""
        try:
            # This would typically connect to broker API to get account info
            # For now, using default values
            self.account_balance = 100000.0  # Default 1 Lakh INR
            self.account_equity = self.account_balance
            self.account_currency = "INR"
            
            self.logger.info(f"Account initialized: Balance={self.account_balance} {self.account_currency}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize account info: {e}")
            return False
    
    def _start_risk_monitoring(self):
        """Start risk monitoring thread"""
        try:
            self.monitor_thread = threading.Thread(target=self._risk_monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Risk monitoring thread started")
            
        except Exception as e:
            self.logger.error(f"Failed to start risk monitoring: {e}")
    
    def _risk_monitor_loop(self):
        """Main risk monitoring loop"""
        while self.is_running:
            try:
                # Update risk metrics
                self._update_risk_metrics()
                
                # Check risk limits
                self._check_risk_limits()
                
                # Monitor correlations
                self._monitor_correlations()
                
                # Update risk level
                self._update_risk_level()
                
                self.last_update = datetime.now()
                time.sleep(10)  # Update every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Risk monitoring error: {e}")
                if self.error_handler:
                    self.error_handler.handle_error("risk_monitor", e)
                time.sleep(30)  # Longer sleep on error
    
    def _update_risk_metrics(self):
        """Update current risk metrics"""
        try:
            with self.position_lock:
                # Calculate unrealized PnL
                total_unrealized = 0.0
                for position in self.open_positions.values():
                    if position.side == "BUY":
                        position.unrealized_pnl = (position.current_price - position.entry_price) * position.size
                    else:
                        position.unrealized_pnl = (position.entry_price - position.current_price) * position.size
                    total_unrealized += position.unrealized_pnl
                
                # Update equity
                self.account_equity = self.account_balance + total_unrealized
                
                # Calculate current drawdown
                if self.account_equity < self.account_balance:
                    self.current_drawdown = ((self.account_balance - self.account_equity) / self.account_balance) * 100
                else:
                    self.current_drawdown = 0.0
                
                # Update max drawdown
                self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
                
                # Calculate portfolio heat
                total_risk = sum(self._calculate_position_risk(pos) for pos in self.open_positions.values())
                self.portfolio_heat = (total_risk / self.account_balance) * 100
                
        except Exception as e:
            self.logger.error(f"Error updating risk metrics: {e}")
    
    def _calculate_position_risk(self, position: Position) -> float:
        """Calculate risk amount for a position"""
        try:
            if position.side == "BUY":
                risk = (position.entry_price - position.stop_loss) * position.size
            else:
                risk = (position.stop_loss - position.entry_price) * position.size
            return abs(risk)
        except:
            return 0.0
    
    def _check_risk_limits(self):
        """Check if any risk limits are breached"""
        try:
            # Check drawdown limit
            if self.current_drawdown > self.max_drawdown_limit:
                self._trigger_emergency_protocol("MAX_DRAWDOWN_EXCEEDED")
            
            # Check daily loss limit
            if self.daily_pnl < -(self.daily_loss_limit * self.account_balance / 100):
                self._trigger_emergency_protocol("DAILY_LOSS_LIMIT_EXCEEDED")
            
            # Check portfolio heat
            if self.portfolio_heat > self.max_portfolio_heat:
                self._trigger_emergency_protocol("PORTFOLIO_HEAT_EXCEEDED")
                
        except Exception as e:
            self.logger.error(f"Error checking risk limits: {e}")
    
    def _monitor_correlations(self):
        """Monitor position correlations"""
        try:
            if len(self.open_positions) < 2:
                return
            
            # Calculate correlations between positions
            symbols = list(self.open_positions.keys())
            high_correlation_pairs = []
            
            for i in range(len(symbols)):
                for j in range(i + 1, len(symbols)):
                    correlation = self._calculate_correlation(symbols[i], symbols[j])
                    if abs(correlation) > self.max_correlation:
                        high_correlation_pairs.append((symbols[i], symbols[j], correlation))
            
            if high_correlation_pairs:
                self.logger.warning(f"High correlation detected: {high_correlation_pairs}")
                
        except Exception as e:
            self.logger.error(f"Error monitoring correlations: {e}")
    
    def _calculate_correlation(self, symbol1: str, symbol2: str) -> float:
        """Calculate correlation between two symbols"""
        # This would typically use historical price data
        # For now, return a placeholder value
        return 0.0
    
    def _update_risk_level(self):
        """Update overall risk level"""
        try:
            if self.emergency_stop:
                self.risk_level = RiskLevel.CRITICAL
            elif self.current_drawdown > 7.0 or self.portfolio_heat > 15.0:
                self.risk_level = RiskLevel.HIGH
            elif self.current_drawdown > 3.0 or self.portfolio_heat > 10.0:
                self.risk_level = RiskLevel.MEDIUM
            else:
                self.risk_level = RiskLevel.LOW
                
        except Exception as e:
            self.logger.error(f"Error updating risk level: {e}")
    
    def _trigger_emergency_protocol(self, reason: str):
        """Trigger emergency risk protocol"""
        try:
            self.logger.critical(f"🚨 EMERGENCY PROTOCOL TRIGGERED: {reason}")
            self.emergency_stop = True
            self.risk_level = RiskLevel.CRITICAL
            
            # Close all positions if configured
            if self.risk_config.get('emergency_protocols', {}).get('auto_close_on_limit', False):
                self._close_all_positions(reason)
            
            # Send alerts
            self._send_emergency_alert(reason)
            
        except Exception as e:
            self.logger.error(f"Error triggering emergency protocol: {e}")
    
    def _close_all_positions(self, reason: str):
        """Close all open positions"""
        try:
            self.logger.info(f"Closing all positions due to: {reason}")
            with self.position_lock:
                for symbol, position in self.open_positions.items():
                    self.logger.info(f"Closing position: {symbol}")
                    # This would send close orders to broker
                    
        except Exception as e:
            self.logger.error(f"Error closing positions: {e}")
    
    def _send_emergency_alert(self, reason: str):
        """Send emergency alert notifications"""
        try:
            alert_message = f"🚨 EMERGENCY: {reason}\n"
            alert_message += f"Account Equity: {self.account_equity:.2f}\n"
            alert_message += f"Current Drawdown: {self.current_drawdown:.2f}%\n"
            alert_message += f"Portfolio Heat: {self.portfolio_heat:.2f}%"
            
            # This would send alerts via configured channels
            self.logger.critical(alert_message)
            
        except Exception as e:
            self.logger.error(f"Error sending emergency alert: {e}")
    
    def calculate_position_size(self, symbol: str, entry_price: float, stop_loss: float, 
                              risk_percent: Optional[float] = None) -> float:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price for the trade
            stop_loss: Stop loss price
            risk_percent: Risk percentage (optional, uses default if None)
            
        Returns:
            Calculated position size
        """
        try:
            # Use provided risk percent or default
            if risk_percent is None:
                risk_percent = self.max_risk_per_trade
            
            # Check if we can take new position
            if self.emergency_stop:
                self.logger.warning("Cannot calculate position size - emergency stop active")
                return 0.0
            
            # Calculate base risk amount
            risk_amount = (risk_percent / 100) * self.account_equity
            
            # Apply VIX-based adjustment
            vix_multiplier = self._get_vix_multiplier()
            adjusted_risk = risk_amount * vix_multiplier
            
            # Calculate position size based on stop loss distance
            price_diff = abs(entry_price - stop_loss)
            if price_diff == 0:
                return 0.0
            
            position_size = adjusted_risk / price_diff
            
            # Apply additional risk checks
            position_size = self._apply_risk_checks(symbol, position_size, adjusted_risk)
            
            self.logger.info(f"Position size calculated for {symbol}: {position_size:.2f}")
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _get_vix_multiplier(self) -> float:
        """Get VIX-based position size multiplier"""
        try:
            if not self.risk_config.get('vix_based_sizing', True):
                return 1.0
            
            for (low, high), multiplier in self.vix_multipliers.items():
                if low <= self.vix_value < high:
                    return multiplier
            
            return 0.2  # Default for extreme volatility
            
        except Exception as e:
            self.logger.error(f"Error getting VIX multiplier: {e}")
            return 1.0
    
    def _apply_risk_checks(self, symbol: str, position_size: float, risk_amount: float) -> float:
        """Apply additional risk checks to position size"""
        try:
            # Check portfolio heat
            new_heat = self.portfolio_heat + ((risk_amount / self.account_balance) * 100)
            if new_heat > self.max_portfolio_heat:
                reduction_factor = self.max_portfolio_heat / new_heat
                position_size *= reduction_factor
                self.logger.warning(f"Position size reduced due to portfolio heat: {reduction_factor:.2f}")
            
            # Check correlation limits
            correlation_factor = self._check_correlation_impact(symbol)
            position_size *= correlation_factor
            
            # Ensure minimum viable size
            min_size = 1.0  # Minimum 1 unit
            if position_size < min_size:
                return 0.0
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error applying risk checks: {e}")
            return position_size
    
    def _check_correlation_impact(self, symbol: str) -> float:
        """Check correlation impact and return adjustment factor"""
        try:
            # Count highly correlated positions
            correlated_count = 0
            for existing_symbol in self.open_positions.keys():
                correlation = self._calculate_correlation(symbol, existing_symbol)
                if abs(correlation) > self.max_correlation:
                    correlated_count += 1
            
            # Reduce position size based on correlation
            if correlated_count == 0:
                return 1.0
            elif correlated_count <= 2:
                return 0.7
            else:
                return 0.5  # Significant reduction for high correlation
                
        except Exception as e:
            self.logger.error(f"Error checking correlation impact: {e}")
            return 1.0
    
    def add_position(self, symbol: str, side: str, size: float, entry_price: float, 
                    stop_loss: float, take_profit: float) -> bool:
        """
        Add new position to tracking
        
        Returns: True if position added successfully
        """
        try:
            with self.position_lock:
                position = Position(
                    symbol=symbol,
                    side=side,
                    size=size,
                    entry_price=entry_price,
                    current_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    unrealized_pnl=0.0,
                    timestamp=datetime.now()
                )
                
                self.open_positions[symbol] = position
                self.logger.info(f"Position added: {symbol} {side} {size} @ {entry_price}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error adding position: {e}")
            return False
    
    def remove_position(self, symbol: str) -> bool:
        """
        Remove position from tracking
        
        Returns: True if position removed successfully
        """
        try:
            with self.position_lock:
                if symbol in self.open_positions:
                    del self.open_positions[symbol]
                    self.logger.info(f"Position removed: {symbol}")
                    return True
                else:
                    self.logger.warning(f"Position not found for removal: {symbol}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error removing position: {e}")
            return False
    
    def update_position_price(self, symbol: str, current_price: float):
        """Update current price for a position"""
        try:
            with self.position_lock:
                if symbol in self.open_positions:
                    self.open_positions[symbol].current_price = current_price
                    
        except Exception as e:
            self.logger.error(f"Error updating position price: {e}")
    
    def can_open_position(self, symbol: str, risk_amount: float) -> Tuple[bool, str]:
        """
        Check if new position can be opened
        
        Returns: (can_open, reason)
        """
        try:
            # Check emergency stop
            if self.emergency_stop:
                return False, "Emergency stop active"
            
            # Check risk level
            if self.risk_level == RiskLevel.CRITICAL:
                return False, "Critical risk level"
            
            # Check portfolio heat
            new_heat = self.portfolio_heat + ((risk_amount / self.account_balance) * 100)
            if new_heat > self.max_portfolio_heat:
                return False, f"Portfolio heat would exceed limit: {new_heat:.2f}%"
            
            # Check drawdown
            if self.current_drawdown > (self.max_drawdown_limit * 0.8):
                return False, f"Approaching max drawdown: {self.current_drawdown:.2f}%"
            
            # Check correlation
            correlated_positions = 0
            for existing_symbol in self.open_positions.keys():
                correlation = self._calculate_correlation(symbol, existing_symbol)
                if abs(correlation) > self.max_correlation:
                    correlated_positions += 1
            
            if correlated_positions >= 3:
                return False, "Too many correlated positions"
            
            return True, "Position allowed"
            
        except Exception as e:
            self.logger.error(f"Error checking position permission: {e}")
            return False, "Error in risk check"
    
    def get_risk_metrics(self) -> RiskMetrics:
        """Get current risk metrics"""
        return RiskMetrics(
            account_balance=self.account_balance,
            equity=self.account_equity,
            margin_used=0.0,  # Would be calculated from broker
            free_margin=self.account_equity,
            margin_level=100.0,
            daily_pnl=self.daily_pnl,
            unrealized_pnl=sum(pos.unrealized_pnl for pos in self.open_positions.values()),
            drawdown_current=self.current_drawdown,
            drawdown_max=self.max_drawdown,
            portfolio_heat=self.portfolio_heat,
            vix_value=self.vix_value,
            correlation_risk=0.0
        )
    
    def update_vix(self, vix_value: float):
        """Update VIX value for position sizing"""
        self.vix_value = vix_value
        self.logger.debug(f"VIX updated: {vix_value}")
    
    def reset_emergency_stop(self) -> bool:
        """Reset emergency stop (manual intervention required)"""
        try:
            self.emergency_stop = False
            self.risk_level = RiskLevel.LOW
            self.logger.info("Emergency stop reset - trading can resume")
            return True
        except Exception as e:
            self.logger.error(f"Error resetting emergency stop: {e}")
            return False
    
    def is_healthy(self) -> bool:
        """Check if risk manager is healthy"""
        try:
            time_since_update = datetime.now() - self.last_update
            return (
                self.is_running and 
                self.is_initialized and 
                time_since_update.total_seconds() < 60 and
                not self.emergency_stop
            )
        except:
            return False
    
    def stop(self):
        """Stop risk manager"""
        try:
            self.is_running = False
            self.logger.info("Risk Manager stopped")
        except Exception as e:
            self.logger.error(f"Error stopping risk manager: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create risk manager
    risk_manager = GlobalFlowRisk()
    
    # Initialize
    if risk_manager.initialize():
        print("✅ Risk Manager initialized successfully")
        
        # Test position size calculation
        position_size = risk_manager.calculate_position_size("NIFTY", 18000, 17800)
        print(f"Calculated position size: {position_size}")
        
        # Test adding position
        risk_manager.add_position("NIFTY", "BUY", position_size, 18000, 17800, 18200)
        
        # Get risk metrics
        metrics = risk_manager.get_risk_metrics()
        print(f"Risk metrics: {metrics}")
        
        # Stop
        risk_manager.stop()
    else:
        print("❌ Risk Manager initialization failed")