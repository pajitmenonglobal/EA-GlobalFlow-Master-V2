#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Advanced Trade Exit Management System
Intelligent exit signal management with multiple exit strategies

Features:
- STR-EXIT based exit signals (ATR 1.5x, Period 20)
- Multiple exit strategies (Trailing, Fixed, Dynamic)
- Partial exit management
- Risk-based exit adjustments
- Time-based exit management
- Profit protection strategies
- Emergency exit protocols

Author: EA GlobalFlow Pro Team
Version: v0.1
Date: 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import talib

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ExitManager:
    """
    Advanced Trade Exit Management System
    
    Manages trade exits using multiple strategies and signals:
    - STR-EXIT SuperTrend signals
    - Technical indicator exit signals
    - Risk-based exits
    - Time-based exits
    - Profit protection exits
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize Exit Manager"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Exit Strategy Parameters
        self.exit_params = {
            'str_exit': {
                'atr_period': 20,        # ATR period for STR-EXIT
                'atr_multiplier': 1.5    # ATR multiplier for STR-EXIT
            },
            'trailing_stop': {
                'initial_distance': 0.02,  # 2% initial distance
                'step_size': 0.005,        # 0.5% step size
                'activation_profit': 0.01  # 1% profit to activate trailing
            },
            'time_exits': {
                'max_hold_time': 24,       # Maximum hours to hold position
                'profit_time_limit': 4,    # Hours to hold profitable position
                'loss_time_limit': 2       # Hours to hold losing position
            },
            'profit_protection': {
                'partial_exit_levels': [0.5, 1.0, 1.5],  # R multiples for partial exits
                'partial_exit_sizes': [25, 25, 25],       # Percentage to exit at each level
                'breakeven_move': 1.0      # R multiple to move stop to breakeven
            }
        }
        
        # Exit Signal Weights
        self.exit_signal_weights = {
            'str_exit': 0.35,           # STR-EXIT SuperTrend (primary)
            'technical_exit': 0.25,     # Technical indicator exits
            'risk_exit': 0.20,          # Risk-based exits
            'time_exit': 0.10,          # Time-based exits
            'profit_protection': 0.10   # Profit protection exits
        }
        
        # Exit Urgency Thresholds
        self.urgency_thresholds = {
            'immediate': 0.9,           # Immediate exit required
            'urgent': 0.7,              # Urgent exit recommended
            'moderate': 0.5,            # Moderate exit consideration
            'low': 0.3                  # Low priority exit
        }
        
        # Active trades tracking
        self.active_trades = {}
        self.exit_history = []
        
        # Performance tracking
        self.exit_performance = {
            'total_exits': 0,
            'profitable_exits': 0,
            'stopped_out_exits': 0,
            'time_exits': 0,
            'avg_hold_time': 0.0,
            'best_exit_return': 0.0,
            'worst_exit_return': 0.0
        }
        
        self.logger.info("🚪 Advanced Exit Manager initialized")
    
    def evaluate_exit_signals(self, price_data: pd.DataFrame, position_info: Dict[str, Any],
                            timeframe: str = "M15") -> Dict[str, Any]:
        """
        Evaluate all exit signals for a position
        
        Args:
            price_data: Current OHLC price data
            position_info: Position details (entry_price, size, direction, entry_time, etc.)
            timeframe: Current timeframe
            
        Returns:
            Comprehensive exit analysis with recommendations
        """
        try:
            if not position_info or 'entry_price' not in position_info:
                return self._empty_exit_result()
            
            # Calculate STR-EXIT signals
            str_exit_signals = self._calculate_str_exit_signals(price_data, position_info)
            
            # Calculate technical exit signals
            technical_exits = self._calculate_technical_exits(price_data, position_info)
            
            # Calculate risk-based exits
            risk_exits = self._calculate_risk_exits(price_data, position_info)
            
            # Calculate time-based exits
            time_exits = self._calculate_time_exits(position_info)
            
            # Calculate profit protection exits
            profit_protection = self._calculate_profit_protection_exits(price_data, position_info)
            
            # Combine all exit signals
            combined_exit_signal = self._combine_exit_signals(
                str_exit_signals, technical_exits, risk_exits, time_exits, profit_protection
            )
            
            # Generate exit recommendations
            exit_recommendations = self._generate_exit_recommendations(
                combined_exit_signal, position_info, str_exit_signals
            )
            
            # Calculate updated stop loss and take profit levels
            updated_levels = self._calculate_updated_levels(
                price_data, position_info, combined_exit_signal
            )
            
            result = {
                'exit_signals': {
                    'str_exit': str_exit_signals,
                    'technical_exits': technical_exits,
                    'risk_exits': risk_exits,
                    'time_exits': time_exits,
                    'profit_protection': profit_protection
                },
                'combined_signal': combined_exit_signal,
                'exit_recommendations': exit_recommendations,
                'updated_levels': updated_levels,
                'position_analysis': self._analyze_position_status(price_data, position_info),
                'exit_urgency': self._calculate_exit_urgency(combined_exit_signal),
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe
            }
            
            # Update active trade tracking
            self._update_trade_tracking(position_info, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Exit signal evaluation error: {e}")
            if self.error_handler:
                self.error_handler.log_error("EXIT_EVALUATION_ERROR", str(e))
            return self._empty_exit_result()
    
    def get_exit_recommendation(self, price_data: pd.DataFrame, position_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get specific exit recommendation for a position
        
        Returns:
            Direct exit recommendation with specific actions
        """
        try:
            exit_analysis = self.evaluate_exit_signals(price_data, position_info)
            
            if not exit_analysis or exit_analysis.get('exit_recommendations', {}).get('action') == 'ERROR':
                return {'action': 'HOLD', 'reason': 'Unable to analyze exit conditions', 'urgency': 'LOW'}
            
            recommendations = exit_analysis['exit_recommendations']
            combined_signal = exit_analysis['combined_signal']
            exit_urgency = exit_analysis['exit_urgency']
            
            # Determine specific action
            action = recommendations.get('primary_action', 'HOLD')
            reason = recommendations.get('primary_reason', 'No exit signal detected')
            
            # Add specific exit details
            exit_details = {
                'action': action,
                'reason': reason,
                'urgency': exit_urgency['urgency_level'],
                'exit_percentage': recommendations.get('exit_percentage', 0),
                'exit_price_target': recommendations.get('suggested_exit_price', 0.0),
                'stop_loss_update': recommendations.get('stop_loss_update', None),
                'take_profit_update': recommendations.get('take_profit_update', None),
                'trailing_stop_level': recommendations.get('trailing_stop_level', None),
                'time_remaining': self._calculate_time_remaining(position_info),
                'profit_loss_status': self._calculate_current_pl_status(price_data, position_info)
            }
            
            return exit_details
            
        except Exception as e:
            self.logger.error(f"Exit recommendation error: {e}")
            return {'action': 'ERROR', 'reason': 'Exit analysis failed', 'urgency': 'HIGH'}
    
    def manage_partial_exits(self, price_data: pd.DataFrame, position_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Manage partial exit strategy
        
        Returns:
            Partial exit recommendations with specific percentages
        """
        try:
            current_price = price_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            position_direction = position_info.get('direction', 'LONG').upper()
            
            # Calculate current R multiple (Risk multiple)
            if 'stop_loss' in position_info and position_info['stop_loss'] > 0:
                if position_direction == 'LONG':
                    risk_amount = entry_price - position_info['stop_loss']
                    current_profit = current_price - entry_price
                else:  # SHORT
                    risk_amount = position_info['stop_loss'] - entry_price
                    current_profit = entry_price - current_price
                
                r_multiple = current_profit / risk_amount if risk_amount > 0 else 0.0
            else:
                # Fallback calculation
                r_multiple = abs(current_price - entry_price) / entry_price * 10
            
            # Check partial exit levels
            partial_exits = []
            partial_exit_levels = self.exit_params['profit_protection']['partial_exit_levels']
            partial_exit_sizes = self.exit_params['profit_protection']['partial_exit_sizes']
            
            remaining_position = 100.0  # Start with 100% position
            
            for i, (r_level, exit_size) in enumerate(zip(partial_exit_levels, partial_exit_sizes)):
                if r_multiple >= r_level:
                    # Check if this level hasn't been executed yet
                    executed_exits = position_info.get('executed_partial_exits', [])
                    if r_level not in executed_exits:
                        partial_exits.append({
                            'r_multiple': r_level,
                            'exit_percentage': exit_size,
                            'trigger_price': self._calculate_partial_exit_price(entry_price, position_direction, r_level, position_info),
                            'reason': f'Partial exit at {r_level}R profit level',
                            'priority': 'HIGH' if r_multiple >= r_level * 1.1 else 'MEDIUM'
                        })
                        remaining_position -= exit_size
            
            # Breakeven stop adjustment
            breakeven_r = self.exit_params['profit_protection']['breakeven_move']
            breakeven_needed = r_multiple >= breakeven_r and not position_info.get('breakeven_moved', False)
            
            return {
                'partial_exits_available': partial_exits,
                'current_r_multiple': round(r_multiple, 2),
                'remaining_position_percent': remaining_position,
                'breakeven_stop_needed': breakeven_needed,
                'next_partial_exit_level': min([pe['r_multiple'] for pe in partial_exits], default=None),
                'total_partial_exits_pending': len(partial_exits)
            }
            
        except Exception as e:
            self.logger.error(f"Partial exit management error: {e}")
            return {'partial_exits_available': [], 'current_r_multiple': 0.0}
    
    def update_trailing_stop(self, price_data: pd.DataFrame, position_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update trailing stop levels
        
        Returns:
            Updated trailing stop information
        """
        try:
            current_price = price_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            position_direction = position_info.get('direction', 'LONG').upper()
            
            # Calculate current profit/loss
            if position_direction == 'LONG':
                current_pl = (current_price - entry_price) / entry_price
            else:  # SHORT
                current_pl = (entry_price - current_price) / entry_price
            
            # Check if trailing should be activated
            activation_profit = self.exit_params['trailing_stop']['activation_profit']
            
            if current_pl < activation_profit:
                return {
                    'trailing_active': False,
                    'activation_needed_profit': activation_profit - current_pl,
                    'current_trailing_stop': None,
                    'reason': 'Insufficient profit to activate trailing stop'
                }
            
            # Calculate trailing stop level
            current_trailing = position_info.get('current_trailing_stop', None)
            initial_distance = self.exit_params['trailing_stop']['initial_distance']
            step_size = self.exit_params['trailing_stop']['step_size']
            
            if position_direction == 'LONG':
                new_trailing_stop = current_price * (1 - initial_distance)
                
                # Only update if it's higher than current trailing stop
                if current_trailing is None or new_trailing_stop > current_trailing:
                    updated_trailing = new_trailing_stop
                    trailing_moved = True
                else:
                    updated_trailing = current_trailing
                    trailing_moved = False
            else:  # SHORT
                new_trailing_stop = current_price * (1 + initial_distance)
                
                # Only update if it's lower than current trailing stop
                if current_trailing is None or new_trailing_stop < current_trailing:
                    updated_trailing = new_trailing_stop
                    trailing_moved = True
                else:
                    updated_trailing = current_trailing
                    trailing_moved = False
            
            # Calculate distance from current price
            if position_direction == 'LONG':
                distance_percent = (current_price - updated_trailing) / current_price
            else:
                distance_percent = (updated_trailing - current_price) / current_price
            
            return {
                'trailing_active': True,
                'current_trailing_stop': round(updated_trailing, 5),
                'trailing_moved': trailing_moved,
                'distance_from_price_percent': round(distance_percent * 100, 2),
                'trailing_profit_locked': round(current_pl * 100, 2),
                'next_trailing_update_price': self._calculate_next_trailing_price(current_price, position_direction, step_size)
            }
            
        except Exception as e:
            self.logger.error(f"Trailing stop update error: {e}")
            return {'trailing_active': False, 'current_trailing_stop': None}
    
    def check_emergency_exits(self, price_data: pd.DataFrame, position_info: Dict[str, Any],
                            market_conditions: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Check for emergency exit conditions
        
        Returns:
            Emergency exit analysis and recommendations
        """
        try:
            emergency_conditions = []
            emergency_level = 'NONE'
            
            current_price = price_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            position_direction = position_info.get('direction', 'LONG').upper()
            
            # Calculate current loss
            if position_direction == 'LONG':
                current_loss = (entry_price - current_price) / entry_price
            else:
                current_loss = (current_price - entry_price) / entry_price
            
            # Excessive loss check
            max_acceptable_loss = 0.10  # 10% maximum loss
            if current_loss > max_acceptable_loss:
                emergency_conditions.append({
                    'type': 'EXCESSIVE_LOSS',
                    'severity': 'CRITICAL',
                    'message': f'Loss exceeds maximum acceptable level: {current_loss*100:.1f}%',
                    'immediate_action_required': True
                })
                emergency_level = 'CRITICAL'
            
            # Rapid adverse movement check
            if len(price_data) >= 5:
                recent_movement = abs(price_data['close'].iloc[-1] - price_data['close'].iloc[-5]) / price_data['close'].iloc[-5]
                if recent_movement > 0.05:  # 5% rapid movement
                    emergency_conditions.append({
                        'type': 'RAPID_ADVERSE_MOVEMENT',
                        'severity': 'HIGH',
                        'message': f'Rapid price movement detected: {recent_movement*100:.1f}%',
                        'immediate_action_required': True
                    })
                    if emergency_level == 'NONE':
                        emergency_level = 'HIGH'
            
            # Market condition emergencies
            if market_conditions:
                volatility = market_conditions.get('volatility', 0.0)
                if volatility > 0.08:  # 8% volatility
                    emergency_conditions.append({
                        'type': 'HIGH_VOLATILITY',
                        'severity': 'MEDIUM',
                        'message': f'Extreme volatility detected: {volatility*100:.1f}%',
                        'immediate_action_required': False
                    })
                    if emergency_level == 'NONE':
                        emergency_level = 'MEDIUM'
                
                # News impact check
                if market_conditions.get('high_impact_news', False):
                    emergency_conditions.append({
                        'type': 'HIGH_IMPACT_NEWS',
                        'severity': 'MEDIUM',
                        'message': 'High impact news event detected',
                        'immediate_action_required': False
                    })
                    if emergency_level == 'NONE':
                        emergency_level = 'MEDIUM'
            
            # Time-based emergency (position held too long)
            if 'entry_time' in position_info:
                entry_time = datetime.fromisoformat(position_info['entry_time']) if isinstance(position_info['entry_time'], str) else position_info['entry_time']
                hold_time_hours = (datetime.now() - entry_time).total_seconds() / 3600
                max_hold_time = self.exit_params['time_exits']['max_hold_time']
                
                if hold_time_hours > max_hold_time:
                    emergency_conditions.append({
                        'type': 'MAXIMUM_HOLD_TIME_EXCEEDED',
                        'severity': 'MEDIUM',
                        'message': f'Position held for {hold_time_hours:.1f} hours (max: {max_hold_time})',
                        'immediate_action_required': True
                    })
                    if emergency_level == 'NONE':
                        emergency_level = 'MEDIUM'
            
            # Generate emergency action
            if emergency_conditions:
                if emergency_level == 'CRITICAL':
                    emergency_action = 'IMMEDIATE_FULL_EXIT'
                elif emergency_level == 'HIGH':
                    emergency_action = 'URGENT_PARTIAL_EXIT'
                else:
                    emergency_action = 'REVIEW_POSITION'
            else:
                emergency_action = 'NO_ACTION'
            
            return {
                'emergency_detected': len(emergency_conditions) > 0,
                'emergency_level': emergency_level,
                'emergency_conditions': emergency_conditions,
                'emergency_action': emergency_action,
                'conditions_count': len(emergency_conditions),
                'highest_severity': max([ec['severity'] for ec in emergency_conditions], default='NONE')
            }
            
        except Exception as e:
            self.logger.error(f"Emergency exit check error: {e}")
            return {'emergency_detected': False, 'emergency_level': 'ERROR'}
    
    # Private Methods
    
    def _calculate_str_exit_signals(self, price_data: pd.DataFrame, position_info: Dict) -> Dict[str, Any]:
        """Calculate STR-EXIT SuperTrend signals"""
        try:
            if len(price_data) < self.exit_params['str_exit']['atr_period'] + 10:
                return {'exit_signal': False, 'reason': 'Insufficient data'}
            
            # Calculate ATR
            atr = talib.ATR(
                price_data['high'].values,
                price_data['low'].values,
                price_data['close'].values,
                timeperiod=self.exit_params['str_exit']['atr_period']
            )
            
            # Calculate SuperTrend for exits
            hl2 = (price_data['high'] + price_data['low']) / 2
            multiplier = self.exit_params['str_exit']['atr_multiplier']
            
            upper_band = hl2 + (multiplier * atr)
            lower_band = hl2 - (multiplier * atr)
            
            # SuperTrend calculation (simplified)
            close_prices = price_data['close'].values
            current_price = close_prices[-1]
            
            # Determine exit signal based on position direction
            position_direction = position_info.get('direction', 'LONG').upper()
            
            if position_direction == 'LONG':
                # For long positions, exit when price breaks below lower band
                exit_level = lower_band.iloc[-1]
                exit_signal = current_price < exit_level
                signal_strength = max(0.0, (exit_level - current_price) / current_price) if exit_signal else 0.0
            else:  # SHORT
                # For short positions, exit when price breaks above upper band
                exit_level = upper_band.iloc[-1]
                exit_signal = current_price > exit_level
                signal_strength = max(0.0, (current_price - exit_level) / current_price) if exit_signal else 0.0
            
            # Calculate signal confidence
            recent_atr = atr[-5:].mean()
            signal_confidence = min(1.0, signal_strength * 2) if exit_signal else 0.0
            
            return {
                'exit_signal': exit_signal,
                'exit_level': round(exit_level, 5),
                'signal_strength': round(signal_strength, 4),
                'signal_confidence': round(signal_confidence, 4),
                'atr_value': round(recent_atr, 5),
                'distance_to_exit': abs(current_price - exit_level),
                'reason': f'STR-EXIT SuperTrend signal (ATR {multiplier}x)'
            }
            
        except Exception as e:
            self.logger.error(f"STR-EXIT calculation error: {e}")
            return {'exit_signal': False, 'reason': 'STR-EXIT calculation failed'}
    
    def _calculate_technical_exits(self, price_data: pd.DataFrame, position_info: Dict) -> Dict[str, Any]:
        """Calculate technical indicator exit signals"""
        try:
            technical_signals = []
            combined_strength = 0.0
            
            if len(price_data) < 20:
                return {'exit_signals': [], 'combined_strength': 0.0}
            
            current_price = price_data['close'].iloc[-1]
            position_direction = position_info.get('direction', 'LONG').upper()
            
            # RSI overbought/oversold exits
            rsi = talib.RSI(price_data['close'].values, timeperiod=14)
            current_rsi = rsi[-1]
            
            if position_direction == 'LONG' and current_rsi > 80:
                technical_signals.append({
                    'type': 'RSI_OVERBOUGHT',
                    'strength': (current_rsi - 80) / 20,
                    'reason': f'RSI overbought at {current_rsi:.1f}'
                })
                combined_strength += 0.3
            elif position_direction == 'SHORT' and current_rsi < 20:
                technical_signals.append({
                    'type': 'RSI_OVERSOLD',
                    'strength': (20 - current_rsi) / 20,
                    'reason': f'RSI oversold at {current_rsi:.1f}'
                })
                combined_strength += 0.3
            
            # Moving average exits
            sma_20 = talib.SMA(price_data['close'].values, timeperiod=20)
            sma_50 = talib.SMA(price_data['close'].values, timeperiod=50)
            
            if len(sma_20) > 1 and len(sma_50) > 1:
                if position_direction == 'LONG':
                    # Exit long if price breaks below key MAs
                    if current_price < sma_20[-1] and sma_20[-1] < sma_20[-2]:
                        technical_signals.append({
                            'type': 'MA_BREAKDOWN',
                            'strength': 0.6,
                            'reason': 'Price broke below declining 20 SMA'
                        })
                        combined_strength += 0.4
                else:  # SHORT
                    # Exit short if price breaks above key MAs
                    if current_price > sma_20[-1] and sma_20[-1] > sma_20[-2]:
                        technical_signals.append({
                            'type': 'MA_BREAKUP',
                            'strength': 0.6,
                            'reason': 'Price broke above rising 20 SMA'
                        })
                        combined_strength += 0.4
            
            # MACD exit signals
            macd, macd_signal, macd_histogram = talib.MACD(price_data['close'].values)
            
            if len(macd) > 1 and len(macd_signal) > 1:
                # MACD bearish divergence for longs
                if position_direction == 'LONG' and macd[-1] < macd_signal[-1] and macd[-2] >= macd_signal[-2]:
                    technical_signals.append({
                        'type': 'MACD_BEARISH_CROSS',
                        'strength': 0.5,
                        'reason': 'MACD bearish crossover'
                    })
                    combined_strength += 0.3
                # MACD bullish divergence for shorts
                elif position_direction == 'SHORT' and macd[-1] > macd_signal[-1] and macd[-2] <= macd_signal[-2]:
                    technical_signals.append({
                        'type': 'MACD_BULLISH_CROSS',
                        'strength': 0.5,
                        'reason': 'MACD bullish crossover'
                    })
                    combined_strength += 0.3
            
            # Normalize combined strength
            combined_strength = min(1.0, combined_strength)
            
            return {
                'exit_signals': technical_signals,
                'combined_strength': round(combined_strength, 4),
                'signal_count': len(technical_signals),
                'strongest_signal': max(technical_signals, key=lambda x: x['strength']) if technical_signals else None
            }
            
        except Exception as e:
            self.logger.error(f"Technical exits calculation error: {e}")
            return {'exit_signals': [], 'combined_strength': 0.0}
    
    def _calculate_risk_exits(self, price_data: pd.DataFrame, position_info: Dict) -> Dict[str, Any]:
        """Calculate risk-based exit signals"""
        try:
            risk_signals = []
            risk_level = 0.0
            
            current_price = price_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            position_direction = position_info.get('direction', 'LONG').upper()
            
            # Calculate current P&L
            if position_direction == 'LONG':
                current_pl_percent = (current_price - entry_price) / entry_price
            else:
                current_pl_percent = (entry_price - current_price) / entry_price
            
            # Stop loss hit check
            if 'stop_loss' in position_info and position_info['stop_loss'] > 0:
                stop_loss = position_info['stop_loss']
                if (position_direction == 'LONG' and current_price <= stop_loss) or \
                   (position_direction == 'SHORT' and current_price >= stop_loss):
                    risk_signals.append({
                        'type': 'STOP_LOSS_HIT', 
                        'urgency': 'CRITICAL',
                        'reason': 'Stop loss level reached'
                    })
                    risk_level = 1.0
            
            # Excessive drawdown check
            max_acceptable_loss = -0.05  # 5% maximum loss
            if current_pl_percent < max_acceptable_loss:
                risk_signals.append({
                    'type': 'EXCESSIVE_DRAWDOWN',
                    'urgency': 'HIGH',
                    'reason': f'Loss exceeds acceptable level: {current_pl_percent*100:.1f}%'
                })
                risk_level = max(risk_level, 0.8)
            
            # Volatility risk check
            if len(price_data) >= 10:
                recent_volatility = price_data['close'][-10:].pct_change().std()
                if recent_volatility > 0.03:  # 3% daily volatility
                    risk_signals.append({
                        'type': 'HIGH_VOLATILITY',
                        'urgency': 'MEDIUM',
                        'reason': f'High volatility detected: {recent_volatility*100:.1f}%'
                    })
                    risk_level = max(risk_level, 0.4)
            
            # Account risk check (if position size info available)
            if 'position_size_percent' in position_info:
                position_size = position_info['position_size_percent']
                if position_size > 5.0:  # More than 5% of account
                    risk_signals.append({
                        'type': 'LARGE_POSITION_SIZE',
                        'urgency': 'MEDIUM',
                        'reason': f'Large position size: {position_size:.1f}% of account'
                    })
                    risk_level = max(risk_level, 0.3)
            
            return {
                'risk_signals': risk_signals,
                'risk_level': round(risk_level, 4),
                'current_pl_percent': round(current_pl_percent * 100, 2),
                'risk_assessment': 'HIGH' if risk_level > 0.7 else 'MEDIUM' if risk_level > 0.4 else 'LOW'
            }
            
        except Exception as e:
            self.logger.error(f"Risk exits calculation error: {e}")
            return {'risk_signals': [], 'risk_level': 0.0}
    
    def _calculate_time_exits(self, position_info: Dict) -> Dict[str, Any]:
        """Calculate time-based exit signals"""
        try:
            if 'entry_time' not in position_info:
                return {'time_exit_signal': False, 'reason': 'No entry time available'}
            
            # Parse entry time
            entry_time = datetime.fromisoformat(position_info['entry_time']) if isinstance(position_info['entry_time'], str) else position_info['entry_time']
            current_time = datetime.now()
            hold_time_hours = (current_time - entry_time).total_seconds() / 3600
            
            # Check various time limits
            max_hold_time = self.exit_params['time_exits']['max_hold_time']
            profit_time_limit = self.exit_params['time_exits']['profit_time_limit']
            loss_time_limit = self.exit_params['time_exits']['loss_time_limit']
            
            time_signals = []
            time_exit_strength = 0.0
            
            # Maximum hold time check
            if hold_time_hours > max_hold_time:
                time_signals.append({
                    'type': 'MAX_HOLD_TIME_EXCEEDED',
                    'urgency': 'HIGH',
                    'reason': f'Position held for {hold_time_hours:.1f}h (max: {max_hold_time}h)'
                })
                time_exit_strength = 0.8
            
            # Profit/Loss time limits
            current_pl = position_info.get('current_pl_percent', 0.0)
            
            if current_pl > 0 and hold_time_hours > profit_time_limit:
                time_signals.append({
                    'type': 'PROFITABLE_TIME_LIMIT',
                    'urgency': 'MEDIUM',
                    'reason': f'Profitable position held for {hold_time_hours:.1f}h (limit: {profit_time_limit}h)'
                })
                time_exit_strength = max(time_exit_strength, 0.5)
            elif current_pl < 0 and hold_time_hours > loss_time_limit:
                time_signals.append({
                    'type': 'LOSING_TIME_LIMIT',
                    'urgency': 'MEDIUM',
                    'reason': f'Losing position held for {hold_time_hours:.1f}h (limit: {loss_time_limit}h)'
                })
                time_exit_strength = max(time_exit_strength, 0.6)
            
            # End of trading day/week considerations
            hour = current_time.hour
            weekday = current_time.weekday()
            
            # Friday afternoon close
            if weekday == 4 and hour >= 15:  # Friday 3 PM or later
                time_signals.append({
                    'type': 'END_OF_WEEK',
                    'urgency': 'LOW',
                    'reason': 'Consider closing before weekend'
                })
                time_exit_strength = max(time_exit_strength, 0.2)
            
            return {
                'time_exit_signals': time_signals,
                'time_exit_strength': round(time_exit_strength, 4),
                'hold_time_hours': round(hold_time_hours, 2),
                'time_remaining_max': max(0, max_hold_time - hold_time_hours),
                'time_exit_recommended': time_exit_strength > 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Time exits calculation error: {e}")
            return {'time_exit_signals': [], 'time_exit_strength': 0.0}
    
    def _calculate_profit_protection_exits(self, price_data: pd.DataFrame, position_info: Dict) -> Dict[str, Any]:
        """Calculate profit protection exit signals"""
        try:
            current_price = price_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            position_direction = position_info.get('direction', 'LONG').upper()
            
            # Calculate current profit
            if position_direction == 'LONG':
                current_profit_percent = (current_price - entry_price) / entry_price
            else:
                current_profit_percent = (entry_price - current_price) / entry_price
            
            protection_signals = []
            protection_strength = 0.0
            
            # Only activate profit protection if in profit
            if current_profit_percent <= 0:
                return {
                    'protection_signals': [],
                    'protection_strength': 0.0,
                    'reason': 'No profit to protect'
                }
            
            # Calculate R multiple if stop loss is available
            r_multiple = 0.0
            if 'stop_loss' in position_info and position_info['stop_loss'] > 0:
                stop_loss = position_info['stop_loss']
                if position_direction == 'LONG':
                    risk_amount = entry_price - stop_loss
                    current_profit_amount = current_price - entry_price
                else:
                    risk_amount = stop_loss - entry_price
                    current_profit_amount = entry_price - current_price
                
                r_multiple = current_profit_amount / risk_amount if risk_amount > 0 else 0.0
            
            # Partial profit taking signals
            partial_exit_levels = self.exit_params['profit_protection']['partial_exit_levels']
            for level in partial_exit_levels:
                if r_multiple >= level:
                    executed_exits = position_info.get('executed_partial_exits', [])
                    if level not in executed_exits:
                        protection_signals.append({
                            'type': 'PARTIAL_PROFIT_TAKING',
                            'r_level': level,
                            'urgency': 'MEDIUM',
                            'reason': f'Take partial profits at {level}R level'
                        })
                        protection_strength = max(protection_strength, 0.3)
            
            # Trailing stop activation
            activation_profit = self.exit_params['trailing_stop']['activation_profit']
            if current_profit_percent >= activation_profit:
                if not position_info.get('trailing_stop_active', False):
                    protection_signals.append({
                        'type': 'ACTIVATE_TRAILING_STOP',
                        'urgency': 'LOW',
                        'reason': 'Activate trailing stop to protect profits'
                    })
                    protection_strength = max(protection_strength, 0.2)
            
            # Breakeven stop move
            breakeven_r = self.exit_params['profit_protection']['breakeven_move']
            if r_multiple >= breakeven_r and not position_info.get('breakeven_moved', False):
                protection_signals.append({
                    'type': 'MOVE_TO_BREAKEVEN',
                    'urgency': 'MEDIUM',
                    'reason': f'Move stop to breakeven at {breakeven_r}R profit'
                })
                protection_strength = max(protection_strength, 0.4)
            
            # Profit deterioration protection
            max_profit = position_info.get('max_profit_percent', current_profit_percent)
            if current_profit_percent < max_profit * 0.7:  # 30% profit giveback
                protection_signals.append({
                    'type': 'PROFIT_DETERIORATION',
                    'urgency': 'HIGH',
                    'reason': f'Profits declined from {max_profit*100:.1f}% to {current_profit_percent*100:.1f}%'
                })
                protection_strength = max(protection_strength, 0.7)
            
            return {
                'protection_signals': protection_signals,
                'protection_strength': round(protection_strength, 4),
                'current_profit_percent': round(current_profit_percent * 100, 2),
                'r_multiple': round(r_multiple, 2),
                'max_profit_achieved': round(max_profit * 100, 2),
                'protection_active': len(protection_signals) > 0
            }
            
        except Exception as e:
            self.logger.error(f"Profit protection calculation error: {e}")
            return {'protection_signals': [], 'protection_strength': 0.0}
    
    def _combine_exit_signals(self, str_exit: Dict, technical: Dict, risk: Dict, 
                            time: Dict, protection: Dict) -> Dict[str, Any]:
        """Combine all exit signals into unified signal"""
        try:
            # Extract signal strengths
            str_strength = str_exit.get('signal_strength', 0.0) if str_exit.get('exit_signal', False) else 0.0
            tech_strength = technical.get('combined_strength', 0.0)
            risk_strength = risk.get('risk_level', 0.0)
            time_strength = time.get('time_exit_strength', 0.0)
            protection_strength = protection.get('protection_strength', 0.0)
            
            # Apply weights
            weighted_strength = (
                str_strength * self.exit_signal_weights['str_exit'] +
                tech_strength * self.exit_signal_weights['technical_exit'] +
                risk_strength * self.exit_signal_weights['risk_exit'] +
                time_strength * self.exit_signal_weights['time_exit'] +
                protection_strength * self.exit_signal_weights['profit_protection']
            )
            
            # Determine overall exit recommendation
            if weighted_strength >= 0.8:
                exit_recommendation = 'IMMEDIATE_EXIT'
            elif weighted_strength >= 0.6:
                exit_recommendation = 'STRONG_EXIT'
            elif weighted_strength >= 0.4:
                exit_recommendation = 'MODERATE_EXIT'
            elif weighted_strength >= 0.2:
                exit_recommendation = 'WEAK_EXIT'
            else:
                exit_recommendation = 'HOLD'
            
            # Identify dominant exit reason
            strengths = {
                'STR-EXIT': str_strength,
                'Technical': tech_strength,
                'Risk': risk_strength,
                'Time': time_strength,
                'Protection': protection_strength
            }
            
            dominant_reason = max(strengths.items(), key=lambda x: x[1])[0]
            
            return {
                'combined_strength': round(weighted_strength, 4),
                'exit_recommendation': exit_recommendation,
                'dominant_reason': dominant_reason,
                'component_strengths': {k: round(v, 4) for k, v in strengths.items()},
                'weighted_components': {
                    'str_exit_weighted': round(str_strength * self.exit_signal_weights['str_exit'], 4),
                    'technical_weighted': round(tech_strength * self.exit_signal_weights['technical_exit'], 4),
                    'risk_weighted': round(risk_strength * self.exit_signal_weights['risk_exit'], 4),
                    'time_weighted': round(time_strength * self.exit_signal_weights['time_exit'], 4),
                    'protection_weighted': round(protection_strength * self.exit_signal_weights['profit_protection'], 4)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Exit signal combination error: {e}")
            return {'combined_strength': 0.0, 'exit_recommendation': 'ERROR'}
    
    def _generate_exit_recommendations(self, combined_signal: Dict, position_info: Dict, 
                                     str_exit_signals: Dict) -> Dict[str, Any]:
        """Generate specific exit recommendations"""
        try:
            exit_recommendation = combined_signal.get('exit_recommendation', 'HOLD')
            combined_strength = combined_signal.get('combined_strength', 0.0)
            dominant_reason = combined_signal.get('dominant_reason', 'Unknown')
            
            recommendations = {}
            
            # Primary action determination
            if exit_recommendation == 'IMMEDIATE_EXIT':
                recommendations['primary_action'] = 'EXIT_FULL'
                recommendations['exit_percentage'] = 100
                recommendations['urgency'] = 'IMMEDIATE'
                recommendations['primary_reason'] = f'Immediate exit required - {dominant_reason} signal'
            elif exit_recommendation == 'STRONG_EXIT':
                recommendations['primary_action'] = 'EXIT_PARTIAL'
                recommendations['exit_percentage'] = 75
                recommendations['urgency'] = 'HIGH'
                recommendations['primary_reason'] = f'Strong exit signal - {dominant_reason}'
            elif exit_recommendation == 'MODERATE_EXIT':
                recommendations['primary_action'] = 'EXIT_PARTIAL'
                recommendations['exit_percentage'] = 50
                recommendations['urgency'] = 'MEDIUM'
                recommendations['primary_reason'] = f'Moderate exit signal - {dominant_reason}'
            elif exit_recommendation == 'WEAK_EXIT':
                recommendations['primary_action'] = 'REVIEW_POSITION'
                recommendations['exit_percentage'] = 25
                recommendations['urgency'] = 'LOW'
                recommendations['primary_reason'] = f'Weak exit signal - monitor closely'
            else:
                recommendations['primary_action'] = 'HOLD'
                recommendations['exit_percentage'] = 0
                recommendations['urgency'] = 'NONE'
                recommendations['primary_reason'] = 'No significant exit signals detected'
            
            # Suggested exit price
            if str_exit_signals.get('exit_signal', False):
                recommendations['suggested_exit_price'] = str_exit_signals.get('exit_level', 0.0)
            else:
                # Use current market price as fallback
                recommendations['suggested_exit_price'] = 0.0  # Would be current price in real implementation
            
            # Stop loss updates
            recommendations['stop_loss_update'] = self._suggest_stop_loss_update(position_info, combined_signal)
            
            # Take profit updates
            recommendations['take_profit_update'] = self._suggest_take_profit_update(position_info, combined_signal)
            
            # Trailing stop recommendation
            recommendations['trailing_stop_level'] = self._suggest_trailing_stop_level(position_info, combined_signal)
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Exit recommendations generation error: {e}")
            return {'primary_action': 'ERROR', 'primary_reason': 'Recommendation generation failed'}
    
    def _calculate_updated_levels(self, price_data: pd.DataFrame, position_info: Dict, 
                                combined_signal: Dict) -> Dict[str, Any]:
        """Calculate updated stop loss and take profit levels"""
        try:
            current_price = price_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            position_direction = position_info.get('direction', 'LONG').upper()
            
            updated_levels = {}
            
            # Calculate new stop loss level
            if combined_signal.get('combined_strength', 0) > 0.3:
                # Tighten stop loss when exit signals are present
                if position_direction == 'LONG':
                    # Move stop loss closer to current price
                    current_stop = position_info.get('stop_loss', entry_price * 0.98)
                    suggested_stop = max(current_stop, current_price * 0.99)  # 1% below current price
                else:  # SHORT
                    current_stop = position_info.get('stop_loss', entry_price * 1.02)
                    suggested_stop = min(current_stop, current_price * 1.01)  # 1% above current price
                
                updated_levels['suggested_stop_loss'] = round(suggested_stop, 5)
                updated_levels['stop_loss_reason'] = 'Tightened due to exit signals'
            else:
                updated_levels['suggested_stop_loss'] = position_info.get('stop_loss', 0.0)
                updated_levels['stop_loss_reason'] = 'No change recommended'
            
            # Calculate new take profit level
            current_tp = position_info.get('take_profit', 0.0)
            if current_tp > 0:
                # Keep existing take profit unless profit protection is active
                protection_strength = combined_signal.get('component_strengths', {}).get('Protection', 0.0)
                if protection_strength > 0.5:
                    # Lower take profit to secure profits
                    if position_direction == 'LONG':
                        suggested_tp = min(current_tp, current_price * 1.01)  # 1% above current
                    else:
                        suggested_tp = max(current_tp, current_price * 0.99)  # 1% below current
                    
                    updated_levels['suggested_take_profit'] = round(suggested_tp, 5)
                    updated_levels['take_profit_reason'] = 'Lowered to protect profits'
                else:
                    updated_levels['suggested_take_profit'] = current_tp
                    updated_levels['take_profit_reason'] = 'No change recommended'
            else:
                updated_levels['suggested_take_profit'] = 0.0
                updated_levels['take_profit_reason'] = 'No take profit set'
            
            return updated_levels
            
        except Exception as e:
            self.logger.error(f"Updated levels calculation error: {e}")
            return {'suggested_stop_loss': 0.0, 'suggested_take_profit': 0.0}
    
    def _analyze_position_status(self, price_data: pd.DataFrame, position_info: Dict) -> Dict[str, Any]:
        """Analyze current position status"""
        try:
            current_price = price_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            position_direction = position_info.get('direction', 'LONG').upper()
            
            # Calculate P&L
            if position_direction == 'LONG':
                pl_percent = (current_price - entry_price) / entry_price
                pl_points = current_price - entry_price
            else:
                pl_percent = (entry_price - current_price) / entry_price
                pl_points = entry_price - current_price
            
            # Determine status
            if pl_percent > 0.02:
                status = 'PROFITABLE'
            elif pl_percent > 0:
                status = 'SLIGHTLY_PROFITABLE'
            elif pl_percent > -0.02:
                status = 'BREAKEVEN'
            elif pl_percent > -0.05:
                status = 'SMALL_LOSS'
            else:
                status = 'SIGNIFICANT_LOSS'
            
            # Calculate hold time
            if 'entry_time' in position_info:
                entry_time = datetime.fromisoformat(position_info['entry_time']) if isinstance(position_info['entry_time'], str) else position_info['entry_time']
                hold_time_hours = (datetime.now() - entry_time).total_seconds() / 3600
            else:
                hold_time_hours = 0.0
            
            return {
                'status': status,
                'pl_percent': round(pl_percent * 100, 2),
                'pl_points': round(pl_points, 5),
                'hold_time_hours': round(hold_time_hours, 2),
                'current_price': current_price,
                'entry_price': entry_price,
                'position_direction': position_direction
            }
            
        except Exception as e:
            self.logger.error(f"Position status analysis error: {e}")
            return {'status': 'ERROR', 'pl_percent': 0.0}
    
    def _calculate_exit_urgency(self, combined_signal: Dict) -> Dict[str, Any]:
        """Calculate exit urgency level"""
        try:
            combined_strength = combined_signal.get('combined_strength', 0.0)
            
            if combined_strength >= self.urgency_thresholds['immediate']:
                urgency_level = 'IMMEDIATE'
                urgency_score = 1.0
                urgency_message = 'Exit immediately - critical signals detected'
            elif combined_strength >= self.urgency_thresholds['urgent']:
                urgency_level = 'URGENT' 
                urgency_score = combined_strength
                urgency_message = 'Exit urgently recommended - strong signals'
            elif combined_strength >= self.urgency_thresholds['moderate']:
                urgency_level = 'MODERATE'
                urgency_score = combined_strength
                urgency_message = 'Consider exit - moderate signals present'
            elif combined_strength >= self.urgency_thresholds['low']:
                urgency_level = 'LOW'
                urgency_score = combined_strength
                urgency_message = 'Monitor position - weak exit signals'
            else:
                urgency_level = 'NONE'
                urgency_score = 0.0
                urgency_message = 'No urgent exit required'
            
            return {
                'urgency_level': urgency_level,
                'urgency_score': round(urgency_score, 4),
                'urgency_message': urgency_message,
                'time_to_action': self._estimate_time_to_action(urgency_level)
            }
            
        except Exception as e:
            self.logger.error(f"Exit urgency calculation error: {e}")
            return {'urgency_level': 'ERROR', 'urgency_score': 0.0}
    
    def _estimate_time_to_action(self, urgency_level: str) -> str:
        """Estimate time frame for taking action"""
        time_frames = {
            'IMMEDIATE': 'Within 1 minute',
            'URGENT': 'Within 5 minutes', 
            'MODERATE': 'Within 30 minutes',
            'LOW': 'Within 2 hours',
            'NONE': 'No specific timeframe'
        }
        return time_frames.get(urgency_level, 'Unknown')
    
    def _update_trade_tracking(self, position_info: Dict, exit_result: Dict):
        """Update trade tracking information"""
        try:
            position_id = position_info.get('position_id', f"pos_{datetime.now().timestamp()}")
            
            # Update active trades
            self.active_trades[position_id] = {
                'position_info': position_info,
                'last_exit_analysis': exit_result,
                'analysis_time': datetime.now().isoformat()
            }
            
            # Cleanup old trades (remove trades older than 7 days)
            current_time = datetime.now()
            expired_trades = []
            
            for trade_id, trade_data in self.active_trades.items():
                analysis_time = datetime.fromisoformat(trade_data['analysis_time'])
                if (current_time - analysis_time).days > 7:
                    expired_trades.append(trade_id)
            
            for trade_id in expired_trades:
                del self.active_trades[trade_id]
                
        except Exception as e:
            self.logger.error(f"Trade tracking update error: {e}")
    
    def _calculate_time_remaining(self, position_info: Dict) -> Dict[str, float]:
        """Calculate time remaining before various time-based exits"""
        try:
            if 'entry_time' not in position_info:
                return {'max_hold_time_remaining': 0.0}
            
            entry_time = datetime.fromisoformat(position_info['entry_time']) if isinstance(position_info['entry_time'], str) else position_info['entry_time']
            current_time = datetime.now()
            hold_time_hours = (current_time - entry_time).total_seconds() / 3600
            
            max_hold_time = self.exit_params['time_exits']['max_hold_time']
            
            return {
                'max_hold_time_remaining': max(0.0, max_hold_time - hold_time_hours),
                'current_hold_time': hold_time_hours
            }
            
        except Exception as e:
            self.logger.error(f"Time remaining calculation error: {e}")
            return {'max_hold_time_remaining': 0.0}
    
    def _calculate_current_pl_status(self, price_data: pd.DataFrame, position_info: Dict) -> Dict[str, Any]:
        """Calculate current profit/loss status"""
        try:
            current_price = price_data['close'].iloc[-1]
            entry_price = position_info['entry_price']
            position_direction = position_info.get('direction', 'LONG').upper()
            
            if position_direction == 'LONG':
                pl_percent = (current_price - entry_price) / entry_price
                pl_amount = current_price - entry_price
            else:
                pl_percent = (entry_price - current_price) / entry_price  
                pl_amount = entry_price - current_price
            
            # Calculate position value if size is known
            position_size = position_info.get('position_size', 1.0)
            pl_value = pl_amount * position_size
            
            return {
                'pl_percent': round(pl_percent * 100, 2),
                'pl_amount_per_unit': round(pl_amount, 5),
                'pl_total_value': round(pl_value, 2),
                'is_profitable': pl_percent > 0,
                'profit_loss_ratio': round(pl_percent, 4)
            }
            
        except Exception as e:
            self.logger.error(f"P&L status calculation error: {e}")
            return {'pl_percent': 0.0, 'is_profitable': False}
    
    def _calculate_partial_exit_price(self, entry_price: float, direction: str, r_multiple: float, 
                                    position_info: Dict) -> float:
        """Calculate price for partial exit at specific R multiple"""
        try:
            if 'stop_loss' not in position_info:
                return 0.0
            
            stop_loss = position_info['stop_loss']
            
            if direction == 'LONG':
                risk_per_unit = entry_price - stop_loss
                target_price = entry_price + (risk_per_unit * r_multiple)
            else:  # SHORT
                risk_per_unit = stop_loss - entry_price  
                target_price = entry_price - (risk_per_unit * r_multiple)
            
            return round(target_price, 5)
            
        except Exception as e:
            self.logger.error(f"Partial exit price calculation error: {e}")
            return 0.0
    
    def _calculate_next_trailing_price(self, current_price: float, direction: str, step_size: float) -> float:
        """Calculate next price level to update trailing stop"""
        try:
            if direction == 'LONG':
                return round(current_price * (1 + step_size), 5)
            else:
                return round(current_price * (1 - step_size), 5)
                
        except Exception as e:
            self.logger.error(f"Next trailing price calculation error: {e}")
            return current_price
    
    def _suggest_stop_loss_update(self, position_info: Dict, combined_signal: Dict) -> Optional[float]:
        """Suggest stop loss update based on exit signals"""
        try:
            # Implementation would depend on specific exit signal analysis
            # For now, return None (no update suggested)
            return None
            
        except Exception as e:
            self.logger.error(f"Stop loss update suggestion error: {e}")
            return None
    
    def _suggest_take_profit_update(self, position_info: Dict, combined_signal: Dict) -> Optional[float]:
        """Suggest take profit update based on exit signals"""
        try:
            # Implementation would depend on specific exit signal analysis
            # For now, return None (no update suggested)
            return None
            
        except Exception as e:
            self.logger.error(f"Take profit update suggestion error: {e}")
            return None
    
    def _suggest_trailing_stop_level(self, position_info: Dict, combined_signal: Dict) -> Optional[float]:
        """Suggest trailing stop level based on exit signals"""
        try:
            # Implementation would depend on specific exit signal analysis
            # For now, return None (no trailing stop suggested)
            return None
            
        except Exception as e:
            self.logger.error(f"Trailing stop suggestion error: {e}")
            return None
    
    def _empty_exit_result(self) -> Dict[str, Any]:
        """Return empty exit result"""
        return {
            'exit_signals': {},
            'combined_signal': {'combined_strength': 0.0, 'exit_recommendation': 'ERROR'},
            'exit_recommendations': {'primary_action': 'ERROR', 'primary_reason': 'Exit analysis failed'},
            'updated_levels': {},
            'position_analysis': {'status': 'ERROR'},
            'exit_urgency': {'urgency_level': 'ERROR', 'urgency_score': 0.0},
            'timestamp': datetime.now().isoformat(),
            'error': 'Exit evaluation failed'
        }

# Usage Example (for testing):
if __name__ == "__main__":
    # Sample usage
    exit_manager = ExitManager()
    
    # Sample OHLC data
    sample_data = pd.DataFrame({
        'high': np.random.rand(100) * 100 + 1000,
        'low': np.random.rand(100) * 100 + 950,
        'close': np.random.rand(100) * 100 + 975,
        'open': np.random.rand(100) * 100 + 975,
        'volume': np.random.rand(100) * 1000 + 500
    })
    
    # Sample position info
    sample_position = {
        'position_id': 'test_001',
        'entry_price': 970.0,
        'direction': 'LONG',
        'position_size': 100,
        'stop_loss': 960.0,
        'take_profit': 990.0,
        'entry_time': datetime.now() - timedelta(hours=2)
    }
    
    # Evaluate exit signals
    exit_analysis = exit_manager.evaluate_exit_signals(sample_data, sample_position)
    print("Exit Analysis:", json.dumps(exit_analysis, indent=2, default=str))
    
    # Get exit recommendation
    exit_rec = exit_manager.get_exit_recommendation(sample_data, sample_position)
    print("Exit Recommendation:", exit_rec)
    
    # Manage partial exits
    partial_exits = exit_manager.manage_partial_exits(sample_data, sample_position)
    print("Partial Exits:", partial_exits)
    
    # Update trailing stop
    trailing_update = exit_manager.update_trailing_stop(sample_data, sample_position)
    print("Trailing Stop Update:", trailing_update)
    
    # Check emergency exits
    emergency_check = exit_manager.check_emergency_exits(sample_data, sample_position)
    print("Emergency Check:", emergency_check)