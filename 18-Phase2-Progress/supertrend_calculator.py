#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - SuperTrend Calculator
Advanced SuperTrend calculation with STR-ENTRY and STR-EXIT logic

Features:
- STR-ENTRY: ATR multiplier 1.0, ATR period 20 (for trade entries)
- STR-EXIT: ATR multiplier 1.5, ATR period 20 (for trade exits)
- Multi-timeframe SuperTrend analysis
- Trend change detection and confirmation
- Support/Resistance level identification
- Signal strength and reliability scoring

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

class SuperTrendCalculator:
    """
    Advanced SuperTrend Calculator with Entry/Exit Logic
    
    Implements both STR-ENTRY and STR-EXIT SuperTrend variants:
    - STR-ENTRY: Sensitive entries (ATR 1.0x, 20 period)
    - STR-EXIT: Conservative exits (ATR 1.5x, 20 period)
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize SuperTrend Calculator"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # SuperTrend Parameters (as per EA specification)
        self.params = {
            'str_entry': {
                'atr_period': 20,        # ATR period for entry signals
                'atr_multiplier': 1.0    # ATR multiplier for entry signals
            },
            'str_exit': {
                'atr_period': 20,        # ATR period for exit signals
                'atr_multiplier': 1.5    # ATR multiplier for exit signals
            }
        }
        
        # Analysis thresholds
        self.thresholds = {
            'strong_trend': 0.8,         # Strong trend signal threshold
            'medium_trend': 0.6,         # Medium trend signal threshold
            'weak_trend': 0.4,           # Weak trend signal threshold
            'trend_change_confirmation': 3,  # Bars for trend change confirmation
            'distance_threshold': 0.002   # Minimum distance for reliable signals
        }
        
        # Signal scoring weights
        self.weights = {
            'trend_direction': 0.4,      # Current trend direction
            'trend_strength': 0.3,       # Trend strength based on distance
            'trend_consistency': 0.2,    # Trend consistency over time
            'momentum': 0.1              # Price momentum component
        }
        
        # Historical data cache
        self.supertrend_cache = {}
        self.last_calculation = {}
        
        self.logger.info("📊 SuperTrend Calculator initialized")
    
    def calculate_supertrend(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Calculate both STR-ENTRY and STR-EXIT SuperTrend indicators
        
        Args:
            price_data: OHLC price data
            timeframe: Chart timeframe
            
        Returns:
            Dictionary with both SuperTrend variants and analysis
        """
        try:
            required_bars = max(self.params['str_entry']['atr_period'], 
                              self.params['str_exit']['atr_period']) + 50
            
            if len(price_data) < required_bars:
                self.logger.warning(f"Insufficient data for SuperTrend calculation: {len(price_data)} bars, need {required_bars}")
                return self._empty_supertrend_result()
            
            # Calculate STR-ENTRY SuperTrend
            str_entry_data = self._calculate_single_supertrend(
                price_data, 
                'str_entry',
                self.params['str_entry']['atr_period'],
                self.params['str_entry']['atr_multiplier']
            )
            
            # Calculate STR-EXIT SuperTrend
            str_exit_data = self._calculate_single_supertrend(
                price_data,
                'str_exit', 
                self.params['str_exit']['atr_period'],
                self.params['str_exit']['atr_multiplier']
            )
            
            # Analyze trend signals
            trend_analysis = self._analyze_trend_signals(price_data, str_entry_data, str_exit_data)
            
            # Detect trend changes
            trend_changes = self._detect_trend_changes(str_entry_data, str_exit_data)
            
            # Calculate support/resistance levels
            support_resistance = self._calculate_sr_levels(price_data, str_entry_data, str_exit_data)
            
            # Multi-timeframe analysis (simplified)
            mtf_analysis = self._multi_timeframe_analysis(trend_analysis, timeframe)
            
            result = {
                'str_entry': str_entry_data,
                'str_exit': str_exit_data,
                'trend_analysis': trend_analysis,
                'trend_changes': trend_changes,
                'support_resistance': support_resistance,
                'mtf_analysis': mtf_analysis,
                'current_values': self._get_current_supertrend_values(str_entry_data, str_exit_data),
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'data_points': len(price_data)
            }
            
            # Cache result
            self.last_calculation[timeframe] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"SuperTrend calculation error: {e}")
            if self.error_handler:
                self.error_handler.log_error("SUPERTREND_CALC_ERROR", str(e))
            return self._empty_supertrend_result()
    
    def get_entry_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get STR-ENTRY based trading signals
        
        Returns:
            Entry signal with direction, strength, and confidence
        """
        try:
            supertrend_result = self.calculate_supertrend(price_data)
            if not supertrend_result or not supertrend_result['str_entry']:
                return {'signal': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            str_entry = supertrend_result['str_entry']
            trend_analysis = supertrend_result['trend_analysis']
            
            # Determine entry signal direction
            signal_direction = self._determine_entry_signal_direction(str_entry, trend_analysis)
            
            # Calculate signal strength
            signal_strength = self._calculate_entry_signal_strength(str_entry, trend_analysis)
            
            # Calculate confidence
            confidence = self._calculate_entry_confidence(
                supertrend_result['str_entry'],
                supertrend_result['trend_changes'],
                supertrend_result['mtf_analysis']
            )
            
            return {
                'signal': signal_direction,
                'strength': round(signal_strength, 4),
                'confidence': round(confidence, 4),
                'entry_price': price_data['close'].iloc[-1],
                'supertrend_level': str_entry['supertrend'][-1],
                'trend_direction': str_entry['trend'][-1],
                'distance_to_supertrend': abs(price_data['close'].iloc[-1] - str_entry['supertrend'][-1]),
                'atr_value': str_entry['atr'][-1],
                'details': {
                    'trend_consistency': trend_analysis.get('trend_consistency', 0),
                    'trend_strength': trend_analysis.get('trend_strength', 0),
                    'momentum_score': trend_analysis.get('momentum_score', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Entry signal error: {e}")
            return {'signal': 'ERROR', 'strength': 0.0, 'confidence': 0.0}
    
    def get_exit_signal(self, price_data: pd.DataFrame, current_position: str = 'NONE') -> Dict[str, Any]:
        """
        Get STR-EXIT based exit signals
        
        Args:
            price_data: OHLC price data
            current_position: Current position ('LONG', 'SHORT', 'NONE')
            
        Returns:
            Exit signal with recommendation and levels
        """
        try:
            supertrend_result = self.calculate_supertrend(price_data)
            if not supertrend_result or not supertrend_result['str_exit']:
                return {'exit_signal': 'HOLD', 'exit_reason': 'NO_DATA', 'exit_level': 0.0}
            
            str_exit = supertrend_result['str_exit']
            current_price = price_data['close'].iloc[-1]
            current_trend = str_exit['trend'][-1]
            supertrend_level = str_exit['supertrend'][-1]
            
            # Determine exit signal based on position and trend
            exit_recommendation = self._determine_exit_recommendation(
                current_position, current_trend, current_price, supertrend_level
            )
            
            # Calculate exit urgency
            exit_urgency = self._calculate_exit_urgency(
                current_position, str_exit, price_data
            )
            
            # Get stop loss level
            stop_loss_level = self._calculate_stop_loss_level(
                current_position, str_exit, price_data
            )
            
            return {
                'exit_signal': exit_recommendation['signal'],
                'exit_reason': exit_recommendation['reason'],
                'exit_level': supertrend_level,
                'stop_loss_level': stop_loss_level,
                'exit_urgency': round(exit_urgency, 4),
                'current_trend': current_trend,
                'trend_change_detected': exit_recommendation.get('trend_change', False),
                'distance_to_exit': abs(current_price - supertrend_level),
                'atr_distance': abs(current_price - supertrend_level) / str_exit['atr'][-1],
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Exit signal error: {e}")
            return {'exit_signal': 'ERROR', 'exit_reason': 'CALCULATION_ERROR', 'exit_level': 0.0}
    
    def detect_trend_changes(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect SuperTrend based trend changes
        
        Returns:
            Trend change detection results
        """
        try:
            supertrend_result = self.calculate_supertrend(price_data)
            if not supertrend_result:
                return {'trend_change': False, 'new_trend': 'UNKNOWN', 'confidence': 0.0}
            
            trend_changes = supertrend_result['trend_changes']
            str_entry = supertrend_result['str_entry']
            str_exit = supertrend_result['str_exit']
            
            # Check for recent trend changes
            recent_entry_change = self._check_recent_trend_change(str_entry, 5)
            recent_exit_change = self._check_recent_trend_change(str_exit, 5)
            
            # Determine overall trend change status
            trend_change_detected = recent_entry_change['changed'] or recent_exit_change['changed']
            
            if trend_change_detected:
                # SuperTrend calculation
                if i == 0:
                    # Initialize first values
                    final_upper_band[0] = upper_band[0]
                    final_lower_band[0] = lower_band[0]
                    supertrend[0] = final_upper_band[0]
                    trend[0] = 1  # Start with uptrend
                else:
                    # SuperTrend logic
                    if (supertrend[i-1] == final_upper_band[i-1] and close[i] <= final_upper_band[i]) or \
                       (supertrend[i-1] == final_upper_band[i-1] and close[i] > final_upper_band[i]):
                        supertrend[i] = final_upper_band[i]
                    elif (supertrend[i-1] == final_lower_band[i-1] and close[i] >= final_lower_band[i]) or \
                         (supertrend[i-1] == final_lower_band[i-1] and close[i] < final_lower_band[i]):
                        supertrend[i] = final_lower_band[i]
                    elif supertrend[i-1] == final_upper_band[i-1] and close[i] < final_upper_band[i]:
                        supertrend[i] = final_upper_band[i]
                    else:
                        supertrend[i] = final_lower_band[i]
                    
                    # Trend direction
                    if close[i] <= supertrend[i]:
                        trend[i] = -1  # Downtrend
                    else:
                        trend[i] = 1   # Uptrend
            
            return {
                'supertrend': supertrend,
                'trend': trend,
                'atr': atr,
                'upper_band': final_upper_band,
                'lower_band': final_lower_band,
                'variant': variant
            }
            
        except Exception as e:
            self.logger.error(f"Single SuperTrend calculation error for {variant}: {e}")
            return {}
    
    def _analyze_trend_signals(self, price_data: pd.DataFrame, str_entry: Dict, str_exit: Dict) -> Dict[str, Any]:
        """Analyze trend signals from both SuperTrend variants"""
        try:
            current_price = price_data['close'].iloc[-1]
            
            # Entry trend analysis
            entry_trend = str_entry['trend'][-1]
            entry_supertrend = str_entry['supertrend'][-1]
            entry_distance = abs(current_price - entry_supertrend) / current_price
            
            # Exit trend analysis
            exit_trend = str_exit['trend'][-1]
            exit_supertrend = str_exit['supertrend'][-1]
            exit_distance = abs(current_price - exit_supertrend) / current_price
            
            # Trend consistency (how long current trend has been active)
            entry_consistency = self._calculate_trend_consistency(str_entry['trend'])
            exit_consistency = self._calculate_trend_consistency(str_exit['trend'])
            
            # Trend strength based on price distance from SuperTrend
            entry_strength = min(1.0, entry_distance / self.thresholds['distance_threshold'])
            exit_strength = min(1.0, exit_distance / self.thresholds['distance_threshold'])
            
            # Momentum calculation
            momentum_score = self._calculate_momentum_score(price_data, str_entry)
            
            # Overall trend analysis
            trend_agreement = 1.0 if entry_trend == exit_trend else 0.5
            
            return {
                'entry_trend': int(entry_trend),
                'exit_trend': int(exit_trend),
                'trend_agreement': trend_agreement,
                'entry_strength': round(entry_strength, 4),
                'exit_strength': round(exit_strength, 4),
                'entry_consistency': round(entry_consistency, 4),
                'exit_consistency': round(exit_consistency, 4),
                'momentum_score': round(momentum_score, 4),
                'entry_distance': round(entry_distance, 6),
                'exit_distance': round(exit_distance, 6),
                'trend_strength': round((entry_strength + exit_strength) / 2, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Trend signal analysis error: {e}")
            return {}
    
    def _detect_trend_changes(self, str_entry: Dict, str_exit: Dict) -> Dict[str, Any]:
        """Detect trend changes in SuperTrend indicators"""
        try:
            # Check for recent trend changes in both variants
            entry_changes = self._find_trend_changes(str_entry['trend'], 10)
            exit_changes = self._find_trend_changes(str_exit['trend'], 10)
            
            # Calculate change strength
            change_strength = 0.0
            if entry_changes or exit_changes:
                # Recent change detected
                entry_change_strength = len(entry_changes) * 0.3
                exit_change_strength = len(exit_changes) * 0.2
                change_strength = min(1.0, entry_change_strength + exit_change_strength)
            
            return {
                'entry_changes': entry_changes,
                'exit_changes': exit_changes,
                'recent_changes': len(entry_changes) + len(exit_changes),
                'change_strength': round(change_strength, 4),
                'last_entry_change': entry_changes[0] if entry_changes else None,
                'last_exit_change': exit_changes[0] if exit_changes else None
            }
            
        except Exception as e:
            self.logger.error(f"Trend change detection error: {e}")
            return {}
    
    def _calculate_sr_levels(self, price_data: pd.DataFrame, str_entry: Dict, str_exit: Dict) -> Dict[str, Any]:
        """Calculate support/resistance levels from SuperTrend"""
        try:
            current_price = price_data['close'].iloc[-1]
            
            support_levels = []
            resistance_levels = []
            
            # Current SuperTrend levels
            entry_level = str_entry['supertrend'][-1]
            exit_level = str_exit['supertrend'][-1]
            entry_trend = str_entry['trend'][-1]
            exit_trend = str_exit['trend'][-1]
            
            # Assign levels based on trend direction
            if entry_trend == 1:  # Uptrend
                support_levels.append(entry_level)
            else:  # Downtrend
                resistance_levels.append(entry_level)
            
            if exit_trend == 1:  # Uptrend
                support_levels.append(exit_level)
            else:  # Downtrend
                resistance_levels.append(exit_level)
            
            # Historical levels from trend changes
            historical_levels = self._get_historical_supertrend_levels(str_entry, str_exit)
            support_levels.extend(historical_levels['support'])
            resistance_levels.extend(historical_levels['resistance'])
            
            # Remove duplicates and sort
            support_levels = sorted(list(set([level for level in support_levels if level > 0])))
            resistance_levels = sorted(list(set([level for level in resistance_levels if level > 0])))
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'dynamic_support': entry_level if entry_trend == 1 else 0.0,
                'dynamic_resistance': entry_level if entry_trend == -1 else 0.0,
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance calculation error: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _multi_timeframe_analysis(self, trend_analysis: Dict, current_tf: str) -> Dict[str, Any]:
        """Perform multi-timeframe analysis (simplified)"""
        try:
            # Simplified MTF analysis - would need actual higher timeframe data
            return {
                'higher_tf_trend': 'UNKNOWN',
                'lower_tf_trend': 'UNKNOWN',
                'mtf_agreement': 0.5,
                'confidence_boost': 0.0,
                'current_tf': current_tf
            }
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe analysis error: {e}")
            return {}
    
    def _calculate_trend_consistency(self, trend_array: np.ndarray) -> float:
        """Calculate trend consistency over recent periods"""
        try:
            if len(trend_array) < 5:
                return 0.0
            
            recent_trend = trend_array[-10:]  # Last 10 periods
            current_trend = trend_array[-1]
            
            # Count periods with same trend
            same_trend_count = np.sum(recent_trend == current_trend)
            consistency = same_trend_count / len(recent_trend)
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"Trend consistency calculation error: {e}")
            return 0.0
    
    def _calculate_momentum_score(self, price_data: pd.DataFrame, str_entry: Dict) -> float:
        """Calculate momentum score"""
        try:
            if len(price_data) < 5:
                return 0.0
            
            # Price momentum over last 5 periods
            recent_closes = price_data['close'][-5:].values
            price_change = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            
            # SuperTrend momentum
            recent_supertrend = str_entry['supertrend'][-5:]
            supertrend_change = (recent_supertrend[-1] - recent_supertrend[0]) / recent_supertrend[0]
            
            # Combined momentum score
            momentum = (price_change + supertrend_change) / 2
            
            # Normalize to [-1, 1] range
            return max(-1.0, min(1.0, momentum * 100))
            
        except Exception as e:
            self.logger.error(f"Momentum score calculation error: {e}")
            return 0.0
    
    def _find_trend_changes(self, trend_array: np.ndarray, lookback: int) -> List[Dict[str, Any]]:
        """Find trend changes in the specified lookback period"""
        try:
            changes = []
            
            if len(trend_array) < 2:
                return changes
            
            # Look for trend changes in recent periods
            search_length = min(lookback, len(trend_array) - 1)
            
            for i in range(1, search_length + 1):
                if trend_array[-i] != trend_array[-(i+1)]:
                    changes.append({
                        'bars_ago': i,
                        'from_trend': int(trend_array[-(i+1)]),
                        'to_trend': int(trend_array[-i]),
                        'change_type': 'BULLISH' if trend_array[-i] == 1 else 'BEARISH'
                    })
            
            return changes
            
        except Exception as e:
            self.logger.error(f"Trend change finding error: {e}")
            return []
    
    def _get_historical_supertrend_levels(self, str_entry: Dict, str_exit: Dict) -> Dict[str, List[float]]:
        """Get historical SuperTrend levels from trend changes"""
        try:
            support_levels = []
            resistance_levels = []
            
            # Get levels from recent trend changes (last 20 periods)
            if len(str_entry['supertrend']) >= 20:
                recent_entry_st = str_entry['supertrend'][-20:]
                recent_entry_trend = str_entry['trend'][-20:]
                
                for i in range(1, len(recent_entry_trend)):
                    if recent_entry_trend[i] != recent_entry_trend[i-1]:
                        # Trend change detected
                        level = recent_entry_st[i-1]
                        if recent_entry_trend[i-1] == 1:  # Was uptrend, now downtrend
                            resistance_levels.append(level)
                        else:  # Was downtrend, now uptrend
                            support_levels.append(level)
            
            return {
                'support': support_levels[-3:],  # Last 3 levels
                'resistance': resistance_levels[-3:]  # Last 3 levels
            }
            
        except Exception as e:
            self.logger.error(f"Historical levels calculation error: {e}")
            return {'support': [], 'resistance': []}
    
    def _get_current_supertrend_values(self, str_entry: Dict, str_exit: Dict) -> Dict[str, float]:
        """Get current SuperTrend values"""
        try:
            return {
                'str_entry_level': float(str_entry['supertrend'][-1]),
                'str_entry_trend': int(str_entry['trend'][-1]),
                'str_entry_atr': float(str_entry['atr'][-1]),
                'str_exit_level': float(str_exit['supertrend'][-1]),
                'str_exit_trend': int(str_exit['trend'][-1]),
                'str_exit_atr': float(str_exit['atr'][-1]),
                'entry_upper_band': float(str_entry['upper_band'][-1]),
                'entry_lower_band': float(str_entry['lower_band'][-1]),
                'exit_upper_band': float(str_exit['upper_band'][-1]),
                'exit_lower_band': float(str_exit['lower_band'][-1])
            }
        except Exception as e:
            self.logger.error(f"Current values extraction error: {e}")
            return {}
    
    def _determine_entry_signal_direction(self, str_entry: Dict, trend_analysis: Dict) -> str:
        """Determine entry signal direction"""
        try:
            entry_trend = str_entry['trend'][-1]
            trend_strength = trend_analysis.get('trend_strength', 0)
            trend_consistency = trend_analysis.get('entry_consistency', 0)
            
            if entry_trend == 1:  # Uptrend
                if trend_strength >= self.thresholds['strong_trend'] and trend_consistency >= 0.7:
                    return 'STRONG_BULLISH'
                elif trend_strength >= self.thresholds['medium_trend']:
                    return 'BULLISH'
                elif trend_strength >= self.thresholds['weak_trend']:
                    return 'WEAK_BULLISH'
                else:
                    return 'NEUTRAL_BULLISH'
            else:  # Downtrend
                if trend_strength >= self.thresholds['strong_trend'] and trend_consistency >= 0.7:
                    return 'STRONG_BEARISH'
                elif trend_strength >= self.thresholds['medium_trend']:
                    return 'BEARISH'
                elif trend_strength >= self.thresholds['weak_trend']:
                    return 'WEAK_BEARISH'
                else:
                    return 'NEUTRAL_BEARISH'
                    
        except Exception as e:
            self.logger.error(f"Entry signal direction error: {e}")
            return 'ERROR'
    
    def _calculate_entry_signal_strength(self, str_entry: Dict, trend_analysis: Dict) -> float:
        """Calculate entry signal strength"""
        try:
            # Combine different strength components
            trend_strength = trend_analysis.get('trend_strength', 0) * self.weights['trend_strength']
            trend_consistency = trend_analysis.get('entry_consistency', 0) * self.weights['trend_consistency']
            momentum = abs(trend_analysis.get('momentum_score', 0)) * self.weights['momentum']
            
            # Trend direction component
            trend_direction_score = 1.0 * self.weights['trend_direction']
            
            total_strength = trend_strength + trend_consistency + momentum + trend_direction_score
            return min(1.0, total_strength)
            
        except Exception as e:
            self.logger.error(f"Entry signal strength calculation error: {e}")
            return 0.0
    
    def _calculate_entry_confidence(self, str_entry: Dict, trend_changes: Dict, mtf_analysis: Dict) -> float:
        """Calculate entry signal confidence"""
        try:
            base_confidence = 0.5
            
            # Trend stability (fewer recent changes = higher confidence)
            recent_changes = trend_changes.get('recent_changes', 0)
            stability_boost = max(0.0, (5 - recent_changes) / 5 * 0.3)
            
            # ATR consistency (stable ATR = higher confidence)
            if len(str_entry['atr']) >= 5:
                recent_atr = str_entry['atr'][-5:]
                atr_cv = np.std(recent_atr) / np.mean(recent_atr) if np.mean(recent_atr) > 0 else 1.0
                atr_boost = max(0.0, (0.2 - atr_cv) * 2.5) if atr_cv < 0.2 else 0.0
            else:
                atr_boost = 0.0
            
            # Multi-timeframe boost
            mtf_boost = mtf_analysis.get('confidence_boost', 0)
            
            confidence = base_confidence + stability_boost + atr_boost + mtf_boost
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Entry confidence calculation error: {e}")
            return 0.0
    
    def _determine_exit_recommendation(self, current_position: str, current_trend: int, 
                                     current_price: float, supertrend_level: float) -> Dict[str, Any]:
        """Determine exit recommendation based on position and trend"""
        try:
            if current_position == 'NONE':
                return {'signal': 'NO_POSITION', 'reason': 'No active position'}
            
            if current_position == 'LONG':
                if current_trend == -1:  # Trend turned bearish
                    return {
                        'signal': 'EXIT_LONG',
                        'reason': 'Trend changed to bearish',
                        'trend_change': True
                    }
                elif current_price < supertrend_level:  # Price below SuperTrend
                    return {
                        'signal': 'EXIT_LONG',
                        'reason': 'Price below SuperTrend exit level',
                        'trend_change': False
                    }
                else:
                    return {'signal': 'HOLD_LONG', 'reason': 'Trend still bullish'}
            
            elif current_position == 'SHORT':
                if current_trend == 1:  # Trend turned bullish
                    return {
                        'signal': 'EXIT_SHORT',
                        'reason': 'Trend changed to bullish',
                        'trend_change': True
                    }
                elif current_price > supertrend_level:  # Price above SuperTrend
                    return {
                        'signal': 'EXIT_SHORT',
                        'reason': 'Price above SuperTrend exit level',
                        'trend_change': False
                    }
                else:
                    return {'signal': 'HOLD_SHORT', 'reason': 'Trend still bearish'}
            
            return {'signal': 'HOLD', 'reason': 'Unknown position type'}
            
        except Exception as e:
            self.logger.error(f"Exit recommendation error: {e}")
            return {'signal': 'ERROR', 'reason': 'Calculation error'}
    
    def _calculate_exit_urgency(self, current_position: str, str_exit: Dict, price_data: pd.DataFrame) -> float:
        """Calculate exit urgency"""
        try:
            if current_position == 'NONE':
                return 0.0
            
            current_price = price_data['close'].iloc[-1]
            supertrend_level = str_exit['supertrend'][-1]
            atr_value = str_exit['atr'][-1]
            
            # Distance to SuperTrend in ATR terms
            distance_atr = abs(current_price - supertrend_level) / atr_value
            
            # Higher urgency when closer to SuperTrend level
            urgency = max(0.0, 1.0 - (distance_atr / 2))
            
            # Additional urgency if trend recently changed
            if len(str_exit['trend']) >= 3:
                recent_trends = str_exit['trend'][-3:]
                if not all(t == recent_trends[0] for t in recent_trends):
                    urgency += 0.3
            
            return min(1.0, urgency)
            
        except Exception as e:
            self.logger.error(f"Exit urgency calculation error: {e}")
            return 0.0
    
    def _calculate_stop_loss_level(self, current_position: str, str_exit: Dict, price_data: pd.DataFrame) -> float:
        """Calculate stop loss level based on SuperTrend"""
        try:
            if current_position == 'NONE':
                return 0.0
            
            supertrend_level = str_exit['supertrend'][-1]
            atr_value = str_exit['atr'][-1]
            
            # Stop loss is at SuperTrend level with small buffer
            if current_position == 'LONG':
                stop_loss = supertrend_level - (atr_value * 0.1)  # Slightly below SuperTrend
            else:  # SHORT
                stop_loss = supertrend_level + (atr_value * 0.1)  # Slightly above SuperTrend
            
            return stop_loss
            
        except Exception as e:
            self.logger.error(f"Stop loss calculation error: {e}")
            return 0.0
    
    def _check_recent_trend_change(self, supertrend_data: Dict, lookback: int) -> Dict[str, Any]:
        """Check for recent trend changes"""
        try:
            trend_array = supertrend_data['trend']
            
            if len(trend_array) < 2:
                return {'changed': False, 'bars_ago': 999}
            
            current_trend = trend_array[-1]
            
            # Look for trend change in recent periods
            for i in range(1, min(lookback + 1, len(trend_array))):
                if trend_array[-i-1] != current_trend:
                    return {
                        'changed': True,
                        'bars_ago': i,
                        'from_trend': int(trend_array[-i-1]),
                        'to_trend': int(current_trend),
                        'change_type': 'BULLISH' if current_trend == 1 else 'BEARISH'
                    }
            
            return {'changed': False, 'bars_ago': 999}
            
        except Exception as e:
            self.logger.error(f"Recent trend change check error: {e}")
            return {'changed': False, 'bars_ago': 999}
    
    def _calculate_trend_change_confidence(self, entry_change: Dict, exit_change: Dict, 
                                         supertrend_result: Dict) -> float:
        """Calculate trend change confidence"""
        try:
            confidence = 0.0
            
            # Both indicators changed (higher confidence)
            if entry_change.get('changed', False) and exit_change.get('changed', False):
                confidence += 0.6
                
                # Same direction change
                if entry_change.get('change_type') == exit_change.get('change_type'):
                    confidence += 0.2
            
            # Single indicator change
            elif entry_change.get('changed', False) or exit_change.get('changed', False):
                confidence += 0.4
            
            # Trend agreement boost
            trend_agreement = supertrend_result['trend_analysis'].get('trend_agreement', 0)
            confidence += trend_agreement * 0.2
            
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Trend change confidence calculation error: {e}")
            return 0.0
    
    def _calculate_trend_duration(self, str_entry: Dict) -> int:
        """Calculate how long current trend has been active"""
        try:
            trend_array = str_entry['trend']
            
            if len(trend_array) < 2:
                return 0
            
            current_trend = trend_array[-1]
            duration = 0
            
            # Count consecutive periods with same trend
            for i in range(len(trend_array) - 1, -1, -1):
                if trend_array[i] == current_trend:
                    duration += 1
                else:
                    break
            
            return duration
            
        except Exception as e:
            self.logger.error(f"Trend duration calculation error: {e}")
            return 0
    
    def _empty_supertrend_result(self) -> Dict[str, Any]:
        """Return empty SuperTrend result"""
        return {
            'str_entry': {},
            'str_exit': {},
            'trend_analysis': {},
            'trend_changes': {},
            'support_resistance': {},
            'mtf_analysis': {},
            'current_values': {},
            'timestamp': datetime.now().isoformat(),
            'error': 'Calculation failed'
        }

# Usage Example (for testing):
if __name__ == "__main__":
    # Sample usage
    calculator = SuperTrendCalculator()
    
    # Sample OHLC data
    sample_data = pd.DataFrame({
        'high': np.random.rand(100) * 100 + 1000,
        'low': np.random.rand(100) * 100 + 950,
        'close': np.random.rand(100) * 100 + 975,
        'open': np.random.rand(100) * 100 + 975,
        'volume': np.random.rand(100) * 1000 + 500
    })
    
    # Calculate SuperTrend
    result = calculator.calculate_supertrend(sample_data)
    print("SuperTrend Analysis Result:", json.dumps(result, indent=2, default=str))
    
    # Get entry signal
    entry_signal = calculator.get_entry_signal(sample_data)
    print("Entry Signal:", entry_signal)
    
    # Get exit signal for a LONG position
    exit_signal = calculator.get_exit_signal(sample_data, 'LONG')
    print("Exit Signal:", exit_signal)
    
    # Detect trend changes
    trend_change = calculator.detect_trend_changes(sample_data)
    print("Trend Changes:", trend_change)
    
    # Get SuperTrend levels
    st_levels = calculator.get_supertrend_levels(sample_data)
    print("SuperTrend Levels:", st_levels) Determine new trend direction
                new_trend = str_entry['trend'][-1]  # Use entry trend as primary
                
                # Calculate change confidence
                change_confidence = self._calculate_trend_change_confidence(
                    recent_entry_change, recent_exit_change, supertrend_result
                )
                
                return {
                    'trend_change': True,
                    'new_trend': new_trend,
                    'confidence': round(change_confidence, 4),
                    'entry_change': recent_entry_change,
                    'exit_change': recent_exit_change,
                    'bars_since_change': min(
                        recent_entry_change.get('bars_ago', 999),
                        recent_exit_change.get('bars_ago', 999)
                    ),
                    'change_strength': trend_changes.get('change_strength', 0)
                }
            else:
                return {
                    'trend_change': False,
                    'new_trend': str_entry['trend'][-1],
                    'confidence': 0.0,
                    'current_trend_duration': self._calculate_trend_duration(str_entry)
                }
                
        except Exception as e:
            self.logger.error(f"Trend change detection error: {e}")
            return {'trend_change': False, 'new_trend': 'ERROR', 'confidence': 0.0}
    
    def get_supertrend_levels(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get current SuperTrend support/resistance levels
        
        Returns:
            Current SuperTrend levels for both entry and exit
        """
        try:
            supertrend_result = self.calculate_supertrend(price_data)
            if not supertrend_result:
                return {'entry_level': 0.0, 'exit_level': 0.0, 'support': [], 'resistance': []}
            
            str_entry = supertrend_result['str_entry']
            str_exit = supertrend_result['str_exit']
            support_resistance = supertrend_result['support_resistance']
            current_price = price_data['close'].iloc[-1]
            
            return {
                'entry_level': str_entry['supertrend'][-1],
                'exit_level': str_exit['supertrend'][-1],
                'entry_trend': str_entry['trend'][-1],
                'exit_trend': str_exit['trend'][-1],
                'current_price': current_price,
                'support': support_resistance.get('support_levels', []),
                'resistance': support_resistance.get('resistance_levels', []),
                'dynamic_support': str_entry['supertrend'][-1] if str_entry['trend'][-1] == 1 else 0.0,
                'dynamic_resistance': str_entry['supertrend'][-1] if str_entry['trend'][-1] == -1 else 0.0,
                'atr_entry': str_entry['atr'][-1],
                'atr_exit': str_exit['atr'][-1]
            }
            
        except Exception as e:
            self.logger.error(f"SuperTrend levels error: {e}")
            return {'entry_level': 0.0, 'exit_level': 0.0, 'support': [], 'resistance': []}
    
    # Private Methods
    
    def _calculate_single_supertrend(self, price_data: pd.DataFrame, variant: str, 
                                   atr_period: int, atr_multiplier: float) -> Dict[str, np.ndarray]:
        """Calculate single SuperTrend variant"""
        try:
            high = price_data['high'].values
            low = price_data['low'].values
            close = price_data['close'].values
            
            # Calculate ATR
            atr = talib.ATR(high, low, close, timeperiod=atr_period)
            
            # Calculate HL2 (median price)
            hl2 = (high + low) / 2
            
            # Calculate basic upper and lower bands
            upper_band = hl2 + (atr_multiplier * atr)
            lower_band = hl2 - (atr_multiplier * atr)
            
            # Initialize arrays
            final_upper_band = np.zeros_like(upper_band)
            final_lower_band = np.zeros_like(lower_band)
            supertrend = np.zeros_like(close)
            trend = np.zeros_like(close)
            
            # Calculate final bands and SuperTrend
            for i in range(1, len(close)):
                # Final Upper Band
                if upper_band[i] < final_upper_band[i-1] or close[i-1] > final_upper_band[i-1]:
                    final_upper_band[i] = upper_band[i]
                else:
                    final_upper_band[i] = final_upper_band[i-1]
                
                # Final Lower Band
                if lower_band[i] > final_lower_band[i-1] or close[i-1] < final_lower_band[i-1]:
                    final_lower_band[i] = lower_band[i]
                else:
                    final_lower_band[i] = final_lower_band[i-1]
                
                #