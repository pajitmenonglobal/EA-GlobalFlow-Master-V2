#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Multi-Timeframe Trend Detection Engine
Advanced trend detection using multiple indicators and timeframes

Features:
- Multi-timeframe trend analysis (Major: Daily/4H, Middle: 1H/30min, Minor: 15min/5min)
- Trend strength calculation and scoring
- Trend change detection and confirmation
- Trend alignment analysis across timeframes
- Dynamic trend channels and support/resistance
- Trend momentum and exhaustion signals

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

class TrendDetector:
    """
    Multi-Timeframe Trend Detection Engine
    
    Provides comprehensive trend analysis including:
    - Major, Middle, and Minor trend identification
    - Trend strength and momentum analysis
    - Multi-timeframe trend alignment
    - Trend change detection and confirmation
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize Trend Detector"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Trend Detection Parameters
        self.params = {
            'major_trend': {
                'sma_fast': 50,          # Fast SMA for major trend
                'sma_slow': 200,         # Slow SMA for major trend
                'lookback': 100          # Lookback period for major trend
            },
            'middle_trend': {
                'sma_fast': 20,          # Fast SMA for middle trend
                'sma_slow': 50,          # Slow SMA for middle trend
                'lookback': 50           # Lookback period for middle trend
            },
            'minor_trend': {
                'sma_fast': 8,           # Fast SMA for minor trend
                'sma_slow': 21,          # Slow SMA for minor trend
                'lookback': 25           # Lookback period for minor trend
            },
            'atr_period': 14,            # ATR period for trend strength
            'momentum_period': 14        # RSI period for momentum
        }
        
        # Trend Strength Thresholds
        self.strength_thresholds = {
            'very_strong': 0.8,          # Very strong trend
            'strong': 0.6,               # Strong trend
            'moderate': 0.4,             # Moderate trend
            'weak': 0.2,                 # Weak trend
            'sideways': 0.1              # Sideways/no trend
        }
        
        # Trend Change Thresholds
        self.change_thresholds = {
            'major_change': 0.7,         # Major trend change confidence
            'middle_change': 0.6,        # Middle trend change confidence
            'minor_change': 0.5,         # Minor trend change confidence
            'confirmation_bars': 3       # Bars for trend change confirmation
        }
        
        # Timeframe mappings (for multi-timeframe analysis)
        self.timeframe_hierarchy = {
            'M1': {'major': 'H4', 'middle': 'H1', 'minor': 'M15'},
            'M5': {'major': 'D1', 'middle': 'H4', 'minor': 'H1'},
            'M15': {'major': 'D1', 'middle': 'H4', 'minor': 'H1'},
            'M30': {'major': 'W1', 'middle': 'D1', 'minor': 'H4'},
            'H1': {'major': 'W1', 'middle': 'D1', 'minor': 'H4'},
            'H4': {'major': 'MN1', 'middle': 'W1', 'minor': 'D1'},
            'D1': {'major': 'MN1', 'middle': 'W1', 'minor': 'D1'}
        }
        
        # Trend analysis cache
        self.trend_cache = {}
        self.last_calculation = {}
        
        self.logger.info("📈 Multi-Timeframe Trend Detector initialized")
    
    def detect_trends(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Detect trends across multiple timeframes
        
        Args:
            price_data: OHLC price data
            timeframe: Current chart timeframe
            
        Returns:
            Complete trend analysis across timeframes
        """
        try:
            required_bars = max(
                self.params['major_trend']['lookback'],
                self.params['middle_trend']['lookback'],
                self.params['minor_trend']['lookback']
            ) + 50
            
            if len(price_data) < required_bars:
                self.logger.warning(f"Insufficient data for trend detection: {len(price_data)} bars, need {required_bars}")
                return self._empty_trend_result()
            
            # Detect Major Trend (highest timeframe)
            major_trend = self._detect_single_trend(
                price_data, 'major', self.params['major_trend']
            )
            
            # Detect Middle Trend
            middle_trend = self._detect_single_trend(
                price_data, 'middle', self.params['middle_trend']
            )
            
            # Detect Minor Trend (current timeframe)
            minor_trend = self._detect_single_trend(
                price_data, 'minor', self.params['minor_trend']
            )
            
            # Analyze trend alignment
            trend_alignment = self._analyze_trend_alignment(major_trend, middle_trend, minor_trend)
            
            # Calculate trend strength scores
            trend_strength = self._calculate_trend_strength(
                price_data, major_trend, middle_trend, minor_trend
            )
            
            # Detect trend changes
            trend_changes = self._detect_trend_changes(
                major_trend, middle_trend, minor_trend
            )
            
            # Analyze trend momentum
            trend_momentum = self._analyze_trend_momentum(price_data, trend_strength)
            
            # Generate trend signals
            trend_signals = self._generate_trend_signals(
                trend_alignment, trend_strength, trend_changes, trend_momentum
            )
            
            result = {
                'major_trend': major_trend,
                'middle_trend': middle_trend,
                'minor_trend': minor_trend,
                'trend_alignment': trend_alignment,
                'trend_strength': trend_strength,
                'trend_changes': trend_changes,
                'trend_momentum': trend_momentum,
                'trend_signals': trend_signals,
                'current_values': self._get_current_trend_values(major_trend, middle_trend, minor_trend),
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'data_points': len(price_data)
            }
            
            # Cache result
            self.last_calculation[timeframe] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Trend detection error: {e}")
            if self.error_handler:
                self.error_handler.log_error("TREND_DETECTION_ERROR", str(e))
            return self._empty_trend_result()
    
    def get_trend_signal(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Get comprehensive trend signal
        
        Returns:
            Trend signal with direction, strength, and alignment
        """
        try:
            trend_result = self.detect_trends(price_data, timeframe)
            if not trend_result or not trend_result['trend_signals']:
                return {'signal': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            trend_signals = trend_result['trend_signals']
            trend_alignment = trend_result['trend_alignment']
            trend_strength = trend_result['trend_strength']
            
            # Determine overall trend signal
            signal_direction = self._determine_trend_signal_direction(trend_signals, trend_alignment)
            
            # Calculate signal strength
            signal_strength = self._calculate_trend_signal_strength(trend_strength, trend_alignment)
            
            # Calculate confidence
            confidence = self._calculate_trend_confidence(
                trend_alignment, trend_strength, trend_result['trend_changes']
            )
            
            return {
                'signal': signal_direction,
                'strength': round(signal_strength, 4),
                'confidence': round(confidence, 4),
                'trend_components': {
                    'major_trend': trend_result['major_trend']['direction'],
                    'middle_trend': trend_result['middle_trend']['direction'],
                    'minor_trend': trend_result['minor_trend']['direction']
                },
                'alignment_score': trend_alignment.get('alignment_score', 0),
                'trend_strength_score': trend_strength.get('overall_strength', 0),
                'momentum_score': trend_result['trend_momentum'].get('momentum_score', 0),
                'details': trend_result['current_values']
            }
            
        except Exception as e:
            self.logger.error(f"Trend signal error: {e}")
            return {'signal': 'ERROR', 'strength': 0.0, 'confidence': 0.0}
    
    def detect_trend_changes(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Detect trend changes across timeframes
        
        Returns:
            Trend change analysis with early warning signals
        """
        try:
            trend_result = self.detect_trends(price_data, timeframe)
            if not trend_result:
                return {'trend_change': False, 'timeframe': 'NONE', 'confidence': 0.0}
            
            trend_changes = trend_result['trend_changes']
            
            # Find most significant trend change
            most_significant_change = self._find_most_significant_change(trend_changes)
            
            # Early warning signals
            early_warnings = self._detect_early_trend_warnings(
                trend_result['major_trend'],
                trend_result['middle_trend'],
                trend_result['minor_trend'],
                trend_result['trend_momentum']
            )
            
            return {
                'trend_change': most_significant_change['detected'],
                'changed_timeframe': most_significant_change['timeframe'],
                'new_direction': most_significant_change['new_direction'],
                'change_confidence': most_significant_change['confidence'],
                'bars_since_change': most_significant_change['bars_since_change'],
                'early_warnings': early_warnings,
                'all_changes': trend_changes,
                'change_strength': most_significant_change['strength']
            }
            
        except Exception as e:
            self.logger.error(f"Trend change detection error: {e}")
            return {'trend_change': False, 'timeframe': 'ERROR', 'confidence': 0.0}
    
    def get_trend_levels(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Get trend-based support and resistance levels
        
        Returns:
            Trend levels and channels
        """
        try:
            trend_result = self.detect_trends(price_data, timeframe)
            if not trend_result:
                return {'trend_levels': [], 'channels': {}}
            
            # Calculate trend channels
            trend_channels = self._calculate_trend_channels(
                price_data, 
                trend_result['major_trend'],
                trend_result['middle_trend'],
                trend_result['minor_trend']
            )
            
            # Key trend levels
            trend_levels = self._extract_trend_levels(trend_channels, price_data)
            
            # Dynamic support/resistance based on trends
            dynamic_levels = self._calculate_dynamic_sr_levels(
                price_data, trend_result
            )
            
            return {
                'trend_levels': trend_levels,
                'trend_channels': trend_channels,
                'dynamic_support': dynamic_levels['support'],
                'dynamic_resistance': dynamic_levels['resistance'],
                'channel_width': trend_channels.get('channel_width', 0),
                'trend_direction': trend_result['trend_alignment']['overall_direction']
            }
            
        except Exception as e:
            self.logger.error(f"Trend levels calculation error: {e}")
            return {'trend_levels': [], 'channels': {}}
    
    # Private Methods
    
    def _detect_single_trend(self, price_data: pd.DataFrame, trend_type: str, params: Dict) -> Dict[str, Any]:
        """Detect trend for single timeframe"""
        try:
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            
            # Calculate moving averages
            sma_fast = talib.SMA(close_prices, timeperiod=params['sma_fast'])
            sma_slow = talib.SMA(close_prices, timeperiod=params['sma_slow'])
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(sma_fast, sma_slow, close_prices)
            
            # Calculate trend slope
            trend_slope = self._calculate_trend_slope(sma_fast, sma_slow, params['lookback'])
            
            # Calculate trend consistency
            trend_consistency = self._calculate_trend_consistency(
                sma_fast, sma_slow, close_prices, params['lookback']
            )
            
            # Price position relative to MAs
            price_position = self._calculate_price_position(close_prices, sma_fast, sma_slow)
            
            # Trend age (how long current trend has been active)
            trend_age = self._calculate_trend_age(sma_fast, sma_slow)
            
            return {
                'type': trend_type,
                'direction': trend_direction,
                'slope': round(trend_slope, 6),
                'consistency': round(trend_consistency, 4),
                'price_position': round(price_position, 4),
                'trend_age': trend_age,
                'sma_fast': sma_fast,
                'sma_slow': sma_slow,
                'sma_fast_value': float(sma_fast[-1]) if len(sma_fast) > 0 else 0.0,
                'sma_slow_value': float(sma_slow[-1]) if len(sma_slow) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Single trend detection error for {trend_type}: {e}")
            return {'type': trend_type, 'direction': 'UNKNOWN', 'slope': 0.0}
    
    def _determine_trend_direction(self, sma_fast: np.ndarray, sma_slow: np.ndarray, 
                                 close_prices: np.ndarray) -> str:
        """Determine trend direction"""
        try:
            if len(sma_fast) < 2 or len(sma_slow) < 2:
                return 'UNKNOWN'
            
            # Current MA relationship
            fast_above_slow = sma_fast[-1] > sma_slow[-1]
            
            # Price above both MAs
            price_above_fast = close_prices[-1] > sma_fast[-1]
            price_above_slow = close_prices[-1] > sma_slow[-1]
            
            # MA slope analysis
            fast_slope = sma_fast[-1] - sma_fast[-5] if len(sma_fast) >= 5 else 0
            slow_slope = sma_slow[-1] - sma_slow[-5] if len(sma_slow) >= 5 else 0
            
            # Determine trend
            if fast_above_slow and price_above_fast and price_above_slow:
                if fast_slope > 0 and slow_slope > 0:
                    return 'STRONG_BULLISH'
                else:
                    return 'BULLISH'
            elif not fast_above_slow and not price_above_fast and not price_above_slow:
                if fast_slope < 0 and slow_slope < 0:
                    return 'STRONG_BEARISH'
                else:
                    return 'BEARISH'
            elif fast_above_slow but not (price_above_fast and price_above_slow):
                return 'WEAK_BULLISH'
            elif not fast_above_slow but (price_above_fast or price_above_slow):
                return 'WEAK_BEARISH'
            else:
                return 'SIDEWAYS'
                
        except Exception as e:
            self.logger.error(f"Trend direction determination error: {e}")
            return 'ERROR'
    
    def _calculate_trend_slope(self, sma_fast: np.ndarray, sma_slow: np.ndarray, lookback: int) -> float:
        """Calculate trend slope"""
        try:
            if len(sma_fast) < lookback or len(sma_slow) < lookback:
                return 0.0
            
            # Calculate slopes for both MAs
            fast_slope = np.polyfit(range(lookback), sma_fast[-lookback:], 1)[0]
            slow_slope = np.polyfit(range(lookback), sma_slow[-lookback:], 1)[0]
            
            # Average slope
            avg_slope = (fast_slope + slow_slope) / 2
            
            # Normalize slope
            avg_price = np.mean(sma_fast[-lookback:])
            normalized_slope = avg_slope / avg_price if avg_price > 0 else 0.0
            
            return normalized_slope
            
        except Exception as e:
            self.logger.error(f"Trend slope calculation error: {e}")
            return 0.0
    
    def _calculate_trend_consistency(self, sma_fast: np.ndarray, sma_slow: np.ndarray,
                                   close_prices: np.ndarray, lookback: int) -> float:
        """Calculate trend consistency"""
        try:
            if len(sma_fast) < lookback:
                return 0.0
            
            # Check how consistently fast MA is above/below slow MA
            recent_fast = sma_fast[-lookback:]
            recent_slow = sma_slow[-lookback:]
            
            # Count consistent periods
            fast_above_slow = recent_fast > recent_slow
            current_direction = fast_above_slow[-1]
            
            consistent_periods = np.sum(fast_above_slow == current_direction)
            consistency = consistent_periods / lookback
            
            return consistency
            
        except Exception as e:
            self.logger.error(f"Trend consistency calculation error: {e}")
            return 0.0
    
    def _calculate_price_position(self, close_prices: np.ndarray, sma_fast: np.ndarray, 
                                sma_slow: np.ndarray) -> float:
        """Calculate price position relative to MAs"""
        try:
            if len(close_prices) < 1 or len(sma_fast) < 1 or len(sma_slow) < 1:
                return 0.0
            
            current_price = close_prices[-1]
            current_fast = sma_fast[-1]
            current_slow = sma_slow[-1]
            
            # Calculate relative position
            if current_price > max(current_fast, current_slow):
                # Price above both MAs
                distance = (current_price - max(current_fast, current_slow)) / current_price
                return min(1.0, distance * 10)  # Scale to 0-1
            elif current_price < min(current_fast, current_slow):
                # Price below both MAs
                distance = (min(current_fast, current_slow) - current_price) / current_price
                return max(-1.0, -distance * 10)  # Scale to -1-0
            else:
                # Price between MAs
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Price position calculation error: {e}")
            return 0.0
    
    def _calculate_trend_age(self, sma_fast: np.ndarray, sma_slow: np.ndarray) -> int:
        """Calculate how long current trend has been active"""
        try:
            if len(sma_fast) < 10 or len(sma_slow) < 10:
                return 0
            
            current_direction = sma_fast[-1] > sma_slow[-1]
            age = 0
            
            # Count consecutive periods with same trend direction
            for i in range(len(sma_fast) - 1, 0, -1):
                if (sma_fast[i] > sma_slow[i]) == current_direction:
                    age += 1
                else:
                    break
            
            return age
            
        except Exception as e:
            self.logger.error(f"Trend age calculation error: {e}")
            return 0
    
    def _analyze_trend_alignment(self, major_trend: Dict, middle_trend: Dict, minor_trend: Dict) -> Dict[str, Any]:
        """Analyze alignment between different timeframe trends"""
        try:
            major_dir = major_trend.get('direction', 'UNKNOWN')
            middle_dir = middle_trend.get('direction', 'UNKNOWN')
            minor_dir = minor_trend.get('direction', 'UNKNOWN')
            
            # Simplify directions for alignment analysis
            def simplify_direction(direction):
                if 'BULLISH' in direction:
                    return 'BULLISH'
                elif 'BEARISH' in direction:
                    return 'BEARISH'
                else:
                    return 'NEUTRAL'
            
            major_simple = simplify_direction(major_dir)
            middle_simple = simplify_direction(middle_dir)
            minor_simple = simplify_direction(minor_dir)
            
            # Calculate alignment score
            alignments = [major_simple, middle_simple, minor_simple]
            
            if len(set(alignments)) == 1 and alignments[0] != 'NEUTRAL':
                # Perfect alignment
                alignment_score = 1.0
                overall_direction = alignments[0]
                alignment_strength = 'PERFECT'
            elif alignments.count('BULLISH') >= 2:
                # Bullish majority
                alignment_score = 0.7
                overall_direction = 'BULLISH'
                alignment_strength = 'STRONG'
            elif alignments.count('BEARISH') >= 2:
                # Bearish majority
                alignment_score = 0.7
                overall_direction = 'BEARISH'
                alignment_strength = 'STRONG'
            elif 'NEUTRAL' in alignments:
                # Mixed with neutral
                alignment_score = 0.3
                overall_direction = 'NEUTRAL'
                alignment_strength = 'WEAK'
            else:
                # Conflicting trends
                alignment_score = 0.1
                overall_direction = 'CONFLICTED'
                alignment_strength = 'CONFLICTED'
            
            return {
                'alignment_score': round(alignment_score, 4),
                'overall_direction': overall_direction,
                'alignment_strength': alignment_strength,
                'major_direction': major_simple,
                'middle_direction': middle_simple,
                'minor_direction': minor_simple,
                'trend_agreement': {
                    'major_middle': major_simple == middle_simple,
                    'middle_minor': middle_simple == minor_simple,
                    'major_minor': major_simple == minor_simple
                }
            }
            
        except Exception as e:
            self.logger.error(f"Trend alignment analysis error: {e}")
            return {'alignment_score': 0.0, 'overall_direction': 'ERROR'}
    
    def _calculate_trend_strength(self, price_data: pd.DataFrame, major_trend: Dict,
                                middle_trend: Dict, minor_trend: Dict) -> Dict[str, Any]:
        """Calculate overall trend strength"""
        try:
            # Individual trend strengths
            major_strength = self._calculate_individual_trend_strength(major_trend)
            middle_strength = self._calculate_individual_trend_strength(middle_trend)
            minor_strength = self._calculate_individual_trend_strength(minor_trend)
            
            # Weighted overall strength (major trend has more weight)
            weights = {'major': 0.5, 'middle': 0.3, 'minor': 0.2}
            overall_strength = (
                major_strength * weights['major'] +
                middle_strength * weights['middle'] +
                minor_strength * weights['minor']
            )
            
            # Add ATR-based volatility component
            if len(price_data) >= self.params['atr_period']:
                atr = talib.ATR(
                    price_data['high'].values,
                    price_data['low'].values,
                    price_data['close'].values,
                    timeperiod=self.params['atr_period']
                )
                current_atr = atr[-1]
                avg_atr = np.mean(atr[-20:])
                volatility_factor = min(1.0, current_atr / avg_atr) if avg_atr > 0 else 1.0
            else:
                volatility_factor = 1.0
            
            # Adjust strength based on volatility
            adjusted_strength = overall_strength * (1 + (volatility_factor - 1) * 0.2)
            
            # Determine strength category
            if adjusted_strength >= self.strength_thresholds['very_strong']:
                strength_category = 'VERY_STRONG'
            elif adjusted_strength >= self.strength_thresholds['strong']:
                strength_category = 'STRONG'
            elif adjusted_strength >= self.strength_thresholds['moderate']:
                strength_category = 'MODERATE'
            elif adjusted_strength >= self.strength_thresholds['weak']:
                strength_category = 'WEAK'
            else:
                strength_category = 'SIDEWAYS'
            
            return {
                'overall_strength': round(adjusted_strength, 4),
                'strength_category': strength_category,
                'major_strength': round(major_strength, 4),
                'middle_strength': round(middle_strength, 4),
                'minor_strength': round(minor_strength, 4),
                'volatility_factor': round(volatility_factor, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return {'overall_strength': 0.0, 'strength_category': 'ERROR'}
    
    def _calculate_individual_trend_strength(self, trend: Dict) -> float:
        """Calculate strength for individual trend"""
        try:
            # Components of trend strength
            consistency = trend.get('consistency', 0.0)
            slope_abs = abs(trend.get('slope', 0.0))
            price_position_abs = abs(trend.get('price_position', 0.0))
            
            # Normalize slope (very small slopes should be considered weak)
            slope_strength = min(1.0, slope_abs * 1000)  # Scale factor for slope
            
            # Combine components
            strength = (
                consistency * 0.4 +
                slope_strength * 0.3 +
                price_position_abs * 0.3
            )
            
            return min(1.0, strength)
            
        except Exception as e:
            self.logger.error(f"Individual trend strength calculation error: {e}")
            return 0.0
    
    def _detect_trend_changes(self, major_trend: Dict, middle_trend: Dict, minor_trend: Dict) -> Dict[str, Any]:
        """Detect trend changes across timeframes"""
        try:
            changes = {
                'major_change': self._detect_single_trend_change(major_trend, 'major'),
                'middle_change': self._detect_single_trend_change(middle_trend, 'middle'),
                'minor_change': self._detect_single_trend_change(minor_trend, 'minor')
            }
            
            # Count active changes
            active_changes = sum(1 for change in changes.values() if change['detected'])
            
            # Find most significant change
            most_significant = max(
                changes.values(),
                key=lambda x: x['confidence'] if x['detected'] else 0
            )
            
            return {
                'changes': changes,
                'active_changes': active_changes,
                'most_significant': most_significant,
                'change_cascade': self._analyze_change_cascade(changes)
            }
            
        except Exception as e:
            self.logger.error(f"Trend change detection error: {e}")
            return {'changes': {}, 'active_changes': 0}
    
    def _detect_single_trend_change(self, trend: Dict, trend_type: str) -> Dict[str, Any]:
        """Detect trend change for single timeframe"""
        try:
            sma_fast = trend.get('sma_fast', np.array([]))
            sma_slow = trend.get('sma_slow', np.array([]))
            
            if len(sma_fast) < 10 or len(sma_slow) < 10:
                return {'detected': False, 'confidence': 0.0}
            
            # Check for recent MA crossover
            current_cross = sma_fast[-1] > sma_slow[-1]
            
            # Look for change in recent periods
            change_detected = False
            bars_since_change = 0
            confidence = 0.0
            
            for i in range(1, min(10, len(sma_fast))):
                prev_cross = sma_fast[-(i+1)] > sma_slow[-(i+1)]
                if prev_cross != current_cross:
                    change_detected = True
                    bars_since_change = i
                    
                    # Calculate confidence based on how clear the change is
                    ma_separation = abs(sma_fast[-1] - sma_slow[-1]) / sma_slow[-1]
                    time_factor = max(0.1, 1.0 - (i / 10))  # Recent changes are more confident
                    
                    confidence = min(1.0, ma_separation * 100 * time_factor)
                    break
            
            # Determine new direction
            new_direction = 'BULLISH' if current_cross else 'BEARISH'
            
            return {
                'detected': change_detected,
                'timeframe': trend_type,
                'new_direction': new_direction,
                'bars_since_change': bars_since_change,
                'confidence': round(confidence, 4),
                'strength': min(1.0, confidence + (trend.get('slope', 0.0) * 100))
            }
            
        except Exception as e:
            self.logger.error(f"Single trend change detection error for {trend_type}: {e}")
            return {'detected': False, 'confidence': 0.0}
    
    def _analyze_change_cascade(self, changes: Dict) -> Dict[str, Any]:
        """Analyze trend change cascade (changes flowing from higher to lower TFs)"""
        try:
            major_changed = changes.get('major_change', {}).get('detected', False)
            middle_changed = changes.get('middle_change', {}).get('detected', False)
            minor_changed = changes.get('minor_change', {}).get('detected', False)
            
            # Analyze cascade pattern
            if major_changed and middle_changed and minor_changed:
                cascade_type = 'FULL_CASCADE'
                cascade_strength = 1.0
            elif major_changed and middle_changed:
                cascade_type = 'MAJOR_TO_MIDDLE'
                cascade_strength = 0.8
            elif middle_changed and minor_changed:
                cascade_type = 'MIDDLE_TO_MINOR'
                cascade_strength = 0.6
            elif major_changed:
                cascade_type = 'MAJOR_ONLY'
                cascade_strength = 0.7
            elif middle_changed:
                cascade_type = 'MIDDLE_ONLY'
                cascade_strength = 0.5
            elif minor_changed:
                cascade_type = 'MINOR_ONLY'
                cascade_strength = 0.3
            else:
                cascade_type = 'NO_CASCADE'
                cascade_strength = 0.0
            
            return {
                'cascade_type': cascade_type,
                'cascade_strength': round(cascade_strength, 4),
                'changes_count': sum([major_changed, middle_changed, minor_changed])
            }
            
        except Exception as e:
            self.logger.error(f"Change cascade analysis error: {e}")
            return {'cascade_type': 'ERROR', 'cascade_strength': 0.0}
    
    def _analyze_trend_momentum(self, price_data: pd.DataFrame, trend_strength: Dict) -> Dict[str, Any]:
        """Analyze trend momentum"""
        try:
            if len(price_data) < self.params['momentum_period']:
                return {'momentum_score': 0.0, 'momentum_direction': 'UNKNOWN'}
            
            # Calculate RSI for momentum
            rsi = talib.RSI(price_data['close'].values, timeperiod=self.params['momentum_period'])
            current_rsi = rsi[-1]
            
            # Price momentum
            price_change = (price_data['close'].iloc[-1] - price_data['close'].iloc[-10]) / price_data['close'].iloc[-10]
            
            # Volume momentum (if available)
            volume_momentum = 0.0
            if 'volume' in price_data.columns:
                recent_volume = price_data['volume'][-5:].mean()
                avg_volume = price_data['volume'][-20:].mean()
                volume_momentum = (recent_volume / avg_volume - 1) if avg_volume > 0 else 0.0
            
            # Combined momentum score
            rsi_momentum = (current_rsi - 50) / 50  # Normalize RSI to -1 to 1
            price_momentum_norm = max(-1.0, min(1.0, price_change * 10))  # Normalize price change
            volume_momentum_norm = max(-1.0, min(1.0, volume_momentum))
            
            # Weighted momentum
            momentum_score = (
                rsi_momentum * 0.4 +
                price_momentum_norm * 0.4 +
                volume_momentum_norm * 0.2
            )
            
            # Determine momentum direction
            if momentum_score > 0.2:
                momentum_direction = 'POSITIVE'
            elif momentum_score < -0.2:
                momentum_direction = 'NEGATIVE'
            else:
                momentum_direction = 'NEUTRAL'
            
            # Momentum strength category
            momentum_abs = abs(momentum_score)
            if momentum_abs >= 0.7:
                momentum_strength = 'VERY_STRONG'
            elif momentum_abs >= 0.5:
                momentum_strength = 'STRONG'
            elif momentum_abs >= 0.3:
                momentum_strength = 'MODERATE'
            else:
                momentum_strength = 'WEAK'
            
            return {
                'momentum_score': round(momentum_score, 4),
                'momentum_direction': momentum_direction,
                'momentum_strength': momentum_strength,
                'rsi_value': round(current_rsi, 2),
                'price_momentum': round(price_momentum_norm, 4),
                'volume_momentum': round(volume_momentum_norm, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Trend momentum analysis error: {e}")
            return {'momentum_score': 0.0, 'momentum_direction': 'ERROR'}
    
    def _generate_trend_signals(self, trend_alignment: Dict, trend_strength: Dict,
                              trend_changes: Dict, trend_momentum: Dict) -> Dict[str, Any]:
        """Generate comprehensive trend signals"""
        try:
            # Base signal from alignment
            alignment_signal = self._calculate_alignment_signal(trend_alignment)
            
            # Strength-based signal adjustment
            strength_multiplier = trend_strength.get('overall_strength', 0.5)
            
            # Change-based signal (trend changes can provide early signals)
            change_signal = self._calculate_change_signal(trend_changes)
            
            # Momentum signal
            momentum_signal = trend_momentum.get('momentum_score', 0.0)
            
            # Combined signal
            combined_signal = (
                alignment_signal * 0.4 +
                change_signal * 0.3 +
                momentum_signal * 0.3
            ) * strength_multiplier
            
            # Signal direction
            if combined_signal > 0.3:
                signal_direction = 'BULLISH'
            elif combined_signal < -0.3:
                signal_direction = 'BEARISH'
            else:
                signal_direction = 'NEUTRAL'
            
            # Signal strength
            signal_strength = abs(combined_signal)
            
            # Signal quality (based on alignment and strength)
            signal_quality = (trend_alignment.get('alignment_score', 0) + strength_multiplier) / 2
            
            return {
                'signal_direction': signal_direction,
                'signal_strength': round(signal_strength, 4),
                'signal_quality': round(signal_quality, 4),
                'combined_signal': round(combined_signal, 4),
                'component_signals': {
                    'alignment_signal': round(alignment_signal, 4),
                    'change_signal': round(change_signal, 4),
                    'momentum_signal': round(momentum_signal, 4)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Trend signal generation error: {e}")
            return {'signal_direction': 'ERROR', 'signal_strength': 0.0}
    
    def _calculate_alignment_signal(self, trend_alignment: Dict) -> float:
        """Calculate signal from trend alignment"""
        try:
            overall_direction = trend_alignment.get('overall_direction', 'NEUTRAL')
            alignment_score = trend_alignment.get('alignment_score', 0.0)
            
            if overall_direction == 'BULLISH':
                return alignment_score
            elif overall_direction == 'BEARISH':
                return -alignment_score
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Alignment signal calculation error: {e}")
            return 0.0
    
    def _calculate_change_signal(self, trend_changes: Dict) -> float:
        """Calculate signal from trend changes"""
        try:
            most_significant = trend_changes.get('most_significant', {})
            
            if not most_significant.get('detected', False):
                return 0.0
            
            new_direction = most_significant.get('new_direction', 'NEUTRAL')
            confidence = most_significant.get('confidence', 0.0)
            
            if new_direction == 'BULLISH':
                return confidence * 0.5  # Scale down change signals
            elif new_direction == 'BEARISH':
                return -confidence * 0.5
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Change signal calculation error: {e}")
            return 0.0
    
    def _find_most_significant_change(self, trend_changes: Dict) -> Dict[str, Any]:
        """Find most significant trend change"""
        try:
            changes = trend_changes.get('changes', {})
            
            if not changes:
                return {'detected': False, 'timeframe': 'NONE', 'confidence': 0.0}
            
            # Find change with highest confidence
            best_change = {'detected': False, 'confidence': 0.0}
            
            for timeframe, change in changes.items():
                if change.get('detected', False) and change.get('confidence', 0) > best_change.get('confidence', 0):
                    best_change = change.copy()
                    best_change['timeframe'] = timeframe
            
            return best_change
            
        except Exception as e:
            self.logger.error(f"Most significant change finding error: {e}")
            return {'detected': False, 'timeframe': 'ERROR', 'confidence': 0.0}
    
    def _detect_early_trend_warnings(self, major_trend: Dict, middle_trend: Dict,
                                   minor_trend: Dict, trend_momentum: Dict) -> Dict[str, Any]:
        """Detect early trend change warnings"""
        try:
            warnings = []
            
            # Momentum divergence warning
            momentum_score = trend_momentum.get('momentum_score', 0.0)
            major_direction = major_trend.get('direction', 'UNKNOWN')
            
            if ('BULLISH' in major_direction and momentum_score < -0.3) or \
               ('BEARISH' in major_direction and momentum_score > 0.3):
                warnings.append({
                    'type': 'MOMENTUM_DIVERGENCE',
                    'severity': 'HIGH',
                    'description': 'Momentum diverging from major trend'
                })
            
            # Minor trend change warning
            minor_age = minor_trend.get('trend_age', 0)
            if minor_age <= 3:  # Very recent minor trend change
                warnings.append({
                    'type': 'MINOR_TREND_CHANGE',
                    'severity': 'MEDIUM',
                    'description': 'Recent minor trend change detected'
                })
            
            # Trend exhaustion warning (very old trend)
            major_age = major_trend.get('trend_age', 0)
            if major_age > 100:  # Very old trend
                warnings.append({
                    'type': 'TREND_EXHAUSTION',
                    'severity': 'MEDIUM',
                    'description': 'Major trend showing signs of exhaustion'
                })
            
            return {
                'warnings': warnings,
                'warning_count': len(warnings),
                'highest_severity': max([w['severity'] for w in warnings], default='NONE')
            }
            
        except Exception as e:
            self.logger.error(f"Early trend warning detection error: {e}")
            return {'warnings': [], 'warning_count': 0}
    
    def _calculate_trend_channels(self, price_data: pd.DataFrame, major_trend: Dict,
                                middle_trend: Dict, minor_trend: Dict) -> Dict[str, Any]:
        """Calculate trend channels"""
        try:
            if len(price_data) < 20:
                return {'channel_width': 0, 'upper_channel': 0, 'lower_channel': 0}
            
            # Use middle trend MA as baseline
            middle_ma = middle_trend.get('sma_fast', np.array([]))
            
            if len(middle_ma) < 20:
                return {'channel_width': 0, 'upper_channel': 0, 'lower_channel': 0}
            
            # Calculate ATR for channel width
            atr = talib.ATR(
                price_data['high'].values,
                price_data['low'].values,
                price_data['close'].values,
                timeperiod=14
            )
            
            current_ma = middle_ma[-1]
            current_atr = atr[-1]
            
            # Channel calculation
            channel_multiplier = 2.0  # Standard deviation equivalent
            upper_channel = current_ma + (current_atr * channel_multiplier)
            lower_channel = current_ma - (current_atr * channel_multiplier)
            channel_width = upper_channel - lower_channel
            
            return {
                'upper_channel': round(upper_channel, 5),
                'lower_channel': round(lower_channel, 5),
                'channel_width': round(channel_width, 5),
                'middle_line': round(current_ma, 5),
                'atr_value': round(current_atr, 5)
            }
            
        except Exception as e:
            self.logger.error(f"Trend channel calculation error: {e}")
            return {'channel_width': 0, 'upper_channel': 0, 'lower_channel': 0}
    
    def _extract_trend_levels(self, trend_channels: Dict, price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract key levels from trend channels"""
        try:
            levels = []
            
            upper_channel = trend_channels.get('upper_channel', 0)
            lower_channel = trend_channels.get('lower_channel', 0)
            middle_line = trend_channels.get('middle_line', 0)
            current_price = price_data['close'].iloc[-1]
            
            if upper_channel > 0 and lower_channel > 0:
                # Add channel levels
                levels.append({
                    'price': upper_channel,
                    'type': 'RESISTANCE',
                    'strength': 0.8,
                    'source': 'TREND_CHANNEL'
                })
                
                levels.append({
                    'price': lower_channel,
                    'type': 'SUPPORT',
                    'strength': 0.8,
                    'source': 'TREND_CHANNEL'
                })
                
                levels.append({
                    'price': middle_line,
                    'type': 'DYNAMIC',
                    'strength': 0.6,
                    'source': 'TREND_MIDDLE'
                })
            
            return levels
            
        except Exception as e:
            self.logger.error(f"Trend level extraction error: {e}")
            return []
    
    def _calculate_dynamic_sr_levels(self, price_data: pd.DataFrame, trend_result: Dict) -> Dict[str, List[float]]:
        """Calculate dynamic support/resistance levels"""
        try:
            support_levels = []
            resistance_levels = []
            
            # Get trend data
            major_trend = trend_result.get('major_trend', {})
            middle_trend = trend_result.get('middle_trend', {})
            
            major_ma = major_trend.get('sma_slow_value', 0)
            middle_ma = middle_trend.get('sma_fast_value', 0)
            
            overall_direction = trend_result.get('trend_alignment', {}).get('overall_direction', 'NEUTRAL')
            
            # Assign levels based on trend direction
            if overall_direction == 'BULLISH':
                if major_ma > 0:
                    support_levels.append(major_ma)
                if middle_ma > 0:
                    support_levels.append(middle_ma)
            elif overall_direction == 'BEARISH':
                if major_ma > 0:
                    resistance_levels.append(major_ma)
                if middle_ma > 0:
                    resistance_levels.append(middle_ma)
            else:
                # Neutral - MAs can act as both support and resistance
                if major_ma > 0:
                    support_levels.append(major_ma)
                    resistance_levels.append(major_ma)
            
            return {
                'support': sorted(support_levels),
                'resistance': sorted(resistance_levels, reverse=True)
            }
            
        except Exception as e:
            self.logger.error(f"Dynamic S/R calculation error: {e}")
            return {'support': [], 'resistance': []}
    
    def _get_current_trend_values(self, major_trend: Dict, middle_trend: Dict, minor_trend: Dict) -> Dict[str, Any]:
        """Get current trend values for all timeframes"""
        try:
            return {
                'major_trend_direction': major_trend.get('direction', 'UNKNOWN'),
                'major_trend_slope': major_trend.get('slope', 0.0),
                'major_sma_fast': major_trend.get('sma_fast_value', 0.0),
                'major_sma_slow': major_trend.get('sma_slow_value', 0.0),
                
                'middle_trend_direction': middle_trend.get('direction', 'UNKNOWN'),
                'middle_trend_slope': middle_trend.get('slope', 0.0),
                'middle_sma_fast': middle_trend.get('sma_fast_value', 0.0),
                'middle_sma_slow': middle_trend.get('sma_slow_value', 0.0),
                
                'minor_trend_direction': minor_trend.get('direction', 'UNKNOWN'),
                'minor_trend_slope': minor_trend.get('slope', 0.0),
                'minor_sma_fast': minor_trend.get('sma_fast_value', 0.0),
                'minor_sma_slow': minor_trend.get('sma_slow_value', 0.0),
                
                'trend_ages': {
                    'major': major_trend.get('trend_age', 0),
                    'middle': middle_trend.get('trend_age', 0),
                    'minor': minor_trend.get('trend_age', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Current trend values extraction error: {e}")
            return {}
    
    def _determine_trend_signal_direction(self, trend_signals: Dict, trend_alignment: Dict) -> str:
        """Determine overall trend signal direction"""
        try:
            signal_direction = trend_signals.get('signal_direction', 'NEUTRAL')
            signal_strength = trend_signals.get('signal_strength', 0.0)
            alignment_score = trend_alignment.get('alignment_score', 0.0)
            
            # Enhance signal based on alignment
            if signal_strength >= 0.7 and alignment_score >= 0.8:
                return f'STRONG_{signal_direction}' if signal_direction != 'NEUTRAL' else 'NEUTRAL'
            elif signal_strength >= 0.5:
                return signal_direction
            elif signal_strength >= 0.3:
                return f'WEAK_{signal_direction}' if signal_direction != 'NEUTRAL' else 'NEUTRAL'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"Trend signal direction determination error: {e}")
            return 'ERROR'
    
    def _calculate_trend_signal_strength(self, trend_strength: Dict, trend_alignment: Dict) -> float:
        """Calculate trend signal strength"""
        try:
            overall_strength = trend_strength.get('overall_strength', 0.0)
            alignment_score = trend_alignment.get('alignment_score', 0.0)
            
            # Combine strength components
            combined_strength = (overall_strength * 0.7) + (alignment_score * 0.3)
            
            return min(1.0, combined_strength)
            
        except Exception as e:
            self.logger.error(f"Trend signal strength calculation error: {e}")
            return 0.0
    
    def _calculate_trend_confidence(self, trend_alignment: Dict, trend_strength: Dict, trend_changes: Dict) -> float:
        """Calculate trend signal confidence"""
        try:
            # Base confidence from alignment
            alignment_confidence = trend_alignment.get('alignment_score', 0.0) * 0.4
            
            # Strength confidence
            strength_confidence = trend_strength.get('overall_strength', 0.0) * 0.3
            
            # Stability confidence (fewer recent changes = higher confidence)
            active_changes = trend_changes.get('active_changes', 0)
            stability_confidence = max(0.0, (3 - active_changes) / 3) * 0.3
            
            total_confidence = alignment_confidence + strength_confidence + stability_confidence
            
            return min(1.0, total_confidence)
            
        except Exception as e:
            self.logger.error(f"Trend confidence calculation error: {e}")
            return 0.0
    
    def _empty_trend_result(self) -> Dict[str, Any]:
        """Return empty trend result"""
        return {
            'major_trend': {'direction': 'UNKNOWN', 'slope': 0.0},
            'middle_trend': {'direction': 'UNKNOWN', 'slope': 0.0},
            'minor_trend': {'direction': 'UNKNOWN', 'slope': 0.0},
            'trend_alignment': {'alignment_score': 0.0, 'overall_direction': 'UNKNOWN'},
            'trend_strength': {'overall_strength': 0.0, 'strength_category': 'UNKNOWN'},
            'trend_changes': {'active_changes': 0},
            'trend_momentum': {'momentum_score': 0.0, 'momentum_direction': 'UNKNOWN'},
            'trend_signals': {'signal_direction': 'UNKNOWN', 'signal_strength': 0.0},
            'current_values': {},
            'timestamp': datetime.now().isoformat(),
            'error': 'Trend detection failed'
        }

# Usage Example (for testing):
if __name__ == "__main__":
    # Sample usage
    detector = TrendDetector()
    
    # Sample OHLC data
    sample_data = pd.DataFrame({
        'high': np.random.rand(200) * 100 + 1000,
        'low': np.random.rand(200) * 100 + 950,
        'close': np.random.rand(200) * 100 + 975,
        'open': np.random.rand(200) * 100 + 975,
        'volume': np.random.rand(200) * 1000 + 500
    })
    
    # Detect trends
    result = detector.detect_trends(sample_data)
    print("Trend Detection Result:", json.dumps(result, indent=2, default=str))
    
    # Get trend signal
    trend_signal = detector.get_trend_signal(sample_data)
    print("Trend Signal:", trend_signal)
    
    # Detect trend changes
    trend_changes = detector.detect_trend_changes(sample_data)
    print("Trend Changes:", trend_changes)
    
    # Get trend levels
    trend_levels = detector.get_trend_levels(sample_data)
    print("Trend Levels:", trend_levels)
