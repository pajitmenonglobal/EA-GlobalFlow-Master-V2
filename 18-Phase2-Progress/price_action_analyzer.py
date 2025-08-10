#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Price Action Analysis Engine
Advanced price action analysis with candlestick patterns and market structure

Features:
- Candlestick pattern recognition (20+ patterns)
- Support/Resistance level identification
- Market structure analysis (Higher Highs, Lower Lows, etc.)
- Price action signals and confirmations
- Breakout and reversal pattern detection
- Volume-price relationship analysis

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

class PriceActionAnalyzer:
    """
    Advanced Price Action Analysis Engine
    
    Provides comprehensive price action analysis including:
    - Candlestick pattern recognition
    - Market structure analysis
    - Support/Resistance identification
    - Price action signals and confirmations
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize Price Action Analyzer"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Price Action Parameters
        self.params = {
            'pattern_lookback': 10,      # Bars to look back for patterns
            'structure_lookback': 20,    # Bars for market structure analysis
            'sr_lookback': 50,          # Bars for support/resistance
            'sr_touch_tolerance': 0.002, # Tolerance for S/R level touches
            'breakout_confirmation': 3,  # Bars for breakout confirmation
            'volume_analysis': True      # Enable volume analysis
        }
        
        # Pattern recognition thresholds
        self.pattern_thresholds = {
            'body_size_min': 0.3,       # Minimum body size for pattern validity
            'wick_size_ratio': 2.0,     # Wick to body ratio for pin bars
            'engulfing_ratio': 1.1,     # Minimum ratio for engulfing patterns
            'inside_bar_ratio': 0.95,   # Maximum ratio for inside bars
            'outside_bar_ratio': 1.05   # Minimum ratio for outside bars
        }
        
        # Signal strength thresholds
        self.signal_thresholds = {
            'strong_signal': 0.8,
            'medium_signal': 0.6,
            'weak_signal': 0.4,
            'pattern_confirmation': 0.7,
            'structure_confirmation': 0.6
        }
        
        # Candlestick patterns to recognize
        self.candlestick_patterns = [
            'HAMMER', 'INVERTED_HAMMER', 'HANGING_MAN', 'SHOOTING_STAR',
            'DOJI', 'DRAGONFLY_DOJI', 'GRAVESTONE_DOJI',
            'BULLISH_ENGULFING', 'BEARISH_ENGULFING',
            'MORNING_STAR', 'EVENING_STAR',
            'PIERCING_LINE', 'DARK_CLOUD_COVER',
            'INSIDE_BAR', 'OUTSIDE_BAR',
            'PIN_BAR_BULLISH', 'PIN_BAR_BEARISH',
            'MARUBOZU_BULLISH', 'MARUBOZU_BEARISH'
        ]
        
        # Cache for analysis results
        self.analysis_cache = {}
        self.last_calculation = {}
        
        self.logger.info("📊 Price Action Analyzer initialized")
    
    def analyze_price_action(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Perform comprehensive price action analysis
        
        Args:
            price_data: OHLC price data with volume if available
            timeframe: Chart timeframe
            
        Returns:
            Complete price action analysis results
        """
        try:
            required_bars = max(self.params.values()) + 20
            if len(price_data) < required_bars:
                self.logger.warning(f"Insufficient data for price action analysis: {len(price_data)} bars, need {required_bars}")
                return self._empty_pa_result()
            
            # Candlestick pattern recognition
            candlestick_analysis = self._analyze_candlestick_patterns(price_data)
            
            # Market structure analysis
            structure_analysis = self._analyze_market_structure(price_data)
            
            # Support/Resistance level identification
            sr_analysis = self._identify_support_resistance(price_data)
            
            # Price action signals
            signal_analysis = self._analyze_price_action_signals(
                price_data, candlestick_analysis, structure_analysis, sr_analysis
            )
            
            # Volume-price analysis (if volume available)
            volume_analysis = self._analyze_volume_price_relationship(price_data)
            
            # Breakout/Reversal pattern detection
            pattern_analysis = self._detect_breakout_reversal_patterns(
                price_data, structure_analysis, sr_analysis
            )
            
            result = {
                'candlestick_analysis': candlestick_analysis,
                'structure_analysis': structure_analysis,
                'sr_analysis': sr_analysis,
                'signal_analysis': signal_analysis,
                'volume_analysis': volume_analysis,
                'pattern_analysis': pattern_analysis,
                'current_values': self._get_current_pa_values(price_data),
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'data_points': len(price_data)
            }
            
            # Cache result
            self.last_calculation[timeframe] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Price action analysis error: {e}")
            if self.error_handler:
                self.error_handler.log_error("PA_ANALYSIS_ERROR", str(e))
            return self._empty_pa_result()
    
    def get_price_action_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get price action trading signals
        
        Returns:
            Price action signal with direction, strength, and patterns
        """
        try:
            pa_result = self.analyze_price_action(price_data)
            if not pa_result or not pa_result['signal_analysis']:
                return {'signal': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            signal_analysis = pa_result['signal_analysis']
            candlestick_analysis = pa_result['candlestick_analysis']
            structure_analysis = pa_result['structure_analysis']
            pattern_analysis = pa_result['pattern_analysis']
            
            # Determine overall signal direction
            signal_direction = self._determine_pa_signal_direction(
                signal_analysis, candlestick_analysis, structure_analysis, pattern_analysis
            )
            
            # Calculate signal strength
            signal_strength = self._calculate_pa_signal_strength(
                signal_analysis, candlestick_analysis, pattern_analysis
            )
            
            # Calculate confidence
            confidence = self._calculate_pa_confidence(
                candlestick_analysis, structure_analysis, pattern_analysis
            )
            
            return {
                'signal': signal_direction,
                'strength': round(signal_strength, 4),
                'confidence': round(confidence, 4),
                'components': {
                    'candlestick_signal': signal_analysis.get('candlestick_signal', 0),
                    'structure_signal': signal_analysis.get('structure_signal', 0),
                    'sr_signal': signal_analysis.get('sr_signal', 0),
                    'volume_signal': signal_analysis.get('volume_signal', 0)
                },
                'patterns_detected': candlestick_analysis.get('patterns_found', []),
                'key_levels': pa_result['sr_analysis'].get('key_levels', {}),
                'market_structure': structure_analysis.get('current_structure', 'UNKNOWN'),
                'details': pa_result['current_values']
            }
            
        except Exception as e:
            self.logger.error(f"Price action signal error: {e}")
            return {'signal': 'ERROR', 'strength': 0.0, 'confidence': 0.0}
    
    def detect_candlestick_patterns(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect candlestick patterns in recent price data
        
        Returns:
            Detailed candlestick pattern analysis
        """
        try:
            patterns_detected = []
            
            if len(price_data) < 5:
                return {'patterns': [], 'recent_patterns': [], 'pattern_strength': 0.0}
            
            # Check for patterns in recent bars
            for i in range(min(self.params['pattern_lookback'], len(price_data) - 3), len(price_data)):
                bar_patterns = self._identify_patterns_at_bar(price_data, i)
                if bar_patterns:
                    for pattern in bar_patterns:
                        pattern['bar_index'] = i
                        pattern['bars_ago'] = len(price_data) - 1 - i
                        patterns_detected.append(pattern)
            
            # Get most recent patterns (last 3 bars)
            recent_patterns = [p for p in patterns_detected if p['bars_ago'] <= 3]
            
            # Calculate overall pattern strength
            pattern_strength = self._calculate_pattern_strength(patterns_detected)
            
            # Pattern reliability assessment
            reliability_analysis = self._assess_pattern_reliability(
                patterns_detected, price_data
            )
            
            return {
                'patterns': patterns_detected,
                'recent_patterns': recent_patterns,
                'pattern_count': len(patterns_detected),
                'pattern_strength': round(pattern_strength, 4),
                'reliability_analysis': reliability_analysis,
                'strongest_pattern': max(patterns_detected, key=lambda x: x['strength']) if patterns_detected else None
            }
            
        except Exception as e:
            self.logger.error(f"Candlestick pattern detection error: {e}")
            return {'patterns': [], 'recent_patterns': [], 'pattern_strength': 0.0}
    
    def identify_support_resistance(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Identify key support and resistance levels
        
        Returns:
            Support/Resistance analysis with key levels
        """
        try:
            if len(price_data) < self.params['sr_lookback']:
                return {'support_levels': [], 'resistance_levels': [], 'key_levels': {}}
            
            # Identify swing highs and lows
            swing_highs = self._find_swing_points(price_data['high'], 'high')
            swing_lows = self._find_swing_points(price_data['low'], 'low')
            
            # Cluster levels to find key support/resistance
            support_levels = self._cluster_levels(swing_lows, price_data)
            resistance_levels = self._cluster_levels(swing_highs, price_data)
            
            # Identify key levels based on touches and strength
            key_levels = self._identify_key_levels(
                support_levels + resistance_levels, price_data
            )
            
            # Current price position analysis
            position_analysis = self._analyze_price_position_vs_levels(
                price_data, support_levels, resistance_levels
            )
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'key_levels': key_levels,
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'position_analysis': position_analysis,
                'level_count': len(support_levels) + len(resistance_levels)
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance identification error: {e}")
            return {'support_levels': [], 'resistance_levels': [], 'key_levels': {}}
    
    def analyze_market_structure(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze market structure (HH, HL, LH, LL patterns)
        
        Returns:
            Market structure analysis
        """
        try:
            if len(price_data) < self.params['structure_lookback']:
                return {'structure': 'UNKNOWN', 'trend': 'NEUTRAL', 'strength': 0.0}
            
            # Find significant highs and lows
            significant_highs = self._find_significant_points(price_data['high'], 'high')
            significant_lows = self._find_significant_points(price_data['low'], 'low')
            
            # Analyze structure patterns
            structure_pattern = self._analyze_structure_pattern(
                significant_highs, significant_lows, price_data
            )
            
            # Determine current trend
            trend_analysis = self._determine_structure_trend(structure_pattern)
            
            # Calculate structure strength
            structure_strength = self._calculate_structure_strength(structure_pattern)
            
            # Trend change signals
            trend_change_signals = self._detect_structure_trend_changes(
                structure_pattern, price_data
            )
            
            return {
                'current_structure': structure_pattern.get('current_pattern', 'UNKNOWN'),
                'trend_direction': trend_analysis.get('direction', 'NEUTRAL'),
                'trend_strength': trend_analysis.get('strength', 0.0),
                'structure_strength': round(structure_strength, 4),
                'significant_highs': significant_highs,
                'significant_lows': significant_lows,
                'trend_change_signals': trend_change_signals,
                'structure_pattern': structure_pattern
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return {'structure': 'ERROR', 'trend': 'ERROR', 'strength': 0.0}
    
    # Private Methods
    
    def _analyze_candlestick_patterns(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze candlestick patterns"""
        try:
            patterns_found = []
            
            if len(price_data) < 3:
                return {'patterns_found': [], 'pattern_score': 0.0}
            
            # Analyze recent bars for patterns
            lookback = min(self.params['pattern_lookback'], len(price_data) - 2)
            
            for i in range(len(price_data) - lookback, len(price_data)):
                bar_patterns = self._identify_patterns_at_bar(price_data, i)
                patterns_found.extend(bar_patterns)
            
            # Calculate overall pattern score
            pattern_score = self._calculate_overall_pattern_score(patterns_found)
            
            # Filter for most significant patterns
            significant_patterns = [p for p in patterns_found if p['strength'] >= 0.6]
            
            return {
                'patterns_found': patterns_found,
                'significant_patterns': significant_patterns,
                'pattern_count': len(patterns_found),
                'pattern_score': round(pattern_score, 4),
                'bullish_patterns': [p for p in patterns_found if p['bias'] == 'BULLISH'],
                'bearish_patterns': [p for p in patterns_found if p['bias'] == 'BEARISH']
            }
            
        except Exception as e:
            self.logger.error(f"Candlestick pattern analysis error: {e}")
            return {'patterns_found': [], 'pattern_score': 0.0}
    
    def _identify_patterns_at_bar(self, price_data: pd.DataFrame, bar_index: int) -> List[Dict[str, Any]]:
        """Identify patterns at specific bar"""
        try:
            patterns = []
            
            if bar_index < 2 or bar_index >= len(price_data):
                return patterns
            
            # Get OHLC data for current and previous bars
            current = price_data.iloc[bar_index]
            prev1 = price_data.iloc[bar_index - 1]
            prev2 = price_data.iloc[bar_index - 2] if bar_index >= 2 else None
            
            # Single bar patterns
            single_patterns = self._identify_single_bar_patterns(current, prev1)
            patterns.extend(single_patterns)
            
            # Two bar patterns
            two_bar_patterns = self._identify_two_bar_patterns(current, prev1)
            patterns.extend(two_bar_patterns)
            
            # Three bar patterns (if available)
            if prev2 is not None:
                three_bar_patterns = self._identify_three_bar_patterns(current, prev1, prev2)
                patterns.extend(three_bar_patterns)
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Pattern identification at bar {bar_index} error: {e}")
            return []
    
    def _identify_single_bar_patterns(self, current: pd.Series, prev: pd.Series) -> List[Dict[str, Any]]:
        """Identify single candlestick patterns"""
        try:
            patterns = []
            
            # Calculate candlestick properties
            body_size = abs(current['close'] - current['open'])
            total_range = current['high'] - current['low']
            upper_wick = current['high'] - max(current['open'], current['close'])
            lower_wick = min(current['open'], current['close']) - current['low']
            
            if total_range == 0:
                return patterns
            
            body_ratio = body_size / total_range
            upper_wick_ratio = upper_wick / total_range
            lower_wick_ratio = lower_wick / total_range
            
            # Doji patterns
            if body_ratio < 0.1:  # Very small body
                if upper_wick_ratio > 0.4 and lower_wick_ratio < 0.1:
                    patterns.append({
                        'name': 'GRAVESTONE_DOJI',
                        'bias': 'BEARISH',
                        'strength': 0.7,
                        'type': 'REVERSAL'
                    })
                elif lower_wick_ratio > 0.4 and upper_wick_ratio < 0.1:
                    patterns.append({
                        'name': 'DRAGONFLY_DOJI',
                        'bias': 'BULLISH',
                        'strength': 0.7,
                        'type': 'REVERSAL'
                    })
                else:
                    patterns.append({
                        'name': 'DOJI',
                        'bias': 'NEUTRAL',
                        'strength': 0.5,
                        'type': 'INDECISION'
                    })
            
            # Hammer patterns
            elif body_ratio < 0.3 and lower_wick_ratio > 0.6:
                if current['close'] > current['open']:  # Bullish hammer
                    patterns.append({
                        'name': 'HAMMER',
                        'bias': 'BULLISH',
                        'strength': 0.8,
                        'type': 'REVERSAL'
                    })
                else:  # Hanging man (in uptrend)
                    patterns.append({
                        'name': 'HANGING_MAN',
                        'bias': 'BEARISH',
                        'strength': 0.7,
                        'type': 'REVERSAL'
                    })
            
            # Shooting star / Inverted hammer
            elif body_ratio < 0.3 and upper_wick_ratio > 0.6:
                if current['close'] < current['open']:  # Bearish shooting star
                    patterns.append({
                        'name': 'SHOOTING_STAR',
                        'bias': 'BEARISH',
                        'strength': 0.8,
                        'type': 'REVERSAL'
                    })
                else:  # Inverted hammer
                    patterns.append({
                        'name': 'INVERTED_HAMMER',
                        'bias': 'BULLISH',
                        'strength': 0.7,
                        'type': 'REVERSAL'
                    })
            
            # Marubozu (very large body, small wicks)
            elif body_ratio > 0.8:
                if current['close'] > current['open']:
                    patterns.append({
                        'name': 'MARUBOZU_BULLISH',
                        'bias': 'BULLISH',
                        'strength': 0.8,
                        'type': 'CONTINUATION'
                    })
                else:
                    patterns.append({
                        'name': 'MARUBOZU_BEARISH',
                        'bias': 'BEARISH',
                        'strength': 0.8,
                        'type': 'CONTINUATION'
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Single bar pattern identification error: {e}")
            return []
    
    def _identify_two_bar_patterns(self, current: pd.Series, prev: pd.Series) -> List[Dict[str, Any]]:
        """Identify two-bar patterns"""
        try:
            patterns = []
            
            # Engulfing patterns
            prev_body = abs(prev['close'] - prev['open'])
            current_body = abs(current['close'] - current['open'])
            
            if current_body > prev_body * self.pattern_thresholds['engulfing_ratio']:
                # Bullish engulfing
                if (prev['close'] < prev['open'] and current['close'] > current['open'] and
                    current['open'] < prev['close'] and current['close'] > prev['open']):
                    patterns.append({
                        'name': 'BULLISH_ENGULFING',
                        'bias': 'BULLISH',
                        'strength': 0.9,
                        'type': 'REVERSAL'
                    })
                
                # Bearish engulfing
                elif (prev['close'] > prev['open'] and current['close'] < current['open'] and
                      current['open'] > prev['close'] and current['close'] < prev['open']):
                    patterns.append({
                        'name': 'BEARISH_ENGULFING',
                        'bias': 'BEARISH',
                        'strength': 0.9,
                        'type': 'REVERSAL'
                    })
            
            # Inside bar
            if (current['high'] < prev['high'] and current['low'] > prev['low']):
                patterns.append({
                    'name': 'INSIDE_BAR',
                    'bias': 'NEUTRAL',
                    'strength': 0.6,
                    'type': 'CONSOLIDATION'
                })
            
            # Outside bar
            elif (current['high'] > prev['high'] and current['low'] < prev['low']):
                bias = 'BULLISH' if current['close'] > current['open'] else 'BEARISH'
                patterns.append({
                    'name': 'OUTSIDE_BAR',
                    'bias': bias,
                    'strength': 0.7,
                    'type': 'BREAKOUT'
                })
            
            # Piercing Line / Dark Cloud Cover
            if prev['close'] < prev['open'] and current['close'] > current['open']:
                # Piercing line (bullish reversal)
                if (current['open'] < prev['low'] and 
                    current['close'] > (prev['open'] + prev['close']) / 2):
                    patterns.append({
                        'name': 'PIERCING_LINE',
                        'bias': 'BULLISH',
                        'strength': 0.8,
                        'type': 'REVERSAL'
                    })
            
            elif prev['close'] > prev['open'] and current['close'] < current['open']:
                # Dark cloud cover (bearish reversal)
                if (current['open'] > prev['high'] and 
                    current['close'] < (prev['open'] + prev['close']) / 2):
                    patterns.append({
                        'name': 'DARK_CLOUD_COVER',
                        'bias': 'BEARISH',
                        'strength': 0.8,
                        'type': 'REVERSAL'
                    })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Two bar pattern identification error: {e}")
            return []
    
    def _identify_three_bar_patterns(self, current: pd.Series, prev1: pd.Series, prev2: pd.Series) -> List[Dict[str, Any]]:
        """Identify three-bar patterns"""
        try:
            patterns = []
            
            # Morning Star pattern
            if (prev2['close'] < prev2['open'] and  # First bar bearish
                abs(prev1['close'] - prev1['open']) < abs(prev2['close'] - prev2['open']) * 0.3 and  # Middle bar small
                current['close'] > current['open'] and  # Third bar bullish
                current['close'] > (prev2['open'] + prev2['close']) / 2):  # Closes above midpoint
                patterns.append({
                    'name': 'MORNING_STAR',
                    'bias': 'BULLISH',
                    'strength': 0.9,
                    'type': 'REVERSAL'
                })
            
            # Evening Star pattern
            elif (prev2['close'] > prev2['open'] and  # First bar bullish
                  abs(prev1['close'] - prev1['open']) < abs(prev2['close'] - prev2['open']) * 0.3 and  # Middle bar small
                  current['close'] < current['open'] and  # Third bar bearish
                  current['close'] < (prev2['open'] + prev2['close']) / 2):  # Closes below midpoint
                patterns.append({
                    'name': 'EVENING_STAR',
                    'bias': 'BEARISH',
                    'strength': 0.9,
                    'type': 'REVERSAL'
                })
            
            return patterns
            
        except Exception as e:
            self.logger.error(f"Three bar pattern identification error: {e}")
            return []
    
    def _analyze_market_structure(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze market structure"""
        try:
            # Find swing highs and lows
            swing_highs = self._find_swing_points(price_data['high'], 'high')
            swing_lows = self._find_swing_points(price_data['low'], 'low')
            
            # Analyze structure pattern
            if len(swing_highs) >= 2 and len(swing_lows) >= 2:
                structure_analysis = self._analyze_hh_hl_lh_ll_pattern(swing_highs, swing_lows)
            else:
                structure_analysis = {'pattern': 'INSUFFICIENT_DATA', 'trend': 'UNKNOWN'}
            
            # Current structure state
            current_structure = self._determine_current_structure(
                swing_highs, swing_lows, price_data
            )
            
            return {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows,
                'structure_pattern': structure_analysis.get('pattern', 'UNKNOWN'),
                'trend_direction': structure_analysis.get('trend', 'NEUTRAL'),
                'current_structure': current_structure,
                'structure_strength': structure_analysis.get('strength', 0.0)
            }
            
        except Exception as e:
            self.logger.error(f"Market structure analysis error: {e}")
            return {'structure_pattern': 'ERROR', 'trend_direction': 'ERROR'}
    
    def _identify_support_resistance(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Identify support and resistance levels"""
        try:
            # Find pivot points
            pivot_highs = self._find_pivot_points(price_data['high'], 'high')
            pivot_lows = self._find_pivot_points(price_data['low'], 'low')
            
            # Cluster similar levels
            resistance_levels = self._cluster_price_levels(pivot_highs, price_data)
            support_levels = self._cluster_price_levels(pivot_lows, price_data)
            
            # Identify most significant levels
            key_resistance = self._rank_levels_by_significance(resistance_levels, price_data, 'resistance')
            key_support = self._rank_levels_by_significance(support_levels, price_data, 'support')
            
            return {
                'support_levels': support_levels,
                'resistance_levels': resistance_levels,
                'key_support': key_support[:5],  # Top 5 levels
                'key_resistance': key_resistance[:5],  # Top 5 levels
                'pivot_highs': pivot_highs,
                'pivot_lows': pivot_lows
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance identification error: {e}")
            return {'support_levels': [], 'resistance_levels': []}
    
    def _analyze_price_action_signals(self, price_data: pd.DataFrame, candlestick_analysis: Dict,
                                    structure_analysis: Dict, sr_analysis: Dict) -> Dict[str, Any]:
        """Analyze price action signals"""
        try:
            # Candlestick signal
            candlestick_signal = self._calculate_candlestick_signal(candlestick_analysis)
            
            # Structure signal
            structure_signal = self._calculate_structure_signal(structure_analysis)
            
            # Support/Resistance signal
            sr_signal = self._calculate_sr_signal(price_data, sr_analysis)
            
            # Volume signal (if available)
            volume_signal = self._calculate_volume_signal(price_data)
            
            # Combined signal
            combined_signal = self._combine_pa_signals(
                candlestick_signal, structure_signal, sr_signal, volume_signal
            )
            
            return {
                'candlestick_signal': candlestick_signal,
                'structure_signal': structure_signal,
                'sr_signal': sr_signal,
                'volume_signal': volume_signal,
                'combined_signal': combined_signal,
                'signal_strength': abs(combined_signal),
                'signal_direction': 'BULLISH' if combined_signal > 0 else 'BEARISH' if combined_signal < 0 else 'NEUTRAL'
            }
            
        except Exception as e:
            self.logger.error(f"Price action signal analysis error: {e}")
            return {'combined_signal': 0.0, 'signal_strength': 0.0}
    
    def _analyze_volume_price_relationship(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume-price relationship"""
        try:
            if 'volume' not in price_data.columns:
                return {'volume_available': False, 'vp_signal': 0.0}
            
            volume = price_data['volume'].values
            close = price_data['close'].values
            
            if len(volume) < 10:
                return {'volume_available': True, 'vp_signal': 0.0, 'insufficient_data': True}
            
            # Volume trend analysis
            recent_volume = volume[-5:]
            avg_volume = np.mean(volume[-20:])
            volume_ratio = np.mean(recent_volume) / avg_volume
            
            # Price-volume correlation
            recent_price_change = (close[-1] - close[-5]) / close[-5]
            volume_confirmation = 1.0 if volume_ratio > 1.2 else 0.5 if volume_ratio > 1.0 else 0.0
            
            # Volume signal
            if recent_price_change > 0 and volume_confirmation > 0.5:
                vp_signal = 0.5 * volume_confirmation
            elif recent_price_change < 0 and volume_confirmation > 0.5:
                vp_signal = -0.5 * volume_confirmation
            else:
                vp_signal = 0.0
            
            return {
                'volume_available': True,
                'vp_signal': round(vp_signal, 4),
                'volume_ratio': round(volume_ratio, 4),
                'volume_confirmation': volume_confirmation,
                'recent_volume_avg': np.mean(recent_volume),
                'historical_volume_avg': avg_volume
            }
            
        except Exception as e:
            self.logger.error(f"Volume-price analysis error: {e}")
            return {'volume_available': False, 'vp_signal': 0.0}
    
    def _detect_breakout_reversal_patterns(self, price_data: pd.DataFrame, 
                                         structure_analysis: Dict, sr_analysis: Dict) -> Dict[str, Any]:
        """Detect breakout and reversal patterns"""
        try:
            current_price = price_data['close'].iloc[-1]
            
            # Breakout detection
            breakout_analysis = self._detect_breakouts(price_data, sr_analysis)
            
            # Reversal pattern detection
            reversal_analysis = self._detect_reversal_patterns(price_data, structure_analysis)
            
            # Pattern confirmation
            pattern_confirmation = self._confirm_patterns(
                price_data, breakout_analysis, reversal_analysis
            )
            
            return {
                'breakout_analysis': breakout_analysis,
                'reversal_analysis': reversal_analysis,
                'pattern_confirmation': pattern_confirmation,
                'active_patterns': self._get_active_patterns(breakout_analysis, reversal_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Breakout/Reversal pattern detection error: {e}")
            return {'breakout_analysis': {}, 'reversal_analysis': {}}
    
    # Helper Methods for Pattern Recognition
    
    def _find_swing_points(self, data: pd.Series, point_type: str) -> List[Dict[str, Any]]:
        """Find swing highs or lows"""
        try:
            swing_points = []
            data_array = data.values
            
            if len(data_array) < 5:
                return swing_points
            
            # Find local extremes
            for i in range(2, len(data_array) - 2):
                if point_type == 'high':
                    # Check if it's a local maximum
                    if (data_array[i] > data_array[i-1] and data_array[i] > data_array[i-2] and
                        data_array[i] > data_array[i+1] and data_array[i] > data_array[i+2]):
                        swing_points.append({
                            'index': i,
                            'price': data_array[i],
                            'type': 'SWING_HIGH'
                        })
                else:  # point_type == 'low'
                    # Check if it's a local minimum
                    if (data_array[i] < data_array[i-1] and data_array[i] < data_array[i-2] and
                        data_array[i] < data_array[i+1] and data_array[i] < data_array[i+2]):
                        swing_points.append({
                            'index': i,
                            'price': data_array[i],
                            'type': 'SWING_LOW'
                        })
            
            return swing_points
            
        except Exception as e:
            self.logger.error(f"Swing point finding error: {e}")
            return []
    
    def _find_significant_points(self, data: pd.Series, point_type: str) -> List[Dict[str, Any]]:
        """Find significant highs or lows using a more relaxed criteria"""
        try:
            significant_points = []
            data_array = data.values
            
            if len(data_array) < 7:
                return significant_points
            
            # Use a wider window for significant points
            window = 3
            
            for i in range(window, len(data_array) - window):
                if point_type == 'high':
                    # Check if it's the highest in the window
                    is_highest = all(data_array[i] >= data_array[j] for j in range(i-window, i+window+1) if j != i)
                    if is_highest:
                        significant_points.append({
                            'index': i,
                            'price': data_array[i],
                            'type': 'SIGNIFICANT_HIGH'
                        })
                else:  # point_type == 'low'
                    # Check if it's the lowest in the window
                    is_lowest = all(data_array[i] <= data_array[j] for j in range(i-window, i+window+1) if j != i)
                    if is_lowest:
                        significant_points.append({
                            'index': i,
                            'price': data_array[i],
                            'type': 'SIGNIFICANT_LOW'
                        })
            
            return significant_points
            
        except Exception as e:
            self.logger.error(f"Significant point finding error: {e}")
            return []
    
    def _analyze_structure_pattern(self, highs: List[Dict], lows: List[Dict], 
                                 price_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze Higher Highs, Lower Lows pattern"""
        try:
            if len(highs) < 2 or len(lows) < 2:
                return {'current_pattern': 'INSUFFICIENT_DATA'}
            
            # Sort by index
            highs_sorted = sorted(highs, key=lambda x: x['index'])
            lows_sorted = sorted(lows, key=lambda x: x['index'])
            
            # Analyze recent highs (last 2)
            recent_highs = highs_sorted[-2:]
            higher_high = recent_highs[1]['price'] > recent_highs[0]['price']
            
            # Analyze recent lows (last 2)
            recent_lows = lows_sorted[-2:]
            higher_low = recent_lows[1]['price'] > recent_lows[0]['price']
            
            # Determine pattern
            if higher_high and higher_low:
                current_pattern = 'HIGHER_HIGHS_HIGHER_LOWS'
                trend = 'BULLISH'
            elif not higher_high and not higher_low:
                current_pattern = 'LOWER_HIGHS_LOWER_LOWS'  
                trend = 'BEARISH'
            elif higher_high and not higher_low:
                current_pattern = 'HIGHER_HIGHS_LOWER_LOWS'
                trend = 'NEUTRAL'
            else:  # not higher_high and higher_low
                current_pattern = 'LOWER_HIGHS_HIGHER_LOWS'
                trend = 'NEUTRAL'
            
            return {
                'current_pattern': current_pattern,
                'trend': trend,
                'recent_highs': recent_highs,
                'recent_lows': recent_lows,
                'higher_high': higher_high,
                'higher_low': higher_low
            }
            
        except Exception as e:
            self.logger.error(f"Structure pattern analysis error: {e}")
            return {'current_pattern': 'ERROR'}
    
    def _determine_structure_trend(self, structure_pattern: Dict) -> Dict[str, Any]:
        """Determine trend from structure pattern"""
        try:
            pattern = structure_pattern.get('current_pattern', 'UNKNOWN')
            
            if pattern == 'HIGHER_HIGHS_HIGHER_LOWS':
                return {'direction': 'BULLISH', 'strength': 0.8}
            elif pattern == 'LOWER_HIGHS_LOWER_LOWS':
                return {'direction': 'BEARISH', 'strength': 0.8}
            elif pattern in ['HIGHER_HIGHS_LOWER_LOWS', 'LOWER_HIGHS_HIGHER_LOWS']:
                return {'direction': 'NEUTRAL', 'strength': 0.3}
            else:
                return {'direction': 'UNKNOWN', 'strength': 0.0}
                
        except Exception as e:
            self.logger.error(f"Structure trend determination error: {e}")
            return {'direction': 'ERROR', 'strength': 0.0}
    
    def _calculate_structure_strength(self, structure_pattern: Dict) -> float:
        """Calculate structure strength"""
        try:
            pattern = structure_pattern.get('current_pattern', 'UNKNOWN')
            
            if pattern in ['HIGHER_HIGHS_HIGHER_LOWS', 'LOWER_HIGHS_LOWER_LOWS']:
                return 0.8  # Strong trending structure
            elif pattern in ['HIGHER_HIGHS_LOWER_LOWS', 'LOWER_HIGHS_HIGHER_LOWS']:
                return 0.4  # Neutral/consolidating structure
            else:
                return 0.0  # Unknown structure
                
        except Exception as e:
            self.logger.error(f"Structure strength calculation error: {e}")
            return 0.0
    
    def _detect_structure_trend_changes(self, structure_pattern: Dict, price_data: pd.DataFrame) -> Dict[str, Any]:
        """Detect trend change signals from structure"""
        try:
            # Simple trend change detection based on recent price action
            if len(price_data) < 10:
                return {'trend_change_signal': False, 'confidence': 0.0}
            
            # Check for potential trend change
            recent_closes = price_data['close'][-5:].values
            price_momentum = (recent_closes[-1] - recent_closes[0]) / recent_closes[0]
            
            pattern = structure_pattern.get('current_pattern', 'UNKNOWN')
            current_trend = structure_pattern.get('trend', 'NEUTRAL')
            
            # Detect divergence between price momentum and structure
            trend_change_signal = False
            confidence = 0.0
            
            if current_trend == 'BULLISH' and price_momentum < -0.02:
                trend_change_signal = True
                confidence = 0.6
            elif current_trend == 'BEARISH' and price_momentum > 0.02:
                trend_change_signal = True
                confidence = 0.6
            
            return {
                'trend_change_signal': trend_change_signal,
                'confidence': round(confidence, 4),
                'price_momentum': round(price_momentum, 6),
                'current_trend': current_trend
            }
            
        except Exception as e:
            self.logger.error(f"Trend change detection error: {e}")
            return {'trend_change_signal': False, 'confidence': 0.0}
    
    def _find_pivot_points(self, data: pd.Series, point_type: str) -> List[Dict[str, Any]]:
        """Find pivot points for support/resistance"""
        try:
            pivot_points = []
            data_array = data.values
            
            if len(data_array) < 5:
                return pivot_points
            
            # Find pivot points with 2-bar confirmation on each side
            for i in range(2, len(data_array) - 2):
                if point_type == 'high':
                    if (data_array[i] >= data_array[i-1] and data_array[i] >= data_array[i-2] and
                        data_array[i] >= data_array[i+1] and data_array[i] >= data_array[i+2]):
                        pivot_points.append({
                            'index': i,
                            'price': data_array[i],
                            'type': 'PIVOT_HIGH',
                            'strength': self._calculate_pivot_strength(data_array, i, point_type)
                        })
                else:  # point_type == 'low'
                    if (data_array[i] <= data_array[i-1] and data_array[i] <= data_array[i-2] and
                        data_array[i] <= data_array[i+1] and data_array[i] <= data_array[i+2]):
                        pivot_points.append({
                            'index': i,
                            'price': data_array[i],
                            'type': 'PIVOT_LOW',
                            'strength': self._calculate_pivot_strength(data_array, i, point_type)
                        })
            
            return pivot_points
            
        except Exception as e:
            self.logger.error(f"Pivot point finding error: {e}")
            return []
    
    def _calculate_pivot_strength(self, data_array: np.ndarray, index: int, point_type: str) -> float:
        """Calculate strength of a pivot point"""
        try:
            if index < 5 or index >= len(data_array) - 5:
                return 0.5
            
            # Look at wider range to assess strength
            left_range = data_array[index-5:index]
            right_range = data_array[index+1:index+6]
            pivot_price = data_array[index]
            
            if point_type == 'high':
                # Higher pivot = stronger resistance
                left_max = np.max(left_range)
                right_max = np.max(right_range)
                strength = min(1.0, (pivot_price - max(left_max, right_max)) / pivot_price + 0.5)
            else:  # point_type == 'low'
                # Lower pivot = stronger support
                left_min = np.min(left_range)
                right_min = np.min(right_range)
                strength = min(1.0, (min(left_min, right_min) - pivot_price) / pivot_price + 0.5)
            
            return max(0.1, strength)
            
        except Exception as e:
            self.logger.error(f"Pivot strength calculation error: {e}")
            return 0.5
    
    def _cluster_price_levels(self, pivot_points: List[Dict], price_data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Cluster similar price levels"""
        try:
            if not pivot_points:
                return []
            
            # Sort by price
            sorted_pivots = sorted(pivot_points, key=lambda x: x['price'])
            
            clustered_levels = []
            current_cluster = [sorted_pivots[0]]
            
            tolerance = self.params['sr_touch_tolerance']
            
            for i in range(1, len(sorted_pivots)):
                pivot = sorted_pivots[i]
                last_in_cluster = current_cluster[-1]
                
                # Check if pivot belongs to current cluster
                if abs(pivot['price'] - last_in_cluster['price']) / last_in_cluster['price'] <= tolerance:
                    current_cluster.append(pivot)
                else:
                    # Finalize current cluster and start new one
                    if current_cluster:
                        clustered_levels.append(self._finalize_cluster(current_cluster))
                    current_cluster = [pivot]
            
            # Don't forget the last cluster
            if current_cluster:
                clustered_levels.append(self._finalize_cluster(current_cluster))
            
            return clustered_levels
            
        except Exception as e:
            self.logger.error(f"Price level clustering error: {e}")
            return []
    
    def _finalize_cluster(self, cluster: List[Dict]) -> Dict[str, Any]:
        """Finalize a cluster of pivot points"""
        try:
            # Calculate average price and total strength
            avg_price = np.mean([p['price'] for p in cluster])
            total_strength = sum(p.get('strength', 0.5) for p in cluster)
            touch_count = len(cluster)
            
            # Most recent touch
            most_recent = max(cluster, key=lambda x: x['index'])
            
            return {
                'price': avg_price,
                'strength': min(1.0, total_strength / touch_count),
                'touch_count': touch_count,
                'most_recent_index': most_recent['index'],
                'cluster_points': cluster
            }
            
        except Exception as e:
            self.logger.error(f"Cluster finalization error: {e}")
            return {'price': 0.0, 'strength': 0.0, 'touch_count': 0}
    
    def _rank_levels_by_significance(self, levels: List[Dict], price_data: pd.DataFrame, level_type: str) -> List[Dict]:
        """Rank levels by significance"""
        try:
            if not levels:
                return []
            
            # Calculate significance score for each level
            for level in levels:
                significance = (
                    level.get('strength', 0) * 0.4 +
                    min(1.0, level.get('touch_count', 0) / 5) * 0.3 +
                    self._calculate_recency_score(level, len(price_data)) * 0.3
                )
                level['significance'] = significance
            
            # Sort by significance (descending)
            return sorted(levels, key=lambda x: x.get('significance', 0), reverse=True)
            
        except Exception as e:
            self.logger.error(f"Level ranking error: {e}")
            return levels
    
    def _calculate_recency_score(self, level: Dict, total_bars: int) -> float:
        """Calculate how recent the level is"""
        try:
            most_recent_index = level.get('most_recent_index', 0)
            bars_ago = total_bars - 1 - most_recent_index
            
            # More recent = higher score
            if bars_ago <= 5:
                return 1.0
            elif bars_ago <= 20:
                return 0.8
            elif bars_ago <= 50:
                return 0.5
            else:
                return 0.2
                
        except Exception as e:
            self.logger.error(f"Recency score calculation error: {e}")
            return 0.5
    
    # Signal Calculation Methods
    
    def _calculate_candlestick_signal(self, candlestick_analysis: Dict) -> float:
        """Calculate candlestick signal strength"""
        try:
            patterns = candlestick_analysis.get('patterns_found', [])
            
            if not patterns:
                return 0.0
            
            # Weight recent patterns more heavily
            signal = 0.0
            total_weight = 0.0
            
            for pattern in patterns:
                # Calculate weight based on strength and recency
                bars_ago = pattern.get('bars_ago', 10)
                recency_weight = max(0.1, 1.0 - (bars_ago / 10))
                pattern_weight = pattern.get('strength', 0.5) * recency_weight
                
                # Apply bias
                pattern_signal = pattern_weight if pattern.get('bias') == 'BULLISH' else -pattern_weight if pattern.get('bias') == 'BEARISH' else 0
                
                signal += pattern_signal
                total_weight += pattern_weight
            
            # Normalize
            return signal / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Candlestick signal calculation error: {e}")
            return 0.0
    
    def _calculate_structure_signal(self, structure_analysis: Dict) -> float:
        """Calculate structure signal strength"""
        try:
            trend_direction = structure_analysis.get('trend_direction', 'NEUTRAL')
            trend_strength = structure_analysis.get('trend_strength', 0.0)
            
            if trend_direction == 'BULLISH':
                return trend_strength
            elif trend_direction == 'BEARISH':
                return -trend_strength
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Structure signal calculation error: {e}")
            return 0.0
    
    def _calculate_sr_signal(self, price_data: pd.DataFrame, sr_analysis: Dict) -> float:
        """Calculate support/resistance signal"""
        try:
            if len(price_data) < 1:
                return 0.0
            
            current_price = price_data['close'].iloc[-1]
            support_levels = sr_analysis.get('key_support', [])
            resistance_levels = sr_analysis.get('key_resistance', [])
            
            signal = 0.0
            
            # Check proximity to support/resistance levels
            for support in support_levels[:3]:  # Top 3 support levels
                distance = abs(current_price - support['price']) / current_price
                if distance < self.params['sr_touch_tolerance']:
                    # Close to support - potential bounce (bullish)
                    signal += support.get('significance', 0.5) * 0.3
            
            for resistance in resistance_levels[:3]:  # Top 3 resistance levels
                distance = abs(current_price - resistance['price']) / current_price
                if distance < self.params['sr_touch_tolerance']:
                    # Close to resistance - potential rejection (bearish)
                    signal -= resistance.get('significance', 0.5) * 0.3
            
            return max(-1.0, min(1.0, signal))
            
        except Exception as e:
            self.logger.error(f"S/R signal calculation error: {e}")
            return 0.0
    
    def _calculate_volume_signal(self, price_data: pd.DataFrame) -> float:
        """Calculate volume-based signal"""
        try:
            if 'volume' not in price_data.columns or len(price_data) < 10:
                return 0.0
            
            volume = price_data['volume'].values
            close = price_data['close'].values
            
            # Volume trend analysis
            recent_volume = np.mean(volume[-3:])
            avg_volume = np.mean(volume[-20:])
            volume_ratio = recent_volume / avg_volume
            
            # Price change
            price_change = (close[-1] - close[-3]) / close[-3]
            
            # Volume confirmation signal
            if price_change > 0 and volume_ratio > 1.2:
                return min(0.5, (volume_ratio - 1.0) * 0.5)
            elif price_change < 0 and volume_ratio > 1.2:
                return max(-0.5, -(volume_ratio - 1.0) * 0.5)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Volume signal calculation error: {e}")
            return 0.0
    
    def _combine_pa_signals(self, candlestick_signal: float, structure_signal: float, 
                           sr_signal: float, volume_signal: float) -> float:
        """Combine all price action signals"""
        try:
            # Weighted combination
            weights = {
                'candlestick': 0.3,
                'structure': 0.4,
                'sr': 0.2,
                'volume': 0.1
            }
            
            combined = (
                candlestick_signal * weights['candlestick'] +
                structure_signal * weights['structure'] +
                sr_signal * weights['sr'] +
                volume_signal * weights['volume']
            )
            
            return max(-1.0, min(1.0, combined))
            
        except Exception as e:
            self.logger.error(f"Signal combination error: {e}")
            return 0.0
    
    # Additional helper methods for completeness
    
    def _calculate_overall_pattern_score(self, patterns_found: List[Dict]) -> float:
        """Calculate overall pattern score"""
        try:
            if not patterns_found:
                return 0.0
            
            total_strength = sum(p.get('strength', 0) for p in patterns_found)
            return min(1.0, total_strength / len(patterns_found))
            
        except Exception as e:
            self.logger.error(f"Overall pattern score calculation error: {e}")
            return 0.0
    
    def _calculate_pattern_strength(self, patterns_detected: List[Dict]) -> float:
        """Calculate pattern strength"""
        try:
            if not patterns_detected:
                return 0.0
            
            # Recent patterns are weighted more heavily
            weighted_strength = 0.0
            total_weight = 0.0
            
            for pattern in patterns_detected:
                bars_ago = pattern.get('bars_ago', 10)
                weight = max(0.1, 1.0 - (bars_ago / 20))
                strength = pattern.get('strength', 0.5)
                
                weighted_strength += strength * weight
                total_weight += weight
            
            return weighted_strength / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Pattern strength calculation error: {e}")
            return 0.0
    
    def _assess_pattern_reliability(self, patterns_detected: List[Dict], price_data: pd.DataFrame) -> Dict[str, Any]:
        """Assess pattern reliability"""
        try:
            if not patterns_detected:
                return {'reliability_score': 0.0, 'high_reliability_patterns': 0}
            
            high_reliability_count = 0
            total_reliability = 0.0
            
            for pattern in patterns_detected:
                # Base reliability on pattern strength and type
                base_reliability = pattern.get('strength', 0.5)
                
                # Adjust based on pattern type
                if pattern.get('type') == 'REVERSAL':
                    type_multiplier = 0.9
                elif pattern.get('type') == 'CONTINUATION':
                    type_multiplier = 0.8
                else:
                    type_multiplier = 0.6
                
                pattern_reliability = base_reliability * type_multiplier
                total_reliability += pattern_reliability
                
                if pattern_reliability >= 0.7:
                    high_reliability_count += 1
            
            avg_reliability = total_reliability / len(patterns_detected)
            
            return {
                'reliability_score': round(avg_reliability, 4),
                'high_reliability_patterns': high_reliability_count,
                'total_patterns': len(patterns_detected),
                'reliability_percentage': round((high_reliability_count / len(patterns_detected)) * 100, 2)
            }
            
        except Exception as e:
            self.logger.error(f"Pattern reliability assessment error: {e}")
            return {'reliability_score': 0.0, 'high_reliability_patterns': 0}
    
    def _detect_breakouts(self, price_data: pd.DataFrame, sr_analysis: Dict) -> Dict[str, Any]:
        """Detect breakout patterns"""
        try:
            if len(price_data) < 5:
                return {'breakout_detected': False}
            
            current_price = price_data['close'].iloc[-1]
            recent_high = price_data['high'][-3:].max()
            recent_low = price_data['low'][-3:].min()
            
            resistance_levels = sr_analysis.get('key_resistance', [])
            support_levels = sr_analysis.get('key_support', [])
            
            breakout_detected = False
            breakout_type = 'NONE'
            breakout_level = 0.0
            
            # Check for resistance breakout
            for resistance in resistance_levels[:3]:
                if recent_high > resistance['price'] * 1.001:  # Small buffer
                    breakout_detected = True
                    breakout_type = 'BULLISH_BREAKOUT'
                    breakout_level = resistance['price']
                    break
            
            # Check for support breakdown
            if not breakout_detected:
                for support in support_levels[:3]:
                    if recent_low < support['price'] * 0.999:  # Small buffer
                        breakout_detected = True
                        breakout_type = 'BEARISH_BREAKDOWN'
                        breakout_level = support['price']
                        break
            
            return {
                'breakout_detected': breakout_detected,
                'breakout_type': breakout_type,
                'breakout_level': breakout_level,
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Breakout detection error: {e}")
            return {'breakout_detected': False}
    
    def _detect_reversal_patterns(self, price_data: pd.DataFrame, structure_analysis: Dict) -> Dict[str, Any]:
        """Detect reversal patterns"""
        try:
            # Simple reversal detection based on structure
            current_structure = structure_analysis.get('current_structure', 'UNKNOWN')
            trend_direction = structure_analysis.get('trend_direction', 'NEUTRAL')
            
            # Check for potential reversal signals
            reversal_detected = False
            reversal_type = 'NONE'
            
            if len(price_data) >= 10:
                recent_price_action = price_data['close'][-5:]
                price_momentum = (recent_price_action.iloc[-1] - recent_price_action.iloc[0]) / recent_price_action.iloc[0]
                
                # Check for momentum divergence
                if trend_direction == 'BULLISH' and price_momentum < -0.02:
                    reversal_detected = True
                    reversal_type = 'BEARISH_REVERSAL'
                elif trend_direction == 'BEARISH' and price_momentum > 0.02:
                    reversal_detected = True
                    reversal_type = 'BULLISH_REVERSAL'
            
            return {
                'reversal_detected': reversal_detected,
                'reversal_type': reversal_type,
                'current_trend': trend_direction
            }
            
        except Exception as e:
            self.logger.error(f"Reversal pattern detection error: {e}")
            return {'reversal_detected': False}
    
    def _confirm_patterns(self, price_data: pd.DataFrame, breakout_analysis: Dict, reversal_analysis: Dict) -> Dict[str, Any]:
        """Confirm detected patterns"""
        try:
            confirmations = {
                'breakout_confirmed': False,
                'reversal_confirmed': False,
                'confirmation_strength': 0.0
            }
            
            if len(price_data) < 3:
                return confirmations
            
            # Volume confirmation (if available)
            volume_confirmation = False
            if 'volume' in price_data.columns:
                recent_volume = price_data['volume'][-3:].mean()
                avg_volume = price_data['volume'][-20:].mean()
                volume_confirmation = recent_volume > avg_volume * 1.2
            
            # Price confirmation
            recent_closes = price_data['close'][-3:]
            price_consistency = len([c for c in recent_closes if c == recent_closes.iloc[-1]]) >= 2
            
            # Confirm breakout
            if breakout_analysis.get('breakout_detected', False):
                confirmations['breakout_confirmed'] = volume_confirmation or price_consistency
                if confirmations['breakout_confirmed']:
                    confirmations['confirmation_strength'] += 0.5
            
            # Confirm reversal
            if reversal_analysis.get('reversal_detected', False):
                confirmations['reversal_confirmed'] = volume_confirmation or price_consistency
                if confirmations['reversal_confirmed']:
                    confirmations['confirmation_strength'] += 0.5
            
            return confirmations
            
        except Exception as e:
            self.logger.error(f"Pattern confirmation error: {e}")
            return {'breakout_confirmed': False, 'reversal_confirmed': False}
    
    def _get_active_patterns(self, breakout_analysis: Dict, reversal_analysis: Dict) -> List[str]:
        """Get list of active patterns"""
        try:
            active_patterns = []
            
            if breakout_analysis.get('breakout_detected', False):
                active_patterns.append(breakout_analysis.get('breakout_type', 'BREAKOUT'))
            
            if reversal_analysis.get('reversal_detected', False):
                active_patterns.append(reversal_analysis.get('reversal_type', 'REVERSAL'))
            
            return active_patterns
            
        except Exception as e:
            self.logger.error(f"Active patterns retrieval error: {e}")
            return []
    
    def _get_current_pa_values(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """Get current price action values"""
        try:
            if len(price_data) < 1:
                return {}
            
            current_bar = price_data.iloc[-1]
            
            return {
                'current_price': float(current_bar['close']),
                'current_high': float(current_bar['high']),
                'current_low': float(current_bar['low']),
                'current_open': float(current_bar['open']),
                'body_size': float(abs(current_bar['close'] - current_bar['open'])),
                'total_range': float(current_bar['high'] - current_bar['low']),
                'upper_wick': float(current_bar['high'] - max(current_bar['open'], current_bar['close'])),
                'lower_wick': float(min(current_bar['open'], current_bar['close']) - current_bar['low']),
                'current_volume': float(current_bar['volume']) if 'volume' in price_data.columns else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Current values extraction error: {e}")
            return {}
    
    def _determine_pa_signal_direction(self, signal_analysis: Dict, candlestick_analysis: Dict,
                                     structure_analysis: Dict, pattern_analysis: Dict) -> str:
        """Determine overall price action signal direction"""
        try:
            combined_signal = signal_analysis.get('combined_signal', 0.0)
            signal_strength = abs(combined_signal)
            
            if signal_strength >= self.signal_thresholds['strong_signal']:
                return 'STRONG_BULLISH' if combined_signal > 0 else 'STRONG_BEARISH'
            elif signal_strength >= self.signal_thresholds['medium_signal']:
                return 'BULLISH' if combined_signal > 0 else 'BEARISH'
            elif signal_strength >= self.signal_thresholds['weak_signal']:
                return 'WEAK_BULLISH' if combined_signal > 0 else 'WEAK_BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"PA signal direction error: {e}")
            return 'ERROR'
    
    def _calculate_pa_signal_strength(self, signal_analysis: Dict, candlestick_analysis: Dict, 
                                    pattern_analysis: Dict) -> float:
        """Calculate price action signal strength"""
        try:
            base_strength = abs(signal_analysis.get('combined_signal', 0.0))
            pattern_boost = min(0.2, candlestick_analysis.get('pattern_score', 0.0) * 0.2)
            
            # Boost from confirmed patterns
            breakout_confirmed = pattern_analysis.get('pattern_confirmation', {}).get('breakout_confirmed', False)
            reversal_confirmed = pattern_analysis.get('pattern_confirmation', {}).get('reversal_confirmed', False)
            
            confirmation_boost = 0.1 if (breakout_confirmed or reversal_confirmed) else 0.0
            
            total_strength = base_strength + pattern_boost + confirmation_boost
            return min(1.0, total_strength)
            
        except Exception as e:
            self.logger.error(f"PA signal strength calculation error: {e}")
            return 0.0
    
    def _calculate_pa_confidence(self, candlestick_analysis: Dict, structure_analysis: Dict, 
                               pattern_analysis: Dict) -> float:
        """Calculate price action confidence"""
        try:
            # Base confidence from pattern reliability
            pattern_reliability = candlestick_analysis.get('pattern_score', 0.0) * 0.3
            
            # Structure consistency boost
            structure_strength = structure_analysis.get('structure_strength', 0.0) * 0.4
            
            # Pattern confirmation boost
            confirmation_strength = pattern_analysis.get('pattern_confirmation', {}).get('confirmation_strength', 0.0) * 0.3
            
            confidence = pattern_reliability + structure_strength + confirmation_strength
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"PA confidence calculation error: {e}")
            return 0.0
    
    def _analyze_price_position_vs_levels(self, price_data: pd.DataFrame, 
                                        support_levels: List[Dict], resistance_levels: List[Dict]) -> Dict[str, Any]:
        """Analyze current price position relative to S/R levels"""
        try:
            if len(price_data) < 1:
                return {'position': 'UNKNOWN'}
            
            current_price = price_data['close'].iloc[-1]
            
            # Find nearest support and resistance
            nearest_support = None
            nearest_resistance = None
            
            for support in support_levels:
                if support['price'] < current_price:
                    if nearest_support is None or support['price'] > nearest_support['price']:
                        nearest_support = support
            
            for resistance in resistance_levels:
                if resistance['price'] > current_price:
                    if nearest_resistance is None or resistance['price'] < nearest_resistance['price']:
                        nearest_resistance = resistance
            
            # Determine position
            position = 'NEUTRAL'
            if nearest_support and nearest_resistance:
                support_distance = current_price - nearest_support['price']
                resistance_distance = nearest_resistance['price'] - current_price
                total_range = resistance_distance + support_distance
                
                if total_range > 0:
                    position_ratio = support_distance / total_range
                    if position_ratio < 0.3:
                        position = 'NEAR_SUPPORT'
                    elif position_ratio > 0.7:
                        position = 'NEAR_RESISTANCE'
                    else:
                        position = 'MIDDLE_RANGE'
            
            return {
                'position': position,
                'nearest_support': nearest_support,
                'nearest_resistance': nearest_resistance,
                'current_price': current_price
            }
            
        except Exception as e:
            self.logger.error(f"Price position analysis error: {e}")
            return {'position': 'ERROR'}
    
    def _determine_current_structure(self, swing_highs: List[Dict], swing_lows: List[Dict], 
                                   price_data: pd.DataFrame) -> str:
        """Determine current market structure"""
        try:
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return 'INSUFFICIENT_DATA'
            
            # Get most recent swings
            recent_highs = sorted(swing_highs, key=lambda x: x['index'])[-2:]
            recent_lows = sorted(swing_lows, key=lambda x: x['index'])[-2:]
            
            # Analyze pattern
            higher_high = recent_highs[1]['price'] > recent_highs[0]['price']
            higher_low = recent_lows[1]['price'] > recent_lows[0]['price']
            
            if higher_high and higher_low:
                return 'UPTREND'
            elif not higher_high and not higher_low:
                return 'DOWNTREND'
            else:
                return 'SIDEWAYS'
                
        except Exception as e:
            self.logger.error(f"Current structure determination error: {e}")
            return 'ERROR'
    
    def _analyze_hh_hl_lh_ll_pattern(self, swing_highs: List[Dict], swing_lows: List[Dict]) -> Dict[str, Any]:
        """Analyze Higher High/Lower Low patterns"""
        try:
            if len(swing_highs) < 2 or len(swing_lows) < 2:
                return {'pattern': 'INSUFFICIENT_DATA', 'trend': 'UNKNOWN'}
            
            # Sort by index (time)
            highs_sorted = sorted(swing_highs, key=lambda x: x['index'])
            lows_sorted = sorted(swing_lows, key=lambda x: x['index'])
            
            # Analyze last two highs and lows
            recent_highs = highs_sorted[-2:]
            recent_lows = lows_sorted[-2:]
            
            higher_high = recent_highs[1]['price'] > recent_highs[0]['price']
            higher_low = recent_lows[1]['price'] > recent_lows[0]['price']
            
            # Determine pattern and trend
            if higher_high and higher_low:
                pattern = 'HH_HL'
                trend = 'BULLISH'
                strength = 0.8
            elif not higher_high and not higher_low:
                pattern = 'LH_LL'
                trend = 'BEARISH'
                strength = 0.8
            elif higher_high and not higher_low:
                pattern = 'HH_LL'
                trend = 'NEUTRAL'
                strength = 0.4
            else:  # LH and HL
                pattern = 'LH_HL'
                trend = 'NEUTRAL'
                strength = 0.4
            
            return {
                'pattern': pattern,
                'trend': trend,
                'strength': strength,
                'higher_high': higher_high,
                'higher_low': higher_low
            }
            
        except Exception as e:
            self.logger.error(f"HH/HL/LH/LL pattern analysis error: {e}")
            return {'pattern': 'ERROR', 'trend': 'ERROR'}
    
    def _empty_pa_result(self) -> Dict[str, Any]:
        """Return empty price action result"""
        return {
            'candlestick_analysis': {},
            'structure_analysis': {},
            'sr_analysis': {},
            'signal_analysis': {},
            'volume_analysis': {},
            'pattern_analysis': {},
            'current_values': {},
            'timestamp': datetime.now().isoformat(),
            'error': 'Analysis failed'
        }

# Usage Example (for testing):
if __name__ == "__main__":
    # Sample usage
    analyzer = PriceActionAnalyzer()
    
    # Sample OHLC data
    sample_data = pd.DataFrame({
        'high': np.random.rand(100) * 100 + 1000,
        'low': np.random.rand(100) * 100 + 950,
        'close': np.random.rand(100) * 100 + 975,
        'open': np.random.rand(100) * 100 + 975,
        'volume': np.random.rand(100) * 1000 + 500
    })
    
    # Perform price action analysis
    result = analyzer.analyze_price_action(sample_data)
    print("Price Action Analysis Result:", json.dumps(result, indent=2, default=str))
    
    # Get price action signal
    pa_signal = analyzer.get_price_action_signal(sample_data)
    print("Price Action Signal:", pa_signal)
    
    # Detect candlestick patterns
    patterns = analyzer.detect_candlestick_patterns(sample_data)
    print("Candlestick Patterns:", patterns)
    
    # Identify support/resistance
    sr_levels = analyzer.identify_support_resistance(sample_data)
    print("Support/Resistance Levels:", sr_levels)
    
    # Analyze market structure
    structure = analyzer.analyze_market_structure(sample_data)
    print("Market Structure:", structure)
                '