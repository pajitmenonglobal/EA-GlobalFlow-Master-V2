#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Bollinger Bands Analysis Engine
Advanced Bollinger Bands analysis with squeeze detection and breakout signals

Features:
- Complete Bollinger Bands calculation (Upper, Middle, Lower)
- Bollinger Band Squeeze detection
- Breakout analysis and confirmation
- Price position analysis relative to bands
- Band width analysis for volatility measurement
- Multi-timeframe band analysis
- %B and Bandwidth indicators

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

class BollingerAnalyzer:
    """
    Advanced Bollinger Bands Analysis Engine
    
    Provides comprehensive Bollinger Bands analysis including:
    - Standard Bollinger Bands calculations
    - Squeeze detection and analysis
    - Breakout pattern recognition
    - Volatility analysis through band width
    - %B calculation for price position
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize Bollinger Bands Analyzer"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Bollinger Bands Parameters (customizable)
        self.params = {
            'bb_period': 20,            # Bollinger Bands period
            'bb_deviation': 2.0,        # Standard deviation multiplier
            'squeeze_period': 20,       # Period for squeeze detection
            'squeeze_threshold': 0.1,   # Threshold for squeeze detection
            'breakout_threshold': 0.2   # Threshold for breakout confirmation
        }
        
        # Analysis thresholds
        self.thresholds = {
            'strong_squeeze': 0.05,     # Very tight bands
            'normal_squeeze': 0.1,      # Normal squeeze level
            'high_volatility': 0.3,     # High volatility threshold
            'breakout_strength': 0.15,  # Minimum breakout strength
            'trend_confirmation': 3     # Bars for trend confirmation
        }
        
        # %B levels for analysis
        self.percent_b_levels = {
            'oversold': 0.0,           # %B = 0 (price at lower band)
            'oversold_warning': 0.2,   # %B = 0.2
            'neutral_low': 0.4,        # %B = 0.4
            'neutral_high': 0.6,       # %B = 0.6
            'overbought_warning': 0.8, # %B = 0.8
            'overbought': 1.0          # %B = 1 (price at upper band)
        }
        
        # Historical data cache
        self.bb_cache = {}
        self.last_calculation = {}
        
        self.logger.info("📈 Bollinger Bands Analyzer initialized")
    
    def calculate_bollinger_bands(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Calculate complete Bollinger Bands analysis
        
        Args:
            price_data: OHLC price data
            timeframe: Chart timeframe
            
        Returns:
            Dictionary with all Bollinger Bands data and analysis
        """
        try:
            required_bars = self.params['bb_period'] + 50  # Extra buffer
            if len(price_data) < required_bars:
                self.logger.warning(f"Insufficient data for BB calculation: {len(price_data)} bars, need {required_bars}")
                return self._empty_bb_result()
            
            # Calculate basic Bollinger Bands
            bb_data = self._calculate_basic_bands(price_data['close'])
            if not bb_data:
                return self._empty_bb_result()
            
            # Calculate additional indicators
            additional_indicators = self._calculate_additional_indicators(price_data, bb_data)
            
            # Detect squeeze conditions
            squeeze_analysis = self._analyze_squeeze_conditions(bb_data, additional_indicators)
            
            # Analyze breakout patterns
            breakout_analysis = self._analyze_breakout_patterns(price_data, bb_data)
            
            # Price position analysis
            position_analysis = self._analyze_price_position(price_data, bb_data, additional_indicators)
            
            # Volatility analysis
            volatility_analysis = self._analyze_volatility(bb_data, additional_indicators)
            
            result = {
                'bb_data': bb_data,
                'additional_indicators': additional_indicators,
                'squeeze_analysis': squeeze_analysis,
                'breakout_analysis': breakout_analysis,
                'position_analysis': position_analysis,
                'volatility_analysis': volatility_analysis,
                'current_values': self._get_current_bb_values(bb_data, additional_indicators),
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'data_points': len(price_data)
            }
            
            # Cache result
            self.last_calculation[timeframe] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Bollinger Bands calculation error: {e}")
            if self.error_handler:
                self.error_handler.log_error("BB_CALC_ERROR", str(e))
            return self._empty_bb_result()
    
    def get_bb_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get Bollinger Bands trading signals
        
        Returns:
            BB signal with direction, strength, and confidence
        """
        try:
            bb_result = self.calculate_bollinger_bands(price_data)
            if not bb_result or not bb_result['bb_data']:
                return {'signal': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            squeeze_analysis = bb_result['squeeze_analysis']
            breakout_analysis = bb_result['breakout_analysis']
            position_analysis = bb_result['position_analysis']
            volatility_analysis = bb_result['volatility_analysis']
            
            # Determine signal direction
            signal_direction = self._determine_bb_signal_direction(
                squeeze_analysis, breakout_analysis, position_analysis
            )
            
            # Calculate signal strength
            signal_strength = self._calculate_bb_signal_strength(
                breakout_analysis, position_analysis, volatility_analysis
            )
            
            # Calculate confidence
            confidence = self._calculate_bb_confidence(
                squeeze_analysis, breakout_analysis, position_analysis
            )
            
            return {
                'signal': signal_direction,
                'strength': round(signal_strength, 4),
                'confidence': round(confidence, 4),
                'components': {
                    'squeeze_status': squeeze_analysis.get('squeeze_status', 'NORMAL'),
                    'breakout_direction': breakout_analysis.get('breakout_direction', 'NONE'),
                    'price_position': position_analysis.get('position_status', 'NEUTRAL'),
                    'volatility_state': volatility_analysis.get('volatility_state', 'NORMAL')
                },
                'details': bb_result['current_values'],
                'squeeze_strength': squeeze_analysis.get('squeeze_strength', 0),
                'breakout_strength': breakout_analysis.get('breakout_strength', 0)
            }
            
        except Exception as e:
            self.logger.error(f"BB signal error: {e}")
            return {'signal': 'ERROR', 'strength': 0.0, 'confidence': 0.0}
    
    def detect_bb_squeeze(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bollinger Band Squeeze conditions
        
        Returns:
            Squeeze detection results with strength and duration
        """
        try:
            bb_result = self.calculate_bollinger_bands(price_data)
            if not bb_result:
                return {'squeeze': False, 'strength': 0.0, 'duration': 0}
            
            squeeze_analysis = bb_result['squeeze_analysis']
            
            return {
                'squeeze': squeeze_analysis.get('is_squeezed', False),
                'strength': squeeze_analysis.get('squeeze_strength', 0.0),
                'duration': squeeze_analysis.get('squeeze_duration', 0),
                'squeeze_type': squeeze_analysis.get('squeeze_type', 'NONE'),
                'expected_breakout': squeeze_analysis.get('expected_breakout', 'UNKNOWN'),
                'historical_squeezes': squeeze_analysis.get('historical_count', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"BB squeeze detection error: {e}")
            return {'squeeze': False, 'strength': 0.0, 'duration': 0}
    
    def detect_bb_breakout(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Bollinger Band breakouts
        
        Returns:
            Breakout detection results
        """
        try:
            bb_result = self.calculate_bollinger_bands(price_data)
            if not bb_result:
                return {'breakout': False, 'direction': 'NONE', 'strength': 0.0}
            
            breakout_analysis = bb_result['breakout_analysis']
            
            return {
                'breakout': breakout_analysis.get('has_breakout', False),
                'direction': breakout_analysis.get('breakout_direction', 'NONE'),
                'strength': breakout_analysis.get('breakout_strength', 0.0),
                'confirmation_bars': breakout_analysis.get('confirmation_bars', 0),
                'volume_confirmation': breakout_analysis.get('volume_confirmed', False),
                'breakout_type': breakout_analysis.get('breakout_type', 'NONE'),
                'target_projection': breakout_analysis.get('target_projection', 0.0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"BB breakout detection error: {e}")
            return {'breakout': False, 'direction': 'ERROR', 'strength': 0.0}
    
    def get_bb_support_resistance(self, price_data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Get Bollinger Bands based support and resistance levels
        
        Returns:
            Support and resistance levels from BB analysis
        """
        try:
            bb_result = self.calculate_bollinger_bands(price_data)
            if not bb_result:
                return {'support': [], 'resistance': []}
            
            bb_data = bb_result['bb_data']
            current_price = price_data['close'].iloc[-1]
            
            # Current band levels
            upper_band = bb_data['upper_band'][-1]
            middle_band = bb_data['middle_band'][-1]
            lower_band = bb_data['lower_band'][-1]
            
            support_levels = []
            resistance_levels = []
            
            # Determine levels based on price position
            if current_price > upper_band:
                # Price above upper band
                resistance_levels.append(current_price * 1.01)  # Projected resistance
                support_levels.extend([upper_band, middle_band, lower_band])
            elif current_price < lower_band:
                # Price below lower band
                support_levels.append(current_price * 0.99)  # Projected support
                resistance_levels.extend([lower_band, middle_band, upper_band])
            else:
                # Price within bands
                if current_price > middle_band:
                    resistance_levels.append(upper_band)
                    support_levels.extend([middle_band, lower_band])
                else:
                    support_levels.append(lower_band)
                    resistance_levels.extend([middle_band, upper_band])
            
            # Add historical pivot levels from recent band touches
            historical_levels = self._get_historical_band_levels(bb_data, price_data)
            support_levels.extend(historical_levels['support'])
            resistance_levels.extend(historical_levels['resistance'])
            
            # Remove duplicates and sort
            support_levels = sorted(list(set([level for level in support_levels if level > 0])))
            resistance_levels = sorted(list(set([level for level in resistance_levels if level > 0])))
            
            return {
                'support': support_levels,
                'resistance': resistance_levels,
                'key_levels': {
                    'upper_band': upper_band,
                    'middle_band': middle_band,
                    'lower_band': lower_band,
                    'current_price': current_price
                },
                'band_width': upper_band - lower_band,
                'price_position_pct': (current_price - lower_band) / (upper_band - lower_band) * 100
            }
            
        except Exception as e:
            self.logger.error(f"BB support/resistance calculation error: {e}")
            return {'support': [], 'resistance': []}
    
    # Private Methods
    
    def _calculate_basic_bands(self, close_prices: pd.Series) -> Dict[str, np.ndarray]:
        """Calculate basic Bollinger Bands"""
        try:
            # Calculate Bollinger Bands using TA-Lib
            upper_band, middle_band, lower_band = talib.BBANDS(
                close_prices.values,
                timeperiod=self.params['bb_period'],
                nbdevup=self.params['bb_deviation'],
                nbdevdn=self.params['bb_deviation'],
                matype=0  # Simple Moving Average
            )
            
            return {
                'upper_band': upper_band,
                'middle_band': middle_band,
                'lower_band': lower_band
            }
            
        except Exception as e:
            self.logger.error(f"Basic bands calculation error: {e}")
            return {}
    
    def _calculate_additional_indicators(self, price_data: pd.DataFrame, bb_data: Dict) -> Dict[str, np.ndarray]:
        """Calculate additional BB indicators"""
        try:
            close_prices = price_data['close'].values
            upper_band = bb_data['upper_band']
            middle_band = bb_data['middle_band']
            lower_band = bb_data['lower_band']
            
            # %B (Percent B) - Price position within bands
            percent_b = (close_prices - lower_band) / (upper_band - lower_band)
            
            # Bandwidth - Measure of volatility
            bandwidth = (upper_band - lower_band) / middle_band
            
            # Band width ratio (current vs historical average)
            bandwidth_sma = talib.SMA(bandwidth, timeperiod=self.params['squeeze_period'])
            bandwidth_ratio = bandwidth / bandwidth_sma
            
            # Price distance from middle band
            price_distance = (close_prices - middle_band) / middle_band
            
            # Band squeeze indicator
            squeeze_indicator = self._calculate_squeeze_indicator(bandwidth)
            
            return {
                'percent_b': percent_b,
                'bandwidth': bandwidth,
                'bandwidth_ratio': bandwidth_ratio,
                'bandwidth_sma': bandwidth_sma,
                'price_distance': price_distance,
                'squeeze_indicator': squeeze_indicator
            }
            
        except Exception as e:
            self.logger.error(f"Additional indicators calculation error: {e}")
            return {}
    
    def _calculate_squeeze_indicator(self, bandwidth: np.ndarray) -> np.ndarray:
        """Calculate squeeze indicator"""
        try:
            # Normalize bandwidth to 0-1 scale for squeeze detection
            bandwidth_min = np.nanmin(bandwidth[~np.isnan(bandwidth)])
            bandwidth_max = np.nanmax(bandwidth[~np.isnan(bandwidth)])
            
            if bandwidth_max > bandwidth_min:
                normalized_bandwidth = (bandwidth - bandwidth_min) / (bandwidth_max - bandwidth_min)
                squeeze_indicator = 1.0 - normalized_bandwidth  # Inverted (1 = tight, 0 = wide)
            else:
                squeeze_indicator = np.full_like(bandwidth, 0.5)
            
            return squeeze_indicator
            
        except Exception as e:
            self.logger.error(f"Squeeze indicator calculation error: {e}")
            return np.full_like(bandwidth, 0.5)
    
    def _analyze_squeeze_conditions(self, bb_data: Dict, additional_indicators: Dict) -> Dict[str, Any]:
        """Analyze Bollinger Band squeeze conditions"""
        try:
            bandwidth = additional_indicators['bandwidth']
            squeeze_indicator = additional_indicators['squeeze_indicator']
            
            if len(bandwidth) < self.params['squeeze_period']:
                return {'is_squeezed': False, 'squeeze_strength': 0.0}
            
            # Current squeeze level
            current_squeeze = squeeze_indicator[-1]
            current_bandwidth = bandwidth[-1]
            
            # Historical comparison
            recent_bandwidth = bandwidth[-self.params['squeeze_period']:]
            bandwidth_percentile = np.percentile(recent_bandwidth[~np.isnan(recent_bandwidth)], 20)  # 20th percentile
            
            # Determine squeeze status
            is_squeezed = current_bandwidth <= bandwidth_percentile
            squeeze_strength = current_squeeze
            
            # Squeeze type classification
            if squeeze_strength >= 0.8:
                squeeze_type = 'EXTREME_SQUEEZE'
            elif squeeze_strength >= 0.6:
                squeeze_type = 'STRONG_SQUEEZE'
            elif squeeze_strength >= 0.4:
                squeeze_type = 'MODERATE_SQUEEZE'
            else:
                squeeze_type = 'NO_SQUEEZE'
            
            # Calculate squeeze duration
            squeeze_duration = self._calculate_squeeze_duration(squeeze_indicator)
            
            # Expected breakout direction (simplified)
            expected_breakout = self._predict_breakout_direction(bb_data, additional_indicators)
            
            # Historical squeeze count
            historical_count = self._count_historical_squeezes(squeeze_indicator)
            
            return {
                'is_squeezed': is_squeezed,
                'squeeze_strength': round(squeeze_strength, 4),
                'squeeze_type': squeeze_type,
                'squeeze_duration': squeeze_duration,
                'expected_breakout': expected_breakout,
                'historical_count': historical_count,
                'current_bandwidth': round(current_bandwidth, 6),
                'bandwidth_percentile': round(bandwidth_percentile, 6)
            }
            
        except Exception as e:
            self.logger.error(f"Squeeze analysis error: {e}")
            return {'is_squeezed': False, 'squeeze_strength': 0.0}
    
    def _analyze_breakout_patterns(self, price_data: pd.DataFrame, bb_data: Dict) -> Dict[str, Any]:
        """Analyze Bollinger Band breakout patterns"""
        try:
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values
            low_prices = price_data['low'].values
            upper_band = bb_data['upper_band']
            lower_band = bb_data['lower_band']
            
            if len(close_prices) < 5:
                return {'has_breakout': False, 'breakout_direction': 'NONE'}
            
            # Check for recent breakouts
            recent_closes = close_prices[-5:]
            recent_highs = high_prices[-5:]
            recent_lows = low_prices[-5:]
            recent_upper = upper_band[-5:]
            recent_lower = lower_band[-5:]
            
            # Detect breakout patterns
            breakout_result = self._detect_breakout_pattern(
                recent_closes, recent_highs, recent_lows,
                recent_upper, recent_lower
            )
            
            # Calculate breakout strength
            breakout_strength = self._calculate_breakout_strength(
                breakout_result, close_prices, upper_band, lower_band
            )
            
            # Volume confirmation (if volume data available)
            volume_confirmed = self._check_volume_confirmation(price_data, breakout_result)
            
            # Target projection
            target_projection = self._calculate_target_projection(
                breakout_result, bb_data
            )
            
            return {
                'has_breakout': breakout_result['has_breakout'],
                'breakout_direction': breakout_result['direction'],
                'breakout_type': breakout_result['type'],
                'breakout_strength': round(breakout_strength, 4),
                'confirmation_bars': breakout_result['confirmation_bars'],
                'volume_confirmed': volume_confirmed,
                'target_projection': round(target_projection, 6),
                'breakout_details': breakout_result
            }
            
        except Exception as e:
            self.logger.error(f"Breakout analysis error: {e}")
            return {'has_breakout': False, 'breakout_direction': 'ERROR'}
    
    def _analyze_price_position(self, price_data: pd.DataFrame, bb_data: Dict, additional_indicators: Dict) -> Dict[str, Any]:
        """Analyze price position relative to Bollinger Bands"""
        try:
            current_price = price_data['close'].iloc[-1]
            percent_b = additional_indicators['percent_b'][-1]
            
            # Determine position status
            if percent_b >= self.percent_b_levels['overbought']:
                position_status = 'AT_UPPER_BAND'
                signal_bias = 'BEARISH_REVERSAL'
            elif percent_b >= self.percent_b_levels['overbought_warning']:
                position_status = 'NEAR_UPPER_BAND'
                signal_bias = 'BEARISH_BIAS'
            elif percent_b >= self.percent_b_levels['neutral_high']:
                position_status = 'UPPER_NEUTRAL'
                signal_bias = 'NEUTRAL_BULLISH'
            elif percent_b >= self.percent_b_levels['neutral_low']:
                position_status = 'LOWER_NEUTRAL'
                signal_bias = 'NEUTRAL_BEARISH'
            elif percent_b >= self.percent_b_levels['oversold_warning']:
                position_status = 'NEAR_LOWER_BAND'
                signal_bias = 'BULLISH_BIAS'
            else:
                position_status = 'AT_LOWER_BAND'
                signal_bias = 'BULLISH_REVERSAL'
            
            # Calculate position strength
            position_strength = self._calculate_position_strength(percent_b)
            
            # Historical %B analysis
            historical_percent_b = percent_b if len(additional_indicators['percent_b']) > 20 else None
            extremes_analysis = self._analyze_percent_b_extremes(historical_percent_b)
            
            return {
                'position_status': position_status,
                'signal_bias': signal_bias,
                'position_strength': round(position_strength, 4),
                'percent_b': round(percent_b, 4),
                'extremes_analysis': extremes_analysis,
                'current_price': current_price,
                'upper_band': bb_data['upper_band'][-1],
                'middle_band': bb_data['middle_band'][-1],
                'lower_band': bb_data['lower_band'][-1]
            }
            
        except Exception as e:
            self.logger.error(f"Price position analysis error: {e}")
            return {'position_status': 'ERROR', 'signal_bias': 'NEUTRAL'}
    
    def _analyze_volatility(self, bb_data: Dict, additional_indicators: Dict) -> Dict[str, Any]:
        """Analyze volatility through Bollinger Bands"""
        try:
            bandwidth = additional_indicators['bandwidth']
            bandwidth_ratio = additional_indicators['bandwidth_ratio']
            
            if len(bandwidth) < 10:
                return {'volatility_state': 'UNKNOWN', 'volatility_level': 0.5}
            
            current_bandwidth = bandwidth[-1]
            current_ratio = bandwidth_ratio[-1]
            
            # Historical volatility comparison
            recent_bandwidth = bandwidth[-20:]  # Last 20 periods
            bandwidth_percentile = np.percentile(recent_bandwidth[~np.isnan(recent_bandwidth)], 50)
            
            # Determine volatility state
            if current_bandwidth <= self.thresholds['strong_squeeze']:
                volatility_state = 'EXTREMELY_LOW'
                volatility_level = 0.1
            elif current_bandwidth <= self.thresholds['normal_squeeze']:
                volatility_state = 'LOW'
                volatility_level = 0.3
            elif current_bandwidth >= self.thresholds['high_volatility']:
                volatility_state = 'HIGH'
                volatility_level = 0.8
            elif current_bandwidth >= bandwidth_percentile * 1.5:
                volatility_state = 'ABOVE_AVERAGE'
                volatility_level = 0.7
            elif current_bandwidth <= bandwidth_percentile * 0.5:
                volatility_state = 'BELOW_AVERAGE'
                volatility_level = 0.3
            else:
                volatility_state = 'NORMAL'
                volatility_level = 0.5
            
            # Volatility trend
            if len(bandwidth) >= 5:
                recent_trend = np.polyfit(range(5), bandwidth[-5:], 1)[0]
                volatility_trend = 'INCREASING' if recent_trend > 0 else 'DECREASING' if recent_trend < 0 else 'STABLE'
            else:
                volatility_trend = 'UNKNOWN'
            
            return {
                'volatility_state': volatility_state,
                'volatility_level': round(volatility_level, 4),
                'volatility_trend': volatility_trend,
                'current_bandwidth': round(current_bandwidth, 6),
                'bandwidth_ratio': round(current_ratio, 4),
                'bandwidth_percentile': round(bandwidth_percentile, 6)
            }
            
        except Exception as e:
            self.logger.error(f"Volatility analysis error: {e}")
            return {'volatility_state': 'ERROR', 'volatility_level': 0.5}
    
    def _calculate_squeeze_duration(self, squeeze_indicator: np.ndarray) -> int:
        """Calculate how long the squeeze has been active"""
        try:
            if len(squeeze_indicator) < 2:
                return 0
            
            duration = 0
            threshold = self.thresholds['normal_squeeze']
            
            # Count consecutive periods above squeeze threshold
            for i in range(len(squeeze_indicator) - 1, -1, -1):
                if squeeze_indicator[i] >= threshold:
                    duration += 1
                else:
                    break
            
            return duration
            
        except Exception as e:
            self.logger.error(f"Squeeze duration calculation error: {e}")
            return 0
    
    def _predict_breakout_direction(self, bb_data: Dict, additional_indicators: Dict) -> str:
        """Predict likely breakout direction"""
        try:
            price_distance = additional_indicators['price_distance']
            
            if len(price_distance) < 5:
                return 'UNKNOWN'
            
            # Simple trend analysis
            recent_distance = price_distance[-5:]
            trend = np.polyfit(range(5), recent_distance, 1)[0]
            
            if trend > 0.001:
                return 'BULLISH'
            elif trend < -0.001:
                return 'BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"Breakout prediction error: {e}")
            return 'UNKNOWN'
    
    def _count_historical_squeezes(self, squeeze_indicator: np.ndarray) -> int:
        """Count historical squeeze occurrences"""
        try:
            if len(squeeze_indicator) < 50:
                return 0
            
            # Look back 50 periods
            historical_data = squeeze_indicator[-50:]
            threshold = self.thresholds['normal_squeeze']
            
            # Count squeeze periods
            in_squeeze = False
            squeeze_count = 0
            
            for value in historical_data:
                if not in_squeeze and value >= threshold:
                    squeeze_count += 1
                    in_squeeze = True
                elif in_squeeze and value < threshold:
                    in_squeeze = False
            
            return squeeze_count
            
        except Exception as e:
            self.logger.error(f"Historical squeeze count error: {e}")
            return 0
    
    def _detect_breakout_pattern(self, closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
                                upper_band: np.ndarray, lower_band: np.ndarray) -> Dict[str, Any]:
        """Detect breakout patterns"""
        try:
            # Check for upper band breakout
            upper_breakout = any(closes[-3:] > upper_band[-3:]) or any(highs[-3:] > upper_band[-3:])
            
            # Check for lower band breakout
            lower_breakout = any(closes[-3:] < lower_band[-3:]) or any(lows[-3:] < lower_band[-3:])
            
            if upper_breakout:
                return {
                    'has_breakout': True,
                    'direction': 'BULLISH',
                    'type': 'UPPER_BREAKOUT',
                    'confirmation_bars': self._count_confirmation_bars(closes, upper_band, 'UPPER')
                }
            elif lower_breakout:
                return {
                    'has_breakout': True,
                    'direction': 'BEARISH',
                    'type': 'LOWER_BREAKOUT',
                    'confirmation_bars': self._count_confirmation_bars(closes, lower_band, 'LOWER')
                }
            else:
                return {
                    'has_breakout': False,
                    'direction': 'NONE',
                    'type': 'NO_BREAKOUT',
                    'confirmation_bars': 0
                }
                
        except Exception as e:
            self.logger.error(f"Breakout pattern detection error: {e}")
            return {'has_breakout': False, 'direction': 'ERROR', 'type': 'ERROR'}
    
    def _count_confirmation_bars(self, closes: np.ndarray, band: np.ndarray, direction: str) -> int:
        """Count confirmation bars for breakout"""
        try:
            confirmation_bars = 0
            
            if direction == 'UPPER':
                for i in range(len(closes) - 1, -1, -1):
                    if closes[i] > band[i]:
                        confirmation_bars += 1
                    else:
                        break
            elif direction == 'LOWER':
                for i in range(len(closes) - 1, -1, -1):
                    if closes[i] < band[i]:
                        confirmation_bars += 1
                    else:
                        break
            
            return confirmation_bars
            
        except Exception as e:
            self.logger.error(f"Confirmation bars count error: {e}")
            return 0
    
    def _calculate_breakout_strength(self, breakout_result: Dict, closes: np.ndarray,
                                   upper_band: np.ndarray, lower_band: np.ndarray) -> float:
        """Calculate breakout strength"""
        try:
            if not breakout_result['has_breakout']:
                return 0.0
            
            current_price = closes[-1]
            band_width = upper_band[-1] - lower_band[-1]
            
            if breakout_result['direction'] == 'BULLISH':
                penetration = current_price - upper_band[-1]
            else:
                penetration = lower_band[-1] - current_price
            
            if band_width > 0:
                strength = penetration / band_width
                return max(0.0, min(1.0, strength))
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Breakout strength calculation error: {e}")
            return 0.0
    
    def _check_volume_confirmation(self, price_data: pd.DataFrame, breakout_result: Dict) -> bool:
        """Check volume confirmation for breakout"""
        try:
            if 'volume' not in price_data.columns or not breakout_result['has_breakout']:
                return False
            
            # Simple volume confirmation - current volume > average
            recent_volume = price_data['volume'][-10:].mean()
            current_volume = price_data['volume'].iloc[-1]
            
            return current_volume > recent_volume * 1.2
            
        except Exception as e:
            self.logger.error(f"Volume confirmation error: {e}")
            return False
    
    def _calculate_target_projection(self, breakout_result: Dict, bb_data: Dict) -> float:
        """Calculate target projection for breakout"""
        try:
            if not breakout_result['has_breakout']:
                return 0.0
            
            # Simple projection based on band width
            upper_band = bb_data['upper_band'][-1]
            lower_band = bb_data['lower_band'][-1]
            band_width = upper_band - lower_band
            
            if breakout_result['direction'] == 'BULLISH':
                return upper_band + band_width
            else:
                return lower_band - band_width
                
        except Exception as e:
            self.logger.error(f"Target projection calculation error: {e}")
            return 0.0
    
    def _calculate_position_strength(self, percent_b: float) -> float:
        """Calculate position strength based on %B"""
        try:
            # Distance from neutral (0.5)
            distance_from_neutral = abs(percent_b - 0.5)
            
            # Strength increases as we move away from neutral
            strength = distance_from_neutral * 2  # Scale to 0-1
            
            return min(1.0, strength)
            
        except Exception as e:
            self.logger.error(f"Position strength calculation error: {e}")
            return 0.0
    
    def _analyze_percent_b_extremes(self, percent_b_history: Optional[np.ndarray]) -> Dict[str, Any]:
        """Analyze %B extremes for reversal signals"""
        try:
            if percent_b_history is None or len(percent_b_history) < 20:
                return {'extreme_readings': 0, 'reversal_probability': 0.0}
            
            # Count extreme readings in recent history
            recent_data = percent_b_history[-20:]
            extreme_high = np.sum(recent_data >= 0.9)
            extreme_low = np.sum(recent_data <= 0.1)
            
            extreme_readings = extreme_high + extreme_low
            
            # Calculate reversal probability based on extremes
            if extreme_readings >= 3:
                reversal_probability = min(0.8, extreme_readings * 0.15)
            else:
                reversal_probability = 0.2
            
            return {
                'extreme_readings': int(extreme_readings),
                'extreme_high_count': int(extreme_high),
                'extreme_low_count': int(extreme_low),
                'reversal_probability': round(reversal_probability, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Percent B extremes analysis error: {e}")
            return {'extreme_readings': 0, 'reversal_probability': 0.0}
    
    def _get_historical_band_levels(self, bb_data: Dict, price_data: pd.DataFrame) -> Dict[str, List[float]]:
        """Get historical support/resistance levels from band touches"""
        try:
            if len(price_data) < 50:
                return {'support': [], 'resistance': []}
            
            # Look back 50 periods for band touches
            recent_highs = price_data['high'][-50:].values
            recent_lows = price_data['low'][-50:].values
            recent_upper = bb_data['upper_band'][-50:]
            recent_lower = bb_data['lower_band'][-50:]
            
            support_levels = []
            resistance_levels = []
            
            # Find areas where price touched bands and reversed
            for i in range(1, len(recent_highs) - 1):
                # Upper band touches (resistance)
                if recent_highs[i] >= recent_upper[i] * 0.999:  # Close to upper band
                    if recent_highs[i] > recent_highs[i-1] and recent_highs[i] > recent_highs[i+1]:
                        resistance_levels.append(recent_upper[i])
                
                # Lower band touches (support)
                if recent_lows[i] <= recent_lower[i] * 1.001:  # Close to lower band
                    if recent_lows[i] < recent_lows[i-1] and recent_lows[i] < recent_lows[i+1]:
                        support_levels.append(recent_lower[i])
            
            return {
                'support': support_levels[-5:],  # Last 5 levels
                'resistance': resistance_levels[-5:]  # Last 5 levels
            }
            
        except Exception as e:
            self.logger.error(f"Historical levels calculation error: {e}")
            return {'support': [], 'resistance': []}
    
    def _get_current_bb_values(self, bb_data: Dict, additional_indicators: Dict) -> Dict[str, float]:
        """Get current Bollinger Bands values"""
        try:
            return {
                'upper_band': float(bb_data['upper_band'][-1]),
                'middle_band': float(bb_data['middle_band'][-1]),
                'lower_band': float(bb_data['lower_band'][-1]),
                'percent_b': float(additional_indicators['percent_b'][-1]),
                'bandwidth': float(additional_indicators['bandwidth'][-1]),
                'bandwidth_ratio': float(additional_indicators['bandwidth_ratio'][-1]),
                'squeeze_indicator': float(additional_indicators['squeeze_indicator'][-1])
            }
        except Exception as e:
            self.logger.error(f"Current values extraction error: {e}")
            return {}
    
    def _determine_bb_signal_direction(self, squeeze_analysis: Dict, breakout_analysis: Dict, position_analysis: Dict) -> str:
        """Determine overall BB signal direction"""
        try:
            # Priority: Breakout > Position > Squeeze
            if breakout_analysis.get('has_breakout', False):
                direction = breakout_analysis.get('breakout_direction', 'NONE')
                strength = breakout_analysis.get('breakout_strength', 0)
                
                if strength >= 0.7:
                    return f'STRONG_{direction}'
                elif strength >= 0.4:
                    return direction
                else:
                    return f'WEAK_{direction}'
            
            # Position-based signals
            signal_bias = position_analysis.get('signal_bias', 'NEUTRAL')
            position_strength = position_analysis.get('position_strength', 0)
            
            if 'REVERSAL' in signal_bias and position_strength >= 0.6:
                return signal_bias.replace('_REVERSAL', '')
            elif 'BIAS' in signal_bias and position_strength >= 0.4:
                return signal_bias.replace('_BIAS', '')
            
            # Squeeze-based expectation
            if squeeze_analysis.get('is_squeezed', False):
                expected = squeeze_analysis.get('expected_breakout', 'UNKNOWN')
                if expected in ['BULLISH', 'BEARISH']:
                    return f'POTENTIAL_{expected}'
            
            return 'NEUTRAL'
            
        except Exception as e:
            self.logger.error(f"BB signal direction error: {e}")
            return 'ERROR'
    
    def _calculate_bb_signal_strength(self, breakout_analysis: Dict, position_analysis: Dict, volatility_analysis: Dict) -> float:
        """Calculate BB signal strength"""
        try:
            # Combine different strength components
            breakout_strength = breakout_analysis.get('breakout_strength', 0) * 0.5
            position_strength = position_analysis.get('position_strength', 0) * 0.3
            volatility_boost = min(0.2, volatility_analysis.get('volatility_level', 0.5) * 0.2)
            
            total_strength = breakout_strength + position_strength + volatility_boost
            return min(1.0, total_strength)
            
        except Exception as e:
            self.logger.error(f"BB signal strength calculation error: {e}")
            return 0.0
    
    def _calculate_bb_confidence(self, squeeze_analysis: Dict, breakout_analysis: Dict, position_analysis: Dict) -> float:
        """Calculate BB signal confidence"""
        try:
            confidence_factors = []
            
            # Breakout confidence
            if breakout_analysis.get('has_breakout', False):
                conf_bars = breakout_analysis.get('confirmation_bars', 0)
                volume_conf = breakout_analysis.get('volume_confirmed', False)
                breakout_conf = min(1.0, (conf_bars / 3) + (0.2 if volume_conf else 0))
                confidence_factors.append(breakout_conf * 0.4)
            
            # Position confidence
            pos_strength = position_analysis.get('position_strength', 0)
            extremes_prob = position_analysis.get('extremes_analysis', {}).get('reversal_probability', 0)
            position_conf = (pos_strength + extremes_prob) / 2
            confidence_factors.append(position_conf * 0.3)
            
            # Squeeze confidence
            if squeeze_analysis.get('is_squeezed', False):
                squeeze_strength = squeeze_analysis.get('squeeze_strength', 0)
                squeeze_duration = min(1.0, squeeze_analysis.get('squeeze_duration', 0) / 10)
                squeeze_conf = (squeeze_strength + squeeze_duration) / 2
                confidence_factors.append(squeeze_conf * 0.3)
            
            return min(1.0, sum(confidence_factors))
            
        except Exception as e:
            self.logger.error(f"BB confidence calculation error: {e}")
            return 0.0
    
    def _empty_bb_result(self) -> Dict[str, Any]:
        """Return empty BB result"""
        return {
            'bb_data': {},
            'additional_indicators': {},
            'squeeze_analysis': {},
            'breakout_analysis': {},
            'position_analysis': {},
            'volatility_analysis': {},
            'current_values': {},
            'timestamp': datetime.now().isoformat(),
            'error': 'Calculation failed'
        }

# Usage Example (for testing):
if __name__ == "__main__":
    # Sample usage
    analyzer = BollingerAnalyzer()
    
    # Sample OHLC data
    sample_data = pd.DataFrame({
        'high': np.random.rand(100) * 100 + 1000,
        'low': np.random.rand(100) * 100 + 950,
        'close': np.random.rand(100) * 100 + 975,
        'open': np.random.rand(100) * 100 + 975,
        'volume': np.random.rand(100) * 1000 + 500
    })
    
    # Calculate Bollinger Bands analysis
    result = analyzer.calculate_bollinger_bands(sample_data)
    print("BB Analysis Result:", json.dumps(result, indent=2, default=str))
    
    # Get BB signal
    bb_signal = analyzer.get_bb_signal(sample_data)
    print("BB Signal:", bb_signal)
    
    # Detect squeeze
    squeeze_result = analyzer.detect_bb_squeeze(sample_data)
    print("BB Squeeze:", squeeze_result)
