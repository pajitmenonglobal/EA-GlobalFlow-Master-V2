#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Ichimoku Kinko Hyo Analysis Engine
Advanced Ichimoku analysis for trend detection and signal generation

Features:
- Complete Ichimoku calculations (Tenkan, Kijun, Senkou Span A/B, Chikou)
- Kumo (cloud) analysis and trend detection
- Multi-timeframe Ichimoku analysis
- Signal strength scoring
- Cloud breakout detection
- Price action relative to cloud

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

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class IchimokuAnalyzer:
    """
    Advanced Ichimoku Kinko Hyo Analysis Engine
    
    Provides comprehensive Ichimoku analysis including:
    - Standard Ichimoku calculations
    - Cloud analysis and trend detection
    - Multi-timeframe analysis
    - Signal generation and scoring
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize Ichimoku Analyzer"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Ichimoku Parameters (customizable)
        self.params = {
            'tenkan_period': 9,      # Tenkan-sen (Conversion Line)
            'kijun_period': 26,      # Kijun-sen (Base Line)
            'senkou_b_period': 52,   # Senkou Span B period
            'displacement': 26       # Displacement for Senkou spans
        }
        
        # Analysis thresholds
        self.thresholds = {
            'strong_signal': 0.8,
            'medium_signal': 0.6,
            'weak_signal': 0.4,
            'cloud_thickness_min': 0.0005,  # Minimum cloud thickness
            'price_distance_threshold': 0.002  # Price distance from cloud
        }
        
        # Historical data cache
        self.data_cache = {}
        self.last_calculation = {}
        
        self.logger.info("🔹 Ichimoku Analyzer initialized")
    
    def calculate_ichimoku(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Calculate complete Ichimoku indicators
        
        Args:
            price_data: OHLC price data
            timeframe: Chart timeframe
            
        Returns:
            Dictionary with all Ichimoku values and analysis
        """
        try:
            if len(price_data) < max(self.params.values()) + self.params['displacement']:
                self.logger.warning(f"Insufficient data for Ichimoku calculation: {len(price_data)} bars")
                return self._empty_ichimoku_result()
            
            # Calculate basic Ichimoku lines
            ichimoku_data = self._calculate_basic_lines(price_data)
            
            # Calculate cloud (Kumo)
            cloud_data = self._calculate_cloud(ichimoku_data)
            
            # Analyze current market state
            analysis = self._analyze_ichimoku_signals(price_data, ichimoku_data, cloud_data)
            
            # Multi-timeframe analysis
            mtf_analysis = self._multi_timeframe_analysis(price_data, timeframe)
            
            result = {
                'ichimoku_data': ichimoku_data,
                'cloud_data': cloud_data,
                'analysis': analysis,
                'mtf_analysis': mtf_analysis,
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'data_points': len(price_data)
            }
            
            # Cache result
            self.last_calculation[timeframe] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ichimoku calculation error: {e}")
            if self.error_handler:
                self.error_handler.log_error("ICHIMOKU_CALC_ERROR", str(e))
            return self._empty_ichimoku_result()
    
    def get_trend_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get Ichimoku trend signals
        
        Returns:
            Trend signal with strength and direction
        """
        try:
            ichimoku_result = self.calculate_ichimoku(price_data)
            if not ichimoku_result or not ichimoku_result['analysis']:
                return {'signal': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            analysis = ichimoku_result['analysis']
            
            # Determine trend direction
            trend_direction = self._determine_trend_direction(analysis)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(analysis)
            
            # Calculate confidence
            confidence = self._calculate_confidence(analysis, ichimoku_result['mtf_analysis'])
            
            return {
                'signal': trend_direction,
                'strength': round(signal_strength, 4),
                'confidence': round(confidence, 4),
                'components': {
                    'tenkan_kijun_cross': analysis.get('tenkan_kijun_signal', 0),
                    'price_cloud_position': analysis.get('price_cloud_signal', 0),
                    'chikou_confirmation': analysis.get('chikou_signal', 0),
                    'cloud_color': analysis.get('cloud_signal', 0)
                },
                'details': {
                    'current_price': analysis.get('current_price', 0),
                    'tenkan_sen': analysis.get('tenkan_sen', 0),
                    'kijun_sen': analysis.get('kijun_sen', 0),
                    'cloud_top': analysis.get('cloud_top', 0),
                    'cloud_bottom': analysis.get('cloud_bottom', 0)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Trend signal error: {e}")
            return {'signal': 'ERROR', 'strength': 0.0, 'confidence': 0.0}
    
    def detect_kumo_breakout(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect Kumo (cloud) breakouts
        
        Returns:
            Breakout detection result
        """
        try:
            if len(price_data) < 5:
                return {'breakout': False, 'direction': 'NONE', 'strength': 0.0}
            
            ichimoku_result = self.calculate_ichimoku(price_data)
            if not ichimoku_result:
                return {'breakout': False, 'direction': 'NONE', 'strength': 0.0}
            
            cloud_data = ichimoku_result['cloud_data']
            current_price = price_data['close'].iloc[-1]
            
            # Get recent cloud data
            recent_cloud_top = cloud_data['senkou_span_a'][-5:]
            recent_cloud_bottom = cloud_data['senkou_span_b'][-5:]
            recent_prices = price_data['close'][-5:]
            
            # Detect breakout patterns
            breakout_result = self._analyze_breakout_pattern(
                recent_prices, recent_cloud_top, recent_cloud_bottom
            )
            
            # Calculate breakout strength
            strength = self._calculate_breakout_strength(
                current_price, cloud_data, breakout_result
            )
            
            return {
                'breakout': breakout_result['has_breakout'],
                'direction': breakout_result['direction'],
                'strength': round(strength, 4),
                'cloud_thickness': breakout_result.get('cloud_thickness', 0),
                'price_penetration': breakout_result.get('penetration_depth', 0),
                'confirmation_bars': breakout_result.get('confirmation_bars', 0),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Kumo breakout detection error: {e}")
            return {'breakout': False, 'direction': 'ERROR', 'strength': 0.0}
    
    def get_support_resistance_levels(self, price_data: pd.DataFrame) -> Dict[str, List[float]]:
        """
        Get Ichimoku-based support and resistance levels
        
        Returns:
            Support and resistance levels
        """
        try:
            ichimoku_result = self.calculate_ichimoku(price_data)
            if not ichimoku_result:
                return {'support': [], 'resistance': []}
            
            ichimoku_data = ichimoku_result['ichimoku_data']
            cloud_data = ichimoku_result['cloud_data']
            
            support_levels = []
            resistance_levels = []
            
            # Current values
            current_tenkan = ichimoku_data['tenkan_sen'][-1]
            current_kijun = ichimoku_data['kijun_sen'][-1]
            current_cloud_top = max(cloud_data['senkou_span_a'][-1], cloud_data['senkou_span_b'][-1])
            current_cloud_bottom = min(cloud_data['senkou_span_a'][-1], cloud_data['senkou_span_b'][-1])
            current_price = price_data['close'].iloc[-1]
            
            # Determine levels based on price position
            if current_price > current_cloud_top:
                # Price above cloud - cloud acts as support
                support_levels.extend([current_cloud_top, current_cloud_bottom])
                resistance_levels.extend([current_tenkan, current_kijun])
            elif current_price < current_cloud_bottom:
                # Price below cloud - cloud acts as resistance
                resistance_levels.extend([current_cloud_top, current_cloud_bottom])
                support_levels.extend([current_tenkan, current_kijun])
            else:
                # Price inside cloud
                support_levels.append(current_cloud_bottom)
                resistance_levels.append(current_cloud_top)
            
            # Remove duplicates and sort
            support_levels = sorted(list(set([level for level in support_levels if level > 0])))
            resistance_levels = sorted(list(set([level for level in resistance_levels if level > 0])))
            
            return {
                'support': support_levels,
                'resistance': resistance_levels,
                'key_levels': {
                    'tenkan_sen': current_tenkan,
                    'kijun_sen': current_kijun,
                    'cloud_top': current_cloud_top,
                    'cloud_bottom': current_cloud_bottom
                }
            }
            
        except Exception as e:
            self.logger.error(f"Support/Resistance calculation error: {e}")
            return {'support': [], 'resistance': []}
    
    # Private Methods
    
    def _calculate_basic_lines(self, price_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Calculate basic Ichimoku lines"""
        try:
            high = price_data['high'].values
            low = price_data['low'].values
            close = price_data['close'].values
            
            # Tenkan-sen (Conversion Line)
            tenkan_high = pd.Series(high).rolling(window=self.params['tenkan_period']).max()
            tenkan_low = pd.Series(low).rolling(window=self.params['tenkan_period']).min()
            tenkan_sen = (tenkan_high + tenkan_low) / 2
            
            # Kijun-sen (Base Line)
            kijun_high = pd.Series(high).rolling(window=self.params['kijun_period']).max()
            kijun_low = pd.Series(low).rolling(window=self.params['kijun_period']).min()
            kijun_sen = (kijun_high + kijun_low) / 2
            
            # Senkou Span A (Leading Span A)
            senkou_span_a = (tenkan_sen + kijun_sen) / 2
            
            # Senkou Span B (Leading Span B)
            senkou_b_high = pd.Series(high).rolling(window=self.params['senkou_b_period']).max()
            senkou_b_low = pd.Series(low).rolling(window=self.params['senkou_b_period']).min()
            senkou_span_b = (senkou_b_high + senkou_b_low) / 2
            
            # Chikou Span (Lagging Span) - shifted backward
            chikou_span = np.roll(close, -self.params['displacement'])
            
            return {
                'tenkan_sen': tenkan_sen.values,
                'kijun_sen': kijun_sen.values,
                'senkou_span_a': senkou_span_a.values,
                'senkou_span_b': senkou_span_b.values,
                'chikou_span': chikou_span
            }
            
        except Exception as e:
            self.logger.error(f"Basic lines calculation error: {e}")
            return {}
    
    def _calculate_cloud(self, ichimoku_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate cloud (Kumo) properties"""
        try:
            senkou_a = ichimoku_data['senkou_span_a']
            senkou_b = ichimoku_data['senkou_span_b']
            
            # Shift spans forward for cloud projection
            displaced_senkou_a = np.roll(senkou_a, self.params['displacement'])
            displaced_senkou_b = np.roll(senkou_b, self.params['displacement'])
            
            # Cloud top and bottom
            cloud_top = np.maximum(displaced_senkou_a, displaced_senkou_b)
            cloud_bottom = np.minimum(displaced_senkou_a, displaced_senkou_b)
            
            # Cloud thickness
            cloud_thickness = cloud_top - cloud_bottom
            
            # Cloud color (green when Senkou A > Senkou B)
            cloud_color = np.where(displaced_senkou_a > displaced_senkou_b, 1, -1)  # 1=Green, -1=Red
            
            return {
                'senkou_span_a': displaced_senkou_a,
                'senkou_span_b': displaced_senkou_b,
                'cloud_top': cloud_top,
                'cloud_bottom': cloud_bottom,
                'cloud_thickness': cloud_thickness,
                'cloud_color': cloud_color
            }
            
        except Exception as e:
            self.logger.error(f"Cloud calculation error: {e}")
            return {}
    
    def _analyze_ichimoku_signals(self, price_data: pd.DataFrame, ichimoku_data: Dict, cloud_data: Dict) -> Dict[str, Any]:
        """Analyze Ichimoku signals"""
        try:
            current_price = price_data['close'].iloc[-1]
            
            analysis = {
                'current_price': current_price,
                'tenkan_sen': ichimoku_data['tenkan_sen'][-1],
                'kijun_sen': ichimoku_data['kijun_sen'][-1],
                'cloud_top': cloud_data['cloud_top'][-1],
                'cloud_bottom': cloud_data['cloud_bottom'][-1]
            }
            
            # Tenkan-Kijun Cross Signal
            analysis['tenkan_kijun_signal'] = self._analyze_tenkan_kijun_cross(ichimoku_data)
            
            # Price vs Cloud Position
            analysis['price_cloud_signal'] = self._analyze_price_cloud_position(current_price, cloud_data)
            
            # Chikou Span Confirmation
            analysis['chikou_signal'] = self._analyze_chikou_confirmation(price_data, ichimoku_data)
            
            # Cloud Color Signal
            analysis['cloud_signal'] = self._analyze_cloud_color(cloud_data)
            
            # Overall trend strength
            analysis['trend_strength'] = self._calculate_trend_strength(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Signal analysis error: {e}")
            return {}
    
    def _analyze_tenkan_kijun_cross(self, ichimoku_data: Dict) -> float:
        """Analyze Tenkan-Kijun cross signals"""
        try:
            tenkan = ichimoku_data['tenkan_sen']
            kijun = ichimoku_data['kijun_sen']
            
            if len(tenkan) < 2 or len(kijun) < 2:
                return 0.0
            
            # Current and previous values
            tenkan_current = tenkan[-1]
            tenkan_prev = tenkan[-2]
            kijun_current = kijun[-1]
            kijun_prev = kijun[-2]
            
            # Check for crosses
            if tenkan_prev <= kijun_prev and tenkan_current > kijun_current:
                # Bullish cross
                return 1.0
            elif tenkan_prev >= kijun_prev and tenkan_current < kijun_current:
                # Bearish cross
                return -1.0
            elif tenkan_current > kijun_current:
                # Tenkan above Kijun (bullish)
                return 0.5
            elif tenkan_current < kijun_current:
                # Tenkan below Kijun (bearish)
                return -0.5
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Tenkan-Kijun analysis error: {e}")
            return 0.0
    
    def _analyze_price_cloud_position(self, current_price: float, cloud_data: Dict) -> float:
        """Analyze price position relative to cloud"""
        try:
            cloud_top = cloud_data['cloud_top'][-1]
            cloud_bottom = cloud_data['cloud_bottom'][-1]
            cloud_thickness = cloud_data['cloud_thickness'][-1]
            
            if current_price > cloud_top:
                # Price above cloud (bullish)
                distance = (current_price - cloud_top) / current_price
                return min(1.0, distance / self.thresholds['price_distance_threshold'])
            elif current_price < cloud_bottom:
                # Price below cloud (bearish)
                distance = (cloud_bottom - current_price) / current_price
                return max(-1.0, -distance / self.thresholds['price_distance_threshold'])
            else:
                # Price inside cloud (neutral to weak)
                if cloud_thickness > 0:
                    position = (current_price - cloud_bottom) / cloud_thickness
                    return (position - 0.5) * 0.5  # Scale to [-0.25, 0.25]
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Price-cloud analysis error: {e}")
            return 0.0
    
    def _analyze_chikou_confirmation(self, price_data: pd.DataFrame, ichimoku_data: Dict) -> float:
        """Analyze Chikou Span confirmation"""
        try:
            chikou = ichimoku_data['chikou_span']
            current_price = price_data['close'].iloc[-1]
            
            # Compare current price with historical price (Chikou logic)
            displacement = self.params['displacement']
            if len(price_data) > displacement:
                historical_price = price_data['close'].iloc[-(displacement+1)]
                
                if current_price > historical_price:
                    return 0.5  # Bullish confirmation
                elif current_price < historical_price:
                    return -0.5  # Bearish confirmation
                    
            return 0.0
            
        except Exception as e:
            self.logger.error(f"Chikou analysis error: {e}")
            return 0.0
    
    def _analyze_cloud_color(self, cloud_data: Dict) -> float:
        """Analyze cloud color signals"""
        try:
            cloud_color = cloud_data['cloud_color'][-1]
            
            # Green cloud (bullish) or Red cloud (bearish)
            return 0.3 * cloud_color  # Scale factor for cloud color contribution
            
        except Exception as e:
            self.logger.error(f"Cloud color analysis error: {e}")
            return 0.0
    
    def _calculate_trend_strength(self, analysis: Dict) -> float:
        """Calculate overall trend strength"""
        try:
            signals = [
                analysis.get('tenkan_kijun_signal', 0),
                analysis.get('price_cloud_signal', 0),
                analysis.get('chikou_signal', 0),
                analysis.get('cloud_signal', 0)
            ]
            
            # Weighted average of signals
            weights = [0.4, 0.3, 0.2, 0.1]  # Adjust weights as needed
            
            strength = sum(signal * weight for signal, weight in zip(signals, weights))
            return max(-1.0, min(1.0, strength))  # Clamp to [-1, 1]
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return 0.0
    
    def _multi_timeframe_analysis(self, price_data: pd.DataFrame, current_tf: str) -> Dict[str, Any]:
        """Perform multi-timeframe Ichimoku analysis"""
        try:
            # Simplified MTF analysis (would need actual MTF data in production)
            mtf_result = {
                'higher_tf_trend': 'NEUTRAL',
                'lower_tf_trend': 'NEUTRAL',
                'alignment_score': 0.5,
                'confidence_boost': 0.0
            }
            
            # This would be expanded with actual multi-timeframe data
            return mtf_result
            
        except Exception as e:
            self.logger.error(f"Multi-timeframe analysis error: {e}")
            return {}
    
    def _determine_trend_direction(self, analysis: Dict) -> str:
        """Determine overall trend direction"""
        try:
            trend_strength = analysis.get('trend_strength', 0)
            
            if trend_strength >= self.thresholds['strong_signal']:
                return 'STRONG_BULLISH'
            elif trend_strength >= self.thresholds['medium_signal']:
                return 'BULLISH'
            elif trend_strength >= self.thresholds['weak_signal']:
                return 'WEAK_BULLISH'
            elif trend_strength <= -self.thresholds['strong_signal']:
                return 'STRONG_BEARISH'
            elif trend_strength <= -self.thresholds['medium_signal']:
                return 'BEARISH'
            elif trend_strength <= -self.thresholds['weak_signal']:
                return 'WEAK_BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"Trend direction error: {e}")
            return 'ERROR'
    
    def _calculate_signal_strength(self, analysis: Dict) -> float:
        """Calculate signal strength"""
        try:
            return abs(analysis.get('trend_strength', 0))
        except Exception as e:
            self.logger.error(f"Signal strength calculation error: {e}")
            return 0.0
    
    def _calculate_confidence(self, analysis: Dict, mtf_analysis: Dict) -> float:
        """Calculate signal confidence"""
        try:
            base_confidence = abs(analysis.get('trend_strength', 0))
            mtf_boost = mtf_analysis.get('confidence_boost', 0)
            
            # Component agreement
            signals = [
                analysis.get('tenkan_kijun_signal', 0),
                analysis.get('price_cloud_signal', 0),
                analysis.get('chikou_signal', 0),
                analysis.get('cloud_signal', 0)
            ]
            
            agreement = sum(1 for s in signals if abs(s) > 0.1) / len(signals)
            
            confidence = base_confidence * agreement + mtf_boost
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.0
    
    def _analyze_breakout_pattern(self, prices: pd.Series, cloud_top: np.ndarray, cloud_bottom: np.ndarray) -> Dict[str, Any]:
        """Analyze breakout patterns"""
        try:
            current_price = prices.iloc[-1]
            prev_prices = prices.iloc[:-1]
            
            current_cloud_top = cloud_top[-1]
            current_cloud_bottom = cloud_bottom[-1]
            
            # Check if price was inside cloud and now outside
            was_inside = any(
                cloud_bottom[i] <= prev_prices.iloc[i] <= cloud_top[i] 
                for i in range(len(prev_prices))
            )
            
            is_outside_now = current_price > current_cloud_top or current_price < current_cloud_bottom
            
            has_breakout = was_inside and is_outside_now
            
            if has_breakout:
                direction = 'BULLISH' if current_price > current_cloud_top else 'BEARISH'
                cloud_thickness = current_cloud_top - current_cloud_bottom
                penetration_depth = abs(current_price - (current_cloud_top if direction == 'BULLISH' else current_cloud_bottom))
                
                return {
                    'has_breakout': True,
                    'direction': direction,
                    'cloud_thickness': cloud_thickness,
                    'penetration_depth': penetration_depth,
                    'confirmation_bars': 1
                }
            
            return {'has_breakout': False, 'direction': 'NONE'}
            
        except Exception as e:
            self.logger.error(f"Breakout pattern analysis error: {e}")
            return {'has_breakout': False, 'direction': 'ERROR'}
    
    def _calculate_breakout_strength(self, current_price: float, cloud_data: Dict, breakout_result: Dict) -> float:
        """Calculate breakout strength"""
        try:
            if not breakout_result.get('has_breakout', False):
                return 0.0
            
            cloud_thickness = breakout_result.get('cloud_thickness', 0)
            penetration_depth = breakout_result.get('penetration_depth', 0)
            
            if cloud_thickness > 0:
                strength = penetration_depth / cloud_thickness
                return min(1.0, strength)
            
            return 0.5  # Default strength for thin clouds
            
        except Exception as e:
            self.logger.error(f"Breakout strength calculation error: {e}")
            return 0.0
    
    def _empty_ichimoku_result(self) -> Dict[str, Any]:
        """Return empty Ichimoku result"""
        return {
            'ichimoku_data': {},
            'cloud_data': {},
            'analysis': {},
            'mtf_analysis': {},
            'timestamp': datetime.now().isoformat(),
            'error': 'Calculation failed'
        }

# Usage Example (for testing):
if __name__ == "__main__":
    # Sample usage
    analyzer = IchimokuAnalyzer()
    
    # Sample OHLC data (normally from MT5 or market feed)
    sample_data = pd.DataFrame({
        'high': np.random.rand(100) * 100 + 1000,
        'low': np.random.rand(100) * 100 + 950,
        'close': np.random.rand(100) * 100 + 975,
        'open': np.random.rand(100) * 100 + 975
    })
    
    # Calculate Ichimoku
    result = analyzer.calculate_ichimoku(sample_data)
    print("Ichimoku Analysis Result:", json.dumps(result, indent=2, default=str))
    
    # Get trend signal
    trend_signal = analyzer.get_trend_signal(sample_data)
    print("Trend Signal:", trend_signal)
