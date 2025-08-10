#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Traders Dynamic Index (TDI) Calculator
Advanced TDI calculation and signal analysis

Features:
- Complete TDI calculation (RSI, Price Line, Signal Line, Market Base Line)
- Multi-timeframe TDI analysis
- Overbought/Oversold detection
- Trend strength measurement
- Signal line crossovers
- Volatility band analysis

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

class TDICalculator:
    """
    Advanced Traders Dynamic Index (TDI) Calculator
    
    TDI Components:
    - RSI Price Line (Green Line)
    - RSI Signal Line (Red Line) - MA of RSI
    - Market Base Line (Yellow Line) - MA of RSI with different period
    - Volatility Bands (Bollinger Bands around RSI)
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize TDI Calculator"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # TDI Parameters (customizable)
        self.params = {
            'rsi_period': 13,           # RSI calculation period
            'rsi_price_line': 2,        # Price line smoothing (MA of RSI)
            'rsi_signal_line': 7,       # Signal line period (MA of RSI)
            'rsi_market_base': 5,       # Market base line period
            'volatility_band': 34,      # Volatility band period (BB period)
            'band_deviation': 1.6185    # Band deviation multiplier
        }
        
        # TDI Levels and Thresholds
        self.levels = {
            'overbought_high': 68,      # Strong overbought
            'overbought_mid': 50,       # Medium overbought
            'oversold_high': 32,        # Strong oversold
            'oversold_mid': 50,         # Medium oversold
            'middle_line': 50           # Middle line
        }
        
        # Signal thresholds
        self.thresholds = {
            'strong_signal': 0.8,
            'medium_signal': 0.6,
            'weak_signal': 0.4,
            'trend_threshold': 5,       # Points above/below middle for trend
            'cross_confirmation': 3     # Bars for cross confirmation
        }
        
        # Data cache
        self.tdi_cache = {}
        self.last_calculation = {}
        
        self.logger.info("📊 TDI Calculator initialized")
    
    def calculate_tdi(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Calculate complete TDI (Traders Dynamic Index)
        
        Args:
            price_data: OHLC price data
            timeframe: Chart timeframe
            
        Returns:
            Dictionary with all TDI components and analysis
        """
        try:
            required_bars = max(self.params.values()) + 50  # Extra buffer for MA calculations
            if len(price_data) < required_bars:
                self.logger.warning(f"Insufficient data for TDI calculation: {len(price_data)} bars, need {required_bars}")
                return self._empty_tdi_result()
            
            # Calculate base RSI
            rsi_values = self._calculate_rsi(price_data['close'])
            if rsi_values is None:
                return self._empty_tdi_result()
            
            # Calculate TDI components
            tdi_components = self._calculate_tdi_components(rsi_values)
            
            # Calculate volatility bands
            volatility_bands = self._calculate_volatility_bands(rsi_values)
            
            # Analyze TDI signals
            signal_analysis = self._analyze_tdi_signals(tdi_components, volatility_bands)
            
            # Market state analysis
            market_state = self._analyze_market_state(tdi_components, volatility_bands)
            
            result = {
                'tdi_components': tdi_components,
                'volatility_bands': volatility_bands,
                'signal_analysis': signal_analysis,
                'market_state': market_state,
                'current_values': self._get_current_values(tdi_components, volatility_bands),
                'timestamp': datetime.now().isoformat(),
                'timeframe': timeframe,
                'data_points': len(price_data)
            }
            
            # Cache result
            self.last_calculation[timeframe] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"TDI calculation error: {e}")
            if self.error_handler:
                self.error_handler.log_error("TDI_CALC_ERROR", str(e))
            return self._empty_tdi_result()
    
    def get_tdi_signal(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get TDI trading signals
        
        Returns:
            TDI signal with direction, strength, and confidence
        """
        try:
            tdi_result = self.calculate_tdi(price_data)
            if not tdi_result or not tdi_result['signal_analysis']:
                return {'signal': 'NEUTRAL', 'strength': 0.0, 'confidence': 0.0}
            
            signal_analysis = tdi_result['signal_analysis']
            market_state = tdi_result['market_state']
            
            # Determine signal direction
            signal_direction = self._determine_signal_direction(signal_analysis, market_state)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(signal_analysis)
            
            # Calculate confidence
            confidence = self._calculate_confidence(signal_analysis, market_state)
            
            return {
                'signal': signal_direction,
                'strength': round(signal_strength, 4),
                'confidence': round(confidence, 4),
                'components': {
                    'price_signal_cross': signal_analysis.get('price_signal_cross', 0),
                    'market_base_position': signal_analysis.get('market_base_signal', 0),
                    'overbought_oversold': signal_analysis.get('ob_os_signal', 0),
                    'volatility_breakout': signal_analysis.get('volatility_signal', 0)
                },
                'market_state': market_state.get('state', 'UNKNOWN'),
                'trend_strength': market_state.get('trend_strength', 0),
                'details': tdi_result['current_values']
            }
            
        except Exception as e:
            self.logger.error(f"TDI signal error: {e}")
            return {'signal': 'ERROR', 'strength': 0.0, 'confidence': 0.0}
    
    def detect_tdi_crossovers(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect TDI line crossovers
        
        Returns:
            Crossover detection results
        """
        try:
            tdi_result = self.calculate_tdi(price_data)
            if not tdi_result:
                return {'crossovers': [], 'recent_cross': None}
            
            tdi_components = tdi_result['tdi_components']
            
            # Detect crossovers
            crossovers = self._detect_crossovers(tdi_components)
            
            # Find most recent significant crossover
            recent_cross = self._find_recent_crossover(crossovers)
            
            return {
                'crossovers': crossovers,
                'recent_cross': recent_cross,
                'cross_strength': self._calculate_cross_strength(recent_cross, tdi_components),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"TDI crossover detection error: {e}")
            return {'crossovers': [], 'recent_cross': None}
    
    def get_overbought_oversold_levels(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get overbought/oversold analysis
        
        Returns:
            OB/OS levels and signals
        """
        try:
            tdi_result = self.calculate_tdi(price_data)
            if not tdi_result:
                return {'status': 'NEUTRAL', 'level': 50, 'strength': 0}
            
            current_values = tdi_result['current_values']
            price_line = current_values.get('price_line', 50)
            signal_line = current_values.get('signal_line', 50)
            
            # Determine OB/OS status
            ob_os_status = self._determine_ob_os_status(price_line, signal_line)
            
            # Calculate reversal probability
            reversal_prob = self._calculate_reversal_probability(
                tdi_result['tdi_components'], 
                tdi_result['volatility_bands']
            )
            
            return {
                'status': ob_os_status['status'],
                'level': ob_os_status['level'],
                'strength': ob_os_status['strength'],
                'reversal_probability': round(reversal_prob, 4),
                'price_line': price_line,
                'signal_line': signal_line,
                'distance_from_middle': abs(price_line - self.levels['middle_line'])
            }
            
        except Exception as e:
            self.logger.error(f"OB/OS analysis error: {e}")
            return {'status': 'ERROR', 'level': 50, 'strength': 0}
    
    # Private Methods
    
    def _calculate_rsi(self, close_prices: pd.Series) -> Optional[np.ndarray]:
        """Calculate RSI using TA-Lib"""
        try:
            rsi = talib.RSI(close_prices.values, timeperiod=self.params['rsi_period'])
            return rsi
        except Exception as e:
            self.logger.error(f"RSI calculation error: {e}")
            return None
    
    def _calculate_tdi_components(self, rsi_values: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate TDI components"""
        try:
            # RSI Price Line (smoothed RSI)
            price_line = self._smooth_data(rsi_values, self.params['rsi_price_line'])
            
            # RSI Signal Line (MA of RSI)
            signal_line = self._smooth_data(rsi_values, self.params['rsi_signal_line'])
            
            # Market Base Line (different MA period)
            market_base_line = self._smooth_data(rsi_values, self.params['rsi_market_base'])
            
            return {
                'rsi': rsi_values,
                'price_line': price_line,
                'signal_line': signal_line,
                'market_base_line': market_base_line
            }
            
        except Exception as e:
            self.logger.error(f"TDI components calculation error: {e}")
            return {}
    
    def _calculate_volatility_bands(self, rsi_values: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate volatility bands (Bollinger Bands around RSI)"""
        try:
            # Calculate Bollinger Bands on RSI
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                rsi_values,
                timeperiod=self.params['volatility_band'],
                nbdevup=self.params['band_deviation'],
                nbdevdn=self.params['band_deviation']
            )
            
            # Calculate band width
            band_width = bb_upper - bb_lower
            
            return {
                'upper_band': bb_upper,
                'middle_band': bb_middle,
                'lower_band': bb_lower,
                'band_width': band_width
            }
            
        except Exception as e:
            self.logger.error(f"Volatility bands calculation error: {e}")
            return {}
    
    def _smooth_data(self, data: np.ndarray, period: int) -> np.ndarray:
        """Smooth data using Simple Moving Average"""
        try:
            return talib.SMA(data, timeperiod=period)
        except Exception as e:
            self.logger.error(f"Data smoothing error: {e}")
            return data
    
    def _analyze_tdi_signals(self, tdi_components: Dict, volatility_bands: Dict) -> Dict[str, Any]:
        """Analyze TDI signals"""
        try:
            analysis = {}
            
            # Price Line vs Signal Line Cross
            analysis['price_signal_cross'] = self._analyze_price_signal_cross(tdi_components)
            
            # Market Base Line signals
            analysis['market_base_signal'] = self._analyze_market_base_signals(tdi_components)
            
            # Overbought/Oversold signals
            analysis['ob_os_signal'] = self._analyze_ob_os_signals(tdi_components)
            
            # Volatility breakout signals
            analysis['volatility_signal'] = self._analyze_volatility_signals(tdi_components, volatility_bands)
            
            # Overall signal strength
            analysis['overall_strength'] = self._calculate_overall_strength(analysis)
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"TDI signal analysis error: {e}")
            return {}
    
    def _analyze_price_signal_cross(self, tdi_components: Dict) -> float:
        """Analyze Price Line vs Signal Line crossovers"""
        try:
            price_line = tdi_components['price_line']
            signal_line = tdi_components['signal_line']
            
            if len(price_line) < 2 or len(signal_line) < 2:
                return 0.0
            
            # Current and previous values
            price_curr = price_line[-1]
            price_prev = price_line[-2]
            signal_curr = signal_line[-1]
            signal_prev = signal_line[-2]
            
            # Check for crosses
            if price_prev <= signal_prev and price_curr > signal_curr:
                # Bullish cross
                return 1.0
            elif price_prev >= signal_prev and price_curr < signal_curr:
                # Bearish cross
                return -1.0
            elif price_curr > signal_curr:
                # Price above signal (bullish bias)
                separation = (price_curr - signal_curr) / 100  # Normalize
                return min(0.5, separation * 10)
            elif price_curr < signal_curr:
                # Price below signal (bearish bias)
                separation = (signal_curr - price_curr) / 100  # Normalize
                return max(-0.5, -separation * 10)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Price-signal cross analysis error: {e}")
            return 0.0
    
    def _analyze_market_base_signals(self, tdi_components: Dict) -> float:
        """Analyze Market Base Line signals"""
        try:
            price_line = tdi_components['price_line']
            market_base = tdi_components['market_base_line']
            
            if len(price_line) < 1 or len(market_base) < 1:
                return 0.0
            
            price_curr = price_line[-1]
            base_curr = market_base[-1]
            middle = self.levels['middle_line']
            
            # Price position relative to market base and middle line
            if price_curr > base_curr and price_curr > middle:
                # Bullish: price above base and middle
                strength = min((price_curr - middle) / 50, 1.0)
                return strength
            elif price_curr < base_curr and price_curr < middle:
                # Bearish: price below base and middle
                strength = min((middle - price_curr) / 50, 1.0)
                return -strength
            else:
                # Mixed signals
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Market base analysis error: {e}")
            return 0.0
    
    def _analyze_ob_os_signals(self, tdi_components: Dict) -> float:
        """Analyze overbought/oversold signals"""
        try:
            price_line = tdi_components['price_line']
            signal_line = tdi_components['signal_line']
            
            if len(price_line) < 1 or len(signal_line) < 1:
                return 0.0
            
            price_curr = price_line[-1]
            signal_curr = signal_line[-1]
            
            # Average of price and signal lines
            avg_level = (price_curr + signal_curr) / 2
            
            if avg_level >= self.levels['overbought_high']:
                # Strong overbought - potential bearish reversal
                return -0.8
            elif avg_level >= self.levels['overbought_mid']:
                # Medium overbought
                return -0.5
            elif avg_level <= self.levels['oversold_high']:
                # Strong oversold - potential bullish reversal
                return 0.8
            elif avg_level <= self.levels['oversold_mid']:
                # Medium oversold
                return 0.5
            else:
                # Neutral zone
                return 0.0
                
        except Exception as e:
            self.logger.error(f"OB/OS analysis error: {e}")
            return 0.0
    
    def _analyze_volatility_signals(self, tdi_components: Dict, volatility_bands: Dict) -> float:
        """Analyze volatility band signals"""
        try:
            price_line = tdi_components['price_line']
            upper_band = volatility_bands['upper_band']
            lower_band = volatility_bands['lower_band']
            band_width = volatility_bands['band_width']
            
            if len(price_line) < 2:
                return 0.0
            
            price_curr = price_line[-1]
            price_prev = price_line[-2]
            upper_curr = upper_band[-1]
            lower_curr = lower_band[-1]
            width_curr = band_width[-1]
            
            # Band breakout signals
            if price_prev < upper_curr and price_curr >= upper_curr:
                # Breakout above upper band (bullish)
                return 0.7
            elif price_prev > lower_curr and price_curr <= lower_curr:
                # Breakout below lower band (bearish)
                return -0.7
            elif width_curr < 10:  # Tight bands - low volatility
                # Potential breakout setup
                if price_curr > (upper_curr + lower_curr) / 2:
                    return 0.3  # Slight bullish bias
                else:
                    return -0.3  # Slight bearish bias
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Volatility analysis error: {e}")
            return 0.0
    
    def _calculate_overall_strength(self, analysis: Dict) -> float:
        """Calculate overall signal strength"""
        try:
            signals = [
                analysis.get('price_signal_cross', 0),
                analysis.get('market_base_signal', 0),
                analysis.get('ob_os_signal', 0),
                analysis.get('volatility_signal', 0)
            ]
            
            # Weighted average
            weights = [0.4, 0.3, 0.2, 0.1]
            
            strength = sum(signal * weight for signal, weight in zip(signals, weights))
            return max(-1.0, min(1.0, strength))
            
        except Exception as e:
            self.logger.error(f"Overall strength calculation error: {e}")
            return 0.0
    
    def _analyze_market_state(self, tdi_components: Dict, volatility_bands: Dict) -> Dict[str, Any]:
        """Analyze current market state"""
        try:
            price_line = tdi_components['price_line']
            signal_line = tdi_components['signal_line']
            band_width = volatility_bands['band_width']
            
            if len(price_line) < 1:
                return {'state': 'UNKNOWN', 'trend_strength': 0}
            
            price_curr = price_line[-1]
            signal_curr = signal_line[-1]
            width_curr = band_width[-1]
            
            # Determine market state
            if width_curr < 8:
                state = 'LOW_VOLATILITY'
            elif width_curr > 20:
                state = 'HIGH_VOLATILITY'
            elif price_curr > 60 and signal_curr > 60:
                state = 'BULLISH_MOMENTUM'
            elif price_curr < 40 and signal_curr < 40:
                state = 'BEARISH_MOMENTUM'
            elif abs(price_curr - signal_curr) < 2:
                state = 'CONSOLIDATION'
            else:
                state = 'TRENDING'
            
            # Calculate trend strength
            trend_strength = self._calculate_trend_strength(price_curr, signal_curr)
            
            return {
                'state': state,
                'trend_strength': trend_strength,
                'volatility_level': 'HIGH' if width_curr > 15 else 'LOW' if width_curr < 10 else 'MEDIUM'
            }
            
        except Exception as e:
            self.logger.error(f"Market state analysis error: {e}")
            return {'state': 'ERROR', 'trend_strength': 0}
    
    def _calculate_trend_strength(self, price_line: float, signal_line: float) -> float:
        """Calculate trend strength"""
        try:
            middle = self.levels['middle_line']
            
            # Distance from middle line
            price_distance = abs(price_line - middle) / 50
            signal_distance = abs(signal_line - middle) / 50
            
            # Average distance (trend strength)
            avg_distance = (price_distance + signal_distance) / 2
            
            # Direction consistency
            direction_consistency = 1.0 if (price_line - middle) * (signal_line - middle) > 0 else 0.5
            
            return min(1.0, avg_distance * direction_consistency)
            
        except Exception as e:
            self.logger.error(f"Trend strength calculation error: {e}")
            return 0.0
    
    def _get_current_values(self, tdi_components: Dict, volatility_bands: Dict) -> Dict[str, float]:
        """Get current TDI values"""
        try:
            return {
                'rsi': float(tdi_components['rsi'][-1]) if len(tdi_components['rsi']) > 0 else 50.0,
                'price_line': float(tdi_components['price_line'][-1]) if len(tdi_components['price_line']) > 0 else 50.0,
                'signal_line': float(tdi_components['signal_line'][-1]) if len(tdi_components['signal_line']) > 0 else 50.0,
                'market_base_line': float(tdi_components['market_base_line'][-1]) if len(tdi_components['market_base_line']) > 0 else 50.0,
                'upper_band': float(volatility_bands['upper_band'][-1]) if len(volatility_bands['upper_band']) > 0 else 70.0,
                'lower_band': float(volatility_bands['lower_band'][-1]) if len(volatility_bands['lower_band']) > 0 else 30.0,
                'band_width': float(volatility_bands['band_width'][-1]) if len(volatility_bands['band_width']) > 0 else 40.0
            }
        except Exception as e:
            self.logger.error(f"Current values extraction error: {e}")
            return {}
    
    def _determine_signal_direction(self, signal_analysis: Dict, market_state: Dict) -> str:
        """Determine overall signal direction"""
        try:
            overall_strength = signal_analysis.get('overall_strength', 0)
            
            if overall_strength >= self.thresholds['strong_signal']:
                return 'STRONG_BULLISH'
            elif overall_strength >= self.thresholds['medium_signal']:
                return 'BULLISH'
            elif overall_strength >= self.thresholds['weak_signal']:
                return 'WEAK_BULLISH'
            elif overall_strength <= -self.thresholds['strong_signal']:
                return 'STRONG_BEARISH'
            elif overall_strength <= -self.thresholds['medium_signal']:
                return 'BEARISH'
            elif overall_strength <= -self.thresholds['weak_signal']:
                return 'WEAK_BEARISH'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            self.logger.error(f"Signal direction error: {e}")
            return 'ERROR'
    
    def _calculate_signal_strength(self, signal_analysis: Dict) -> float:
        """Calculate signal strength"""
        try:
            return abs(signal_analysis.get('overall_strength', 0))
        except Exception as e:
            self.logger.error(f"Signal strength calculation error: {e}")
            return 0.0
    
    def _calculate_confidence(self, signal_analysis: Dict, market_state: Dict) -> float:
        """Calculate signal confidence"""
        try:
            base_strength = abs(signal_analysis.get('overall_strength', 0))
            
            # Component agreement
            signals = [
                signal_analysis.get('price_signal_cross', 0),
                signal_analysis.get('market_base_signal', 0),
                signal_analysis.get('ob_os_signal', 0),
                signal_analysis.get('volatility_signal', 0)
            ]
            
            # Count signals in same direction
            positive_signals = sum(1 for s in signals if s > 0.1)
            negative_signals = sum(1 for s in signals if s < -0.1)
            total_signals = len([s for s in signals if abs(s) > 0.1])
            
            if total_signals > 0:
                agreement = max(positive_signals, negative_signals) / len(signals)
            else:
                agreement = 0.0
            
            # Trend strength boost
            trend_strength = market_state.get('trend_strength', 0)
            
            confidence = base_strength * agreement + (trend_strength * 0.1)
            return min(1.0, confidence)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.0
    
    def _detect_crossovers(self, tdi_components: Dict) -> List[Dict[str, Any]]:
        """Detect TDI line crossovers"""
        try:
            crossovers = []
            
            price_line = tdi_components['price_line']
            signal_line = tdi_components['signal_line']
            
            if len(price_line) < 10 or len(signal_line) < 10:
                return crossovers
            
            # Check last 10 bars for crossovers
            for i in range(1, min(10, len(price_line))):
                prev_price = price_line[-(i+1)]
                curr_price = price_line[-i]
                prev_signal = signal_line[-(i+1)]
                curr_signal = signal_line[-i]
                
                # Detect crosses
                if prev_price <= prev_signal and curr_price > curr_signal:
                    crossovers.append({
                        'type': 'BULLISH_CROSS',
                        'bars_ago': i,
                        'price_level': curr_price,
                        'signal_level': curr_signal
                    })
                elif prev_price >= prev_signal and curr_price < curr_signal:
                    crossovers.append({
                        'type': 'BEARISH_CROSS',
                        'bars_ago': i,
                        'price_level': curr_price,
                        'signal_level': curr_signal
                    })
            
            return crossovers
            
        except Exception as e:
            self.logger.error(f"Crossover detection error: {e}")
            return []
    
    def _find_recent_crossover(self, crossovers: List[Dict]) -> Optional[Dict[str, Any]]:
        """Find most recent significant crossover"""
        try:
            if not crossovers:
                return None
            
            # Return the most recent crossover
            return min(crossovers, key=lambda x: x.get('bars_ago', 999))
            
        except Exception as e:
            self.logger.error(f"Recent crossover search error: {e}")
            return None
    
    def _calculate_cross_strength(self, recent_cross: Optional[Dict], tdi_components: Dict) -> float:
        """Calculate crossover strength"""
        try:
            if not recent_cross:
                return 0.0
            
            # Simple strength calculation based on level and age
            bars_ago = recent_cross.get('bars_ago', 10)
            price_level = recent_cross.get('price_level', 50)
            
            # Fresher crosses are stronger
            time_factor = max(0.1, 1.0 - (bars_ago / 10))
            
            # Distance from middle line
            distance_factor = abs(price_level - 50) / 50
            
            return min(1.0, time_factor * (1 + distance_factor))
            
        except Exception as e:
            self.logger.error(f"Cross strength calculation error: {e}")
            return 0.0
    
    def _determine_ob_os_status(self, price_line: float, signal_line: float) -> Dict[str, Any]:
        """Determine overbought/oversold status"""
        try:
            avg_level = (price_line + signal_line) / 2
            
            if avg_level >= self.levels['overbought_high']:
                return {'status': 'STRONG_OVERBOUGHT', 'level': avg_level, 'strength': 0.8}
            elif avg_level >= 60:
                return {'status': 'OVERBOUGHT', 'level': avg_level, 'strength': 0.6}
            elif avg_level <= self.levels['oversold_high']:
                return {'status': 'STRONG_OVERSOLD', 'level': avg_level, 'strength': 0.8}
            elif avg_level <= 40:
                return {'status': 'OVERSOLD', 'level': avg_level, 'strength': 0.6}
            else:
                return {'status': 'NEUTRAL', 'level': avg_level, 'strength': 0.0}
                
        except Exception as e:
            self.logger.error(f"OB/OS status error: {e}")
            return {'status': 'ERROR', 'level': 50, 'strength': 0}
    
    def _calculate_reversal_probability(self, tdi_components: Dict, volatility_bands: Dict) -> float:
        """Calculate reversal probability"""
        try:
            price_line = tdi_components['price_line']
            signal_line = tdi_components['signal_line']
            upper_band = volatility_bands['upper_band']
            lower_band = volatility_bands['lower_band']
            
            if len(price_line) < 1:
                return 0.0
            
            price_curr = price_line[-1]
            signal_curr = signal_line[-1]
            upper_curr = upper_band[-1]
            lower_curr = lower_band[-1]
            
            # Higher probability near extreme levels
            if price_curr >= upper_curr or signal_curr >= upper_curr:
                return 0.7  # High probability of bearish reversal
            elif price_curr <= lower_curr or signal_curr <= lower_curr:
                return 0.7  # High probability of bullish reversal
            elif price_curr >= 70 or signal_curr >= 70:
                return 0.5  # Medium probability of bearish reversal
            elif price_curr <= 30 or signal_curr <= 30:
                return 0.5  # Medium probability of bullish reversal
            else:
                return 0.2  # Low probability of reversal
                
        except Exception as e:
            self.logger.error(f"Reversal probability calculation error: {e}")
            return 0.0
    
    def _empty_tdi_result(self) -> Dict[str, Any]:
        """Return empty TDI result"""
        return {
            'tdi_components': {},
            'volatility_bands': {},
            'signal_analysis': {},
            'market_state': {},
            'current_values': {},
            'timestamp': datetime.now().isoformat(),
            'error': 'Calculation failed'
        }

# Usage Example (for testing):
if __name__ == "__main__":
    # Sample usage
    calculator = TDICalculator()
    
    # Sample OHLC data
    sample_data = pd.DataFrame({
        'high': np.random.rand(100) * 100 + 1000,
        'low': np.random.rand(100) * 100 + 950,
        'close': np.random.rand(100) * 100 + 975,
        'open': np.random.rand(100) * 100 + 975,
        'volume': np.random.rand(100) * 1000 + 500
    })
    
    # Calculate TDI
    result = calculator.calculate_tdi(sample_data)
    print("TDI Analysis Result:", json.dumps(result, indent=2, default=str))
    
    # Get TDI signal
    tdi_signal = calculator.get_tdi_signal(sample_data)
    print("TDI Signal:", tdi_signal)
    
    # Detect crossovers
    crossover_result = calculator.detect_tdi_crossovers(sample_data)
    print("TDI Crossovers:", crossover_result)
    
    # Get OB/OS levels
    ob_os_result = calculator.get_overbought_oversold_levels(sample_data)
    print("TDI OB/OS Analysis:", ob_os_result)