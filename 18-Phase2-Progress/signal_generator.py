#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Unified Signal Generation Engine
Combines all technical analysis components to generate unified trading signals

Features:
- Integrates signals from all analyzers (Ichimoku, TDI, BB, SuperTrend, Price Action, Trend)
- Multi-layered signal confirmation system
- Signal strength and confidence scoring
- Risk-adjusted signal generation
- Real-time signal monitoring and alerts
- Signal performance tracking

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

# Import all analyzers (will be available when other files are created)
try:
    from ichimoku_analyzer import IchimokuAnalyzer
    from tdi_calculator import TDICalculator
    from bollinger_analyzer import BollingerAnalyzer
    from supertrend_calculator import SuperTrendCalculator
    from price_action_analyzer import PriceActionAnalyzer
    from trend_detector import TrendDetector
except ImportError as e:
    # For development - will work when all files are created
    print(f"Import note: {e} - This is expected during development")

class SignalGenerator:
    """
    Unified Signal Generation Engine
    
    Combines and processes signals from all technical analysis components:
    - Ichimoku Kinko Hyo signals
    - TDI (Traders Dynamic Index) signals
    - Bollinger Bands signals
    - SuperTrend Entry/Exit signals
    - Price Action signals
    - Multi-timeframe trend signals
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize Signal Generator"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger(__name__)
        
        # Initialize all analyzers
        self.analyzers = {}
        self._initialize_analyzers()
        
        # Signal combination weights
        self.signal_weights = {
            'ichimoku': 0.20,        # Ichimoku trend and momentum
            'tdi': 0.15,             # TDI overbought/oversold and momentum
            'bollinger': 0.15,       # Bollinger Bands squeeze and breakouts
            'supertrend': 0.25,      # SuperTrend entry signals (most weight)
            'price_action': 0.15,    # Price action patterns
            'trend': 0.10            # Multi-timeframe trend alignment
        }
        
        # Signal confirmation requirements
        self.confirmation_settings = {
            'minimum_confirmations': 3,    # Minimum number of confirming signals
            'strong_signal_threshold': 0.7, # Threshold for strong signal
            'medium_signal_threshold': 0.5, # Threshold for medium signal
            'weak_signal_threshold': 0.3,  # Threshold for weak signal
            'conflict_threshold': 0.2      # Max acceptable signal conflict
        }
        
        # Risk filtering settings
        self.risk_filters = {
            'max_volatility': 0.05,        # Maximum acceptable volatility
            'min_liquidity_score': 0.3,    # Minimum liquidity requirement
            'news_impact_filter': True,    # Filter signals during high impact news
            'session_filter': True         # Filter signals outside trading hours
        }
        
        # Signal tracking
        self.signal_history = []
        self.performance_stats = {
            'total_signals': 0,
            'successful_signals': 0,
            'failed_signals': 0,
            'win_rate': 0.0,
            'avg_strength': 0.0,
            'avg_confidence': 0.0
        }
        
        self.logger.info("🎯 Unified Signal Generator initialized")
    
    def generate_signal(self, price_data: pd.DataFrame, timeframe: str = "M15", 
                       additional_data: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate unified trading signal from all analyzers
        
        Args:
            price_data: OHLC price data
            timeframe: Current timeframe
            additional_data: Additional market data (volume, news, etc.)
            
        Returns:
            Unified signal with comprehensive analysis
        """
        try:
            if len(price_data) < 100:
                self.logger.warning(f"Insufficient data for signal generation: {len(price_data)} bars")
                return self._empty_signal_result()
            
            # Get signals from all analyzers
            individual_signals = self._collect_individual_signals(price_data, timeframe)
            
            # Apply risk filters
            risk_assessment = self._assess_signal_risk(price_data, individual_signals, additional_data)
            
            # Combine signals
            combined_signal = self._combine_signals(individual_signals, risk_assessment)
            
            # Validate and confirm signal
            signal_validation = self._validate_signal(combined_signal, individual_signals, risk_assessment)
            
            # Generate final signal recommendation
            final_signal = self._generate_final_signal(
                combined_signal, signal_validation, individual_signals, risk_assessment
            )
            
            # Add metadata and tracking
            final_signal.update({
                'individual_signals': individual_signals,
                'risk_assessment': risk_assessment,
                'signal_validation': signal_validation,
                'generation_time': datetime.now().isoformat(),
                'timeframe': timeframe,
                'data_quality': self._assess_data_quality(price_data),
                'signal_id': self._generate_signal_id()
            })
            
            # Update performance tracking
            self._update_signal_tracking(final_signal)
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Signal generation error: {e}")
            if self.error_handler:
                self.error_handler.log_error("SIGNAL_GENERATION_ERROR", str(e))
            return self._empty_signal_result()
    
    def get_signal_strength_breakdown(self, price_data: pd.DataFrame, 
                                    timeframe: str = "M15") -> Dict[str, Any]:
        """
        Get detailed breakdown of signal strength components
        
        Returns:
            Detailed analysis of each signal component
        """
        try:
            individual_signals = self._collect_individual_signals(price_data, timeframe)
            
            # Calculate component contributions
            component_breakdown = {}
            total_weighted_strength = 0.0
            
            for analyzer_name, signal_data in individual_signals.items():
                if signal_data and 'strength' in signal_data:
                    weight = self.signal_weights.get(analyzer_name, 0.0)
                    weighted_strength = signal_data['strength'] * weight
                    total_weighted_strength += weighted_strength
                    
                    component_breakdown[analyzer_name] = {
                        'raw_strength': signal_data['strength'],
                        'weight': weight,
                        'weighted_contribution': weighted_strength,
                        'signal_direction': signal_data.get('signal', 'NEUTRAL'),
                        'confidence': signal_data.get('confidence', 0.0),
                        'percentage_contribution': 0.0  # Will be calculated below
                    }
            
            # Calculate percentage contributions
            if total_weighted_strength > 0:
                for analyzer_name in component_breakdown:
                    weighted_contrib = component_breakdown[analyzer_name]['weighted_contribution']
                    component_breakdown[analyzer_name]['percentage_contribution'] = \
                        (weighted_contrib / total_weighted_strength) * 100
            
            return {
                'component_breakdown': component_breakdown,
                'total_weighted_strength': round(total_weighted_strength, 4),
                'strongest_component': max(component_breakdown.items(), 
                                         key=lambda x: x[1]['weighted_contribution'])[0] if component_breakdown else 'NONE',
                'weakest_component': min(component_breakdown.items(), 
                                       key=lambda x: x[1]['weighted_contribution'])[0] if component_breakdown else 'NONE',
                'signal_consensus': self._calculate_signal_consensus(individual_signals),
                'conflict_analysis': self._analyze_signal_conflicts(individual_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Signal strength breakdown error: {e}")
            return {'component_breakdown': {}, 'total_weighted_strength': 0.0}
    
    def validate_signal_quality(self, price_data: pd.DataFrame, timeframe: str = "M15") -> Dict[str, Any]:
        """
        Validate the quality of generated signals
        
        Returns:
            Signal quality assessment and recommendations
        """
        try:
            # Generate signal for quality assessment
            signal_result = self.generate_signal(price_data, timeframe)
            
            # Quality criteria
            quality_scores = {
                'data_quality': self._assess_data_quality(price_data),
                'signal_consistency': self._assess_signal_consistency(signal_result),
                'confirmation_strength': self._assess_confirmation_strength(signal_result),
                'risk_level': self._assess_risk_level(signal_result),
                'timing_quality': self._assess_timing_quality(price_data, signal_result)
            }
            
            # Overall quality score
            weights = {'data_quality': 0.2, 'signal_consistency': 0.3, 'confirmation_strength': 0.2,
                      'risk_level': 0.15, 'timing_quality': 0.15}
            
            overall_quality = sum(score * weights[category] for category, score in quality_scores.items())
            
            # Quality rating
            if overall_quality >= 0.8:
                quality_rating = 'EXCELLENT'
            elif overall_quality >= 0.6:
                quality_rating = 'GOOD'
            elif overall_quality >= 0.4:
                quality_rating = 'FAIR'
            else:
                quality_rating = 'POOR'
            
            # Recommendations
            recommendations = self._generate_quality_recommendations(quality_scores, overall_quality)
            
            return {
                'overall_quality': round(overall_quality, 4),
                'quality_rating': quality_rating,
                'quality_scores': {k: round(v, 4) for k, v in quality_scores.items()},
                'recommendations': recommendations,
                'signal_reliability': 'HIGH' if overall_quality >= 0.7 else 'MEDIUM' if overall_quality >= 0.5 else 'LOW'
            }
            
        except Exception as e:
            self.logger.error(f"Signal quality validation error: {e}")
            return {'overall_quality': 0.0, 'quality_rating': 'ERROR'}
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get signal generator performance statistics"""
        try:
            # Update win rate
            if self.performance_stats['total_signals'] > 0:
                self.performance_stats['win_rate'] = (
                    self.performance_stats['successful_signals'] / 
                    self.performance_stats['total_signals']
                )
            
            # Recent performance (last 50 signals)
            recent_signals = self.signal_history[-50:] if len(self.signal_history) >= 50 else self.signal_history
            recent_performance = self._calculate_recent_performance(recent_signals)
            
            return {
                'overall_performance': self.performance_stats.copy(),
                'recent_performance': recent_performance,
                'signal_distribution': self._analyze_signal_distribution(),
                'analyzer_performance': self._analyze_analyzer_performance(),
                'improvement_suggestions': self._generate_improvement_suggestions()
            }
            
        except Exception as e:
            self.logger.error(f"Performance statistics error: {e}")
            return {'overall_performance': {}, 'recent_performance': {}}
    
    # Private Methods
    
    def _initialize_analyzers(self):
        """Initialize all technical analysis components"""
        try:
            # Initialize analyzers (with error handling for development)
            try:
                self.analyzers['ichimoku'] = IchimokuAnalyzer(self.config_manager, self.error_handler)
            except:
                self.analyzers['ichimoku'] = None
                self.logger.warning("Ichimoku analyzer not available")
            
            try:
                self.analyzers['tdi'] = TDICalculator(self.config_manager, self.error_handler)
            except:
                self.analyzers['tdi'] = None
                self.logger.warning("TDI calculator not available")
            
            try:
                self.analyzers['bollinger'] = BollingerAnalyzer(self.config_manager, self.error_handler)
            except:
                self.analyzers['bollinger'] = None
                self.logger.warning("Bollinger analyzer not available")
            
            try:
                self.analyzers['supertrend'] = SuperTrendCalculator(self.config_manager, self.error_handler)
            except:
                self.analyzers['supertrend'] = None
                self.logger.warning("SuperTrend calculator not available")
            
            try:
                self.analyzers['price_action'] = PriceActionAnalyzer(self.config_manager, self.error_handler)
            except:
                self.analyzers['price_action'] = None
                self.logger.warning("Price Action analyzer not available")
            
            try:
                self.analyzers['trend'] = TrendDetector(self.config_manager, self.error_handler)
            except:
                self.analyzers['trend'] = None
                self.logger.warning("Trend detector not available")
            
        except Exception as e:
            self.logger.error(f"Analyzer initialization error: {e}")
    
    def _collect_individual_signals(self, price_data: pd.DataFrame, timeframe: str) -> Dict[str, Any]:
        """Collect signals from all analyzers"""
        try:
            signals = {}
            
            # Ichimoku signals
            if self.analyzers.get('ichimoku'):
                try:
                    ichimoku_signal = self.analyzers['ichimoku'].get_trend_signal(price_data)
                    signals['ichimoku'] = ichimoku_signal
                except Exception as e:
                    self.logger.warning(f"Ichimoku signal collection error: {e}")
                    signals['ichimoku'] = None
            
            # TDI signals
            if self.analyzers.get('tdi'):
                try:
                    tdi_signal = self.analyzers['tdi'].get_tdi_signal(price_data)
                    signals['tdi'] = tdi_signal
                except Exception as e:
                    self.logger.warning(f"TDI signal collection error: {e}")
                    signals['tdi'] = None
            
            # Bollinger Bands signals
            if self.analyzers.get('bollinger'):
                try:
                    bb_signal = self.analyzers['bollinger'].get_bb_signal(price_data)
                    signals['bollinger'] = bb_signal
                except Exception as e:
                    self.logger.warning(f"Bollinger signal collection error: {e}")
                    signals['bollinger'] = None
            
            # SuperTrend signals
            if self.analyzers.get('supertrend'):
                try:
                    st_signal = self.analyzers['supertrend'].get_entry_signal(price_data)
                    signals['supertrend'] = st_signal
                except Exception as e:
                    self.logger.warning(f"SuperTrend signal collection error: {e}")
                    signals['supertrend'] = None
            
            # Price Action signals
            if self.analyzers.get('price_action'):
                try:
                    pa_signal = self.analyzers['price_action'].get_price_action_signal(price_data)
                    signals['price_action'] = pa_signal
                except Exception as e:
                    self.logger.warning(f"Price Action signal collection error: {e}")
                    signals['price_action'] = None
            
            # Trend signals
            if self.analyzers.get('trend'):
                try:
                    trend_signal = self.analyzers['trend'].get_trend_signal(price_data, timeframe)
                    signals['trend'] = trend_signal
                except Exception as e:
                    self.logger.warning(f"Trend signal collection error: {e}")
                    signals['trend'] = None
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Signal collection error: {e}")
            return {}
    
    def _assess_signal_risk(self, price_data: pd.DataFrame, individual_signals: Dict, 
                          additional_data: Optional[Dict]) -> Dict[str, Any]:
        """Assess risk factors for signal generation"""
        try:
            risk_factors = {}
            
            # Volatility risk
            if len(price_data) >= 20:
                recent_returns = price_data['close'].pct_change().dropna()[-20:]
                volatility = recent_returns.std()
                risk_factors['volatility'] = min(1.0, volatility / self.risk_filters['max_volatility'])
            else:
                risk_factors['volatility'] = 0.5
            
            # Liquidity risk (based on volume if available)
            if 'volume' in price_data.columns:
                recent_volume = price_data['volume'][-10:].mean()
                avg_volume = price_data['volume'][-50:].mean()
                liquidity_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
                risk_factors['liquidity'] = max(0.0, min(1.0, liquidity_ratio))
            else:
                risk_factors['liquidity'] = 0.7  # Default moderate liquidity
            
            # Signal conflict risk
            signal_directions = [
                s.get('signal', 'NEUTRAL') for s in individual_signals.values() 
                if s and 'signal' in s
            ]
            bullish_count = sum(1 for s in signal_directions if 'BULLISH' in s)
            bearish_count = sum(1 for s in signal_directions if 'BEARISH' in s)
            total_directional = bullish_count + bearish_count
            
            if total_directional > 0:
                conflict_ratio = min(bullish_count, bearish_count) / total_directional
                risk_factors['signal_conflict'] = conflict_ratio
            else:
                risk_factors['signal_conflict'] = 0.0
            
            # Time-based risk (market hours, news events)
            risk_factors['timing_risk'] = self._assess_timing_risk(additional_data)
            
            # Overall risk score
            risk_weights = {'volatility': 0.3, 'liquidity': 0.2, 'signal_conflict': 0.3, 'timing_risk': 0.2}
            overall_risk = sum(risk_factors[factor] * risk_weights[factor] for factor in risk_factors)
            
            # Risk category
            if overall_risk <= 0.3:
                risk_category = 'LOW'
            elif overall_risk <= 0.6:
                risk_category = 'MEDIUM'
            else:
                risk_category = 'HIGH'
            
            return {
                'risk_factors': risk_factors,
                'overall_risk': round(overall_risk, 4),
                'risk_category': risk_category,
                'risk_acceptable': overall_risk <= 0.7  # Risk threshold
            }
            
        except Exception as e:
            self.logger.error(f"Risk assessment error: {e}")
            return {'overall_risk': 1.0, 'risk_category': 'HIGH', 'risk_acceptable': False}
    
    def _combine_signals(self, individual_signals: Dict, risk_assessment: Dict) -> Dict[str, Any]:
        """Combine individual signals into unified signal"""
        try:
            # Initialize combined signal
            combined_strength = 0.0
            combined_confidence = 0.0
            signal_contributions = {}
            valid_signals = 0
            
            # Process each analyzer signal
            for analyzer_name, signal_data in individual_signals.items():
                if not signal_data or 'signal' in signal_data and signal_data['signal'] in ['ERROR', 'NEUTRAL']:
                    continue
                
                # Get signal components
                signal_direction = signal_data.get('signal', 'NEUTRAL')
                signal_strength = signal_data.get('strength', 0.0)
                signal_confidence = signal_data.get('confidence', 0.0)
                
                # Apply signal weight
                weight = self.signal_weights.get(analyzer_name, 0.0)
                
                # Convert direction to numeric value
                direction_value = self._signal_direction_to_value(signal_direction)
                
                # Calculate weighted contribution
                weighted_strength = signal_strength * weight * abs(direction_value)
                weighted_confidence = signal_confidence * weight
                
                # Apply direction
                combined_strength += weighted_strength * direction_value
                combined_confidence += weighted_confidence
                
                # Track contribution
                signal_contributions[analyzer_name] = {
                    'direction': signal_direction,
                    'strength': signal_strength,
                    'confidence': signal_confidence,
                    'weight': weight,
                    'contribution': weighted_strength * direction_value
                }
                
                valid_signals += 1
            
            # Normalize confidence by number of valid signals
            if valid_signals > 0:
                combined_confidence /= valid_signals
            
            # Apply risk adjustment
            risk_multiplier = 1.0 - (risk_assessment.get('overall_risk', 0.5) * 0.3)
            combined_strength *= risk_multiplier
            combined_confidence *= risk_multiplier
            
            # Determine combined direction
            if combined_strength > 0.1:
                combined_direction = 'BULLISH'
            elif combined_strength < -0.1:
                combined_direction = 'BEARISH'
            else:
                combined_direction = 'NEUTRAL'
            
            # Adjust direction based on strength
            strength_abs = abs(combined_strength)
            if strength_abs >= 0.7:
                combined_direction = f'STRONG_{combined_direction}' if combined_direction != 'NEUTRAL' else 'NEUTRAL'
            elif strength_abs >= 0.4:
                combined_direction = combined_direction
            elif strength_abs >= 0.2:
                combined_direction = f'WEAK_{combined_direction}' if combined_direction != 'NEUTRAL' else 'NEUTRAL'
            
            return {
                'combined_direction': combined_direction,
                'combined_strength': round(abs(combined_strength), 4),
                'combined_confidence': round(combined_confidence, 4),
                'signal_contributions': signal_contributions,
                'valid_signals_count': valid_signals,
                'risk_adjusted': True
            }
            
        except Exception as e:
            self.logger.error(f"Signal combination error: {e}")
            return {'combined_direction': 'ERROR', 'combined_strength': 0.0, 'combined_confidence': 0.0}
    
    def _validate_signal(self, combined_signal: Dict, individual_signals: Dict, 
                        risk_assessment: Dict) -> Dict[str, Any]:
        """Validate the combined signal"""
        try:
            validation_results = {}
            
            # Confirmation count validation
            confirming_signals = self._count_confirming_signals(individual_signals)
            validation_results['confirmation_count'] = confirming_signals
            validation_results['sufficient_confirmations'] = (
                confirming_signals >= self.confirmation_settings['minimum_confirmations']
            )
            
            # Signal strength validation
            combined_strength = combined_signal.get('combined_strength', 0.0)
            validation_results['strength_validation'] = {
                'strength_level': self._categorize_strength(combined_strength),
                'meets_minimum': combined_strength >= self.confirmation_settings['weak_signal_threshold']
            }
            
            # Risk validation
            validation_results['risk_validation'] = {
                'risk_acceptable': risk_assessment.get('risk_acceptable', False),
                'risk_level': risk_assessment.get('risk_category', 'HIGH')
            }
            
            # Conflict validation
            conflict_level = risk_assessment.get('risk_factors', {}).get('signal_conflict', 1.0)
            validation_results['conflict_validation'] = {
                'conflict_level': round(conflict_level, 4),
                'acceptable_conflict': conflict_level <= self.confirmation_settings['conflict_threshold']
            }
            
            # Overall validation
            validation_passed = (
                validation_results['sufficient_confirmations'] and
                validation_results['strength_validation']['meets_minimum'] and
                validation_results['risk_validation']['risk_acceptable'] and
                validation_results['conflict_validation']['acceptable_conflict']
            )
            
            validation_results['overall_validation'] = validation_passed
            validation_results['validation_score'] = self._calculate_validation_score(validation_results)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return {'overall_validation': False, 'validation_score': 0.0}
    
    def _generate_final_signal(self, combined_signal: Dict, signal_validation: Dict,
                             individual_signals: Dict, risk_assessment: Dict) -> Dict[str, Any]:
        """Generate the final trading signal recommendation"""
        try:
            # Base signal from combination
            base_direction = combined_signal.get('combined_direction', 'NEUTRAL')
            base_strength = combined_signal.get('combined_strength', 0.0)
            base_confidence = combined_signal.get('combined_confidence', 0.0)
            
            # Apply validation adjustments
            if not signal_validation.get('overall_validation', False):
                # Reduce strength and confidence for failed validation
                validation_score = signal_validation.get('validation_score', 0.0)
                base_strength *= validation_score
                base_confidence *= validation_score
                
                # Potentially change direction to neutral if very low validation
                if validation_score < 0.3:
                    base_direction = 'NEUTRAL'
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                base_direction, base_strength, base_confidence, risk_assessment
            )
            
            # Calculate position sizing suggestion
            position_sizing = self._calculate_position_sizing(base_strength, base_confidence, risk_assessment)
            
            # Generate stop loss and take profit suggestions
            stop_take_levels = self._calculate_stop_take_levels(individual_signals, base_direction, base_strength)
            
            # Final signal package
            final_signal = {
                'signal': base_direction,
                'strength': round(base_strength, 4),
                'confidence': round(base_confidence, 4),
                'action': self._determine_action(base_direction, base_strength, base_confidence),
                'recommendations': recommendations,
                'position_sizing': position_sizing,
                'stop_take_levels': stop_take_levels,
                'signal_quality': signal_validation.get('validation_score', 0.0),
                'risk_level': risk_assessment.get('risk_category', 'HIGH'),
                'confirming_indicators': self._list_confirming_indicators(individual_signals),
                'conflicting_indicators': self._list_conflicting_indicators(individual_signals),
                'signal_summary': self._generate_signal_summary(base_direction, base_strength, base_confidence)
            }
            
            return final_signal
            
        except Exception as e:
            self.logger.error(f"Final signal generation error: {e}")
            return {'signal': 'ERROR', 'strength': 0.0, 'confidence': 0.0, 'action': 'NO_ACTION'}
    
    def _signal_direction_to_value(self, direction: str) -> float:
        """Convert signal direction to numeric value"""
        try:
            direction_upper = direction.upper()
            
            if 'STRONG_BULLISH' in direction_upper:
                return 1.0
            elif 'BULLISH' in direction_upper and 'WEAK' not in direction_upper:
                return 0.7
            elif 'WEAK_BULLISH' in direction_upper:
                return 0.3
            elif 'STRONG_BEARISH' in direction_upper:
                return -1.0
            elif 'BEARISH' in direction_upper and 'WEAK' not in direction_upper:
                return -0.7
            elif 'WEAK_BEARISH' in direction_upper:
                return -0.3
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Direction conversion error: {e}")
            return 0.0
    
    def _count_confirming_signals(self, individual_signals: Dict) -> int:
        """Count signals that confirm the overall direction"""
        try:
            # Determine overall direction first
            signal_values = []
            for signal_data in individual_signals.values():
                if signal_data and 'signal' in signal_data:
                    value = self._signal_direction_to_value(signal_data['signal'])
                    if value != 0:
                        signal_values.append(value)
            
            if not signal_values:
                return 0
            
            # Overall direction
            overall_direction = 1 if sum(signal_values) > 0 else -1
            
            # Count confirming signals
            confirming_count = sum(1 for value in signal_values if (value > 0) == (overall_direction > 0))
            
            return confirming_count
            
        except Exception as e:
            self.logger.error(f"Confirming signals count error: {e}")
            return 0
    
    def _categorize_strength(self, strength: float) -> str:
        """Categorize signal strength"""
        try:
            if strength >= self.confirmation_settings['strong_signal_threshold']:
                return 'STRONG'
            elif strength >= self.confirmation_settings['medium_signal_threshold']:
                return 'MEDIUM'
            elif strength >= self.confirmation_settings['weak_signal_threshold']:
                return 'WEAK'
            else:
                return 'VERY_WEAK'
                
        except Exception as e:
            self.logger.error(f"Strength categorization error: {e}")
            return 'ERROR'
    
    def _calculate_validation_score(self, validation_results: Dict) -> float:
        """Calculate overall validation score"""
        try:
            score_components = []
            
            # Confirmation score
            if validation_results.get('sufficient_confirmations', False):
                score_components.append(1.0)
            else:
                confirmations = validation_results.get('confirmation_count', 0)
                required = self.confirmation_settings['minimum_confirmations']
                score_components.append(confirmations / required if required > 0 else 0.0)
            
            # Strength score
            if validation_results.get('strength_validation', {}).get('meets_minimum', False):
                score_components.append(1.0)
            else:
                score_components.append(0.3)
            
            # Risk score
            if validation_results.get('risk_validation', {}).get('risk_acceptable', False):
                score_components.append(1.0)
            else:
                score_components.append(0.2)
            
            # Conflict score
            if validation_results.get('conflict_validation', {}).get('acceptable_conflict', False):
                score_components.append(1.0)
            else:
                conflict_level = validation_results.get('conflict_validation', {}).get('conflict_level', 1.0)
                score_components.append(max(0.0, 1.0 - conflict_level))
            
            # Average score
            return sum(score_components) / len(score_components) if score_components else 0.0
            
        except Exception as e:
            self.logger.error(f"Validation score calculation error: {e}")
            return 0.0
    
    def _assess_timing_risk(self, additional_data: Optional[Dict]) -> float:
        """Assess timing-based risk factors"""
        try:
            risk_score = 0.0
            
            # Market hours risk
            current_time = datetime.now()
            hour = current_time.hour
            
            # High risk during low liquidity hours (example: 22:00-02:00 UTC)
            if hour >= 22 or hour <= 2:
                risk_score += 0.3
            
            # Medium risk during lunch hours (12:00-13:00 UTC)
            elif 12 <= hour <= 13:
                risk_score += 0.1
            
            # News impact risk
            if additional_data and additional_data.get('high_impact_news', False):
                risk_score += 0.4
            
            # Day of week risk (Friday afternoon, Sunday night)
            weekday = current_time.weekday()
            if weekday == 4 and hour >= 15:  # Friday afternoon
                risk_score += 0.2
            elif weekday == 6 and hour >= 20:  # Sunday night
                risk_score += 0.3
            
            return min(1.0, risk_score)
            
        except Exception as e:
            self.logger.error(f"Timing risk assessment error: {e}")
            return 0.5
    
    def _generate_recommendations(self, direction: str, strength: float, confidence: float, 
                                risk_assessment: Dict) -> List[str]:
        """Generate trading recommendations"""
        try:
            recommendations = []
            
            # Direction-based recommendations
            if direction != 'NEUTRAL' and direction != 'ERROR':
                if strength >= 0.7 and confidence >= 0.7:
                    recommendations.append(f"Strong {direction.lower()} signal - Consider full position")
                elif strength >= 0.5 and confidence >= 0.5:
                    recommendations.append(f"Moderate {direction.lower()} signal - Consider partial position")
                else:
                    recommendations.append(f"Weak {direction.lower()} signal - Consider small position or wait")
            else:
                recommendations.append("No clear directional signal - Avoid trading")
            
            # Risk-based recommendations
            risk_level = risk_assessment.get('risk_category', 'HIGH')
            if risk_level == 'HIGH':
                recommendations.append("High risk environment - Reduce position sizes")
            elif risk_level == 'MEDIUM':
                recommendations.append("Moderate risk - Use standard position sizing")
            else:
                recommendations.append("Low risk environment - Consider slightly larger positions")
            
            # Volatility recommendations
            volatility_risk = risk_assessment.get('risk_factors', {}).get('volatility', 0.5)
            if volatility_risk > 0.7:
                recommendations.append("High volatility detected - Use wider stops")
            
            # Liquidity recommendations
            liquidity_risk = risk_assessment.get('risk_factors', {}).get('liquidity', 0.7)
            if liquidity_risk < 0.3:
                recommendations.append("Low liquidity warning - Avoid large positions")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Recommendations generation error: {e}")
            return ["Error generating recommendations"]
    
    def _calculate_position_sizing(self, strength: float, confidence: float, risk_assessment: Dict) -> Dict[str, Any]:
        """Calculate suggested position sizing"""
        try:
            # Base position size (as percentage of account)
            base_size = 0.02  # 2% base risk
            
            # Adjust based on signal strength and confidence
            signal_multiplier = (strength + confidence) / 2
            
            # Adjust based on risk
            risk_multiplier = 1.0 - (risk_assessment.get('overall_risk', 0.5) * 0.5)
            
            # Calculate suggested position size
            suggested_size = base_size * signal_multiplier * risk_multiplier
            
            # Apply limits
            min_size = 0.005  # 0.5% minimum
            max_size = 0.05   # 5% maximum
            
            suggested_size = max(min_size, min(max_size, suggested_size))
            
            return {
                'suggested_position_percent': round(suggested_size * 100, 2),
                'risk_per_trade_percent': round(suggested_size * 100, 2),
                'position_sizing_rationale': f"Based on signal strength ({strength:.2f}), confidence ({confidence:.2f}), and risk level",
                'max_recommended_percent': round(max_size * 100, 1),
                'min_recommended_percent': round(min_size * 100, 1)
            }
            
        except Exception as e:
            self.logger.error(f"Position sizing calculation error: {e}")
            return {'suggested_position_percent': 1.0, 'risk_per_trade_percent': 1.0}
    
    def _calculate_stop_take_levels(self, individual_signals: Dict, direction: str, strength: float) -> Dict[str, Any]:
        """Calculate stop loss and take profit levels"""
        try:
            # Extract relevant levels from individual signals
            supertrend_levels = individual_signals.get('supertrend', {})
            bb_levels = individual_signals.get('bollinger', {})
            
            # Default ratios
            default_stop_ratio = 0.02  # 2% stop loss
            default_take_ratio = 0.04  # 2:1 risk/reward
            
            # Adjust based on signal strength
            stop_ratio = default_stop_ratio * (2.0 - strength)  # Stronger signals = tighter stops
            take_ratio = default_take_ratio * (1.0 + strength)  # Stronger signals = larger targets
            
            suggestions = {
                'stop_loss_percent': round(stop_ratio * 100, 2),
                'take_profit_percent': round(take_ratio * 100, 2),
                'risk_reward_ratio': round(take_ratio / stop_ratio, 1),
                'suggested_method': 'Percentage-based with signal strength adjustment'
            }
            
            # Add specific level suggestions if available
            if supertrend_levels and 'details' in supertrend_levels:
                suggestions['supertrend_stop'] = supertrend_levels['details'].get('supertrend_level', 0)
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Stop/Take levels calculation error: {e}")
            return {'stop_loss_percent': 2.0, 'take_profit_percent': 4.0, 'risk_reward_ratio': 2.0}
    
    def _determine_action(self, direction: str, strength: float, confidence: float) -> str:
        """Determine recommended trading action"""
        try:
            if direction in ['ERROR', 'NEUTRAL'] or strength < 0.2 or confidence < 0.3:
                return 'NO_ACTION'
            elif 'BULLISH' in direction and strength >= 0.5 and confidence >= 0.5:
                return 'BUY'
            elif 'BEARISH' in direction and strength >= 0.5 and confidence >= 0.5:
                return 'SELL'
            elif strength >= 0.3 and confidence >= 0.4:
                return 'WATCH'
            else:
                return 'NO_ACTION'
                
        except Exception as e:
            self.logger.error(f"Action determination error: {e}")
            return 'ERROR'
    
    def _list_confirming_indicators(self, individual_signals: Dict) -> List[str]:
        """List indicators that confirm the main signal"""
        try:
            # This would be implemented based on the actual signal directions
            # For now, return a placeholder
            confirming = []
            
            for analyzer_name, signal_data in individual_signals.items():
                if signal_data and signal_data.get('signal', 'NEUTRAL') not in ['NEUTRAL', 'ERROR']:
                    if signal_data.get('strength', 0) >= 0.4:
                        confirming.append(analyzer_name.title().replace('_', ' '))
            
            return confirming
            
        except Exception as e:
            self.logger.error(f"Confirming indicators listing error: {e}")
            return []
    
    def _list_conflicting_indicators(self, individual_signals: Dict) -> List[str]:
        """List indicators that conflict with the main signal"""
        try:
            # This would be implemented based on the actual signal directions
            # For now, return a placeholder
            conflicting = []
            
            # Simplified conflict detection
            signal_directions = {}
            for analyzer_name, signal_data in individual_signals.items():
                if signal_data and 'signal' in signal_data:
                    direction = signal_data['signal']
                    if direction not in ['NEUTRAL', 'ERROR']:
                        signal_directions[analyzer_name] = direction
            
            # Find conflicts (this is simplified)
            bullish_analyzers = [name for name, direction in signal_directions.items() if 'BULLISH' in direction]
            bearish_analyzers = [name for name, direction in signal_directions.items() if 'BEARISH' in direction]
            
            if len(bullish_analyzers) > 0 and len(bearish_analyzers) > 0:
                # There are conflicts
                if len(bullish_analyzers) >= len(bearish_analyzers):
                    conflicting = [name.title().replace('_', ' ') for name in bearish_analyzers]
                else:
                    conflicting = [name.title().replace('_', ' ') for name in bullish_analyzers]
            
            return conflicting
            
        except Exception as e:
            self.logger.error(f"Conflicting indicators listing error: {e}")
            return []
    
    def _generate_signal_summary(self, direction: str, strength: float, confidence: float) -> str:
        """Generate human-readable signal summary"""
        try:
            if direction == 'ERROR':
                return "Signal generation error - Unable to analyze market conditions"
            elif direction == 'NEUTRAL':
                return "No clear directional bias - Market appears neutral or conflicted"
            else:
                strength_desc = self._categorize_strength(strength).lower()
                confidence_desc = "high" if confidence >= 0.7 else "moderate" if confidence >= 0.5 else "low"
                
                action = "bullish" if 'BULLISH' in direction else "bearish"
                
                return f"{strength_desc.title()} {action} signal with {confidence_desc} confidence"
                
        except Exception as e:
            self.logger.error(f"Signal summary generation error: {e}")
            return "Error generating signal summary"
    
    def _generate_signal_id(self) -> str:
        """Generate unique signal ID"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"SIG_{timestamp}_{len(self.signal_history):04d}"
        except Exception as e:
            self.logger.error(f"Signal ID generation error: {e}")
            return f"SIG_ERROR_{datetime.now().timestamp()}"
    
    def _update_signal_tracking(self, signal_result: Dict):
        """Update signal tracking and performance statistics"""
        try:
            # Add to signal history
            self.signal_history.append({
                'signal_id': signal_result.get('signal_id', ''),
                'signal': signal_result.get('signal', 'UNKNOWN'),
                'strength': signal_result.get('strength', 0.0),
                'confidence': signal_result.get('confidence', 0.0),
                'timestamp': signal_result.get('generation_time', ''),
                'action': signal_result.get('action', 'NO_ACTION')
            })
            
            # Limit history size
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
            
            # Update statistics
            self.performance_stats['total_signals'] += 1
            
            # Update averages
            total = self.performance_stats['total_signals']
            current_avg_strength = self.performance_stats['avg_strength']
            current_avg_confidence = self.performance_stats['avg_confidence']
            
            new_strength = signal_result.get('strength', 0.0)
            new_confidence = signal_result.get('confidence', 0.0)
            
            self.performance_stats['avg_strength'] = (
                (current_avg_strength * (total - 1) + new_strength) / total
            )
            self.performance_stats['avg_confidence'] = (
                (current_avg_confidence * (total - 1) + new_confidence) / total
            )
            
        except Exception as e:
            self.logger.error(f"Signal tracking update error: {e}")
    
    def _assess_data_quality(self, price_data: pd.DataFrame) -> float:
        """Assess quality of input data"""
        try:
            quality_score = 1.0
            
            # Check for missing data
            if price_data.isnull().any().any():
                quality_score -= 0.2
            
            # Check data length
            if len(price_data) < 100:
                quality_score -= 0.3
            elif len(price_data) < 200:
                quality_score -= 0.1
            
            # Check for unrealistic price movements (gaps > 10%)
            price_changes = price_data['close'].pct_change().abs()
            large_gaps = (price_changes > 0.1).sum()
            if large_gaps > 0:
                quality_score -= min(0.3, large_gaps * 0.1)
            
            # Check for constant prices (no movement)
            if price_changes.max() == 0:
                quality_score -= 0.5
            
            return max(0.0, quality_score)
            
        except Exception as e:
            self.logger.error(f"Data quality assessment error: {e}")
            return 0.5
    
    def _assess_signal_consistency(self, signal_result: Dict) -> float:
        """Assess consistency of signal components"""
        try:
            individual_signals = signal_result.get('individual_signals', {})
            
            if not individual_signals:
                return 0.0
            
            # Check signal direction consistency
            directions = []
            for signal_data in individual_signals.values():
                if signal_data and 'signal' in signal_data:
                    direction = signal_data['signal']
                    if direction not in ['NEUTRAL', 'ERROR']:
                        direction_value = self._signal_direction_to_value(direction)
                        directions.append(direction_value)
            
            if not directions:
                return 0.0
            
            # Calculate consistency (how aligned are the directions)
            avg_direction = np.mean(directions)
            consistency = 1.0 - (np.std(directions) / 2.0)  # Normalize std to 0-1 range
            
            return max(0.0, min(1.0, consistency))
            
        except Exception as e:
            self.logger.error(f"Signal consistency assessment error: {e}")
            return 0.0
    
    def _assess_confirmation_strength(self, signal_result: Dict) -> float:
        """Assess strength of signal confirmations"""
        try:
            validation = signal_result.get('signal_validation', {})
            
            confirmation_count = validation.get('confirmation_count', 0)
            required_confirmations = self.confirmation_settings['minimum_confirmations']
            
            # Base score from confirmation count
            confirmation_score = min(1.0, confirmation_count / required_confirmations)
            
            # Adjust based on signal strength
            signal_strength = signal_result.get('strength', 0.0)
            strength_bonus = signal_strength * 0.2
            
            total_score = confirmation_score + strength_bonus
            
            return min(1.0, total_score)
            
        except Exception as e:
            self.logger.error(f"Confirmation strength assessment error: {e}")
            return 0.0
    
    def _assess_risk_level(self, signal_result: Dict) -> float:
        """Assess risk level (lower risk = higher score)"""
        try:
            risk_assessment = signal_result.get('risk_assessment', {})
            overall_risk = risk_assessment.get('overall_risk', 1.0)
            
            # Convert risk to quality score (inverse relationship)
            risk_quality_score = 1.0 - overall_risk
            
            return max(0.0, risk_quality_score)
            
        except Exception as e:
            self.logger.error(f"Risk level assessment error: {e}")
            return 0.0
    
    def _assess_timing_quality(self, price_data: pd.DataFrame, signal_result: Dict) -> float:
        """Assess timing quality of the signal"""
        try:
            # Simple timing quality based on recent volatility and trend
            if len(price_data) < 20:
                return 0.5
            
            # Recent volatility
            recent_volatility = price_data['close'][-10:].pct_change().std()
            
            # Timing is better with moderate volatility (not too high, not too low)
            optimal_volatility = 0.02  # 2% daily volatility
            volatility_score = 1.0 - abs(recent_volatility - optimal_volatility) / optimal_volatility
            volatility_score = max(0.0, min(1.0, volatility_score))
            
            # Add trend momentum factor
            price_momentum = (price_data['close'].iloc[-1] - price_data['close'].iloc[-5]) / price_data['close'].iloc[-5]
            momentum_score = min(1.0, abs(price_momentum) * 10)  # Scale momentum
            
            # Combined timing score
            timing_score = (volatility_score * 0.6) + (momentum_score * 0.4)
            
            return timing_score
            
        except Exception as e:
            self.logger.error(f"Timing quality assessment error: {e}")
            return 0.5
    
    def _generate_quality_recommendations(self, quality_scores: Dict, overall_quality: float) -> List[str]:
        """Generate recommendations for improving signal quality"""
        try:
            recommendations = []
            
            # Data quality recommendations
            if quality_scores.get('data_quality', 1.0) < 0.7:
                recommendations.append("Improve data quality - check for missing or invalid data points")
            
            # Signal consistency recommendations
            if quality_scores.get('signal_consistency', 1.0) < 0.6:
                recommendations.append("Low signal consistency - consider waiting for clearer market conditions")
            
            # Confirmation recommendations
            if quality_scores.get('confirmation_strength', 1.0) < 0.6:
                recommendations.append("Weak confirmations - wait for more indicators to align")
            
            # Risk recommendations
            if quality_scores.get('risk_level', 1.0) < 0.5:
                recommendations.append("High risk environment - reduce position sizes or avoid trading")
            
            # Timing recommendations
            if quality_scores.get('timing_quality', 1.0) < 0.5:
                recommendations.append("Poor timing conditions - consider waiting for better market conditions")
            
            # Overall recommendations
            if overall_quality < 0.4:
                recommendations.append("Overall signal quality is poor - strongly recommend avoiding this trade")
            elif overall_quality < 0.6:
                recommendations.append("Signal quality is moderate - proceed with caution and reduced position size")
            
            return recommendations if recommendations else ["Signal quality is acceptable"]
            
        except Exception as e:
            self.logger.error(f"Quality recommendations generation error: {e}")
            return ["Error generating quality recommendations"]
    
    def _calculate_recent_performance(self, recent_signals: List[Dict]) -> Dict[str, Any]:
        """Calculate performance metrics for recent signals"""
        try:
            if not recent_signals:
                return {'recent_win_rate': 0.0, 'recent_avg_strength': 0.0, 'signal_count': 0}
            
            # Calculate recent averages
            strengths = [s.get('strength', 0.0) for s in recent_signals]
            confidences = [s.get('confidence', 0.0) for s in recent_signals]
            
            return {
                'recent_win_rate': 0.0,  # Would need actual trade results to calculate
                'recent_avg_strength': np.mean(strengths) if strengths else 0.0,
                'recent_avg_confidence': np.mean(confidences) if confidences else 0.0,
                'signal_count': len(recent_signals),
                'strong_signals': sum(1 for s in strengths if s >= 0.7),
                'weak_signals': sum(1 for s in strengths if s < 0.4)
            }
            
        except Exception as e:
            self.logger.error(f"Recent performance calculation error: {e}")
            return {'recent_win_rate': 0.0, 'recent_avg_strength': 0.0, 'signal_count': 0}
    
    def _analyze_signal_distribution(self) -> Dict[str, Any]:
        """Analyze distribution of generated signals"""
        try:
            if not self.signal_history:
                return {'distribution': {}, 'total_count': 0}
            
            # Count signal types
            signal_counts = {}
            for signal in self.signal_history:
                signal_type = signal.get('signal', 'UNKNOWN')
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            total_signals = len(self.signal_history)
            
            # Calculate percentages
            distribution = {}
            for signal_type, count in signal_counts.items():
                distribution[signal_type] = {
                    'count': count,
                    'percentage': round((count / total_signals) * 100, 2)
                }
            
            return {
                'distribution': distribution,
                'total_count': total_signals,
                'most_common': max(signal_counts.items(), key=lambda x: x[1])[0] if signal_counts else 'NONE'
            }
            
        except Exception as e:
            self.logger.error(f"Signal distribution analysis error: {e}")
            return {'distribution': {}, 'total_count': 0}
    
    def _analyze_analyzer_performance(self) -> Dict[str, Any]:
        """Analyze performance of individual analyzers"""
        try:
            # This would require tracking individual analyzer success rates
            # For now, return placeholder data
            return {
                'analyzer_contribution': {
                    analyzer: {'weight': weight, 'effectiveness': 0.7}  # Placeholder
                    for analyzer, weight in self.signal_weights.items()
                },
                'best_performer': 'supertrend',  # Placeholder
                'worst_performer': 'tdi'  # Placeholder
            }
            
        except Exception as e:
            self.logger.error(f"Analyzer performance analysis error: {e}")
            return {'analyzer_contribution': {}}
    
    def _generate_improvement_suggestions(self) -> List[str]:
        """Generate suggestions for improving signal generation"""
        try:
            suggestions = []
            
            # Based on performance stats
            if self.performance_stats['avg_strength'] < 0.5:
                suggestions.append("Consider adjusting signal strength thresholds")
            
            if self.performance_stats['avg_confidence'] < 0.6:
                suggestions.append("Review confirmation requirements to improve confidence")
            
            # General suggestions
            suggestions.extend([
                "Regularly review and adjust analyzer weights based on market conditions",
                "Consider adding market regime detection for adaptive signal generation",
                "Implement machine learning for dynamic weight optimization"
            ])
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Improvement suggestions generation error: {e}")
            return ["Error generating improvement suggestions"]
    
    def _calculate_signal_consensus(self, individual_signals: Dict) -> Dict[str, Any]:
        """Calculate consensus among signals"""
        try:
            signal_values = []
            for signal_data in individual_signals.values():
                if signal_data and 'signal' in signal_data:
                    value = self._signal_direction_to_value(signal_data['signal'])
                    if value != 0:
                        signal_values.append(value)
            
            if not signal_values:
                return {'consensus_level': 0.0, 'consensus_direction': 'NONE'}
            
            # Calculate consensus
            avg_signal = np.mean(signal_values)
            signal_std = np.std(signal_values)
            
            # Consensus level (higher when signals agree)
            consensus_level = 1.0 - (signal_std / 2.0)  # Normalize std
            consensus_level = max(0.0, min(1.0, consensus_level))
            
            # Consensus direction
            if avg_signal > 0.2:
                consensus_direction = 'BULLISH'
            elif avg_signal < -0.2:
                consensus_direction = 'BEARISH'
            else:
                consensus_direction = 'NEUTRAL'
            
            return {
                'consensus_level': round(consensus_level, 4),
                'consensus_direction': consensus_direction,
                'signal_count': len(signal_values),
                'average_signal_value': round(avg_signal, 4)
            }
            
        except Exception as e:
            self.logger.error(f"Signal consensus calculation error: {e}")
            return {'consensus_level': 0.0, 'consensus_direction': 'ERROR'}
    
    def _analyze_signal_conflicts(self, individual_signals: Dict) -> Dict[str, Any]:
        """Analyze conflicts between signals"""
        try:
            bullish_signals = []
            bearish_signals = []
            neutral_signals = []
            
            for analyzer_name, signal_data in individual_signals.items():
                if not signal_data or 'signal' not in signal_data:
                    continue
                
                signal = signal_data['signal']
                if 'BULLISH' in signal:
                    bullish_signals.append(analyzer_name)
                elif 'BEARISH' in signal:
                    bearish_signals.append(analyzer_name)
                else:
                    neutral_signals.append(analyzer_name)
            
            total_directional = len(bullish_signals) + len(bearish_signals)
            
            if total_directional == 0:
                conflict_level = 0.0
                conflict_type = 'NO_DIRECTIONAL_SIGNALS'
            else:
                minority_count = min(len(bullish_signals), len(bearish_signals))
                conflict_level = minority_count / total_directional
                
                if conflict_level == 0.0:
                    conflict_type = 'NO_CONFLICT'
                elif conflict_level < 0.3:
                    conflict_type = 'LOW_CONFLICT'
                elif conflict_level < 0.5:
                    conflict_type = 'MODERATE_CONFLICT'
                else:
                    conflict_type = 'HIGH_CONFLICT'
            
            return {
                'conflict_level': round(conflict_level, 4),
                'conflict_type': conflict_type,
                'bullish_count': len(bullish_signals),
                'bearish_count': len(bearish_signals),
                'neutral_count': len(neutral_signals),
                'bullish_analyzers': bullish_signals,
                'bearish_analyzers': bearish_signals,
                'conflicting_pairs': self._identify_conflicting_pairs(bullish_signals, bearish_signals)
            }
            
        except Exception as e:
            self.logger.error(f"Signal conflict analysis error: {e}")
            return {'conflict_level': 1.0, 'conflict_type': 'ERROR'}
    
    def _identify_conflicting_pairs(self, bullish_signals: List[str], bearish_signals: List[str]) -> List[Tuple[str, str]]:
        """Identify pairs of conflicting signals"""
        try:
            conflicting_pairs = []
            
            # Create pairs of conflicting analyzers
            for bullish in bullish_signals:
                for bearish in bearish_signals:
                    conflicting_pairs.append((bullish, bearish))
            
            return conflicting_pairs
            
        except Exception as e:
            self.logger.error(f"Conflicting pairs identification error: {e}")
            return []
    
    def _empty_signal_result(self) -> Dict[str, Any]:
        """Return empty signal result"""
        return {
            'signal': 'ERROR',
            'strength': 0.0,
            'confidence': 0.0,
            'action': 'NO_ACTION',
            'recommendations': ['Signal generation failed'],
            'position_sizing': {'suggested_position_percent': 0.0},
            'stop_take_levels': {'stop_loss_percent': 2.0, 'take_profit_percent': 4.0},
            'signal_quality': 0.0,
            'risk_level': 'HIGH',
            'confirming_indicators': [],
            'conflicting_indicators': [],
            'signal_summary': 'Error generating signal',
            'generation_time': datetime.now().isoformat(),
            'error': 'Signal generation failed'
        }

# Usage Example (for testing):
if __name__ == "__main__":
    # Sample usage
    generator = SignalGenerator()
    
    # Sample OHLC data
    sample_data = pd.DataFrame({
        'high': np.random.rand(200) * 100 + 1000,
        'low': np.random.rand(200) * 100 + 950,
        'close': np.random.rand(200) * 100 + 975,
        'open': np.random.rand(200) * 100 + 975,
        'volume': np.random.rand(200) * 1000 + 500
    })
    
    # Generate unified signal
    signal = generator.generate_signal(sample_data)
    print("Unified Signal:", json.dumps(signal, indent=2, default=str))
    
    # Get signal strength breakdown
    breakdown = generator.get_signal_strength_breakdown(sample_data)
    print("Signal Breakdown:", breakdown)
    
    # Validate signal quality
    quality = generator.validate_signal_quality(sample_data)
    print("Signal Quality:", quality)
    
    # Get performance statistics
    performance = generator.get_performance_statistics()
    print("Performance Stats:", performance)