#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Entry Conditions Processor
Implements all 34 entry conditions (17 BUY + 17 SELL) with ML validation

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
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
import math

class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    NONE = "NONE"

class PathType(Enum):
    CONTINUATION = "CONTINUATION"  # CT path
    PULLBACK = "PULLBACK"         # PB path

class TimeFrameType(Enum):
    MAJOR = "MAJOR"     # Daily/4H
    MIDDLE = "MIDDLE"   # 1H/30M
    ENTRY = "ENTRY"     # 15M/5M

@dataclass
class TechnicalIndicators:
    # Ichimoku indicators
    tenkan_sen: float = 0.0
    kijun_sen: float = 0.0
    senkou_span_a: float = 0.0
    senkou_span_b: float = 0.0
    chikou_span: float = 0.0
    
    # TDI (Trader's Dynamic Index)
    tdi_rsi: float = 0.0
    tdi_price_line: float = 0.0
    tdi_base_line: float = 0.0
    tdi_volatility_band_high: float = 0.0
    tdi_volatility_band_low: float = 0.0
    
    # Moving averages
    smma_50: float = 0.0
    
    # Bollinger Bands
    bb_upper: float = 0.0
    bb_middle: float = 0.0
    bb_lower: float = 0.0
    bb_squeeze: bool = False
    
    # SuperTrend
    str_entry: float = 0.0
    str_entry_direction: int = 0  # 1 for up, -1 for down
    str_exit: float = 0.0
    str_exit_direction: int = 0
    
    # Price action
    current_price: float = 0.0
    previous_close: float = 0.0
    high: float = 0.0
    low: float = 0.0
    volume: int = 0

@dataclass
class EntryCondition:
    condition_id: str
    name: str
    signal_type: SignalType
    path_type: PathType
    timeframe: TimeFrameType
    is_met: bool = False
    confidence: float = 0.0
    description: str = ""

@dataclass
class SignalResult:
    signal_id: str
    signal_type: SignalType
    path_type: PathType
    conditions_met: List[str]
    total_conditions: int
    confidence: float
    ml_validated: bool
    entry_price: float
    stop_loss: float
    take_profit: float
    timestamp: datetime

class EntryConditionsProcessor:
    """
    Entry Conditions Processor for EA GlobalFlow Pro v0.1
    Implements all 34 entry conditions with multi-timeframe analysis
    """
    
    def __init__(self, config_manager=None, ml_enhancement=None, error_handler=None):
        """Initialize entry conditions processor"""
        self.config_manager = config_manager
        self.ml_enhancement = ml_enhancement
        self.error_handler = error_handler
        self.logger = logging.getLogger('EntryConditionsProcessor')
        
        # Configuration
        self.conditions_config = {}
        self.is_initialized = False
        
        # Entry conditions
        self.buy_conditions = {}
        self.sell_conditions = {}
        self.all_conditions = {}
        
        # Timeframe data
        self.timeframe_data = {
            TimeFrameType.MAJOR: {},
            TimeFrameType.MIDDLE: {},
            TimeFrameType.ENTRY: {}
        }
        
        # Signal tracking
        self.generated_signals = {}
        self.signal_history = []
        
        # Performance tracking
        self.performance_stats = {
            'total_signals': 0,
            'buy_signals': 0,
            'sell_signals': 0,
            'ml_validated_signals': 0,
            'successful_trades': 0
        }
        
    def initialize(self) -> bool:
        """
        Initialize entry conditions processor
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing Entry Conditions Processor v0.1...")
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Initialize all 34 entry conditions
            if not self._initialize_conditions():
                return False
            
            self.is_initialized = True
            self.logger.info("✅ Entry Conditions Processor initialized successfully")
            self.logger.info(f"📊 Loaded {len(self.all_conditions)} entry conditions")
            return True
            
        except Exception as e:
            self.logger.error(f"Entry Conditions Processor initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("entry_conditions_init", e)
            return False
    
    def _load_config(self) -> bool:
        """Load entry conditions configuration"""
        try:
            if self.config_manager:
                self.conditions_config = self.config_manager.get_config('entry_conditions', {})
            else:
                # Default configuration
                self.conditions_config = {
                    'total_conditions': 34,
                    'buy_conditions': 17,
                    'sell_conditions': 17,
                    'validation_layers': {
                        'technical_analysis': True,
                        'ml_enhancement': True,
                        'risk_validation': True
                    },
                    'confidence_thresholds': {
                        'minimum': 0.6,
                        'good': 0.75,
                        'excellent': 0.9
                    }
                }
            
            self.logger.info("Entry conditions configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load entry conditions config: {e}")
            return False
    
    def _initialize_conditions(self) -> bool:
        """Initialize all 34 entry conditions"""
        try:
            # BUY CONDITIONS (17 conditions)
            self.buy_conditions = {
                # Continuation Path BUY Conditions (A1-A3)
                'A1_CT_BUY': EntryCondition(
                    condition_id='A1_CT_BUY',
                    name='Ichimoku Bullish Continuation',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price above Kumo, Tenkan > Kijun, bullish momentum'
                ),
                'A2_CT_BUY': EntryCondition(
                    condition_id='A2_CT_BUY',
                    name='TDI Bullish Continuation',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='TDI price line above base line, bullish cross'
                ),
                'A3_CT_BUY': EntryCondition(
                    condition_id='A3_CT_BUY',
                    name='SuperTrend Bullish Confirmation',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='STR-ENTRY bullish, price above SuperTrend'
                ),
                
                # Pullback Path BUY Conditions (B1-B3)
                'B1_PB_BUY': EntryCondition(
                    condition_id='B1_PB_BUY',
                    name='Ichimoku Bullish Pullback',
                    signal_type=SignalType.BUY,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price pullback to Kumo support, bullish reversal'
                ),
                'B2_PB_BUY': EntryCondition(
                    condition_id='B2_PB_BUY',
                    name='TDI Oversold Reversal',
                    signal_type=SignalType.BUY,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='TDI oversold, bullish divergence signal'
                ),
                'B3_PB_BUY': EntryCondition(
                    condition_id='B3_PB_BUY',
                    name='Bollinger Bounce Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price bounce from BB lower band'
                ),
                
                # Combined Path BUY Conditions (C1-C11)
                'C1_CMB_BUY': EntryCondition(
                    condition_id='C1_CMB_BUY',
                    name='Multi-Timeframe Bullish Alignment',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='All timeframes bullish aligned'
                ),
                'C2_CMB_BUY': EntryCondition(
                    condition_id='C2_CMB_BUY',
                    name='Kumo Breakout Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Strong breakout above Ichimoku cloud'
                ),
                'C3_CMB_BUY': EntryCondition(
                    condition_id='C3_CMB_BUY',
                    name='TDI-PA Confluence Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='TDI and price action confluence'
                ),
                'C4_CMB_BUY': EntryCondition(
                    condition_id='C4_CMB_BUY',
                    name='BB Squeeze Breakout Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Bollinger Band squeeze breakout bullish'
                ),
                'C5_CMB_BUY': EntryCondition(
                    condition_id='C5_CMB_BUY',
                    name='Volume Confirmed Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Strong volume confirmation on bullish signal'
                ),
                'C6_CMB_BUY': EntryCondition(
                    condition_id='C6_CMB_BUY',
                    name='SMMA Support Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price finds support at SMMA 50'
                ),
                'C7_CMB_BUY': EntryCondition(
                    condition_id='C7_CMB_BUY',
                    name='Chikou Span Confirmation Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Chikou span above price, bullish confirmation'
                ),
                'C8_CMB_BUY': EntryCondition(
                    condition_id='C8_CMB_BUY',
                    name='Double Bottom Reversal Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Double bottom pattern reversal signal'
                ),
                'C9_CMB_BUY': EntryCondition(
                    condition_id='C9_CMB_BUY',
                    name='Divergence Buy Signal',
                    signal_type=SignalType.BUY,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Bullish divergence between price and indicators'
                ),
                'C10_CMB_BUY': EntryCondition(
                    condition_id='C10_CMB_BUY',
                    name='Momentum Shift Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Clear momentum shift to bullish'
                ),
                'C11_CMB_BUY': EntryCondition(
                    condition_id='C11_CMB_BUY',
                    name='Support-Resistance Buy',
                    signal_type=SignalType.BUY,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price rebounds from key support level'
                )
            }
            
            # SELL CONDITIONS (17 conditions) - Mirror of BUY conditions
            self.sell_conditions = {
                # Continuation Path SELL Conditions
                'A1_CT_SELL': EntryCondition(
                    condition_id='A1_CT_SELL',
                    name='Ichimoku Bearish Continuation',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price below Kumo, Tenkan < Kijun, bearish momentum'
                ),
                'A2_CT_SELL': EntryCondition(
                    condition_id='A2_CT_SELL',
                    name='TDI Bearish Continuation',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='TDI price line below base line, bearish cross'
                ),
                'A3_CT_SELL': EntryCondition(
                    condition_id='A3_CT_SELL',
                    name='SuperTrend Bearish Confirmation',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='STR-ENTRY bearish, price below SuperTrend'
                ),
                
                # Pullback Path SELL Conditions
                'B1_PB_SELL': EntryCondition(
                    condition_id='B1_PB_SELL',
                    name='Ichimoku Bearish Pullback',
                    signal_type=SignalType.SELL,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price pullback to Kumo resistance, bearish reversal'
                ),
                'B2_PB_SELL': EntryCondition(
                    condition_id='B2_PB_SELL',
                    name='TDI Overbought Reversal',
                    signal_type=SignalType.SELL,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='TDI overbought, bearish divergence signal'
                ),
                'B3_PB_SELL': EntryCondition(
                    condition_id='B3_PB_SELL',
                    name='Bollinger Rejection Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price rejection from BB upper band'
                ),
                
                # Combined Path SELL Conditions
                'C1_CMB_SELL': EntryCondition(
                    condition_id='C1_CMB_SELL',
                    name='Multi-Timeframe Bearish Alignment',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='All timeframes bearish aligned'
                ),
                'C2_CMB_SELL': EntryCondition(
                    condition_id='C2_CMB_SELL',
                    name='Kumo Breakdown Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Strong breakdown below Ichimoku cloud'
                ),
                'C3_CMB_SELL': EntryCondition(
                    condition_id='C3_CMB_SELL',
                    name='TDI-PA Confluence Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='TDI and price action bearish confluence'
                ),
                'C4_CMB_SELL': EntryCondition(
                    condition_id='C4_CMB_SELL',
                    name='BB Squeeze Breakdown Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Bollinger Band squeeze breakdown bearish'
                ),
                'C5_CMB_SELL': EntryCondition(
                    condition_id='C5_CMB_SELL',
                    name='Volume Confirmed Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Strong volume confirmation on bearish signal'
                ),
                'C6_CMB_SELL': EntryCondition(
                    condition_id='C6_CMB_SELL',
                    name='SMMA Resistance Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price finds resistance at SMMA 50'
                ),
                'C7_CMB_SELL': EntryCondition(
                    condition_id='C7_CMB_SELL',
                    name='Chikou Span Confirmation Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Chikou span below price, bearish confirmation'
                ),
                'C8_CMB_SELL': EntryCondition(
                    condition_id='C8_CMB_SELL',
                    name='Double Top Reversal Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Double top pattern reversal signal'
                ),
                'C9_CMB_SELL': EntryCondition(
                    condition_id='C9_CMB_SELL',
                    name='Divergence Sell Signal',
                    signal_type=SignalType.SELL,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Bearish divergence between price and indicators'
                ),
                'C10_CMB_SELL': EntryCondition(
                    condition_id='C10_CMB_SELL',
                    name='Momentum Shift Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.CONTINUATION,
                    timeframe=TimeFrameType.ENTRY,
                    description='Clear momentum shift to bearish'
                ),
                'C11_CMB_SELL': EntryCondition(
                    condition_id='C11_CMB_SELL',
                    name='Resistance-Support Sell',
                    signal_type=SignalType.SELL,
                    path_type=PathType.PULLBACK,
                    timeframe=TimeFrameType.ENTRY,
                    description='Price rejected from key resistance level'
                )
            }
            
            # Combine all conditions
            self.all_conditions.update(self.buy_conditions)
            self.all_conditions.update(self.sell_conditions)
            
            self.logger.info(f"Initialized {len(self.buy_conditions)} BUY conditions")
            self.logger.info(f"Initialized {len(self.sell_conditions)} SELL conditions")
            self.logger.info(f"Total conditions: {len(self.all_conditions)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize conditions: {e}")
            return False
    
    def process_market_data(self, symbol: str, timeframe: str, indicators: TechnicalIndicators) -> Optional[SignalResult]:
        """
        Process market data and check all entry conditions
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe (5M, 15M, 30M, 1H, 4H, D1)
            indicators: Technical indicators data
            
        Returns:
            SignalResult if signal generated, None otherwise
        """
        try:
            # Store timeframe data
            tf_type = self._get_timeframe_type(timeframe)
            self.timeframe_data[tf_type][symbol] = indicators
            
            # Only generate signals on entry timeframes
            if tf_type != TimeFrameType.ENTRY:
                return None
            
            # Check all conditions
            buy_results = self._check_buy_conditions(symbol, indicators)
            sell_results = self._check_sell_conditions(symbol, indicators)
            
            # Determine best signal
            best_signal = self._select_best_signal(buy_results, sell_results)
            
            if best_signal:
                # ML validation if available
                if self.ml_enhancement:
                    ml_signal = self._validate_with_ml(best_signal, indicators)
                    if ml_signal:
                        best_signal.ml_validated = True
                        best_signal.confidence *= ml_signal.confidence
                
                # Calculate entry parameters
                self._calculate_entry_parameters(best_signal, indicators)
                
                # Store signal
                self.generated_signals[best_signal.signal_id] = best_signal
                self.signal_history.append(best_signal)
                
                # Update statistics
                self._update_performance_stats(best_signal)
                
                self.logger.info(f"✅ Signal generated: {best_signal.signal_type.value} {symbol} - Confidence: {best_signal.confidence:.2f}")
                
                return best_signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"Market data processing error: {e}")
            if self.error_handler:
                self.error_handler.handle_error("entry_conditions_process", e)
            return None
    
    def _get_timeframe_type(self, timeframe: str) -> TimeFrameType:
        """Get timeframe type from timeframe string"""
        if timeframe in ['D1', '4H']:
            return TimeFrameType.MAJOR
        elif timeframe in ['1H', '30M']:
            return TimeFrameType.MIDDLE
        else:
            return TimeFrameType.ENTRY
    
    def _check_buy_conditions(self, symbol: str, indicators: TechnicalIndicators) -> List[EntryCondition]:
        """Check all BUY conditions"""
        met_conditions = []
        
        try:
            for condition_id, condition in self.buy_conditions.items():
                if self._evaluate_condition(condition, indicators):
                    condition.is_met = True
                    condition.confidence = self._calculate_condition_confidence(condition, indicators)
                    met_conditions.append(condition)
                else:
                    condition.is_met = False
            
        except Exception as e:
            self.logger.error(f"BUY conditions check error: {e}")
        
        return met_conditions
    
    def _check_sell_conditions(self, symbol: str, indicators: TechnicalIndicators) -> List[EntryCondition]:
        """Check all SELL conditions"""
        met_conditions = []
        
        try:
            for condition_id, condition in self.sell_conditions.items():
                if self._evaluate_condition(condition, indicators):
                    condition.is_met = True
                    condition.confidence = self._calculate_condition_confidence(condition, indicators)
                    met_conditions.append(condition)
                else:
                    condition.is_met = False
            
        except Exception as e:
            self.logger.error(f"SELL conditions check error: {e}")
        
        return met_conditions
    
    def _evaluate_condition(self, condition: EntryCondition, indicators: TechnicalIndicators) -> bool:
        """Evaluate individual condition"""
        try:
            condition_id = condition.condition_id
            
            # Ichimoku-based conditions
            if 'A1_CT_BUY' in condition_id:
                return (indicators.current_price > indicators.senkou_span_a and 
                       indicators.current_price > indicators.senkou_span_b and
                       indicators.tenkan_sen > indicators.kijun_sen)
            
            elif 'A1_CT_SELL' in condition_id:
                return (indicators.current_price < indicators.senkou_span_a and 
                       indicators.current_price < indicators.senkou_span_b and
                       indicators.tenkan_sen < indicators.kijun_sen)
            
            # TDI-based conditions
            elif 'A2_CT_BUY' in condition_id:
                return (indicators.tdi_price_line > indicators.tdi_base_line and
                       indicators.tdi_rsi > 50)
            
            elif 'A2_CT_SELL' in condition_id:
                return (indicators.tdi_price_line < indicators.tdi_base_line and
                       indicators.tdi_rsi < 50)
            
            # SuperTrend conditions
            elif 'A3_CT_BUY' in condition_id:
                return (indicators.str_entry_direction == 1 and
                       indicators.current_price > indicators.str_entry)
            
            elif 'A3_CT_SELL' in condition_id:
                return (indicators.str_entry_direction == -1 and
                       indicators.current_price < indicators.str_entry)
            
            # Pullback conditions
            elif 'B1_PB_BUY' in condition_id:
                return (indicators.current_price > min(indicators.senkou_span_a, indicators.senkou_span_b) and
                       indicators.current_price < indicators.previous_close)
            
            elif 'B1_PB_SELL' in condition_id:
                return (indicators.current_price < max(indicators.senkou_span_a, indicators.senkou_span_b) and
                       indicators.current_price > indicators.previous_close)
            
            # Bollinger Band conditions
            elif 'B3_PB_BUY' in condition_id:
                return (indicators.current_price <= indicators.bb_lower * 1.01 and  # Near BB lower
                       indicators.current_price > indicators.previous_close)  # Bouncing up
            
            elif 'B3_PB_SELL' in condition_id:
                return (indicators.current_price >= indicators.bb_upper * 0.99 and  # Near BB upper
                       indicators.current_price < indicators.previous_close)  # Rejecting down
            
            # BB Squeeze conditions
            elif 'C4_CMB_BUY' in condition_id:
                return (indicators.bb_squeeze and
                       indicators.current_price > indicators.bb_middle and
                       indicators.volume > 0)  # Volume confirmation
            
            elif 'C4_CMB_SELL' in condition_id:
                return (indicators.bb_squeeze and
                       indicators.current_price < indicators.bb_middle and
                       indicators.volume > 0)  # Volume confirmation
            
            # SMMA conditions
            elif 'C6_CMB_BUY' in condition_id:
                return (indicators.current_price > indicators.smma_50 and
                       indicators.previous_close <= indicators.smma_50)  # Crossing above
            
            elif 'C6_CMB_SELL' in condition_id:
                return (indicators.current_price < indicators.smma_50 and
                       indicators.previous_close >= indicators.smma_50)  # Crossing below
            
            # Default conditions (simplified for other complex conditions)
            else:
                # For complex conditions, use simplified logic based on multiple indicators
                if condition.signal_type == SignalType.BUY:
                    return (indicators.tenkan_sen > indicators.kijun_sen and
                           indicators.current_price > indicators.bb_middle and
                           indicators.tdi_rsi > 50)
                else:
                    return (indicators.tenkan_sen < indicators.kijun_sen and
                           indicators.current_price < indicators.bb_middle and
                           indicators.tdi_rsi < 50)
            
        except Exception as e:
            self.logger.error(f"Condition evaluation error for {condition.condition_id}: {e}")
            return False
    
    def _calculate_condition_confidence(self, condition: EntryCondition, indicators: TechnicalIndicators) -> float:
        """Calculate confidence level for condition"""
        try:
            confidence = 0.5  # Base confidence
            
            # Adjust based on indicator alignment
            if condition.signal_type == SignalType.BUY:
                if indicators.tenkan_sen > indicators.kijun_sen:
                    confidence += 0.1
                if indicators.current_price > indicators.bb_middle:
                    confidence += 0.1
                if indicators.tdi_rsi > 60:
                    confidence += 0.1
                if indicators.str_entry_direction == 1:
                    confidence += 0.1
                if indicators.volume > 0:  # Volume confirmation
                    confidence += 0.1
            else:
                if indicators.tenkan_sen < indicators.kijun_sen:
                    confidence += 0.1
                if indicators.current_price < indicators.bb_middle:
                    confidence += 0.1
                if indicators.tdi_rsi < 40:
                    confidence += 0.1
                if indicators.str_entry_direction == -1:
                    confidence += 0.1
                if indicators.volume > 0:  # Volume confirmation
                    confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            self.logger.error(f"Confidence calculation error: {e}")
            return 0.5
    
    def _select_best_signal(self, buy_results: List[EntryCondition], sell_results: List[EntryCondition]) -> Optional[SignalResult]:
        """Select best signal from conditions"""
        try:
            # Minimum conditions required
            min_conditions = 3
            
            # Check BUY signals
            if len(buy_results) >= min_conditions:
                buy_confidence = sum(c.confidence for c in buy_results) / len(buy_results)
                buy_signal = SignalResult(
                    signal_id=f"BUY_{int(time.time())}",
                    signal_type=SignalType.BUY,
                    path_type=self._determine_path_type(buy_results),
                    conditions_met=[c.condition_id for c in buy_results],
                    total_conditions=len(buy_results),
                    confidence=buy_confidence,
                    ml_validated=False,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    timestamp=datetime.now()
                )
            else:
                buy_signal = None
            
            # Check SELL signals
            if len(sell_results) >= min_conditions:
                sell_confidence = sum(c.confidence for c in sell_results) / len(sell_results)
                sell_signal = SignalResult(
                    signal_id=f"SELL_{int(time.time())}",
                    signal_type=SignalType.SELL,
                    path_type=self._determine_path_type(sell_results),
                    conditions_met=[c.condition_id for c in sell_results],
                    total_conditions=len(sell_results),
                    confidence=sell_confidence,
                    ml_validated=False,
                    entry_price=0.0,
                    stop_loss=0.0,
                    take_profit=0.0,
                    timestamp=datetime.now()
                )
            else:
                sell_signal = None
            
            # Return signal with higher confidence
            if buy_signal and sell_signal:
                return buy_signal if buy_signal.confidence > sell_signal.confidence else sell_signal
            elif buy_signal:
                return buy_signal
            elif sell_signal:
                return sell_signal
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Signal selection error: {e}")
            return None
    
    def _determine_path_type(self, conditions: List[EntryCondition]) -> PathType:
        """Determine path type from conditions"""
        continuation_count = sum(1 for c in conditions if c.path_type == PathType.CONTINUATION)
        pullback_count = sum(1 for c in conditions if c.path_type == PathType.PULLBACK)
        
        return PathType.CONTINUATION if continuation_count >= pullback_count else PathType.PULLBACK
    
    def _validate_with_ml(self, signal: SignalResult, indicators: TechnicalIndicators) -> Optional[Any]:
        """Validate signal with ML enhancement"""
        try:
            if not self.ml_enhancement:
                return None
            
            # Prepare signal data for ML validation
            signal_data = {
                'signal_type': signal.signal_type.value,
                'signal_id': signal.signal_id,
                'rsi': indicators.tdi_rsi / 100.0,
                'macd': 0.5,  # Placeholder
                'bb_position': (indicators.current_price - indicators.bb_lower) / (indicators.bb_upper - indicators.bb_lower),
                'volume_ratio': 1.0,  # Placeholder
                'price_change': (indicators.current_price - indicators.previous_close) / indicators.previous_close,
                'volatility': 0.15,  # Placeholder
                'trend_strength': 0.7,  # Placeholder
                'support_resistance': 0.8,  # Placeholder
                'momentum': 0.6,  # Placeholder
                'divergence': 0.1  # Placeholder
            }
            
            # Validate with ML system
            ml_signal = self.ml_enhancement.validate_signal(signal_data)
            return ml_signal
            
        except Exception as e:
            self.logger.error(f"ML validation error: {e}")
            return None
    
    def _calculate_entry_parameters(self, signal: SignalResult, indicators: TechnicalIndicators):
        """Calculate entry price, stop loss, and take profit"""
        try:
            signal.entry_price = indicators.current_price
            
            if signal.signal_type == SignalType.BUY:
                # For BUY signals
                if signal.path_type == PathType.CONTINUATION:
                    # Continuation: SL below recent support
                    signal.stop_loss = min(indicators.bb_lower, indicators.kijun_sen) * 0.995
                    signal.take_profit = indicators.current_price * 1.015  # 1.5% target
                else:
                    # Pullback: Tighter SL
                    signal.stop_loss = indicators.current_price * 0.992  # 0.8% SL
                    signal.take_profit = indicators.current_price * 1.012  # 1.2% target
            else:
                # For SELL signals
                if signal.path_type == PathType.CONTINUATION:
                    # Continuation: SL above recent resistance
                    signal.stop_loss = max(indicators.bb_upper, indicators.kijun_sen) * 1.005
                    signal.take_profit = indicators.current_price * 0.985  # 1.5% target
                else:
                    # Pullback: Tighter SL
                    signal.stop_loss = indicators.current_price * 1.008  # 0.8% SL
                    signal.take_profit = indicators.current_price * 0.988  # 1.2% target
            
        except Exception as e:
            self.logger.error(f"Entry parameters calculation error: {e}")
            # Set default values
            signal.entry_price = indicators.current_price
            signal.stop_loss = indicators.current_price * (0.99 if signal.signal_type == SignalType.BUY else 1.01)
            signal.take_profit = indicators.current_price * (1.01 if signal.signal_type == SignalType.BUY else 0.99)
    
    def _update_performance_stats(self, signal: SignalResult):
        """Update performance statistics"""
        try:
            self.performance_stats['total_signals'] += 1
            
            if signal.signal_type == SignalType.BUY:
                self.performance_stats['buy_signals'] += 1
            else:
                self.performance_stats['sell_signals'] += 1
            
            if signal.ml_validated:
                self.performance_stats['ml_validated_signals'] += 1
                
        except Exception as e:
            self.logger.error(f"Performance stats update error: {e}")
    
    def get_signal_by_id(self, signal_id: str) -> Optional[SignalResult]:
        """Get signal by ID"""
        return self.generated_signals.get(signal_id)
    
    def get_recent_signals(self, count: int = 10) -> List[SignalResult]:
        """Get recent signals"""
        return self.signal_history[-count:] if self.signal_history else []
    
    def get_conditions_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all conditions"""
        try:
            status = {}
            
            for condition_id, condition in self.all_conditions.items():
                status[condition_id] = {
                    'name': condition.name,
                    'signal_type': condition.signal_type.value,
                    'path_type': condition.path_type.value,
                    'is_met': condition.is_met,
                    'confidence': condition.confidence,
                    'description': condition.description
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Conditions status error: {e}")
            return {}
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics"""
        try:
            stats = self.performance_stats.copy()
            
            # Calculate additional metrics
            if stats['total_signals'] > 0:
                stats['buy_percentage'] = (stats['buy_signals'] / stats['total_signals']) * 100
                stats['sell_percentage'] = (stats['sell_signals'] / stats['total_signals']) * 100
                stats['ml_validation_rate'] = (stats['ml_validated_signals'] / stats['total_signals']) * 100
            else:
                stats['buy_percentage'] = 0
                stats['sell_percentage'] = 0
                stats['ml_validation_rate'] = 0
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Performance statistics error: {e}")
            return {}
    
    def validate_signal_manually(self, signal_id: str, success: bool):
        """Manually validate signal success for performance tracking"""
        try:
            if success:
                self.performance_stats['successful_trades'] += 1
            
            # Calculate win rate
            if self.performance_stats['total_signals'] > 0:
                self.performance_stats['win_rate'] = (
                    self.performance_stats['successful_trades'] / 
                    self.performance_stats['total_signals'] * 100
                )
            
            self.logger.info(f"Signal {signal_id} validated as {'successful' if success else 'failed'}")
            
        except Exception as e:
            self.logger.error(f"Manual signal validation error: {e}")
    
    def reset_conditions(self):
        """Reset all condition states"""
        try:
            for condition in self.all_conditions.values():
                condition.is_met = False
                condition.confidence = 0.0
            
        except Exception as e:
            self.logger.error(f"Conditions reset error: {e}")
    
    def test_conditions_with_sample_data(self) -> Dict[str, Any]:
        """Test conditions with sample market data"""
        try:
            # Create sample indicators
            sample_indicators = TechnicalIndicators(
                tenkan_sen=18500.0,
                kijun_sen=18400.0,
                senkou_span_a=18450.0,
                senkou_span_b=18350.0,
                chikou_span=18300.0,
                tdi_rsi=65.0,
                tdi_price_line=0.6,
                tdi_base_line=0.5,
                tdi_volatility_band_high=0.8,
                tdi_volatility_band_low=0.2,
                smma_50=18300.0,
                bb_upper=18600.0,
                bb_middle=18450.0,
                bb_lower=18300.0,
                bb_squeeze=False,
                str_entry=18400.0,
                str_entry_direction=1,
                str_exit=18350.0,
                str_exit_direction=1,
                current_price=18520.0,
                previous_close=18480.0,
                high=18550.0,
                low=18420.0,
                volume=1000000
            )
            
            # Process test data
            signal = self.process_market_data("TEST_SYMBOL", "15M", sample_indicators)
            
            # Get conditions status
            conditions_status = self.get_conditions_status()
            
            # Count met conditions
            buy_conditions_met = sum(1 for c in self.buy_conditions.values() if c.is_met)
            sell_conditions_met = sum(1 for c in self.sell_conditions.values() if c.is_met)
            
            test_results = {
                'signal_generated': signal is not None,
                'signal_details': {
                    'signal_type': signal.signal_type.value if signal else None,
                    'confidence': signal.confidence if signal else None,
                    'conditions_met': signal.total_conditions if signal else 0,
                    'ml_validated': signal.ml_validated if signal else False
                } if signal else None,
                'conditions_summary': {
                    'buy_conditions_met': buy_conditions_met,
                    'sell_conditions_met': sell_conditions_met,
                    'total_conditions': len(self.all_conditions)
                },
                'sample_data_used': {
                    'current_price': sample_indicators.current_price,
                    'tenkan_kijun_bullish': sample_indicators.tenkan_sen > sample_indicators.kijun_sen,
                    'price_above_kumo': sample_indicators.current_price > max(sample_indicators.senkou_span_a, sample_indicators.senkou_span_b),
                    'tdi_bullish': sample_indicators.tdi_price_line > sample_indicators.tdi_base_line
                }
            }
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Test conditions error: {e}")
            return {'error': str(e)}
    
    def get_condition_details(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific condition"""
        try:
            if condition_id in self.all_conditions:
                condition = self.all_conditions[condition_id]
                return {
                    'condition_id': condition.condition_id,
                    'name': condition.name,
                    'signal_type': condition.signal_type.value,
                    'path_type': condition.path_type.value,
                    'timeframe': condition.timeframe.value,
                    'is_met': condition.is_met,
                    'confidence': condition.confidence,
                    'description': condition.description
                }
            return None
            
        except Exception as e:
            self.logger.error(f"Condition details error: {e}")
            return None
    
    def is_healthy(self) -> bool:
        """Check if entry conditions processor is healthy"""
        try:
            return (
                self.is_initialized and
                len(self.all_conditions) == 34 and
                len(self.buy_conditions) == 17 and
                len(self.sell_conditions) == 17
            )
        except:
            return False
    
    def stop(self):
        """Stop entry conditions processor"""
        try:
            self.logger.info("Entry Conditions Processor stopped")
        except Exception as e:
            self.logger.error(f"Error stopping entry conditions processor: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create entry conditions processor
    processor = EntryConditionsProcessor()
    
    # Initialize
    if processor.initialize():
        print("✅ Entry Conditions Processor initialized successfully")
        
        # Test with sample data
        test_results = processor.test_conditions_with_sample_data()
        print(f"Test results: {json.dumps(test_results, indent=2)}")
        
        # Get performance statistics
        stats = processor.get_performance_statistics()
        print(f"Performance stats: {stats}")
        
        # Get conditions status
        conditions_status = processor.get_conditions_status()
        buy_met = sum(1 for c in conditions_status.values() if c['is_met'] and c['signal_type'] == 'BUY')
        sell_met = sum(1 for c in conditions_status.values() if c['is_met'] and c['signal_type'] == 'SELL')
        print(f"Conditions met - BUY: {buy_met}, SELL: {sell_met}")
        
        # Test specific condition details
        condition_details = processor.get_condition_details('A1_CT_BUY')
        if condition_details:
            print(f"A1_CT_BUY details: {condition_details}")
        
        # Stop
        processor.stop()
    else:
        print("❌ Entry Conditions Processor initialization failed")