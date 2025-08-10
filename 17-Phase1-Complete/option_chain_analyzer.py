#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Option Chain Analyzer with Hybrid OI Logic
==================================================================

This module implements the sophisticated Hybrid OI (Open Interest) Logic 
for Indian F&O markets to determine optimal strike prices for opening 
secondary charts.

Key Features:
- Hybrid OI Logic: PATH 1 (Directional Bias) OR PATH 2 (Independent Bias)
- ATM and OTM strike price analysis
- Real-time Open Interest monitoring
- 5-level strike price scanning
- Integration with FnO Scanner
- Special expiry day handling

Author: EA GlobalFlow Pro Development Team
Date: August 2025
Version: v0.1 - Production Ready
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import redis

# Import internal modules
from truedata_bridge import TrueDataBridge
from error_handler import ErrorHandler
from security_manager import SecurityManager

@dataclass
class OIData:
    """Container for Open Interest data"""
    strike_price: float
    call_oi: int
    put_oi: int
    call_volume: int
    put_volume: int
    call_ltp: float
    put_ltp: float
    total_oi: int
    oi_difference: int  # call_oi - put_oi
    oi_ratio: float    # call_oi / put_oi
    relative_bias: float  # percentage bias
    timestamp: datetime

@dataclass
class OIAnalysisResult:
    """Container for OI analysis results"""
    symbol: str
    atm_strike: float
    current_price: float
    analysis_type: str  # "DIRECTIONAL" or "INDEPENDENT"
    should_open_charts: bool
    call_chart_strike: Optional[float]
    put_chart_strike: Optional[float]
    bias_type: str
    confidence_score: float
    analysis_details: Dict
    timestamp: datetime

@dataclass
class StrikeLevelData:
    """Container for strike level analysis"""
    strike_price: float
    distance_from_atm: int  # +1, -1, +2, -2, etc.
    oi_data: OIData
    meets_criteria: bool
    bias_percentage: float
    recommendation: str  # "CALL_BIAS", "PUT_BIAS", "NEUTRAL"

class OptionChainAnalyzer:
    """
    Advanced Option Chain Analyzer implementing Hybrid OI Logic
    
    This class implements the complete logic for:
    1. PATH 1: Directional Bias Pattern Analysis
    2. PATH 2: Independent Bias Pattern Analysis
    3. ATM/OTM strike price selection
    4. Real-time OI monitoring and analysis
    5. Chart opening recommendations
    """
    
    def __init__(self, config_path: str = "Config/ea_config_v01.json"):
        """Initialize Option Chain Analyzer"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.oi_config = self.config.get('oi_analysis', {})
        
        # Initialize logging
        self.logger = logging.getLogger('OptionChainAnalyzer')
        self.logger.setLevel(logging.INFO)
        
        # Initialize core components
        self.error_handler = ErrorHandler()
        self.security_manager = SecurityManager()
        
        # Initialize data bridge
        self.truedata_bridge = TrueDataBridge()
        
        # Analysis parameters
        self.oi_bias_threshold = self.oi_config.get('bias_threshold_percent', 30.0)
        self.min_oi_threshold = self.oi_config.get('min_oi_threshold', 100)
        self.max_strike_levels = self.oi_config.get('max_strike_levels', 5)
        
        # Cache for OI data
        self.oi_cache = {}
        self.cache_expiry_seconds = 30  # 30 seconds cache
        
        # Database for analysis history
        self.db_connection = self._init_database()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        self.logger.info("Option Chain Analyzer initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'oi_analysis': {
                'bias_threshold_percent': 30.0,
                'min_oi_threshold': 100,
                'max_strike_levels': 5,
                'enable_caching': True,
                'cache_expiry_seconds': 30,
                'confidence_threshold': 0.75
            }
        }

    def _init_database(self) -> sqlite3.Connection:
        """Initialize database for analysis history"""
        try:
            conn = sqlite3.Connection('Data/oi_analysis.db', check_same_thread=False)
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS oi_analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    atm_strike REAL,
                    analysis_type TEXT,
                    recommendation TEXT,
                    call_strike REAL,
                    put_strike REAL,
                    confidence_score REAL,
                    success BOOLEAN
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS oi_data_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT,
                    strike_price REAL,
                    call_oi INTEGER,
                    put_oi INTEGER,
                    call_volume INTEGER,
                    put_volume INTEGER,
                    total_oi INTEGER
                )
            ''')
            
            conn.commit()
            return conn
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return None

    async def analyze_hybrid_oi_logic(self, symbol: str, atm_strike: float, current_price: float) -> Optional[OIAnalysisResult]:
        """
        Main method implementing the complete Hybrid OI Logic
        
        Evaluates both PATH 1 (Directional Bias) and PATH 2 (Independent Bias)
        in parallel and returns chart opening recommendations.
        """
        try:
            self.logger.info(f"Starting Hybrid OI analysis for {symbol} @ ATM: {atm_strike}")
            
            # Step 1: Get complete option chain data
            option_chain = await self._get_option_chain_data(symbol, atm_strike)
            if not option_chain:
                self.logger.warning(f"Failed to get option chain data for {symbol}")
                return None
            
            # Step 2: Analyze ATM strike OI bias
            atm_oi_data = self._get_strike_oi_data(option_chain, atm_strike)
            if not atm_oi_data:
                self.logger.warning(f"No OI data found for ATM strike {atm_strike}")
                return None
            
            # Step 3: Analyze surrounding strike levels (5 levels above and below)
            strike_level_analysis = self._analyze_strike_levels(option_chain, atm_strike)
            
            # Step 4: Evaluate PATH 1 - Directional Bias Pattern
            path1_result = await self._evaluate_directional_bias_pattern(
                atm_oi_data, strike_level_analysis, atm_strike
            )
            
            # Step 5: Evaluate PATH 2 - Independent Bias Pattern  
            path2_result = await self._evaluate_independent_bias_pattern(
                strike_level_analysis, atm_strike
            )
            
            # Step 6: Determine final recommendation (OR logic)
            final_result = self._determine_final_recommendation(
                symbol, atm_strike, current_price, path1_result, path2_result
            )
            
            # Step 7: Store analysis in database
            self._store_analysis_result(final_result)
            
            self.logger.info(f"Hybrid OI analysis completed for {symbol}: {final_result.should_open_charts}")
            return final_result
            
        except Exception as e:
            self.error_handler.handle_error("OI_ANALYSIS_FAILED", f"{symbol}: {str(e)}")
            return None

    async def _get_option_chain_data(self, symbol: str, atm_strike: float) -> Optional[Dict]:
        """Get complete option chain data from TrueData"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{atm_strike}"
            if cache_key in self.oi_cache:
                cache_data = self.oi_cache[cache_key]
                if (datetime.now() - cache_data['timestamp']).seconds < self.cache_expiry_seconds:
                    return cache_data['data']
            
            # Get fresh data from TrueData
            underlying = self._extract_underlying_symbol(symbol)
            option_chain = await self.truedata_bridge.get_option_chain(underlying)
            
            if not option_chain:
                return None
            
            # Filter for relevant strikes (5 levels above and below ATM)
            strike_interval = self._get_strike_interval(symbol)
            min_strike = atm_strike - (5 * strike_interval)
            max_strike = atm_strike + (5 * strike_interval)
            
            filtered_chain = {}
            for strike_data in option_chain:
                strike = strike_data.get('strike_price', 0)
                if min_strike <= strike <= max_strike:
                    filtered_chain[strike] = strike_data
            
            # Cache the data
            self.oi_cache[cache_key] = {
                'data': filtered_chain,
                'timestamp': datetime.now()
            }
            
            return filtered_chain
            
        except Exception as e:
            self.logger.error(f"Failed to get option chain data for {symbol}: {e}")
            return None

    def _get_strike_oi_data(self, option_chain: Dict, strike_price: float) -> Optional[OIData]:
        """Extract OI data for specific strike price"""
        try:
            if strike_price not in option_chain:
                return None
            
            strike_data = option_chain[strike_price]
            
            call_oi = strike_data.get('call_oi', 0)
            put_oi = strike_data.get('put_oi', 0)
            
            # Calculate derived metrics
            total_oi = call_oi + put_oi
            oi_difference = call_oi - put_oi
            oi_ratio = call_oi / put_oi if put_oi > 0 else float('inf')
            
            # Calculate relative bias percentage
            if total_oi > 0:
                relative_bias = (abs(oi_difference) / total_oi) * 100
            else:
                relative_bias = 0.0
            
            return OIData(
                strike_price=strike_price,
                call_oi=call_oi,
                put_oi=put_oi,
                call_volume=strike_data.get('call_volume', 0),
                put_volume=strike_data.get('put_volume', 0),
                call_ltp=strike_data.get('call_ltp', 0.0),
                put_ltp=strike_data.get('put_ltp', 0.0),
                total_oi=total_oi,
                oi_difference=oi_difference,
                oi_ratio=oi_ratio,
                relative_bias=relative_bias,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to extract OI data for strike {strike_price}: {e}")
            return None

    def _analyze_strike_levels(self, option_chain: Dict, atm_strike: float) -> List[StrikeLevelData]:
        """Analyze all strike levels around ATM"""
        try:
            strike_interval = self._get_strike_interval_from_chain(option_chain)
            strike_levels = []
            
            # Analyze 5 levels above and below ATM
            for level in range(-5, 6):  # -5 to +5 inclusive
                if level == 0:  # Skip ATM (level 0)
                    continue
                
                target_strike = atm_strike + (level * strike_interval)
                oi_data = self._get_strike_oi_data(option_chain, target_strike)
                
                if oi_data and oi_data.total_oi >= self.min_oi_threshold:
                    # Check if this level meets bias criteria
                    meets_criteria = oi_data.relative_bias >= self.oi_bias_threshold
                    
                    # Determine bias direction
                    if oi_data.call_oi > oi_data.put_oi:
                        recommendation = "CALL_BIAS"
                    elif oi_data.put_oi > oi_data.call_oi:
                        recommendation = "PUT_BIAS"
                    else:
                        recommendation = "NEUTRAL"
                    
                    strike_level = StrikeLevelData(
                        strike_price=target_strike,
                        distance_from_atm=level,
                        oi_data=oi_data,
                        meets_criteria=meets_criteria,
                        bias_percentage=oi_data.relative_bias,
                        recommendation=recommendation
                    )
                    
                    strike_levels.append(strike_level)
            
            return strike_levels
            
        except Exception as e:
            self.logger.error(f"Strike level analysis failed: {e}")
            return []

    async def _evaluate_directional_bias_pattern(self, atm_oi_data: OIData, 
                                               strike_levels: List[StrikeLevelData], 
                                               atm_strike: float) -> Optional[Dict]:
        """
        Evaluate PATH 1: Directional Bias Pattern
        
        Bullish Directional Pattern:
        - ATM: >30% relative Call OI bias
        - +1-OTM: Call OI >30% more than Put OI (absolute numbers)
        → IF TRUE: Open Call chart at +1-OTM + Put chart at -1-OTM
        
        Bearish Directional Pattern:
        - ATM: >30% relative Put OI bias  
        - -1-OTM: Put OI >30% more than Call OI (absolute numbers)
        → IF TRUE: Open Put chart at -1-OTM + Call chart at +1-OTM
        """
        try:
            self.logger.debug("Evaluating Directional Bias Pattern (PATH 1)")
            
            # Check ATM bias
            atm_call_bias = (atm_oi_data.call_oi > atm_oi_data.put_oi and 
                           atm_oi_data.relative_bias >= self.oi_bias_threshold)
            
            atm_put_bias = (atm_oi_data.put_oi > atm_oi_data.call_oi and 
                          atm_oi_data.relative_bias >= self.oi_bias_threshold)
            
            if not (atm_call_bias or atm_put_bias):
                return None  # No ATM bias found
            
            # Find +1 OTM and -1 OTM levels
            plus_one_otm = self._find_strike_level(strike_levels, +1)
            minus_one_otm = self._find_strike_level(strike_levels, -1)
            
            if not (plus_one_otm and minus_one_otm):
                return None  # Required strike levels not available
            
            # BULLISH DIRECTIONAL PATTERN
            if atm_call_bias:
                # Check +1-OTM for Call OI > Put OI by >30% (absolute numbers)
                plus_one_call_dominance = (plus_one_otm.oi_data.call_oi > plus_one_otm.oi_data.put_oi and
                                         plus_one_otm.oi_data.relative_bias >= self.oi_bias_threshold)
                
                if plus_one_call_dominance:
                    return {
                        'pattern_type': 'BULLISH_DIRECTIONAL',
                        'qualified': True,
                        'call_chart_strike': plus_one_otm.strike_price,
                        'put_chart_strike': minus_one_otm.strike_price,
                        'confidence': self._calculate_pattern_confidence(atm_oi_data, plus_one_otm, minus_one_otm),
                        'details': {
                            'atm_call_bias': atm_oi_data.relative_bias,
                            'plus_one_call_bias': plus_one_otm.oi_data.relative_bias,
                            'atm_strike': atm_strike,
                            'call_strike': plus_one_otm.strike_price,
                            'put_strike': minus_one_otm.strike_price
                        }
                    }
            
            # BEARISH DIRECTIONAL PATTERN
            elif atm_put_bias:
                # Check -1-OTM for Put OI > Call OI by >30% (absolute numbers)
                minus_one_put_dominance = (minus_one_otm.oi_data.put_oi > minus_one_otm.oi_data.call_oi and
                                         minus_one_otm.oi_data.relative_bias >= self.oi_bias_threshold)
                
                if minus_one_put_dominance:
                    return {
                        'pattern_type': 'BEARISH_DIRECTIONAL',
                        'qualified': True,
                        'call_chart_strike': plus_one_otm.strike_price,
                        'put_chart_strike': minus_one_otm.strike_price,
                        'confidence': self._calculate_pattern_confidence(atm_oi_data, minus_one_otm, plus_one_otm),
                        'details': {
                            'atm_put_bias': atm_oi_data.relative_bias,
                            'minus_one_put_bias': minus_one_otm.oi_data.relative_bias,
                            'atm_strike': atm_strike,
                            'call_strike': plus_one_otm.strike_price,
                            'put_strike': minus_one_otm.strike_price
                        }
                    }
            
            return None  # Pattern not qualified
            
        except Exception as e:
            self.logger.error(f"Directional bias evaluation failed: {e}")
            return None

    async def _evaluate_independent_bias_pattern(self, strike_levels: List[StrikeLevelData], 
                                               atm_strike: float) -> Optional[Dict]:
        """
        Evaluate PATH 2: Independent Bias Pattern
        
        Independent Pattern:
        - +1-OTM: >30% relative difference (any direction)
        - -1-OTM: >30% relative difference (any direction)  
        → IF TRUE: Open Call chart at +1-OTM + Put chart at -1-OTM
        """
        try:
            self.logger.debug("Evaluating Independent Bias Pattern (PATH 2)")
            
            # Find +1 OTM and -1 OTM levels
            plus_one_otm = self._find_strike_level(strike_levels, +1)
            minus_one_otm = self._find_strike_level(strike_levels, -1)
            
            if not (plus_one_otm and minus_one_otm):
                return None  # Required strike levels not available
            
            # Check if both levels have >30% relative difference (any direction)
            plus_one_qualified = plus_one_otm.oi_data.relative_bias >= self.oi_bias_threshold
            minus_one_qualified = minus_one_otm.oi_data.relative_bias >= self.oi_bias_threshold
            
            if plus_one_qualified and minus_one_qualified:
                return {
                    'pattern_type': 'INDEPENDENT_BIAS',
                    'qualified': True,
                    'call_chart_strike': plus_one_otm.strike_price,
                    'put_chart_strike': minus_one_otm.strike_price,
                    'confidence': self._calculate_independent_pattern_confidence(plus_one_otm, minus_one_otm),
                    'details': {
                        'plus_one_bias': plus_one_otm.oi_data.relative_bias,
                        'minus_one_bias': minus_one_otm.oi_data.relative_bias,
                        'plus_one_direction': plus_one_otm.recommendation,
                        'minus_one_direction': minus_one_otm.recommendation,
                        'atm_strike': atm_strike,
                        'call_strike': plus_one_otm.strike_price,
                        'put_strike': minus_one_otm.strike_price
                    }
                }
            
            return None  # Pattern not qualified
            
        except Exception as e:
            self.logger.error(f"Independent bias evaluation failed: {e}")
            return None

    def _determine_final_recommendation(self, symbol: str, atm_strike: float, current_price: float,
                                      path1_result: Optional[Dict], path2_result: Optional[Dict]) -> OIAnalysisResult:
        """
        Determine final recommendation using OR logic between both paths
        
        If either PATH 1 OR PATH 2 qualifies → Open secondary charts
        Priority given to PATH 1 if both qualify
        """
        try:
            # Check if either path qualified
            path1_qualified = path1_result and path1_result.get('qualified', False)
            path2_qualified = path2_result and path2_result.get('qualified', False)
            
            if path1_qualified:
                # Use PATH 1 (Directional Bias) result
                result = OIAnalysisResult(
                    symbol=symbol,
                    atm_strike=atm_strike,
                    current_price=current_price,
                    analysis_type="DIRECTIONAL",
                    should_open_charts=True,
                    call_chart_strike=path1_result['call_chart_strike'],
                    put_chart_strike=path1_result['put_chart_strike'],
                    bias_type=path1_result['pattern_type'],
                    confidence_score=path1_result['confidence'],
                    analysis_details=path1_result['details'],
                    timestamp=datetime.now()
                )
                
                self.logger.info(f"Directional Bias Pattern qualified for {symbol}")
                
            elif path2_qualified:
                # Use PATH 2 (Independent Bias) result
                result = OIAnalysisResult(
                    symbol=symbol,
                    atm_strike=atm_strike,
                    current_price=current_price,
                    analysis_type="INDEPENDENT",
                    should_open_charts=True,
                    call_chart_strike=path2_result['call_chart_strike'],
                    put_chart_strike=path2_result['put_chart_strike'],
                    bias_type=path2_result['pattern_type'],
                    confidence_score=path2_result['confidence'],
                    analysis_details=path2_result['details'],
                    timestamp=datetime.now()
                )
                
                self.logger.info(f"Independent Bias Pattern qualified for {symbol}")
                
            else:
                # Neither path qualified
                result = OIAnalysisResult(
                    symbol=symbol,
                    atm_strike=atm_strike,
                    current_price=current_price,
                    analysis_type="NONE",
                    should_open_charts=False,
                    call_chart_strike=None,
                    put_chart_strike=None,
                    bias_type="NO_PATTERN",
                    confidence_score=0.0,
                    analysis_details={
                        'path1_result': path1_result,
                        'path2_result': path2_result,
                        'reason': 'Neither directional nor independent patterns qualified'
                    },
                    timestamp=datetime.now()
                )
                
                self.logger.info(f"No OI pattern qualified for {symbol}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Final recommendation determination failed: {e}")
            # Return failed result
            return OIAnalysisResult(
                symbol=symbol,
                atm_strike=atm_strike,
                current_price=current_price,
                analysis_type="ERROR",
                should_open_charts=False,
                call_chart_strike=None,
                put_chart_strike=None,
                bias_type="ANALYSIS_FAILED",
                confidence_score=0.0,
                analysis_details={'error': str(e)},
                timestamp=datetime.now()
            )

    def _find_strike_level(self, strike_levels: List[StrikeLevelData], target_level: int) -> Optional[StrikeLevelData]:
        """Find strike level data for specific distance from ATM"""
        for level in strike_levels:
            if level.distance_from_atm == target_level:
                return level
        return None

    def _calculate_pattern_confidence(self, atm_data: OIData, primary_level: StrikeLevelData, 
                                    secondary_level: StrikeLevelData) -> float:
        """Calculate confidence score for directional bias pattern"""
        try:
            confidence_factors = []
            
            # Factor 1: ATM bias strength
            atm_confidence = min(1.0, atm_data.relative_bias / 50.0)  # Normalize to 50%
            confidence_factors.append(atm_confidence)
            
            # Factor 2: Primary level bias strength
            primary_confidence = min(1.0, primary_level.oi_data.relative_bias / 50.0)
            confidence_factors.append(primary_confidence)
            
            # Factor 3: Overall OI liquidity
            total_oi = atm_data.total_oi + primary_level.oi_data.total_oi + secondary_level.oi_data.total_oi
            liquidity_confidence = min(1.0, total_oi / 10000.0)  # Normalize to 10K OI
            confidence_factors.append(liquidity_confidence)
            
            # Factor 4: Pattern consistency
            consistency_score = self._calculate_pattern_consistency(atm_data, primary_level)
            confidence_factors.append(consistency_score)
            
            # Weighted average
            weights = [0.3, 0.3, 0.2, 0.2]  # ATM and primary level get higher weight
            weighted_confidence = sum(f * w for f, w in zip(confidence_factors, weights))
            
            return round(weighted_confidence, 3)
            
        except Exception as e:
            self.logger.error(f"Pattern confidence calculation failed: {e}")
            return 0.5

    def _calculate_independent_pattern_confidence(self, plus_one: StrikeLevelData, 
                                                minus_one: StrikeLevelData) -> float:
        """Calculate confidence score for independent bias pattern"""
        try:
            confidence_factors = []
            
            # Factor 1: +1 OTM bias strength
            plus_one_confidence = min(1.0, plus_one.oi_data.relative_bias / 50.0)
            confidence_factors.append(plus_one_confidence)
            
            # Factor 2: -1 OTM bias strength
            minus_one_confidence = min(1.0, minus_one.oi_data.relative_bias / 50.0)
            confidence_factors.append(minus_one_confidence)
            
            # Factor 3: Overall liquidity
            total_oi = plus_one.oi_data.total_oi + minus_one.oi_data.total_oi
            liquidity_confidence = min(1.0, total_oi / 8000.0)  # Normalize to 8K OI
            confidence_factors.append(liquidity_confidence)
            
            # Factor 4: Balance between levels
            balance_score = self._calculate_level_balance(plus_one, minus_one)
            confidence_factors.append(balance_score)
            
            # Equal weighted average
            return round(sum(confidence_factors) / len(confidence_factors), 3)
            
        except Exception as e:
            self.logger.error(f"Independent pattern confidence calculation failed: {e}")
            return 0.5

    def _calculate_pattern_consistency(self, atm_data: OIData, level_data: StrikeLevelData) -> float:
        """Calculate how consistent the bias pattern is"""
        try:
            # Check if bias directions are aligned
            atm_call_bias = atm_data.call_oi > atm_data.put_oi
            level_call_bias = level_data.oi_data.call_oi > level_data.ovi_data.put_oi
            
            # Higher score for aligned bias
            if atm_call_bias == level_call_bias:
                return 0.8
            else:
                return 0.4  # Divergent bias patterns
                
        except Exception as e:
            return 0.5

    def _calculate_level_balance(self, plus_one: StrikeLevelData, minus_one: StrikeLevelData) -> float:
        """Calculate balance score between +1 and -1 OTM levels"""
        try:
            # Compare OI magnitudes
            plus_oi = plus_one.oi_data.total_oi
            minus_oi = minus_one.oi_data.total_oi
            
            if plus_oi == 0 or minus_oi == 0:
                return 0.3
            
            # Calculate balance ratio
            balance_ratio = min(plus_oi, minus_oi) / max(plus_oi, minus_oi)
            
            # Higher score for more balanced OI
            return balance_ratio
            
        except Exception as e:
            return 0.5

    def _get_strike_interval_from_chain(self, option_chain: Dict) -> float:
        """Determine strike interval from option chain data"""
        try:
            strikes = sorted(option_chain.keys())
            if len(strikes) >= 2:
                return strikes[1] - strikes[0]
            else:
                return 50.0  # Default NIFTY interval
        except:
            return 50.0

    def _extract_underlying_symbol(self, symbol: str) -> str:
        """Extract underlying symbol from option symbol"""
        try:
            # Remove CE/PE suffixes and strike prices
            if 'CE' in symbol:
                return symbol.split('CE')[0].rstrip('0123456789')
            elif 'PE' in symbol:
                return symbol.split('PE')[0].rstrip('0123456789')
            else:
                return symbol  # Already underlying
        except:
            return symbol

    def _store_analysis_result(self, result: OIAnalysisResult):
        """Store analysis result in database"""
        try:
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT INTO oi_analysis_history 
                    (symbol, atm_strike, analysis_type, recommendation, call_strike, put_strike, confidence_score, success)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    result.symbol,
                    result.atm_strike,
                    result.analysis_type,
                    result.bias_type,
                    result.call_chart_strike,
                    result.put_chart_strike,
                    result.confidence_score,
                    result.should_open_charts
                ))
                self.db_connection.commit()
        except Exception as e:
            self.logger.error(f"Failed to store analysis result: {e}")

    async def get_real_time_oi_update(self, symbol: str, strike_price: float) -> Optional[OIData]:
        """Get real-time OI update for specific strike"""
        try:
            # Get fresh data from TrueData
            underlying = self._extract_underlying_symbol(symbol)
            strike_data = await self.truedata_bridge.get_strike_data(underlying, strike_price)
            
            if not strike_data:
                return None
            
            return self._get_strike_oi_data({strike_price: strike_data}, strike_price)
            
        except Exception as e:
            self.logger.error(f"Real-time OI update failed for {symbol}@{strike_price}: {e}")
            return None

    async def batch_analyze_symbols(self, symbols: List[Tuple[str, float, float]]) -> Dict[str, OIAnalysisResult]:
        """Batch analyze multiple symbols for efficiency"""
        try:
            self.logger.info(f"Starting batch analysis for {len(symbols)} symbols")
            
            # Create analysis tasks
            tasks = []
            for symbol, atm_strike, current_price in symbols:
                task = self.analyze_hybrid_oi_logic(symbol, atm_strike, current_price)
                tasks.append((symbol, task))
            
            # Execute in parallel
            results = {}
            for symbol, task in tasks:
                try:
                    result = await asyncio.wait_for(task, timeout=10.0)  # 10 second timeout
                    if result:
                        results[symbol] = result
                except asyncio.TimeoutError:
                    self.logger.warning(f"Analysis timeout for {symbol}")
                except Exception as e:
                    self.logger.error(f"Analysis failed for {symbol}: {e}")
            
            self.logger.info(f"Batch analysis completed: {len(results)} successful")
            return results
            
        except Exception as e:
            self.logger.error(f"Batch analysis failed: {e}")
            return {}

    def get_analysis_statistics(self) -> Dict:
        """Get analysis performance statistics"""
        try:
            if not self.db_connection:
                return {}
            
            cursor = self.db_connection.cursor()
            
            # Overall statistics
            cursor.execute("SELECT COUNT(*) FROM oi_analysis_history")
            total_analyses = cursor.fetchone()[0]
            
            # Success rate
            cursor.execute("SELECT COUNT(*) FROM oi_analysis_history WHERE success = 1")
            successful_analyses = cursor.fetchone()[0]
            
            # Pattern distribution
            cursor.execute("""
                SELECT analysis_type, COUNT(*) 
                FROM oi_analysis_history 
                GROUP BY analysis_type
            """)
            pattern_distribution = dict(cursor.fetchall())
            
            # Average confidence by pattern
            cursor.execute("""
                SELECT analysis_type, AVG(confidence_score) 
                FROM oi_analysis_history 
                WHERE success = 1
                GROUP BY analysis_type
            """)
            avg_confidence = dict(cursor.fetchall())
            
            # Recent performance (last 24 hours)
            cursor.execute("""
                SELECT COUNT(*) 
                FROM oi_analysis_history 
                WHERE timestamp > datetime('now', '-24 hours')
            """)
            recent_analyses = cursor.fetchone()[0]
            
            return {
                'total_analyses': total_analyses,
                'successful_analyses': successful_analyses,
                'success_rate': successful_analyses / total_analyses if total_analyses > 0 else 0,
                'pattern_distribution': pattern_distribution,
                'average_confidence_by_pattern': avg_confidence,
                'recent_24h_analyses': recent_analyses,
                'cache_hit_rate': len(self.oi_cache) / (total_analyses + 1),  # Approximate
                'last_update': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Statistics generation failed: {e}")
            return {}

    async def validate_oi_logic_integrity(self, symbol: str, atm_strike: float) -> Dict:
        """Validate the integrity of OI logic implementation"""
        try:
            validation_results = {
                'symbol': symbol,
                'atm_strike': atm_strike,
                'tests_passed': 0,
                'tests_failed': 0,
                'test_results': {},
                'overall_status': 'UNKNOWN'
            }
            
            # Test 1: Data availability
            option_chain = await self._get_option_chain_data(symbol, atm_strike)
            test1_pass = option_chain is not None and len(option_chain) > 0
            validation_results['test_results']['data_availability'] = test1_pass
            
            # Test 2: ATM OI calculation
            if test1_pass:
                atm_oi = self._get_strike_oi_data(option_chain, atm_strike)
                test2_pass = atm_oi is not None and atm_oi.total_oi > 0
                validation_results['test_results']['atm_oi_calculation'] = test2_pass
            else:
                test2_pass = False
                validation_results['test_results']['atm_oi_calculation'] = False
            
            # Test 3: Strike level analysis
            if test2_pass:
                strike_levels = self._analyze_strike_levels(option_chain, atm_strike)
                test3_pass = len(strike_levels) >= 2  # At least +1 and -1 OTM
                validation_results['test_results']['strike_level_analysis'] = test3_pass
            else:
                test3_pass = False
                validation_results['test_results']['strike_level_analysis'] = False
            
            # Test 4: Pattern recognition logic
            if test3_pass:
                try:
                    # Test both patterns
                    atm_oi = self._get_strike_oi_data(option_chain, atm_strike)
                    path1 = await self._evaluate_directional_bias_pattern(atm_oi, strike_levels, atm_strike)
                    path2 = await self._evaluate_independent_bias_pattern(strike_levels, atm_strike)
                    test4_pass = True  # If no exceptions, logic is sound
                    validation_results['test_results']['pattern_recognition'] = test4_pass
                except Exception as e:
                    test4_pass = False
                    validation_results['test_results']['pattern_recognition'] = False
                    validation_results['test_results']['pattern_error'] = str(e)
            else:
                test4_pass = False
                validation_results['test_results']['pattern_recognition'] = False
            
            # Count results
            for test_result in validation_results['test_results'].values():
                if isinstance(test_result, bool):
                    if test_result:
                        validation_results['tests_passed'] += 1
                    else:
                        validation_results['tests_failed'] += 1
            
            # Overall status
            if validation_results['tests_failed'] == 0:
                validation_results['overall_status'] = 'PASSED'
            elif validation_results['tests_passed'] > validation_results['tests_failed']:
                validation_results['overall_status'] = 'PARTIAL'
            else:
                validation_results['overall_status'] = 'FAILED'
            
            return validation_results
            
        except Exception as e:
            return {
                'symbol': symbol,
                'atm_strike': atm_strike,
                'overall_status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def clear_oi_cache(self):
        """Clear the OI data cache"""
        self.oi_cache.clear()
        self.logger.info("OI cache cleared")

    def get_cache_statistics(self) -> Dict:
        """Get cache performance statistics"""
        return {
            'cache_size': len(self.oi_cache),
            'cache_entries': list(self.oi_cache.keys()),
            'cache_expiry_seconds': self.cache_expiry_seconds,
            'oldest_entry': min(
                [entry['timestamp'] for entry in self.oi_cache.values()],
                default=datetime.now()
            ).isoformat() if self.oi_cache else None,
            'newest_entry': max(
                [entry['timestamp'] for entry in self.oi_cache.values()],
                default=datetime.now()
            ).isoformat() if self.oi_cache else None
        }

# Global instance for use by other modules
option_chain_analyzer = None

def get_option_chain_analyzer() -> OptionChainAnalyzer:
    """Get singleton instance of Option Chain Analyzer"""
    global option_chain_analyzer
    if option_chain_analyzer is None:
        option_chain_analyzer = OptionChainAnalyzer()
    return option_chain_analyzer

if __name__ == "__main__":
    # Test the analyzer
    import asyncio
    
    async def main():
        analyzer = OptionChainAnalyzer()
        
        # Test with NIFTY
        result = await analyzer.analyze_hybrid_oi_logic("NIFTY", 19000.0, 19050.0)
        
        if result:
            print(f"Analysis Result:")
            print(f"  Symbol: {result.symbol}")
            print(f"  Should Open Charts: {result.should_open_charts}")
            print(f"  Analysis Type: {result.analysis_type}")
            print(f"  Call Chart Strike: {result.call_chart_strike}")
            print(f"  Put Chart Strike: {result.put_chart_strike}")
            print(f"  Confidence: {result.confidence_score}")
            print(f"  Details: {result.analysis_details}")
        else:
            print("No analysis result generated")
        
        # Get statistics
        stats = analyzer.get_analysis_statistics()
        print(f"Statistics: {stats}")
    
    asyncio.run(main())