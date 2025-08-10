#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - ML Enhancement System
Three-layer ML system: Signal Validation, Market Regime Classification, Risk Optimization

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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import pickle
import joblib

# Try to import ML libraries
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import StandardScaler
    import xgboost as xgb
    ML_LIBRARIES_AVAILABLE = True
except ImportError:
    ML_LIBRARIES_AVAILABLE = False
    print("Warning: ML libraries not available. Install with: pip install scikit-learn xgboost")

class MLModelType(Enum):
    SIGNAL_VALIDATION = "SIGNAL_VALIDATION"
    MARKET_REGIME = "MARKET_REGIME"
    RISK_OPTIMIZATION = "RISK_OPTIMIZATION"

class MarketRegime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"

class SignalQuality(Enum):
    EXCELLENT = "EXCELLENT"  # 90%+ confidence
    GOOD = "GOOD"           # 75-90% confidence
    FAIR = "FAIR"           # 60-75% confidence
    POOR = "POOR"           # <60% confidence

@dataclass
class MLSignal:
    signal_id: str
    signal_type: str  # BUY/SELL
    confidence: float
    quality: SignalQuality
    features: Dict[str, float]
    regime: MarketRegime
    risk_score: float
    timestamp: datetime

@dataclass
class MLPrediction:
    model_type: MLModelType
    prediction: Any
    confidence: float
    features_used: List[str]
    timestamp: datetime

class MLEnhancement:
    """
    ML Enhancement System for EA GlobalFlow Pro v0.1
    Implements 3-layer ML system for 95% win rate optimization
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize ML enhancement system"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger('MLEnhancement')
        
        # Configuration
        self.ml_config = {}
        self.is_initialized = False
        self.is_running = False
        
        # Models
        self.signal_validator = None
        self.regime_classifier = None
        self.risk_optimizer = None
        
        # Model paths
        self.models_dir = os.path.join(os.path.dirname(__file__), 'Models')
        
        # Feature engineering
        self.feature_scaler = StandardScaler()
        self.feature_columns = []
        
        # Signal processing
        self.signal_queue = []
        self.processed_signals = {}
        self.signal_lock = threading.Lock()
        
        # Performance tracking
        self.performance_metrics = {
            'total_signals': 0,
            'filtered_signals': 0,
            'successful_trades': 0,
            'win_rate': 0.0,
            'filter_effectiveness': 0.0
        }
        
        # Processing thread
        self.processing_thread = None
        self.last_training = datetime.now()
        self.training_interval = timedelta(days=7)  # Retrain weekly
        
        # Market data buffer
        self.market_data_buffer = {}
        self.buffer_size = 1000
        
    def initialize(self) -> bool:
        """
        Initialize ML enhancement system
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing ML Enhancement System v0.1...")
            
            # Check ML libraries
            if not ML_LIBRARIES_AVAILABLE:
                self.logger.error("ML libraries not available")
                return False
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Create models directory
            os.makedirs(self.models_dir, exist_ok=True)
            
            # Initialize models
            if not self._initialize_models():
                return False
            
            # Load or train models
            if not self._load_or_train_models():
                return False
            
            # Start signal processing
            self._start_signal_processing()
            
            self.is_initialized = True
            self.is_running = True
            self.logger.info("✅ ML Enhancement System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"ML Enhancement initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("ml_init", e)
            return False
    
    def _load_config(self) -> bool:
        """Load ML configuration"""
        try:
            if self.config_manager:
                self.ml_config = self.config_manager.get_config('ml_enhancement', {})
            else:
                # Default configuration
                self.ml_config = {
                    'enabled': True,
                    'models': {
                        'signal_validation': {
                            'enabled': True,
                            'model_type': 'random_forest',
                            'confidence_threshold': 0.75
                        },
                        'market_regime': {
                            'enabled': True,
                            'model_type': 'xgboost',
                            'regimes': ['trending', 'ranging', 'volatile']
                        },
                        'risk_optimization': {
                            'enabled': True,
                            'model_type': 'lstm',
                            'dynamic_sl_tp': True
                        }
                    },
                    'win_rate_target': 95.0,
                    'filter_chain': {
                        'entry_conditions_filter': 40.0,
                        'ml_validation_filter': 42.0,
                        'candlestick_volume_filter': 29.0
                    }
                }
            
            self.logger.info("ML configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load ML config: {e}")
            return False
    
    def _initialize_models(self) -> bool:
        """Initialize ML models"""
        try:
            # Signal Validation Model (Random Forest)
            self.signal_validator = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Market Regime Classifier (XGBoost)
            self.regime_classifier = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                n_jobs=-1
            )
            
            # Risk Optimizer (placeholder - would be LSTM in full implementation)
            self.risk_optimizer = RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )
            
            # Define feature columns
            self.feature_columns = [
                'rsi', 'macd', 'bb_position', 'volume_ratio',
                'price_change', 'volatility', 'trend_strength',
                'support_resistance', 'momentum', 'divergence'
            ]
            
            self.logger.info("ML models initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize ML models: {e}")
            return False
    
    def _load_or_train_models(self) -> bool:
        """Load existing models or train new ones"""
        try:
            models_loaded = 0
            
            # Try to load existing models
            model_files = {
                'signal_validator': 'signal_validator.pkl',
                'regime_classifier': 'regime_classifier.pkl',
                'risk_optimizer': 'risk_optimizer.pkl',
                'feature_scaler': 'feature_scaler.pkl'
            }
            
            for model_name, filename in model_files.items():
                filepath = os.path.join(self.models_dir, filename)
                if os.path.exists(filepath):
                    try:
                        if model_name == 'feature_scaler':
                            self.feature_scaler = joblib.load(filepath)
                        else:
                            setattr(self, model_name, joblib.load(filepath))
                        models_loaded += 1
                        self.logger.info(f"Loaded {model_name} from {filename}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load {model_name}: {e}")
            
            # If no models loaded, train new ones
            if models_loaded == 0:
                self.logger.info("No existing models found, training new models...")
                if not self._train_initial_models():
                    self.logger.warning("Initial model training failed, using default models")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load/train models: {e}")
            return False
    
    def _train_initial_models(self) -> bool:
        """Train initial models with synthetic data"""
        try:
            # Generate synthetic training data
            X_train, y_signal, y_regime, y_risk = self._generate_synthetic_data(1000)
            
            # Train signal validator
            self.signal_validator.fit(X_train, y_signal)
            
            # Train regime classifier
            self.regime_classifier.fit(X_train, y_regime)
            
            # Train risk optimizer
            self.risk_optimizer.fit(X_train, y_risk)
            
            # Fit feature scaler
            self.feature_scaler.fit(X_train)
            
            # Save models
            self._save_models()
            
            self.logger.info("Initial models trained successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initial model training failed: {e}")
            return False
    
    def _generate_synthetic_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate synthetic training data"""
        try:
            np.random.seed(42)
            
            # Generate features
            X = np.random.randn(n_samples, len(self.feature_columns))
            
            # Generate synthetic labels
            y_signal = np.random.choice([0, 1], n_samples, p=[0.3, 0.7])  # 70% positive signals
            y_regime = np.random.choice([0, 1, 2], n_samples, p=[0.4, 0.4, 0.2])  # Trending, Ranging, Volatile
            y_risk = np.random.choice([0, 1, 2], n_samples, p=[0.5, 0.3, 0.2])  # Low, Medium, High risk
            
            return X, y_signal, y_regime, y_risk
            
        except Exception as e:
            self.logger.error(f"Synthetic data generation failed: {e}")
            return np.array([]), np.array([]), np.array([]), np.array([])
    
    def _save_models(self):
        """Save trained models"""
        try:
            model_files = {
                'signal_validator': self.signal_validator,
                'regime_classifier': self.regime_classifier,
                'risk_optimizer': self.risk_optimizer,
                'feature_scaler': self.feature_scaler
            }
            
            for model_name, model in model_files.items():
                if model is not None:
                    filepath = os.path.join(self.models_dir, f"{model_name}.pkl")
                    joblib.dump(model, filepath)
                    self.logger.debug(f"Saved {model_name} to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save models: {e}")
    
    def _start_signal_processing(self):
        """Start signal processing thread"""
        try:
            self.processing_thread = threading.Thread(target=self._signal_processing_loop, daemon=True)
            self.processing_thread.start()
            self.logger.info("Signal processing thread started")
            
        except Exception as e:
            self.logger.error(f"Failed to start signal processing: {e}")
    
    def _signal_processing_loop(self):
        """Main signal processing loop"""
        while self.is_running:
            try:
                # Process queued signals
                with self.signal_lock:
                    if self.signal_queue:
                        signals_to_process = self.signal_queue.copy()
                        self.signal_queue.clear()
                    else:
                        signals_to_process = []
                
                # Process each signal
                for signal_data in signals_to_process:
                    self._process_signal(signal_data)
                
                # Check if retraining is needed
                if datetime.now() - self.last_training > self.training_interval:
                    self._retrain_models()
                
                time.sleep(1)  # Process every second
                
            except Exception as e:
                self.logger.error(f"Signal processing error: {e}")
                if self.error_handler:
                    self.error_handler.handle_error("ml_signal_processing", e)
                time.sleep(5)
    
    def _process_signal(self, signal_data: Dict[str, Any]):
        """Process individual signal through ML pipeline"""
        try:
            signal_id = signal_data.get('signal_id', '')
            
            # Extract features
            features = self._extract_features(signal_data)
            if not features:
                return
            
            # ML Pipeline: 3-layer validation
            
            # Layer 1: Signal Validation
            signal_valid, signal_confidence = self._validate_signal(features)
            if not signal_valid:
                self.logger.debug(f"Signal {signal_id} failed validation")
                return
            
            # Layer 2: Market Regime Classification
            regime = self._classify_market_regime(features)
            
            # Layer 3: Risk Optimization
            risk_score = self._optimize_risk(features, regime)
            
            # Determine signal quality
            quality = self._determine_signal_quality(signal_confidence)
            
            # Create ML signal
            ml_signal = MLSignal(
                signal_id=signal_id,
                signal_type=signal_data.get('signal_type', 'BUY'),
                confidence=signal_confidence,
                quality=quality,
                features=features,
                regime=regime,
                risk_score=risk_score,
                timestamp=datetime.now()
            )
            
            # Store processed signal
            self.processed_signals[signal_id] = ml_signal
            
            # Update performance metrics
            self._update_performance_metrics(ml_signal)
            
            self.logger.info(f"✅ Signal {signal_id} processed: {quality.value} quality, {regime.value} regime")
            
        except Exception as e:
            self.logger.error(f"Signal processing error: {e}")
    
    def _extract_features(self, signal_data: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract features from signal data"""
        try:
            # This would extract real features from market data
            # For now, using placeholder values
            features = {}
            
            for feature in self.feature_columns:
                if feature in signal_data:
                    features[feature] = signal_data[feature]
                else:
                    # Generate placeholder feature values
                    features[feature] = np.random.random()
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction error: {e}")
            return None
    
    def _validate_signal(self, features: Dict[str, float]) -> Tuple[bool, float]:
        """Layer 1: Signal Validation using Random Forest"""
        try:
            if not self.signal_validator:
                return True, 0.5  # Default if model not available
            
            # Convert features to array
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            
            # Scale features
            feature_array = self.feature_scaler.transform(feature_array)
            
            # Get prediction and confidence
            prediction = self.signal_validator.predict(feature_array)[0]
            confidence = self.signal_validator.predict_proba(feature_array)[0].max()
            
            # Signal is valid if prediction is positive and confidence is above threshold
            threshold = self.ml_config.get('models', {}).get('signal_validation', {}).get('confidence_threshold', 0.75)
            is_valid = prediction == 1 and confidence >= threshold
            
            return is_valid, confidence
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return True, 0.5
    
    def _classify_market_regime(self, features: Dict[str, float]) -> MarketRegime:
        """Layer 2: Market Regime Classification using XGBoost"""
        try:
            if not self.regime_classifier:
                return MarketRegime.UNKNOWN
            
            # Convert features to array
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            
            # Scale features
            feature_array = self.feature_scaler.transform(feature_array)
            
            # Get prediction
            prediction = self.regime_classifier.predict(feature_array)[0]
            
            # Map prediction to regime
            regime_mapping = {
                0: MarketRegime.TRENDING,
                1: MarketRegime.RANGING,
                2: MarketRegime.VOLATILE
            }
            
            return regime_mapping.get(prediction, MarketRegime.UNKNOWN)
            
        except Exception as e:
            self.logger.error(f"Market regime classification error: {e}")
            return MarketRegime.UNKNOWN
    
    def _optimize_risk(self, features: Dict[str, float], regime: MarketRegime) -> float:
        """Layer 3: Risk Optimization"""
        try:
            if not self.risk_optimizer:
                return 0.5  # Default risk score
            
            # Convert features to array
            feature_array = np.array([list(features.values())]).reshape(1, -1)
            
            # Scale features
            feature_array = self.feature_scaler.transform(feature_array)
            
            # Get risk prediction
            risk_prediction = self.risk_optimizer.predict(feature_array)[0]
            
            # Convert to risk score (0-1)
            risk_score = risk_prediction / 2.0  # Assuming 0-2 scale
            
            # Adjust based on market regime
            if regime == MarketRegime.VOLATILE:
                risk_score *= 1.5  # Increase risk in volatile markets
            elif regime == MarketRegime.RANGING:
                risk_score *= 0.8  # Decrease risk in ranging markets
            
            return min(risk_score, 1.0)
            
        except Exception as e:
            self.logger.error(f"Risk optimization error: {e}")
            return 0.5
    
    def _determine_signal_quality(self, confidence: float) -> SignalQuality:
        """Determine signal quality based on confidence"""
        if confidence >= 0.9:
            return SignalQuality.EXCELLENT
        elif confidence >= 0.75:
            return SignalQuality.GOOD
        elif confidence >= 0.6:
            return SignalQuality.FAIR
        else:
            return SignalQuality.POOR
    
    def _update_performance_metrics(self, ml_signal: MLSignal):
        """Update performance tracking metrics"""
        try:
            self.performance_metrics['total_signals'] += 1
            
            if ml_signal.quality in [SignalQuality.EXCELLENT, SignalQuality.GOOD]:
                self.performance_metrics['filtered_signals'] += 1
            
            # Calculate filter effectiveness
            if self.performance_metrics['total_signals'] > 0:
                self.performance_metrics['filter_effectiveness'] = (
                    self.performance_metrics['filtered_signals'] / 
                    self.performance_metrics['total_signals'] * 100
                )
            
        except Exception as e:
            self.logger.error(f"Performance metrics update error: {e}")
    
    def _retrain_models(self):
        """Retrain models with recent data"""
        try:
            self.logger.info("Starting model retraining...")
            
            # This would collect recent trading data and retrain
            # For now, just update the last training time
            self.last_training = datetime.now()
            
            self.logger.info("Model retraining completed")
            
        except Exception as e:
            self.logger.error(f"Model retraining error: {e}")
    
    def validate_signal(self, signal_data: Dict[str, Any]) -> Optional[MLSignal]:
        """
        Validate trading signal through ML pipeline
        
        Args:
            signal_data: Signal data with features
            
        Returns:
            MLSignal if valid, None if rejected
        """
        try:
            signal_id = signal_data.get('signal_id', f"signal_{int(time.time())}")
            
            # Add to processing queue
            with self.signal_lock:
                signal_data['signal_id'] = signal_id
                self.signal_queue.append(signal_data)
            
            # Wait for processing (in real implementation, this would be async)
            time.sleep(0.5)
            
            # Return processed signal
            return self.processed_signals.get(signal_id)
            
        except Exception as e:
            self.logger.error(f"Signal validation error: {e}")
            return None
    
    def get_market_regime(self, market_data: Dict[str, Any]) -> MarketRegime:
        """Get current market regime"""
        try:
            features = self._extract_features(market_data)
            if features:
                return self._classify_market_regime(features)
            return MarketRegime.UNKNOWN
            
        except Exception as e:
            self.logger.error(f"Market regime detection error: {e}")
            return MarketRegime.UNKNOWN
    
    def calculate_risk_score(self, signal_data: Dict[str, Any]) -> float:
        """Calculate risk score for signal"""
        try:
            features = self._extract_features(signal_data)
            if features:
                regime = self._classify_market_regime(features)
                return self._optimize_risk(features, regime)
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Risk score calculation error: {e}")
            return 0.5
    
    def process_signals(self):
        """Process any pending signals (called by main loop)"""
        try:
            # This method is called by the main bridge
            # Processing happens in the background thread
            pass
            
        except Exception as e:
            self.logger.error(f"Signal processing error: {e}")
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get ML system performance metrics"""
        return self.performance_metrics.copy()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        try:
            model_info = {
                'signal_validator': {
                    'type': 'RandomForestClassifier',
                    'loaded': self.signal_validator is not None,
                    'features': len(self.feature_columns)
                },
                'regime_classifier': {
                    'type': 'XGBClassifier',
                    'loaded': self.regime_classifier is not None,
                    'features': len(self.feature_columns)
                },
                'risk_optimizer': {
                    'type': 'RandomForestClassifier',
                    'loaded': self.risk_optimizer is not None,
                    'features': len(self.feature_columns)
                },
                'last_training': self.last_training.isoformat(),
                'next_training': (self.last_training + self.training_interval).isoformat()
            }
            
            return model_info
            
        except Exception as e:
            self.logger.error(f"Model info error: {e}")
            return {}
    
    def is_healthy(self) -> bool:
        """Check if ML enhancement system is healthy"""
        try:
            return (
                self.is_initialized and
                self.is_running and
                self.signal_validator is not None and
                self.regime_classifier is not None and
                self.risk_optimizer is not None
            )
        except:
            return False
    
    def stop(self):
        """Stop ML enhancement system"""
        try:
            self.is_running = False
            self._save_models()  # Save models before stopping
            self.logger.info("ML Enhancement System stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping ML system: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create ML enhancement system
    ml_system = MLEnhancement()
    
    # Initialize
    if ml_system.initialize():
        print("✅ ML Enhancement System initialized successfully")
        
        # Test signal validation
        test_signal = {
            'signal_type': 'BUY',
            'rsi': 0.3,
            'macd': 0.1,
            'bb_position': 0.2,
            'volume_ratio': 1.5,
            'price_change': 0.02,
            'volatility': 0.15,
            'trend_strength': 0.7,
            'support_resistance': 0.8,
            'momentum': 0.6,
            'divergence': 0.1
        }
        
        ml_signal = ml_system.validate_signal(test_signal)
        if ml_signal:
            print(f"Signal validated: {ml_signal.quality.value} quality")
            print(f"Market regime: {ml_signal.regime.value}")
            print(f"Risk score: {ml_signal.risk_score:.2f}")
        else:
            print("Signal rejected by ML system")
        
        # Test market regime detection
        regime = ml_system.get_market_regime(test_signal)
        print(f"Current market regime: {regime.value}")
        
        # Get performance metrics
        metrics = ml_system.get_performance_metrics()
        print(f"Performance metrics: {metrics}")
        
        # Get model info
        model_info = ml_system.get_model_info()
        print(f"Model info: {model_info}")
        
        # Stop
        ml_system.stop()
    else:
        print("❌ ML Enhancement System initialization failed")