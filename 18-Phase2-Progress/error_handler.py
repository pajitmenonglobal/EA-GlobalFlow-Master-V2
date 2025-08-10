#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Centralized Error Handling System
=========================================================

Comprehensive error handling solution providing:
- Centralized error management and logging
- MT5 error code integration and handling
- Error categorization and severity assessment
- Recovery strategies and fallback mechanisms
- Error tracking and reporting
- Integration with alerting system

Author: EA GlobalFlow Pro Development Team
Date: August 2025
Version: v0.1 - Production Ready
"""

import logging
import json
import sqlite3
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import sys
import os

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    FATAL = "FATAL"

class ErrorCategory(Enum):
    """Error category classification"""
    SYSTEM = "SYSTEM"
    NETWORK = "NETWORK"
    API = "API"
    DATABASE = "DATABASE"
    TRADING = "TRADING"
    ML = "ML"
    DATA = "DATA"
    CONFIGURATION = "CONFIGURATION"
    SECURITY = "SECURITY"
    USER = "USER"
    UNKNOWN = "UNKNOWN"

class RecoveryStrategy(Enum):
    """Recovery strategy types"""
    RETRY = "RETRY"
    FALLBACK = "FALLBACK"
    RESTART = "RESTART"
    ESCALATE = "ESCALATE"
    IGNORE = "IGNORE"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"

@dataclass
class ErrorContext:
    """Container for error context information"""
    component: str
    function: str
    line_number: int
    file_name: str
    timestamp: datetime
    user_data: Dict[str, Any] = field(default_factory=dict)
    system_state: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ErrorRecord:
    """Container for complete error information"""
    error_id: str
    error_code: str
    error_type: str
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    description: str
    context: ErrorContext
    stack_trace: str
    recovery_strategy: RecoveryStrategy
    recovery_attempted: bool = False
    recovery_successful: bool = False
    resolution_time: Optional[datetime] = None
    occurrence_count: int = 1
    first_occurrence: datetime = field(default_factory=datetime.now)
    last_occurrence: datetime = field(default_factory=datetime.now)
    related_errors: List[str] = field(default_factory=list)

class ErrorHandler:
    """
    Advanced Centralized Error Handling System
    
    Provides comprehensive error management including:
    - Error detection, classification, and logging
    - MT5 error code mapping and handling
    - Recovery strategy implementation
    - Error tracking and analytics
    - Integration with monitoring and alerting
    - Performance impact assessment
    """
    
    def __init__(self, config_path: str = "Config/ea_config_v01.json"):
        """Initialize Error Handler"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.error_config = self.config.get('error_handler', {})
        
        # Initialize logging
        self.logger = logging.getLogger('ErrorHandler')
        self.logger.setLevel(logging.INFO)
        
        # Error tracking
        self.error_records: Dict[str, ErrorRecord] = {}
        self.error_patterns: Dict[str, int] = {}
        
        # MT5 error codes mapping
        self.mt5_error_codes = self._initialize_mt5_error_codes()
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, Callable] = {}
        self.recovery_history: List[Dict] = []
        
        # Configuration parameters
        self.max_error_records = self.error_config.get('max_error_records', 10000)
        self.error_retention_days = self.error_config.get('error_retention_days', 30)
        self.enable_auto_recovery = self.error_config.get('enable_auto_recovery', True)
        self.max_recovery_attempts = self.error_config.get('max_recovery_attempts', 3)
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.error_lock = threading.RLock()
        
        # Database for persistence
        self.db_connection = self._init_database()
        
        # Error thresholds for alerting
        self.error_thresholds = self.error_config.get('thresholds', {
            'critical_errors_per_hour': 5,
            'total_errors_per_hour': 50,
            'error_rate_threshold': 0.1
        })
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
        
        self.logger.info("Error Handler initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load error handler configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load error handler config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default configuration if file loading fails"""
        return {
            'error_handler': {
                'max_error_records': 10000,
                'error_retention_days': 30,
                'enable_auto_recovery': True,
                'max_recovery_attempts': 3,
                'enable_email_alerts': True,
                'enable_phone_alerts': True,
                'critical_alert_email': "pajitmenonai@gmail.com",
                'critical_alert_phone': "+971507423656",
                'thresholds': {
                    'critical_errors_per_hour': 5,
                    'total_errors_per_hour': 50,
                    'error_rate_threshold': 0.1
                }
            }
        }

    def _init_database(self) -> sqlite3.Connection:
        """Initialize database for error tracking"""
        try:
            db_path = Path("Data/error_handler.db")
            db_path.parent.mkdir(exist_ok=True)
            
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS error_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    error_id TEXT UNIQUE,
                    error_code TEXT,
                    error_type TEXT,
                    severity TEXT,
                    category TEXT,
                    message TEXT,
                    description TEXT,
                    component TEXT,
                    function_name TEXT,
                    stack_trace TEXT,
                    recovery_strategy TEXT,
                    recovery_attempted BOOLEAN,
                    recovery_successful BOOLEAN,
                    occurrence_count INTEGER,
                    first_occurrence DATETIME,
                    last_occurrence DATETIME,
                    resolution_time DATETIME,
                    context_data TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS recovery_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    error_id TEXT,
                    recovery_strategy TEXT,
                    recovery_attempt INTEGER,
                    success BOOLEAN,
                    duration_seconds REAL,
                    details TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS error_patterns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    pattern_signature TEXT,
                    occurrence_count INTEGER,
                    severity TEXT,
                    category TEXT,
                    components_affected TEXT,
                    impact_assessment TEXT
                )
            ''')
            
            conn.commit()
            return conn
            
        except Exception as e:
            print(f"Error handler database initialization failed: {e}")
            return None

    def _initialize_mt5_error_codes(self) -> Dict[int, Dict[str, Any]]:
        """Initialize MT5 error codes with detailed information"""
        return {
            # Trade server errors
            0: {"name": "ERR_SUCCESS", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.LOW, "recovery": RecoveryStrategy.IGNORE},
            1: {"name": "ERR_NO_ERROR", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.LOW, "recovery": RecoveryStrategy.IGNORE},
            2: {"name": "ERR_COMMON_ERROR", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            3: {"name": "ERR_INVALID_TRADE_PARAMETERS", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.FALLBACK},
            4: {"name": "ERR_SERVER_BUSY", "category": ErrorCategory.NETWORK, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            5: {"name": "ERR_OLD_VERSION", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.MANUAL_INTERVENTION},
            6: {"name": "ERR_NO_CONNECTION", "category": ErrorCategory.NETWORK, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.RETRY},
            7: {"name": "ERR_NOT_ENOUGH_RIGHTS", "category": ErrorCategory.SECURITY, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.MANUAL_INTERVENTION},
            8: {"name": "ERR_TOO_FREQUENT_REQUESTS", "category": ErrorCategory.API, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            9: {"name": "ERR_MALFUNCTIONAL_TRADE", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.FALLBACK},
            
            # Account errors
            64: {"name": "ERR_ACCOUNT_DISABLED", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.CRITICAL, "recovery": RecoveryStrategy.MANUAL_INTERVENTION},
            65: {"name": "ERR_INVALID_ACCOUNT", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.CRITICAL, "recovery": RecoveryStrategy.MANUAL_INTERVENTION},
            
            # Trade errors
            128: {"name": "ERR_TRADE_TIMEOUT", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            129: {"name": "ERR_INVALID_PRICE", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            130: {"name": "ERR_INVALID_STOPS", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            131: {"name": "ERR_INVALID_TRADE_VOLUME", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            132: {"name": "ERR_MARKET_CLOSED", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            133: {"name": "ERR_TRADE_DISABLED", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.MANUAL_INTERVENTION},
            134: {"name": "ERR_NOT_ENOUGH_MONEY", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.FALLBACK},
            135: {"name": "ERR_PRICE_CHANGED", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.LOW, "recovery": RecoveryStrategy.RETRY},
            136: {"name": "ERR_OFF_QUOTES", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            137: {"name": "ERR_BROKER_BUSY", "category": ErrorCategory.NETWORK, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            138: {"name": "ERR_REQUOTE", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.LOW, "recovery": RecoveryStrategy.RETRY},
            139: {"name": "ERR_ORDER_LOCKED", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            140: {"name": "ERR_LONG_POSITIONS_ONLY_ALLOWED", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            141: {"name": "ERR_TOO_MANY_REQUESTS", "category": ErrorCategory.API, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            
            # OrderSend errors
            4000: {"name": "ERR_NO_MQLERROR", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.LOW, "recovery": RecoveryStrategy.IGNORE},
            4001: {"name": "ERR_WRONG_FUNCTION_POINTER", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.RESTART},
            4002: {"name": "ERR_ARRAY_INDEX_OUT_OF_RANGE", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.FALLBACK},
            4003: {"name": "ERR_NO_MEMORY_FOR_FUNCTION_CALL_STACK", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.CRITICAL, "recovery": RecoveryStrategy.RESTART},
            4004: {"name": "ERR_RECURSIVE_STACK_OVERFLOW", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.CRITICAL, "recovery": RecoveryStrategy.RESTART},
            4005: {"name": "ERR_NOT_ENOUGH_STACK_FOR_PARAMETER", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.RESTART},
            4006: {"name": "ERR_NO_MEMORY_FOR_PARAMETER_STRING", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.RESTART},
            4007: {"name": "ERR_NO_MEMORY_FOR_TEMP_STRING", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.RESTART},
            4008: {"name": "ERR_NOT_INITIALIZED_STRING", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4009: {"name": "ERR_NOT_INITIALIZED_ARRAYSTRING", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4010: {"name": "ERR_NO_MEMORY_FOR_ARRAYSTRING", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.RESTART},
            4011: {"name": "ERR_TOO_LONG_STRING", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4012: {"name": "ERR_REMAINDER_FROM_ZERO_DIVIDE", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.FALLBACK},
            4013: {"name": "ERR_ZERO_DIVIDE", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.FALLBACK},
            4014: {"name": "ERR_UNKNOWN_COMMAND", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.FALLBACK},
            4015: {"name": "ERR_WRONG_JUMP", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.RESTART},
            
            # File errors
            4050: {"name": "ERR_CANNOT_OPEN_FILE", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.HIGH, "recovery": RecoveryStrategy.FALLBACK},
            4051: {"name": "ERR_INCOMPATIBLE_FILEACCESS", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4052: {"name": "ERR_NO_ORDER_SELECTED", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4053: {"name": "ERR_UNKNOWN_SYMBOL", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4054: {"name": "ERR_INVALID_PRICE_PARAM", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4055: {"name": "ERR_INVALID_TICKET", "category": ErrorCategory.TRADING, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            
            # Object errors
            4200: {"name": "ERR_OBJECT_DOES_NOT_EXIST", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4201: {"name": "ERR_UNKNOWN_OBJECT_PROPERTY", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4202: {"name": "ERR_CANNOT_READ_OBJECT_PROPERTY", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4203: {"name": "ERR_CANNOT_CHANGE_OBJECT_PROPERTY", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            
            # MarketInfo errors
            4106: {"name": "ERR_NO_HISTORY_DATA", "category": ErrorCategory.DATA, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.RETRY},
            4107: {"name": "ERR_HISTORY_WILL_UPDATED", "category": ErrorCategory.DATA, "severity": ErrorSeverity.LOW, "recovery": RecoveryStrategy.RETRY},
            
            # GlobalVariable errors
            4501: {"name": "ERR_GLOBAL_VARIABLE_NOT_FOUND", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK},
            4502: {"name": "ERR_FUNC_NOT_ALLOWED_IN_TESTING", "category": ErrorCategory.SYSTEM, "severity": ErrorSeverity.LOW, "recovery": RecoveryStrategy.FALLBACK},
            
            # Custom EA errors
            65536: {"name": "ERR_USER_ERROR_FIRST", "category": ErrorCategory.USER, "severity": ErrorSeverity.MEDIUM, "recovery": RecoveryStrategy.FALLBACK}
        }

    def _initialize_recovery_strategies(self):
        """Initialize recovery strategy implementations"""
        self.recovery_strategies = {
            RecoveryStrategy.RETRY.value: self._recovery_retry,
            RecoveryStrategy.FALLBACK.value: self._recovery_fallback,
            RecoveryStrategy.RESTART.value: self._recovery_restart,
            RecoveryStrategy.ESCALATE.value: self._recovery_escalate,
            RecoveryStrategy.IGNORE.value: self._recovery_ignore,
            RecoveryStrategy.MANUAL_INTERVENTION.value: self._recovery_manual_intervention
        }

    def handle_error(self, error_code: Union[str, int], message: str, 
                    component: str = "UNKNOWN", function: str = "UNKNOWN",
                    user_data: Optional[Dict] = None, severity: Optional[ErrorSeverity] = None) -> ErrorRecord:
        """
        Main error handling entry point
        
        Handles errors with comprehensive analysis, classification, and recovery
        """
        try:
            # Create error context
            context = self._create_error_context(component, function, user_data)
            
            # Generate unique error ID
            error_id = self._generate_error_id(error_code, component, function)
            
            # Check if this is a known error pattern
            existing_error = self._check_existing_error(error_id)
            
            if existing_error:
                # Update existing error
                existing_error.occurrence_count += 1
                existing_error.last_occurrence = datetime.now()
                error_record = existing_error
            else:
                # Create new error record
                error_record = self._create_error_record(
                    error_id, error_code, message, context, severity
                )
                
                # Store in memory
                with self.error_lock:
                    self.error_records[error_id] = error_record
            
            # Log the error
            self._log_error(error_record)
            
            # Store in database
            self._store_error_record(error_record)
            
            # Attempt recovery if enabled
            if self.enable_auto_recovery and not error_record.recovery_attempted:
                self._attempt_recovery(error_record)
            
            # Check for error patterns and thresholds
            self._analyze_error_patterns(error_record)
            
            # Trigger alerts if necessary
            self._check_alert_thresholds(error_record)
            
            return error_record
            
        except Exception as e:
            # Fallback error handling to prevent infinite loops
            self.logger.critical(f"Error handler failure: {str(e)}")
            return self._create_fallback_error_record(error_code, message)

    def _create_error_context(self, component: str, function: str, user_data: Optional[Dict]) -> ErrorContext:
        """Create error context with system information"""
        try:
            # Get current frame information
            frame = sys._getframe(2)  # Go back 2 frames to get caller info
            
            return ErrorContext(
                component=component,
                function=function,
                line_number=frame.f_lineno,
                file_name=os.path.basename(frame.f_code.co_filename),
                timestamp=datetime.now(),
                user_data=user_data or {},
                system_state=self._capture_system_state()
            )
            
        except Exception as e:
            return ErrorContext(
                component=component,
                function=function,
                line_number=0,
                file_name="unknown",
                timestamp=datetime.now(),
                user_data=user_data or {},
                system_state={}
            )

    def _capture_system_state(self) -> Dict[str, Any]:
        """Capture current system state for error context"""
        try:
            import psutil
            return {
                'cpu_usage': psutil.cpu_percent(),
                'memory_usage': psutil.virtual_memory().percent,
                'thread_count': threading.active_count(),
                'timestamp': datetime.now().isoformat()
            }
        except:
            return {'timestamp': datetime.now().isoformat()}

    def _generate_error_id(self, error_code: Union[str, int], component: str, function: str) -> str:
        """Generate unique error ID"""
        import hashlib
        
        # Create signature from error components
        signature = f"{error_code}_{component}_{function}"
        
        # Generate hash
        error_hash = hashlib.md5(signature.encode()).hexdigest()[:8]
        
        return f"ERR_{error_hash}"

    def _check_existing_error(self, error_id: str) -> Optional[ErrorRecord]:
        """Check if error already exists"""
        with self.error_lock:
            return self.error_records.get(error_id)

    def _create_error_record(self, error_id: str, error_code: Union[str, int], 
                           message: str, context: ErrorContext, 
                           severity: Optional[ErrorSeverity]) -> ErrorRecord:
        """Create comprehensive error record"""
        try:
            # Determine error properties
            if isinstance(error_code, int) and error_code in self.mt5_error_codes:
                mt5_info = self.mt5_error_codes[error_code]
                error_type = mt5_info["name"]
                category = mt5_info["category"]
                auto_severity = mt5_info["severity"]
                recovery_strategy = mt5_info["recovery"]
            else:
                error_type = str(error_code)
                category = self._classify_error_category(message, context)
                auto_severity = self._assess_error_severity(message, context)
                recovery_strategy = self._determine_recovery_strategy(category, auto_severity)
            
            # Use provided severity or auto-determined
            final_severity = severity or auto_severity
            
            # Get stack trace
            stack_trace = traceback.format_exc()
            
            return ErrorRecord(
                error_id=error_id,
                error_code=str(error_code),
                error_type=error_type,
                severity=final_severity,
                category=category,
                message=message,
                description=self._generate_error_description(error_code, message, context),
                context=context,
                stack_trace=stack_trace,
                recovery_strategy=recovery_strategy,
                first_occurrence=datetime.now(),
                last_occurrence=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error record creation failed: {e}")
            return self._create_fallback_error_record(error_code, message)

    def _classify_error_category(self, message: str, context: ErrorContext) -> ErrorCategory:
        """Classify error into appropriate category"""
        message_lower = message.lower()
        component_lower = context.component.lower()
        
        # Network related
        if any(keyword in message_lower for keyword in ['connection', 'network', 'timeout', 'socket']):
            return ErrorCategory.NETWORK
        
        # API related
        if any(keyword in message_lower for keyword in ['api', 'request', 'response', 'authentication']):
            return ErrorCategory.API
        
        # Database related
        if any(keyword in message_lower for keyword in ['database', 'sql', 'sqlite', 'db']):
            return ErrorCategory.DATABASE
        
        # Trading related
        if any(keyword in message_lower for keyword in ['trade', 'order', 'position', 'symbol', 'market']):
            return ErrorCategory.TRADING
        
        # ML related
        if any(keyword in message_lower for keyword in ['ml', 'model', 'prediction', 'training']):
            return ErrorCategory.ML
        
        # Data related
        if any(keyword in message_lower for keyword in ['data', 'parse', 'format', 'csv', 'json']):
            return ErrorCategory.DATA
        
        # Security related
        if any(keyword in message_lower for keyword in ['security', 'auth', 'permission', 'access']):
            return ErrorCategory.SECURITY
        
        # Component-based classification
        if 'api' in component_lower:
            return ErrorCategory.API
        elif 'ml' in component_lower:
            return ErrorCategory.ML
        elif 'trade' in component_lower:
            return ErrorCategory.TRADING
        elif 'data' in component_lower:
            return ErrorCategory.DATA
        
        return ErrorCategory.SYSTEM

    def _assess_error_severity(self, message: str, context: ErrorContext) -> ErrorSeverity:
        """Assess error severity based on content and context"""
        message_lower = message.lower()
        
        # Critical keywords
        if any(keyword in message_lower for keyword in ['critical', 'fatal', 'crash', 'corruption']):
            return ErrorSeverity.CRITICAL
        
        # High severity keywords
        if any(keyword in message_lower for keyword in ['failed', 'cannot', 'unable', 'error', 'exception']):
            return ErrorSeverity.HIGH
        
        # Medium severity keywords
        if any(keyword in message_lower for keyword in ['warning', 'retry', 'timeout', 'invalid']):
            return ErrorSeverity.MEDIUM
        
        # Default to medium for unknown errors
        return ErrorSeverity.MEDIUM

    def _determine_recovery_strategy(self, category: ErrorCategory, severity: ErrorSeverity) -> RecoveryStrategy:
        """Determine appropriate recovery strategy"""
        # Critical errors require manual intervention
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.MANUAL_INTERVENTION
        
        # Category-based strategies
        if category == ErrorCategory.NETWORK:
            return RecoveryStrategy.RETRY
        elif category == ErrorCategory.API:
            return RecoveryStrategy.RETRY
        elif category == ErrorCategory.DATABASE:
            return RecoveryStrategy.FALLBACK
        elif category == ErrorCategory.TRADING:
            return RecoveryStrategy.FALLBACK
        elif category == ErrorCategory.SYSTEM:
            return RecoveryStrategy.RESTART if severity == ErrorSeverity.HIGH else RecoveryStrategy.FALLBACK
        
        return RecoveryStrategy.RETRY

    def _generate_error_description(self, error_code: Union[str, int], message: str, context: ErrorContext) -> str:
        """Generate detailed error description"""
        try:
            description_parts = [f"Error occurred in {context.component}.{context.function}"]
            
            if isinstance(error_code, int) and error_code in self.mt5_error_codes:
                mt5_info = self.mt5_error_codes[error_code]
                description_parts.append(f"MT5 Error: {mt5_info['name']}")
            
            description_parts.append(f"Message: {message}")
            description_parts.append(f"Location: {context.file_name}:{context.line_number}")
            
            if context.user_data:
                description_parts.append(f"Context: {context.user_data}")
            
            return " | ".join(description_parts)
            
        except Exception as e:
            return f"Error description generation failed: {str(e)}"

    def _create_fallback_error_record(self, error_code: Union[str, int], message: str) -> ErrorRecord:
        """Create minimal error record as fallback"""
        return ErrorRecord(
            error_id=f"FALLBACK_{int(time.time())}",
            error_code=str(error_code),
            error_type="FALLBACK_ERROR",
            severity=ErrorSeverity.HIGH,
            category=ErrorCategory.SYSTEM,
            message=message,
            description=f"Fallback error record: {message}",
            context=ErrorContext(
                component="UNKNOWN",
                function="UNKNOWN",
                line_number=0,
                file_name="unknown",
                timestamp=datetime.now()
            ),
            stack_trace="Fallback - no stack trace available",
            recovery_strategy=RecoveryStrategy.MANUAL_INTERVENTION
        )

    def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        log_message = f"[{error_record.error_id}] {error_record.category.value}: {error_record.message}"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        else:
            self.logger.info(log_message)

    def _store_error_record(self, error_record: ErrorRecord):
        """Store error record in database"""
        try:
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT OR REPLACE INTO error_records 
                    (error_id, error_code, error_type, severity, category, message, description,
                     component, function_name, stack_trace, recovery_strategy, recovery_attempted,
                     recovery_successful, occurrence_count, first_occurrence, last_occurrence,
                     resolution_time, context_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    error_record.error_id,
                    error_record.error_code,
                    error_record.error_type,
                    error_record.severity.value,
                    error_record.category.value,
                    error_record.message,
                    error_record.description,
                    error_record.context.component,
                    error_record.context.function,
                    error_record.stack_trace,
                    error_record.recovery_strategy.value,
                    error_record.recovery_attempted,
                    error_record.recovery_successful,
                    error_record.occurrence_count,
                    error_record.first_occurrence,
                    error_record.last_occurrence,
                    error_record.resolution_time,
                    json.dumps(error_record.context.user_data)
                ))
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store error record: {e}")

    def _attempt_recovery(self, error_record: ErrorRecord):
        """Attempt error recovery using appropriate strategy"""
        try:
            if error_record.recovery_attempted:
                return
            
            error_record.recovery_attempted = True
            recovery_strategy = error_record.recovery_strategy.value
            
            if recovery_strategy in self.recovery_strategies:
                self.logger.info(f"Attempting recovery for {error_record.error_id} using {recovery_strategy}")
                
                recovery_start = time.time()
                recovery_func = self.recovery_strategies[recovery_strategy]
                success = recovery_func(error_record)
                recovery_duration = time.time() - recovery_start
                
                error_record.recovery_successful = success
                if success:
                    error_record.resolution_time = datetime.now()
                
                # Log recovery attempt
                recovery_log = {
                    'timestamp': datetime.now(),
                    'error_id': error_record.error_id,
                    'strategy': recovery_strategy,
                    'success': success,
                    'duration': recovery_duration,
                    'details': f"Recovery attempt for {error_record.error_type}"
                }
                
                self.recovery_history.append(recovery_log)
                
                # Store in database
                if self.db_connection:
                    self.db_connection.execute('''
                        INSERT INTO recovery_history 
                        (error_id, recovery_strategy, recovery_attempt, success, duration_seconds, details)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        error_record.error_id,
                        recovery_strategy,
                        1,  # First attempt
                        success,
                        recovery_duration,
                        recovery_log['details']
                    ))
                    self.db_connection.commit()
                
                self.logger.info(f"Recovery {'succeeded' if success else 'failed'} for {error_record.error_id}")
            
        except Exception as e:
            self.logger.error(f"Recovery attempt failed: {e}")
            error_record.recovery_successful = False

    # Recovery strategy implementations
    def _recovery_retry(self, error_record: ErrorRecord) -> bool:
        """Retry recovery strategy"""
        try:
            # Simple retry - log and return true to indicate retry should be attempted
            self.logger.info(f"Retry strategy applied for {error_record.error_id}")
            return True
        except Exception as e:
            self.logger.error(f"Retry recovery failed: {e}")
            return False

    def _recovery_fallback(self, error_record: ErrorRecord) -> bool:
        """Fallback recovery strategy"""
        try:
            # Fallback to safe defaults
            self.logger.info(f"Fallback strategy applied for {error_record.error_id}")
            
            # Component-specific fallback logic could be implemented here
            if error_record.category == ErrorCategory.TRADING:
                # Fallback for trading errors - stop trading temporarily
                return True
            elif error_record.category == ErrorCategory.ML:
                # Fallback for ML errors - disable ML temporarily
                return True
            
            return True
        except Exception as e:
            self.logger.error(f"Fallback recovery failed: {e}")
            return False

    def _recovery_restart(self, error_record: ErrorRecord) -> bool:
        """Restart recovery strategy"""
        try:
            # Component restart logic
            self.logger.warning(f"Restart strategy applied for {error_record.error_id}")
            
            # This would restart specific components based on the error
            # For now, just log the restart intention
            return True
        except Exception as e:
            self.logger.error(f"Restart recovery failed: {e}")
            return False

    def _recovery_escalate(self, error_record: ErrorRecord) -> bool:
        """Escalate recovery strategy"""
        try:
            # Escalate to higher authority or different recovery method
            self.logger.warning(f"Escalating error {error_record.error_id}")
            
            # Send alert to administrators
            self._send_escalation_alert(error_record)
            
            return True
        except Exception as e:
            self.logger.error(f"Escalation recovery failed: {e}")
            return False

    def _recovery_ignore(self, error_record: ErrorRecord) -> bool:
        """Ignore recovery strategy"""
        try:
            # Simply ignore the error
            self.logger.debug(f"Ignoring error {error_record.error_id}")
            return True
        except Exception as e:
            return False

    def _recovery_manual_intervention(self, error_record: ErrorRecord) -> bool:
        """Manual intervention recovery strategy"""
        try:
            # Flag for manual intervention
            self.logger.critical(f"Manual intervention required for {error_record.error_id}")
            
            # Send critical alert
            self._send_critical_alert(error_record)
            
            return False  # Cannot auto-recover
        except Exception as e:
            self.logger.error(f"Manual intervention flagging failed: {e}")
            return False

    def _analyze_error_patterns(self, error_record: ErrorRecord):
        """Analyze error patterns for trends"""
        try:
            # Create pattern signature
            pattern_signature = f"{error_record.category.value}_{error_record.error_type}"
            
            # Update pattern count
            with self.error_lock:
                if pattern_signature in self.error_patterns:
                    self.error_patterns[pattern_signature] += 1
                else:
                    self.error_patterns[pattern_signature] = 1
            
            # Check for concerning patterns
            pattern_count = self.error_patterns[pattern_signature]
            if pattern_count > 10:  # More than 10 occurrences
                self.logger.warning(f"Error pattern detected: {pattern_signature} occurred {pattern_count} times")
                
                # Store pattern analysis
                if self.db_connection:
                    self.db_connection.execute('''
                        INSERT OR REPLACE INTO error_patterns 
                        (pattern_signature, occurrence_count, severity, category, components_affected, impact_assessment)
                        VALUES (?, ?, ?, ?, ?, ?)
                    ''', (
                        pattern_signature,
                        pattern_count,
                        error_record.severity.value,
                        error_record.category.value,
                        error_record.context.component,
                        f"Pattern detected with {pattern_count} occurrences"
                    ))
                    self.db_connection.commit()
                    
        except Exception as e:
            self.logger.error(f"Error pattern analysis failed: {e}")

    def _check_alert_thresholds(self, error_record: ErrorRecord):
        """Check if error thresholds are exceeded"""
        try:
            current_time = datetime.now()
            hour_ago = current_time - timedelta(hours=1)
            
            # Count recent errors
            recent_critical = 0
            recent_total = 0
            
            with self.error_lock:
                for error in self.error_records.values():
                    if error.last_occurrence > hour_ago:
                        recent_total += error.occurrence_count
                        if error.severity == ErrorSeverity.CRITICAL:
                            recent_critical += error.occurrence_count
            
            # Check thresholds
            critical_threshold = self.error_thresholds.get('critical_errors_per_hour', 5)
            total_threshold = self.error_thresholds.get('total_errors_per_hour', 50)
            
            if recent_critical >= critical_threshold:
                self.logger.critical(f"Critical error threshold exceeded: {recent_critical} in last hour")
                self._send_threshold_alert('CRITICAL_ERRORS_THRESHOLD', recent_critical, critical_threshold)
            
            if recent_total >= total_threshold:
                self.logger.error(f"Total error threshold exceeded: {recent_total} in last hour")
                self._send_threshold_alert('TOTAL_ERRORS_THRESHOLD', recent_total, total_threshold)
                
        except Exception as e:
            self.logger.error(f"Threshold checking failed: {e}")

    def _send_escalation_alert(self, error_record: ErrorRecord):
        """Send escalation alert"""
        try:
            alert_message = f"Error escalated: {error_record.error_id} - {error_record.message}"
            self.logger.warning(f"ESCALATION ALERT: {alert_message}")
            
            # This would integrate with actual alerting system
            if self.error_config.get('enable_email_alerts', False):
                self._send_email_alert("Error Escalation", alert_message)
                
        except Exception as e:
            self.logger.error(f"Escalation alert failed: {e}")

    def _send_critical_alert(self, error_record: ErrorRecord):
        """Send critical error alert"""
        try:
            alert_message = f"CRITICAL ERROR: {error_record.error_id} - {error_record.message}"
            self.logger.critical(f"CRITICAL ALERT: {alert_message}")
            
            # This would integrate with actual alerting system
            if self.error_config.get('enable_phone_alerts', False):
                self._send_phone_alert(alert_message)
                
        except Exception as e:
            self.logger.error(f"Critical alert failed: {e}")

    def _send_threshold_alert(self, threshold_type: str, actual: int, threshold: int):
        """Send threshold exceeded alert"""
        try:
            alert_message = f"ERROR THRESHOLD EXCEEDED: {threshold_type} - {actual}/{threshold}"
            self.logger.error(f"THRESHOLD ALERT: {alert_message}")
            
            # This would integrate with actual alerting system
            
        except Exception as e:
            self.logger.error(f"Threshold alert failed: {e}")

    def _send_email_alert(self, subject: str, message: str):
        """Send email alert (placeholder)"""
        try:
            email = self.error_config.get('critical_alert_email')
            if email:
                self.logger.info(f"Email alert would be sent to {email}: {subject} - {message}")
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")

    def _send_phone_alert(self, message: str):
        """Send phone alert (placeholder)"""
        try:
            phone = self.error_config.get('critical_alert_phone')
            if phone:
                self.logger.info(f"Phone alert would be sent to {phone}: {message}")
        except Exception as e:
            self.logger.error(f"Phone alert failed: {e}")

    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.error_lock:
                # Filter errors for time period
                recent_errors = [
                    error for error in self.error_records.values()
                    if error.last_occurrence > cutoff_time
                ]
            
            if not recent_errors:
                return {'error': 'No errors in specified time period'}
            
            # Calculate statistics
            total_errors = len(recent_errors)
            total_occurrences = sum(error.occurrence_count for error in recent_errors)
            
            # Group by severity
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = len([
                    e for e in recent_errors if e.severity == severity
                ])
            
            # Group by category
            category_counts = {}
            for category in ErrorCategory:
                category_counts[category.value] = len([
                    e for e in recent_errors if e.category == category
                ])
            
            # Recovery statistics
            recovery_attempted = len([e for e in recent_errors if e.recovery_attempted])
            recovery_successful = len([e for e in recent_errors if e.recovery_successful])
            
            # Most common errors
            error_frequency = {}
            for error in recent_errors:
                error_frequency[error.error_type] = error_frequency.get(error.error_type, 0) + error.occurrence_count
            
            most_common = sorted(error_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                'time_period_hours': hours,
                'total_unique_errors': total_errors,
                'total_occurrences': total_occurrences,
                'severity_breakdown': severity_counts,
                'category_breakdown': category_counts,
                'recovery_statistics': {
                    'attempted': recovery_attempted,
                    'successful': recovery_successful,
                    'success_rate': recovery_successful / recovery_attempted if recovery_attempted > 0 else 0
                },
                'most_common_errors': most_common,
                'error_patterns_detected': len(self.error_patterns),
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error statistics generation failed: {e}")
            return {'error': str(e)}

    def resolve_error(self, error_id: str, resolution_notes: str = "") -> bool:
        """Manually resolve an error"""
        try:
            with self.error_lock:
                if error_id in self.error_records:
                    error_record = self.error_records[error_id]
                    error_record.resolution_time = datetime.now()
                    error_record.recovery_successful = True
                    
                    # Update database
                    if self.db_connection:
                        self.db_connection.execute(
                            "UPDATE error_records SET resolution_time = ?, recovery_successful = 1 WHERE error_id = ?",
                            (error_record.resolution_time, error_id)
                        )
                        self.db_connection.commit()
                    
                    self.logger.info(f"Error {error_id} manually resolved: {resolution_notes}")
                    return True
                else:
                    self.logger.warning(f"Error {error_id} not found for resolution")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error resolution failed: {e}")
            return False

    def cleanup_old_errors(self):
        """Cleanup old error records"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.error_retention_days)
            
            # Cleanup memory
            with self.error_lock:
                old_error_ids = [
                    error_id for error_id, error in self.error_records.items()
                    if error.first_occurrence < cutoff_time and error.resolution_time is not None
                ]
                
                for error_id in old_error_ids:
                    del self.error_records[error_id]
                
                # Keep only recent patterns
                self.error_patterns = {k: v for k, v in self.error_patterns.items() if v > 1}
            
            # Cleanup database
            if self.db_connection:
                self.db_connection.execute(
                    "DELETE FROM error_records WHERE first_occurrence < ? AND resolution_time IS NOT NULL",
                    (cutoff_time,)
                )
                self.db_connection.execute(
                    "DELETE FROM recovery_history WHERE timestamp < ?",
                    (cutoff_time,)
                )
                self.db_connection.commit()
            
            self.logger.info(f"Cleaned up {len(old_error_ids)} old error records")
            
        except Exception as e:
            self.logger.error(f"Error cleanup failed: {e}")

    def get_system_health_status(self) -> Dict[str, Any]:
        """Get system health status based on error patterns"""
        try:
            recent_time = datetime.now() - timedelta(hours=1)
            
            with self.error_lock:
                recent_errors = [
                    error for error in self.error_records.values()
                    if error.last_occurrence > recent_time
                ]
            
            critical_errors = len([e for e in recent_errors if e.severity == ErrorSeverity.CRITICAL])
            high_errors = len([e for e in recent_errors if e.severity == ErrorSeverity.HIGH])
            total_errors = len(recent_errors)
            
            # Determine health status
            if critical_errors > 0:
                health_status = "CRITICAL"
            elif high_errors > 5:
                health_status = "UNHEALTHY"
            elif total_errors > 20:
                health_status = "DEGRADED"
            elif total_errors > 5:
                health_status = "WARNING"
            else:
                health_status = "HEALTHY"
            
            return {
                'health_status': health_status,
                'recent_errors': {
                    'critical': critical_errors,
                    'high': high_errors,
                    'total': total_errors
                },
                'error_rate_per_hour': total_errors,
                'active_patterns': len(self.error_patterns),
                'recovery_rate': self._calculate_recovery_rate(),
                'last_critical_error': self._get_last_critical_error(),
                'assessment_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Health status assessment failed: {e}")
            return {'health_status': 'UNKNOWN', 'error': str(e)}

    def _calculate_recovery_rate(self) -> float:
        """Calculate recent recovery success rate"""
        try:
            recent_recoveries = [r for r in self.recovery_history if 
                               (datetime.now() - r['timestamp']).total_seconds() < 3600]
            
            if not recent_recoveries:
                return 0.0
            
            successful = len([r for r in recent_recoveries if r['success']])
            return successful / len(recent_recoveries)
            
        except Exception as e:
            return 0.0

    def _get_last_critical_error(self) -> Optional[str]:
        """Get timestamp of last critical error"""
        try:
            with self.error_lock:
                critical_errors = [
                    error for error in self.error_records.values()
                    if error.severity == ErrorSeverity.CRITICAL
                ]
            
            if critical_errors:
                latest = max(critical_errors, key=lambda x: x.last_occurrence)
                return latest.last_occurrence.isoformat()
            
            return None
            
        except Exception as e:
            return None

# Global instance
error_handler = None

def get_error_handler() -> ErrorHandler:
    """Get singleton instance of Error Handler"""
    global error_handler
    if error_handler is None:
        error_handler = ErrorHandler()
    return error_handler

if __name__ == "__main__":
    # Test the Error Handler
    handler = ErrorHandler()
    
    # Test various error scenarios
    print("Testing Error Handler...")
    
    # Test MT5 error
    error1 = handler.handle_error(134, "Not enough money", "TradingEngine", "place_order")
    print(f"MT5 Error: {error1.error_id} - {error1.message}")
    
    # Test custom error
    error2 = handler.handle_error("API_TIMEOUT", "API request timed out", "APIBridge", "make_request")
    print(f"Custom Error: {error2.error_id} - {error2.message}")
    
    # Test duplicate error
    error3 = handler.handle_error("API_TIMEOUT", "API request timed out", "APIBridge", "make_request")
    print(f"Duplicate Error: {error3.error_id} - Occurrences: {error3.occurrence_count}")
    
    # Get statistics
    stats = handler.get_error_statistics(1)
    print(f"Error Statistics: {json.dumps(stats, indent=2)}")
    
    # Get health status
    health = handler.get_system_health_status()
    print(f"System Health: {json.dumps(health, indent=2)}")
