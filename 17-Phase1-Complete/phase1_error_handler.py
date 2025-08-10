#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Centralized Error Handler
Comprehensive error handling, logging, and recovery system

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
import traceback
import smtplib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import pickle

class ErrorSeverity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ErrorCategory(Enum):
    SYSTEM = "SYSTEM"
    API = "API"
    TRADING = "TRADING"
    ML = "ML"
    NETWORK = "NETWORK"
    DATA = "DATA"
    SECURITY = "SECURITY"
    UNKNOWN = "UNKNOWN"

class RecoveryAction(Enum):
    NONE = "NONE"
    RETRY = "RETRY"
    RESTART_COMPONENT = "RESTART_COMPONENT"
    RESTART_SYSTEM = "RESTART_SYSTEM"
    EMERGENCY_STOP = "EMERGENCY_STOP"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"

@dataclass
class ErrorReport:
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    message: str
    exception: Optional[Exception]
    traceback_str: str
    context: Dict[str, Any]
    recovery_action: RecoveryAction
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class ErrorPattern:
    pattern_id: str
    category: ErrorCategory
    keywords: List[str]
    severity: ErrorSeverity
    recovery_action: RecoveryAction
    threshold_count: int = 5
    threshold_time_minutes: int = 10

class ErrorHandler:
    """
    Centralized Error Handler for EA GlobalFlow Pro v0.1
    Handles error logging, categorization, and automated recovery
    """
    
    def __init__(self, config_manager=None):
        """Initialize error handler"""
        self.config_manager = config_manager
        self.logger = logging.getLogger('ErrorHandler')
        
        # Configuration
        self.error_config = {}
        self.is_initialized = False
        
        # Error storage
        self.error_reports = []
        self.error_patterns = {}
        self.error_statistics = {}
        
        # Recovery actions
        self.recovery_callbacks = {}
        self.auto_recovery_enabled = True
        
        # Notification settings
        self.email_notifications = False
        self.sms_notifications = False
        self.sound_notifications = False
        
        # Rate limiting
        self.error_rate_limits = {}
        self.max_errors_per_minute = 10
        
        # Threading
        self.error_lock = threading.Lock()
        self.cleanup_thread = None
        self.is_running = False
        
        # File paths
        self.error_log_dir = os.path.join(os.path.dirname(__file__), 'Logs', 'Errors')
        self.error_reports_file = os.path.join(self.error_log_dir, 'error_reports.pkl')
        
        # Statistics
        self.stats = {
            'total_errors': 0,
            'critical_errors': 0,
            'resolved_errors': 0,
            'auto_recoveries': 0,
            'manual_interventions': 0,
            'system_restarts': 0
        }
        
    def initialize(self) -> bool:
        """
        Initialize error handler
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing Error Handler v0.1...")
            
            # Create directories
            os.makedirs(self.error_log_dir, exist_ok=True)
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Initialize error patterns
            if not self._initialize_error_patterns():
                return False
            
            # Load previous error reports
            self._load_error_reports()
            
            # Start cleanup thread
            self._start_cleanup_thread()
            
            self.is_initialized = True
            self.is_running = True
            self.logger.info("✅ Error Handler initialized successfully")
            return True
            
        except Exception as e:
            print(f"Error Handler initialization failed: {e}")
            return False
    
    def _load_config(self) -> bool:
        """Load error handler configuration"""
        try:
            if self.config_manager:
                self.error_config = self.config_manager.get_config('error_handling', {})
            else:
                # Default configuration
                self.error_config = {
                    'auto_recovery_enabled': True,
                    'max_errors_per_minute': 10,
                    'max_error_reports': 10000,
                    'cleanup_interval_hours': 24,
                    'notifications': {
                        'email_enabled': True,
                        'sms_enabled': False,
                        'sound_enabled': False
                    },
                    'email_settings': {
                        'smtp_server': 'smtp.gmail.com',
                        'smtp_port': 587,
                        'username': '',
                        'password': '',
                        'recipient': 'pajitmenonai@gmail.com'
                    }
                }
            
            # Update settings
            self.auto_recovery_enabled = self.error_config.get('auto_recovery_enabled', True)
            self.max_errors_per_minute = self.error_config.get('max_errors_per_minute', 10)
            
            notifications = self.error_config.get('notifications', {})
            self.email_notifications = notifications.get('email_enabled', True)
            self.sms_notifications = notifications.get('sms_enabled', False)
            self.sound_notifications = notifications.get('sound_enabled', False)
            
            self.logger.info("Error handler configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load error handler config: {e}")
            return False
    
    def _initialize_error_patterns(self) -> bool:
        """Initialize error pattern recognition"""
        try:
            patterns = [
                ErrorPattern(
                    pattern_id='API_CONNECTION_ERROR',
                    category=ErrorCategory.API,
                    keywords=['connection', 'timeout', 'network', 'unreachable'],
                    severity=ErrorSeverity.HIGH,
                    recovery_action=RecoveryAction.RETRY,
                    threshold_count=3,
                    threshold_time_minutes=5
                ),
                ErrorPattern(
                    pattern_id='API_AUTHENTICATION_ERROR',
                    category=ErrorCategory.API,
                    keywords=['authentication', 'unauthorized', 'token', 'login'],
                    severity=ErrorSeverity.CRITICAL,
                    recovery_action=RecoveryAction.RESTART_COMPONENT,
                    threshold_count=2,
                    threshold_time_minutes=10
                ),
                ErrorPattern(
                    pattern_id='MEMORY_ERROR',
                    category=ErrorCategory.SYSTEM,
                    keywords=['memory', 'allocation', 'out of memory'],
                    severity=ErrorSeverity.CRITICAL,
                    recovery_action=RecoveryAction.RESTART_SYSTEM,
                    threshold_count=1,
                    threshold_time_minutes=1
                ),
                ErrorPattern(
                    pattern_id='TRADING_ORDER_ERROR',
                    category=ErrorCategory.TRADING,
                    keywords=['order', 'trade', 'execution', 'broker'],
                    severity=ErrorSeverity.HIGH,
                    recovery_action=RecoveryAction.MANUAL_INTERVENTION,
                    threshold_count=5,
                    threshold_time_minutes=15
                ),
                ErrorPattern(
                    pattern_id='ML_MODEL_ERROR',
                    category=ErrorCategory.ML,
                    keywords=['model', 'prediction', 'training', 'ml'],
                    severity=ErrorSeverity.MEDIUM,
                    recovery_action=RecoveryAction.RESTART_COMPONENT,
                    threshold_count=10,
                    threshold_time_minutes=30
                ),
                ErrorPattern(
                    pattern_id='DATA_CORRUPTION_ERROR',
                    category=ErrorCategory.DATA,
                    keywords=['corrupt', 'invalid', 'parse', 'format'],
                    severity=ErrorSeverity.HIGH,
                    recovery_action=RecoveryAction.RESTART_COMPONENT,
                    threshold_count=3,
                    threshold_time_minutes=10
                ),
                ErrorPattern(
                    pattern_id='SECURITY_ERROR',
                    category=ErrorCategory.SECURITY,
                    keywords=['security', 'encryption', 'decrypt', 'certificate'],
                    severity=ErrorSeverity.CRITICAL,
                    recovery_action=RecoveryAction.EMERGENCY_STOP,
                    threshold_count=1,
                    threshold_time_minutes=1
                )
            ]
            
            for pattern in patterns:
                self.error_patterns[pattern.pattern_id] = pattern
            
            self.logger.info(f"Initialized {len(self.error_patterns)} error patterns")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize error patterns: {e}")
            return False
    
    def _load_error_reports(self):
        """Load previous error reports from file"""
        try:
            if os.path.exists(self.error_reports_file):
                with open(self.error_reports_file, 'rb') as f:
                    self.error_reports = pickle.load(f)
                self.logger.info(f"Loaded {len(self.error_reports)} previous error reports")
            
        except Exception as e:
            self.logger.warning(f"Failed to load previous error reports: {e}")
            self.error_reports = []
    
    def _save_error_reports(self):
        """Save error reports to file"""
        try:
            with open(self.error_reports_file, 'wb') as f:
                pickle.dump(self.error_reports, f)
            
        except Exception as e:
            self.logger.error(f"Failed to save error reports: {e}")
    
    def _start_cleanup_thread(self):
        """Start cleanup thread for old error reports"""
        try:
            self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self.cleanup_thread.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start cleanup thread: {e}")
    
    def _cleanup_loop(self):
        """Cleanup loop for old error reports"""
        cleanup_interval = self.error_config.get('cleanup_interval_hours', 24) * 3600
        
        while self.is_running:
            try:
                time.sleep(cleanup_interval)
                self._cleanup_old_reports()
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                time.sleep(3600)  # 1 hour on error
    
    def _cleanup_old_reports(self):
        """Clean up old error reports"""
        try:
            with self.error_lock:
                # Keep only last 30 days of reports
                cutoff_date = datetime.now() - timedelta(days=30)
                self.error_reports = [
                    report for report in self.error_reports
                    if report.timestamp > cutoff_date
                ]
                
                # Keep only max number of reports
                max_reports = self.error_config.get('max_error_reports', 10000)
                if len(self.error_reports) > max_reports:
                    self.error_reports = self.error_reports[-max_reports:]
                
                # Save cleaned reports
                self._save_error_reports()
                
            self.logger.info(f"Cleaned up error reports, {len(self.error_reports)} remaining")
            
        except Exception as e:
            self.logger.error(f"Error cleanup failed: {e}")
    
    def handle_error(self, component: str, error: Exception, context: Dict[str, Any] = None) -> str:
        """
        Handle error with automatic categorization and recovery
        
        Args:
            component: Component where error occurred
            error: Exception object
            context: Additional context information
            
        Returns:
            Error ID for tracking
        """
        try:
            # Check rate limiting
            if not self._check_rate_limit(component):
                self.logger.warning(f"Rate limit exceeded for component: {component}")
                return ""
            
            # Generate error ID
            error_id = f"ERR_{int(time.time())}_{hash(str(error)) % 10000}"
            
            # Get error information
            error_message = str(error)
            traceback_str = traceback.format_exc()
            
            # Categorize error
            category, severity = self._categorize_error(error_message, traceback_str)
            
            # Determine recovery action
            recovery_action = self._determine_recovery_action(category, severity, error_message)
            
            # Create error report
            error_report = ErrorReport(
                error_id=error_id,
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                component=component,
                message=error_message,
                exception=error,
                traceback_str=traceback_str,
                context=context or {},
                recovery_action=recovery_action
            )
            
            # Store error report
            with self.error_lock:
                self.error_reports.append(error_report)
                self.stats['total_errors'] += 1
                if severity == ErrorSeverity.CRITICAL:
                    self.stats['critical_errors'] += 1
            
            # Log error
            self._log_error(error_report)
            
            # Send notifications
            self._send_notifications(error_report)
            
            # Execute recovery action
            if self.auto_recovery_enabled and recovery_action != RecoveryAction.NONE:
                self._execute_recovery_action(error_report)
            
            # Save reports periodically
            if len(self.error_reports) % 10 == 0:
                self._save_error_reports()
            
            return error_id
            
        except Exception as e:
            # Fallback error handling
            self.logger.critical(f"Error handler failed: {e}")
            print(f"CRITICAL: Error handler failed: {e}")
            return ""
    
    def _check_rate_limit(self, component: str) -> bool:
        """Check if component is within rate limits"""
        try:
            current_time = datetime.now()
            minute_ago = current_time - timedelta(minutes=1)
            
            if component not in self.error_rate_limits:
                self.error_rate_limits[component] = []
            
            # Remove old timestamps
            self.error_rate_limits[component] = [
                timestamp for timestamp in self.error_rate_limits[component]
                if timestamp > minute_ago
            ]
            
            # Check limit
            if len(self.error_rate_limits[component]) >= self.max_errors_per_minute:
                return False
            
            # Add current timestamp
            self.error_rate_limits[component].append(current_time)
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error
    
    def _categorize_error(self, error_message: str, traceback_str: str) -> tuple[ErrorCategory, ErrorSeverity]:
        """Categorize error based on message and traceback"""
        try:
            error_text = (error_message + " " + traceback_str).lower()
            
            # Check against known patterns
            for pattern in self.error_patterns.values():
                if any(keyword in error_text for keyword in pattern.keywords):
                    return pattern.category, pattern.severity
            
            # Default categorization based on common keywords
            if any(keyword in error_text for keyword in ['connection', 'network', 'timeout']):
                return ErrorCategory.NETWORK, ErrorSeverity.HIGH
            elif any(keyword in error_text for keyword in ['api', 'authentication', 'token']):
                return ErrorCategory.API, ErrorSeverity.HIGH
            elif any(keyword in error_text for keyword in ['memory', 'allocation']):
                return ErrorCategory.SYSTEM, ErrorSeverity.CRITICAL
            elif any(keyword in error_text for keyword in ['trade', 'order', 'broker']):
                return ErrorCategory.TRADING, ErrorSeverity.HIGH
            elif any(keyword in error_text for keyword in ['model', 'prediction', 'ml']):
                return ErrorCategory.ML, ErrorSeverity.MEDIUM
            elif any(keyword in error_text for keyword in ['data', 'parse', 'format']):
                return ErrorCategory.DATA, ErrorSeverity.MEDIUM
            else:
                return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
                
        except Exception as e:
            self.logger.error(f"Error categorization failed: {e}")
            return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    def _determine_recovery_action(self, category: ErrorCategory, severity: ErrorSeverity, message: str) -> RecoveryAction:
        """Determine appropriate recovery action"""
        try:
            # Critical errors
            if severity == ErrorSeverity.CRITICAL:
                if category == ErrorCategory.SECURITY:
                    return RecoveryAction.EMERGENCY_STOP
                elif category == ErrorCategory.SYSTEM:
                    return RecoveryAction.RESTART_SYSTEM
                else:
                    return RecoveryAction.RESTART_COMPONENT
            
            # High severity errors
            elif severity == ErrorSeverity.HIGH:
                if category == ErrorCategory.TRADING:
                    return RecoveryAction.MANUAL_INTERVENTION
                elif category in [ErrorCategory.API, ErrorCategory.NETWORK]:
                    return RecoveryAction.RETRY
                else:
                    return RecoveryAction.RESTART_COMPONENT
            
            # Medium severity errors
            elif severity == ErrorSeverity.MEDIUM:
                if category == ErrorCategory.ML:
                    return RecoveryAction.RESTART_COMPONENT
                else:
                    return RecoveryAction.RETRY
            
            # Low severity errors
            else:
                return RecoveryAction.NONE
                
        except Exception as e:
            self.logger.error(f"Recovery action determination failed: {e}")
            return RecoveryAction.NONE
    
    def _log_error(self, error_report: ErrorReport):
        """Log error with appropriate level"""
        try:
            log_message = f"[{error_report.error_id}] {error_report.component}: {error_report.message}"
            
            if error_report.severity == ErrorSeverity.CRITICAL:
                self.logger.critical(log_message)
            elif error_report.severity == ErrorSeverity.MEDIUM:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
            
            # Log full traceback for debugging
            if error_report.traceback_str:
                self.logger.debug(f"Full traceback for {error_report.error_id}:\n{error_report.traceback_str}")
            
        except Exception as e:
            print(f"Error logging failed: {e}")
    
    def _send_notifications(self, error_report: ErrorReport):
        """Send error notifications"""
        try:
            # Only send notifications for high severity and critical errors
            if error_report.severity not in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                return
            
            # Email notification
            if self.email_notifications:
                self._send_email_notification(error_report)
            
            # SMS notification (placeholder)
            if self.sms_notifications:
                self._send_sms_notification(error_report)
            
            # Sound notification (placeholder)
            if self.sound_notifications:
                self._play_error_sound(error_report)
                
        except Exception as e:
            self.logger.error(f"Notification sending failed: {e}")
    
    def _send_email_notification(self, error_report: ErrorReport):
        """Send email notification for error"""
        try:
            email_settings = self.error_config.get('email_settings', {})
            
            if not all([email_settings.get('username'), email_settings.get('password'), email_settings.get('recipient')]):
                return
            
            # Create email message
            subject = f"🚨 EA GlobalFlow Pro Alert - {error_report.severity.value} Error"
            
            body = f"""
EA GlobalFlow Pro Error Alert

Error ID: {error_report.error_id}
Timestamp: {error_report.timestamp}
Severity: {error_report.severity.value}
Category: {error_report.category.value}
Component: {error_report.component}

Error Message:
{error_report.message}

Recovery Action: {error_report.recovery_action.value}

Context:
{json.dumps(error_report.context, indent=2)}

Please check the system logs for more details.
            """
            
            # Send email
            msg = MimeMultipart()
            msg['From'] = email_settings['username']
            msg['To'] = email_settings['recipient']
            msg['Subject'] = subject
            msg.attach(MimeText(body, 'plain'))
            
            server = smtplib.SMTP(email_settings['smtp_server'], email_settings['smtp_port'])
            server.starttls()
            server.login(email_settings['username'], email_settings['password'])
            server.send_message(msg)
            server.quit()
            
            self.logger.info(f"Email notification sent for error {error_report.error_id}")
            
        except Exception as e:
            self.logger.error(f"Email notification failed: {e}")
    
    def _send_sms_notification(self, error_report: ErrorReport):
        """Send SMS notification (placeholder)"""
        try:
            # This would integrate with SMS service
            self.logger.info(f"SMS notification would be sent for error {error_report.error_id}")
            
        except Exception as e:
            self.logger.error(f"SMS notification failed: {e}")
    
    def _play_error_sound(self, error_report: ErrorReport):
        """Play error sound (placeholder)"""
        try:
            # This would play system sound
            self.logger.info(f"Sound notification would be played for error {error_report.error_id}")
            
        except Exception as e:
            self.logger.error(f"Sound notification failed: {e}")
    
    def _execute_recovery_action(self, error_report: ErrorReport):
        """Execute recovery action for error"""
        try:
            action = error_report.recovery_action
            
            if action == RecoveryAction.RETRY:
                self._execute_retry_action(error_report)
            elif action == RecoveryAction.RESTART_COMPONENT:
                self._execute_restart_component(error_report)
            elif action == RecoveryAction.RESTART_SYSTEM:
                self._execute_restart_system(error_report)
            elif action == RecoveryAction.EMERGENCY_STOP:
                self._execute_emergency_stop(error_report)
            elif action == RecoveryAction.MANUAL_INTERVENTION:
                self._request_manual_intervention(error_report)
            
            self.stats['auto_recoveries'] += 1
            
        except Exception as e:
            self.logger.error(f"Recovery action execution failed: {e}")
    
    def _execute_retry_action(self, error_report: ErrorReport):
        """Execute retry recovery action"""
        try:
            component = error_report.component
            
            if component in self.recovery_callbacks:
                callback = self.recovery_callbacks[component]
                if callable(callback):
                    callback('retry', error_report)
            
            self.logger.info(f"Retry action executed for {error_report.error_id}")
            
        except Exception as e:
            self.logger.error(f"Retry action failed: {e}")
    
    def _execute_restart_component(self, error_report: ErrorReport):
        """Execute component restart recovery action"""
        try:
            component = error_report.component
            
            if component in self.recovery_callbacks:
                callback = self.recovery_callbacks[component]
                if callable(callback):
                    callback('restart', error_report)
            
            self.logger.warning(f"Component restart action executed for {error_report.error_id}")
            
        except Exception as e:
            self.logger.error(f"Component restart action failed: {e}")
    
    def _execute_restart_system(self, error_report: ErrorReport):
        """Execute system restart recovery action"""
        try:
            self.logger.critical(f"System restart requested for error {error_report.error_id}")
            self.stats['system_restarts'] += 1
            
            # This would trigger system restart
            # For safety, just log the request
            
        except Exception as e:
            self.logger.error(f"System restart action failed: {e}")
    
    def _execute_emergency_stop(self, error_report: ErrorReport):
        """Execute emergency stop recovery action"""
        try:
            self.logger.critical(f"EMERGENCY STOP triggered for error {error_report.error_id}")
            
            # This would trigger emergency trading stop
            if 'emergency_stop' in self.recovery_callbacks:
                callback = self.recovery_callbacks['emergency_stop']
                if callable(callback):
                    callback('emergency_stop', error_report)
            
        except Exception as e:
            self.logger.error(f"Emergency stop action failed: {e}")
    
    def _request_manual_intervention(self, error_report: ErrorReport):
        """Request manual intervention"""
        try:
            self.logger.critical(f"Manual intervention required for error {error_report.error_id}")
            self.stats['manual_interventions'] += 1
            
            # Send high-priority notification
            # This would trigger immediate alerts
            
        except Exception as e:
            self.logger.error(f"Manual intervention request failed: {e}")
    
    def register_recovery_callback(self, component: str, callback: Callable):
        """Register recovery callback for component"""
        try:
            self.recovery_callbacks[component] = callback
            self.logger.info(f"Recovery callback registered for {component}")
            
        except Exception as e:
            self.logger.error(f"Recovery callback registration failed: {e}")
    
    def resolve_error(self, error_id: str, resolution_notes: str = ""):
        """Mark error as resolved"""
        try:
            with self.error_lock:
                for report in self.error_reports:
                    if report.error_id == error_id:
                        report.resolved = True
                        report.resolution_time = datetime.now()
                        self.stats['resolved_errors'] += 1
                        
                        self.logger.info(f"Error {error_id} marked as resolved: {resolution_notes}")
                        return True
            
            self.logger.warning(f"Error {error_id} not found for resolution")
            return False
            
        except Exception as e:
            self.logger.error(f"Error resolution failed: {e}")
            return False
    
    def get_error_report(self, error_id: str) -> Optional[ErrorReport]:
        """Get specific error report"""
        try:
            for report in self.error_reports:
                if report.error_id == error_id:
                    return report
            return None
            
        except Exception as e:
            self.logger.error(f"Get error report failed: {e}")
            return None
    
    def get_recent_errors(self, hours: int = 24, severity: ErrorSeverity = None) -> List[ErrorReport]:
        """Get recent error reports"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            recent_errors = [
                report for report in self.error_reports
                if report.timestamp > cutoff_time
            ]
            
            if severity:
                recent_errors = [
                    report for report in recent_errors
                    if report.severity == severity
                ]
            
            return sorted(recent_errors, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Get recent errors failed: {e}")
            return []
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        try:
            stats = self.stats.copy()
            
            # Calculate additional metrics
            if stats['total_errors'] > 0:
                stats['resolution_rate'] = (stats['resolved_errors'] / stats['total_errors']) * 100
                stats['critical_error_rate'] = (stats['critical_errors'] / stats['total_errors']) * 100
            else:
                stats['resolution_rate'] = 0
                stats['critical_error_rate'] = 0
            
            # Recent error counts
            recent_24h = len(self.get_recent_errors(hours=24))
            recent_1h = len(self.get_recent_errors(hours=1))
            
            stats['errors_last_24h'] = recent_24h
            stats['errors_last_1h'] = recent_1h
            
            # Error breakdown by category
            category_counts = {}
            for report in self.error_reports[-100:]:  # Last 100 errors
                category = report.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
            
            stats['category_breakdown'] = category_counts
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Get error statistics failed: {e}")
            return self.stats.copy()
    
    def get_unresolved_errors(self) -> List[ErrorReport]:
        """Get all unresolved errors"""
        try:
            return [report for report in self.error_reports if not report.resolved]
            
        except Exception as e:
            self.logger.error(f"Get unresolved errors failed: {e}")
            return []
    
    def get_critical_errors(self, hours: int = 24) -> List[ErrorReport]:
        """Get critical errors from specified time period"""
        try:
            return self.get_recent_errors(hours=hours, severity=ErrorSeverity.CRITICAL)
            
        except Exception as e:
            self.logger.error(f"Get critical errors failed: {e}")
            return []
    
    def clear_resolved_errors(self, older_than_days: int = 7):
        """Clear resolved errors older than specified days"""
        try:
            cutoff_time = datetime.now() - timedelta(days=older_than_days)
            
            with self.error_lock:
                original_count = len(self.error_reports)
                self.error_reports = [
                    report for report in self.error_reports
                    if not (report.resolved and report.resolution_time and report.resolution_time < cutoff_time)
                ]
                
                cleared_count = original_count - len(self.error_reports)
                self.logger.info(f"Cleared {cleared_count} resolved errors older than {older_than_days} days")
                
                # Save updated reports
                self._save_error_reports()
            
        except Exception as e:
            self.logger.error(f"Clear resolved errors failed: {e}")
    
    def export_error_report(self, filepath: str, hours: int = 24) -> bool:
        """Export error report to file"""
        try:
            recent_errors = self.get_recent_errors(hours=hours)
            
            report_data = {
                'export_timestamp': datetime.now().isoformat(),
                'time_period_hours': hours,
                'total_errors': len(recent_errors),
                'statistics': self.get_error_statistics(),
                'errors': []
            }
            
            for error in recent_errors:
                error_data = {
                    'error_id': error.error_id,
                    'timestamp': error.timestamp.isoformat(),
                    'severity': error.severity.value,
                    'category': error.category.value,
                    'component': error.component,
                    'message': error.message,
                    'recovery_action': error.recovery_action.value,
                    'resolved': error.resolved,
                    'resolution_time': error.resolution_time.isoformat() if error.resolution_time else None,
                    'context': error.context
                }
                report_data['errors'].append(error_data)
            
            with open(filepath, 'w') as f:
                json.dump(report_data, f, indent=2)
            
            self.logger.info(f"Error report exported to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error report export failed: {e}")
            return False
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling system"""
        try:
            test_results = {
                'timestamp': datetime.now().isoformat(),
                'tests_passed': 0,
                'tests_failed': 0,
                'test_details': []
            }
            
            # Test 1: Handle a test error
            try:
                test_error = Exception("Test error for validation")
                error_id = self.handle_error("test_component", test_error, {'test': True})
                
                if error_id:
                    test_results['tests_passed'] += 1
                    test_results['test_details'].append("✅ Error handling test passed")
                else:
                    test_results['tests_failed'] += 1
                    test_results['test_details'].append("❌ Error handling test failed")
            except Exception as e:
                test_results['tests_failed'] += 1
                test_results['test_details'].append(f"❌ Error handling test exception: {e}")
            
            # Test 2: Categorization
            try:
                category, severity = self._categorize_error("connection timeout error", "network traceback")
                if category == ErrorCategory.NETWORK and severity == ErrorSeverity.HIGH:
                    test_results['tests_passed'] += 1
                    test_results['test_details'].append("✅ Error categorization test passed")
                else:
                    test_results['tests_failed'] += 1
                    test_results['test_details'].append("❌ Error categorization test failed")
            except Exception as e:
                test_results['tests_failed'] += 1
                test_results['test_details'].append(f"❌ Error categorization test exception: {e}")
            
            # Test 3: Statistics
            try:
                stats = self.get_error_statistics()
                if isinstance(stats, dict) and 'total_errors' in stats:
                    test_results['tests_passed'] += 1
                    test_results['test_details'].append("✅ Statistics test passed")
                else:
                    test_results['tests_failed'] += 1
                    test_results['test_details'].append("❌ Statistics test failed")
            except Exception as e:
                test_results['tests_failed'] += 1
                test_results['test_details'].append(f"❌ Statistics test exception: {e}")
            
            return test_results
            
        except Exception as e:
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'tests_passed': 0,
                'tests_failed': 1
            }
    
    def is_healthy(self) -> bool:
        """Check if error handler is healthy"""
        try:
            return (
                self.is_initialized and
                self.is_running and
                len(self.error_patterns) > 0
            )
        except:
            return False
    
    def stop(self):
        """Stop error handler"""
        try:
            self.is_running = False
            self._save_error_reports()
            self.logger.info("Error Handler stopped")
            
        except Exception as e:
            print(f"Error stopping error handler: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create error handler
    error_handler = ErrorHandler()
    
    # Initialize
    if error_handler.initialize():
        print("✅ Error Handler initialized successfully")
        
        # Test error handling
        test_results = error_handler.test_error_handling()
        print(f"Test results: {json.dumps(test_results, indent=2)}")
        
        # Test with sample errors
        try:
            raise ConnectionError("Test connection error")
        except Exception as e:
            error_id = error_handler.handle_error("test_component", e, {'test_context': 'sample_error'})
            print(f"Handled error with ID: {error_id}")
        
        # Get statistics
        stats = error_handler.get_error_statistics()
        print(f"Error statistics: {stats}")
        
        # Get recent errors
        recent_errors = error_handler.get_recent_errors(hours=1)
        print(f"Recent errors: {len(recent_errors)}")
        
        # Export error report
        export_file = "test_error_report.json"
        if error_handler.export_error_report(export_file, hours=24):
            print(f"Error report exported to {export_file}")
        
        # Stop
        error_handler.stop()
    else:
        print("❌ Error Handler initialization failed")everity.HIGH:
                self.logger.error(log_message)
            elif error_report.severity == ErrorS