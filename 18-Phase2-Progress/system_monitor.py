#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - System Health Monitoring and Performance Tracking
=========================================================================

Comprehensive system monitoring solution providing:
- Real-time system health monitoring
- Performance metrics tracking
- Resource usage monitoring
- Component status tracking
- Alert generation for system issues
- Historical performance analysis

Author: EA GlobalFlow Pro Development Team
Date: August 2025
Version: v0.1 - Production Ready
"""

import asyncio
import json
import logging
import psutil
import threading
import time
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import platform
import sys
import traceback
import gc
import weakref

# Internal imports
from error_handler import ErrorHandler
from security_manager import SecurityManager

class HealthStatus(Enum):
    """System health status enumeration"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    ERROR = "ERROR"
    UNKNOWN = "UNKNOWN"

class ComponentType(Enum):
    """Component type enumeration"""
    CORE_SYSTEM = "CORE_SYSTEM"
    API_BRIDGE = "API_BRIDGE"
    ML_SYSTEM = "ML_SYSTEM"
    TRADING_ENGINE = "TRADING_ENGINE"
    DATA_FEED = "DATA_FEED"
    RISK_MANAGER = "RISK_MANAGER"
    DATABASE = "DATABASE"
    NETWORK = "NETWORK"
    EXTERNAL_SERVICE = "EXTERNAL_SERVICE"

@dataclass
class SystemMetrics:
    """Container for system performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    memory_available: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    thread_count: int
    open_files: int
    uptime_seconds: float
    load_average: List[float] = field(default_factory=list)

@dataclass
class ComponentHealth:
    """Container for component health information"""
    component_name: str
    component_type: ComponentType
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_count: int
    warning_count: int
    uptime_seconds: float
    last_error: Optional[str] = None
    last_warning: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceAlert:
    """Container for performance alerts"""
    alert_id: str
    alert_type: str
    severity: HealthStatus
    component: str
    message: str
    timestamp: datetime
    threshold_value: float
    actual_value: float
    resolved: bool = False
    resolution_time: Optional[datetime] = None

@dataclass
class TradingMetrics:
    """Container for trading-specific metrics"""
    timestamp: datetime
    signals_processed: int
    signals_generated: int
    trades_executed: int
    avg_processing_time_ms: float
    ml_predictions: int
    ml_accuracy: float
    api_calls_made: int
    api_errors: int
    data_feed_latency_ms: float

class SystemMonitor:
    """
    Advanced System Health Monitoring and Performance Tracking
    
    Provides comprehensive monitoring of:
    - System resources (CPU, memory, disk, network)
    - Component health and status
    - Performance metrics and trends
    - Error and warning tracking
    - Alert generation and notification
    - Historical analysis and reporting
    """
    
    def __init__(self, config_path: str = "Config/ea_config_v01.json"):
        """Initialize System Monitor"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.monitor_config = self.config.get('system_monitor', {})
        
        # Initialize logging
        self.logger = logging.getLogger('SystemMonitor')
        self.logger.setLevel(logging.INFO)
        
        # Initialize core components
        self.error_handler = ErrorHandler() if 'ErrorHandler' in globals() else None
        self.security_manager = SecurityManager() if 'SecurityManager' in globals() else None
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.start_time = datetime.now()
        
        # Component registry
        self.registered_components: Dict[str, ComponentHealth] = {}
        self.component_callbacks: Dict[str, callable] = {}
        
        # Metrics storage
        self.system_metrics_history: List[SystemMetrics] = []
        self.trading_metrics_history: List[TradingMetrics] = []
        self.active_alerts: Dict[str, PerformanceAlert] = {}
        self.alert_history: List[PerformanceAlert] = []
        
        # Configuration parameters
        self.check_interval_seconds = self.monitor_config.get('check_interval_seconds', 30)
        self.max_history_items = self.monitor_config.get('max_history_items', 1000)
        self.alert_thresholds = self.monitor_config.get('alert_thresholds', {})
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.monitoring_lock = threading.RLock()
        
        # Database for persistence
        self.db_connection = self._init_database()
        
        # System information
        self.system_info = self._get_system_info()
        
        # Performance baselines
        self.performance_baselines = self._load_performance_baselines()
        
        self.logger.info("System Monitor initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load system monitor configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default configuration if file loading fails"""
        return {
            'system_monitor': {
                'check_interval_seconds': 30,
                'max_history_items': 1000,
                'alert_thresholds': {
                    'cpu_usage': 80.0,
                    'memory_usage': 85.0,
                    'disk_usage': 90.0,
                    'response_time_ms': 5000,
                    'error_rate': 5.0,
                    'api_error_rate': 10.0
                },
                'enable_email_alerts': True,
                'enable_phone_alerts': True,
                'critical_alert_email': "pajitmenonai@gmail.com",
                'critical_alert_phone': "+971507423656"
            }
        }

    def _init_database(self) -> sqlite3.Connection:
        """Initialize database for monitoring data"""
        try:
            db_path = Path("Data/system_monitor.db")
            db_path.parent.mkdir(exist_ok=True)
            
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    cpu_usage REAL,
                    memory_usage REAL,
                    memory_available REAL,
                    disk_usage REAL,
                    network_bytes_sent INTEGER,
                    network_bytes_recv INTEGER,
                    process_count INTEGER,
                    thread_count INTEGER,
                    uptime_seconds REAL
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS component_health (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    component_name TEXT,
                    component_type TEXT,
                    status TEXT,
                    response_time_ms REAL,
                    error_count INTEGER,
                    warning_count INTEGER,
                    last_error TEXT,
                    custom_metrics TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS performance_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    alert_id TEXT UNIQUE,
                    alert_type TEXT,
                    severity TEXT,
                    component TEXT,
                    message TEXT,
                    threshold_value REAL,
                    actual_value REAL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolution_time DATETIME
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trading_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    signals_processed INTEGER,
                    signals_generated INTEGER,
                    trades_executed INTEGER,
                    avg_processing_time_ms REAL,
                    ml_predictions INTEGER,
                    ml_accuracy REAL,
                    api_calls_made INTEGER,
                    api_errors INTEGER,
                    data_feed_latency_ms REAL
                )
            ''')
            
            conn.commit()
            return conn
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            return None

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        try:
            return {
                'platform': platform.platform(),
                'system': platform.system(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'python_version': sys.version,
                'cpu_count': psutil.cpu_count(),
                'total_memory': psutil.virtual_memory().total,
                'total_disk': sum(partition.total for partition in psutil.disk_usage('/') if hasattr(partition, 'total')),
                'network_interfaces': list(psutil.net_if_addrs().keys()),
                'boot_time': psutil.boot_time(),
                'timezone': str(datetime.now().astimezone().tzinfo)
            }
        except Exception as e:
            self.logger.error(f"Failed to get system info: {e}")
            return {}

    def _load_performance_baselines(self) -> Dict[str, float]:
        """Load or calculate performance baselines"""
        try:
            baseline_file = Path("Data/performance_baselines.json")
            if baseline_file.exists():
                with open(baseline_file, 'r') as f:
                    return json.load(f)
            else:
                # Default baselines
                return {
                    'cpu_baseline': 20.0,
                    'memory_baseline': 40.0,
                    'response_time_baseline': 100.0,
                    'processing_time_baseline': 50.0
                }
        except Exception as e:
            self.logger.error(f"Failed to load performance baselines: {e}")
            return {}

    async def start_monitoring(self) -> bool:
        """Start system monitoring"""
        try:
            if self.is_monitoring:
                self.logger.warning("Monitoring already started")
                return True
            
            self.logger.info("🔍 Starting System Health Monitoring...")
            
            self.is_monitoring = True
            self.start_time = datetime.now()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitor_thread.start()
            
            # Initial system check
            await self._perform_system_check()
            
            self.logger.info("✅ System monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring: {e}")
            return False

    async def stop_monitoring(self) -> bool:
        """Stop system monitoring"""
        try:
            if not self.is_monitoring:
                self.logger.warning("Monitoring not running")
                return True
            
            self.logger.info("🔄 Stopping System Health Monitoring...")
            
            self.is_monitoring = False
            
            # Wait for monitoring thread to finish
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.monitor_thread.join(timeout=5.0)
            
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("✅ System monitoring stopped successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop monitoring: {e}")
            return False

    def _monitoring_loop(self):
        """Main monitoring loop - runs in separate thread"""
        self.logger.info("Monitoring loop started")
        
        while self.is_monitoring:
            try:
                # Perform system check
                asyncio.run(self._perform_system_check())
                
                # Check component health
                self._check_all_components()
                
                # Process alerts
                self._process_alerts()
                
                # Cleanup old data
                self._cleanup_old_data()
                
                # Sleep until next check
                time.sleep(self.check_interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(5)  # Error cooldown

    async def _perform_system_check(self):
        """Perform comprehensive system health check"""
        try:
            # Collect system metrics
            metrics = await self._collect_system_metrics()
            
            # Store metrics
            self._store_system_metrics(metrics)
            
            # Check for threshold violations
            await self._check_system_thresholds(metrics)
            
        except Exception as e:
            self.logger.error(f"System check failed: {e}")

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system metrics"""
        try:
            # CPU metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            memory_available = memory.available
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage = disk.percent
            
            # Network metrics
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process metrics
            process_count = len(psutil.pids())
            
            # Thread count for current process
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            # Open files count
            try:
                open_files = len(current_process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            # Uptime
            uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Load average (Unix systems)
            load_average = []
            try:
                if hasattr(psutil, 'getloadavg'):
                    load_average = list(psutil.getloadavg())
            except:
                pass
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                memory_available=memory_available,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                thread_count=thread_count,
                open_files=open_files,
                uptime_seconds=uptime_seconds,
                load_average=load_average
            )
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=0.0,
                memory_usage=0.0,
                memory_available=0.0,
                disk_usage=0.0,
                network_io={},
                process_count=0,
                thread_count=0,
                open_files=0,
                uptime_seconds=0.0
            )

    def _store_system_metrics(self, metrics: SystemMetrics):
        """Store system metrics in memory and database"""
        try:
            # Add to memory storage
            with self.monitoring_lock:
                self.system_metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.system_metrics_history) > self.max_history_items:
                    self.system_metrics_history = self.system_metrics_history[-self.max_history_items:]
            
            # Store in database
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT INTO system_metrics 
                    (cpu_usage, memory_usage, memory_available, disk_usage, 
                     network_bytes_sent, network_bytes_recv, process_count, 
                     thread_count, uptime_seconds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.cpu_usage,
                    metrics.memory_usage,
                    metrics.memory_available,
                    metrics.disk_usage,
                    metrics.network_io.get('bytes_sent', 0),
                    metrics.network_io.get('bytes_recv', 0),
                    metrics.process_count,
                    metrics.thread_count,
                    metrics.uptime_seconds
                ))
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store system metrics: {e}")

    async def _check_system_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds and generate alerts"""
        try:
            thresholds = self.alert_thresholds
            
            # CPU usage check
            if metrics.cpu_usage > thresholds.get('cpu_usage', 80.0):
                await self._generate_alert(
                    'CPU_USAGE_HIGH',
                    HealthStatus.WARNING if metrics.cpu_usage < 90 else HealthStatus.CRITICAL,
                    'SYSTEM',
                    f'CPU usage is {metrics.cpu_usage:.1f}%',
                    thresholds.get('cpu_usage', 80.0),
                    metrics.cpu_usage
                )
            
            # Memory usage check
            if metrics.memory_usage > thresholds.get('memory_usage', 85.0):
                await self._generate_alert(
                    'MEMORY_USAGE_HIGH',
                    HealthStatus.WARNING if metrics.memory_usage < 95 else HealthStatus.CRITICAL,
                    'SYSTEM',
                    f'Memory usage is {metrics.memory_usage:.1f}%',
                    thresholds.get('memory_usage', 85.0),
                    metrics.memory_usage
                )
            
            # Disk usage check
            if metrics.disk_usage > thresholds.get('disk_usage', 90.0):
                await self._generate_alert(
                    'DISK_USAGE_HIGH',
                    HealthStatus.WARNING if metrics.disk_usage < 95 else HealthStatus.CRITICAL,
                    'SYSTEM',
                    f'Disk usage is {metrics.disk_usage:.1f}%',
                    thresholds.get('disk_usage', 90.0),
                    metrics.disk_usage
                )
            
            # Thread count check
            if metrics.thread_count > 50:  # Threshold for thread count
                await self._generate_alert(
                    'THREAD_COUNT_HIGH',
                    HealthStatus.WARNING,
                    'SYSTEM',
                    f'Thread count is {metrics.thread_count}',
                    50,
                    metrics.thread_count
                )
                
        except Exception as e:
            self.logger.error(f"Threshold checking failed: {e}")

    def register_component(self, component_name: str, component_type: ComponentType, 
                          health_check_callback: Optional[callable] = None) -> bool:
        """Register a component for monitoring"""
        try:
            with self.monitoring_lock:
                component_health = ComponentHealth(
                    component_name=component_name,
                    component_type=component_type,
                    status=HealthStatus.UNKNOWN,
                    last_check=datetime.now(),
                    response_time_ms=0.0,
                    error_count=0,
                    warning_count=0,
                    uptime_seconds=0.0
                )
                
                self.registered_components[component_name] = component_health
                
                if health_check_callback:
                    self.component_callbacks[component_name] = health_check_callback
            
            self.logger.info(f"Component registered: {component_name} ({component_type.value})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register component {component_name}: {e}")
            return False

    def unregister_component(self, component_name: str) -> bool:
        """Unregister a component from monitoring"""
        try:
            with self.monitoring_lock:
                if component_name in self.registered_components:
                    del self.registered_components[component_name]
                
                if component_name in self.component_callbacks:
                    del self.component_callbacks[component_name]
            
            self.logger.info(f"Component unregistered: {component_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unregister component {component_name}: {e}")
            return False

    def _check_all_components(self):
        """Check health of all registered components"""
        try:
            for component_name in list(self.registered_components.keys()):
                self._check_component_health(component_name)
                
        except Exception as e:
            self.logger.error(f"Component health check failed: {e}")

    def _check_component_health(self, component_name: str):
        """Check health of a specific component"""
        try:
            if component_name not in self.registered_components:
                return
            
            component = self.registered_components[component_name]
            callback = self.component_callbacks.get(component_name)
            
            start_time = time.time()
            
            if callback:
                try:
                    # Execute health check callback
                    health_result = callback()
                    
                    # Process result
                    if isinstance(health_result, dict):
                        component.status = HealthStatus(health_result.get('status', 'UNKNOWN'))
                        component.custom_metrics = health_result.get('metrics', {})
                        
                        if 'error' in health_result:
                            component.error_count += 1
                            component.last_error = health_result['error']
                        
                        if 'warning' in health_result:
                            component.warning_count += 1
                            component.last_warning = health_result['warning']
                    else:
                        # Simple boolean result
                        component.status = HealthStatus.HEALTHY if health_result else HealthStatus.ERROR
                        
                except Exception as callback_error:
                    component.status = HealthStatus.ERROR
                    component.error_count += 1
                    component.last_error = str(callback_error)
            else:
                # No callback - assume healthy if recently updated
                time_since_check = (datetime.now() - component.last_check).total_seconds()
                if time_since_check > 300:  # 5 minutes
                    component.status = HealthStatus.WARNING
            
            # Update component metrics
            component.response_time_ms = (time.time() - start_time) * 1000
            component.last_check = datetime.now()
            component.uptime_seconds = (datetime.now() - self.start_time).total_seconds()
            
            # Store component health
            self._store_component_health(component)
            
        except Exception as e:
            self.logger.error(f"Component {component_name} health check failed: {e}")

    def _store_component_health(self, component: ComponentHealth):
        """Store component health in database"""
        try:
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT INTO component_health 
                    (component_name, component_type, status, response_time_ms, 
                     error_count, warning_count, last_error, custom_metrics)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    component.component_name,
                    component.component_type.value,
                    component.status.value,
                    component.response_time_ms,
                    component.error_count,
                    component.warning_count,
                    component.last_error,
                    json.dumps(component.custom_metrics)
                ))
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store component health: {e}")

    async def _generate_alert(self, alert_type: str, severity: HealthStatus, 
                            component: str, message: str, threshold: float, actual: float):
        """Generate a performance alert"""
        try:
            alert_id = f"{alert_type}_{component}_{int(time.time())}"
            
            # Check if similar alert is already active
            existing_alert = None
            for alert in self.active_alerts.values():
                if (alert.alert_type == alert_type and 
                    alert.component == component and 
                    not alert.resolved):
                    existing_alert = alert
                    break
            
            if existing_alert:
                # Update existing alert
                existing_alert.actual_value = actual
                existing_alert.timestamp = datetime.now()
            else:
                # Create new alert
                alert = PerformanceAlert(
                    alert_id=alert_id,
                    alert_type=alert_type,
                    severity=severity,
                    component=component,
                    message=message,
                    timestamp=datetime.now(),
                    threshold_value=threshold,
                    actual_value=actual
                )
                
                self.active_alerts[alert_id] = alert
                self.alert_history.append(alert)
                
                # Store in database
                await self._store_alert(alert)
                
                # Send notifications for critical alerts
                if severity == HealthStatus.CRITICAL:
                    await self._send_critical_alert_notification(alert)
                
                self.logger.warning(f"Alert generated: {alert_type} - {message}")
                
        except Exception as e:
            self.logger.error(f"Failed to generate alert: {e}")

    async def _store_alert(self, alert: PerformanceAlert):
        """Store alert in database"""
        try:
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT OR REPLACE INTO performance_alerts 
                    (alert_id, alert_type, severity, component, message, 
                     threshold_value, actual_value, resolved, resolution_time)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    alert.alert_id,
                    alert.alert_type,
                    alert.severity.value,
                    alert.component,
                    alert.message,
                    alert.threshold_value,
                    alert.actual_value,
                    alert.resolved,
                    alert.resolution_time
                ))
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to store alert: {e}")

    async def _send_critical_alert_notification(self, alert: PerformanceAlert):
        """Send critical alert notifications"""
        try:
            if self.monitor_config.get('enable_email_alerts', False):
                await self._send_email_alert(alert)
            
            if self.monitor_config.get('enable_phone_alerts', False):
                await self._send_phone_alert(alert)
                
        except Exception as e:
            self.logger.error(f"Failed to send critical alert notification: {e}")

    async def _send_email_alert(self, alert: PerformanceAlert):
        """Send email alert (placeholder implementation)"""
        try:
            # This would integrate with actual email service
            email = self.monitor_config.get('critical_alert_email')
            if email:
                self.logger.info(f"Email alert would be sent to {email}: {alert.message}")
                
        except Exception as e:
            self.logger.error(f"Email alert failed: {e}")

    async def _send_phone_alert(self, alert: PerformanceAlert):
        """Send phone alert (placeholder implementation)"""
        try:
            # This would integrate with actual SMS/WhatsApp service
            phone = self.monitor_config.get('critical_alert_phone')
            if phone:
                self.logger.info(f"Phone alert would be sent to {phone}: {alert.message}")
                
        except Exception as e:
            self.logger.error(f"Phone alert failed: {e}")

    def _process_alerts(self):
        """Process and manage active alerts"""
        try:
            current_time = datetime.now()
            resolved_alerts = []
            
            for alert_id, alert in self.active_alerts.items():
                # Check if alert should be auto-resolved
                if not alert.resolved:
                    # Auto-resolve alerts older than 1 hour if condition improved
                    if (current_time - alert.timestamp).total_seconds() > 3600:
                        if self._check_alert_resolution(alert):
                            alert.resolved = True
                            alert.resolution_time = current_time
                            resolved_alerts.append(alert_id)
            
            # Remove resolved alerts from active list
            for alert_id in resolved_alerts:
                if alert_id in self.active_alerts:
                    del self.active_alerts[alert_id]
                    self.logger.info(f"Alert auto-resolved: {alert_id}")
                    
        except Exception as e:
            self.logger.error(f"Alert processing failed: {e}")

    def _check_alert_resolution(self, alert: PerformanceAlert) -> bool:
        """Check if alert condition has been resolved"""
        try:
            # Get latest metrics to check if threshold violation persists
            if self.system_metrics_history:
                latest_metrics = self.system_metrics_history[-1]
                
                if alert.alert_type == 'CPU_USAGE_HIGH':
                    return latest_metrics.cpu_usage < alert.threshold_value
                elif alert.alert_type == 'MEMORY_USAGE_HIGH':
                    return latest_metrics.memory_usage < alert.threshold_value
                elif alert.alert_type == 'DISK_USAGE_HIGH':
                    return latest_metrics.disk_usage < alert.threshold_value
            
            return False
            
        except Exception as e:
            self.logger.error(f"Alert resolution check failed: {e}")
            return False

    def _cleanup_old_data(self):
        """Cleanup old monitoring data"""
        try:
            current_time = datetime.now()
            
            # Cleanup old metrics (keep last 24 hours in memory)
            cutoff_time = current_time - timedelta(hours=24)
            
            with self.monitoring_lock:
                self.system_metrics_history = [
                    m for m in self.system_metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                self.trading_metrics_history = [
                    m for m in self.trading_metrics_history 
                    if m.timestamp > cutoff_time
                ]
                
                # Cleanup old alerts (keep last 7 days)
                alert_cutoff = current_time - timedelta(days=7)
                self.alert_history = [
                    a for a in self.alert_history 
                    if a.timestamp > alert_cutoff
                ]
            
            # Cleanup database (keep last 30 days)
            if self.db_connection:
                db_cutoff = current_time - timedelta(days=30)
                self.db_connection.execute(
                    "DELETE FROM system_metrics WHERE timestamp < ?",
                    (db_cutoff,)
                )
                self.db_connection.execute(
                    "DELETE FROM component_health WHERE timestamp < ?",
                    (db_cutoff,)
                )
                self.db_connection.execute(
                    "DELETE FROM performance_alerts WHERE timestamp < ? AND resolved = 1",
                    (db_cutoff,)
                )
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Data cleanup failed: {e}")

    def update_trading_metrics(self, metrics: TradingMetrics):
        """Update trading-specific metrics"""
        try:
            with self.monitoring_lock:
                self.trading_metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.trading_metrics_history) > self.max_history_items:
                    self.trading_metrics_history = self.trading_metrics_history[-self.max_history_items:]
            
            # Store in database
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT INTO trading_metrics 
                    (signals_processed, signals_generated, trades_executed, 
                     avg_processing_time_ms, ml_predictions, ml_accuracy, 
                     api_calls_made, api_errors, data_feed_latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    metrics.signals_processed,
                    metrics.signals_generated,
                    metrics.trades_executed,
                    metrics.avg_processing_time_ms,
                    metrics.ml_predictions,
                    metrics.ml_accuracy,
                    metrics.api_calls_made,
                    metrics.api_errors,
                    metrics.data_feed_latency_ms
                ))
                self.db_connection.commit()
                
        except Exception as e:
            self.logger.error(f"Failed to update trading metrics: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            with self.monitoring_lock:
                latest_metrics = self.system_metrics_history[-1] if self.system_metrics_history else None
                latest_trading = self.trading_metrics_history[-1] if self.trading_metrics_history else None
                
                # Overall health assessment
                overall_health = self._assess_overall_health()
                
                return {
                    'monitoring_active': self.is_monitoring,
                    'uptime_seconds': (datetime.now() - self.start_time).total_seconds(),
                    'overall_health': overall_health.value,
                    'system_info': self.system_info,
                    'latest_system_metrics': {
                        'cpu_usage': latest_metrics.cpu_usage if latest_metrics else 0,
                        'memory_usage': latest_metrics.memory_usage if latest_metrics else 0,
                        'disk_usage': latest_metrics.disk_usage if latest_metrics else 0,
                        'thread_count': latest_metrics.thread_count if latest_metrics else 0,
                        'timestamp': latest_metrics.timestamp.isoformat() if latest_metrics else None
                    },
                    'latest_trading_metrics': {
                        'signals_processed': latest_trading.signals_processed if latest_trading else 0,
                        'signals_generated': latest_trading.signals_generated if latest_trading else 0,
                        'trades_executed': latest_trading.trades_executed if latest_trading else 0,
                        'ml_accuracy': latest_trading.ml_accuracy if latest_trading else 0,
                        'timestamp': latest_trading.timestamp.isoformat() if latest_trading else None
                    },
                    'registered_components': {
                        name: {
                            'type': comp.component_type.value,
                            'status': comp.status.value,
                            'response_time_ms': comp.response_time_ms,
                            'error_count': comp.error_count,
                            'last_check': comp.last_check.isoformat()
                        } for name, comp in self.registered_components.items()
                    },
                    'active_alerts': len(self.active_alerts),
                    'critical_alerts': len([a for a in self.active_alerts.values() if a.severity == HealthStatus.CRITICAL]),
                    'performance_baselines': self.performance_baselines
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get system status: {e}")
            return {'error': str(e)}

    def _assess_overall_health(self) -> HealthStatus:
        """Assess overall system health"""
        try:
            if not self.system_metrics_history:
                return HealthStatus.UNKNOWN
            
            # Check for critical alerts
            critical_alerts = [a for a in self.active_alerts.values() if a.severity == HealthStatus.CRITICAL]
            if critical_alerts:
                return HealthStatus.CRITICAL
            
            # Check component health
            unhealthy_components = [
                c for c in self.registered_components.values() 
                if c.status in [HealthStatus.ERROR, HealthStatus.CRITICAL]
            ]
            if unhealthy_components:
                return HealthStatus.ERROR
            
            # Check warning alerts
            warning_alerts = [a for a in self.active_alerts.values() if a.severity == HealthStatus.WARNING]
            if warning_alerts:
                return HealthStatus.WARNING
            
            # Check latest metrics
            latest_metrics = self.system_metrics_history[-1]
            if (latest_metrics.cpu_usage > 80 or 
                latest_metrics.memory_usage > 85 or 
                latest_metrics.disk_usage > 90):
                return HealthStatus.WARNING
            
            return HealthStatus.HEALTHY
            
        except Exception as e:
            self.logger.error(f"Health assessment failed: {e}")
            return HealthStatus.UNKNOWN

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate performance report for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            # Filter metrics for time period
            period_system_metrics = [
                m for m in self.system_metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            period_trading_metrics = [
                m for m in self.trading_metrics_history 
                if m.timestamp > cutoff_time
            ]
            
            if not period_system_metrics:
                return {'error': 'No metrics available for specified period'}
            
            # Calculate averages and statistics
            avg_cpu = sum(m.cpu_usage for m in period_system_metrics) / len(period_system_metrics)
            max_cpu = max(m.cpu_usage for m in period_system_metrics)
            avg_memory = sum(m.memory_usage for m in period_system_metrics) / len(period_system_metrics)
            max_memory = max(m.memory_usage for m in period_system_metrics)
            
            # Trading statistics
            if period_trading_metrics:
                total_signals = sum(m.signals_processed for m in period_trading_metrics)
                total_generated = sum(m.signals_generated for m in period_trading_metrics)
                avg_processing_time = sum(m.avg_processing_time_ms for m in period_trading_metrics) / len(period_trading_metrics)
                avg_ml_accuracy = sum(m.ml_accuracy for m in period_trading_metrics) / len(period_trading_metrics)
            else:
                total_signals = total_generated = avg_processing_time = avg_ml_accuracy = 0
            
            return {
                'report_period_hours': hours,
                'metrics_count': len(period_system_metrics),
                'system_performance': {
                    'avg_cpu_usage': round(avg_cpu, 2),
                    'max_cpu_usage': round(max_cpu, 2),
                    'avg_memory_usage': round(avg_memory, 2),
                    'max_memory_usage': round(max_memory, 2)
                },
                'trading_performance': {
                    'total_signals_processed': total_signals,
                    'total_signals_generated': total_generated,
                    'avg_processing_time_ms': round(avg_processing_time, 2),
                    'avg_ml_accuracy': round(avg_ml_accuracy, 3)
                },
                'alerts_summary': {
                    'total_alerts': len([a for a in self.alert_history if a.timestamp > cutoff_time]),
                    'critical_alerts': len([a for a in self.alert_history if a.timestamp > cutoff_time and a.severity == HealthStatus.CRITICAL]),
                    'resolved_alerts': len([a for a in self.alert_history if a.timestamp > cutoff_time and a.resolved])
                },
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Performance report generation failed: {e}")
            return {'error': str(e)}

# Global instance
system_monitor = None

def get_system_monitor() -> SystemMonitor:
    """Get singleton instance of System Monitor"""
    global system_monitor
    if system_monitor is None:
        system_monitor = SystemMonitor()
    return system_monitor

if __name__ == "__main__":
    # Test the System Monitor
    import asyncio
    
    async def main():
        monitor = SystemMonitor()
        
        # Start monitoring
        await monitor.start_monitoring()
        
        # Register a test component
        def test_health_check():
            return {'status': 'HEALTHY', 'metrics': {'test_value': 42}}
        
        monitor.register_component("TestComponent", ComponentType.CORE_SYSTEM, test_health_check)
        
        # Let it run for a while
        await asyncio.sleep(60)
        
        # Get system status
        status = monitor.get_system_status()
        print(f"System Status: {json.dumps(status, indent=2)}")
        
        # Get performance report
        report = monitor.get_performance_report(1)
        print(f"Performance Report: {json.dumps(report, indent=2)}")
        
        # Stop monitoring
        await monitor.stop_monitoring()
    
    asyncio.run(main())
