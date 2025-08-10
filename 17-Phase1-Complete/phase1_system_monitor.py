#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - System Health Monitor
Comprehensive system monitoring with performance tracking and alerts

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
import psutil
import platform
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import socket
import subprocess

class HealthStatus(Enum):
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    UNKNOWN = "UNKNOWN"

class AlertLevel(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, int]
    process_count: int
    uptime: float
    temperature: Optional[float] = None

@dataclass
class ComponentHealth:
    component_name: str
    status: HealthStatus
    last_heartbeat: datetime
    error_count: int
    performance_score: float
    details: Dict[str, Any]

@dataclass
class SystemAlert:
    alert_id: str
    level: AlertLevel
    component: str
    message: str
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class SystemMonitor:
    """
    System Health Monitor for EA GlobalFlow Pro v0.1
    Monitors system resources, component health, and performance
    """
    
    def __init__(self, config_manager=None, error_handler=None):
        """Initialize system monitor"""
        self.config_manager = config_manager
        self.error_handler = error_handler
        self.logger = logging.getLogger('SystemMonitor')
        
        # Configuration
        self.monitor_config = {}
        self.is_initialized = False
        self.is_monitoring = False
        
        # Monitoring intervals
        self.system_check_interval = 10  # seconds
        self.component_check_interval = 30  # seconds
        self.performance_check_interval = 60  # seconds
        
        # System metrics
        self.current_metrics = None
        self.metrics_history = []
        self.max_history_size = 1440  # 24 hours at 1-minute intervals
        
        # Component health tracking
        self.component_health = {}
        self.registered_components = {}
        
        # Alert system
        self.active_alerts = {}
        self.alert_history = []
        self.alert_callbacks = []
        
        # Thresholds
        self.cpu_warning_threshold = 80.0
        self.cpu_critical_threshold = 95.0
        self.memory_warning_threshold = 85.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 90.0
        self.disk_critical_threshold = 98.0
        
        # Performance tracking
        self.performance_metrics = {
            'system_uptime': 0.0,
            'ea_uptime': 0.0,
            'total_trades': 0,
            'successful_trades': 0,
            'api_calls_made': 0,
            'api_errors': 0,
            'ml_predictions': 0,
            'system_restarts': 0
        }
        
        # Monitoring threads
        self.system_thread = None
        self.component_thread = None
        self.performance_thread = None
        
        # Start time
        self.start_time = datetime.now()
        
    def initialize(self) -> bool:
        """
        Initialize system monitor
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing System Monitor v0.1...")
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Initialize system info
            self._initialize_system_info()
            
            # Register core components
            self._register_core_components()
            
            self.is_initialized = True
            self.logger.info("✅ System Monitor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System Monitor initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("system_monitor_init", e)
            return False
    
    def _load_config(self) -> bool:
        """Load system monitor configuration"""
        try:
            if self.config_manager:
                self.monitor_config = self.config_manager.get_config('system_monitor', {})
            else:
                # Default configuration
                self.monitor_config = {
                    'enabled': True,
                    'intervals': {
                        'system_check': 10,
                        'component_check': 30,
                        'performance_check': 60
                    },
                    'thresholds': {
                        'cpu_warning': 80.0,
                        'cpu_critical': 95.0,
                        'memory_warning': 85.0,
                        'memory_critical': 95.0,
                        'disk_warning': 90.0,
                        'disk_critical': 98.0
                    },
                    'alerts': {
                        'email_enabled': True,
                        'sms_enabled': False,
                        'sound_enabled': False
                    }
                }
            
            # Update intervals and thresholds
            intervals = self.monitor_config.get('intervals', {})
            self.system_check_interval = intervals.get('system_check', 10)
            self.component_check_interval = intervals.get('component_check', 30)
            self.performance_check_interval = intervals.get('performance_check', 60)
            
            thresholds = self.monitor_config.get('thresholds', {})
            self.cpu_warning_threshold = thresholds.get('cpu_warning', 80.0)
            self.cpu_critical_threshold = thresholds.get('cpu_critical', 95.0)
            self.memory_warning_threshold = thresholds.get('memory_warning', 85.0)
            self.memory_critical_threshold = thresholds.get('memory_critical', 95.0)
            self.disk_warning_threshold = thresholds.get('disk_warning', 90.0)
            self.disk_critical_threshold = thresholds.get('disk_critical', 98.0)
            
            self.logger.info("System monitor configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load system monitor config: {e}")
            return False
    
    def _initialize_system_info(self):
        """Initialize system information"""
        try:
            system_info = {
                'platform': platform.platform(),
                'processor': platform.processor(),
                'architecture': platform.architecture(),
                'python_version': platform.python_version(),
                'cpu_count': psutil.cpu_count(),
                'memory_total': psutil.virtual_memory().total,
                'disk_total': psutil.disk_usage('/').total if os.name != 'nt' else psutil.disk_usage('C:').total,
                'hostname': socket.gethostname(),
                'ip_address': socket.gethostbyname(socket.gethostname())
            }
            
            self.logger.info(f"System Info: {system_info['platform']}")
            self.logger.info(f"CPU Cores: {system_info['cpu_count']}")
            self.logger.info(f"Total Memory: {system_info['memory_total'] / (1024**3):.1f} GB")
            
        except Exception as e:
            self.logger.error(f"System info initialization error: {e}")
    
    def _register_core_components(self):
        """Register core EA components for monitoring"""
        core_components = [
            'main_bridge',
            'risk_manager',
            'fyers_bridge',
            'truedata_bridge',
            'ml_enhancement',
            'entry_processor',
            'market_scanner',
            'fno_scanner',
            'option_analyzer'
        ]
        
        for component in core_components:
            self.register_component(component)
    
    def start_monitoring(self) -> bool:
        """
        Start system monitoring
        Returns: True if started successfully
        """
        try:
            if not self.is_initialized:
                self.logger.error("System monitor not initialized")
                return False
            
            self.is_monitoring = True
            
            # Start monitoring threads
            self.system_thread = threading.Thread(target=self._system_monitor_loop, daemon=True)
            self.system_thread.start()
            
            self.component_thread = threading.Thread(target=self._component_monitor_loop, daemon=True)
            self.component_thread.start()
            
            self.performance_thread = threading.Thread(target=self._performance_monitor_loop, daemon=True)
            self.performance_thread.start()
            
            self.logger.info("✅ System monitoring started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system monitoring: {e}")
            return False
    
    def _system_monitor_loop(self):
        """System resources monitoring loop"""
        while self.is_monitoring:
            try:
                # Collect system metrics
                metrics = self._collect_system_metrics()
                if metrics:
                    self.current_metrics = metrics
                    self._store_metrics(metrics)
                    self._check_system_thresholds(metrics)
                
                time.sleep(self.system_check_interval)
                
            except Exception as e:
                self.logger.error(f"System monitor loop error: {e}")
                if self.error_handler:
                    self.error_handler.handle_error("system_monitor_loop", e)
                time.sleep(self.system_check_interval * 2)
    
    def _component_monitor_loop(self):
        """Component health monitoring loop"""
        while self.is_monitoring:
            try:
                # Check all registered components
                for component_name in self.registered_components:
                    health = self._check_component_health(component_name)
                    if health:
                        self.component_health[component_name] = health
                        self._check_component_alerts(health)
                
                time.sleep(self.component_check_interval)
                
            except Exception as e:
                self.logger.error(f"Component monitor loop error: {e}")
                if self.error_handler:
                    self.error_handler.handle_error("component_monitor_loop", e)
                time.sleep(self.component_check_interval * 2)
    
    def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        while self.is_monitoring:
            try:
                # Update performance metrics
                self._update_performance_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                time.sleep(self.performance_check_interval)
                
            except Exception as e:
                self.logger.error(f"Performance monitor loop error: {e}")
                if self.error_handler:
                    self.error_handler.handle_error("performance_monitor_loop", e)
                time.sleep(self.performance_check_interval * 2)
    
    def _collect_system_metrics(self) -> Optional[SystemMetrics]:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_usage = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_usage = memory.percent
            
            # Disk usage
            disk_path = '/' if os.name != 'nt' else 'C:'
            disk = psutil.disk_usage(disk_path)
            disk_usage = (disk.used / disk.total) * 100
            
            # Network I/O
            network = psutil.net_io_counters()
            network_io = {
                'bytes_sent': network.bytes_sent,
                'bytes_recv': network.bytes_recv,
                'packets_sent': network.packets_sent,
                'packets_recv': network.packets_recv
            }
            
            # Process count
            process_count = len(psutil.pids())
            
            # System uptime
            uptime = time.time() - psutil.boot_time()
            
            # Temperature (if available)
            temperature = None
            try:
                if hasattr(psutil, 'sensors_temperatures'):
                    temps = psutil.sensors_temperatures()
                    if temps:
                        # Get average temperature
                        all_temps = []
                        for sensor_family in temps.values():
                            for sensor in sensor_family:
                                if sensor.current:
                                    all_temps.append(sensor.current)
                        if all_temps:
                            temperature = sum(all_temps) / len(all_temps)
            except:
                pass
            
            return SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=cpu_usage,
                memory_usage=memory_usage,
                disk_usage=disk_usage,
                network_io=network_io,
                process_count=process_count,
                uptime=uptime,
                temperature=temperature
            )
            
        except Exception as e:
            self.logger.error(f"System metrics collection error: {e}")
            return None
    
    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in history"""
        try:
            self.metrics_history.append(metrics)
            
            # Keep only recent history
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]
                
        except Exception as e:
            self.logger.error(f"Metrics storage error: {e}")
    
    def _check_system_thresholds(self, metrics: SystemMetrics):
        """Check system metrics against thresholds"""
        try:
            # CPU threshold checks
            if metrics.cpu_usage >= self.cpu_critical_threshold:
                self._create_alert(AlertLevel.CRITICAL, "system", 
                                f"CPU usage critical: {metrics.cpu_usage:.1f}%")
            elif metrics.cpu_usage >= self.cpu_warning_threshold:
                self._create_alert(AlertLevel.WARNING, "system", 
                                f"CPU usage high: {metrics.cpu_usage:.1f}%")
            
            # Memory threshold checks
            if metrics.memory_usage >= self.memory_critical_threshold:
                self._create_alert(AlertLevel.CRITICAL, "system", 
                                f"Memory usage critical: {metrics.memory_usage:.1f}%")
            elif metrics.memory_usage >= self.memory_warning_threshold:
                self._create_alert(AlertLevel.WARNING, "system", 
                                f"Memory usage high: {metrics.memory_usage:.1f}%")
            
            # Disk threshold checks
            if metrics.disk_usage >= self.disk_critical_threshold:
                self._create_alert(AlertLevel.CRITICAL, "system", 
                                f"Disk usage critical: {metrics.disk_usage:.1f}%")
            elif metrics.disk_usage >= self.disk_warning_threshold:
                self._create_alert(AlertLevel.WARNING, "system", 
                                f"Disk usage high: {metrics.disk_usage:.1f}%")
            
            # Temperature checks (if available)
            if metrics.temperature and metrics.temperature > 80:
                self._create_alert(AlertLevel.WARNING, "system", 
                                f"System temperature high: {metrics.temperature:.1f}°C")
                
        except Exception as e:
            self.logger.error(f"Threshold check error: {e}")
    
    def _check_component_health(self, component_name: str) -> Optional[ComponentHealth]:
        """Check health of a specific component"""
        try:
            component_info = self.registered_components.get(component_name)
            if not component_info:
                return None
            
            # Get component object
            component = component_info.get('object')
            if not component:
                return ComponentHealth(
                    component_name=component_name,
                    status=HealthStatus.UNKNOWN,
                    last_heartbeat=datetime.now(),
                    error_count=0,
                    performance_score=0.0,
                    details={'error': 'Component object not found'}
                )
            
            # Check if component has health check method
            if hasattr(component, 'is_healthy'):
                try:
                    is_healthy = component.is_healthy()
                    status = HealthStatus.HEALTHY if is_healthy else HealthStatus.WARNING
                except Exception as e:
                    status = HealthStatus.CRITICAL
                    self.logger.error(f"Health check failed for {component_name}: {e}")
            else:
                status = HealthStatus.UNKNOWN
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(component_name, component)
            
            # Get component details
            details = self._get_component_details(component_name, component)
            
            return ComponentHealth(
                component_name=component_name,
                status=status,
                last_heartbeat=datetime.now(),
                error_count=component_info.get('error_count', 0),
                performance_score=performance_score,
                details=details
            )
            
        except Exception as e:
            self.logger.error(f"Component health check error for {component_name}: {e}")
            return ComponentHealth(
                component_name=component_name,
                status=HealthStatus.CRITICAL,
                last_heartbeat=datetime.now(),
                error_count=0,
                performance_score=0.0,
                details={'error': str(e)}
            )
    
    def _calculate_performance_score(self, component_name: str, component) -> float:
        """Calculate performance score for component"""
        try:
            score = 50.0  # Base score
            
            # Check if component is running
            if hasattr(component, 'is_running') and component.is_running:
                score += 20.0
            
            # Check if component is initialized
            if hasattr(component, 'is_initialized') and component.is_initialized:
                score += 15.0
            
            # Check response time (if available)
            if hasattr(component, 'last_response_time'):
                response_time = getattr(component, 'last_response_time', 1.0)
                if response_time < 0.5:
                    score += 15.0
                elif response_time < 1.0:
                    score += 10.0
                elif response_time < 2.0:
                    score += 5.0
            else:
                score += 10.0  # Default if no response time available
            
            return min(score, 100.0)
            
        except Exception as e:
            self.logger.error(f"Performance score calculation error for {component_name}: {e}")
            return 0.0
    
    def _get_component_details(self, component_name: str, component) -> Dict[str, Any]:
        """Get detailed information about component"""
        try:
            details = {
                'type': type(component).__name__,
                'has_health_check': hasattr(component, 'is_healthy'),
                'has_status_method': hasattr(component, 'get_status'),
                'last_checked': datetime.now().isoformat()
            }
            
            # Get status if available
            if hasattr(component, 'get_status'):
                try:
                    status = component.get_status()
                    details['status'] = status
                except Exception as e:
                    details['status_error'] = str(e)
            
            return details
            
        except Exception as e:
            self.logger.error(f"Component details error for {component_name}: {e}")
            return {'error': str(e)}
    
    def _check_component_alerts(self, health: ComponentHealth):
        """Check component health and create alerts if needed"""
        try:
            if health.status == HealthStatus.CRITICAL:
                self._create_alert(AlertLevel.CRITICAL, health.component_name,
                                f"Component {health.component_name} is in critical state")
            elif health.status == HealthStatus.WARNING:
                self._create_alert(AlertLevel.WARNING, health.component_name,
                                f"Component {health.component_name} has warnings")
            elif health.performance_score < 30:
                self._create_alert(AlertLevel.WARNING, health.component_name,
                                f"Component {health.component_name} performance low: {health.performance_score:.1f}")
                
        except Exception as e:
            self.logger.error(f"Component alert check error: {e}")
    
    def _create_alert(self, level: AlertLevel, component: str, message: str):
        """Create system alert"""
        try:
            alert_key = f"{component}_{level.value}_{hash(message) % 10000}"
            
            # Check if similar alert already exists
            if alert_key in self.active_alerts and not self.active_alerts[alert_key].resolved:
                return  # Don't spam alerts
            
            alert = SystemAlert(
                alert_id=alert_key,
                level=level,
                component=component,
                message=message,
                timestamp=datetime.now()
            )
            
            self.active_alerts[alert_key] = alert
            self.alert_history.append(alert)
            
            # Log alert
            log_level = {
                AlertLevel.INFO: logging.INFO,
                AlertLevel.WARNING: logging.WARNING,
                AlertLevel.CRITICAL: logging.CRITICAL,
                AlertLevel.EMERGENCY: logging.CRITICAL
            }.get(level, logging.INFO)
            
            self.logger.log(log_level, f"🚨 ALERT [{level.value}] {component}: {message}")
            
            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback error: {e}")
            
        except Exception as e:
            self.logger.error(f"Alert creation error: {e}")
    
    def _update_performance_metrics(self):
        """Update system performance metrics"""
        try:
            # EA uptime
            self.performance_metrics['ea_uptime'] = (datetime.now() - self.start_time).total_seconds()
            
            # System uptime
            self.performance_metrics['system_uptime'] = time.time() - psutil.boot_time()
            
        except Exception as e:
            self.logger.error(f"Performance metrics update error: {e}")
    
    def _cleanup_old_data(self):
        """Clean up old monitoring data"""
        try:
            # Clean up old alerts (keep last 1000)
            if len(self.alert_history) > 1000:
                self.alert_history = self.alert_history[-1000:]
            
            # Clean up resolved alerts older than 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            self.active_alerts = {
                k: v for k, v in self.active_alerts.items()
                if not v.resolved or v.resolution_time > cutoff_time
            }
            
        except Exception as e:
            self.logger.error(f"Data cleanup error: {e}")
    
    def register_component(self, component_name: str, component_object=None, health_check_func: Callable = None):
        """Register component for monitoring"""
        try:
            self.registered_components[component_name] = {
                'object': component_object,
                'health_check': health_check_func,
                'registered_at': datetime.now(),
                'error_count': 0
            }
            
            self.logger.info(f"Component registered for monitoring: {component_name}")
            
        except Exception as e:
            self.logger.error(f"Component registration error: {e}")
    
    def unregister_component(self, component_name: str):
        """Unregister component from monitoring"""
        try:
            if component_name in self.registered_components:
                del self.registered_components[component_name]
                self.logger.info(f"Component unregistered: {component_name}")
            
        except Exception as e:
            self.logger.error(f"Component unregistration error: {e}")
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for alerts"""
        self.alert_callbacks.append(callback)
    
    def resolve_alert(self, alert_id: str):
        """Resolve an active alert"""
        try:
            if alert_id in self.active_alerts:
                self.active_alerts[alert_id].resolved = True
                self.active_alerts[alert_id].resolution_time = datetime.now()
                self.logger.info(f"Alert resolved: {alert_id}")
            
        except Exception as e:
            self.logger.error(f"Alert resolution error: {e}")
    
    def perform_health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check"""
        try:
            health_report = {
                'timestamp': datetime.now().isoformat(),
                'system_metrics': None,
                'component_health': {},
                'active_alerts': len(self.active_alerts),
                'critical_alerts': 0,
                'overall_health': HealthStatus.HEALTHY.value,
                'performance_metrics': self.performance_metrics.copy()
            }
            
            # Current system metrics
            if self.current_metrics:
                health_report['system_metrics'] = {
                    'cpu_usage': self.current_metrics.cpu_usage,
                    'memory_usage': self.current_metrics.memory_usage,
                    'disk_usage': self.current_metrics.disk_usage,
                    'process_count': self.current_metrics.process_count,
                    'temperature': self.current_metrics.temperature
                }
            
            # Component health
            critical_components = 0
            warning_components = 0
            
            for component_name, health in self.component_health.items():
                health_report['component_health'][component_name] = {
                    'status': health.status.value,
                    'performance_score': health.performance_score,
                    'error_count': health.error_count
                }
                
                if health.status == HealthStatus.CRITICAL:
                    critical_components += 1
                elif health.status == HealthStatus.WARNING:
                    warning_components += 1
            
            # Count critical alerts
            health_report['critical_alerts'] = sum(
                1 for alert in self.active_alerts.values()
                if alert.level == AlertLevel.CRITICAL and not alert.resolved
            )
            
            # Determine overall health
            if critical_components > 0 or health_report['critical_alerts'] > 0:
                health_report['overall_health'] = HealthStatus.CRITICAL.value
            elif warning_components > 0:
                health_report['overall_health'] = HealthStatus.WARNING.value
            
            return health_report
            
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
            return {
                'timestamp': datetime.now().isoformat(),
                'error': str(e),
                'overall_health': HealthStatus.UNKNOWN.value
            }
    
    def get_system_metrics(self, hours: int = 1) -> List[SystemMetrics]:
        """Get system metrics for specified hours"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [m for m in self.metrics_history if m.timestamp > cutoff_time]
            
        except Exception as e:
            self.logger.error(f"Get system metrics error: {e}")
            return []
    
    def get_active_alerts(self) -> List[SystemAlert]:
        """Get all active alerts"""
        return [alert for alert in self.active_alerts.values() if not alert.resolved]
    
    def get_component_status(self, component_name: str) -> Optional[ComponentHealth]:
        """Get status of specific component"""
        return self.component_health.get(component_name)
    
    def update_performance_metric(self, metric_name: str, value: Any):
        """Update performance metric"""
        try:
            self.performance_metrics[metric_name] = value
        except Exception as e:
            self.logger.error(f"Performance metric update error: {e}")
    
    def is_healthy(self) -> bool:
        """Check if system monitor is healthy"""
        try:
            return (
                self.is_initialized and
                self.is_monitoring and
                self.system_thread and self.system_thread.is_alive() and
                self.component_thread and self.component_thread.is_alive()
            )
        except:
            return False
    
    def stop(self):
        """Stop system monitoring"""
        try:
            self.is_monitoring = False
            self.logger.info("System Monitor stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping system monitor: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create system monitor
    monitor = SystemMonitor()
    
    # Initialize
    if monitor.initialize():
        print("✅ System Monitor initialized successfully")
        
        # Start monitoring
        if monitor.start_monitoring():
            print("✅ System monitoring started")
            
            # Test for a few seconds
            time.sleep(5)
            
            # Perform health check
            health_report = monitor.perform_health_check()
            print(f"Health report: {json.dumps(health_report, indent=2)}")
            
            # Get active alerts
            alerts = monitor.get_active_alerts()
            print(f"Active alerts: {len(alerts)}")
            
            # Get system metrics
            metrics = monitor.get_system_metrics(hours=1)
            print(f"Metrics collected: {len(metrics)}")
            
            # Stop
            monitor.stop()
        else:
            print("❌ Failed to start system monitoring")
    else:
        print("❌ System Monitor initialization failed")