#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Main Bridge Controller
Core system controller that coordinates all components

Author: EA GlobalFlow Pro Team
Version: v0.1
Date: 2025
"""

import os
import sys
import json
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import traceback

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Core'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Risk'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'API'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'ML'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Trading'))

try:
    # Core imports
    from system_monitor import SystemMonitor
    from error_handler import ErrorHandler
    from config_manager import ConfigManager
    from security_manager import SecurityManager
    
    # Risk management
    from globalflow_risk_v01 import GlobalFlowRisk
    
    # API integrations
    from fyers_bridge import FyersBridge
    from truedata_bridge import TruedataBridge
    from totp_autologin_v01 import TOTPAutoLogin
    
    # ML and Analytics
    from ml_enhancement_v01 import MLEnhancement
    
    # Trading logic
    from entry_conditions_processor import EntryConditionsProcessor
    from market_scanner_v01 import MarketScanner
    
    # F&O specific
    from fno_scanner import FnOScanner
    from option_chain_analyzer import OptionChainAnalyzer
    
except ImportError as e:
    print(f"Critical import error: {e}")
    sys.exit(1)

class MainBridge:
    """
    Main Bridge Controller - Coordinates all EA GlobalFlow Pro v0.1 components
    """
    
    def __init__(self):
        """Initialize main bridge controller"""
        self.version = "v0.1"
        self.start_time = datetime.now()
        self.is_running = False
        self.is_initialized = False
        
        # Component instances
        self.config_manager = None
        self.security_manager = None
        self.error_handler = None
        self.system_monitor = None
        self.risk_manager = None
        self.fyers_bridge = None
        self.truedata_bridge = None
        self.totp_login = None
        self.ml_enhancement = None
        self.entry_processor = None
        self.market_scanner = None
        self.fno_scanner = None
        self.option_analyzer = None
        
        # Threading
        self.main_thread = None
        self.monitor_thread = None
        self.api_thread = None
        self.ml_thread = None
        
        # Status tracking
        self.components_status = {}
        self.last_heartbeat = datetime.now()
        
        # Initialize logging first
        self._setup_logging()
        
    def _setup_logging(self):
        """Setup centralized logging"""
        try:
            log_dir = os.path.join(os.path.dirname(__file__), 'Logs')
            os.makedirs(log_dir, exist_ok=True)
            
            log_file = os.path.join(log_dir, f'main_bridge_{datetime.now().strftime("%Y%m%d")}.log')
            
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler(sys.stdout)
                ]
            )
            
            self.logger = logging.getLogger('MainBridge')
            self.logger.info(f"EA GlobalFlow Pro {self.version} - Main Bridge Starting...")
            
        except Exception as e:
            print(f"Failed to setup logging: {e}")
            sys.exit(1)
    
    def initialize(self) -> bool:
        """
        Initialize all core components
        Returns: True if successful, False otherwise
        """
        try:
            self.logger.info("Starting component initialization...")
            
            # Step 1: Initialize configuration manager
            if not self._init_config_manager():
                return False
                
            # Step 2: Initialize security manager
            if not self._init_security_manager():
                return False
                
            # Step 3: Initialize error handler
            if not self._init_error_handler():
                return False
                
            # Step 4: Initialize system monitor
            if not self._init_system_monitor():
                return False
                
            # Step 5: Initialize risk manager
            if not self._init_risk_manager():
                return False
                
            # Step 6: Initialize API bridges
            if not self._init_api_bridges():
                return False
                
            # Step 7: Initialize ML enhancement
            if not self._init_ml_enhancement():
                return False
                
            # Step 8: Initialize trading components
            if not self._init_trading_components():
                return False
                
            # Step 9: Initialize F&O components
            if not self._init_fno_components():
                return False
                
            self.is_initialized = True
            self.logger.info("All components initialized successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    def _init_config_manager(self) -> bool:
        """Initialize configuration manager"""
        try:
            self.config_manager = ConfigManager()
            if self.config_manager.load_configurations():
                self.components_status['config_manager'] = 'ACTIVE'
                self.logger.info("✅ Configuration Manager initialized")
                return True
            else:
                self.logger.error("❌ Configuration Manager failed to load configs")
                return False
        except Exception as e:
            self.logger.error(f"ConfigManager init error: {e}")
            return False
    
    def _init_security_manager(self) -> bool:
        """Initialize security manager"""
        try:
            self.security_manager = SecurityManager(self.config_manager)
            if self.security_manager.initialize():
                self.components_status['security_manager'] = 'ACTIVE'
                self.logger.info("✅ Security Manager initialized")
                return True
            else:
                self.logger.error("❌ Security Manager initialization failed")
                return False
        except Exception as e:
            self.logger.error(f"SecurityManager init error: {e}")
            return False
    
    def _init_error_handler(self) -> bool:
        """Initialize error handler"""
        try:
            self.error_handler = ErrorHandler(self.config_manager)
            self.components_status['error_handler'] = 'ACTIVE'
            self.logger.info("✅ Error Handler initialized")
            return True
        except Exception as e:
            self.logger.error(f"ErrorHandler init error: {e}")
            return False
    
    def _init_system_monitor(self) -> bool:
        """Initialize system monitor"""
        try:
            self.system_monitor = SystemMonitor(self.config_manager, self.error_handler)
            if self.system_monitor.start_monitoring():
                self.components_status['system_monitor'] = 'ACTIVE'
                self.logger.info("✅ System Monitor initialized")
                return True
            else:
                self.logger.error("❌ System Monitor failed to start")
                return False
        except Exception as e:
            self.logger.error(f"SystemMonitor init error: {e}")
            return False
    
    def _init_risk_manager(self) -> bool:
        """Initialize risk manager"""
        try:
            self.risk_manager = GlobalFlowRisk(self.config_manager, self.error_handler)
            if self.risk_manager.initialize():
                self.components_status['risk_manager'] = 'ACTIVE'
                self.logger.info("✅ Risk Manager initialized")
                return True
            else:
                self.logger.error("❌ Risk Manager initialization failed")
                return False
        except Exception as e:
            self.logger.error(f"RiskManager init error: {e}")
            return False
    
    def _init_api_bridges(self) -> bool:
        """Initialize API bridges"""
        try:
            # Initialize TOTP auto-login first
            self.totp_login = TOTPAutoLogin(self.config_manager, self.security_manager)
            
            # Initialize Fyers bridge
            self.fyers_bridge = FyersBridge(self.config_manager, self.totp_login, self.error_handler)
            if not self.fyers_bridge.initialize():
                self.logger.error("❌ Fyers Bridge initialization failed")
                return False
            
            # Initialize TrueData bridge
            self.truedata_bridge = TruedataBridge(self.config_manager, self.error_handler)
            if not self.truedata_bridge.initialize():
                self.logger.error("❌ TrueData Bridge initialization failed")
                return False
            
            self.components_status['fyers_bridge'] = 'ACTIVE'
            self.components_status['truedata_bridge'] = 'ACTIVE'
            self.components_status['totp_login'] = 'ACTIVE'
            self.logger.info("✅ API Bridges initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"API Bridges init error: {e}")
            return False
    
    def _init_ml_enhancement(self) -> bool:
        """Initialize ML enhancement system"""
        try:
            self.ml_enhancement = MLEnhancement(self.config_manager, self.error_handler)
            if self.ml_enhancement.initialize():
                self.components_status['ml_enhancement'] = 'ACTIVE'
                self.logger.info("✅ ML Enhancement initialized")
                return True
            else:
                self.logger.error("❌ ML Enhancement initialization failed")
                return False
        except Exception as e:
            self.logger.error(f"ML Enhancement init error: {e}")
            return False
    
    def _init_trading_components(self) -> bool:
        """Initialize trading components"""
        try:
            # Initialize entry conditions processor
            self.entry_processor = EntryConditionsProcessor(
                self.config_manager, 
                self.ml_enhancement,
                self.error_handler
            )
            
            # Initialize market scanner
            self.market_scanner = MarketScanner(
                self.config_manager,
                self.fyers_bridge,
                self.truedata_bridge,
                self.error_handler
            )
            
            if self.entry_processor.initialize() and self.market_scanner.initialize():
                self.components_status['entry_processor'] = 'ACTIVE'
                self.components_status['market_scanner'] = 'ACTIVE'
                self.logger.info("✅ Trading Components initialized")
                return True
            else:
                self.logger.error("❌ Trading Components initialization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Trading Components init error: {e}")
            return False
    
    def _init_fno_components(self) -> bool:
        """Initialize F&O specific components"""
        try:
            # Initialize F&O scanner
            self.fno_scanner = FnOScanner(
                self.config_manager,
                self.truedata_bridge,
                self.error_handler
            )
            
            # Initialize option chain analyzer
            self.option_analyzer = OptionChainAnalyzer(
                self.config_manager,
                self.fyers_bridge,
                self.truedata_bridge,
                self.error_handler
            )
            
            if self.fno_scanner.initialize() and self.option_analyzer.initialize():
                self.components_status['fno_scanner'] = 'ACTIVE'
                self.components_status['option_analyzer'] = 'ACTIVE'
                self.logger.info("✅ F&O Components initialized")
                return True
            else:
                self.logger.error("❌ F&O Components initialization failed")
                return False
                
        except Exception as e:
            self.logger.error(f"F&O Components init error: {e}")
            return False
    
    def start(self) -> bool:
        """
        Start the main bridge system
        Returns: True if started successfully
        """
        try:
            if not self.is_initialized:
                self.logger.error("Cannot start - system not initialized")
                return False
            
            self.is_running = True
            
            # Start main processing thread
            self.main_thread = threading.Thread(target=self._main_loop, daemon=True)
            self.main_thread.start()
            
            # Start monitoring thread
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            
            # Start API processing thread
            self.api_thread = threading.Thread(target=self._api_loop, daemon=True)
            self.api_thread.start()
            
            # Start ML processing thread
            self.ml_thread = threading.Thread(target=self._ml_loop, daemon=True)
            self.ml_thread.start()
            
            self.logger.info(f"🚀 EA GlobalFlow Pro {self.version} started successfully!")
            self.logger.info(f"📊 All {len(self.components_status)} components active")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start system: {e}")
            return False
    
    def _main_loop(self):
        """Main processing loop"""
        while self.is_running:
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Process market scanning
                if self.market_scanner and self.market_scanner.should_scan():
                    self.market_scanner.perform_scan()
                
                # Process F&O scanning
                if self.fno_scanner and self.fno_scanner.should_scan():
                    self.fno_scanner.perform_scan()
                
                # Sleep for main loop interval
                time.sleep(1)  # 1 second interval
                
            except Exception as e:
                self.error_handler.handle_error("main_loop", e)
                time.sleep(5)  # Longer sleep on error
    
    def _monitor_loop(self):
        """System monitoring loop"""
        while self.is_running:
            try:
                if self.system_monitor:
                    self.system_monitor.perform_health_check()
                
                # Monitor component status
                self._check_component_health()
                
                time.sleep(30)  # 30 second monitoring interval
                
            except Exception as e:
                self.error_handler.handle_error("monitor_loop", e)
                time.sleep(60)
    
    def _api_loop(self):
        """API processing loop"""
        while self.is_running:
            try:
                # Maintain API connections
                if self.fyers_bridge:
                    self.fyers_bridge.maintain_connection()
                
                if self.truedata_bridge:
                    self.truedata_bridge.maintain_connection()
                
                time.sleep(10)  # 10 second API maintenance
                
            except Exception as e:
                self.error_handler.handle_error("api_loop", e)
                time.sleep(30)
    
    def _ml_loop(self):
        """ML processing loop"""
        while self.is_running:
            try:
                if self.ml_enhancement:
                    self.ml_enhancement.process_signals()
                
                time.sleep(5)  # 5 second ML processing
                
            except Exception as e:
                self.error_handler.handle_error("ml_loop", e)
                time.sleep(15)
    
    def _check_component_health(self):
        """Check health of all components"""
        try:
            for component_name, status in self.components_status.items():
                component = getattr(self, component_name, None)
                if component and hasattr(component, 'is_healthy'):
                    if not component.is_healthy():
                        self.logger.warning(f"⚠️ Component {component_name} is unhealthy")
                        self.components_status[component_name] = 'WARNING'
                    else:
                        self.components_status[component_name] = 'ACTIVE'
                        
        except Exception as e:
            self.logger.error(f"Health check error: {e}")
    
    def stop(self):
        """Stop the main bridge system"""
        try:
            self.logger.info("Stopping EA GlobalFlow Pro...")
            self.is_running = False
            
            # Stop all components
            components = [
                self.system_monitor, self.risk_manager, self.fyers_bridge,
                self.truedata_bridge, self.ml_enhancement, self.market_scanner,
                self.fno_scanner, self.option_analyzer
            ]
            
            for component in components:
                if component and hasattr(component, 'stop'):
                    try:
                        component.stop()
                    except Exception as e:
                        self.logger.error(f"Error stopping component: {e}")
            
            self.logger.info("✅ EA GlobalFlow Pro stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        uptime = datetime.now() - self.start_time
        
        return {
            'version': self.version,
            'is_running': self.is_running,
            'is_initialized': self.is_initialized,
            'uptime': str(uptime),
            'last_heartbeat': self.last_heartbeat.isoformat(),
            'components': self.components_status,
            'component_count': len(self.components_status)
        }

def main():
    """Main entry point"""
    bridge = None
    try:
        print("🚀 Starting EA GlobalFlow Pro v0.1...")
        
        # Create main bridge
        bridge = MainBridge()
        
        # Initialize system
        if not bridge.initialize():
            print("❌ Failed to initialize system")
            return 1
        
        # Start system
        if not bridge.start():
            print("❌ Failed to start system")
            return 1
        
        print("✅ System started successfully!")
        print("Press Ctrl+C to stop...")
        
        # Keep running until interrupted
        try:
            while bridge.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️ Shutdown requested...")
        
        return 0
        
    except Exception as e:
        print(f"❌ Critical error: {e}")
        return 1
        
    finally:
        if bridge:
            bridge.stop()

if __name__ == "__main__":
    sys.exit(main())