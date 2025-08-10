#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - TOTP Auto-Login System
Automated TOTP-based authentication for broker APIs

Author: EA GlobalFlow Pro Team
Version: v0.1
Date: 2025
"""

import os
import sys
import json
import time
import logging
import pyotp
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
import urllib.parse
import re

# Try to import browser automation libraries
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.chrome.options import Options as ChromeOptions
    from selenium.webdriver.firefox.options import Options as FirefoxOptions
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("Warning: Selenium not available. Install with: pip install selenium")

class BrokerType(Enum):
    FYERS = "FYERS"
    ZERODHA = "ZERODHA"
    UPSTOX = "UPSTOX"
    ANGEL = "ANGEL"

class AuthStatus(Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    EXPIRED = "EXPIRED"

@dataclass
class AuthCredentials:
    broker: BrokerType
    username: str
    password: str
    totp_secret: str
    pin: str = ""
    client_id: str = ""
    app_id: str = ""

@dataclass
class AuthResult:
    status: AuthStatus
    auth_code: str
    access_token: str
    message: str
    expires_at: Optional[datetime] = None

class TOTPAutoLogin:
    """
    TOTP Auto-Login System for EA GlobalFlow Pro v0.1
    Handles automated authentication with TOTP for various brokers
    """
    
    def __init__(self, config_manager=None, security_manager=None):
        """Initialize TOTP auto-login system"""
        self.config_manager = config_manager
        self.security_manager = security_manager
        self.logger = logging.getLogger('TOTPAutoLogin')
        
        # Configuration
        self.totp_config = {}
        self.credentials = {}
        self.is_initialized = False
        
        # Browser automation
        self.driver = None
        self.browser_type = "chrome"  # chrome, firefox, edge
        self.headless = True
        self.page_timeout = 30
        
        # Authentication cache
        self.auth_cache = {}
        self.token_cache = {}
        
        # Rate limiting
        self.auth_attempts = {}
        self.max_attempts_per_hour = 5
        
    def initialize(self) -> bool:
        """
        Initialize TOTP auto-login system
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing TOTP Auto-Login System v0.1...")
            
            # Check dependencies
            if not SELENIUM_AVAILABLE:
                self.logger.error("Selenium not available - required for browser automation")
                return False
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Load credentials
            if not self._load_credentials():
                return False
            
            # Initialize browser driver
            if not self._init_browser_driver():
                return False
            
            self.is_initialized = True
            self.logger.info("✅ TOTP Auto-Login initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"TOTP Auto-Login initialization failed: {e}")
            return False
    
    def _load_config(self) -> bool:
        """Load TOTP configuration"""
        try:
            if self.config_manager:
                self.totp_config = self.config_manager.get_config('security', {}).get('totp', {})
            else:
                # Default configuration
                self.totp_config = {
                    'enabled': True,
                    'browser_type': 'chrome',
                    'headless': True,
                    'page_timeout': 30,
                    'max_attempts_per_hour': 5
                }
            
            self.browser_type = self.totp_config.get('browser_type', 'chrome')
            self.headless = self.totp_config.get('headless', True)
            self.page_timeout = self.totp_config.get('page_timeout', 30)
            self.max_attempts_per_hour = self.totp_config.get('max_attempts_per_hour', 5)
            
            self.logger.info("TOTP configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load TOTP config: {e}")
            return False
    
    def _load_credentials(self) -> bool:
        """Load authentication credentials"""
        try:
            if self.config_manager:
                api_credentials = self.config_manager.get_config('api_credentials', {})
            else:
                # Load from file
                config_file = os.path.join(os.path.dirname(__file__), 'Config', 'api_credentials_v01.json')
                if os.path.exists(config_file):
                    with open(config_file, 'r') as f:
                        api_credentials = json.load(f)
                else:
                    self.logger.error("API credentials file not found")
                    return False
            
            # Process credentials for each broker
            for broker_name, creds in api_credentials.items():
                if broker_name.upper() in [b.value for b in BrokerType]:
                    broker_type = BrokerType(broker_name.upper())
                    
                    self.credentials[broker_type] = AuthCredentials(
                        broker=broker_type,
                        username=creds.get('user_id', creds.get('username', '')),
                        password=creds.get('password', ''),
                        totp_secret=creds.get('totp_secret', ''),
                        pin=creds.get('pin', ''),
                        client_id=creds.get('client_id', ''),
                        app_id=creds.get('app_id', '')
                    )
            
            self.logger.info(f"Loaded credentials for {len(self.credentials)} brokers")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
            return False
    
    def _init_browser_driver(self) -> bool:
        """Initialize browser driver"""
        try:
            if self.browser_type.lower() == 'chrome':
                options = ChromeOptions()
                if self.headless:
                    options.add_argument('--headless')
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                
                self.driver = webdriver.Chrome(options=options)
                
            elif self.browser_type.lower() == 'firefox':
                options = FirefoxOptions()
                if self.headless:
                    options.add_argument('--headless')
                
                self.driver = webdriver.Firefox(options=options)
                
            else:
                self.logger.error(f"Unsupported browser type: {self.browser_type}")
                return False
            
            self.driver.set_page_load_timeout(self.page_timeout)
            self.logger.info(f"Browser driver initialized: {self.browser_type}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize browser driver: {e}")
            return False
    
    def generate_totp(self, broker: BrokerType) -> Optional[str]:
        """
        Generate TOTP code for broker
        
        Args:
            broker: Broker type
            
        Returns:
            TOTP code or None if failed
        """
        try:
            if broker not in self.credentials:
                self.logger.error(f"No credentials found for {broker.value}")
                return None
            
            totp_secret = self.credentials[broker].totp_secret
            if not totp_secret:
                self.logger.error(f"No TOTP secret found for {broker.value}")
                return None
            
            # Generate TOTP code
            totp = pyotp.TOTP(totp_secret)
            code = totp.now()
            
            self.logger.debug(f"Generated TOTP code for {broker.value}: {code}")
            return code
            
        except Exception as e:
            self.logger.error(f"TOTP generation error for {broker.value}: {e}")
            return None
    
    def get_fyers_auth_code(self, auth_url: str) -> Optional[str]:
        """
        Get Fyers authorization code using automated login
        
        Args:
            auth_url: Fyers authorization URL
            
        Returns:
            Authorization code or None if failed
        """
        try:
            if not self._check_rate_limit(BrokerType.FYERS):
                return None
            
            self.logger.info("Starting Fyers automated login...")
            
            # Get credentials
            if BrokerType.FYERS not in self.credentials:
                self.logger.error("Fyers credentials not found")
                return None
            
            creds = self.credentials[BrokerType.FYERS]
            
            # Navigate to auth URL
            self.driver.get(auth_url)
            
            # Wait for login page
            wait = WebDriverWait(self.driver, 10)
            
            # Enter username
            username_field = wait.until(EC.presence_of_element_located((By.NAME, "fy_id")))
            username_field.clear()
            username_field.send_keys(creds.username)
            
            # Enter password
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.clear()
            password_field.send_keys(creds.password)
            
            # Click login
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for TOTP page
            time.sleep(3)
            
            # Check if TOTP is required
            try:
                totp_field = self.driver.find_element(By.NAME, "totp")
                
                # Generate and enter TOTP
                totp_code = self.generate_totp(BrokerType.FYERS)
                if not totp_code:
                    raise Exception("Failed to generate TOTP")
                
                totp_field.clear()
                totp_field.send_keys(totp_code)
                
                # Submit TOTP
                totp_submit = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                totp_submit.click()
                
            except:
                self.logger.info("TOTP not required or already handled")
            
            # Wait for redirect and extract auth code
            time.sleep(5)
            
            # Get current URL and extract auth code
            current_url = self.driver.current_url
            parsed_url = urllib.parse.urlparse(current_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            auth_code = query_params.get('auth_code', [None])[0]
            if auth_code:
                self.logger.info("✅ Fyers auth code obtained successfully")
                self._record_auth_attempt(BrokerType.FYERS, True)
                return auth_code
            else:
                self.logger.error("❌ Failed to extract Fyers auth code")
                self._record_auth_attempt(BrokerType.FYERS, False)
                return None
                
        except Exception as e:
            self.logger.error(f"Fyers auth code error: {e}")
            self._record_auth_attempt(BrokerType.FYERS, False)
            return None
    
    def get_zerodha_auth_code(self, auth_url: str) -> Optional[str]:
        """
        Get Zerodha authorization code using automated login
        
        Args:
            auth_url: Zerodha authorization URL
            
        Returns:
            Authorization code or None if failed
        """
        try:
            if not self._check_rate_limit(BrokerType.ZERODHA):
                return None
            
            self.logger.info("Starting Zerodha automated login...")
            
            # Get credentials
            if BrokerType.ZERODHA not in self.credentials:
                self.logger.error("Zerodha credentials not found")
                return None
            
            creds = self.credentials[BrokerType.ZERODHA]
            
            # Navigate to auth URL
            self.driver.get(auth_url)
            
            # Wait for login page
            wait = WebDriverWait(self.driver, 10)
            
            # Enter username
            username_field = wait.until(EC.presence_of_element_located((By.ID, "userid")))
            username_field.clear()
            username_field.send_keys(creds.username)
            
            # Enter password
            password_field = self.driver.find_element(By.ID, "password")
            password_field.clear()
            password_field.send_keys(creds.password)
            
            # Click login
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for PIN page
            time.sleep(3)
            
            # Enter PIN
            try:
                pin_field = wait.until(EC.presence_of_element_located((By.ID, "pin")))
                pin_field.clear()
                pin_field.send_keys(creds.pin)
                
                # Submit PIN
                pin_submit = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                pin_submit.click()
                
            except:
                self.logger.info("PIN not required or already handled")
            
            # Check for TOTP
            time.sleep(3)
            try:
                totp_field = self.driver.find_element(By.ID, "totp")
                
                # Generate and enter TOTP
                totp_code = self.generate_totp(BrokerType.ZERODHA)
                if not totp_code:
                    raise Exception("Failed to generate TOTP")
                
                totp_field.clear()
                totp_field.send_keys(totp_code)
                
                # Submit TOTP
                totp_submit = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                totp_submit.click()
                
            except:
                self.logger.info("TOTP not required or already handled")
            
            # Wait for redirect and extract request token
            time.sleep(5)
            
            current_url = self.driver.current_url
            parsed_url = urllib.parse.urlparse(current_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            request_token = query_params.get('request_token', [None])[0]
            if request_token:
                self.logger.info("✅ Zerodha request token obtained successfully")
                self._record_auth_attempt(BrokerType.ZERODHA, True)
                return request_token
            else:
                self.logger.error("❌ Failed to extract Zerodha request token")
                self._record_auth_attempt(BrokerType.ZERODHA, False)
                return None
                
        except Exception as e:
            self.logger.error(f"Zerodha auth code error: {e}")
            self._record_auth_attempt(BrokerType.ZERODHA, False)
            return None
    
    def get_upstox_auth_code(self, auth_url: str) -> Optional[str]:
        """
        Get Upstox authorization code using automated login
        
        Args:
            auth_url: Upstox authorization URL
            
        Returns:
            Authorization code or None if failed
        """
        try:
            if not self._check_rate_limit(BrokerType.UPSTOX):
                return None
            
            self.logger.info("Starting Upstox automated login...")
            
            # Get credentials
            if BrokerType.UPSTOX not in self.credentials:
                self.logger.error("Upstox credentials not found")
                return None
            
            creds = self.credentials[BrokerType.UPSTOX]
            
            # Navigate to auth URL
            self.driver.get(auth_url)
            
            # Wait for login page
            wait = WebDriverWait(self.driver, 10)
            
            # Enter mobile number/username
            username_field = wait.until(EC.presence_of_element_located((By.NAME, "username")))
            username_field.clear()
            username_field.send_keys(creds.username)
            
            # Enter password
            password_field = self.driver.find_element(By.NAME, "password")
            password_field.clear()
            password_field.send_keys(creds.password)
            
            # Click login
            login_button = self.driver.find_element(By.XPATH, "//button[@type='submit']")
            login_button.click()
            
            # Wait for PIN/TOTP page
            time.sleep(3)
            
            # Handle 2FA
            try:
                # Check for PIN field
                pin_field = self.driver.find_element(By.NAME, "pin")
                pin_field.clear()
                pin_field.send_keys(creds.pin)
                
                # Submit PIN
                pin_submit = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                pin_submit.click()
                
            except:
                # Check for TOTP field
                try:
                    totp_field = self.driver.find_element(By.NAME, "totp")
                    
                    # Generate and enter TOTP
                    totp_code = self.generate_totp(BrokerType.UPSTOX)
                    if not totp_code:
                        raise Exception("Failed to generate TOTP")
                    
                    totp_field.clear()
                    totp_field.send_keys(totp_code)
                    
                    # Submit TOTP
                    totp_submit = self.driver.find_element(By.XPATH, "//button[@type='submit']")
                    totp_submit.click()
                    
                except:
                    self.logger.info("2FA not required or already handled")
            
            # Wait for redirect and extract auth code
            time.sleep(5)
            
            current_url = self.driver.current_url
            parsed_url = urllib.parse.urlparse(current_url)
            query_params = urllib.parse.parse_qs(parsed_url.query)
            
            auth_code = query_params.get('code', [None])[0]
            if auth_code:
                self.logger.info("✅ Upstox auth code obtained successfully")
                self._record_auth_attempt(BrokerType.UPSTOX, True)
                return auth_code
            else:
                self.logger.error("❌ Failed to extract Upstox auth code")
                self._record_auth_attempt(BrokerType.UPSTOX, False)
                return None
                
        except Exception as e:
            self.logger.error(f"Upstox auth code error: {e}")
            self._record_auth_attempt(BrokerType.UPSTOX, False)
            return None
    
    def authenticate_broker(self, broker: BrokerType, auth_url: str) -> AuthResult:
        """
        Authenticate with broker using automated login
        
        Args:
            broker: Broker type
            auth_url: Authorization URL
            
        Returns:
            AuthResult object
        """
        try:
            self.logger.info(f"Authenticating with {broker.value}...")
            
            # Check cache first
            if broker in self.auth_cache:
                cached_result = self.auth_cache[broker]
                if cached_result.expires_at and cached_result.expires_at > datetime.now():
                    self.logger.info(f"Using cached auth for {broker.value}")
                    return cached_result
            
            # Perform authentication based on broker type
            auth_code = None
            
            if broker == BrokerType.FYERS:
                auth_code = self.get_fyers_auth_code(auth_url)
            elif broker == BrokerType.ZERODHA:
                auth_code = self.get_zerodha_auth_code(auth_url)
            elif broker == BrokerType.UPSTOX:
                auth_code = self.get_upstox_auth_code(auth_url)
            else:
                return AuthResult(
                    status=AuthStatus.FAILED,
                    auth_code="",
                    access_token="",
                    message=f"Unsupported broker: {broker.value}"
                )
            
            if auth_code:
                result = AuthResult(
                    status=AuthStatus.SUCCESS,
                    auth_code=auth_code,
                    access_token="",  # Will be set by broker-specific code
                    message="Authentication successful",
                    expires_at=datetime.now() + timedelta(hours=6)  # 6-hour cache
                )
                
                # Cache the result
                self.auth_cache[broker] = result
                
                return result
            else:
                return AuthResult(
                    status=AuthStatus.FAILED,
                    auth_code="",
                    access_token="",
                    message="Failed to obtain auth code"
                )
                
        except Exception as e:
            self.logger.error(f"Broker authentication error: {e}")
            return AuthResult(
                status=AuthStatus.FAILED,
                auth_code="",
                access_token="",
                message=str(e)
            )
    
    def _check_rate_limit(self, broker: BrokerType) -> bool:
        """Check if rate limit allows new authentication attempt"""
        try:
            current_time = datetime.now()
            hour_ago = current_time - timedelta(hours=1)
            
            if broker not in self.auth_attempts:
                self.auth_attempts[broker] = []
            
            # Remove old attempts
            self.auth_attempts[broker] = [
                attempt for attempt in self.auth_attempts[broker]
                if attempt['timestamp'] > hour_ago
            ]
            
            # Check if under limit
            if len(self.auth_attempts[broker]) >= self.max_attempts_per_hour:
                self.logger.warning(f"Rate limit exceeded for {broker.value}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Rate limit check error: {e}")
            return True  # Allow on error
    
    def _record_auth_attempt(self, broker: BrokerType, success: bool):
        """Record authentication attempt"""
        try:
            if broker not in self.auth_attempts:
                self.auth_attempts[broker] = []
            
            self.auth_attempts[broker].append({
                'timestamp': datetime.now(),
                'success': success
            })
            
        except Exception as e:
            self.logger.error(f"Error recording auth attempt: {e}")
    
    def validate_totp_secret(self, broker: BrokerType, test_code: str = None) -> bool:
        """
        Validate TOTP secret by comparing generated code with expected
        
        Args:
            broker: Broker type
            test_code: Optional test code to validate against
            
        Returns:
            True if TOTP secret is valid
        """
        try:
            if broker not in self.credentials:
                return False
            
            totp_secret = self.credentials[broker].totp_secret
            if not totp_secret:
                return False
            
            # Generate current TOTP code
            totp = pyotp.TOTP(totp_secret)
            current_code = totp.now()
            
            if test_code:
                # Validate against provided test code
                return current_code == test_code
            else:
                # Just check if we can generate a code
                return len(current_code) == 6 and current_code.isdigit()
                
        except Exception as e:
            self.logger.error(f"TOTP validation error: {e}")
            return False
    
    def get_backup_codes(self, broker: BrokerType) -> List[str]:
        """
        Generate backup TOTP codes (current + next few periods)
        
        Args:
            broker: Broker type
            
        Returns:
            List of backup codes
        """
        try:
            if broker not in self.credentials:
                return []
            
            totp_secret = self.credentials[broker].totp_secret
            if not totp_secret:
                return []
            
            totp = pyotp.TOTP(totp_secret)
            backup_codes = []
            
            # Generate codes for current and next 2 periods (90 seconds coverage)
            for i in range(3):
                timestamp = datetime.now() + timedelta(seconds=i * 30)
                code = totp.at(timestamp)
                backup_codes.append(code)
            
            return backup_codes
            
        except Exception as e:
            self.logger.error(f"Backup codes generation error: {e}")
            return []
    
    def clear_auth_cache(self, broker: BrokerType = None):
        """
        Clear authentication cache
        
        Args:
            broker: Specific broker to clear, or None for all
        """
        try:
            if broker:
                if broker in self.auth_cache:
                    del self.auth_cache[broker]
                    self.logger.info(f"Cleared auth cache for {broker.value}")
            else:
                self.auth_cache.clear()
                self.logger.info("Cleared all auth cache")
                
        except Exception as e:
            self.logger.error(f"Clear auth cache error: {e}")
    
    def get_auth_status(self, broker: BrokerType) -> Dict[str, Any]:
        """
        Get authentication status for broker
        
        Args:
            broker: Broker type
            
        Returns:
            Status dictionary
        """
        try:
            status = {
                'broker': broker.value,
                'has_credentials': broker in self.credentials,
                'has_totp_secret': False,
                'cached_auth': False,
                'cache_expires': None,
                'recent_attempts': 0,
                'rate_limited': False
            }
            
            if broker in self.credentials:
                status['has_totp_secret'] = bool(self.credentials[broker].totp_secret)
            
            if broker in self.auth_cache:
                cached_result = self.auth_cache[broker]
                status['cached_auth'] = True
                status['cache_expires'] = cached_result.expires_at.isoformat() if cached_result.expires_at else None
            
            # Check recent attempts
            if broker in self.auth_attempts:
                hour_ago = datetime.now() - timedelta(hours=1)
                recent_attempts = [
                    attempt for attempt in self.auth_attempts[broker]
                    if attempt['timestamp'] > hour_ago
                ]
                status['recent_attempts'] = len(recent_attempts)
                status['rate_limited'] = len(recent_attempts) >= self.max_attempts_per_hour
            
            return status
            
        except Exception as e:
            self.logger.error(f"Get auth status error: {e}")
            return {}
    
    def test_browser_automation(self) -> bool:
        """
        Test browser automation functionality
        
        Returns:
            True if browser automation is working
        """
        try:
            if not self.driver:
                return False
            
            # Navigate to a simple test page
            test_url = "https://httpbin.org/html"
            self.driver.get(test_url)
            
            # Check if page loaded
            return "Herman Melville" in self.driver.page_source
            
        except Exception as e:
            self.logger.error(f"Browser automation test error: {e}")
            return False
    
    def take_screenshot(self, filename: str = None) -> Optional[str]:
        """
        Take screenshot of current browser state
        
        Args:
            filename: Optional filename for screenshot
            
        Returns:
            Screenshot filename or None if failed
        """
        try:
            if not self.driver:
                return None
            
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.png"
            
            # Ensure screenshots directory exists
            screenshots_dir = os.path.join(os.path.dirname(__file__), 'Screenshots')
            os.makedirs(screenshots_dir, exist_ok=True)
            
            filepath = os.path.join(screenshots_dir, filename)
            
            if self.driver.save_screenshot(filepath):
                self.logger.info(f"Screenshot saved: {filepath}")
                return filepath
            else:
                self.logger.error("Failed to save screenshot")
                return None
                
        except Exception as e:
            self.logger.error(f"Screenshot error: {e}")
            return None
    
    def is_healthy(self) -> bool:
        """Check if TOTP auto-login system is healthy"""
        try:
            return (
                self.is_initialized and
                self.driver is not None and
                len(self.credentials) > 0
            )
        except:
            return False
    
    def stop(self):
        """Stop TOTP auto-login system"""
        try:
            if self.driver:
                self.driver.quit()
                self.driver = None
            
            self.logger.info("TOTP Auto-Login stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping TOTP auto-login: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create TOTP auto-login
    totp_login = TOTPAutoLogin()
    
    # Initialize
    if totp_login.initialize():
        print("✅ TOTP Auto-Login initialized successfully")
        
        # Test TOTP generation
        for broker in [BrokerType.FYERS, BrokerType.ZERODHA]:
            totp_code = totp_login.generate_totp(broker)
            if totp_code:
                print(f"{broker.value} TOTP: {totp_code}")
            
            # Test TOTP validation
            is_valid = totp_login.validate_totp_secret(broker)
            print(f"{broker.value} TOTP secret valid: {is_valid}")
            
            # Get auth status
            status = totp_login.get_auth_status(broker)
            print(f"{broker.value} status: {status}")
        
        # Test browser automation
        browser_test = totp_login.test_browser_automation()
        print(f"Browser automation test: {'✅ PASS' if browser_test else '❌ FAIL'}")
        
        # Stop
        totp_login.stop()
    else:
        print("❌ TOTP Auto-Login initialization failed")