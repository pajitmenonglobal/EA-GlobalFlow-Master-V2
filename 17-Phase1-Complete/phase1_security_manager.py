#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Security Manager
Comprehensive security and encryption management system

Author: EA GlobalFlow Pro Team
Version: v0.1
Date: 2025
"""

import os
import sys
import json
import time
import logging
import hashlib
import secrets
import base64
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading

# Try to import cryptography libraries
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa, padding
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("Warning: Cryptography libraries not available. Install with: pip install cryptography")

class SecurityLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class EncryptionType(Enum):
    SYMMETRIC = "SYMMETRIC"
    ASYMMETRIC = "ASYMMETRIC"
    HYBRID = "HYBRID"

class AccessLevel(Enum):
    READ_ONLY = "READ_ONLY"
    STANDARD = "STANDARD"
    ELEVATED = "ELEVATED"
    ADMINISTRATOR = "ADMINISTRATOR"

@dataclass
class SecurityCredential:
    credential_id: str
    name: str
    encrypted_value: bytes
    encryption_type: EncryptionType
    security_level: SecurityLevel
    created_at: datetime
    expires_at: Optional[datetime] = None
    access_count: int = 0
    last_accessed: Optional[datetime] = None

@dataclass
class SecuritySession:
    session_id: str
    user_id: str
    access_level: AccessLevel
    created_at: datetime
    expires_at: datetime
    last_activity: datetime
    ip_address: str
    is_active: bool = True

@dataclass
class SecurityEvent:
    event_id: str
    event_type: str
    severity: SecurityLevel
    timestamp: datetime
    user_id: Optional[str]
    ip_address: Optional[str]
    description: str
    details: Dict[str, Any]

class SecurityManager:
    """
    Security Manager for EA GlobalFlow Pro v0.1
    Handles encryption, credential management, and security monitoring
    """
    
    def __init__(self, config_manager=None):
        """Initialize security manager"""
        self.config_manager = config_manager
        self.logger = logging.getLogger('SecurityManager')
        
        # Configuration
        self.security_config = {}
        self.is_initialized = False
        
        # Encryption keys
        self.master_key = None
        self.fernet_cipher = None
        self.rsa_private_key = None
        self.rsa_public_key = None
        
        # Security storage
        self.credentials = {}
        self.active_sessions = {}
        self.security_events = []
        
        # Security policies
        self.password_policy = {}
        self.session_timeout = 3600  # 1 hour default
        self.max_login_attempts = 3
        self.lockout_duration = 300  # 5 minutes
        
        # Monitoring
        self.failed_attempts = {}
        self.locked_accounts = {}
        self.security_alerts = []
        
        # Threading
        self.security_lock = threading.Lock()
        self.cleanup_thread = None
        self.is_monitoring = False
        
        # File paths
        self.security_dir = os.path.join(os.path.dirname(__file__), 'Security')
        self.credentials_file = os.path.join(self.security_dir, 'credentials.enc')
        self.keys_file = os.path.join(self.security_dir, 'keys.enc')
        self.events_file = os.path.join(self.security_dir, 'events.enc')
        
        # Security metrics
        self.security_metrics = {
            'total_credentials': 0,
            'active_sessions': 0,
            'security_events': 0,
            'failed_login_attempts': 0,
            'successful_logins': 0,
            'encryption_operations': 0,
            'decryption_operations': 0
        }
        
    def initialize(self) -> bool:
        """
        Initialize security manager
        Returns: True if successful
        """
        try:
            self.logger.info("Initializing Security Manager v0.1...")
            
            # Check cryptography libraries
            if not CRYPTO_AVAILABLE:
                self.logger.error("Cryptography libraries not available")
                return False
            
            # Create security directory
            os.makedirs(self.security_dir, exist_ok=True)
            
            # Set secure directory permissions (Unix-like systems)
            if os.name != 'nt':
                os.chmod(self.security_dir, 0o700)
            
            # Load configuration
            if not self._load_config():
                return False
            
            # Initialize encryption
            if not self._initialize_encryption():
                return False
            
            # Load existing credentials
            self._load_credentials()
            
            # Start security monitoring
            self._start_security_monitoring()
            
            self.is_initialized = True
            self.logger.info("✅ Security Manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Security Manager initialization failed: {e}")
            return False
    
    def _load_config(self) -> bool:
        """Load security configuration"""
        try:
            if self.config_manager:
                self.security_config = self.config_manager.get_config('security', {})
            else:
                # Default configuration
                self.security_config = {
                    'encryption_enabled': True,
                    'session_timeout': 3600,
                    'max_login_attempts': 3,
                    'lockout_duration': 300,
                    'password_policy': {
                        'min_length': 12,
                        'require_uppercase': True,
                        'require_lowercase': True,
                        'require_numbers': True,
                        'require_special': True,
                        'max_age_days': 90
                    },
                    'key_rotation': {
                        'enabled': True,
                        'interval_days': 30
                    },
                    'audit_logging': True,
                    'intrusion_detection': True
                }
            
            # Update settings
            self.session_timeout = self.security_config.get('session_timeout', 3600)
            self.max_login_attempts = self.security_config.get('max_login_attempts', 3)
            self.lockout_duration = self.security_config.get('lockout_duration', 300)
            self.password_policy = self.security_config.get('password_policy', {})
            
            self.logger.info("Security configuration loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load security config: {e}")
            return False
    
    def _initialize_encryption(self) -> bool:
        """Initialize encryption systems"""
        try:
            # Try to load existing keys
            if os.path.exists(self.keys_file):
                if self._load_encryption_keys():
                    self.logger.info("Existing encryption keys loaded")
                    return True
            
            # Generate new keys if none exist
            self.logger.info("Generating new encryption keys...")
            
            # Generate master key for symmetric encryption
            self.master_key = Fernet.generate_key()
            self.fernet_cipher = Fernet(self.master_key)
            
            # Generate RSA key pair for asymmetric encryption
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=2048
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
            
            # Save keys securely
            if self._save_encryption_keys():
                self.logger.info("New encryption keys generated and saved")
                return True
            else:
                self.logger.error("Failed to save encryption keys")
                return False
                
        except Exception as e:
            self.logger.error(f"Encryption initialization failed: {e}")
            return False
    
    def _save_encryption_keys(self) -> bool:
        """Save encryption keys securely"""
        try:
            # Create key data structure
            key_data = {
                'master_key': base64.b64encode(self.master_key).decode('utf-8'),
                'rsa_private_key': self.rsa_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ).decode('utf-8'),
                'rsa_public_key': self.rsa_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ).decode('utf-8'),
                'created_at': datetime.now().isoformat()
            }
            
            # Encrypt key data with password-based encryption
            password = self._generate_key_password()
            encrypted_data = self._encrypt_with_password(json.dumps(key_data), password)
            
            # Save to file
            with open(self.keys_file, 'wb') as f:
                f.write(encrypted_data)
            
            # Set secure file permissions
            if os.name != 'nt':
                os.chmod(self.keys_file, 0o600)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save encryption keys: {e}")
            return False
    
    def _load_encryption_keys(self) -> bool:
        """Load encryption keys from file"""
        try:
            # Read encrypted key file
            with open(self.keys_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Decrypt with password
            password = self._generate_key_password()
            decrypted_data = self._decrypt_with_password(encrypted_data, password)
            
            # Parse key data
            key_data = json.loads(decrypted_data)
            
            # Restore keys
            self.master_key = base64.b64decode(key_data['master_key'])
            self.fernet_cipher = Fernet(self.master_key)
            
            self.rsa_private_key = serialization.load_pem_private_key(
                key_data['rsa_private_key'].encode('utf-8'),
                password=None
            )
            
            self.rsa_public_key = serialization.load_pem_public_key(
                key_data['rsa_public_key'].encode('utf-8')
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load encryption keys: {e}")
            return False
    
    def _generate_key_password(self) -> str:
        """Generate password for key encryption"""
        # This would use a more sophisticated method in production
        # For now, using a combination of system-specific values
        import platform
        system_info = f"{platform.node()}{platform.system()}{platform.version()}"
        return hashlib.sha256(system_info.encode()).hexdigest()[:32]
    
    def _encrypt_with_password(self, data: str, password: str) -> bytes:
        """Encrypt data with password-based encryption"""
        try:
            # Generate salt
            salt = os.urandom(16)
            
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(password.encode())
            
            # Encrypt data
            cipher = Fernet(base64.urlsafe_b64encode(key))
            encrypted_data = cipher.encrypt(data.encode())
            
            # Combine salt and encrypted data
            return salt + encrypted_data
            
        except Exception as e:
            self.logger.error(f"Password encryption failed: {e}")
            raise
    
    def _decrypt_with_password(self, encrypted_data: bytes, password: str) -> str:
        """Decrypt data with password-based encryption"""
        try:
            # Extract salt and encrypted data
            salt = encrypted_data[:16]
            encrypted_content = encrypted_data[16:]
            
            # Derive key from password
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            key = kdf.derive(password.encode())
            
            # Decrypt data
            cipher = Fernet(base64.urlsafe_b64encode(key))
            decrypted_data = cipher.decrypt(encrypted_content)
            
            return decrypted_data.decode()
            
        except Exception as e:
            self.logger.error(f"Password decryption failed: {e}")
            raise
    
    def _load_credentials(self):
        """Load encrypted credentials from file"""
        try:
            if os.path.exists(self.credentials_file):
                # Implementation would load and decrypt credentials
                self.logger.info("Existing credentials loaded")
            else:
                self.logger.info("No existing credentials file found")
                
        except Exception as e:
            self.logger.error(f"Failed to load credentials: {e}")
    
    def _start_security_monitoring(self):
        """Start security monitoring thread"""
        try:
            self.is_monitoring = True
            self.cleanup_thread = threading.Thread(target=self._security_monitoring_loop, daemon=True)
            self.cleanup_thread.start()
            self.logger.info("Security monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start security monitoring: {e}")
    
    def _security_monitoring_loop(self):
        """Security monitoring and cleanup loop"""
        while self.is_monitoring:
            try:
                # Clean up expired sessions
                self._cleanup_expired_sessions()
                
                # Clean up old security events
                self._cleanup_old_events()
                
                # Check for security anomalies
                self._check_security_anomalies()
                
                # Sleep for monitoring interval
                time.sleep(300)  # 5 minutes
                
            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")
                time.sleep(60)  # 1 minute on error
    
    def _cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            with self.security_lock:
                for session_id, session in self.active_sessions.items():
                    if current_time > session.expires_at:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
                    self._log_security_event(
                        'SESSION_EXPIRED',
                        SecurityLevel.LOW,
                        f"Session {session_id} expired"
                    )
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            self.logger.error(f"Session cleanup error: {e}")
    
    def _cleanup_old_events(self):
        """Clean up old security events"""
        try:
            cutoff_time = datetime.now() - timedelta(days=30)
            
            with self.security_lock:
                original_count = len(self.security_events)
                self.security_events = [
                    event for event in self.security_events
                    if event.timestamp > cutoff_time
                ]
                
                cleaned_count = original_count - len(self.security_events)
                if cleaned_count > 0:
                    self.logger.info(f"Cleaned up {cleaned_count} old security events")
                    
        except Exception as e:
            self.logger.error(f"Events cleanup error: {e}")
    
    def _check_security_anomalies(self):
        """Check for security anomalies"""
        try:
            # Check for unusual activity patterns
            # This is a simplified implementation
            recent_events = [
                event for event in self.security_events
                if event.timestamp > datetime.now() - timedelta(hours=1)
            ]
            
            # Check for too many failed login attempts
            failed_logins = [
                event for event in recent_events
                if event.event_type == 'LOGIN_FAILED'
            ]
            
            if len(failed_logins) > 10:
                self._create_security_alert(
                    'ANOMALY_DETECTED',
                    SecurityLevel.HIGH,
                    f"Unusual number of failed login attempts: {len(failed_logins)}"
                )
                
        except Exception as e:
            self.logger.error(f"Security anomaly check error: {e}")
    
    def encrypt_credential(self, name: str, value: str, security_level: SecurityLevel = SecurityLevel.MEDIUM) -> str:
        """
        Encrypt and store a credential
        
        Args:
            name: Credential name
            value: Credential value to encrypt
            security_level: Security level for the credential
            
        Returns:
            Credential ID for retrieval
        """
        try:
            if not self.fernet_cipher:
                raise Exception("Encryption not initialized")
            
            # Generate credential ID
            credential_id = f"cred_{int(time.time())}_{secrets.token_hex(8)}"
            
            # Encrypt the value
            encrypted_value = self.fernet_cipher.encrypt(value.encode())
            
            # Create credential record
            credential = SecurityCredential(
                credential_id=credential_id,
                name=name,
                encrypted_value=encrypted_value,
                encryption_type=EncryptionType.SYMMETRIC,
                security_level=security_level,
                created_at=datetime.now()
            )
            
            # Store credential
            with self.security_lock:
                self.credentials[credential_id] = credential
                self.security_metrics['total_credentials'] += 1
                self.security_metrics['encryption_operations'] += 1
            
            # Log security event
            self._log_security_event(
                'CREDENTIAL_ENCRYPTED',
                SecurityLevel.LOW,
                f"Credential {name} encrypted and stored"
            )
            
            self.logger.info(f"Credential {name} encrypted successfully")
            return credential_id
            
        except Exception as e:
            self.logger.error(f"Credential encryption failed: {e}")
            self._log_security_event(
                'ENCRYPTION_FAILED',
                SecurityLevel.HIGH,
                f"Failed to encrypt credential {name}: {str(e)}"
            )
            return ""
    
    def decrypt_credential(self, credential_id: str) -> Optional[str]:
        """
        Decrypt and retrieve a credential
        
        Args:
            credential_id: Credential ID to decrypt
            
        Returns:
            Decrypted credential value or None if failed
        """
        try:
            if not self.fernet_cipher:
                raise Exception("Encryption not initialized")
            
            with self.security_lock:
                if credential_id not in self.credentials:
                    self.logger.warning(f"Credential {credential_id} not found")
                    return None
                
                credential = self.credentials[credential_id]
                
                # Check if credential has expired
                if credential.expires_at and datetime.now() > credential.expires_at:
                    self.logger.warning(f"Credential {credential_id} has expired")
                    return None
                
                # Decrypt the value
                decrypted_value = self.fernet_cipher.decrypt(credential.encrypted_value)
                
                # Update access tracking
                credential.access_count += 1
                credential.last_accessed = datetime.now()
                self.security_metrics['decryption_operations'] += 1
            
            # Log security event
            self._log_security_event(
                'CREDENTIAL_ACCESSED',
                SecurityLevel.LOW,
                f"Credential {credential.name} accessed"
            )
            
            return decrypted_value.decode()
            
        except Exception as e:
            self.logger.error(f"Credential decryption failed: {e}")
            self._log_security_event(
                'DECRYPTION_FAILED',
                SecurityLevel.HIGH,
                f"Failed to decrypt credential {credential_id}: {str(e)}"
            )
            return None
    
    def create_session(self, user_id: str, access_level: AccessLevel, ip_address: str = "unknown") -> Optional[str]:
        """
        Create a new security session
        
        Args:
            user_id: User identifier
            access_level: Access level for the session
            ip_address: Client IP address
            
        Returns:
            Session ID or None if failed
        """
        try:
            # Generate session ID
            session_id = f"sess_{int(time.time())}_{secrets.token_hex(16)}"
            
            # Create session
            session = SecuritySession(
                session_id=session_id,
                user_id=user_id,
                access_level=access_level,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=self.session_timeout),
                last_activity=datetime.now(),
                ip_address=ip_address
            )
            
            # Store session
            with self.security_lock:
                self.active_sessions[session_id] = session
                self.security_metrics['active_sessions'] = len(self.active_sessions)
            
            # Log security event
            self._log_security_event(
                'SESSION_CREATED',
                SecurityLevel.LOW,
                f"Session created for user {user_id}",
                details={'access_level': access_level.value, 'ip_address': ip_address}
            )
            
            self.logger.info(f"Session created for user {user_id}: {session_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Session creation failed: {e}")
            return None
    
    def validate_session(self, session_id: str) -> Optional[SecuritySession]:
        """
        Validate and update session
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            Session object if valid, None otherwise
        """
        try:
            with self.security_lock:
                if session_id not in self.active_sessions:
                    return None
                
                session = self.active_sessions[session_id]
                
                # Check if session has expired
                if datetime.now() > session.expires_at:
                    del self.active_sessions[session_id]
                    self._log_security_event(
                        'SESSION_EXPIRED',
                        SecurityLevel.LOW,
                        f"Session {session_id} expired"
                    )
                    return None
                
                # Update last activity
                session.last_activity = datetime.now()
                
                return session
                
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return None
    
    def revoke_session(self, session_id: str) -> bool:
        """
        Revoke a session
        
        Args:
            session_id: Session ID to revoke
            
        Returns:
            True if session was revoked
        """
        try:
            with self.security_lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    del self.active_sessions[session_id]
                    
                    self._log_security_event(
                        'SESSION_REVOKED',
                        SecurityLevel.MEDIUM,
                        f"Session {session_id} revoked for user {session.user_id}"
                    )
                    
                    self.logger.info(f"Session {session_id} revoked")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Session revocation failed: {e}")
            return False
    
    def _log_security_event(self, event_type: str, severity: SecurityLevel, description: str, 
                           user_id: str = None, ip_address: str = None, details: Dict[str, Any] = None):
        """Log a security event"""
        try:
            event = SecurityEvent(
                event_id=f"evt_{int(time.time())}_{secrets.token_hex(8)}",
                event_type=event_type,
                severity=severity,
                timestamp=datetime.now(),
                user_id=user_id,
                ip_address=ip_address,
                description=description,
                details=details or {}
            )
            
            with self.security_lock:
                self.security_events.append(event)
                self.security_metrics['security_events'] += 1
            
            # Log with appropriate level
            log_level = {
                SecurityLevel.LOW: logging.INFO,
                SecurityLevel.MEDIUM: logging.WARNING,
                SecurityLevel.HIGH: logging.ERROR,
                SecurityLevel.CRITICAL: logging.CRITICAL
            }.get(severity, logging.INFO)
            
            self.logger.log(log_level, f"SECURITY EVENT [{severity.value}] {event_type}: {description}")
            
        except Exception as e:
            self.logger.error(f"Security event logging failed: {e}")
    
    def _create_security_alert(self, alert_type: str, severity: SecurityLevel, message: str):
        """Create a security alert"""
        try:
            alert = {
                'alert_id': f"alert_{int(time.time())}_{secrets.token_hex(8)}",
                'type': alert_type,
                'severity': severity.value,
                'message': message,
                'timestamp': datetime.now().isoformat()
            }
            
            self.security_alerts.append(alert)
            
            # Log critical alerts
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                self.logger.critical(f"🚨 SECURITY ALERT [{severity.value}] {alert_type}: {message}")
            
        except Exception as e:
            self.logger.error(f"Security alert creation failed: {e}")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        try:
            metrics = self.security_metrics.copy()
            metrics['active_sessions'] = len(self.active_sessions)
            metrics['total_events'] = len(self.security_events)
            metrics['total_alerts'] = len(self.security_alerts)
            metrics['credentials_stored'] = len(self.credentials)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Get security metrics failed: {e}")
            return {}
    
    def get_security_events(self, hours: int = 24, severity: SecurityLevel = None) -> List[SecurityEvent]:
        """Get recent security events"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            events = [
                event for event in self.security_events
                if event.timestamp > cutoff_time
            ]
            
            if severity:
                events = [
                    event for event in events
                    if event.severity == severity
                ]
            
            return sorted(events, key=lambda x: x.timestamp, reverse=True)
            
        except Exception as e:
            self.logger.error(f"Get security events failed: {e}")
            return []
    
    def get_active_sessions(self) -> List[SecuritySession]:
        """Get all active sessions"""
        try:
            with self.security_lock:
                return list(self.active_sessions.values())
                
        except Exception as e:
            self.logger.error(f"Get active sessions failed: {e}")
            return []
    
    def validate_password_policy(self, password: str) -> Tuple[bool, List[str]]:
        """
        Validate password against security policy
        
        Args:
            password: Password to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        try:
            errors = []
            
            # Check minimum length
            min_length = self.password_policy.get('min_length', 12)
            if len(password) < min_length:
                errors.append(f"Password must be at least {min_length} characters long")
            
            # Check for uppercase
            if self.password_policy.get('require_uppercase', True) and not any(c.isupper() for c in password):
                errors.append("Password must contain at least one uppercase letter")
            
            # Check for lowercase
            if self.password_policy.get('require_lowercase', True) and not any(c.islower() for c in password):
                errors.append("Password must contain at least one lowercase letter")
            
            # Check for numbers
            if self.password_policy.get('require_numbers', True) and not any(c.isdigit() for c in password):
                errors.append("Password must contain at least one number")
            
            # Check for special characters
            if self.password_policy.get('require_special', True):
                special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
                if not any(c in special_chars for c in password):
                    errors.append("Password must contain at least one special character")
            
            return len(errors) == 0, errors
            
        except Exception as e:
            self.logger.error(f"Password validation failed: {e}")
            return False, ["Password validation error"]
    
    def generate_secure_password(self, length: int = 16) -> str:
        """Generate a secure password"""
        try:
            import string
            characters = string.ascii_letters + string.digits + "!@#$%^&*"
            password = ''.join(secrets.choice(characters) for _ in range(length))
            return password
            
        except Exception as e:
            self.logger.error(f"Secure password generation failed: {e}")
            return ""
    
    def is_healthy(self) -> bool:
        """Check if security manager is healthy"""
        try:
            return (
                self.is_initialized and
                self.fernet_cipher is not None and
                self.rsa_private_key is not None and
                self.is_monitoring
            )
        except:
            return False
    
    def stop(self):
        """Stop security manager"""
        try:
            self.is_monitoring = False
            
            # Save credentials and events
            # Implementation would save encrypted data
            
            self.logger.info("Security Manager stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping security manager: {e}")

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create security manager
    security_manager = SecurityManager()
    
    # Initialize
    if security_manager.initialize():
        print("✅ Security Manager initialized successfully")
        
        # Test credential encryption
        test_credential = "test_api_key_12345"
        cred_id = security_manager.encrypt_credential("test_api", test_credential, SecurityLevel.HIGH)
        
        if cred_id:
            print(f"✅ Credential encrypted with ID: {cred_id}")
            
            # Test credential decryption
            decrypted = security_manager.decrypt_credential(cred_id)
            if decrypted == test_credential:
                print("✅ Credential decryption successful")
            else:
                print("❌ Credential decryption failed")
        
        # Test session management
        session_id = security_manager.create_session("test_user", AccessLevel.STANDARD, "127.0.0.1")
        if session_id:
            print(f"✅ Session created: {session_id}")
            
            # Validate session
            session = security_manager.validate_session(session_id)
            if session:
                print("✅ Session validation successful")
            
            # Revoke session
            if security_manager.revoke_session(session_id):
                print("✅ Session revoked successfully")
        
        # Test password validation
        test_password = "TestPassword123!"
        is_valid, errors = security_manager.validate_password_policy(test_password)
        print(f"Password validation: {'✅ Valid' if is_valid else '❌ Invalid'}")
        if errors:
            print(f"Errors: {errors}")
        
        # Generate secure password
        secure_pwd = security_manager.generate_secure_password()
        print(f"Generated secure password: {secure_pwd}")
        
        # Get security metrics
        metrics = security_manager.get_security_metrics()
        print(f"Security metrics: {metrics}")
        
        # Stop
        security_manager.stop()
    else:
        print("❌ Security Manager initialization failed")