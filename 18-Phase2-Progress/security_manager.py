#!/usr/bin/env python3
"""
EA GlobalFlow Pro v0.1 - Security and Encryption Manager
=======================================================

Comprehensive security management solution providing:
- Advanced encryption and decryption services
- Secure credential management
- Access control and authentication
- Security audit logging
- Threat detection and prevention
- Secure communication protocols
- Hardware fingerprinting and license validation

Author: EA GlobalFlow Pro Development Team
Date: August 2025
Version: v0.1 - Production Ready
"""

import base64
import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import platform
import uuid
import psutil

# Internal imports
from error_handler import ErrorHandler

class SecurityLevel(Enum):
    """Security level enumeration"""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AccessLevel(Enum):
    """Access level enumeration"""
    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"
    ADMIN = "ADMIN"
    SYSTEM = "SYSTEM"

class EncryptionType(Enum):
    """Encryption type enumeration"""
    SYMMETRIC = "SYMMETRIC"
    ASYMMETRIC = "ASYMMETRIC"
    HYBRID = "HYBRID"

@dataclass
class SecurityEvent:
    """Container for security events"""
    event_id: str
    event_type: str
    severity: SecurityLevel
    timestamp: datetime
    user_id: str
    component: str
    action: str
    resource: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class EncryptionKey:
    """Container for encryption key information"""
    key_id: str
    key_type: EncryptionType
    algorithm: str
    key_data: bytes
    created_at: datetime
    expires_at: Optional[datetime] = None
    usage_count: int = 0
    max_usage: Optional[int] = None

@dataclass
class AccessPermission:
    """Container for access permissions"""
    user_id: str
    resource: str
    access_level: AccessLevel
    granted_at: datetime
    granted_by: str
    expires_at: Optional[datetime] = None
    conditions: Dict[str, Any] = field(default_factory=dict)

class SecurityManager:
    """
    Advanced Security and Encryption Manager
    
    Provides comprehensive security services including:
    - Symmetric and asymmetric encryption
    - Secure credential storage and retrieval
    - Access control and permission management
    - Security event logging and monitoring
    - Hardware fingerprinting and license validation
    - Threat detection and prevention
    - Secure communication protocols
    """
    
    def __init__(self, config_path: str = "Config/ea_config_v01.json"):
        """Initialize Security Manager"""
        
        # Load configuration
        self.config = self._load_config(config_path)
        self.security_config = self.config.get('security', {})
        
        # Initialize logging
        self.logger = logging.getLogger('SecurityManager')
        self.logger.setLevel(logging.INFO)
        
        # Initialize error handler
        self.error_handler = ErrorHandler() if 'ErrorHandler' in globals() else None
        
        # Security state
        self.is_initialized = False
        self.master_key = None
        self.hardware_fingerprint = None
        
        # Key management
        self.encryption_keys: Dict[str, EncryptionKey] = {}
        self.key_rotation_interval = self.security_config.get('key_rotation_hours', 24)
        
        # Access control
        self.access_permissions: Dict[str, List[AccessPermission]] = {}
        self.active_sessions: Dict[str, Dict] = {}
        
        # Security monitoring
        self.security_events: List[SecurityEvent] = []
        self.threat_patterns: Dict[str, int] = {}
        self.max_events_memory = self.security_config.get('max_events_memory', 10000)
        
        # Configuration parameters
        self.encryption_algorithm = self.security_config.get('encryption_algorithm', 'AES-256-GCM')
        self.key_derivation_iterations = self.security_config.get('key_derivation_iterations', 100000)
        self.session_timeout_minutes = self.security_config.get('session_timeout_minutes', 480)
        self.max_failed_attempts = self.security_config.get('max_failed_attempts', 5)
        
        # Thread management
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.security_lock = threading.RLock()
        
        # Database for persistence
        self.db_connection = self._init_database()
        
        # Security paths
        self.secure_data_dir = Path("Data/Secure")
        self.secure_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize security subsystems
        self._initialize_encryption_system()
        self._initialize_hardware_fingerprint()
        
        self.logger.info("Security Manager initialized successfully")

    def _load_config(self, config_path: str) -> Dict:
        """Load security configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load security config: {e}")
            return self._get_default_config()

    def _get_default_config(self) -> Dict:
        """Default security configuration"""
        return {
            'security': {
                'encryption_algorithm': 'AES-256-GCM',
                'key_derivation_iterations': 100000,
                'key_rotation_hours': 24,
                'session_timeout_minutes': 480,
                'max_failed_attempts': 5,
                'max_events_memory': 10000,
                'enable_hardware_binding': True,
                'enable_audit_logging': True,
                'master_password': 'masterpass2024',  # Default - should be changed
                'hardware_id': '5E811D5000000024'  # Default - should be system-specific
            }
        }

    def _init_database(self) -> sqlite3.Connection:
        """Initialize security database"""
        try:
            db_path = self.secure_data_dir / "security.db"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            
            # Create tables
            conn.execute('''
                CREATE TABLE IF NOT EXISTS security_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE,
                    event_type TEXT,
                    severity TEXT,
                    timestamp DATETIME,
                    user_id TEXT,
                    component TEXT,
                    action TEXT,
                    resource TEXT,
                    success BOOLEAN,
                    details TEXT,
                    ip_address TEXT,
                    session_id TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS encryption_keys (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT UNIQUE,
                    key_type TEXT,
                    algorithm TEXT,
                    key_data_encrypted BLOB,
                    created_at DATETIME,
                    expires_at DATETIME,
                    usage_count INTEGER DEFAULT 0,
                    max_usage INTEGER
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS access_permissions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    resource TEXT,
                    access_level TEXT,
                    granted_at DATETIME,
                    granted_by TEXT,
                    expires_at DATETIME,
                    conditions TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS secure_credentials (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    credential_id TEXT UNIQUE,
                    service_name TEXT,
                    username TEXT,
                    encrypted_password BLOB,
                    encrypted_data BLOB,
                    created_at DATETIME,
                    updated_at DATETIME,
                    access_count INTEGER DEFAULT 0
                )
            ''')
            
            conn.commit()
            return conn
            
        except Exception as e:
            self.logger.error(f"Security database initialization failed: {e}")
            return None

    def _initialize_encryption_system(self):
        """Initialize encryption system with master key"""
        try:
            # Generate or load master key
            master_password = self.security_config.get('master_password', 'masterpass2024')
            self.master_key = self._derive_master_key(master_password)
            
            # Initialize default encryption key
            self._generate_encryption_key('default', EncryptionType.SYMMETRIC)
            
            self.is_initialized = True
            self.logger.info("Encryption system initialized")
            
        except Exception as e:
            self.logger.error(f"Encryption system initialization failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("ENCRYPTION_INIT_FAILED", str(e), "SecurityManager", "_initialize_encryption_system")

    def _derive_master_key(self, password: str) -> bytes:
        """Derive master key from password using PBKDF2"""
        try:
            # Use hardware fingerprint as salt for additional security
            salt = self._get_hardware_salt()
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,  # 256 bits
                salt=salt,
                iterations=self.key_derivation_iterations,
            )
            
            return kdf.derive(password.encode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Master key derivation failed: {e}")
            # Fallback to simple hash (less secure)
            return hashlib.sha256(password.encode('utf-8')).digest()

    def _initialize_hardware_fingerprint(self):
        """Initialize hardware fingerprinting system"""
        try:
            self.hardware_fingerprint = self._generate_hardware_fingerprint()
            
            # Validate against expected hardware ID if configured
            expected_hw_id = self.security_config.get('hardware_id')
            if expected_hw_id and self.hardware_fingerprint != expected_hw_id:
                self.logger.warning("Hardware fingerprint mismatch detected")
                self._log_security_event(
                    "HARDWARE_MISMATCH",
                    SecurityLevel.HIGH,
                    "SYSTEM",
                    "hardware_check",
                    "system",
                    False,
                    {'expected': expected_hw_id, 'actual': self.hardware_fingerprint}
                )
            
            self.logger.info(f"Hardware fingerprint: {self.hardware_fingerprint}")
            
        except Exception as e:
            self.logger.error(f"Hardware fingerprint initialization failed: {e}")

    def _generate_hardware_fingerprint(self) -> str:
        """Generate unique hardware fingerprint"""
        try:
            # Collect hardware information
            hardware_info = []
            
            # CPU information
            hardware_info.append(platform.processor())
            hardware_info.append(str(psutil.cpu_count()))
            
            # Memory information
            hardware_info.append(str(psutil.virtual_memory().total))
            
            # Disk information
            try:
                disk_info = psutil.disk_usage('/')
                hardware_info.append(str(disk_info.total))
            except:
                hardware_info.append('unknown_disk')
            
            # Network MAC address
            try:
                mac = ':'.join(['{:02x}'.format((uuid.getnode() >> elements) & 0xff) 
                               for elements in range(0, 2*6, 2)][::-1])
                hardware_info.append(mac)
            except:
                hardware_info.append('unknown_mac')
            
            # Platform information
            hardware_info.append(platform.system())
            hardware_info.append(platform.release())
            
            # Create fingerprint hash
            fingerprint_data = '|'.join(hardware_info)
            fingerprint_hash = hashlib.sha256(fingerprint_data.encode('utf-8')).hexdigest()
            
            # Return first 16 characters for readability
            return fingerprint_hash[:16].upper()
            
        except Exception as e:
            self.logger.error(f"Hardware fingerprint generation failed: {e}")
            return "UNKNOWN_HARDWARE"

    def _get_hardware_salt(self) -> bytes:
        """Get hardware-specific salt for key derivation"""
        try:
            if self.hardware_fingerprint:
                return hashlib.sha256(self.hardware_fingerprint.encode('utf-8')).digest()[:16]
            else:
                # Fallback salt
                return b'GlobalFlowSalt16'
        except:
            return b'GlobalFlowSalt16'

    def _generate_encryption_key(self, key_id: str, key_type: EncryptionType) -> EncryptionKey:
        """Generate new encryption key"""
        try:
            if key_type == EncryptionType.SYMMETRIC:
                # Generate AES key
                key_data = Fernet.generate_key()
                algorithm = 'Fernet'
                
            elif key_type == EncryptionType.ASYMMETRIC:
                # Generate RSA key pair
                private_key = rsa.generate_private_key(
                    public_exponent=65537,
                    key_size=2048
                )
                key_data = private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                )
                algorithm = 'RSA-2048'
                
            else:
                raise ValueError(f"Unsupported key type: {key_type}")
            
            # Create encryption key object
            encryption_key = EncryptionKey(
                key_id=key_id,
                key_type=key_type,
                algorithm=algorithm,
                key_data=key_data,
                created_at=datetime.now()
            )
            
            # Store in memory
            with self.security_lock:
                self.encryption_keys[key_id] = encryption_key
            
            # Store in database (encrypted)
            self._store_encryption_key(encryption_key)
            
            self.logger.info(f"Generated {key_type.value} encryption key: {key_id}")
            return encryption_key
            
        except Exception as e:
            self.logger.error(f"Encryption key generation failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("KEY_GENERATION_FAILED", str(e), "SecurityManager", "_generate_encryption_key")
            raise

    def _store_encryption_key(self, encryption_key: EncryptionKey):
        """Store encryption key in database (encrypted with master key)"""
        try:
            if not self.db_connection or not self.master_key:
                return
            
            # Encrypt key data with master key
            fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
            encrypted_key_data = fernet.encrypt(encryption_key.key_data)
            
            self.db_connection.execute('''
                INSERT OR REPLACE INTO encryption_keys 
                (key_id, key_type, algorithm, key_data_encrypted, created_at, expires_at, max_usage)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                encryption_key.key_id,
                encryption_key.key_type.value,
                encryption_key.algorithm,
                encrypted_key_data,
                encryption_key.created_at,
                encryption_key.expires_at,
                encryption_key.max_usage
            ))
            self.db_connection.commit()
            
        except Exception as e:
            self.logger.error(f"Failed to store encryption key: {e}")

    def encrypt_data(self, data: Union[str, bytes], key_id: str = 'default') -> Dict[str, Any]:
        """Encrypt data using specified key"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Security manager not initialized")
            
            # Get encryption key
            if key_id not in self.encryption_keys:
                self._generate_encryption_key(key_id, EncryptionType.SYMMETRIC)
            
            encryption_key = self.encryption_keys[key_id]
            
            # Convert data to bytes if string
            if isinstance(data, str):
                data_bytes = data.encode('utf-8')
            else:
                data_bytes = data
            
            # Encrypt based on key type
            if encryption_key.key_type == EncryptionType.SYMMETRIC:
                fernet = Fernet(encryption_key.key_data)
                encrypted_data = fernet.encrypt(data_bytes)
                
            elif encryption_key.key_type == EncryptionType.ASYMMETRIC:
                # Load RSA key
                private_key = serialization.load_pem_private_key(
                    encryption_key.key_data,
                    password=None
                )
                public_key = private_key.public_key()
                
                # RSA can only encrypt small amounts of data
                max_chunk_size = 190  # For 2048-bit key
                encrypted_chunks = []
                
                for i in range(0, len(data_bytes), max_chunk_size):
                    chunk = data_bytes[i:i + max_chunk_size]
                    encrypted_chunk = public_key.encrypt(
                        chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    encrypted_chunks.append(encrypted_chunk)
                
                encrypted_data = b''.join(encrypted_chunks)
            
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_key.key_type}")
            
            # Update usage count
            encryption_key.usage_count += 1
            
            # Log security event
            self._log_security_event(
                "DATA_ENCRYPTED",
                SecurityLevel.LOW,
                "SYSTEM",
                "encrypt_data",
                f"key:{key_id}",
                True,
                {'data_size': len(data_bytes), 'algorithm': encryption_key.algorithm}
            )
            
            return {
                'encrypted_data': base64.b64encode(encrypted_data).decode('utf-8'),
                'key_id': key_id,
                'algorithm': encryption_key.algorithm,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Data encryption failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("ENCRYPTION_FAILED", str(e), "SecurityManager", "encrypt_data")
            raise

    def decrypt_data(self, encrypted_data_info: Dict[str, Any]) -> Union[str, bytes]:
        """Decrypt data using specified key"""
        try:
            if not self.is_initialized:
                raise RuntimeError("Security manager not initialized")
            
            # Extract information
            encrypted_data_b64 = encrypted_data_info['encrypted_data']
            key_id = encrypted_data_info['key_id']
            
            # Decode base64
            encrypted_data = base64.b64decode(encrypted_data_b64.encode('utf-8'))
            
            # Get encryption key
            if key_id not in self.encryption_keys:
                # Try to load from database
                self._load_encryption_key(key_id)
            
            if key_id not in self.encryption_keys:
                raise RuntimeError(f"Encryption key not found: {key_id}")
            
            encryption_key = self.encryption_keys[key_id]
            
            # Decrypt based on key type
            if encryption_key.key_type == EncryptionType.SYMMETRIC:
                fernet = Fernet(encryption_key.key_data)
                decrypted_data = fernet.decrypt(encrypted_data)
                
            elif encryption_key.key_type == EncryptionType.ASYMMETRIC:
                # Load RSA key
                private_key = serialization.load_pem_private_key(
                    encryption_key.key_data,
                    password=None
                )
                
                # RSA decryption
                chunk_size = 256  # For 2048-bit key
                decrypted_chunks = []
                
                for i in range(0, len(encrypted_data), chunk_size):
                    chunk = encrypted_data[i:i + chunk_size]
                    decrypted_chunk = private_key.decrypt(
                        chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    decrypted_chunks.append(decrypted_chunk)
                
                decrypted_data = b''.join(decrypted_chunks)
            
            else:
                raise ValueError(f"Unsupported encryption type: {encryption_key.key_type}")
            
            # Log security event
            self._log_security_event(
                "DATA_DECRYPTED",
                SecurityLevel.LOW,
                "SYSTEM",
                "decrypt_data",
                f"key:{key_id}",
                True,
                {'data_size': len(decrypted_data)}
            )
            
            # Try to decode as UTF-8 string, otherwise return bytes
            try:
                return decrypted_data.decode('utf-8')
            except UnicodeDecodeError:
                return decrypted_data
                
        except Exception as e:
            self.logger.error(f"Data decryption failed: {e}")
            
            # Log security event for failed decryption
            self._log_security_event(
                "DECRYPTION_FAILED",
                SecurityLevel.MEDIUM,
                "SYSTEM",
                "decrypt_data",
                f"key:{encrypted_data_info.get('key_id', 'unknown')}",
                False,
                {'error': str(e)}
            )
            
            if self.error_handler:
                self.error_handler.handle_error("DECRYPTION_FAILED", str(e), "SecurityManager", "decrypt_data")
            raise

    def _load_encryption_key(self, key_id: str):
        """Load encryption key from database"""
        try:
            if not self.db_connection or not self.master_key:
                return
            
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT key_type, algorithm, key_data_encrypted, created_at, expires_at, usage_count, max_usage FROM encryption_keys WHERE key_id = ?",
                (key_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                return
            
            # Decrypt key data
            fernet = Fernet(base64.urlsafe_b64encode(self.master_key))
            key_data = fernet.decrypt(row[2])
            
            # Create encryption key object
            encryption_key = EncryptionKey(
                key_id=key_id,
                key_type=EncryptionType(row[0]),
                algorithm=row[1],
                key_data=key_data,
                created_at=datetime.fromisoformat(row[3]),
                expires_at=datetime.fromisoformat(row[4]) if row[4] else None,
                usage_count=row[5],
                max_usage=row[6]
            )
            
            # Store in memory
            with self.security_lock:
                self.encryption_keys[key_id] = encryption_key
            
            self.logger.info(f"Loaded encryption key from database: {key_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to load encryption key {key_id}: {e}")

    def store_secure_credential(self, credential_id: str, service_name: str, 
                              username: str, password: str, additional_data: Optional[Dict] = None) -> bool:
        """Store credentials securely"""
        try:
            # Encrypt password and additional data
            password_encrypted = self.encrypt_data(password, 'credentials')
            
            additional_data_encrypted = None
            if additional_data:
                additional_data_encrypted = self.encrypt_data(json.dumps(additional_data), 'credentials')
            
            # Store in database
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT OR REPLACE INTO secure_credentials 
                    (credential_id, service_name, username, encrypted_password, encrypted_data, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    credential_id,
                    service_name,
                    username,
                    json.dumps(password_encrypted).encode('utf-8'),
                    json.dumps(additional_data_encrypted).encode('utf-8') if additional_data_encrypted else None,
                    datetime.now(),
                    datetime.now()
                ))
                self.db_connection.commit()
            
            # Log security event
            self._log_security_event(
                "CREDENTIAL_STORED",
                SecurityLevel.MEDIUM,
                "SYSTEM",
                "store_credential",
                f"service:{service_name}",
                True,
                {'credential_id': credential_id, 'username': username}
            )
            
            self.logger.info(f"Stored secure credential: {credential_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store secure credential: {e}")
            if self.error_handler:
                self.error_handler.handle_error("CREDENTIAL_STORE_FAILED", str(e), "SecurityManager", "store_secure_credential")
            return False

    def retrieve_secure_credential(self, credential_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve credentials securely"""
        try:
            if not self.db_connection:
                return None
            
            cursor = self.db_connection.cursor()
            cursor.execute(
                "SELECT service_name, username, encrypted_password, encrypted_data, access_count FROM secure_credentials WHERE credential_id = ?",
                (credential_id,)
            )
            
            row = cursor.fetchone()
            if not row:
                # Log failed access attempt
                self._log_security_event(
                    "CREDENTIAL_ACCESS_FAILED",
                    SecurityLevel.MEDIUM,
                    "SYSTEM",
                    "retrieve_credential",
                    f"credential:{credential_id}",
                    False,
                    {'reason': 'not_found'}
                )
                return None
            
            # Decrypt password
            password_encrypted = json.loads(row[2].decode('utf-8'))
            password = self.decrypt_data(password_encrypted)
            
            # Decrypt additional data if exists
            additional_data = None
            if row[3]:
                additional_data_encrypted = json.loads(row[3].decode('utf-8'))
                additional_data_json = self.decrypt_data(additional_data_encrypted)
                additional_data = json.loads(additional_data_json)
            
            # Update access count
            self.db_connection.execute(
                "UPDATE secure_credentials SET access_count = access_count + 1 WHERE credential_id = ?",
                (credential_id,)
            )
            self.db_connection.commit()
            
            # Log successful access
            self._log_security_event(
                "CREDENTIAL_ACCESSED",
                SecurityLevel.LOW,
                "SYSTEM",
                "retrieve_credential",
                f"credential:{credential_id}",
                True,
                {'service_name': row[0], 'username': row[1]}
            )
            
            return {
                'credential_id': credential_id,
                'service_name': row[0],
                'username': row[1],
                'password': password,
                'additional_data': additional_data,
                'access_count': row[4]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to retrieve secure credential: {e}")
            
            # Log failed access attempt
            self._log_security_event(
                "CREDENTIAL_ACCESS_ERROR",
                SecurityLevel.HIGH,
                "SYSTEM",
                "retrieve_credential",
                f"credential:{credential_id}",
                False,
                {'error': str(e)}
            )
            
            if self.error_handler:
                self.error_handler.handle_error("CREDENTIAL_RETRIEVE_FAILED", str(e), "SecurityManager", "retrieve_secure_credential")
            return None

    def create_secure_session(self, user_id: str, access_level: AccessLevel, 
                            ip_address: Optional[str] = None) -> str:
        """Create secure session with authentication"""
        try:
            # Generate session ID
            session_id = self._generate_session_id()
            
            # Create session data
            session_data = {
                'session_id': session_id,
                'user_id': user_id,
                'access_level': access_level.value,
                'created_at': datetime.now(),
                'last_activity': datetime.now(),
                'ip_address': ip_address,
                'expires_at': datetime.now() + timedelta(minutes=self.session_timeout_minutes),
                'is_active': True
            }
            
            # Store session
            with self.security_lock:
                self.active_sessions[session_id] = session_data
            
            # Log security event
            self._log_security_event(
                "SESSION_CREATED",
                SecurityLevel.MEDIUM,
                user_id,
                "create_session",
                f"session:{session_id}",
                True,
                {'access_level': access_level.value, 'ip_address': ip_address}
            )
            
            self.logger.info(f"Created secure session: {session_id} for user: {user_id}")
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to create secure session: {e}")
            if self.error_handler:
                self.error_handler.handle_error("SESSION_CREATE_FAILED", str(e), "SecurityManager", "create_secure_session")
            raise

    def validate_session(self, session_id: str, required_access: Optional[AccessLevel] = None) -> bool:
        """Validate session and check access level"""
        try:
            with self.security_lock:
                if session_id not in self.active_sessions:
                    self._log_security_event(
                        "SESSION_VALIDATION_FAILED",
                        SecurityLevel.MEDIUM,
                        "UNKNOWN",
                        "validate_session",
                        f"session:{session_id}",
                        False,
                        {'reason': 'session_not_found'}
                    )
                    return False
                
                session = self.active_sessions[session_id]
                
                # Check if session is active
                if not session['is_active']:
                    self._log_security_event(
                        "SESSION_VALIDATION_FAILED",
                        SecurityLevel.MEDIUM,
                        session['user_id'],
                        "validate_session",
                        f"session:{session_id}",
                        False,
                        {'reason': 'session_inactive'}
                    )
                    return False
                
                # Check if session has expired
                if datetime.now() > session['expires_at']:
                    session['is_active'] = False
                    self._log_security_event(
                        "SESSION_EXPIRED",
                        SecurityLevel.MEDIUM,
                        session['user_id'],
                        "validate_session",
                        f"session:{session_id}",
                        False,
                        {'reason': 'session_expired'}
                    )
                    return False
                
                # Check access level if required
                if required_access:
                    session_access = AccessLevel(session['access_level'])
                    if not self._check_access_level(session_access, required_access):
                        self._log_security_event(
                            "ACCESS_DENIED",
                            SecurityLevel.HIGH,
                            session['user_id'],
                            "validate_session",
                            f"session:{session_id}",
                            False,
                            {'required_access': required_access.value, 'session_access': session_access.value}
                        )
                        return False
                
                # Update last activity
                session['last_activity'] = datetime.now()
                
                return True
                
        except Exception as e:
            self.logger.error(f"Session validation failed: {e}")
            return False

    def _check_access_level(self, session_access: AccessLevel, required_access: AccessLevel) -> bool:
        """Check if session access level meets requirements"""
        access_hierarchy = {
            AccessLevel.READ_ONLY: 1,
            AccessLevel.READ_WRITE: 2,
            AccessLevel.ADMIN: 3,
            AccessLevel.SYSTEM: 4
        }
        
        return access_hierarchy.get(session_access, 0) >= access_hierarchy.get(required_access, 0)

    def invalidate_session(self, session_id: str) -> bool:
        """Invalidate a session"""
        try:
            with self.security_lock:
                if session_id in self.active_sessions:
                    session = self.active_sessions[session_id]
                    session['is_active'] = False
                    
                    # Log security event
                    self._log_security_event(
                        "SESSION_INVALIDATED",
                        SecurityLevel.MEDIUM,
                        session['user_id'],
                        "invalidate_session",
                        f"session:{session_id}",
                        True,
                        {}
                    )
                    
                    del self.active_sessions[session_id]
                    self.logger.info(f"Session invalidated: {session_id}")
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to invalidate session: {e}")
            return False

    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        return secrets.token_urlsafe(32)

    def generate_secure_token(self, purpose: str, validity_hours: int = 24) -> Dict[str, Any]:
        """Generate secure token for specific purpose"""
        try:
            # Generate token
            token = secrets.token_urlsafe(32)
            
            # Create token data
            token_data = {
                'token': token,
                'purpose': purpose,
                'created_at': datetime.now(),
                'expires_at': datetime.now() + timedelta(hours=validity_hours),
                'used': False
            }
            
            # Sign token with HMAC
            signature = self._sign_token(token, token_data)
            token_data['signature'] = signature
            
            # Log security event
            self._log_security_event(
                "TOKEN_GENERATED",
                SecurityLevel.MEDIUM,
                "SYSTEM",
                "generate_token",
                f"purpose:{purpose}",
                True,
                {'validity_hours': validity_hours}
            )
            
            return token_data
            
        except Exception as e:
            self.logger.error(f"Token generation failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("TOKEN_GENERATION_FAILED", str(e), "SecurityManager", "generate_secure_token")
            raise

    def validate_secure_token(self, token_data: Dict[str, Any]) -> bool:
        """Validate secure token"""
        try:
            # Check expiration
            if datetime.now() > datetime.fromisoformat(token_data['expires_at']):
                self._log_security_event(
                    "TOKEN_VALIDATION_FAILED",
                    SecurityLevel.MEDIUM,
                    "SYSTEM",
                    "validate_token",
                    f"purpose:{token_data.get('purpose', 'unknown')}",
                    False,
                    {'reason': 'token_expired'}
                )
                return False
            
            # Check if already used (for one-time tokens)
            if token_data.get('used', False):
                self._log_security_event(
                    "TOKEN_VALIDATION_FAILED",
                    SecurityLevel.HIGH,
                    "SYSTEM",
                    "validate_token",
                    f"purpose:{token_data.get('purpose', 'unknown')}",
                    False,
                    {'reason': 'token_already_used'}
                )
                return False
            
            # Validate signature
            expected_signature = self._sign_token(token_data['token'], {
                k: v for k, v in token_data.items() if k != 'signature'
            })
            
            if not hmac.compare_digest(token_data['signature'], expected_signature):
                self._log_security_event(
                    "TOKEN_VALIDATION_FAILED",
                    SecurityLevel.HIGH,
                    "SYSTEM",
                    "validate_token",
                    f"purpose:{token_data.get('purpose', 'unknown')}",
                    False,
                    {'reason': 'invalid_signature'}
                )
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Token validation failed: {e}")
            return False

    def _sign_token(self, token: str, token_data: Dict[str, Any]) -> str:
        """Sign token with HMAC"""
        try:
            # Create signature payload
            payload = f"{token}|{json.dumps(token_data, sort_keys=True, default=str)}"
            
            # Sign with master key
            signature = hmac.new(
                self.master_key,
                payload.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            self.logger.error(f"Token signing failed: {e}")
            return ""

    def _log_security_event(self, event_type: str, severity: SecurityLevel, 
                          user_id: str, action: str, resource: str, success: bool,
                          details: Optional[Dict] = None, ip_address: Optional[str] = None,
                          session_id: Optional[str] = None):
        """Log security event"""
        try:
            event_id = f"SEC_{int(time.time())}_{secrets.token_hex(4)}"
            
            event = SecurityEvent(
                event_id=event_id,
                event_type=event_type,
                severity=severity,
                timestamp=datetime.now(),
                user_id=user_id,
                component="SecurityManager",
                action=action,
                resource=resource,
                success=success,
                details=details or {},
                ip_address=ip_address,
                session_id=session_id
            )
            
            # Add to memory
            with self.security_lock:
                self.security_events.append(event)
                
                # Keep only recent events in memory
                if len(self.security_events) > self.max_events_memory:
                    self.security_events = self.security_events[-self.max_events_memory:]
            
            # Store in database
            if self.db_connection:
                self.db_connection.execute('''
                    INSERT INTO security_events 
                    (event_id, event_type, severity, timestamp, user_id, component, action, 
                     resource, success, details, ip_address, session_id)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    event.event_id,
                    event.event_type,
                    event.severity.value,
                    event.timestamp,
                    event.user_id,
                    event.component,
                    event.action,
                    event.resource,
                    event.success,
                    json.dumps(event.details),
                    event.ip_address,
                    event.session_id
                ))
                self.db_connection.commit()
            
            # Check for threat patterns
            self._analyze_threat_patterns(event)
            
        except Exception as e:
            self.logger.error(f"Security event logging failed: {e}")

    def _analyze_threat_patterns(self, event: SecurityEvent):
        """Analyze security events for threat patterns"""
        try:
            # Pattern: Multiple failed login attempts
            if event.event_type in ["SESSION_VALIDATION_FAILED", "ACCESS_DENIED"] and not event.success:
                pattern_key = f"failed_access_{event.user_id}"
                self.threat_patterns[pattern_key] = self.threat_patterns.get(pattern_key, 0) + 1
                
                if self.threat_patterns[pattern_key] >= self.max_failed_attempts:
                    self._trigger_security_alert(
                        "BRUTE_FORCE_DETECTED",
                        SecurityLevel.HIGH,
                        f"Multiple failed access attempts detected for user: {event.user_id}",
                        {'pattern_count': self.threat_patterns[pattern_key], 'user_id': event.user_id}
                    )
            
            # Pattern: Unusual access patterns
            if event.success and event.event_type == "CREDENTIAL_ACCESSED":
                current_hour = datetime.now().hour
                if current_hour < 6 or current_hour > 22:  # Outside normal hours
                    self._trigger_security_alert(
                        "UNUSUAL_ACCESS_TIME",
                        SecurityLevel.MEDIUM,
                        f"Credential access outside normal hours: {event.user_id}",
                        {'hour': current_hour, 'user_id': event.user_id}
                    )
            
            # Pattern: Hardware fingerprint mismatches
            if event.event_type == "HARDWARE_MISMATCH":
                self._trigger_security_alert(
                    "HARDWARE_MISMATCH_DETECTED",
                    SecurityLevel.CRITICAL,
                    "Hardware fingerprint mismatch detected - possible unauthorized access",
                    event.details
                )
                
        except Exception as e:
            self.logger.error(f"Threat pattern analysis failed: {e}")

    def _trigger_security_alert(self, alert_type: str, severity: SecurityLevel, 
                              message: str, details: Dict[str, Any]):
        """Trigger security alert"""
        try:
            alert_data = {
                'alert_type': alert_type,
                'severity': severity.value,
                'message': message,
                'details': details,
                'timestamp': datetime.now().isoformat()
            }
            
            # Log critical alert
            if severity in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
                self.logger.critical(f"SECURITY ALERT: {alert_type} - {message}")
            else:
                self.logger.warning(f"Security Alert: {alert_type} - {message}")
            
            # This would integrate with external alerting systems
            self._send_security_alert(alert_data)
            
        except Exception as e:
            self.logger.error(f"Security alert triggering failed: {e}")

    def _send_security_alert(self, alert_data: Dict[str, Any]):
        """Send security alert (placeholder for external integration)"""
        try:
            # This would integrate with email, SMS, or other alerting systems
            if alert_data['severity'] in ['HIGH', 'CRITICAL']:
                self.logger.info(f"Critical security alert would be sent: {alert_data['message']}")
                
        except Exception as e:
            self.logger.error(f"Security alert sending failed: {e}")

    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        try:
            recent_time = datetime.now() - timedelta(hours=24)
            
            with self.security_lock:
                # Recent security events
                recent_events = [e for e in self.security_events if e.timestamp > recent_time]
                
                # Event statistics
                event_stats = {
                    'total': len(recent_events),
                    'successful': len([e for e in recent_events if e.success]),
                    'failed': len([e for e in recent_events if not e.success]),
                    'critical': len([e for e in recent_events if e.severity == SecurityLevel.CRITICAL]),
                    'high': len([e for e in recent_events if e.severity == SecurityLevel.HIGH])
                }
                
                # Active sessions
                active_session_count = len([s for s in self.active_sessions.values() if s['is_active']])
                
                # Threat patterns
                active_threats = len([p for p in self.threat_patterns.values() if p >= 3])
            
            # Overall security health
            security_health = self._assess_security_health(event_stats, active_threats)
            
            return {
                'security_health': security_health,
                'hardware_fingerprint': self.hardware_fingerprint,
                'encryption_system_status': 'OPERATIONAL' if self.is_initialized else 'OFFLINE',
                'active_sessions': active_session_count,
                'recent_events_24h': event_stats,
                'active_threat_patterns': active_threats,
                'encryption_keys_loaded': len(self.encryption_keys),
                'last_key_rotation': self._get_last_key_rotation(),
                'system_locked': not self.is_initialized,
                'assessment_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Security status assessment failed: {e}")
            return {'error': str(e)}

    def _assess_security_health(self, event_stats: Dict[str, int], active_threats: int) -> str:
        """Assess overall security health"""
        try:
            # Critical conditions
            if event_stats['critical'] > 0:
                return "CRITICAL"
            
            # High risk conditions
            if active_threats > 0 or event_stats['high'] > 5:
                return "HIGH_RISK"
            
            # Medium risk conditions
            if event_stats['failed'] > event_stats['successful'] * 0.1:  # More than 10% failure rate
                return "MEDIUM_RISK"
            
            # Warning conditions
            if event_stats['failed'] > 10:
                return "WARNING"
            
            return "HEALTHY"
            
        except Exception as e:
            return "UNKNOWN"

    def _get_last_key_rotation(self) -> Optional[str]:
        """Get timestamp of last key rotation"""
        try:
            if self.encryption_keys:
                latest_key = max(self.encryption_keys.values(), key=lambda k: k.created_at)
                return latest_key.created_at.isoformat()
            return None
        except:
            return None

    def rotate_encryption_keys(self) -> bool:
        """Rotate encryption keys for enhanced security"""
        try:
            self.logger.info("Starting encryption key rotation...")
            
            old_keys = list(self.encryption_keys.keys())
            
            # Generate new keys
            for key_id in old_keys:
                old_key = self.encryption_keys[key_id]
                
                # Create new key with same type
                new_key_id = f"{key_id}_rotated_{int(time.time())}"
                self._generate_encryption_key(new_key_id, old_key.key_type)
                
                # Mark old key as expired
                old_key.expires_at = datetime.now()
            
            # Log security event
            self._log_security_event(
                "KEY_ROTATION_COMPLETED",
                SecurityLevel.MEDIUM,
                "SYSTEM",
                "rotate_keys",
                "encryption_keys",
                True,
                {'rotated_keys': len(old_keys)}
            )
            
            self.logger.info(f"Encryption key rotation completed: {len(old_keys)} keys rotated")
            return True
            
        except Exception as e:
            self.logger.error(f"Key rotation failed: {e}")
            if self.error_handler:
                self.error_handler.handle_error("KEY_ROTATION_FAILED", str(e), "SecurityManager", "rotate_encryption_keys")
            return False

    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            with self.security_lock:
                for session_id, session in self.active_sessions.items():
                    if current_time > session['expires_at'] or not session['is_active']:
                        expired_sessions.append(session_id)
                
                for session_id in expired_sessions:
                    del self.active_sessions[session_id]
            
            if expired_sessions:
                self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
                
        except Exception as e:
            self.logger.error(f"Session cleanup failed: {e}")

    def get_security_audit_log(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get security audit log for specified period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            
            with self.security_lock:
                audit_events = [
                    {
                        'event_id': event.event_id,
                        'event_type': event.event_type,
                        'severity': event.severity.value,
                        'timestamp': event.timestamp.isoformat(),
                        'user_id': event.user_id,
                        'action': event.action,
                        'resource': event.resource,
                        'success': event.success,
                        'details': event.details,
                        'ip_address': event.ip_address
                    }
                    for event in self.security_events
                    if event.timestamp > cutoff_time
                ]
            
            return sorted(audit_events, key=lambda x: x['timestamp'], reverse=True)
            
        except Exception as e:
            self.logger.error(f"Security audit log generation failed: {e}")
            return []

    async def shutdown(self):
        """Gracefully shutdown security manager"""
        try:
            self.logger.info("🔄 Shutting down Security Manager...")
            
            # Invalidate all active sessions
            with self.security_lock:
                for session_id in list(self.active_sessions.keys()):
                    self.invalidate_session(session_id)
            
            # Clear sensitive data from memory
            self.master_key = None
            self.encryption_keys.clear()
            
            # Close database connection
            if self.db_connection:
                self.db_connection.close()
            
            # Shutdown executor
            self.executor.shutdown(wait=True)
            
            self.logger.info("✅ Security Manager shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Security Manager shutdown error: {e}")

# Global instance
security_manager = None

def get_security_manager() -> SecurityManager:
    """Get singleton instance of Security Manager"""
    global security_manager
    if security_manager is None:
        security_manager = SecurityManager()
    return security_manager

if __name__ == "__main__":
    # Test the Security Manager
    import asyncio
    
    async def main():
        security = SecurityManager()
        
        # Test encryption
        test_data = "Sensitive trading data: API_KEY=secret123"
        encrypted = security.encrypt_data(test_data)
        print(f"Encrypted: {encrypted}")
        
        decrypted = security.decrypt_data(encrypted)
        print(f"Decrypted: {decrypted}")
        
        # Test credential storage
        success = security.store_secure_credential(
            "fyers_api",
            "Fyers",
            "M3ZSJC8Q7I-100",
            "T7T0EDKGBT",
            {"app_id": "M3ZSJC8Q7I-100", "secret_key": "T7T0EDKGBT"}
        )
        print(f"Credential stored: {success}")
        
        # Test credential retrieval
        credential = security.retrieve_secure_credential("fyers_api")
        if credential:
            print(f"Retrieved credential: {credential['username']}")
        
        # Test session management
        session_id = security.create_secure_session("user123", AccessLevel.ADMIN)
        print(f"Session created: {session_id}")
        
        valid = security.validate_session(session_id, AccessLevel.READ_WRITE)
        print(f"Session valid: {valid}")
        
        # Test security status
        status = security.get_security_status()
        print(f"Security Status: {json.dumps(status, indent=2)}")
        
        # Shutdown
        await security.shutdown()
    
    asyncio.run(main())
