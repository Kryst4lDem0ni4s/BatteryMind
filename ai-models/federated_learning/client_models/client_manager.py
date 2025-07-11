"""
BatteryMind - Federated Learning Client Manager

Comprehensive client management system for federated learning with battery
health prediction models. Handles client lifecycle, communication protocols,
model synchronization, and fault tolerance.

Features:
- Dynamic client registration and discovery
- Secure communication protocols with encryption
- Model synchronization and version management
- Fault tolerance and recovery mechanisms
- Performance monitoring and resource management
- Privacy-preserving client coordination

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import asyncio
import aiohttp
import ssl
import json
import time
import logging
import hashlib
import threading
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import uuid
from datetime import datetime, timedelta
from collections import defaultdict, deque
import websockets
import grpc
from concurrent.futures import ThreadPoolExecutor
import pickle
import zlib

# Cryptographic imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import secrets

# Network and communication
import socket
import select
from urllib.parse import urlparse

# Local imports
from .local_trainer import LocalTrainer
from .privacy_engine import PrivacyEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientConfig:
    """
    Configuration for federated learning client.
    
    Attributes:
        client_id (str): Unique client identifier
        server_address (str): Federated learning server address
        server_port (int): Server port number
        
        # Communication settings
        communication_protocol (str): Protocol type ('http', 'websocket', 'grpc')
        encryption_enabled (bool): Enable end-to-end encryption
        compression_enabled (bool): Enable data compression
        
        # Model settings
        model_type (str): Type of model ('battery_health', 'degradation')
        local_epochs (int): Number of local training epochs
        batch_size (int): Local training batch size
        learning_rate (float): Local learning rate
        
        # Privacy settings
        differential_privacy (bool): Enable differential privacy
        privacy_budget (float): Privacy budget for DP
        secure_aggregation (bool): Enable secure aggregation
        
        # Resource management
        max_memory_usage (int): Maximum memory usage in MB
        max_cpu_usage (float): Maximum CPU usage percentage
        heartbeat_interval (int): Heartbeat interval in seconds
        
        # Fault tolerance
        max_retries (int): Maximum retry attempts
        timeout_seconds (int): Communication timeout
        reconnect_delay (int): Delay between reconnection attempts
    """
    client_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    server_address: str = "localhost"
    server_port: int = 8080
    
    # Communication settings
    communication_protocol: str = "websocket"
    encryption_enabled: bool = True
    compression_enabled: bool = True
    
    # Model settings
    model_type: str = "battery_health"
    local_epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 1e-4
    
    # Privacy settings
    differential_privacy: bool = True
    privacy_budget: float = 1.0
    secure_aggregation: bool = True
    
    # Resource management
    max_memory_usage: int = 2048  # MB
    max_cpu_usage: float = 80.0   # Percentage
    heartbeat_interval: int = 30  # Seconds
    
    # Fault tolerance
    max_retries: int = 3
    timeout_seconds: int = 60
    reconnect_delay: int = 5

@dataclass
class ClientStatus:
    """
    Current status of a federated learning client.
    """
    client_id: str
    status: str  # 'idle', 'training', 'uploading', 'error', 'disconnected'
    last_seen: datetime
    round_number: int
    model_version: str
    data_samples: int
    training_accuracy: float
    memory_usage: float
    cpu_usage: float
    network_latency: float
    error_message: Optional[str] = None

class SecureCommunicator:
    """
    Secure communication handler for federated learning clients.
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.encryption_key = None
        self.session_token = None
        
        if config.encryption_enabled:
            self._setup_encryption()
    
    def _setup_encryption(self):
        """Setup encryption keys and certificates."""
        # Generate or load encryption keys
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048
        )
        self.public_key = self.private_key.public_key()
        
        # Generate symmetric key for session encryption
        self.encryption_key = Fernet.generate_key()
        self.cipher_suite = Fernet(self.encryption_key)
        
        logger.info("Encryption setup completed")
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        if not self.config.encryption_enabled:
            return data
        
        return self.cipher_suite.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        if not self.config.encryption_enabled:
            return encrypted_data
        
        return self.cipher_suite.decrypt(encrypted_data)
    
    def compress_data(self, data: bytes) -> bytes:
        """Compress data for efficient transmission."""
        if not self.config.compression_enabled:
            return data
        
        return zlib.compress(data, level=6)
    
    def decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress received data."""
        if not self.config.compression_enabled:
            return compressed_data
        
        return zlib.decompress(compressed_data)
    
    def prepare_message(self, message: Dict[str, Any]) -> bytes:
        """Prepare message for transmission with encryption and compression."""
        # Serialize message
        serialized = pickle.dumps(message)
        
        # Compress if enabled
        if self.config.compression_enabled:
            serialized = self.compress_data(serialized)
        
        # Encrypt if enabled
        if self.config.encryption_enabled:
            serialized = self.encrypt_data(serialized)
        
        return serialized
    
    def parse_message(self, raw_data: bytes) -> Dict[str, Any]:
        """Parse received message with decryption and decompression."""
        # Decrypt if enabled
        if self.config.encryption_enabled:
            raw_data = self.decrypt_data(raw_data)
        
        # Decompress if enabled
        if self.config.compression_enabled:
            raw_data = self.decompress_data(raw_data)
        
        # Deserialize message
        return pickle.loads(raw_data)

class ResourceMonitor:
    """
    Monitor client resource usage and performance.
    """
    
    def __init__(self, config: ClientConfig):
        self.config = config
        self.metrics_history = deque(maxlen=100)
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        import psutil
        
        while self.monitoring_active:
            try:
                # Get current metrics
                metrics = {
                    'timestamp': datetime.now(),
                    'memory_usage_mb': psutil.virtual_memory().used / (1024 * 1024),
                    'memory_percent': psutil.virtual_memory().percent,
                    'cpu_percent': psutil.cpu_percent(interval=1),
                    'network_io': psutil.net_io_counters(),
                    'disk_io': psutil.disk_io_counters()
                }
                
                self.metrics_history.append(metrics)
                
                # Check resource limits
                if metrics['memory_usage_mb'] > self.config.max_memory_usage:
                    logger.warning(f"Memory usage exceeds limit: {metrics['memory_usage_mb']:.1f}MB")
                
                if metrics['cpu_percent'] > self.config.max_cpu_usage:
                    logger.warning(f"CPU usage exceeds limit: {metrics['cpu_percent']:.1f}%")
                
                time.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                time.sleep(10)
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current resource metrics."""
        if not self.metrics_history:
            return {'memory_usage': 0.0, 'cpu_usage': 0.0, 'network_latency': 0.0}
        
        latest = self.metrics_history[-1]
        return {
            'memory_usage': latest['memory_usage_mb'],
            'cpu_usage': latest['cpu_percent'],
            'network_latency': 0.0  # Will be updated by communication layer
        }

class FederatedClient:
    """
    Main federated learning client implementation.
    """
    
    def __init__(self, config: ClientConfig, local_trainer: LocalTrainer, 
                 privacy_engine: Optional[PrivacyEngine] = None):
        self.config = config
        self.local_trainer = local_trainer
        self.privacy_engine = privacy_engine or PrivacyEngine(config)
        
        # Communication components
        self.communicator = SecureCommunicator(config)
        self.resource_monitor = ResourceMonitor(config)
        
        # Client state
        self.status = ClientStatus(
            client_id=config.client_id,
            status='idle',
            last_seen=datetime.now(),
            round_number=0,
            model_version='1.0.0',
            data_samples=0,
            training_accuracy=0.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            network_latency=0.0
        )
        
        # Connection management
        self.connected = False
        self.websocket = None
        self.session = None
        self.heartbeat_task = None
        
        # Event handlers
        self.event_handlers = {
            'model_update': self._handle_model_update,
            'training_request': self._handle_training_request,
            'aggregation_result': self._handle_aggregation_result,
            'heartbeat': self._handle_heartbeat,
            'shutdown': self._handle_shutdown
        }
        
        logger.info(f"Federated client {config.client_id} initialized")
    
    async def connect(self) -> bool:
        """
        Connect to federated learning server.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            if self.config.communication_protocol == 'websocket':
                return await self._connect_websocket()
            elif self.config.communication_protocol == 'http':
                return await self._connect_http()
            elif self.config.communication_protocol == 'grpc':
                return await self._connect_grpc()
            else:
                raise ValueError(f"Unsupported protocol: {self.config.communication_protocol}")
        
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            return False
    
    async def _connect_websocket(self) -> bool:
        """Connect using WebSocket protocol."""
        try:
            uri = f"ws://{self.config.server_address}:{self.config.server_port}/federated"
            
            # Setup SSL context if needed
            ssl_context = None
            if self.config.encryption_enabled:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            
            self.websocket = await websockets.connect(
                uri, 
                ssl=ssl_context,
                timeout=self.config.timeout_seconds
            )
            
            # Send registration message
            registration_msg = {
                'type': 'register',
                'client_id': self.config.client_id,
                'model_type': self.config.model_type,
                'capabilities': {
                    'differential_privacy': self.config.differential_privacy,
                    'secure_aggregation': self.config.secure_aggregation,
                    'data_samples': self.local_trainer.get_data_size()
                }
            }
            
            await self._send_message(registration_msg)
            
            # Wait for registration confirmation
            response = await self._receive_message()
            
            if response.get('status') == 'registered':
                self.connected = True
                self.status.status = 'idle'
                self.status.last_seen = datetime.now()
                
                # Start heartbeat
                self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                
                # Start resource monitoring
                self.resource_monitor.start_monitoring()
                
                logger.info(f"Client {self.config.client_id} connected successfully")
                return True
            else:
                logger.error(f"Registration failed: {response}")
                return False
        
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def _connect_http(self) -> bool:
        """Connect using HTTP protocol."""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.config.timeout_seconds)
            )
            
            # Test connection with registration
            registration_data = {
                'client_id': self.config.client_id,
                'model_type': self.config.model_type,
                'capabilities': {
                    'differential_privacy': self.config.differential_privacy,
                    'secure_aggregation': self.config.secure_aggregation,
                    'data_samples': self.local_trainer.get_data_size()
                }
            }
            
            url = f"http://{self.config.server_address}:{self.config.server_port}/register"
            
            async with self.session.post(url, json=registration_data) as response:
                if response.status == 200:
                    result = await response.json()
                    if result.get('status') == 'registered':
                        self.connected = True
                        self.status.status = 'idle'
                        self.status.last_seen = datetime.now()
                        
                        # Start heartbeat
                        self.heartbeat_task = asyncio.create_task(self._heartbeat_loop())
                        
                        # Start resource monitoring
                        self.resource_monitor.start_monitoring()
                        
                        logger.info(f"Client {self.config.client_id} connected via HTTP")
                        return True
            
            return False
        
        except Exception as e:
            logger.error(f"HTTP connection failed: {e}")
            return False
    
    async def _connect_grpc(self) -> bool:
        """Connect using gRPC protocol."""
        # gRPC implementation would go here
        # For now, return False as not implemented
        logger.warning("gRPC protocol not yet implemented")
        return False
    
    async def disconnect(self):
        """Disconnect from federated learning server."""
        try:
            self.connected = False
            
            # Stop heartbeat
            if self.heartbeat_task:
                self.heartbeat_task.cancel()
                try:
                    await self.heartbeat_task
                except asyncio.CancelledError:
                    pass
            
            # Stop resource monitoring
            self.resource_monitor.stop_monitoring()
            
            # Send disconnect message
            if self.websocket:
                disconnect_msg = {
                    'type': 'disconnect',
                    'client_id': self.config.client_id
                }
                await self._send_message(disconnect_msg)
                await self.websocket.close()
                self.websocket = None
            
            if self.session:
                await self.session.close()
                self.session = None
            
            self.status.status = 'disconnected'
            logger.info(f"Client {self.config.client_id} disconnected")
        
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    async def _send_message(self, message: Dict[str, Any]):
        """Send message to server."""
        try:
            if self.config.communication_protocol == 'websocket' and self.websocket:
                # Prepare message with encryption/compression
                prepared_data = self.communicator.prepare_message(message)
                await self.websocket.send(prepared_data)
            
            elif self.config.communication_protocol == 'http' and self.session:
                url = f"http://{self.config.server_address}:{self.config.server_port}/message"
                async with self.session.post(url, json=message) as response:
                    return await response.json()
        
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            raise
    
    async def _receive_message(self) -> Dict[str, Any]:
        """Receive message from server."""
        try:
            if self.config.communication_protocol == 'websocket' and self.websocket:
                raw_data = await self.websocket.recv()
                return self.communicator.parse_message(raw_data)
            
            elif self.config.communication_protocol == 'http':
                # HTTP uses request-response pattern, no continuous receiving
                return {}
        
        except Exception as e:
            logger.error(f"Error receiving message: {e}")
            raise
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages to server."""
        while self.connected:
            try:
                # Get current resource metrics
                metrics = self.resource_monitor.get_current_metrics()
                
                heartbeat_msg = {
                    'type': 'heartbeat',
                    'client_id': self.config.client_id,
                    'timestamp': datetime.now().isoformat(),
                    'status': self.status.status,
                    'metrics': metrics
                }
                
                await self._send_message(heartbeat_msg)
                self.status.last_seen = datetime.now()
                
                await asyncio.sleep(self.config.heartbeat_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")
                await asyncio.sleep(self.config.heartbeat_interval)
    
    async def run(self):
        """Main client execution loop."""
        try:
            # Connect to server
            if not await self.connect():
                logger.error("Failed to connect to server")
                return
            
            logger.info("Client running, waiting for server commands...")
            
            # Main message handling loop
            while self.connected:
                try:
                    message = await self._receive_message()
                    
                    if message:
                        await self._handle_message(message)
                
                except websockets.exceptions.ConnectionClosed:
                    logger.warning("Connection closed by server")
                    break
                except Exception as e:
                    logger.error(f"Error in main loop: {e}")
                    await asyncio.sleep(1)
        
        except Exception as e:
            logger.error(f"Critical error in client run: {e}")
        
        finally:
            await self.disconnect()
    
    async def _handle_message(self, message: Dict[str, Any]):
        """Handle incoming message from server."""
        try:
            message_type = message.get('type')
            
            if message_type in self.event_handlers:
                await self.event_handlers[message_type](message)
            else:
                logger.warning(f"Unknown message type: {message_type}")
        
        except Exception as e:
            logger.error(f"Error handling message {message.get('type')}: {e}")
    
    async def _handle_model_update(self, message: Dict[str, Any]):
        """Handle global model update from server."""
        try:
            self.status.status = 'updating'
            
            # Extract model parameters
            model_params = message.get('model_parameters')
            round_number = message.get('round_number', 0)
            
            if model_params:
                # Update local model
                self.local_trainer.update_model(model_params)
                self.status.round_number = round_number
                self.status.model_version = message.get('model_version', '1.0.0')
                
                logger.info(f"Model updated for round {round_number}")
            
            self.status.status = 'idle'
        
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            self.status.status = 'error'
            self.status.error_message = str(e)
    
    async def _handle_training_request(self, message: Dict[str, Any]):
        """Handle training request from server."""
        try:
            self.status.status = 'training'
            
            # Extract training parameters
            training_config = message.get('training_config', {})
            round_number = message.get('round_number', 0)
            
            # Perform local training
            training_result = await self._perform_local_training(training_config)
            
            # Apply privacy mechanisms
            if self.config.differential_privacy:
                training_result = self.privacy_engine.apply_differential_privacy(
                    training_result, self.config.privacy_budget
                )
            
            # Send training result back to server
            response_msg = {
                'type': 'training_result',
                'client_id': self.config.client_id,
                'round_number': round_number,
                'model_update': training_result['model_update'],
                'training_metrics': training_result['metrics'],
                'data_samples': training_result['data_samples']
            }
            
            await self._send_message(response_msg)
            
            self.status.status = 'idle'
            self.status.training_accuracy = training_result['metrics'].get('accuracy', 0.0)
            self.status.data_samples = training_result['data_samples']
            
            logger.info(f"Training completed for round {round_number}")
        
        except Exception as e:
            logger.error(f"Error during training: {e}")
            self.status.status = 'error'
            self.status.error_message = str(e)
    
    async def _perform_local_training(self, training_config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform local model training."""
        try:
            # Update training configuration
            epochs = training_config.get('epochs', self.config.local_epochs)
            batch_size = training_config.get('batch_size', self.config.batch_size)
            learning_rate = training_config.get('learning_rate', self.config.learning_rate)
            
            # Train model
            training_result = await asyncio.get_event_loop().run_in_executor(
                None, 
                self.local_trainer.train,
                epochs,
                batch_size,
                learning_rate
            )
            
            return training_result
        
        except Exception as e:
            logger.error(f"Local training failed: {e}")
            raise
    
    async def _handle_aggregation_result(self, message: Dict[str, Any]):
        """Handle aggregation result from server."""
        try:
            aggregation_metrics = message.get('aggregation_metrics', {})
            global_accuracy = aggregation_metrics.get('global_accuracy', 0.0)
            
            logger.info(f"Aggregation completed. Global accuracy: {global_accuracy:.4f}")
        
        except Exception as e:
            logger.error(f"Error handling aggregation result: {e}")
    
    async def _handle_heartbeat(self, message: Dict[str, Any]):
        """Handle heartbeat response from server."""
        self.status.last_seen = datetime.now()
    
    async def _handle_shutdown(self, message: Dict[str, Any]):
        """Handle shutdown command from server."""
        logger.info("Shutdown command received from server")
        self.connected = False

class ClientManager:
    """
    Manager for multiple federated learning clients.
    """
    
    def __init__(self):
        self.clients: Dict[str, FederatedClient] = {}
        self.client_configs: Dict[str, ClientConfig] = {}
        self.running_tasks: Dict[str, asyncio.Task] = {}
        
    def register_client(self, config: ClientConfig, local_trainer: LocalTrainer,
                       privacy_engine: Optional[PrivacyEngine] = None) -> str:
        """
        Register a new federated learning client.
        
        Args:
            config (ClientConfig): Client configuration
            local_trainer (LocalTrainer): Local training implementation
            privacy_engine (PrivacyEngine, optional): Privacy engine
            
        Returns:
            str: Client ID
        """
        client = FederatedClient(config, local_trainer, privacy_engine)
        self.clients[config.client_id] = client
        self.client_configs[config.client_id] = config
        
        logger.info(f"Client {config.client_id} registered")
        return config.client_id
    
    async def start_client(self, client_id: str) -> bool:
        """
        Start a specific client.
        
        Args:
            client_id (str): Client identifier
            
        Returns:
            bool: True if started successfully
        """
        if client_id not in self.clients:
            logger.error(f"Client {client_id} not found")
            return False
        
        try:
            client = self.clients[client_id]
            task = asyncio.create_task(client.run())
            self.running_tasks[client_id] = task
            
            logger.info(f"Client {client_id} started")
            return True
        
        except Exception as e:
            logger.error(f"Failed to start client {client_id}: {e}")
            return False
    
    async def stop_client(self, client_id: str) -> bool:
        """
        Stop a specific client.
        
        Args:
            client_id (str): Client identifier
            
        Returns:
            bool: True if stopped successfully
        """
        if client_id not in self.clients:
            logger.error(f"Client {client_id} not found")
            return False
        
        try:
            # Disconnect client
            client = self.clients[client_id]
            await client.disconnect()
            
            # Cancel running task
            if client_id in self.running_tasks:
                task = self.running_tasks[client_id]
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                del self.running_tasks[client_id]
            
            logger.info(f"Client {client_id} stopped")
            return True
        
        except Exception as e:
            logger.error(f"Failed to stop client {client_id}: {e}")
            return False
    
    async def start_all_clients(self) -> List[str]:
        """
        Start all registered clients.
        
        Returns:
            List[str]: List of successfully started client IDs
        """
        started_clients = []
        
        for client_id in self.clients.keys():
            if await self.start_client(client_id):
                started_clients.append(client_id)
        
        return started_clients
    
    async def stop_all_clients(self) -> List[str]:
        """
        Stop all running clients.
        
        Returns:
            List[str]: List of successfully stopped client IDs
        """
        stopped_clients = []
        
        for client_id in list(self.running_tasks.keys()):
            if await self.stop_client(client_id):
                stopped_clients.append(client_id)
        
        return stopped_clients
    
    def get_client_status(self, client_id: str) -> Optional[ClientStatus]:
        """
        Get status of a specific client.
        
        Args:
            client_id (str): Client identifier
            
        Returns:
            Optional[ClientStatus]: Client status or None if not found
        """
        if client_id in self.clients:
            return self.clients[client_id].status
        return None
    
    def get_all_client_statuses(self) -> Dict[str, ClientStatus]:
        """
        Get status of all registered clients.
        
        Returns:
            Dict[str, ClientStatus]: Dictionary of client statuses
        """
        return {
            client_id: client.status 
            for client_id, client in self.clients.items()
        }
    
    def remove_client(self, client_id: str) -> bool:
        """
        Remove a client from the manager.
        
        Args:
            client_id (str): Client identifier
            
        Returns:
            bool: True if removed successfully
        """
        if client_id not in self.clients:
            return False
        
        # Stop client if running
        if client_id in self.running_tasks:
            asyncio.create_task(self.stop_client(client_id))
        
        # Remove from manager
        del self.clients[client_id]
        del self.client_configs[client_id]
        
        logger.info(f"Client {client_id} removed")
        return True

# Factory functions for easy client creation
def create_federated_client(server_address: str, server_port: int,
                          local_trainer: LocalTrainer,
                          model_type: str = "battery_health",
                          **kwargs) -> FederatedClient:
    """
    Factory function to create a federated learning client.
    
    Args:
        server_address (str): Server address
        server_port (int): Server port
        local_trainer (LocalTrainer): Local training implementation
        model_type (str): Type of model
        **kwargs: Additional configuration parameters
        
    Returns:
        FederatedClient: Configured client instance
    """
    config = ClientConfig(
        server_address=server_address,
        server_port=server_port,
        model_type=model_type,
        **kwargs
    )
    
    privacy_engine = PrivacyEngine(config)
    return FederatedClient(config, local_trainer, privacy_engine)

def create_client_manager() -> ClientManager:
    """
    Factory function to create a client manager.
    
    Returns:
        ClientManager: New client manager instance
    """
    return ClientManager()
