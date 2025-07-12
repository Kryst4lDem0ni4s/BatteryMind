"""
BatteryMind - Federated Learning Client Simulator

Detailed simulation of individual federated learning clients with realistic
battery management system constraints and behaviors.

Features:
- Realistic client behavior simulation
- Resource constraint modeling
- Network condition simulation
- Data heterogeneity handling
- Privacy-preserving training simulation
- Performance profiling and metrics collection

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import json
import threading
import queue
from collections import deque
import psutil
import gc

# Local imports
from ..client_models.client_manager import ClientManager, ClientConfig
from ..privacy_preserving.differential_privacy import DifferentialPrivacyEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ClientSystemProfile:
    """
    System profile for a federated learning client.
    
    Attributes:
        cpu_cores (int): Number of CPU cores
        memory_gb (float): Available memory in GB
        storage_gb (float): Available storage in GB
        network_bandwidth_mbps (float): Network bandwidth in Mbps
        battery_capacity_mah (int): Battery capacity in mAh (for mobile clients)
        is_mobile (bool): Whether client is mobile device
        availability_pattern (str): Client availability pattern
        compute_capability (float): Relative compute capability (0.1 to 2.0)
    """
    cpu_cores: int = 4
    memory_gb: float = 8.0
    storage_gb: float = 64.0
    network_bandwidth_mbps: float = 10.0
    battery_capacity_mah: int = 4000
    is_mobile: bool = False
    availability_pattern: str = "random"  # "random", "periodic", "always_on"
    compute_capability: float = 1.0

@dataclass
class ClientTrainingProfile:
    """
    Training profile for a federated learning client.
    
    Attributes:
        data_quality (float): Quality of local data (0.0 to 1.0)
        label_noise_rate (float): Rate of label noise in local data
        feature_noise_std (float): Standard deviation of feature noise
        data_freshness (float): How recent the data is (0.0 to 1.0)
        domain_shift (float): Amount of domain shift from global distribution
        class_imbalance (float): Degree of class imbalance in local data
        seasonal_variation (bool): Whether data has seasonal patterns
        privacy_sensitivity (float): Privacy sensitivity level (0.0 to 1.0)
    """
    data_quality: float = 1.0
    label_noise_rate: float = 0.0
    feature_noise_std: float = 0.0
    data_freshness: float = 1.0
    domain_shift: float = 0.0
    class_imbalance: float = 0.0
    seasonal_variation: bool = False
    privacy_sensitivity: float = 0.5

@dataclass
class ClientSimulationMetrics:
    """
    Comprehensive metrics for client simulation.
    """
    client_id: str = ""
    round_number: int = 0
    training_time: float = 0.0
    communication_time: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_utilization: float = 0.0
    energy_consumption_mah: float = 0.0
    data_samples_used: int = 0
    local_accuracy: float = 0.0
    local_loss: float = 0.0
    gradient_norm: float = 0.0
    model_divergence: float = 0.0
    privacy_budget_used: float = 0.0
    network_latency: float = 0.0
    packet_loss_rate: float = 0.0
    successful_transmission: bool = True

class ResourceMonitor:
    """
    Monitor system resources during client training.
    """
    
    def __init__(self, client_profile: ClientSystemProfile):
        self.profile = client_profile
        self.monitoring = False
        self.metrics = deque(maxlen=1000)
        self.monitor_thread = None
        
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def stop_monitoring(self) -> Dict[str, float]:
        """Stop monitoring and return average metrics."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join()
        
        if not self.metrics:
            return {
                'avg_memory_usage_mb': 0.0,
                'avg_cpu_utilization': 0.0,
                'peak_memory_usage_mb': 0.0,
                'peak_cpu_utilization': 0.0
            }
        
        memory_values = [m['memory_mb'] for m in self.metrics]
        cpu_values = [m['cpu_percent'] for m in self.metrics]
        
        return {
            'avg_memory_usage_mb': np.mean(memory_values),
            'avg_cpu_utilization': np.mean(cpu_values),
            'peak_memory_usage_mb': np.max(memory_values),
            'peak_cpu_utilization': np.max(cpu_values)
        }
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop."""
        while self.monitoring:
            try:
                # Get current resource usage
                memory_info = psutil.virtual_memory()
                cpu_percent = psutil.cpu_percent()
                
                # Scale based on client profile
                memory_mb = (memory_info.used / (1024 * 1024)) * (self.profile.memory_gb / 16.0)
                cpu_scaled = cpu_percent * self.profile.compute_capability
                
                self.metrics.append({
                    'timestamp': time.time(),
                    'memory_mb': memory_mb,
                    'cpu_percent': cpu_scaled
                })
                
                time.sleep(0.1)  # Monitor every 100ms
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                break

class NetworkSimulator:
    """
    Simulate network conditions for federated learning client.
    """
    
    def __init__(self, client_profile: ClientSystemProfile):
        self.profile = client_profile
        self.rng = np.random.RandomState()
        
    def simulate_download(self, data_size_mb: float) -> Tuple[float, bool]:
        """
        Simulate model download from server.
        
        Args:
            data_size_mb (float): Size of data to download in MB
            
        Returns:
            Tuple[float, bool]: (download_time, success)
        """
        # Base download time
        bandwidth = self.profile.network_bandwidth_mbps
        base_time = data_size_mb * 8 / bandwidth  # Convert MB to Mb
        
        # Add network variability
        latency = self.rng.exponential(0.1)  # 100ms average latency
        jitter = self.rng.normal(0, 0.02)    # 20ms jitter
        
        # Simulate packet loss
        packet_loss_prob = 0.01 if not self.profile.is_mobile else 0.05
        packet_loss = self.rng.random() < packet_loss_prob
        
        if packet_loss:
            # Simulate retransmission
            base_time *= 1.5
            success = self.rng.random() > 0.1  # 90% success after retry
        else:
            success = True
        
        total_time = base_time + latency + abs(jitter)
        return total_time, success
    
    def simulate_upload(self, data_size_mb: float) -> Tuple[float, bool]:
        """
        Simulate model upload to server.
        
        Args:
            data_size_mb (float): Size of data to upload in MB
            
        Returns:
            Tuple[float, bool]: (upload_time, success)
        """
        # Upload typically slower than download
        upload_bandwidth = self.profile.network_bandwidth_mbps * 0.3
        base_time = data_size_mb * 8 / upload_bandwidth
        
        # Add network variability
        latency = self.rng.exponential(0.15)  # Higher latency for upload
        jitter = self.rng.normal(0, 0.03)
        
        # Simulate connection drops (more common on mobile)
        drop_prob = 0.02 if not self.profile.is_mobile else 0.1
        connection_drop = self.rng.random() < drop_prob
        
        if connection_drop:
            # Simulate reconnection and retry
            base_time *= 2.0
            success = self.rng.random() > 0.2  # 80% success after retry
        else:
            success = True
        
        total_time = base_time + latency + abs(jitter)
        return total_time, success

class DataSimulator:
    """
    Simulate realistic data conditions for federated learning clients.
    """
    
    def __init__(self, training_profile: ClientTrainingProfile):
        self.profile = training_profile
        self.rng = np.random.RandomState()
    
    def add_noise_to_data(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Add realistic noise to training data.
        
        Args:
            data (np.ndarray): Original training data
            labels (np.ndarray): Original labels
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Noisy data and labels
        """
        noisy_data = data.copy()
        noisy_labels = labels.copy()
        
        # Add feature noise
        if self.profile.feature_noise_std > 0:
            noise = self.rng.normal(0, self.profile.feature_noise_std, data.shape)
            noisy_data += noise
        
        # Add label noise
        if self.profile.label_noise_rate > 0:
            num_samples = len(labels)
            noise_indices = self.rng.choice(
                num_samples,
                int(num_samples * self.profile.label_noise_rate),
                replace=False
            )
            
            # Flip labels randomly
            unique_labels = np.unique(labels)
            for idx in noise_indices:
                current_label = labels[idx]
                other_labels = [l for l in unique_labels if l != current_label]
                if other_labels:
                    noisy_labels[idx] = self.rng.choice(other_labels)
        
        # Simulate data quality degradation
        if self.profile.data_quality < 1.0:
            # Randomly corrupt some features
            corruption_rate = 1.0 - self.profile.data_quality
            num_corrupted = int(data.size * corruption_rate)
            
            flat_data = noisy_data.flatten()
            corrupt_indices = self.rng.choice(len(flat_data), num_corrupted, replace=False)
            
            # Set corrupted values to random values from the data distribution
            for idx in corrupt_indices:
                flat_data[idx] = self.rng.choice(flat_data)
            
            noisy_data = flat_data.reshape(data.shape)
        
        return noisy_data, noisy_labels
    
    def simulate_domain_shift(self, data: np.ndarray) -> np.ndarray:
        """Simulate domain shift in client data."""
        if self.profile.domain_shift == 0:
            return data
        
        shifted_data = data.copy()
        
        # Apply systematic shift
        shift_magnitude = self.profile.domain_shift * np.std(data, axis=0)
        shift_direction = self.rng.normal(0, 1, data.shape[1])
        shift_direction = shift_direction / np.linalg.norm(shift_direction)
        
        shift = shift_magnitude * shift_direction
        shifted_data += shift
        
        return shifted_data
    
    def simulate_class_imbalance(self, data: np.ndarray, labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate class imbalance in client data."""
        if self.profile.class_imbalance == 0:
            return data, labels
        
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return data, labels
        
        # Create imbalanced distribution
        num_classes = len(unique_labels)
        class_probs = np.ones(num_classes)
        
        # Make one class dominant
        dominant_class = self.rng.choice(num_classes)
        class_probs[dominant_class] *= (1 + self.profile.class_imbalance * 10)
        class_probs = class_probs / np.sum(class_probs)
        
        # Sample according to imbalanced distribution
        target_samples = int(len(data) * 0.8)  # Keep 80% of data
        selected_indices = []
        
        for class_idx, label in enumerate(unique_labels):
            class_indices = np.where(labels == label)[0]
            num_samples = int(target_samples * class_probs[class_idx])
            num_samples = min(num_samples, len(class_indices))
            
            if num_samples > 0:
                selected = self.rng.choice(class_indices, num_samples, replace=False)
                selected_indices.extend(selected)
        
        if selected_indices:
            return data[selected_indices], labels[selected_indices]
        else:
            return data, labels

class FederatedClientSimulator:
    """
    Comprehensive simulator for federated learning clients.
    """
    
    def __init__(self, client_id: str, client_config: ClientConfig,
                 system_profile: ClientSystemProfile,
                 training_profile: ClientTrainingProfile):
        self.client_id = client_id
        self.client_config = client_config
        self.system_profile = system_profile
        self.training_profile = training_profile
        
        # Initialize components
        self.resource_monitor = ResourceMonitor(system_profile)
        self.network_simulator = NetworkSimulator(system_profile)
        self.data_simulator = DataSimulator(training_profile)
        
        # Client state
        self.local_model = None
        self.local_data = None
        self.local_labels = None
        self.is_available = True
        self.current_round = 0
        
        # Privacy engine
        if training_profile.privacy_sensitivity > 0:
            self.privacy_engine = DifferentialPrivacyEngine(
                epsilon=training_profile.privacy_sensitivity,
                delta=1e-5
            )
        else:
            self.privacy_engine = None
        
        # Metrics tracking
        self.simulation_metrics = []
        
        logger.info(f"Client {client_id} simulator initialized")
    
    def set_data(self, data: np.ndarray, labels: np.ndarray) -> None:
        """Set local training data for the client."""
        # Apply data simulation effects
        noisy_data, noisy_labels = self.data_simulator.add_noise_to_data(data, labels)
        shifted_data = self.data_simulator.simulate_domain_shift(noisy_data)
        final_data, final_labels = self.data_simulator.simulate_class_imbalance(
            shifted_data, noisy_labels
        )
        
        self.local_data = final_data
        self.local_labels = final_labels
        
        logger.info(f"Client {self.client_id} received {len(final_data)} training samples")
    
    def set_model(self, model: nn.Module) -> None:
        """Set local model for the client."""
        self.local_model = model
    
    def check_availability(self) -> bool:
        """Check if client is available for training."""
        if self.system_profile.availability_pattern == "always_on":
            return True
        elif self.system_profile.availability_pattern == "periodic":
            # Simulate periodic availability (e.g., business hours)
            hour = (time.time() / 3600) % 24
            return 8 <= hour <= 18  # Available during business hours
        else:  # random
            # Random availability based on system profile
            availability_prob = 0.9 if not self.system_profile.is_mobile else 0.7
            return np.random.random() < availability_prob
    
    def simulate_training_round(self, global_model_state: Dict,
                              round_number: int) -> Optional[ClientSimulationMetrics]:
        """
        Simulate a complete training round for the client.
        
        Args:
            global_model_state (Dict): Global model parameters
            round_number (int): Current federated learning round
            
        Returns:
            Optional[ClientSimulationMetrics]: Training metrics or None if unavailable
        """
        self.current_round = round_number
        
        # Check availability
        if not self.check_availability():
            logger.info(f"Client {self.client_id} unavailable for round {round_number}")
            return None
        
        metrics = ClientSimulationMetrics(
            client_id=self.client_id,
            round_number=round_number
        )
        
        round_start_time = time.time()
        
        try:
            # Start resource monitoring
            self.resource_monitor.start_monitoring()
            
            # Simulate model download
            model_size_mb = self._estimate_model_size(global_model_state)
            download_time, download_success = self.network_simulator.simulate_download(model_size_mb)
            
            if not download_success:
                metrics.successful_transmission = False
                metrics.communication_time = download_time
                return metrics
            
            # Update local model
            self.local_model.load_state_dict(global_model_state)
            
            # Perform local training
            training_start_time = time.time()
            training_results = self._simulate_local_training()
            training_time = time.time() - training_start_time
            
            # Simulate model upload
            upload_time, upload_success = self.network_simulator.simulate_upload(model_size_mb)
            
            if not upload_success:
                metrics.successful_transmission = False
                metrics.communication_time = download_time + upload_time
                return metrics
            
            # Stop resource monitoring
            resource_metrics = self.resource_monitor.stop_monitoring()
            
            # Populate metrics
            metrics.training_time = training_time
            metrics.communication_time = download_time + upload_time
            metrics.memory_usage_mb = resource_metrics['avg_memory_usage_mb']
            metrics.cpu_utilization = resource_metrics['avg_cpu_utilization']
            metrics.data_samples_used = len(self.local_data) if self.local_data is not None else 0
            metrics.local_accuracy = training_results['accuracy']
            metrics.local_loss = training_results['loss']
            metrics.gradient_norm = training_results['gradient_norm']
            metrics.successful_transmission = True
            
            # Simulate energy consumption (for mobile clients)
            if self.system_profile.is_mobile:
                base_consumption = training_time * 100  # 100 mAh per second base
                compute_consumption = metrics.cpu_utilization * 50
                network_consumption = metrics.communication_time * 200
                metrics.energy_consumption_mah = base_consumption + compute_consumption + network_consumption
            
            # Apply privacy if enabled
            if self.privacy_engine:
                training_results['model_update'] = self.privacy_engine.add_noise(
                    training_results['model_update']
                )
                metrics.privacy_budget_used = self.privacy_engine.get_consumed_budget()
            
            self.simulation_metrics.append(metrics)
            
            logger.info(f"Client {self.client_id} completed round {round_number} "
                       f"(accuracy: {metrics.local_accuracy:.4f}, "
                       f"time: {metrics.training_time:.2f}s)")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Client {self.client_id} training failed: {e}")
            metrics.successful_transmission = False
            return metrics
        
        finally:
            # Ensure resource monitoring is stopped
            try:
                self.resource_monitor.stop_monitoring()
            except:
                pass
    
    def _simulate_local_training(self) -> Dict[str, Any]:
        """Simulate local model training."""
        if self.local_data is None or self.local_model is None:
            raise ValueError("Local data or model not set")
        
        # Create data loader
        dataset = TensorDataset(
            torch.FloatTensor(self.local_data),
            torch.LongTensor(self.local_labels)
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.client_config.batch_size,
            shuffle=True
        )
        
        # Setup optimizer
        optimizer = optim.SGD(
            self.local_model.parameters(),
            lr=self.client_config.learning_rate
        )
        
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.local_model.train()
        total_loss = 0.0
        total_samples = 0
        gradient_norms = []
        
        for epoch in range(self.client_config.local_epochs):
            for batch_data, batch_labels in dataloader:
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.local_model(batch_data)
                loss = criterion(outputs, batch_labels)
                
                # Backward pass
                loss.backward()
                
                # Calculate gradient norm
                grad_norm = torch.norm(torch.stack([
                    torch.norm(p.grad.detach()) for p in self.local_model.parameters()
                    if p.grad is not None
                ]))
                gradient_norms.append(grad_norm.item())
                
                optimizer.step()
                
                total_loss += loss.item() * len(batch_data)
                total_samples += len(batch_data)
        
        # Evaluate local model
        self.local_model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                outputs = self.local_model(batch_data)
                _, predicted = torch.max(outputs.data, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
        
        accuracy = correct / total if total > 0 else 0.0
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        avg_grad_norm = np.mean(gradient_norms) if gradient_norms else 0.0
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'gradient_norm': avg_grad_norm,
            'model_update': self.local_model.state_dict(),
            'num_samples': total_samples
        }
    
    def _estimate_model_size(self, model_state: Dict) -> float:
        """Estimate model size in MB."""
        total_params = 0
        for param in model_state.values():
            if hasattr(param, 'numel'):
                total_params += param.numel()
        
        # Assume 4 bytes per parameter (float32)
        size_mb = total_params * 4 / (1024 * 1024)
        return size_mb
    
    def get_client_summary(self) -> Dict[str, Any]:
        """Get comprehensive client simulation summary."""
        if not self.simulation_metrics:
            return {
                'client_id': self.client_id,
                'total_rounds': 0,
                'system_profile': self.system_profile.__dict__,
                'training_profile': self.training_profile.__dict__
            }
        
        successful_rounds = [m for m in self.simulation_metrics if m.successful_transmission]
        
        return {
            'client_id': self.client_id,
            'total_rounds': len(self.simulation_metrics),
            'successful_rounds': len(successful_rounds),
            'success_rate': len(successful_rounds) / len(self.simulation_metrics),
            'average_accuracy': np.mean([m.local_accuracy for m in successful_rounds]) if successful_rounds else 0,
            'average_training_time': np.mean([m.training_time for m in successful_rounds]) if successful_rounds else 0,
            'average_communication_time': np.mean([m.communication_time for m in self.simulation_metrics]),
            'total_energy_consumption': sum(m.energy_consumption_mah for m in self.simulation_metrics),
            'total_privacy_budget_used': sum(m.privacy_budget_used for m in self.simulation_metrics),
            'system_profile': self.system_profile.__dict__,
            'training_profile': self.training_profile.__dict__,
            'metrics_history': [m.__dict__ for m in self.simulation_metrics]
        }

# Factory functions for creating client simulators
def create_client_simulator(client_id: str, client_config: ClientConfig,
                           system_profile: Optional[ClientSystemProfile] = None,
                           training_profile: Optional[ClientTrainingProfile] = None) -> FederatedClientSimulator:
    """
    Factory function to create a federated client simulator.
    
    Args:
        client_id (str): Unique client identifier
        client_config (ClientConfig): Client training configuration
        system_profile (ClientSystemProfile, optional): System resource profile
        training_profile (ClientTrainingProfile, optional): Training data profile
        
    Returns:
        FederatedClientSimulator: Configured client simulator
    """
    if system_profile is None:
        system_profile = ClientSystemProfile()
    
    if training_profile is None:
        training_profile = ClientTrainingProfile()
    
    return FederatedClientSimulator(client_id, client_config, system_profile, training_profile)

def create_heterogeneous_clients(num_clients: int, base_config: ClientConfig) -> List[FederatedClientSimulator]:
    """
    Create a diverse set of heterogeneous client simulators.
    
    Args:
        num_clients (int): Number of clients to create
        base_config (ClientConfig): Base client configuration
        
    Returns:
        List[FederatedClientSimulator]: List of diverse client simulators
    """
    clients = []
    
    for i in range(num_clients):
        client_id = f"client_{i:03d}"
        
        # Create varied system profiles
        system_profile = ClientSystemProfile(
            cpu_cores=np.random.choice([2, 4, 8, 16]),
            memory_gb=np.random.choice([4, 8, 16, 32]),
            network_bandwidth_mbps=np.random.uniform(1, 100),
            is_mobile=np.random.random() < 0.3,  # 30% mobile clients
            compute_capability=np.random.uniform(0.5, 2.0),
            availability_pattern=np.random.choice(["random", "periodic", "always_on"])
        )
        
        # Create varied training profiles
        training_profile = ClientTrainingProfile(
            data_quality=np.random.uniform(0.7, 1.0),
            label_noise_rate=np.random.uniform(0.0, 0.1),
            feature_noise_std=np.random.uniform(0.0, 0.05),
            domain_shift=np.random.uniform(0.0, 0.3),
            class_imbalance=np.random.uniform(0.0, 0.5),
            privacy_sensitivity=np.random.uniform(0.1, 1.0)
        )
        
        # Vary client configuration
        client_config = ClientConfig(
            client_id=client_id,
            learning_rate=base_config.learning_rate * np.random.uniform(0.5, 2.0),
            batch_size=max(8, int(base_config.batch_size * np.random.uniform(0.5, 2.0))),
            local_epochs=max(1, int(base_config.local_epochs * np.random.uniform(0.5, 2.0)))
        )
        
        client = create_client_simulator(client_id, client_config, system_profile, training_profile)
        clients.append(client)
    
    return clients
