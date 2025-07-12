"""
BatteryMind - Federated Learning Simulator

Comprehensive simulation framework for federated learning in battery management
systems. Provides realistic simulation of distributed battery networks with
privacy-preserving learning capabilities.

Features:
- Multi-client federated learning simulation
- Realistic network conditions and heterogeneity
- Privacy-preserving aggregation simulation
- Performance evaluation and metrics collection
- Integration with battery health prediction models
- Scalable simulation for large battery fleets

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
import pickle
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import threading
from collections import defaultdict, deque
import random
import copy

# Scientific computing imports
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score

# Federated learning imports
from ..server.federated_server import FederatedServer, FederationConfig
from ..client_models.client_manager import ClientManager, ClientConfig
from ..privacy_preserving.differential_privacy import DifferentialPrivacyEngine
from ..privacy_preserving.secure_aggregation import SecureAggregationProtocol

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SimulationConfig:
    """
    Configuration for federated learning simulation.
    
    Attributes:
        # Simulation parameters
        num_clients (int): Number of federated clients
        num_rounds (int): Number of federated learning rounds
        clients_per_round (int): Number of clients participating per round
        simulation_duration (float): Maximum simulation duration in seconds
        
        # Client heterogeneity
        data_heterogeneity (str): Type of data heterogeneity ('iid', 'non_iid', 'extreme')
        system_heterogeneity (bool): Enable system heterogeneity simulation
        client_availability (float): Average client availability probability
        
        # Network simulation
        network_latency_mean (float): Mean network latency in seconds
        network_latency_std (float): Standard deviation of network latency
        bandwidth_mbps (float): Available bandwidth in Mbps
        packet_loss_rate (float): Network packet loss rate
        
        # Privacy settings
        enable_differential_privacy (bool): Enable differential privacy
        privacy_budget (float): Total privacy budget
        enable_secure_aggregation (bool): Enable secure aggregation
        
        # Performance settings
        target_accuracy (float): Target model accuracy for early stopping
        convergence_threshold (float): Convergence threshold for stopping
        evaluation_frequency (int): Frequency of model evaluation
        
        # Resource constraints
        max_memory_per_client (int): Maximum memory per client in MB
        max_compute_per_client (float): Maximum compute per client in FLOPS
        enable_resource_constraints (bool): Enable resource constraint simulation
        
        # Data distribution
        min_samples_per_client (int): Minimum samples per client
        max_samples_per_client (int): Maximum samples per client
        label_distribution_alpha (float): Dirichlet alpha for label distribution
    """
    # Simulation parameters
    num_clients: int = 100
    num_rounds: int = 100
    clients_per_round: int = 10
    simulation_duration: float = 3600.0  # 1 hour
    
    # Client heterogeneity
    data_heterogeneity: str = "non_iid"
    system_heterogeneity: bool = True
    client_availability: float = 0.8
    
    # Network simulation
    network_latency_mean: float = 0.1  # 100ms
    network_latency_std: float = 0.05   # 50ms std
    bandwidth_mbps: float = 10.0
    packet_loss_rate: float = 0.01
    
    # Privacy settings
    enable_differential_privacy: bool = True
    privacy_budget: float = 1.0
    enable_secure_aggregation: bool = True
    
    # Performance settings
    target_accuracy: float = 0.95
    convergence_threshold: float = 0.001
    evaluation_frequency: int = 5
    
    # Resource constraints
    max_memory_per_client: int = 512  # MB
    max_compute_per_client: float = 1e9  # FLOPS
    enable_resource_constraints: bool = True
    
    # Data distribution
    min_samples_per_client: int = 100
    max_samples_per_client: int = 1000
    label_distribution_alpha: float = 0.5

@dataclass
class SimulationMetrics:
    """
    Comprehensive metrics for federated learning simulation.
    """
    round_number: int = 0
    global_accuracy: float = 0.0
    global_loss: float = 0.0
    convergence_rate: float = 0.0
    communication_cost: float = 0.0
    computation_time: float = 0.0
    privacy_budget_consumed: float = 0.0
    client_participation_rate: float = 0.0
    model_divergence: float = 0.0
    fairness_score: float = 0.0
    robustness_score: float = 0.0

class NetworkSimulator:
    """
    Network condition simulator for federated learning.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.RandomState(42)
        
    def simulate_latency(self) -> float:
        """Simulate network latency."""
        latency = self.rng.normal(
            self.config.network_latency_mean,
            self.config.network_latency_std
        )
        return max(0.001, latency)  # Minimum 1ms latency
    
    def simulate_bandwidth(self) -> float:
        """Simulate available bandwidth."""
        # Simulate bandwidth variation (80-120% of nominal)
        variation = self.rng.uniform(0.8, 1.2)
        return self.config.bandwidth_mbps * variation
    
    def simulate_packet_loss(self, data_size_mb: float) -> bool:
        """Simulate packet loss."""
        # Higher probability of loss for larger data
        loss_prob = self.config.packet_loss_rate * (1 + data_size_mb / 10)
        return self.rng.random() < loss_prob
    
    def calculate_transmission_time(self, data_size_mb: float) -> float:
        """Calculate data transmission time."""
        bandwidth = self.simulate_bandwidth()
        base_time = data_size_mb * 8 / bandwidth  # Convert MB to Mb and divide by Mbps
        
        # Add latency
        latency = self.simulate_latency()
        
        # Simulate retransmissions due to packet loss
        retransmissions = 0
        while self.simulate_packet_loss(data_size_mb) and retransmissions < 3:
            retransmissions += 1
        
        total_time = base_time * (1 + retransmissions) + latency
        return total_time

class DataDistributor:
    """
    Data distribution simulator for federated learning heterogeneity.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.rng = np.random.RandomState(42)
    
    def create_iid_distribution(self, data: np.ndarray, labels: np.ndarray,
                               num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create IID data distribution across clients."""
        total_samples = len(data)
        samples_per_client = total_samples // num_clients
        
        # Shuffle data
        indices = self.rng.permutation(total_samples)
        
        client_data = []
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = (i + 1) * samples_per_client if i < num_clients - 1 else total_samples
            
            client_indices = indices[start_idx:end_idx]
            client_x = data[client_indices]
            client_y = labels[client_indices]
            
            client_data.append((client_x, client_y))
        
        return client_data
    
    def create_non_iid_distribution(self, data: np.ndarray, labels: np.ndarray,
                                   num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create non-IID data distribution using Dirichlet distribution."""
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        # Generate Dirichlet distribution for each client
        label_distributions = self.rng.dirichlet(
            [self.config.label_distribution_alpha] * num_classes,
            num_clients
        )
        
        # Calculate number of samples per client
        total_samples = len(data)
        samples_per_client = self.rng.randint(
            self.config.min_samples_per_client,
            self.config.max_samples_per_client,
            num_clients
        )
        
        # Normalize to match total samples
        samples_per_client = (samples_per_client * total_samples / samples_per_client.sum()).astype(int)
        
        client_data = []
        used_indices = set()
        
        for client_idx in range(num_clients):
            client_samples = samples_per_client[client_idx]
            client_label_dist = label_distributions[client_idx]
            
            # Determine number of samples per class for this client
            samples_per_class = (client_label_dist * client_samples).astype(int)
            
            client_indices = []
            for class_idx, class_label in enumerate(unique_labels):
                class_mask = (labels == class_label)
                available_indices = np.where(class_mask)[0]
                available_indices = [idx for idx in available_indices if idx not in used_indices]
                
                num_samples = min(samples_per_class[class_idx], len(available_indices))
                if num_samples > 0:
                    selected_indices = self.rng.choice(
                        available_indices, num_samples, replace=False
                    )
                    client_indices.extend(selected_indices)
                    used_indices.update(selected_indices)
            
            if client_indices:
                client_x = data[client_indices]
                client_y = labels[client_indices]
                client_data.append((client_x, client_y))
            else:
                # Fallback: assign random samples
                remaining_indices = [i for i in range(len(data)) if i not in used_indices]
                if remaining_indices:
                    fallback_size = min(10, len(remaining_indices))
                    fallback_indices = self.rng.choice(remaining_indices, fallback_size, replace=False)
                    client_data.append((data[fallback_indices], labels[fallback_indices]))
                    used_indices.update(fallback_indices)
        
        return client_data
    
    def create_extreme_non_iid_distribution(self, data: np.ndarray, labels: np.ndarray,
                                          num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Create extreme non-IID distribution (each client has only 1-2 classes)."""
        unique_labels = np.unique(labels)
        num_classes = len(unique_labels)
        
        client_data = []
        
        # Assign 1-2 classes per client
        for client_idx in range(num_clients):
            num_classes_per_client = self.rng.choice([1, 2], p=[0.7, 0.3])
            client_classes = self.rng.choice(
                unique_labels, num_classes_per_client, replace=False
            )
            
            # Get samples for assigned classes
            client_indices = []
            for class_label in client_classes:
                class_indices = np.where(labels == class_label)[0]
                # Take a random subset
                subset_size = min(
                    len(class_indices),
                    self.rng.randint(50, 200)
                )
                selected_indices = self.rng.choice(
                    class_indices, subset_size, replace=False
                )
                client_indices.extend(selected_indices)
            
            if client_indices:
                client_x = data[client_indices]
                client_y = labels[client_indices]
                client_data.append((client_x, client_y))
        
        return client_data
    
    def distribute_data(self, data: np.ndarray, labels: np.ndarray,
                       num_clients: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Distribute data according to configured heterogeneity."""
        if self.config.data_heterogeneity == "iid":
            return self.create_iid_distribution(data, labels, num_clients)
        elif self.config.data_heterogeneity == "non_iid":
            return self.create_non_iid_distribution(data, labels, num_clients)
        elif self.config.data_heterogeneity == "extreme":
            return self.create_extreme_non_iid_distribution(data, labels, num_clients)
        else:
            raise ValueError(f"Unknown data heterogeneity: {self.config.data_heterogeneity}")

class FederatedLearningSimulator:
    """
    Main federated learning simulator for battery management systems.
    """
    
    def __init__(self, config: SimulationConfig, model_factory: Callable,
                 global_test_data: Tuple[np.ndarray, np.ndarray]):
        self.config = config
        self.model_factory = model_factory
        self.global_test_data = global_test_data
        
        # Initialize components
        self.network_simulator = NetworkSimulator(config)
        self.data_distributor = DataDistributor(config)
        
        # Initialize federated server
        federation_config = FederationConfig(
            num_clients=config.num_clients,
            min_clients_per_round=config.clients_per_round,
            max_rounds=config.num_rounds
        )
        self.server = FederatedServer(federation_config)
        
        # Privacy components
        if config.enable_differential_privacy:
            self.privacy_engine = DifferentialPrivacyEngine(
                epsilon=config.privacy_budget,
                delta=1e-5
            )
        
        if config.enable_secure_aggregation:
            self.secure_aggregation = SecureAggregationProtocol(config.num_clients)
        
        # Simulation state
        self.clients = {}
        self.simulation_metrics = []
        self.current_round = 0
        self.start_time = None
        self.converged = False
        
        # Thread pool for parallel client simulation
        self.executor = ThreadPoolExecutor(max_workers=min(config.num_clients, 20))
        
        logger.info(f"Federated Learning Simulator initialized with {config.num_clients} clients")
    
    def setup_clients(self, training_data: List[Tuple[np.ndarray, np.ndarray]]) -> None:
        """Setup federated clients with distributed data."""
        logger.info("Setting up federated clients...")
        
        for client_id in range(self.config.num_clients):
            # Create client configuration
            client_config = ClientConfig(
                client_id=f"client_{client_id}",
                learning_rate=0.01,
                batch_size=32,
                local_epochs=5
            )
            
            # Add system heterogeneity
            if self.config.system_heterogeneity:
                # Vary computational capabilities
                compute_factor = np.random.uniform(0.5, 2.0)
                client_config.local_epochs = max(1, int(client_config.local_epochs * compute_factor))
                client_config.batch_size = max(8, int(client_config.batch_size * compute_factor))
            
            # Create client with data
            if client_id < len(training_data):
                client_data = training_data[client_id]
                client = ClientManager(client_config, self.model_factory())
                client.set_data(client_data[0], client_data[1])
                self.clients[client_id] = client
            
        logger.info(f"Setup {len(self.clients)} clients successfully")
    
    def select_clients(self) -> List[int]:
        """Select clients for current round based on availability."""
        available_clients = []
        
        for client_id in self.clients.keys():
            # Simulate client availability
            if np.random.random() < self.config.client_availability:
                available_clients.append(client_id)
        
        # Select subset for this round
        num_selected = min(self.config.clients_per_round, len(available_clients))
        selected_clients = np.random.choice(
            available_clients, num_selected, replace=False
        ).tolist()
        
        return selected_clients
    
    def simulate_client_training(self, client_id: int, global_model_state: Dict) -> Dict:
        """Simulate training on a single client."""
        client = self.clients[client_id]
        
        # Simulate network delay for model download
        model_size_mb = self._estimate_model_size(global_model_state)
        download_time = self.network_simulator.calculate_transmission_time(model_size_mb)
        
        # Simulate training
        start_time = time.time()
        
        # Update client model with global state
        client.update_model(global_model_state)
        
        # Perform local training
        training_result = client.train()
        
        training_time = time.time() - start_time
        
        # Simulate network delay for model upload
        upload_time = self.network_simulator.calculate_transmission_time(model_size_mb)
        
        # Add privacy noise if enabled
        if self.config.enable_differential_privacy:
            training_result['model_update'] = self.privacy_engine.add_noise(
                training_result['model_update']
            )
        
        return {
            'client_id': client_id,
            'model_update': training_result['model_update'],
            'num_samples': training_result['num_samples'],
            'training_loss': training_result['loss'],
            'training_time': training_time,
            'communication_time': download_time + upload_time
        }
    
    def aggregate_updates(self, client_updates: List[Dict]) -> Dict:
        """Aggregate client updates into global model."""
        if self.config.enable_secure_aggregation:
            # Use secure aggregation protocol
            encrypted_updates = []
            for update in client_updates:
                encrypted_update = self.secure_aggregation.encrypt_update(
                    update['model_update'], update['client_id']
                )
                encrypted_updates.append(encrypted_update)
            
            aggregated_update = self.secure_aggregation.aggregate_updates(encrypted_updates)
        else:
            # Standard federated averaging
            total_samples = sum(update['num_samples'] for update in client_updates)
            
            aggregated_update = {}
            for key in client_updates[0]['model_update'].keys():
                weighted_sum = sum(
                    update['model_update'][key] * update['num_samples']
                    for update in client_updates
                )
                aggregated_update[key] = weighted_sum / total_samples
        
        return aggregated_update
    
    def evaluate_global_model(self) -> Dict[str, float]:
        """Evaluate global model on test data."""
        # Create a temporary model for evaluation
        eval_model = self.model_factory()
        eval_model.load_state_dict(self.server.get_global_model())
        eval_model.eval()
        
        test_x, test_y = self.global_test_data
        
        with torch.no_grad():
            if isinstance(test_x, np.ndarray):
                test_x = torch.FloatTensor(test_x)
            if isinstance(test_y, np.ndarray):
                test_y = torch.LongTensor(test_y)
            
            predictions = eval_model(test_x)
            
            # Calculate metrics
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Classification
                predicted_classes = torch.argmax(predictions, dim=1)
                accuracy = accuracy_score(test_y.numpy(), predicted_classes.numpy())
                loss = nn.CrossEntropyLoss()(predictions, test_y).item()
            else:
                # Regression
                loss = nn.MSELoss()(predictions.squeeze(), test_y.float()).item()
                accuracy = 1.0 / (1.0 + loss)  # Pseudo-accuracy for regression
        
        return {
            'accuracy': accuracy,
            'loss': loss
        }
    
    def calculate_convergence_metrics(self) -> float:
        """Calculate convergence rate based on recent performance."""
        if len(self.simulation_metrics) < 2:
            return 0.0
        
        recent_losses = [m.global_loss for m in self.simulation_metrics[-5:]]
        if len(recent_losses) < 2:
            return 0.0
        
        # Calculate rate of change in loss
        loss_changes = [abs(recent_losses[i] - recent_losses[i-1]) 
                       for i in range(1, len(recent_losses))]
        
        return np.mean(loss_changes)
    
    def calculate_fairness_score(self, client_updates: List[Dict]) -> float:
        """Calculate fairness score based on client performance variation."""
        if not client_updates:
            return 1.0
        
        client_losses = [update['training_loss'] for update in client_updates]
        if len(client_losses) < 2:
            return 1.0
        
        # Use coefficient of variation as fairness metric
        mean_loss = np.mean(client_losses)
        std_loss = np.std(client_losses)
        
        if mean_loss == 0:
            return 1.0
        
        cv = std_loss / mean_loss
        fairness = 1.0 / (1.0 + cv)  # Higher fairness for lower variation
        
        return fairness
    
    def run_simulation(self, training_data: List[Tuple[np.ndarray, np.ndarray]]) -> List[SimulationMetrics]:
        """Run complete federated learning simulation."""
        logger.info("Starting federated learning simulation...")
        self.start_time = time.time()
        
        # Setup clients with distributed data
        self.setup_clients(training_data)
        
        # Initialize global model
        global_model = self.model_factory()
        self.server.initialize_global_model(global_model.state_dict())
        
        # Main simulation loop
        for round_num in range(self.config.num_rounds):
            self.current_round = round_num
            round_start_time = time.time()
            
            logger.info(f"Starting round {round_num + 1}/{self.config.num_rounds}")
            
            # Select clients for this round
            selected_clients = self.select_clients()
            
            if not selected_clients:
                logger.warning(f"No clients available for round {round_num}")
                continue
            
            # Get current global model state
            global_model_state = self.server.get_global_model()
            
            # Simulate parallel client training
            client_futures = []
            for client_id in selected_clients:
                future = self.executor.submit(
                    self.simulate_client_training, client_id, global_model_state
                )
                client_futures.append(future)
            
            # Collect client updates
            client_updates = []
            total_communication_time = 0
            
            for future in client_futures:
                try:
                    update = future.result(timeout=60)  # 60 second timeout
                    client_updates.append(update)
                    total_communication_time += update['communication_time']
                except Exception as e:
                    logger.error(f"Client training failed: {e}")
            
            if not client_updates:
                logger.warning(f"No successful client updates in round {round_num}")
                continue
            
            # Aggregate updates
            aggregated_update = self.aggregate_updates(client_updates)
            
            # Update global model
            self.server.update_global_model(aggregated_update)
            
            # Evaluate global model
            if round_num % self.config.evaluation_frequency == 0:
                evaluation_results = self.evaluate_global_model()
                
                # Calculate additional metrics
                convergence_rate = self.calculate_convergence_metrics()
                fairness_score = self.calculate_fairness_score(client_updates)
                
                # Create round metrics
                round_metrics = SimulationMetrics(
                    round_number=round_num,
                    global_accuracy=evaluation_results['accuracy'],
                    global_loss=evaluation_results['loss'],
                    convergence_rate=convergence_rate,
                    communication_cost=total_communication_time,
                    computation_time=time.time() - round_start_time,
                    privacy_budget_consumed=self.privacy_engine.get_consumed_budget() if self.config.enable_differential_privacy else 0.0,
                    client_participation_rate=len(selected_clients) / self.config.num_clients,
                    fairness_score=fairness_score
                )
                
                self.simulation_metrics.append(round_metrics)
                
                logger.info(f"Round {round_num}: Accuracy={evaluation_results['accuracy']:.4f}, "
                           f"Loss={evaluation_results['loss']:.4f}, "
                           f"Clients={len(selected_clients)}")
                
                # Check convergence
                if (evaluation_results['accuracy'] >= self.config.target_accuracy or
                    convergence_rate < self.config.convergence_threshold):
                    logger.info(f"Convergence achieved at round {round_num}")
                    self.converged = True
                    break
            
            # Check simulation time limit
            if time.time() - self.start_time > self.config.simulation_duration:
                logger.info("Simulation time limit reached")
                break
        
        total_time = time.time() - self.start_time
        logger.info(f"Simulation completed in {total_time:.2f} seconds")
        
        return self.simulation_metrics
    
    def _estimate_model_size(self, model_state: Dict) -> float:
        """Estimate model size in MB."""
        total_params = 0
        for param in model_state.values():
            if hasattr(param, 'numel'):
                total_params += param.numel()
            elif isinstance(param, (list, tuple)):
                total_params += len(param)
        
        # Assume 4 bytes per parameter (float32)
        size_mb = total_params * 4 / (1024 * 1024)
        return size_mb
    
    def get_simulation_summary(self) -> Dict[str, Any]:
        """Get comprehensive simulation summary."""
        if not self.simulation_metrics:
            return {}
        
        final_metrics = self.simulation_metrics[-1]
        
        return {
            'simulation_config': self.config.__dict__,
            'total_rounds': len(self.simulation_metrics),
            'final_accuracy': final_metrics.global_accuracy,
            'final_loss': final_metrics.global_loss,
            'converged': self.converged,
            'total_communication_cost': sum(m.communication_cost for m in self.simulation_metrics),
            'total_computation_time': sum(m.computation_time for m in self.simulation_metrics),
            'average_client_participation': np.mean([m.client_participation_rate for m in self.simulation_metrics]),
            'average_fairness_score': np.mean([m.fairness_score for m in self.simulation_metrics]),
            'privacy_budget_consumed': final_metrics.privacy_budget_consumed,
            'metrics_history': [m.__dict__ for m in self.simulation_metrics]
        }
    
    def save_simulation_results(self, filepath: str) -> None:
        """Save simulation results to file."""
        results = self.get_simulation_summary()
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Simulation results saved to {filepath}")

# Factory function for creating simulators
def create_federated_simulator(config: SimulationConfig, model_factory: Callable,
                              test_data: Tuple[np.ndarray, np.ndarray]) -> FederatedLearningSimulator:
    """
    Factory function to create a federated learning simulator.
    
    Args:
        config (SimulationConfig): Simulation configuration
        model_factory (Callable): Function that creates model instances
        test_data (Tuple[np.ndarray, np.ndarray]): Global test dataset
        
    Returns:
        FederatedLearningSimulator: Configured simulator instance
    """
    return FederatedLearningSimulator(config, model_factory, test_data)

# Utility functions for simulation analysis
def analyze_simulation_results(results_file: str) -> Dict[str, Any]:
    """
    Analyze simulation results and generate insights.
    
    Args:
        results_file (str): Path to simulation results file
        
    Returns:
        Dict[str, Any]: Analysis results
    """
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    metrics_history = results['metrics_history']
    
    analysis = {
        'convergence_analysis': {
            'rounds_to_convergence': len(metrics_history),
            'final_accuracy': metrics_history[-1]['global_accuracy'] if metrics_history else 0,
            'accuracy_improvement': (metrics_history[-1]['global_accuracy'] - metrics_history[0]['global_accuracy']) if len(metrics_history) > 1 else 0
        },
        'efficiency_analysis': {
            'total_communication_cost': results['total_communication_cost'],
            'average_round_time': results['total_computation_time'] / len(metrics_history) if metrics_history else 0,
            'communication_efficiency': results['final_accuracy'] / results['total_communication_cost'] if results['total_communication_cost'] > 0 else 0
        },
        'fairness_analysis': {
            'average_fairness': results['average_fairness_score'],
            'fairness_trend': 'improving' if len(metrics_history) > 1 and metrics_history[-1]['fairness_score'] > metrics_history[0]['fairness_score'] else 'stable'
        },
        'privacy_analysis': {
            'privacy_budget_used': results['privacy_budget_consumed'],
            'privacy_efficiency': results['final_accuracy'] / max(results['privacy_budget_consumed'], 0.1)
        }
    }
    
    return analysis
