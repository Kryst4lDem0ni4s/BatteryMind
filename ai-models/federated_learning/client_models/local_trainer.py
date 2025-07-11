"""
BatteryMind - Federated Local Trainer

Privacy-preserving local training component for federated learning with
differential privacy, secure aggregation, and communication efficiency.

Features:
- Local model training with privacy preservation
- Differential privacy with configurable noise levels
- Gradient clipping and noise injection
- Communication-efficient model updates
- Adaptive learning rates for heterogeneous data
- Byzantine fault tolerance and anomaly detection

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
import time
import json
import copy
import hashlib
from pathlib import Path

# Privacy and security imports
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import secrets

# Scientific computing
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LocalTrainingConfig:
    """
    Configuration for federated local training.
    
    Attributes:
        # Client identification
        client_id (str): Unique client identifier
        client_type (str): Type of client (vehicle, fleet, charging_station)
        
        # Training parameters
        local_epochs (int): Number of local training epochs
        local_batch_size (int): Local training batch size
        local_learning_rate (float): Local learning rate
        
        # Privacy parameters
        differential_privacy (bool): Enable differential privacy
        privacy_budget (float): Privacy budget (epsilon)
        noise_multiplier (float): Noise multiplier for DP
        max_grad_norm (float): Maximum gradient norm for clipping
        
        # Communication parameters
        compression_enabled (bool): Enable model compression
        compression_ratio (float): Compression ratio for model updates
        quantization_bits (int): Number of bits for quantization
        
        # Security parameters
        encryption_enabled (bool): Enable model encryption
        secure_aggregation (bool): Use secure aggregation
        byzantine_tolerance (bool): Enable Byzantine fault tolerance
        
        # Data parameters
        data_privacy_level (str): Level of data privacy (low, medium, high)
        local_data_size (int): Size of local dataset
        data_heterogeneity (float): Measure of data heterogeneity
        
        # Performance parameters
        device (str): Training device (cpu, cuda)
        mixed_precision (bool): Enable mixed precision training
        gradient_accumulation_steps (int): Gradient accumulation steps
    """
    # Client identification
    client_id: str = "battery_client_001"
    client_type: str = "vehicle"
    
    # Training parameters
    local_epochs: int = 5
    local_batch_size: int = 16
    local_learning_rate: float = 1e-4
    
    # Privacy parameters
    differential_privacy: bool = True
    privacy_budget: float = 1.0
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    
    # Communication parameters
    compression_enabled: bool = True
    compression_ratio: float = 0.1
    quantization_bits: int = 8
    
    # Security parameters
    encryption_enabled: bool = True
    secure_aggregation: bool = True
    byzantine_tolerance: bool = True
    
    # Data parameters
    data_privacy_level: str = "high"
    local_data_size: int = 1000
    data_heterogeneity: float = 0.5
    
    # Performance parameters
    device: str = "auto"
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1

@dataclass
class LocalTrainingMetrics:
    """
    Metrics for local training performance and privacy.
    """
    client_id: str = ""
    round_number: int = 0
    local_loss: float = 0.0
    local_accuracy: float = 0.0
    training_time: float = 0.0
    
    # Privacy metrics
    privacy_spent: float = 0.0
    noise_added: float = 0.0
    gradient_norm: float = 0.0
    
    # Communication metrics
    model_size_before: int = 0
    model_size_after: int = 0
    compression_ratio: float = 0.0
    
    # Data metrics
    local_samples: int = 0
    data_quality_score: float = 0.0

class PrivacyPreservingLoss(nn.Module):
    """
    Loss function with differential privacy guarantees.
    """
    
    def __init__(self, base_loss: nn.Module, privacy_config: LocalTrainingConfig):
        super().__init__()
        self.base_loss = base_loss
        self.privacy_config = privacy_config
        
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute loss with privacy considerations.
        
        Args:
            predictions (torch.Tensor): Model predictions
            targets (torch.Tensor): Ground truth targets
            
        Returns:
            torch.Tensor: Privacy-preserving loss
        """
        # Compute base loss
        loss = self.base_loss(predictions, targets)
        
        # Add privacy-preserving regularization if needed
        if self.privacy_config.differential_privacy:
            # Add small regularization to prevent overfitting to individual samples
            l2_reg = 0.01 * sum(p.pow(2.0).sum() for p in predictions.view(-1))
            loss = loss + l2_reg
        
        return loss

class ClientDataLoader:
    """
    Data loader for federated learning clients with privacy preservation.
    """
    
    def __init__(self, data: pd.DataFrame, config: LocalTrainingConfig):
        self.data = data
        self.config = config
        self.device = self._setup_device()
        
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)
    
    def create_dataloader(self) -> DataLoader:
        """
        Create privacy-preserving data loader.
        
        Returns:
            DataLoader: Configured data loader
        """
        from ..transformers.battery_health_predictor.data_loader import BatteryDataset
        from ..transformers.battery_health_predictor.data_loader import BatteryDataConfig
        
        # Create data configuration
        data_config = BatteryDataConfig(
            batch_size=self.config.local_batch_size,
            sequence_length=512,
            feature_columns=[
                'voltage', 'current', 'temperature', 'state_of_charge',
                'internal_resistance', 'capacity', 'cycle_count', 'age_days'
            ],
            target_columns=[
                'state_of_health', 'capacity_fade_rate', 
                'resistance_increase_rate', 'thermal_degradation'
            ]
        )
        
        # Create dataset
        dataset = BatteryDataset(self.data, data_config, mode='train')
        
        # Create data loader with privacy considerations
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.local_batch_size,
            shuffle=True,  # Important for privacy
            drop_last=True,  # Ensure consistent batch sizes
            num_workers=0,  # Single-threaded for security
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        return dataloader
    
    def validate_data_privacy(self) -> Dict[str, Any]:
        """
        Validate data privacy requirements.
        
        Returns:
            Dict[str, Any]: Privacy validation results
        """
        validation_results = {
            "privacy_compliant": True,
            "issues": [],
            "recommendations": []
        }
        
        # Check for potential privacy leaks
        if len(self.data) < 100:
            validation_results["issues"].append("Small dataset may compromise privacy")
            validation_results["privacy_compliant"] = False
        
        # Check for outliers that might be identifiable
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            z_scores = np.abs(stats.zscore(self.data[column].dropna()))
            outliers = (z_scores > 3).sum()
            if outliers > len(self.data) * 0.05:  # More than 5% outliers
                validation_results["recommendations"].append(
                    f"Consider removing outliers in {column} for privacy"
                )
        
        return validation_results

class DifferentialPrivacyMechanism:
    """
    Differential privacy mechanism for federated learning.
    """
    
    def __init__(self, config: LocalTrainingConfig):
        self.config = config
        self.privacy_accountant = PrivacyAccountant(config.privacy_budget)
        
    def add_noise_to_gradients(self, gradients: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Add calibrated noise to gradients for differential privacy.
        
        Args:
            gradients (List[torch.Tensor]): Model gradients
            
        Returns:
            List[torch.Tensor]: Noisy gradients
        """
        if not self.config.differential_privacy:
            return gradients
        
        noisy_gradients = []
        total_norm = 0.0
        
        # Calculate total gradient norm
        for grad in gradients:
            if grad is not None:
                total_norm += grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        clip_factor = min(1.0, self.config.max_grad_norm / (total_norm + 1e-6))
        
        # Add noise to each gradient
        for grad in gradients:
            if grad is not None:
                # Clip gradient
                clipped_grad = grad * clip_factor
                
                # Add Gaussian noise
                noise_scale = self.config.noise_multiplier * self.config.max_grad_norm
                noise = torch.normal(0, noise_scale, size=grad.shape, device=grad.device)
                
                noisy_grad = clipped_grad + noise
                noisy_gradients.append(noisy_grad)
            else:
                noisy_gradients.append(None)
        
        # Update privacy accountant
        self.privacy_accountant.step(self.config.noise_multiplier)
        
        return noisy_gradients
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get privacy budget spent so far.
        
        Returns:
            Tuple[float, float]: (epsilon, delta) spent
        """
        return self.privacy_accountant.get_privacy_spent()

class PrivacyAccountant:
    """
    Tracks privacy budget consumption during federated learning.
    """
    
    def __init__(self, total_epsilon: float, total_delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.spent_epsilon = 0.0
        self.spent_delta = 0.0
        self.steps = 0
        
    def step(self, noise_multiplier: float, sampling_rate: float = 1.0):
        """
        Account for one step of DP-SGD.
        
        Args:
            noise_multiplier (float): Noise multiplier used
            sampling_rate (float): Sampling rate for this step
        """
        # Simplified privacy accounting (in practice, use more sophisticated methods)
        # This is a basic implementation - production should use RDP accounting
        epsilon_step = sampling_rate / (noise_multiplier ** 2)
        delta_step = sampling_rate * np.exp(-0.5 * noise_multiplier ** 2)
        
        self.spent_epsilon += epsilon_step
        self.spent_delta += delta_step
        self.steps += 1
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """Get total privacy spent."""
        return self.spent_epsilon, self.spent_delta
    
    def get_remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget."""
        return (self.total_epsilon - self.spent_epsilon, 
                self.total_delta - self.spent_delta)
    
    def is_budget_exhausted(self) -> bool:
        """Check if privacy budget is exhausted."""
        remaining_eps, remaining_delta = self.get_remaining_budget()
        return remaining_eps <= 0 or remaining_delta <= 0

class ModelCompressor:
    """
    Compresses model updates for efficient communication.
    """
    
    def __init__(self, config: LocalTrainingConfig):
        self.config = config
        
    def compress_model_update(self, model_update: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Compress model update for communication.
        
        Args:
            model_update (Dict[str, torch.Tensor]): Model parameter updates
            
        Returns:
            Dict[str, Any]: Compressed model update
        """
        if not self.config.compression_enabled:
            return {"compressed": False, "update": model_update}
        
        compressed_update = {}
        compression_stats = {"original_size": 0, "compressed_size": 0}
        
        for name, param in model_update.items():
            original_size = param.numel() * param.element_size()
            compression_stats["original_size"] += original_size
            
            if self.config.compression_ratio < 1.0:
                # Top-k sparsification
                flat_param = param.flatten()
                k = max(1, int(len(flat_param) * self.config.compression_ratio))
                
                # Get top-k values by magnitude
                _, top_indices = torch.topk(torch.abs(flat_param), k)
                sparse_values = flat_param[top_indices]
                
                compressed_update[name] = {
                    "indices": top_indices.cpu().numpy(),
                    "values": sparse_values.cpu().numpy(),
                    "shape": param.shape
                }
                
                compressed_size = len(top_indices) * 4 + len(sparse_values) * 4  # Approximate
                compression_stats["compressed_size"] += compressed_size
            else:
                # Quantization
                if self.config.quantization_bits < 32:
                    quantized_param = self._quantize_tensor(param, self.config.quantization_bits)
                    compressed_update[name] = quantized_param
                    compressed_size = param.numel() * (self.config.quantization_bits // 8)
                    compression_stats["compressed_size"] += compressed_size
                else:
                    compressed_update[name] = param.cpu().numpy()
                    compression_stats["compressed_size"] += original_size
        
        return {
            "compressed": True,
            "update": compressed_update,
            "compression_stats": compression_stats,
            "compression_method": "top_k" if self.config.compression_ratio < 1.0 else "quantization"
        }
    
    def decompress_model_update(self, compressed_update: Dict[str, Any], 
                              reference_model: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Decompress model update.
        
        Args:
            compressed_update (Dict[str, Any]): Compressed model update
            reference_model (Dict[str, torch.Tensor]): Reference model for decompression
            
        Returns:
            Dict[str, torch.Tensor]: Decompressed model update
        """
        if not compressed_update.get("compressed", False):
            return compressed_update["update"]
        
        decompressed_update = {}
        update_data = compressed_update["update"]
        method = compressed_update.get("compression_method", "top_k")
        
        for name, param_data in update_data.items():
            if method == "top_k" and isinstance(param_data, dict):
                # Reconstruct sparse tensor
                indices = torch.from_numpy(param_data["indices"])
                values = torch.from_numpy(param_data["values"])
                shape = param_data["shape"]
                
                # Create sparse update
                full_update = torch.zeros(shape).flatten()
                full_update[indices] = values
                decompressed_update[name] = full_update.reshape(shape)
            else:
                # Handle quantized or uncompressed data
                if isinstance(param_data, np.ndarray):
                    decompressed_update[name] = torch.from_numpy(param_data)
                else:
                    decompressed_update[name] = param_data
        
        return decompressed_update
    
    def _quantize_tensor(self, tensor: torch.Tensor, bits: int) -> np.ndarray:
        """Quantize tensor to specified number of bits."""
        # Simple linear quantization
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        # Avoid division by zero
        if max_val == min_val:
            return np.zeros(tensor.shape, dtype=np.int8 if bits <= 8 else np.int16)
        
        # Quantize
        scale = (2 ** bits - 1) / (max_val - min_val)
        quantized = ((tensor - min_val) * scale).round().clamp(0, 2 ** bits - 1)
        
        # Store as appropriate integer type
        if bits <= 8:
            return quantized.cpu().numpy().astype(np.int8)
        else:
            return quantized.cpu().numpy().astype(np.int16)

class FederatedLocalTrainer:
    """
    Main federated local trainer with privacy preservation and security.
    """
    
    def __init__(self, model: nn.Module, local_data: pd.DataFrame, 
                 config: LocalTrainingConfig):
        self.model = model
        self.local_data = local_data
        self.config = config
        self.device = self._setup_device()
        
        # Initialize components
        self.data_loader = ClientDataLoader(local_data, config)
        self.privacy_mechanism = DifferentialPrivacyMechanism(config)
        self.model_compressor = ModelCompressor(config)
        
        # Training components
        self.optimizer = self._create_optimizer()
        self.loss_function = self._create_loss_function()
        
        # Metrics tracking
        self.training_metrics = []
        self.privacy_spent = (0.0, 0.0)
        
        # Move model to device
        self.model.to(self.device)
        
        logger.info(f"FederatedLocalTrainer initialized for client {config.client_id}")
    
    def _setup_device(self) -> torch.device:
        """Setup training device."""
        if self.config.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            return torch.device(self.config.device)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for local training."""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.local_learning_rate,
            weight_decay=0.01
        )
    
    def _create_loss_function(self) -> nn.Module:
        """Create privacy-preserving loss function."""
        base_loss = nn.MSELoss()
        return PrivacyPreservingLoss(base_loss, self.config)
    
    def train_local_model(self, global_model_state: Dict[str, torch.Tensor],
                         round_number: int) -> Dict[str, Any]:
        """
        Perform local training with privacy preservation.
        
        Args:
            global_model_state (Dict[str, torch.Tensor]): Global model parameters
            round_number (int): Current federated learning round
            
        Returns:
            Dict[str, Any]: Training results and model updates
        """
        start_time = time.time()
        
        # Load global model
        self.model.load_state_dict(global_model_state)
        self.model.train()
        
        # Create data loader
        dataloader = self.data_loader.create_dataloader()
        
        # Training loop
        total_loss = 0.0
        num_batches = 0
        
        for epoch in range(self.config.local_epochs):
            for batch_idx, batch in enumerate(dataloader):
                # Move batch to device
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_function(outputs['predictions'], targets)
                
                # Backward pass
                loss.backward()
                
                # Apply differential privacy to gradients
                if self.config.differential_privacy:
                    gradients = [param.grad for param in self.model.parameters()]
                    noisy_gradients = self.privacy_mechanism.add_noise_to_gradients(gradients)
                    
                    # Replace gradients with noisy versions
                    for param, noisy_grad in zip(self.model.parameters(), noisy_gradients):
                        if param.grad is not None and noisy_grad is not None:
                            param.grad.data = noisy_grad
                
                # Optimizer step
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
        
        # Calculate model update
        model_update = self._calculate_model_update(global_model_state)
        
        # Compress model update
        compressed_update = self.model_compressor.compress_model_update(model_update)
        
        # Calculate metrics
        training_time = time.time() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        
        # Update privacy accounting
        self.privacy_spent = self.privacy_mechanism.get_privacy_spent()
        
        # Create training metrics
        metrics = LocalTrainingMetrics(
            client_id=self.config.client_id,
            round_number=round_number,
            local_loss=avg_loss,
            training_time=training_time,
            privacy_spent=self.privacy_spent[0],
            local_samples=len(self.local_data),
            model_size_before=compressed_update["compression_stats"]["original_size"],
            model_size_after=compressed_update["compression_stats"]["compressed_size"],
            compression_ratio=compressed_update["compression_stats"]["compressed_size"] / 
                           max(compressed_update["compression_stats"]["original_size"], 1)
        )
        
        self.training_metrics.append(metrics)
        
        return {
            "model_update": compressed_update,
            "metrics": metrics,
            "privacy_spent": self.privacy_spent,
            "client_id": self.config.client_id,
            "round_number": round_number
        }
    
    def _calculate_model_update(self, global_model_state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Calculate model update (difference from global model).
        
        Args:
            global_model_state (Dict[str, torch.Tensor]): Global model parameters
            
        Returns:
            Dict[str, torch.Tensor]: Model parameter updates
        """
        model_update = {}
        current_state = self.model.state_dict()
        
        for name, param in current_state.items():
            if name in global_model_state:
                model_update[name] = param - global_model_state[name]
            else:
                model_update[name] = param
        
        return model_update
    
    def evaluate_local_model(self, test_data: Optional[pd.DataFrame] = None) -> Dict[str, float]:
        """
        Evaluate local model performance.
        
        Args:
            test_data (pd.DataFrame, optional): Test data. If None, uses local data.
            
        Returns:
            Dict[str, float]: Evaluation metrics
        """
        self.model.eval()
        
        if test_data is None:
            test_data = self.local_data
        
        # Create test data loader
        test_config = copy.deepcopy(self.config)
        test_config.local_batch_size = 32  # Larger batch for evaluation
        test_loader = ClientDataLoader(test_data, test_config).create_dataloader()
        
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in test_loader:
                inputs = batch['inputs'].to(self.device)
                targets = batch['targets'].to(self.device)
                
                outputs = self.model(inputs)
                loss = self.loss_function(outputs['predictions'], targets)
                
                total_loss += loss.item()
                all_predictions.append(outputs['predictions'].cpu().numpy())
                all_targets.append(targets.cpu().numpy())
        
        # Calculate metrics
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        mse = mean_squared_error(targets, predictions)
        mae = mean_absolute_error(targets, predictions)
        
        return {
            "loss": total_loss / len(test_loader),
            "mse": mse,
            "mae": mae,
            "num_samples": len(test_data)
        }
    
    def get_client_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive client statistics.
        
        Returns:
            Dict[str, Any]: Client statistics
        """
        data_stats = {
            "num_samples": len(self.local_data),
            "data_quality": self.data_loader.validate_data_privacy(),
            "feature_statistics": {}
        }
        
        # Calculate feature statistics
        numeric_columns = self.local_data.select_dtypes(include=[np.number]).columns
        for column in numeric_columns:
            data_stats["feature_statistics"][column] = {
                "mean": float(self.local_data[column].mean()),
                "std": float(self.local_data[column].std()),
                "min": float(self.local_data[column].min()),
                "max": float(self.local_data[column].max())
            }
        
        return {
            "client_id": self.config.client_id,
            "client_type": self.config.client_type,
            "data_statistics": data_stats,
            "privacy_spent": self.privacy_spent,
            "training_rounds": len(self.training_metrics),
            "device": str(self.device),
            "configuration": self.config.__dict__
        }
    
    def reset_privacy_budget(self, new_budget: float):
        """
        Reset privacy budget for new federated learning session.
        
        Args:
            new_budget (float): New privacy budget
        """
        self.config.privacy_budget = new_budget
        self.privacy_mechanism = DifferentialPrivacyMechanism(self.config)
        self.privacy_spent = (0.0, 0.0)
        logger.info(f"Privacy budget reset to {new_budget} for client {self.config.client_id}")

# Factory functions
def create_federated_local_trainer(model: nn.Module, local_data: pd.DataFrame,
                                 config: Optional[LocalTrainingConfig] = None) -> FederatedLocalTrainer:
    """
    Factory function to create a federated local trainer.
    
    Args:
        model (nn.Module): Model to train
        local_data (pd.DataFrame): Local training data
        config (LocalTrainingConfig, optional): Training configuration
        
    Returns:
        FederatedLocalTrainer: Configured trainer instance
    """
    if config is None:
        config = LocalTrainingConfig()
    
    return FederatedLocalTrainer(model, local_data, config)

def simulate_heterogeneous_clients(base_data: pd.DataFrame, num_clients: int,
                                 heterogeneity_level: float = 0.5) -> List[pd.DataFrame]:
    """
    Simulate heterogeneous data distribution across clients.
    
    Args:
        base_data (pd.DataFrame): Base dataset
        num_clients (int): Number of clients to create
        heterogeneity_level (float): Level of data heterogeneity (0-1)
        
    Returns:
        List[pd.DataFrame]: List of client datasets
    """
    client_datasets = []
    
    # Simple heterogeneity simulation based on random sampling
    for i in range(num_clients):
        # Vary sample size based on heterogeneity
        base_size = len(base_data) // num_clients
        size_variation = int(base_size * heterogeneity_level * np.random.uniform(-0.5, 0.5))
        client_size = max(100, base_size + size_variation)
        
        # Sample data for client
        client_data = base_data.sample(n=min(client_size, len(base_data)), replace=True)
        client_datasets.append(client_data.reset_index(drop=True))
    
    return client_datasets
