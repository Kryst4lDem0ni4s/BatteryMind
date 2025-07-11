"""
BatteryMind - Federated Learning Client Models

Privacy-preserving distributed learning framework for battery health prediction
across multiple clients while maintaining data confidentiality and security.

This module provides comprehensive federated learning capabilities for battery
management systems, enabling fleet-wide learning without data sharing.

Key Components:
- LocalTrainer: Client-side training with privacy preservation
- ClientManager: Client lifecycle and communication management
- PrivacyEngine: Differential privacy and secure aggregation
- ModelUpdates: Secure model parameter exchange and synchronization

Features:
- Differential privacy with configurable noise levels
- Secure aggregation protocols for model updates
- Client-side data validation and preprocessing
- Adaptive learning rates for heterogeneous clients
- Communication-efficient model compression
- Byzantine fault tolerance for robust aggregation

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .local_trainer import (
    FederatedLocalTrainer,
    LocalTrainingConfig,
    LocalTrainingMetrics,
    ClientDataLoader,
    PrivacyPreservingLoss
)

from .client_manager import (
    FederatedClientManager,
    ClientConfig,
    ClientStatus,
    CommunicationProtocol,
    ClientAuthentication
)

from .privacy_engine import (
    DifferentialPrivacyEngine,
    PrivacyConfig,
    NoiseGenerator,
    PrivacyAccountant,
    SecureAggregation
)

from .model_updates import (
    ModelUpdateManager,
    ModelDelta,
    CompressionScheme,
    UpdateValidator,
    SynchronizationProtocol
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Local training components
    "FederatedLocalTrainer",
    "LocalTrainingConfig",
    "LocalTrainingMetrics",
    "ClientDataLoader",
    "PrivacyPreservingLoss",
    
    # Client management components
    "FederatedClientManager",
    "ClientConfig",
    "ClientStatus",
    "CommunicationProtocol",
    "ClientAuthentication",
    
    # Privacy preservation components
    "DifferentialPrivacyEngine",
    "PrivacyConfig",
    "NoiseGenerator",
    "PrivacyAccountant",
    "SecureAggregation",
    
    # Model update components
    "ModelUpdateManager",
    "ModelDelta",
    "CompressionScheme",
    "UpdateValidator",
    "SynchronizationProtocol"
]

# Federated learning configuration
DEFAULT_FEDERATED_CONFIG = {
    "client": {
        "client_id": "battery_client_001",
        "data_privacy_level": "high",
        "local_epochs": 5,
        "local_batch_size": 16,
        "local_learning_rate": 1e-4,
        "privacy_budget": 1.0,
        "noise_multiplier": 1.1,
        "max_grad_norm": 1.0
    },
    "communication": {
        "compression_ratio": 0.1,
        "quantization_bits": 8,
        "sparsification_threshold": 0.01,
        "communication_rounds": 100,
        "client_fraction": 0.1,
        "min_clients": 2
    },
    "privacy": {
        "differential_privacy": True,
        "secure_aggregation": True,
        "homomorphic_encryption": False,
        "privacy_accounting": True,
        "noise_type": "gaussian",
        "clipping_threshold": 1.0
    },
    "security": {
        "client_authentication": True,
        "message_encryption": True,
        "byzantine_tolerance": True,
        "anomaly_detection": True,
        "secure_channels": True
    }
}

def get_default_federated_config():
    """
    Get default configuration for federated learning clients.
    
    Returns:
        dict: Default federated learning configuration
    """
    return DEFAULT_FEDERATED_CONFIG.copy()

def create_federated_client(client_id: str, config=None):
    """
    Factory function to create a federated learning client.
    
    Args:
        client_id (str): Unique identifier for the client
        config (dict, optional): Client configuration. If None, uses default config.
        
    Returns:
        FederatedClientManager: Configured client instance
    """
    if config is None:
        config = get_default_federated_config()
    
    client_config = ClientConfig(
        client_id=client_id,
        **config["client"]
    )
    
    return FederatedClientManager(client_config)

def create_local_trainer(model, client_data, config=None):
    """
    Factory function to create a local trainer for federated learning.
    
    Args:
        model: Battery health prediction model
        client_data: Local training data
        config (dict, optional): Training configuration. If None, uses default config.
        
    Returns:
        FederatedLocalTrainer: Configured local trainer instance
    """
    if config is None:
        config = get_default_federated_config()
    
    training_config = LocalTrainingConfig(**config["client"])
    return FederatedLocalTrainer(model, client_data, training_config)

def create_privacy_engine(config=None):
    """
    Factory function to create a differential privacy engine.
    
    Args:
        config (dict, optional): Privacy configuration. If None, uses default config.
        
    Returns:
        DifferentialPrivacyEngine: Configured privacy engine instance
    """
    if config is None:
        config = get_default_federated_config()
    
    privacy_config = PrivacyConfig(**config["privacy"])
    return DifferentialPrivacyEngine(privacy_config)

# Federated learning constants
FEDERATED_CONSTANTS = {
    # Privacy parameters
    "DEFAULT_EPSILON": 1.0,          # Differential privacy budget
    "DEFAULT_DELTA": 1e-5,           # Differential privacy delta
    "MIN_NOISE_MULTIPLIER": 0.1,     # Minimum noise for DP
    "MAX_NOISE_MULTIPLIER": 10.0,    # Maximum noise for DP
    
    # Communication parameters
    "MIN_CLIENTS_PER_ROUND": 2,      # Minimum clients for aggregation
    "MAX_CLIENTS_PER_ROUND": 100,    # Maximum clients for aggregation
    "DEFAULT_CLIENT_FRACTION": 0.1,   # Fraction of clients per round
    "COMMUNICATION_TIMEOUT": 300,     # Timeout for client communication (seconds)
    
    # Security parameters
    "KEY_SIZE": 2048,                # RSA key size for encryption
    "SESSION_TIMEOUT": 3600,         # Session timeout (seconds)
    "MAX_FAILED_ATTEMPTS": 3,        # Maximum authentication failures
    
    # Performance parameters
    "DEFAULT_COMPRESSION_RATIO": 0.1, # Model compression ratio
    "MIN_ACCURACY_THRESHOLD": 0.7,   # Minimum accuracy for participation
    "MAX_STALENESS": 5,              # Maximum staleness for async updates
}

def get_federated_constants():
    """
    Get federated learning constants.
    
    Returns:
        dict: Dictionary of federated learning constants
    """
    return FEDERATED_CONSTANTS.copy()

# Privacy budget management
class PrivacyBudgetManager:
    """
    Manages privacy budget allocation across federated learning rounds.
    """
    
    def __init__(self, total_epsilon: float = 1.0, total_delta: float = 1e-5):
        self.total_epsilon = total_epsilon
        self.total_delta = total_delta
        self.used_epsilon = 0.0
        self.used_delta = 0.0
        self.round_budgets = []
    
    def allocate_budget(self, num_rounds: int) -> list:
        """
        Allocate privacy budget across training rounds.
        
        Args:
            num_rounds (int): Number of federated learning rounds
            
        Returns:
            list: Privacy budget allocation per round
        """
        # Simple uniform allocation (can be made adaptive)
        epsilon_per_round = self.total_epsilon / num_rounds
        delta_per_round = self.total_delta / num_rounds
        
        budgets = []
        for round_num in range(num_rounds):
            budget = {
                'round': round_num,
                'epsilon': epsilon_per_round,
                'delta': delta_per_round
            }
            budgets.append(budget)
        
        return budgets
    
    def consume_budget(self, epsilon: float, delta: float) -> bool:
        """
        Consume privacy budget for a training round.
        
        Args:
            epsilon (float): Epsilon to consume
            delta (float): Delta to consume
            
        Returns:
            bool: True if budget available, False otherwise
        """
        if (self.used_epsilon + epsilon <= self.total_epsilon and 
            self.used_delta + delta <= self.total_delta):
            self.used_epsilon += epsilon
            self.used_delta += delta
            return True
        return False
    
    def get_remaining_budget(self) -> dict:
        """
        Get remaining privacy budget.
        
        Returns:
            dict: Remaining epsilon and delta
        """
        return {
            'epsilon': self.total_epsilon - self.used_epsilon,
            'delta': self.total_delta - self.used_delta
        }

# Client selection strategies
class ClientSelectionStrategy:
    """
    Strategies for selecting clients in federated learning rounds.
    """
    
    @staticmethod
    def random_selection(available_clients: list, fraction: float) -> list:
        """Random client selection."""
        import random
        num_clients = max(1, int(len(available_clients) * fraction))
        return random.sample(available_clients, num_clients)
    
    @staticmethod
    def performance_based_selection(clients_performance: dict, fraction: float) -> list:
        """Select clients based on historical performance."""
        sorted_clients = sorted(clients_performance.items(), 
                              key=lambda x: x[1], reverse=True)
        num_clients = max(1, int(len(sorted_clients) * fraction))
        return [client_id for client_id, _ in sorted_clients[:num_clients]]
    
    @staticmethod
    def diversity_based_selection(clients_data_stats: dict, fraction: float) -> list:
        """Select clients to maximize data diversity."""
        # Simplified diversity selection based on data statistics
        # In practice, this would use more sophisticated diversity metrics
        num_clients = max(1, int(len(clients_data_stats) * fraction))
        return list(clients_data_stats.keys())[:num_clients]

# Federated learning utilities
def validate_federated_setup(config: dict) -> dict:
    """
    Validate federated learning configuration.
    
    Args:
        config (dict): Federated learning configuration
        
    Returns:
        dict: Validation results
    """
    validation_results = {
        "valid": True,
        "warnings": [],
        "errors": []
    }
    
    # Check privacy parameters
    if config.get("privacy", {}).get("differential_privacy", False):
        epsilon = config["client"].get("privacy_budget", 1.0)
        if epsilon <= 0:
            validation_results["errors"].append("Privacy budget (epsilon) must be positive")
            validation_results["valid"] = False
        elif epsilon > 10:
            validation_results["warnings"].append("High privacy budget may compromise privacy")
    
    # Check communication parameters
    client_fraction = config["communication"].get("client_fraction", 0.1)
    if client_fraction <= 0 or client_fraction > 1:
        validation_results["errors"].append("Client fraction must be between 0 and 1")
        validation_results["valid"] = False
    
    # Check security settings
    if not config["security"].get("client_authentication", True):
        validation_results["warnings"].append("Client authentication is disabled")
    
    return validation_results

def estimate_communication_cost(model_size: int, num_clients: int, 
                              compression_ratio: float = 0.1) -> dict:
    """
    Estimate communication costs for federated learning.
    
    Args:
        model_size (int): Size of model in bytes
        num_clients (int): Number of participating clients
        compression_ratio (float): Model compression ratio
        
    Returns:
        dict: Communication cost estimates
    """
    compressed_size = model_size * compression_ratio
    
    # Download cost (server to clients)
    download_cost = compressed_size * num_clients
    
    # Upload cost (clients to server)
    upload_cost = compressed_size * num_clients
    
    # Total communication per round
    total_per_round = download_cost + upload_cost
    
    return {
        "model_size_mb": model_size / (1024 * 1024),
        "compressed_size_mb": compressed_size / (1024 * 1024),
        "download_cost_mb": download_cost / (1024 * 1024),
        "upload_cost_mb": upload_cost / (1024 * 1024),
        "total_per_round_mb": total_per_round / (1024 * 1024),
        "compression_ratio": compression_ratio
    }

# Integration with battery health predictor
def create_federated_battery_system(base_model_path: str, client_configs: list):
    """
    Create a federated battery health prediction system.
    
    Args:
        base_model_path (str): Path to base battery health model
        client_configs (list): List of client configurations
        
    Returns:
        dict: Federated system components
    """
    from ..transformers.battery_health_predictor import create_battery_predictor
    
    # Load base model
    base_model = create_battery_predictor(base_model_path)
    
    # Create federated clients
    federated_clients = []
    for client_config in client_configs:
        client = create_federated_client(
            client_config["client_id"], 
            client_config.get("config")
        )
        federated_clients.append(client)
    
    return {
        "base_model": base_model,
        "federated_clients": federated_clients,
        "privacy_budget_manager": PrivacyBudgetManager(),
        "client_selection": ClientSelectionStrategy(),
        "version": __version__
    }

# Module health check
def health_check():
    """
    Perform health check of federated learning client models.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "dependencies_available": True,
        "configuration_valid": True
    }
    
    try:
        # Test basic functionality
        config = get_default_federated_config()
        validation = validate_federated_setup(config)
        health_status["configuration_valid"] = validation["valid"]
        
        # Test privacy engine creation
        privacy_engine = create_privacy_engine(config)
        health_status["privacy_engine"] = True
        
    except Exception as e:
        health_status["error"] = str(e)
        health_status["module_loaded"] = False
    
    return health_status

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Federated Learning Client Models v{__version__} initialized")
