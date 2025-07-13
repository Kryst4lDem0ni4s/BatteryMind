"""
BatteryMind - Federated Learning Module

Advanced federated learning framework for privacy-preserving battery health
prediction and optimization across distributed battery management systems.

This module implements state-of-the-art federated learning algorithms with
differential privacy guarantees, secure aggregation protocols, and
comprehensive simulation frameworks for battery fleet management.

Key Components:
- FederatedServer: Central coordination server for federated learning
- ClientModels: Local training and model update management
- PrivacyPreserving: Differential privacy and secure aggregation
- SimulationFramework: Comprehensive federated learning simulation
- Utils: Communication, serialization, and security utilities

Features:
- FedAvg, FedProx, and SecAgg aggregation algorithms
- Differential privacy with adaptive noise mechanisms
- Homomorphic encryption for secure model updates
- Multi-tier federated learning for hierarchical battery systems
- Real-time federated optimization for battery fleets
- Privacy-preserving analytics and model evaluation

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

from .server import (
    FederatedServer,
    ServerConfig,
    AggregationAlgorithm,
    ModelAggregator,
    GlobalModel
)

from .client_models import (
    LocalTrainer,
    ClientManager,
    PrivacyEngine,
    ModelUpdates,
    ClientConfig
)

from .privacy_preserving import (
    DifferentialPrivacy,
    HomomorphicEncryption,
    SecureAggregation,
    NoiseMechanisms,
    PrivacyBudget
)

from .simulation_framework import (
    FederatedSimulator,
    ClientSimulator,
    NetworkSimulator,
    EvaluationMetrics,
    SimulationConfig
)

from .utils import (
    CommunicationProtocol,
    ModelSerialization,
    SecurityUtils,
    FederatedMetrics
)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module metadata
__all__ = [
    # Server components
    "FederatedServer",
    "ServerConfig", 
    "AggregationAlgorithm",
    "ModelAggregator",
    "GlobalModel",
    
    # Client components
    "LocalTrainer",
    "ClientManager",
    "PrivacyEngine", 
    "ModelUpdates",
    "ClientConfig",
    
    # Privacy-preserving components
    "DifferentialPrivacy",
    "HomomorphicEncryption",
    "SecureAggregation",
    "NoiseMechanisms",
    "PrivacyBudget",
    
    # Simulation components
    "FederatedSimulator",
    "ClientSimulator", 
    "NetworkSimulator",
    "EvaluationMetrics",
    "SimulationConfig",
    
    # Utility components
    "CommunicationProtocol",
    "ModelSerialization",
    "SecurityUtils",
    "FederatedMetrics"
]

# Default federated learning configuration
DEFAULT_FEDERATED_CONFIG = {
    "server": {
        "aggregation_algorithm": "fedavg",
        "min_clients": 3,
        "max_clients": 100,
        "rounds": 100,
        "client_fraction": 0.3,
        "privacy_budget": 1.0
    },
    "client": {
        "local_epochs": 5,
        "batch_size": 32,
        "learning_rate": 0.01,
        "privacy_enabled": True,
        "noise_multiplier": 1.1
    },
    "privacy": {
        "differential_privacy": True,
        "epsilon": 1.0,
        "delta": 1e-5,
        "secure_aggregation": True,
        "homomorphic_encryption": False
    },
    "simulation": {
        "network_latency": 100,  # ms
        "bandwidth_limit": 10,   # Mbps
        "client_availability": 0.8,
        "byzantine_clients": 0.0
    }
}

def get_default_federated_config():
    """
    Get default configuration for federated learning.
    
    Returns:
        dict: Default federated learning configuration
    """
    return DEFAULT_FEDERATED_CONFIG.copy()

def create_federated_server(config=None):
    """
    Factory function to create a federated learning server.
    
    Args:
        config (dict, optional): Server configuration. If None, uses default.
        
    Returns:
        FederatedServer: Configured federated server instance
    """
    if config is None:
        config = get_default_federated_config()
    
    server_config = ServerConfig(**config["server"])
    return FederatedServer(server_config)

def create_client_manager(client_id, config=None):
    """
    Factory function to create a federated learning client.
    
    Args:
        client_id (str): Unique client identifier
        config (dict, optional): Client configuration. If None, uses default.
        
    Returns:
        ClientManager: Configured client manager instance
    """
    if config is None:
        config = get_default_federated_config()
    
    client_config = ClientConfig(client_id=client_id, **config["client"])
    return ClientManager(client_config)

def create_federated_simulator(config=None):
    """
    Factory function to create a federated learning simulator.
    
    Args:
        config (dict, optional): Simulation configuration. If None, uses default.
        
    Returns:
        FederatedSimulator: Configured simulator instance
    """
    if config is None:
        config = get_default_federated_config()
    
    sim_config = SimulationConfig(**config["simulation"])
    return FederatedSimulator(sim_config)

# Federated learning algorithms
FEDERATED_ALGORITHMS = {
    "fedavg": {
        "name": "Federated Averaging",
        "description": "Standard federated averaging algorithm",
        "privacy_compatible": True,
        "communication_efficient": True
    },
    "fedprox": {
        "name": "Federated Proximal",
        "description": "Federated learning with proximal term for heterogeneous data",
        "privacy_compatible": True,
        "communication_efficient": True
    },
    "fedopt": {
        "name": "Federated Optimization",
        "description": "Federated learning with adaptive server optimization",
        "privacy_compatible": True,
        "communication_efficient": False
    },
    "scaffold": {
        "name": "SCAFFOLD",
        "description": "Stochastic controlled averaging for federated learning",
        "privacy_compatible": True,
        "communication_efficient": True
    }
}

def get_federated_algorithms():
    """
    Get available federated learning algorithms.
    
    Returns:
        dict: Dictionary of available algorithms and their properties
    """
    return FEDERATED_ALGORITHMS.copy()

# Privacy mechanisms
PRIVACY_MECHANISMS = {
    "differential_privacy": {
        "gaussian_mechanism": "Gaussian noise for differential privacy",
        "laplace_mechanism": "Laplace noise for differential privacy", 
        "exponential_mechanism": "Exponential mechanism for discrete outputs"
    },
    "secure_aggregation": {
        "secagg": "Secure aggregation protocol",
        "turbo_aggregate": "TurboAggregate for efficient secure aggregation"
    },
    "homomorphic_encryption": {
        "paillier": "Paillier homomorphic encryption",
        "ckks": "CKKS scheme for approximate arithmetic"
    }
}

def get_privacy_mechanisms():
    """
    Get available privacy-preserving mechanisms.
    
    Returns:
        dict: Dictionary of privacy mechanisms
    """
    return PRIVACY_MECHANISMS.copy()

# Battery-specific federated learning configurations
BATTERY_FEDERATED_SCENARIOS = {
    "fleet_health_monitoring": {
        "description": "Federated health prediction across vehicle fleets",
        "model_type": "battery_health_predictor",
        "aggregation": "fedavg",
        "privacy_budget": 2.0,
        "min_clients": 10
    },
    "degradation_forecasting": {
        "description": "Collaborative degradation pattern learning",
        "model_type": "degradation_forecaster", 
        "aggregation": "fedprox",
        "privacy_budget": 1.5,
        "min_clients": 5
    },
    "charging_optimization": {
        "description": "Distributed charging strategy optimization",
        "model_type": "optimization_recommender",
        "aggregation": "scaffold",
        "privacy_budget": 1.0,
        "min_clients": 15
    },
    "anomaly_detection": {
        "description": "Federated anomaly detection for battery safety",
        "model_type": "anomaly_detector",
        "aggregation": "fedavg",
        "privacy_budget": 3.0,
        "min_clients": 20
    }
}

def get_battery_federated_scenarios():
    """
    Get predefined battery-specific federated learning scenarios.
    
    Returns:
        dict: Dictionary of battery federated learning scenarios
    """
    return BATTERY_FEDERATED_SCENARIOS.copy()

def create_battery_federated_system(scenario_name, num_clients=10):
    """
    Create a complete federated learning system for battery applications.
    
    Args:
        scenario_name (str): Name of the battery scenario
        num_clients (int): Number of federated clients
        
    Returns:
        dict: Complete federated system with server and clients
    """
    if scenario_name not in BATTERY_FEDERATED_SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario_name}")
    
    scenario = BATTERY_FEDERATED_SCENARIOS[scenario_name]
    
    # Create server configuration
    server_config = get_default_federated_config()
    server_config["server"].update({
        "aggregation_algorithm": scenario["aggregation"],
        "min_clients": scenario["min_clients"],
        "privacy_budget": scenario["privacy_budget"]
    })
    
    # Create server
    server = create_federated_server(server_config)
    
    # Create clients
    clients = []
    for i in range(num_clients):
        client_id = f"battery_client_{i:03d}"
        client = create_client_manager(client_id, server_config)
        clients.append(client)
    
    return {
        "server": server,
        "clients": clients,
        "scenario": scenario,
        "config": server_config
    }

# Performance monitoring
FEDERATED_METRICS = {
    "accuracy_metrics": [
        "global_accuracy",
        "client_accuracy_variance", 
        "convergence_rate",
        "model_staleness"
    ],
    "privacy_metrics": [
        "privacy_budget_consumed",
        "noise_variance",
        "information_leakage",
        "membership_inference_risk"
    ],
    "efficiency_metrics": [
        "communication_rounds",
        "total_communication_cost",
        "training_time",
        "client_participation_rate"
    ],
    "robustness_metrics": [
        "byzantine_resilience",
        "dropout_tolerance",
        "heterogeneity_handling",
        "fairness_score"
    ]
}

def get_federated_metrics():
    """
    Get comprehensive federated learning metrics.
    
    Returns:
        dict: Dictionary of federated learning metrics
    """
    return FEDERATED_METRICS.copy()

# Integration utilities
def integrate_with_battery_models(federated_system, battery_model_paths):
    """
    Integrate federated learning system with existing battery models.
    
    Args:
        federated_system (dict): Federated learning system
        battery_model_paths (dict): Paths to battery model artifacts
        
    Returns:
        dict: Integrated system with battery models
    """
    from ..transformers.battery_health_predictor import create_battery_predictor
    from ..transformers.degradation_forecaster import create_battery_degradation_forecaster
    from ..transformers.optimization_recommender import create_optimization_recommender
    
    integrated_system = federated_system.copy()
    
    # Load battery models for federated learning
    if "health_predictor" in battery_model_paths:
        health_model = create_battery_predictor(battery_model_paths["health_predictor"])
        integrated_system["base_models"] = integrated_system.get("base_models", {})
        integrated_system["base_models"]["health_predictor"] = health_model
    
    if "degradation_forecaster" in battery_model_paths:
        forecaster_model = create_battery_degradation_forecaster(battery_model_paths["degradation_forecaster"])
        integrated_system["base_models"] = integrated_system.get("base_models", {})
        integrated_system["base_models"]["degradation_forecaster"] = forecaster_model
    
    if "optimization_recommender" in battery_model_paths:
        optimizer_model = create_optimization_recommender(battery_model_paths["optimization_recommender"])
        integrated_system["base_models"] = integrated_system.get("base_models", {})
        integrated_system["base_models"]["optimization_recommender"] = optimizer_model
    
    return integrated_system

# Module health check
def health_check():
    """
    Perform health check of the federated learning module.
    
    Returns:
        dict: Health check results
    """
    health_status = {
        "module_loaded": True,
        "version": __version__,
        "components_available": True,
        "dependencies_satisfied": True
    }
    
    try:
        # Test component imports
        from .server import FederatedServer
        from .client_models import ClientManager
        from .privacy_preserving import DifferentialPrivacy
        from .simulation_framework import FederatedSimulator
        
        health_status["server_available"] = True
        health_status["client_available"] = True
        health_status["privacy_available"] = True
        health_status["simulation_available"] = True
        
        # Test factory functions
        test_config = get_default_federated_config()
        test_server = create_federated_server(test_config)
        test_client = create_client_manager("test_client", test_config)
        
        health_status["factory_functions"] = True
        
    except Exception as e:
        health_status["components_available"] = False
        health_status["error"] = str(e)
    
    return health_status

# Export configuration template
def export_federated_config(file_path="federated_config.yaml"):
    """
    Export federated learning configuration template.
    
    Args:
        file_path (str): Path to save configuration template
    """
    import yaml
    
    config_template = get_default_federated_config()
    config_template["_metadata"] = {
        "version": __version__,
        "description": "BatteryMind Federated Learning Configuration",
        "author": __author__
    }
    
    with open(file_path, 'w') as f:
        yaml.dump(config_template, f, default_flow_style=False, indent=2)

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BatteryMind Federated Learning v{__version__} initialized")

# Constants for federated learning
FEDERATED_CONSTANTS = {
    "MIN_CLIENTS_FOR_TRAINING": 3,
    "MAX_COMMUNICATION_ROUNDS": 1000,
    "DEFAULT_PRIVACY_BUDGET": 1.0,
    "MIN_PRIVACY_BUDGET": 0.1,
    "MAX_PRIVACY_BUDGET": 10.0,
    "DEFAULT_CLIENT_FRACTION": 0.3,
    "MIN_CLIENT_FRACTION": 0.1,
    "MAX_CLIENT_FRACTION": 1.0
}

def get_federated_constants():
    """
    Get federated learning constants.
    
    Returns:
        dict: Dictionary of federated learning constants
    """
    return FEDERATED_CONSTANTS.copy()
