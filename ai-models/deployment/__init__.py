"""
BatteryMind - Deployment Module

Production deployment infrastructure for BatteryMind AI/ML models across
cloud, edge, and hybrid environments. This module provides comprehensive
deployment capabilities including:

- AWS SageMaker deployment and management
- Edge device deployment with optimization
- Container orchestration and scaling
- Model versioning and A/B testing
- Automated deployment pipelines
- Health monitoring and rollback

Features:
- Multi-cloud deployment support
- Automated CI/CD integration
- Real-time model monitoring
- Blue-green deployments
- Canary releases
- Infrastructure as Code

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path

# Import deployment components
from .aws_sagemaker import (
    SageMakerDeployment,
    EndpointConfig,
    ModelDeployment,
    AutoScalingConfig,
    MonitoringConfig
)

from .edge_deployment import (
    EdgeDeployment,
    ModelOptimization,
    QuantizationConfig,
    PruningConfig,
    EdgeRuntime
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Module exports
__all__ = [
    # Core deployment classes
    "DeploymentManager",
    "DeploymentConfig",
    "DeploymentTarget",
    "DeploymentStatus",
    
    # AWS SageMaker components
    "SageMakerDeployment",
    "EndpointConfig",
    "ModelDeployment",
    "AutoScalingConfig",
    "MonitoringConfig",
    
    # Edge deployment components
    "EdgeDeployment",
    "ModelOptimization",
    "QuantizationConfig",
    "PruningConfig",
    "EdgeRuntime",
    
    # Utility functions
    "create_deployment_manager",
    "validate_deployment_config",
    "get_deployment_status",
    "list_active_deployments",
    
    # Constants
    "SUPPORTED_PLATFORMS",
    "DEFAULT_DEPLOYMENT_CONFIG"
]

class DeploymentTarget(Enum):
    """Supported deployment targets."""
    AWS_SAGEMAKER = "aws_sagemaker"
    AWS_LAMBDA = "aws_lambda"
    AWS_ECS = "aws_ecs"
    AWS_EKS = "aws_eks"
    EDGE_DEVICE = "edge_device"
    DOCKER_CONTAINER = "docker_container"
    KUBERNETES = "kubernetes"
    LOCAL = "local"

class DeploymentStatus(Enum):
    """Deployment status states."""
    PENDING = "pending"
    DEPLOYING = "deploying"
    DEPLOYED = "deployed"
    UPDATING = "updating"
    FAILING = "failing"
    FAILED = "failed"
    ROLLING_BACK = "rolling_back"
    TERMINATED = "terminated"

class ModelType(Enum):
    """Supported model types for deployment."""
    TRANSFORMER = "transformer"
    FEDERATED = "federated"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    ENSEMBLE = "ensemble"

@dataclass
class DeploymentConfig:
    """Configuration for model deployment."""
    
    # Basic configuration
    deployment_name: str
    model_name: str
    model_version: str
    target: DeploymentTarget
    
    # Model configuration
    model_type: ModelType
    model_path: str
    model_format: str = "pytorch"  # pytorch, tensorflow, onnx, etc.
    
    # Infrastructure configuration
    instance_type: str = "ml.m5.large"
    instance_count: int = 1
    min_capacity: int = 1
    max_capacity: int = 10
    
    # Scaling configuration
    auto_scaling_enabled: bool = True
    target_invocations_per_instance: int = 1000
    scale_down_cooldown: int = 300  # seconds
    scale_up_cooldown: int = 300    # seconds
    
    # Monitoring configuration
    enable_monitoring: bool = True
    enable_data_capture: bool = True
    sampling_percentage: float = 20.0
    
    # Security configuration
    enable_network_isolation: bool = True
    vpc_config: Optional[Dict[str, Any]] = None
    
    # Advanced configuration
    enable_multi_model: bool = False
    enable_model_caching: bool = True
    max_payload_size_mb: int = 6
    max_concurrent_transforms: int = 100
    
    # Environment variables
    environment_variables: Dict[str, str] = None
    
    def __post_init__(self):
        if self.environment_variables is None:
            self.environment_variables = {}

@dataclass
class DeploymentResult:
    """Result of deployment operation."""
    deployment_name: str
    status: DeploymentStatus
    endpoint_url: Optional[str] = None
    deployment_arn: Optional[str] = None
    creation_time: Optional[str] = None
    last_modified_time: Optional[str] = None
    failure_reason: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}

class DeploymentManager:
    """
    Central deployment manager for BatteryMind models.
    
    This class coordinates deployments across different platforms and
    provides a unified interface for model deployment operations.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize deployment manager.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.active_deployments: Dict[str, DeploymentResult] = {}
        self.deployment_history: List[DeploymentResult] = []
        
        # Initialize platform-specific deployers
        self._initialize_deployers()
        
        logger.info("DeploymentManager initialized successfully")
    
    def _initialize_deployers(self):
        """Initialize platform-specific deployment handlers."""
        try:
            # AWS SageMaker deployer
            from .aws_sagemaker import SageMakerDeployment
            self.sagemaker_deployer = SageMakerDeployment(
                self.config.get('aws_config', {})
            )
            
            # Edge deployer
            from .edge_deployment import EdgeDeployment
            self.edge_deployer = EdgeDeployment(
                self.config.get('edge_config', {})
            )
            
            logger.info("Platform deployers initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize deployers: {e}")
            raise
    
    def deploy_model(self, deployment_config: DeploymentConfig) -> DeploymentResult:
        """
        Deploy a model to the specified target.
        
        Args:
            deployment_config: Deployment configuration
            
        Returns:
            DeploymentResult: Result of deployment operation
        """
        logger.info(f"Starting deployment: {deployment_config.deployment_name}")
        
        try:
            # Validate configuration
            self._validate_deployment_config(deployment_config)
            
            # Route to appropriate deployer
            if deployment_config.target == DeploymentTarget.AWS_SAGEMAKER:
                result = self._deploy_to_sagemaker(deployment_config)
            elif deployment_config.target == DeploymentTarget.EDGE_DEVICE:
                result = self._deploy_to_edge(deployment_config)
            elif deployment_config.target == DeploymentTarget.DOCKER_CONTAINER:
                result = self._deploy_to_docker(deployment_config)
            elif deployment_config.target == DeploymentTarget.KUBERNETES:
                result = self._deploy_to_kubernetes(deployment_config)
            else:
                raise ValueError(f"Unsupported deployment target: {deployment_config.target}")
            
            # Track deployment
            self.active_deployments[deployment_config.deployment_name] = result
            self.deployment_history.append(result)
            
            logger.info(f"Deployment completed: {deployment_config.deployment_name}")
            return result
            
        except Exception as e:
            logger.error(f"Deployment failed: {e}")
            error_result = DeploymentResult(
                deployment_name=deployment_config.deployment_name,
                status=DeploymentStatus.FAILED,
                failure_reason=str(e)
            )
            self.deployment_history.append(error_result)
            return error_result
    
    def _deploy_to_sagemaker(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy model to AWS SageMaker."""
        return self.sagemaker_deployer.deploy(config)
    
    def _deploy_to_edge(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy model to edge device."""
        return self.edge_deployer.deploy(config)
    
    def _deploy_to_docker(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy model using Docker containers."""
        # Placeholder for Docker deployment
        # This would integrate with Docker API
        pass
    
    def _deploy_to_kubernetes(self, config: DeploymentConfig) -> DeploymentResult:
        """Deploy model to Kubernetes cluster."""
        # Placeholder for Kubernetes deployment
        # This would integrate with Kubernetes API
        pass
    
    def _validate_deployment_config(self, config: DeploymentConfig):
        """Validate deployment configuration."""
        # Check required fields
        if not config.deployment_name:
            raise ValueError("Deployment name is required")
        
        if not config.model_path or not os.path.exists(config.model_path):
            raise ValueError(f"Model path does not exist: {config.model_path}")
        
        # Validate target-specific requirements
        if config.target == DeploymentTarget.AWS_SAGEMAKER:
            self._validate_sagemaker_config(config)
        elif config.target == DeploymentTarget.EDGE_DEVICE:
            self._validate_edge_config(config)
    
    def _validate_sagemaker_config(self, config: DeploymentConfig):
        """Validate SageMaker-specific configuration."""
        valid_instances = [
            "ml.t2.medium", "ml.t2.large", "ml.t2.xlarge", "ml.t2.2xlarge",
            "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
            "ml.c5.large", "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge"
        ]
        
        if config.instance_type not in valid_instances:
            logger.warning(f"Instance type {config.instance_type} may not be supported")
    
    def _validate_edge_config(self, config: DeploymentConfig):
        """Validate edge deployment configuration."""
        if config.max_payload_size_mb > 100:
            logger.warning("Large payload size may not be suitable for edge deployment")
    
    def update_deployment(self, deployment_name: str, 
                         new_config: DeploymentConfig) -> DeploymentResult:
        """
        Update an existing deployment.
        
        Args:
            deployment_name: Name of deployment to update
            new_config: New configuration
            
        Returns:
            DeploymentResult: Result of update operation
        """
        if deployment_name not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_name} not found")
        
        logger.info(f"Updating deployment: {deployment_name}")
        
        try:
            # Get current deployment
            current_deployment = self.active_deployments[deployment_name]
            
            # Route to appropriate updater
            if new_config.target == DeploymentTarget.AWS_SAGEMAKER:
                result = self.sagemaker_deployer.update(deployment_name, new_config)
            elif new_config.target == DeploymentTarget.EDGE_DEVICE:
                result = self.edge_deployer.update(deployment_name, new_config)
            else:
                raise ValueError(f"Update not supported for target: {new_config.target}")
            
            # Update tracking
            self.active_deployments[deployment_name] = result
            self.deployment_history.append(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Deployment update failed: {e}")
            raise
    
    def delete_deployment(self, deployment_name: str) -> bool:
        """
        Delete a deployment.
        
        Args:
            deployment_name: Name of deployment to delete
            
        Returns:
            bool: True if successful
        """
        if deployment_name not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_name} not found")
        
        logger.info(f"Deleting deployment: {deployment_name}")
        
        try:
            deployment = self.active_deployments[deployment_name]
            
            # Route to appropriate deleter based on the deployment target
            # This would need to be tracked in the deployment result
            success = True  # Placeholder
            
            if success:
                # Update status
                deployment.status = DeploymentStatus.TERMINATED
                self.deployment_history.append(deployment)
                
                # Remove from active deployments
                del self.active_deployments[deployment_name]
                
                logger.info(f"Deployment deleted: {deployment_name}")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Deployment deletion failed: {e}")
            return False
    
    def get_deployment_status(self, deployment_name: str) -> Optional[DeploymentResult]:
        """
        Get status of a deployment.
        
        Args:
            deployment_name: Name of deployment
            
        Returns:
            DeploymentResult: Current deployment status
        """
        return self.active_deployments.get(deployment_name)
    
    def list_deployments(self) -> List[DeploymentResult]:
        """
        List all active deployments.
        
        Returns:
            List[DeploymentResult]: List of active deployments
        """
        return list(self.active_deployments.values())
    
    def get_deployment_metrics(self, deployment_name: str) -> Dict[str, Any]:
        """
        Get metrics for a deployment.
        
        Args:
            deployment_name: Name of deployment
            
        Returns:
            Dict[str, Any]: Deployment metrics
        """
        if deployment_name not in self.active_deployments:
            raise ValueError(f"Deployment {deployment_name} not found")
        
        deployment = self.active_deployments[deployment_name]
        
        # Get metrics from appropriate platform
        # This would integrate with platform-specific monitoring
        metrics = {
            "invocations": 0,
            "errors": 0,
            "latency_p50": 0.0,
            "latency_p95": 0.0,
            "latency_p99": 0.0
        }
        
        return metrics

# Utility functions
def create_deployment_manager(config: Optional[Dict[str, Any]] = None) -> DeploymentManager:
    """
    Factory function to create a deployment manager.
    
    Args:
        config: Optional configuration
        
    Returns:
        DeploymentManager: Configured deployment manager
    """
    return DeploymentManager(config)

def validate_deployment_config(config: DeploymentConfig) -> List[str]:
    """
    Validate deployment configuration and return any issues.
    
    Args:
        config: Deployment configuration to validate
        
    Returns:
        List[str]: List of validation issues (empty if valid)
    """
    issues = []
    
    # Basic validation
    if not config.deployment_name:
        issues.append("Deployment name is required")
    
    if not config.model_path:
        issues.append("Model path is required")
    elif not os.path.exists(config.model_path):
        issues.append(f"Model path does not exist: {config.model_path}")
    
    # Instance configuration validation
    if config.instance_count < 1:
        issues.append("Instance count must be at least 1")
    
    if config.min_capacity > config.max_capacity:
        issues.append("Min capacity cannot be greater than max capacity")
    
    # Monitoring validation
    if config.sampling_percentage < 0 or config.sampling_percentage > 100:
        issues.append("Sampling percentage must be between 0 and 100")
    
    return issues

def get_deployment_status(deployment_manager: DeploymentManager, 
                         deployment_name: str) -> Optional[DeploymentStatus]:
    """
    Get the status of a specific deployment.
    
    Args:
        deployment_manager: DeploymentManager instance
        deployment_name: Name of deployment
        
    Returns:
        DeploymentStatus: Current status or None if not found
    """
    result = deployment_manager.get_deployment_status(deployment_name)
    return result.status if result else None

def list_active_deployments(deployment_manager: DeploymentManager) -> List[str]:
    """
    Get list of active deployment names.
    
    Args:
        deployment_manager: DeploymentManager instance
        
    Returns:
        List[str]: List of active deployment names
    """
    return list(deployment_manager.active_deployments.keys())

# Constants
SUPPORTED_PLATFORMS = [
    DeploymentTarget.AWS_SAGEMAKER,
    DeploymentTarget.AWS_LAMBDA,
    DeploymentTarget.AWS_ECS,
    DeploymentTarget.EDGE_DEVICE,
    DeploymentTarget.DOCKER_CONTAINER,
    DeploymentTarget.KUBERNETES
]

DEFAULT_DEPLOYMENT_CONFIG = {
    "instance_type": "ml.m5.large",
    "instance_count": 1,
    "min_capacity": 1,
    "max_capacity": 10,
    "auto_scaling_enabled": True,
    "enable_monitoring": True,
    "enable_data_capture": True,
    "sampling_percentage": 20.0,
    "enable_network_isolation": True,
    "enable_model_caching": True,
    "max_payload_size_mb": 6
}

# Module initialization
logger.info(f"BatteryMind Deployment module v{__version__} loaded")
logger.info(f"Supported platforms: {[t.value for t in SUPPORTED_PLATFORMS]}")
