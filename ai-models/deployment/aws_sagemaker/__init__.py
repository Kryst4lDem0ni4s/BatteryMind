"""
BatteryMind - AWS SageMaker Deployment Module

Comprehensive AWS SageMaker deployment capabilities for BatteryMind AI/ML models.
This module provides production-ready deployment infrastructure with:

- Real-time endpoint deployment
- Batch transform jobs
- Multi-model endpoints
- Auto-scaling configuration
- A/B testing and canary deployments
- Comprehensive monitoring and alerting

Features:
- Automated model registration
- Blue-green deployments
- Custom inference containers
- Data capture and monitoring
- Cost optimization
- Security best practices

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import json

# AWS SDK imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False
    boto3 = None
    ClientError = Exception
    NoCredentialsError = Exception

# Internal imports
from .endpoint_config import (
    EndpointConfig,
    EndpointConfigBuilder,
    ProductionVariant,
    DataCaptureConfig
)

from .model_deployment import (
    ModelDeployment,
    ModelRegistration,
    ModelPackage,
    ContainerDefinition
)

from .auto_scaling import (
    AutoScalingConfig,
    ScalingPolicy,
    TargetTrackingConfig,
    StepScalingConfig
)

from .monitoring import (
    MonitoringConfig,
    CloudWatchMetrics,
    ModelMonitor,
    DataQualityMonitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Module metadata
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"

# Module exports
__all__ = [
    # Main classes
    "SageMakerDeployment",
    "SageMakerConfig",
    "DeploymentStrategy",
    
    # Configuration classes
    "EndpointConfig",
    "EndpointConfigBuilder",
    "ProductionVariant",
    "DataCaptureConfig",
    
    # Model deployment
    "ModelDeployment",
    "ModelRegistration", 
    "ModelPackage",
    "ContainerDefinition",
    
    # Auto-scaling
    "AutoScalingConfig",
    "ScalingPolicy",
    "TargetTrackingConfig",
    "StepScalingConfig",
    
    # Monitoring
    "MonitoringConfig",
    "CloudWatchMetrics",
    "ModelMonitor",
    "DataQualityMonitor",
    
    # Utility functions
    "create_sagemaker_deployment",
    "validate_aws_credentials",
    "get_available_instance_types",
    "estimate_deployment_cost",
    
    # Exception classes
    "SageMakerDeploymentError",
    "InvalidConfigurationError",
    "DeploymentTimeoutError"
]

# Exception classes
class SageMakerDeploymentError(Exception):
    """Base exception for SageMaker deployment errors."""
    pass

class InvalidConfigurationError(SageMakerDeploymentError):
    """Exception for invalid configuration."""
    pass

class DeploymentTimeoutError(SageMakerDeploymentError):
    """Exception for deployment timeouts."""
    pass

@dataclass
class SageMakerConfig:
    """Configuration for SageMaker deployment."""
    
    # AWS Configuration
    region_name: str = "us-west-2"
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    role_arn: str = ""
    
    # S3 Configuration
    s3_bucket: str = ""
    s3_model_prefix: str = "models"
    s3_data_prefix: str = "data"
    
    # Default endpoint configuration
    default_instance_type: str = "ml.m5.large"
    default_instance_count: int = 1
    default_max_concurrent_transforms: int = 100
    
    # Monitoring configuration
    enable_monitoring: bool = True
    enable_data_capture: bool = True
    data_capture_percentage: float = 20.0
    
    # Security configuration
    enable_network_isolation: bool = True
    vpc_config: Optional[Dict[str, Any]] = None
    kms_key_id: Optional[str] = None
    
    # Advanced configuration
    enable_model_caching: bool = True
    model_cache_size_gb: int = 10
    async_inference_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.role_arn:
            raise InvalidConfigurationError("SageMaker execution role ARN is required")
        
        if not self.s3_bucket:
            raise InvalidConfigurationError("S3 bucket is required")

class DeploymentStrategy:
    """Deployment strategy definitions."""
    
    SINGLE_MODEL = "single_model"
    MULTI_MODEL = "multi_model"
    BLUE_GREEN = "blue_green"
    CANARY = "canary"
    ROLLING = "rolling"

class SageMakerDeployment:
    """
    Main SageMaker deployment orchestrator.
    
    This class provides comprehensive deployment capabilities for
    BatteryMind models on AWS SageMaker infrastructure.
    """
    
    def __init__(self, config: SageMakerConfig):
        """
        Initialize SageMaker deployment.
        
        Args:
            config: SageMaker configuration
        """
        self.config = config
        self.active_endpoints: Dict[str, Dict[str, Any]] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        
        # Initialize AWS clients
        self._initialize_aws_clients()
        
        # Initialize deployment components
        self.model_deployment = ModelDeployment(self.sagemaker_client, config)
        self.auto_scaling = AutoScalingConfig(self.auto_scaling_client, config)
        self.monitoring = MonitoringConfig(self.cloudwatch_client, config)
        
        logger.info("SageMakerDeployment initialized successfully")
    
    def _initialize_aws_clients(self):
        """Initialize AWS service clients."""
        if not AWS_AVAILABLE:
            raise SageMakerDeploymentError(
                "AWS SDK (boto3) is not available. Please install: pip install boto3"
            )
        
        try:
            # Create session with configuration
            session_kwargs = {
                'region_name': self.config.region_name
            }
            
            if self.config.aws_access_key_id and self.config.aws_secret_access_key:
                session_kwargs.update({
                    'aws_access_key_id': self.config.aws_access_key_id,
                    'aws_secret_access_key': self.config.aws_secret_access_key
                })
            
            session = boto3.Session(**session_kwargs)
            
            # Initialize service clients
            self.sagemaker_client = session.client('sagemaker')
            self.sagemaker_runtime = session.client('sagemaker-runtime')
            self.s3_client = session.client('s3')
            self.auto_scaling_client = session.client('application-autoscaling')
            self.cloudwatch_client = session.client('cloudwatch')
            
            # Validate credentials
            self._validate_credentials()
            
            logger.info("AWS clients initialized successfully")
            
        except NoCredentialsError:
            raise SageMakerDeploymentError(
                "AWS credentials not found. Please configure credentials."
            )
        except Exception as e:
            raise SageMakerDeploymentError(f"Failed to initialize AWS clients: {e}")
    
    def _validate_credentials(self):
        """Validate AWS credentials and permissions."""
        try:
            # Test SageMaker access
            self.sagemaker_client.list_endpoints(MaxResults=1)
            
            # Test S3 access
            self.s3_client.head_bucket(Bucket=self.config.s3_bucket)
            
            logger.info("AWS credentials validated successfully")
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == 'NoSuchBucket':
                raise SageMakerDeploymentError(f"S3 bucket not found: {self.config.s3_bucket}")
            elif error_code == 'AccessDenied':
                raise SageMakerDeploymentError("Insufficient AWS permissions")
            else:
                raise SageMakerDeploymentError(f"AWS validation failed: {e}")
    
    def deploy_real_time_endpoint(self, 
                                 model_name: str,
                                 model_path: str,
                                 endpoint_config: EndpointConfig,
                                 deployment_strategy: str = DeploymentStrategy.SINGLE_MODEL) -> Dict[str, Any]:
        """
        Deploy a real-time inference endpoint.
        
        Args:
            model_name: Name of the model
            model_path: S3 path to model artifacts
            endpoint_config: Endpoint configuration
            deployment_strategy: Deployment strategy
            
        Returns:
            Dict[str, Any]: Deployment result
        """
        logger.info(f"Deploying real-time endpoint: {model_name}")
        
        try:
            # Register model
            model_registration = self.model_deployment.register_model(
                model_name=model_name,
                model_path=model_path,
                container_image=endpoint_config.container_image
            )
            
            # Create endpoint configuration
            endpoint_config_name = f"{model_name}-config-{int(datetime.now().timestamp())}"
            config_response = self._create_endpoint_config(
                endpoint_config_name, model_registration, endpoint_config
            )
            
            # Create endpoint
            endpoint_name = f"{model_name}-endpoint"
            endpoint_response = self._create_endpoint(
                endpoint_name, endpoint_config_name, deployment_strategy
            )
            
            # Setup auto-scaling if enabled
            if endpoint_config.auto_scaling_enabled:
                self.auto_scaling.setup_auto_scaling(
                    endpoint_name, 
                    endpoint_config.auto_scaling_config
                )
            
            # Setup monitoring if enabled
            if self.config.enable_monitoring:
                self.monitoring.setup_endpoint_monitoring(
                    endpoint_name, endpoint_config.monitoring_config
                )
            
            # Track deployment
            deployment_info = {
                'endpoint_name': endpoint_name,
                'model_name': model_name,
                'endpoint_config_name': endpoint_config_name,
                'status': 'Creating',
                'creation_time': datetime.now().isoformat(),
                'deployment_strategy': deployment_strategy
            }
            
            self.active_endpoints[endpoint_name] = deployment_info
            self.deployment_history.append(deployment_info)
            
            logger.info(f"Endpoint deployment initiated: {endpoint_name}")
            
            return {
                'endpoint_name': endpoint_name,
                'endpoint_arn': endpoint_response.get('EndpointArn'),
                'status': 'Creating',
                'deployment_strategy': deployment_strategy
            }
            
        except Exception as e:
            logger.error(f"Endpoint deployment failed: {e}")
            raise SageMakerDeploymentError(f"Deployment failed: {e}")
    
    def _create_endpoint_config(self, 
                               config_name: str,
                               model_registration: Dict[str, Any],
                               endpoint_config: EndpointConfig) -> Dict[str, Any]:
        """Create SageMaker endpoint configuration."""
        
        production_variants = []
        
        for variant in endpoint_config.production_variants:
            production_variants.append({
                'VariantName': variant.variant_name,
                'ModelName': model_registration['ModelName'],
                'InitialInstanceCount': variant.initial_instance_count,
                'InstanceType': variant.instance_type,
                'InitialVariantWeight': variant.initial_variant_weight,
                'AcceleratorType': variant.accelerator_type
            })
        
        config_params = {
            'EndpointConfigName': config_name,
            'ProductionVariants': production_variants
        }
        
        # Add data capture configuration if enabled
        if endpoint_config.data_capture_config:
            config_params['DataCaptureConfig'] = {
                'EnableCapture': True,
                'InitialSamplingPercentage': endpoint_config.data_capture_config.sampling_percentage,
                'DestinationS3Uri': endpoint_config.data_capture_config.s3_destination,
                'CaptureOptions': [
                    {'CaptureMode': 'Input'},
                    {'CaptureMode': 'Output'}
                ]
            }
        
        # Add async inference configuration if provided
        if endpoint_config.async_inference_config:
            config_params['AsyncInferenceConfig'] = endpoint_config.async_inference_config
        
        return self.sagemaker_client.create_endpoint_config(**config_params)
    
    def _create_endpoint(self, 
                        endpoint_name: str,
                        endpoint_config_name: str,
                        deployment_strategy: str) -> Dict[str, Any]:
        """Create SageMaker endpoint."""
        
        endpoint_params = {
            'EndpointName': endpoint_name,
            'EndpointConfigName': endpoint_config_name
        }
        
        # Add deployment configuration based on strategy
        if deployment_strategy == DeploymentStrategy.BLUE_GREEN:
            endpoint_params['DeploymentConfig'] = {
                'BlueGreenUpdatePolicy': {
                    'TrafficRoutingConfiguration': {
                        'Type': 'ALL_AT_ONCE',
                        'WaitIntervalInSeconds': 300
                    },
                    'TerminationWaitInSeconds': 300,
                    'MaximumExecutionTimeoutInSeconds': 1800
                }
            }
        elif deployment_strategy == DeploymentStrategy.CANARY:
            endpoint_params['DeploymentConfig'] = {
                'BlueGreenUpdatePolicy': {
                    'TrafficRoutingConfiguration': {
                        'Type': 'CANARY',
                        'CanarySize': {
                            'Type': 'CAPACITY_PERCENT',
                            'Value': 25
                        },
                        'WaitIntervalInSeconds': 300
                    },
                    'TerminationWaitInSeconds': 300,
                    'MaximumExecutionTimeoutInSeconds': 1800
                }
            }
        
        return self.sagemaker_client.create_endpoint(**endpoint_params)
    
    def update_endpoint(self, 
                       endpoint_name: str,
                       new_endpoint_config: EndpointConfig) -> Dict[str, Any]:
        """
        Update an existing endpoint.
        
        Args:
            endpoint_name: Name of endpoint to update
            new_endpoint_config: New endpoint configuration
            
        Returns:
            Dict[str, Any]: Update result
        """
        logger.info(f"Updating endpoint: {endpoint_name}")
        
        try:
            # Create new endpoint configuration
            config_name = f"{endpoint_name}-config-{int(datetime.now().timestamp())}"
            
            # This would need the model registration info
            # For now, we'll use a placeholder
            model_registration = {'ModelName': f"{endpoint_name}-model"}
            
            self._create_endpoint_config(config_name, model_registration, new_endpoint_config)
            
            # Update endpoint
            response = self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=config_name
            )
            
            # Update tracking
            if endpoint_name in self.active_endpoints:
                self.active_endpoints[endpoint_name]['status'] = 'Updating'
                self.active_endpoints[endpoint_name]['last_updated'] = datetime.now().isoformat()
            
            logger.info(f"Endpoint update initiated: {endpoint_name}")
            
            return {
                'endpoint_name': endpoint_name,
                'endpoint_arn': response.get('EndpointArn'),
                'status': 'Updating'
            }
            
        except Exception as e:
            logger.error(f"Endpoint update failed: {e}")
            raise SageMakerDeploymentError(f"Update failed: {e}")
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """
        Delete an endpoint.
        
        Args:
            endpoint_name: Name of endpoint to delete
            
        Returns:
            bool: True if successful
        """
        logger.info(f"Deleting endpoint: {endpoint_name}")
        
        try:
            # Delete endpoint
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # Remove from tracking
            if endpoint_name in self.active_endpoints:
                self.active_endpoints[endpoint_name]['status'] = 'Deleting'
                
            logger.info(f"Endpoint deletion initiated: {endpoint_name}")
            return True
            
        except Exception as e:
            logger.error(f"Endpoint deletion failed: {e}")
            return False
    
    def get_endpoint_status(self, endpoint_name: str) -> Dict[str, Any]:
        """
        Get endpoint status.
        
        Args:
            endpoint_name: Name of endpoint
            
        Returns:
            Dict[str, Any]: Endpoint status information
        """
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            
            return {
                'endpoint_name': endpoint_name,
                'status': response['EndpointStatus'],
                'creation_time': response['CreationTime'],
                'last_modified_time': response['LastModifiedTime'],
                'endpoint_arn': response['EndpointArn'],
                'production_variants': response.get('ProductionVariants', []),
                'failure_reason': response.get('FailureReason')
            }
            
        except ClientError as e:
            if e.response['Error']['Code'] == 'ValidationException':
                return {'endpoint_name': endpoint_name, 'status': 'NotFound'}
            else:
                raise
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """
        List all endpoints.
        
        Returns:
            List[Dict[str, Any]]: List of endpoints
        """
        try:
            response = self.sagemaker_client.list_endpoints()
            return response.get('Endpoints', [])
            
        except Exception as e:
            logger.error(f"Failed to list endpoints: {e}")
            return []
    
    def invoke_endpoint(self, 
                       endpoint_name: str,
                       payload: Union[str, bytes, Dict[str, Any]],
                       content_type: str = 'application/json') -> Dict[str, Any]:
        """
        Invoke an endpoint for inference.
        
        Args:
            endpoint_name: Name of endpoint
            payload: Input data
            content_type: Content type of payload
            
        Returns:
            Dict[str, Any]: Inference result
        """
        try:
            # Prepare payload
            if isinstance(payload, dict):
                body = json.dumps(payload)
            elif isinstance(payload, str):
                body = payload
            else:
                body = payload
            
            # Invoke endpoint
            response = self.sagemaker_runtime.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType=content_type,
                Body=body
            )
            
            # Parse response
            result = response['Body'].read()
            
            return {
                'status_code': response['ResponseMetadata']['HTTPStatusCode'],
                'content_type': response['ContentType'],
                'result': result.decode('utf-8') if isinstance(result, bytes) else result,
                'invoked_production_variant': response.get('InvokedProductionVariant')
            }
            
        except Exception as e:
            logger.error(f"Endpoint invocation failed: {e}")
            raise SageMakerDeploymentError(f"Invocation failed: {e}")

# Utility functions
def create_sagemaker_deployment(config: Optional[SageMakerConfig] = None) -> SageMakerDeployment:
    """
    Factory function to create SageMaker deployment.
    
    Args:
        config: Optional SageMaker configuration
        
    Returns:
        SageMakerDeployment: Configured deployment instance
    """
    if config is None:
        config = SageMakerConfig()
    
    return SageMakerDeployment(config)

def validate_aws_credentials() -> bool:
    """
    Validate AWS credentials.
    
    Returns:
        bool: True if credentials are valid
    """
    if not AWS_AVAILABLE:
        return False
    
    try:
        session = boto3.Session()
        sts_client = session.client('sts')
        sts_client.get_caller_identity()
        return True
    except:
        return False

def get_available_instance_types() -> List[str]:
    """
    Get list of available SageMaker instance types.
    
    Returns:
        List[str]: Available instance types
    """
    return [
        # General purpose
        "ml.t2.medium", "ml.t2.large", "ml.t2.xlarge", "ml.t2.2xlarge",
        "ml.t3.medium", "ml.t3.large", "ml.t3.xlarge", "ml.t3.2xlarge",
        "ml.m5.large", "ml.m5.xlarge", "ml.m5.2xlarge", "ml.m5.4xlarge",
        "ml.m5.12xlarge", "ml.m5.24xlarge",
        
        # Compute optimized
        "ml.c5.large", "ml.c5.xlarge", "ml.c5.2xlarge", "ml.c5.4xlarge",
        "ml.c5.9xlarge", "ml.c5.18xlarge",
        
        # Memory optimized
        "ml.r5.large", "ml.r5.xlarge", "ml.r5.2xlarge", "ml.r5.4xlarge",
        "ml.r5.12xlarge", "ml.r5.24xlarge",
        
        # GPU instances
        "ml.p3.2xlarge", "ml.p3.8xlarge", "ml.p3.16xlarge",
        "ml.p4d.24xlarge",
        "ml.g4dn.xlarge", "ml.g4dn.2xlarge", "ml.g4dn.4xlarge"
    ]

def estimate_deployment_cost(instance_type: str, 
                           instance_count: int,
                           hours_per_month: int = 730) -> Dict[str, float]:
    """
    Estimate deployment cost.
    
    Args:
        instance_type: SageMaker instance type
        instance_count: Number of instances
        hours_per_month: Hours per month (default: 730)
        
    Returns:
        Dict[str, float]: Cost estimation
    """
    # Simplified cost calculation (prices as of 2024)
    # Real implementation would use AWS Pricing API
    
    hourly_rates = {
        "ml.t2.medium": 0.056,
        "ml.t2.large": 0.112,
        "ml.m5.large": 0.115,
        "ml.m5.xlarge": 0.23,
        "ml.c5.large": 0.102,
        "ml.c5.xlarge": 0.204,
        "ml.p3.2xlarge": 3.825
    }
    
    hourly_rate = hourly_rates.get(instance_type, 0.115)  # Default to m5.large
    
    monthly_cost = hourly_rate * instance_count * hours_per_month
    annual_cost = monthly_cost * 12
    
    return {
        'hourly_rate_per_instance': hourly_rate,
        'monthly_cost': monthly_cost,
        'annual_cost': annual_cost,
        'currency': 'USD'
    }

# Module initialization
if AWS_AVAILABLE:
    logger.info("AWS SageMaker deployment module loaded successfully")
else:
    logger.warning("AWS SDK not available - install boto3 for full functionality")

logger.info(f"Available instance types: {len(get_available_instance_types())}")
