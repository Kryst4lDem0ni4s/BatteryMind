"""
BatteryMind - AWS SageMaker Endpoint Configuration

Production-ready configuration management for deploying BatteryMind AI/ML models
on AWS SageMaker with auto-scaling, monitoring, and multi-model endpoints.

Features:
- Multi-model endpoint configuration
- Auto-scaling configuration
- A/B testing setup
- Real-time and batch transform endpoints
- Custom inference containers
- Model versioning and rollback
- Performance monitoring integration

Author: BatteryMind Development Team
Version: 1.0.0
"""

import json
import boto3
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import time
import uuid
from pathlib import Path

# AWS SDK imports
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.pytorch import PyTorchModel
from sagemaker.tensorflow import TensorFlowModel
from sagemaker.sklearn import SKLearnModel

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...utils.config_parser import ConfigParser
from ...utils.aws_helpers import AWSHelper

# Configure logging
logger = setup_logger(__name__)

@dataclass
class EndpointConfiguration:
    """Configuration for SageMaker endpoint deployment."""
    
    # Endpoint identification
    endpoint_name: str
    model_name: str
    config_name: str
    
    # Instance configuration
    instance_type: str = "ml.m5.large"
    initial_instance_count: int = 1
    max_instance_count: int = 10
    min_instance_count: int = 1
    
    # Auto-scaling configuration
    enable_auto_scaling: bool = True
    target_invocations_per_instance: int = 100
    scale_up_cooldown: int = 300  # seconds
    scale_down_cooldown: int = 300  # seconds
    
    # Performance configuration
    max_concurrent_transforms: int = 1
    max_payload_mb: int = 6
    batch_strategy: str = "MultiRecord"
    
    # Model configuration
    model_data_url: str = ""
    image_uri: str = ""
    role_arn: str = ""
    environment_vars: Dict[str, str] = field(default_factory=dict)
    
    # Security configuration
    enable_network_isolation: bool = False
    vpc_config: Optional[Dict[str, Any]] = None
    
    # Monitoring configuration
    enable_data_capture: bool = True
    data_capture_sampling_percentage: int = 100
    
    # Multi-model configuration
    enable_multi_model: bool = False
    model_cache_size: int = 1000
    
    # A/B testing configuration
    enable_ab_testing: bool = False
    production_variant_weight: int = 90
    canary_variant_weight: int = 10

class SageMakerEndpointConfig:
    """
    AWS SageMaker endpoint configuration manager for BatteryMind models.
    """
    
    def __init__(self, aws_region: str = "us-west-2"):
        self.aws_region = aws_region
        self.sagemaker_session = sagemaker.Session()
        self.sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
        self.aws_helper = AWSHelper(region=aws_region)
        
        # Default configurations
        self.default_configs = self._load_default_configurations()
        
        # Endpoint tracking
        self.active_endpoints = {}
        self.deployment_history = []
        
        logger.info(f"SageMaker endpoint configurator initialized for region: {aws_region}")
    
    def _load_default_configurations(self) -> Dict[str, EndpointConfiguration]:
        """Load default endpoint configurations for different model types."""
        return {
            "transformer_battery_health": EndpointConfiguration(
                endpoint_name="batterymind-transformer-health",
                model_name="transformer-battery-health-v1",
                config_name="transformer-health-config",
                instance_type="ml.m5.xlarge",
                initial_instance_count=2,
                max_instance_count=20,
                enable_auto_scaling=True,
                target_invocations_per_instance=200,
                environment_vars={
                    "MODEL_TYPE": "transformer",
                    "BATCH_SIZE": "32",
                    "MAX_SEQUENCE_LENGTH": "100"
                }
            ),
            
            "federated_global_model": EndpointConfiguration(
                endpoint_name="batterymind-federated-global",
                model_name="federated-global-model-v1",
                config_name="federated-global-config",
                instance_type="ml.m5.large",
                initial_instance_count=1,
                max_instance_count=5,
                enable_auto_scaling=True,
                target_invocations_per_instance=150,
                environment_vars={
                    "MODEL_TYPE": "federated",
                    "PRIVACY_MODE": "differential_privacy"
                }
            ),
            
            "rl_charging_optimizer": EndpointConfiguration(
                endpoint_name="batterymind-rl-optimizer",
                model_name="rl-charging-optimizer-v1",
                config_name="rl-optimizer-config",
                instance_type="ml.c5.large",
                initial_instance_count=1,
                max_instance_count=8,
                enable_auto_scaling=True,
                target_invocations_per_instance=300,
                environment_vars={
                    "MODEL_TYPE": "reinforcement_learning",
                    "ACTION_SPACE_SIZE": "4",
                    "OBSERVATION_SPACE_SIZE": "12"
                }
            ),
            
            "ensemble_predictor": EndpointConfiguration(
                endpoint_name="batterymind-ensemble",
                model_name="ensemble-predictor-v1",
                config_name="ensemble-config",
                instance_type="ml.m5.2xlarge",
                initial_instance_count=2,
                max_instance_count=15,
                enable_auto_scaling=True,
                target_invocations_per_instance=100,
                enable_multi_model=True,
                environment_vars={
                    "MODEL_TYPE": "ensemble",
                    "ENSEMBLE_STRATEGY": "voting",
                    "BASE_MODELS": "transformer,federated,rl"
                }
            )
        }
    
    def create_endpoint_config(self, 
                              model_type: str,
                              custom_config: Optional[EndpointConfiguration] = None) -> Dict[str, Any]:
        """
        Create SageMaker endpoint configuration.
        
        Args:
            model_type: Type of model (transformer, federated, rl, ensemble)
            custom_config: Optional custom configuration
            
        Returns:
            Dictionary containing endpoint configuration details
        """
        try:
            # Get configuration
            if custom_config:
                config = custom_config
            elif model_type in self.default_configs:
                config = self.default_configs[model_type]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Create production variant configuration
            production_variant = {
                'VariantName': 'production',
                'ModelName': config.model_name,
                'InitialInstanceCount': config.initial_instance_count,
                'InstanceType': config.instance_type,
                'InitialVariantWeight': config.production_variant_weight if config.enable_ab_testing else 1
            }
            
            # Production variant configurations
            production_variants = [production_variant]
            
            # Add canary variant for A/B testing
            if config.enable_ab_testing:
                canary_variant = {
                    'VariantName': 'canary',
                    'ModelName': f"{config.model_name}-canary",
                    'InitialInstanceCount': 1,
                    'InstanceType': config.instance_type,
                    'InitialVariantWeight': config.canary_variant_weight
                }
                production_variants.append(canary_variant)
            
            # Data capture configuration
            data_capture_config = None
            if config.enable_data_capture:
                data_capture_config = {
                    'EnableCapture': True,
                    'InitialSamplingPercentage': config.data_capture_sampling_percentage,
                    'DestinationS3Uri': f"s3://batterymind-model-data/data-capture/{config.endpoint_name}/",
                    'CaptureOptions': [
                        {'CaptureMode': 'Input'},
                        {'CaptureMode': 'Output'}
                    ]
                }
            
            # Create endpoint configuration
            endpoint_config = {
                'EndpointConfigName': config.config_name,
                'ProductionVariants': production_variants
            }
            
            # Add data capture if enabled
            if data_capture_config:
                endpoint_config['DataCaptureConfig'] = data_capture_config
            
            # Create the endpoint configuration
            response = self.sagemaker_client.create_endpoint_config(**endpoint_config)
            
            logger.info(f"Endpoint configuration created: {config.config_name}")
            
            return {
                'config_name': config.config_name,
                'config_arn': response['EndpointConfigArn'],
                'model_type': model_type,
                'configuration': config,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create endpoint configuration: {e}")
            raise
    
    def create_multi_model_config(self, 
                                 models: Dict[str, str],
                                 config_name: str,
                                 instance_type: str = "ml.m5.2xlarge") -> Dict[str, Any]:
        """
        Create multi-model endpoint configuration.
        
        Args:
            models: Dictionary of model names and their S3 locations
            config_name: Name for the endpoint configuration
            instance_type: EC2 instance type
            
        Returns:
            Multi-model endpoint configuration
        """
        try:
            # Multi-model configuration
            multi_model_config = {
                'EndpointConfigName': config_name,
                'ProductionVariants': [
                    {
                        'VariantName': 'multimodel',
                        'ModelName': f"{config_name}-multimodel",
                        'InitialInstanceCount': 1,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1
                    }
                ]
            }
            
            # Create the configuration
            response = self.sagemaker_client.create_endpoint_config(**multi_model_config)
            
            logger.info(f"Multi-model endpoint configuration created: {config_name}")
            
            return {
                'config_name': config_name,
                'config_arn': response['EndpointConfigArn'],
                'models': models,
                'instance_type': instance_type,
                'created_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create multi-model configuration: {e}")
            raise
    
    def setup_auto_scaling(self, 
                          endpoint_name: str,
                          config: EndpointConfiguration) -> Dict[str, Any]:
        """
        Setup auto-scaling for SageMaker endpoint.
        
        Args:
            endpoint_name: Name of the endpoint
            config: Endpoint configuration
            
        Returns:
            Auto-scaling configuration details
        """
        try:
            autoscaling_client = boto3.client('application-autoscaling', 
                                            region_name=self.aws_region)
            
            # Resource ID for the endpoint variant
            resource_id = f"endpoint/{endpoint_name}/variant/production"
            
            # Register scalable target
            autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=config.min_instance_count,
                MaxCapacity=config.max_instance_count
            )
            
            # Create scaling policy
            policy_name = f"{endpoint_name}-scaling-policy"
            
            scaling_policy = autoscaling_client.put_scaling_policy(
                PolicyName=policy_name,
                ServiceNamespace='sagemaker',
                ResourceId=resource_id,
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': config.target_invocations_per_instance,
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                    },
                    'ScaleOutCooldown': config.scale_up_cooldown,
                    'ScaleInCooldown': config.scale_down_cooldown
                }
            )
            
            logger.info(f"Auto-scaling configured for endpoint: {endpoint_name}")
            
            return {
                'endpoint_name': endpoint_name,
                'resource_id': resource_id,
                'policy_arn': scaling_policy['PolicyARN'],
                'min_capacity': config.min_instance_count,
                'max_capacity': config.max_instance_count,
                'target_invocations': config.target_invocations_per_instance
            }
            
        except Exception as e:
            logger.error(f"Failed to setup auto-scaling: {e}")
            raise
    
    def create_batch_transform_config(self, 
                                    model_name: str,
                                    job_name: str,
                                    input_s3_uri: str,
                                    output_s3_uri: str,
                                    instance_type: str = "ml.m5.large") -> Dict[str, Any]:
        """
        Create batch transform job configuration.
        
        Args:
            model_name: Name of the model
            job_name: Name for the batch transform job
            input_s3_uri: S3 URI for input data
            output_s3_uri: S3 URI for output data
            instance_type: EC2 instance type
            
        Returns:
            Batch transform configuration
        """
        try:
            transform_config = {
                'TransformJobName': job_name,
                'ModelName': model_name,
                'TransformInput': {
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': input_s3_uri
                        }
                    },
                    'ContentType': 'application/json',
                    'SplitType': 'Line'
                },
                'TransformOutput': {
                    'S3OutputPath': output_s3_uri,
                    'Accept': 'application/json'
                },
                'TransformResources': {
                    'InstanceType': instance_type,
                    'InstanceCount': 1
                }
            }
            
            logger.info(f"Batch transform configuration created: {job_name}")
            
            return transform_config
            
        except Exception as e:
            logger.error(f"Failed to create batch transform configuration: {e}")
            raise
    
    def get_endpoint_config(self, config_name: str) -> Dict[str, Any]:
        """
        Get existing endpoint configuration.
        
        Args:
            config_name: Name of the configuration
            
        Returns:
            Endpoint configuration details
        """
        try:
            response = self.sagemaker_client.describe_endpoint_config(
                EndpointConfigName=config_name
            )
            
            return {
                'config_name': response['EndpointConfigName'],
                'config_arn': response['EndpointConfigArn'],
                'production_variants': response['ProductionVariants'],
                'data_capture_config': response.get('DataCaptureConfig'),
                'creation_time': response['CreationTime'].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get endpoint configuration: {e}")
            raise
    
    def update_endpoint_config(self, 
                              config_name: str,
                              updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update endpoint configuration (creates new version).
        
        Args:
            config_name: Name of the configuration
            updates: Configuration updates
            
        Returns:
            Updated configuration details
        """
        try:
            # Get existing configuration
            existing_config = self.get_endpoint_config(config_name)
            
            # Create new configuration name with timestamp
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            new_config_name = f"{config_name}-{timestamp}"
            
            # Merge updates with existing configuration
            updated_config = existing_config.copy()
            updated_config.update(updates)
            updated_config['EndpointConfigName'] = new_config_name
            
            # Remove read-only fields
            for field in ['EndpointConfigArn', 'CreationTime']:
                updated_config.pop(field, None)
            
            # Create new configuration
            response = self.sagemaker_client.create_endpoint_config(**updated_config)
            
            logger.info(f"Endpoint configuration updated: {new_config_name}")
            
            return {
                'old_config_name': config_name,
                'new_config_name': new_config_name,
                'config_arn': response['EndpointConfigArn'],
                'updates_applied': updates
            }
            
        except Exception as e:
            logger.error(f"Failed to update endpoint configuration: {e}")
            raise
    
    def delete_endpoint_config(self, config_name: str) -> bool:
        """
        Delete endpoint configuration.
        
        Args:
            config_name: Name of the configuration to delete
            
        Returns:
            True if successful
        """
        try:
            self.sagemaker_client.delete_endpoint_config(
                EndpointConfigName=config_name
            )
            
            logger.info(f"Endpoint configuration deleted: {config_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete endpoint configuration: {e}")
            return False
    
    def list_endpoint_configs(self, 
                             name_contains: Optional[str] = None,
                             max_results: int = 10) -> List[Dict[str, Any]]:
        """
        List endpoint configurations.
        
        Args:
            name_contains: Filter by name substring
            max_results: Maximum number of results
            
        Returns:
            List of endpoint configurations
        """
        try:
            params = {
                'MaxResults': max_results,
                'SortBy': 'CreationTime',
                'SortOrder': 'Descending'
            }
            
            if name_contains:
                params['NameContains'] = name_contains
            
            response = self.sagemaker_client.list_endpoint_configs(**params)
            
            configs = []
            for config in response['EndpointConfigs']:
                configs.append({
                    'config_name': config['EndpointConfigName'],
                    'config_arn': config['EndpointConfigArn'],
                    'creation_time': config['CreationTime'].isoformat()
                })
            
            return configs
            
        except Exception as e:
            logger.error(f"Failed to list endpoint configurations: {e}")
            return []
    
    def validate_configuration(self, config: EndpointConfiguration) -> Dict[str, Any]:
        """
        Validate endpoint configuration.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Validation results
        """
        validation_results = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Validate instance type
        valid_instance_types = [
            'ml.t2.medium', 'ml.t2.large', 'ml.t2.xlarge', 'ml.t2.2xlarge',
            'ml.m5.large', 'ml.m5.xlarge', 'ml.m5.2xlarge', 'ml.m5.4xlarge',
            'ml.c5.large', 'ml.c5.xlarge', 'ml.c5.2xlarge', 'ml.c5.4xlarge',
            'ml.p3.2xlarge', 'ml.p3.8xlarge', 'ml.p3.16xlarge'
        ]
        
        if config.instance_type not in valid_instance_types:
            validation_results['errors'].append(f"Invalid instance type: {config.instance_type}")
            validation_results['is_valid'] = False
        
        # Validate instance counts
        if config.initial_instance_count < 1:
            validation_results['errors'].append("Initial instance count must be at least 1")
            validation_results['is_valid'] = False
        
        if config.max_instance_count < config.min_instance_count:
            validation_results['errors'].append("Max instance count cannot be less than min instance count")
            validation_results['is_valid'] = False
        
        # Performance recommendations
        if config.instance_type.startswith('ml.t2') and config.enable_auto_scaling:
            validation_results['warnings'].append("T2 instances may not be ideal for auto-scaling workloads")
        
        if config.target_invocations_per_instance > 1000:
            validation_results['recommendations'].append("Consider using larger instance types for high invocation rates")
        
        return validation_results
    
    def get_configuration_costs(self, config: EndpointConfiguration, 
                               hours_per_month: int = 730) -> Dict[str, float]:
        """
        Estimate monthly costs for endpoint configuration.
        
        Args:
            config: Endpoint configuration
            hours_per_month: Hours per month (default: 730)
            
        Returns:
            Cost estimates
        """
        # Simplified cost calculation (actual costs may vary)
        instance_costs = {
            'ml.t2.medium': 0.047,
            'ml.t2.large': 0.094,
            'ml.t2.xlarge': 0.188,
            'ml.t2.2xlarge': 0.376,
            'ml.m5.large': 0.115,
            'ml.m5.xlarge': 0.23,
            'ml.m5.2xlarge': 0.46,
            'ml.m5.4xlarge': 0.92,
            'ml.c5.large': 0.102,
            'ml.c5.xlarge': 0.204,
            'ml.c5.2xlarge': 0.408,
            'ml.c5.4xlarge': 0.816,
            'ml.p3.2xlarge': 3.825,
            'ml.p3.8xlarge': 15.3,
            'ml.p3.16xlarge': 30.6
        }
        
        hourly_rate = instance_costs.get(config.instance_type, 0.115)  # Default to m5.large
        
        # Minimum cost (initial instances)
        min_monthly_cost = hourly_rate * config.initial_instance_count * hours_per_month
        
        # Maximum cost (all instances)
        max_monthly_cost = hourly_rate * config.max_instance_count * hours_per_month
        
        # Estimated cost (assume 70% of max capacity on average)
        estimated_monthly_cost = min_monthly_cost + (max_monthly_cost - min_monthly_cost) * 0.7
        
        return {
            'minimum_monthly_cost': min_monthly_cost,
            'maximum_monthly_cost': max_monthly_cost,
            'estimated_monthly_cost': estimated_monthly_cost,
            'hourly_rate_per_instance': hourly_rate,
            'currency': 'USD'
        }
