"""
BatteryMind - AWS SageMaker Model Deployment

Production-ready deployment orchestrator for BatteryMind AI/ML models on AWS SageMaker
with comprehensive monitoring, rollback capabilities, and multi-environment support.

Features:
- Automated model deployment pipeline
- Blue-green deployment strategy
- Canary deployments for A/B testing
- Model versioning and rollback
- Health monitoring and alerting
- Multi-environment deployment (dev, staging, prod)
- Custom inference containers
- Real-time and batch inference endpoints

Author: BatteryMind Development Team
Version: 1.0.0
"""

import json
import boto3
import time
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import tarfile
import tempfile

# AWS SDK imports
import sagemaker
from sagemaker.model import Model
from sagemaker.predictor import Predictor
from sagemaker.pytorch import PyTorchModel, PyTorchPredictor
from sagemaker.tensorflow import TensorFlowModel, TensorFlowPredictor
from sagemaker.sklearn import SKLearnModel, SKLearnPredictor
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

# BatteryMind imports
from .endpoint_config import SageMakerEndpointConfig, EndpointConfiguration
from ...utils.logging_utils import setup_logger
from ...utils.aws_helpers import AWSHelper
from ...utils.config_parser import ConfigParser

# Configure logging
logger = setup_logger(__name__)

@dataclass
class DeploymentConfiguration:
    """Configuration for model deployment."""
    
    # Deployment identification
    deployment_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    model_version: str = "1.0.0"
    environment: str = "staging"  # dev, staging, prod
    
    # Model artifacts
    model_data_url: str = ""
    source_dir: str = ""
    entry_point: str = "inference.py"
    framework: str = "pytorch"  # pytorch, tensorflow, sklearn, custom
    
    # Deployment strategy
    deployment_strategy: str = "blue_green"  # blue_green, canary, rolling
    rollback_enabled: bool = True
    health_check_grace_period: int = 300  # seconds
    
    # Endpoint configuration
    endpoint_config: Optional[EndpointConfiguration] = None
    
    # Custom container configuration
    custom_image_uri: Optional[str] = None
    model_server_timeout: int = 60
    model_server_workers: int = 1
    
    # Monitoring configuration
    enable_monitoring: bool = True
    cloudwatch_log_level: str = "INFO"
    
    # Notification configuration
    sns_topic_arn: Optional[str] = None
    
    # Tags
    tags: Dict[str, str] = field(default_factory=dict)

@dataclass
class DeploymentStatus:
    """Status of a deployment."""
    
    deployment_id: str
    status: str  # pending, in_progress, completed, failed, rolled_back
    start_time: datetime
    end_time: Optional[datetime] = None
    endpoint_name: Optional[str] = None
    endpoint_arn: Optional[str] = None
    error_message: Optional[str] = None
    health_check_status: str = "unknown"
    performance_metrics: Dict[str, float] = field(default_factory=dict)

class BatteryMindModelDeployer:
    """
    Comprehensive model deployment orchestrator for BatteryMind on AWS SageMaker.
    """
    
    def __init__(self, aws_region: str = "us-west-2"):
        self.aws_region = aws_region
        self.sagemaker_session = sagemaker.Session()
        self.sagemaker_client = boto3.client('sagemaker', region_name=aws_region)
        self.cloudwatch_client = boto3.client('cloudwatch', region_name=aws_region)
        self.sns_client = boto3.client('sns', region_name=aws_region)
        self.s3_client = boto3.client('s3', region_name=aws_region)
        
        # Initialize endpoint configurator
        self.endpoint_config_manager = SageMakerEndpointConfig(aws_region)
        self.aws_helper = AWSHelper(region=aws_region)
        
        # Deployment tracking
        self.active_deployments = {}
        self.deployment_history = []
        
        # Default configurations
        self.default_role_arn = self._get_default_execution_role()
        
        logger.info(f"Model deployer initialized for region: {aws_region}")
    
    def _get_default_execution_role(self) -> str:
        """Get default SageMaker execution role."""
        try:
            return sagemaker.get_execution_role()
        except Exception:
            # Fallback to a default role ARN pattern
            account_id = boto3.client('sts').get_caller_identity()['Account']
            return f"arn:aws:iam::{account_id}:role/SageMakerExecutionRole"
    
    def deploy_model(self, 
                     deployment_config: DeploymentConfiguration,
                     wait_for_completion: bool = True) -> DeploymentStatus:
        """
        Deploy a model to SageMaker endpoint.
        
        Args:
            deployment_config: Deployment configuration
            wait_for_completion: Whether to wait for deployment completion
            
        Returns:
            DeploymentStatus: Status of the deployment
        """
        deployment_status = DeploymentStatus(
            deployment_id=deployment_config.deployment_id,
            status="pending",
            start_time=datetime.now()
        )
        
        try:
            logger.info(f"Starting deployment {deployment_config.deployment_id}")
            
            # Validate deployment configuration
            self._validate_deployment_config(deployment_config)
            
            # Update status
            deployment_status.status = "in_progress"
            self.active_deployments[deployment_config.deployment_id] = deployment_status
            
            # Create model
            model = self._create_sagemaker_model(deployment_config)
            
            # Create endpoint configuration
            endpoint_config = self._create_or_update_endpoint_config(deployment_config)
            
            # Deploy based on strategy
            if deployment_config.deployment_strategy == "blue_green":
                endpoint_name = self._deploy_blue_green(deployment_config, model, endpoint_config)
            elif deployment_config.deployment_strategy == "canary":
                endpoint_name = self._deploy_canary(deployment_config, model, endpoint_config)
            elif deployment_config.deployment_strategy == "rolling":
                endpoint_name = self._deploy_rolling(deployment_config, model, endpoint_config)
            else:
                raise ValueError(f"Unknown deployment strategy: {deployment_config.deployment_strategy}")
            
            # Update deployment status
            deployment_status.endpoint_name = endpoint_name
            
            # Wait for endpoint to be in service
            if wait_for_completion:
                self._wait_for_endpoint(endpoint_name, deployment_status)
            
            # Perform health checks
            if deployment_status.status != "failed":
                self._perform_health_checks(endpoint_name, deployment_status)
            
            # Setup monitoring
            if deployment_config.enable_monitoring and deployment_status.status == "completed":
                self._setup_endpoint_monitoring(endpoint_name, deployment_config)
            
            # Send notification
            if deployment_config.sns_topic_arn:
                self._send_deployment_notification(deployment_config, deployment_status)
            
            # Record deployment history
            self.deployment_history.append({
                'deployment_id': deployment_config.deployment_id,
                'model_name': deployment_config.model_name,
                'environment': deployment_config.environment,
                'status': deployment_status.status,
                'timestamp': deployment_status.start_time.isoformat(),
                'endpoint_name': endpoint_name
            })
            
            logger.info(f"Deployment {deployment_config.deployment_id} completed with status: {deployment_status.status}")
            
        except Exception as e:
            logger.error(f"Deployment {deployment_config.deployment_id} failed: {e}")
            deployment_status.status = "failed"
            deployment_status.error_message = str(e)
            deployment_status.end_time = datetime.now()
            
            # Attempt rollback if enabled
            if deployment_config.rollback_enabled:
                self._attempt_rollback(deployment_config, deployment_status)
        
        finally:
            # Update final status
            if deployment_status.end_time is None:
                deployment_status.end_time = datetime.now()
            
            self.active_deployments[deployment_config.deployment_id] = deployment_status
        
        return deployment_status
    
    def _validate_deployment_config(self, config: DeploymentConfiguration):
        """Validate deployment configuration."""
        if not config.model_name:
            raise ValueError("Model name is required")
        
        if not config.model_data_url and not config.custom_image_uri:
            raise ValueError("Either model_data_url or custom_image_uri is required")
        
        if config.framework not in ["pytorch", "tensorflow", "sklearn", "custom"]:
            raise ValueError(f"Unsupported framework: {config.framework}")
        
        if config.environment not in ["dev", "staging", "prod"]:
            raise ValueError(f"Invalid environment: {config.environment}")
    
    def _create_sagemaker_model(self, config: DeploymentConfiguration) -> Model:
        """Create SageMaker model object."""
        try:
            model_name = f"{config.model_name}-{config.model_version}-{int(time.time())}"
            
            # Common model parameters
            model_params = {
                'name': model_name,
                'role': self.default_role_arn,
                'sagemaker_session': self.sagemaker_session
            }
            
            # Add environment variables
            if config.endpoint_config and config.endpoint_config.environment_vars:
                model_params['env'] = config.endpoint_config.environment_vars
            
            # Create model based on framework
            if config.framework == "pytorch":
                model = PyTorchModel(
                    model_data=config.model_data_url,
                    entry_point=config.entry_point,
                    source_dir=config.source_dir,
                    framework_version="1.12.0",
                    py_version="py38",
                    **model_params
                )
            
            elif config.framework == "tensorflow":
                model = TensorFlowModel(
                    model_data=config.model_data_url,
                    entry_point=config.entry_point,
                    source_dir=config.source_dir,
                    framework_version="2.8.0",
                    py_version="py39",
                    **model_params
                )
            
            elif config.framework == "sklearn":
                model = SKLearnModel(
                    model_data=config.model_data_url,
                    entry_point=config.entry_point,
                    source_dir=config.source_dir,
                    framework_version="1.0-1",
                    py_version="py3",
                    **model_params
                )
            
            elif config.framework == "custom":
                if not config.custom_image_uri:
                    raise ValueError("Custom image URI required for custom framework")
                
                model = Model(
                    image_uri=config.custom_image_uri,
                    model_data=config.model_data_url,
                    **model_params
                )
            
            logger.info(f"SageMaker model created: {model_name}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create SageMaker model: {e}")
            raise
    
    def _create_or_update_endpoint_config(self, config: DeploymentConfiguration) -> Dict[str, Any]:
        """Create or update endpoint configuration."""
        try:
            if config.endpoint_config:
                # Use provided endpoint configuration
                endpoint_config_result = self.endpoint_config_manager.create_endpoint_config(
                    model_type=config.framework,
                    custom_config=config.endpoint_config
                )
            else:
                # Use default configuration for model type
                endpoint_config_result = self.endpoint_config_manager.create_endpoint_config(
                    model_type=config.framework
                )
            
            return endpoint_config_result
            
        except Exception as e:
            logger.error(f"Failed to create endpoint configuration: {e}")
            raise
    
    def _deploy_blue_green(self, 
                          config: DeploymentConfiguration, 
                          model: Model, 
                          endpoint_config: Dict[str, Any]) -> str:
        """Deploy using blue-green strategy."""
        try:
            endpoint_name = f"{config.model_name}-{config.environment}"
            
            # Check if endpoint already exists
            existing_endpoint = self._get_endpoint_info(endpoint_name)
            
            if existing_endpoint:
                # Update existing endpoint (blue-green)
                logger.info(f"Updating existing endpoint: {endpoint_name}")
                
                # Create new endpoint configuration
                new_config_name = f"{endpoint_config['config_name']}-{int(time.time())}"
                
                # Update endpoint with new configuration
                self.sagemaker_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=new_config_name
                )
            else:
                # Create new endpoint
                logger.info(f"Creating new endpoint: {endpoint_name}")
                
                self.sagemaker_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config['config_name']
                )
            
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Blue-green deployment failed: {e}")
            raise
    
    def _deploy_canary(self, 
                      config: DeploymentConfiguration, 
                      model: Model, 
                      endpoint_config: Dict[str, Any]) -> str:
        """Deploy using canary strategy."""
        try:
            endpoint_name = f"{config.model_name}-{config.environment}"
            
            # For canary deployment, we need A/B testing configuration
            if not config.endpoint_config or not config.endpoint_config.enable_ab_testing:
                raise ValueError("Canary deployment requires A/B testing configuration")
            
            # Create endpoint with canary variant
            if not self._get_endpoint_info(endpoint_name):
                self.sagemaker_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config['config_name']
                )
            else:
                # Update endpoint to include canary variant
                self.sagemaker_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config['config_name']
                )
            
            logger.info(f"Canary deployment initiated for: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Canary deployment failed: {e}")
            raise
    
    def _deploy_rolling(self, 
                       config: DeploymentConfiguration, 
                       model: Model, 
                       endpoint_config: Dict[str, Any]) -> str:
        """Deploy using rolling update strategy."""
        try:
            endpoint_name = f"{config.model_name}-{config.environment}"
            
            # Rolling deployment gradually updates instances
            if self._get_endpoint_info(endpoint_name):
                # Update existing endpoint gradually
                self.sagemaker_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config['config_name'],
                    RetainAllVariantProperties=False
                )
            else:
                # Create new endpoint
                self.sagemaker_client.create_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=endpoint_config['config_name']
                )
            
            logger.info(f"Rolling deployment initiated for: {endpoint_name}")
            return endpoint_name
            
        except Exception as e:
            logger.error(f"Rolling deployment failed: {e}")
            raise
    
    def _wait_for_endpoint(self, endpoint_name: str, deployment_status: DeploymentStatus):
        """Wait for endpoint to be in service."""
        try:
            logger.info(f"Waiting for endpoint {endpoint_name} to be in service...")
            
            waiter = self.sagemaker_client.get_waiter('endpoint_in_service')
            waiter.wait(
                EndpointName=endpoint_name,
                WaiterConfig={
                    'Delay': 30,
                    'MaxAttempts': 60  # 30 minutes maximum wait time
                }
            )
            
            # Get endpoint info
            endpoint_info = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            deployment_status.endpoint_arn = endpoint_info['EndpointArn']
            deployment_status.status = "completed"
            
            logger.info(f"Endpoint {endpoint_name} is now in service")
            
        except Exception as e:
            logger.error(f"Endpoint {endpoint_name} failed to come into service: {e}")
            deployment_status.status = "failed"
            deployment_status.error_message = f"Endpoint failed to start: {e}"
            raise
    
    def _perform_health_checks(self, endpoint_name: str, deployment_status: DeploymentStatus):
        """Perform health checks on deployed endpoint."""
        try:
            logger.info(f"Performing health checks for endpoint: {endpoint_name}")
            
            # Create predictor for health checks
            predictor = Predictor(
                endpoint_name=endpoint_name,
                sagemaker_session=self.sagemaker_session,
                serializer=JSONSerializer(),
                deserializer=JSONDeserializer()
            )
            
            # Test prediction with sample data
            sample_data = self._get_sample_test_data()
            
            start_time = time.time()
            try:
                response = predictor.predict(sample_data)
                inference_time = time.time() - start_time
                
                # Update performance metrics
                deployment_status.performance_metrics = {
                    'inference_time_ms': inference_time * 1000,
                    'health_check_success': True,
                    'response_size_bytes': len(str(response))
                }
                
                deployment_status.health_check_status = "healthy"
                logger.info(f"Health check passed for endpoint: {endpoint_name}")
                
            except Exception as e:
                deployment_status.health_check_status = "unhealthy"
                deployment_status.performance_metrics['health_check_success'] = False
                logger.warning(f"Health check failed for endpoint {endpoint_name}: {e}")
            
        except Exception as e:
            logger.error(f"Health check setup failed: {e}")
            deployment_status.health_check_status = "unknown"
    
    def _get_sample_test_data(self) -> Dict[str, Any]:
        """Get sample test data for health checks."""
        # Sample battery data for testing
        return {
            "voltage": 3.7,
            "current": 2.5,
            "temperature": 25.0,
            "soc": 0.8,
            "cycles": 100,
            "age_days": 365
        }
    
    def _setup_endpoint_monitoring(self, endpoint_name: str, config: DeploymentConfiguration):
        """Setup CloudWatch monitoring for endpoint."""
        try:
            logger.info(f"Setting up monitoring for endpoint: {endpoint_name}")
            
            # Create CloudWatch alarms
            alarms = [
                {
                    'AlarmName': f"{endpoint_name}-HighLatency",
                    'MetricName': 'ModelLatency',
                    'Threshold': 5000,  # 5 seconds
                    'ComparisonOperator': 'GreaterThanThreshold'
                },
                {
                    'AlarmName': f"{endpoint_name}-HighErrorRate",
                    'MetricName': 'Invocation4XXErrors',
                    'Threshold': 10,
                    'ComparisonOperator': 'GreaterThanThreshold'
                },
                {
                    'AlarmName': f"{endpoint_name}-LowInvocations",
                    'MetricName': 'Invocations',
                    'Threshold': 1,
                    'ComparisonOperator': 'LessThanThreshold'
                }
            ]
            
            for alarm in alarms:
                self.cloudwatch_client.put_metric_alarm(
                    AlarmName=alarm['AlarmName'],
                    ComparisonOperator=alarm['ComparisonOperator'],
                    EvaluationPeriods=2,
                    MetricName=alarm['MetricName'],
                    Namespace='AWS/SageMaker',
                    Period=300,
                    Statistic='Average',
                    Threshold=alarm['Threshold'],
                    ActionsEnabled=True,
                    AlarmActions=[config.sns_topic_arn] if config.sns_topic_arn else [],
                    AlarmDescription=f"Alarm for {endpoint_name}",
                    Dimensions=[
                        {
                            'Name': 'EndpointName',
                            'Value': endpoint_name
                        }
                    ]
                )
            
            logger.info(f"Monitoring setup completed for endpoint: {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup monitoring: {e}")
    
    def _send_deployment_notification(self, config: DeploymentConfiguration, status: DeploymentStatus):
        """Send deployment notification via SNS."""
        try:
            if not config.sns_topic_arn:
                return
            
            message = {
                'deployment_id': config.deployment_id,
                'model_name': config.model_name,
                'environment': config.environment,
                'status': status.status,
                'endpoint_name': status.endpoint_name,
                'timestamp': status.start_time.isoformat()
            }
            
            if status.error_message:
                message['error_message'] = status.error_message
            
            self.sns_client.publish(
                TopicArn=config.sns_topic_arn,
                Message=json.dumps(message, indent=2),
                Subject=f"BatteryMind Model Deployment {status.status.title()}"
            )
            
            logger.info(f"Deployment notification sent for: {config.deployment_id}")
            
        except Exception as e:
            logger.error(f"Failed to send notification: {e}")
    
    def _attempt_rollback(self, config: DeploymentConfiguration, status: DeploymentStatus):
        """Attempt to rollback failed deployment."""
        try:
            logger.info(f"Attempting rollback for deployment: {config.deployment_id}")
            
            endpoint_name = f"{config.model_name}-{config.environment}"
            
            # Get previous endpoint configuration
            previous_config = self._get_previous_endpoint_config(endpoint_name)
            
            if previous_config:
                # Rollback to previous configuration
                self.sagemaker_client.update_endpoint(
                    EndpointName=endpoint_name,
                    EndpointConfigName=previous_config
                )
                
                status.status = "rolled_back"
                logger.info(f"Rollback initiated for: {config.deployment_id}")
            else:
                logger.warning(f"No previous configuration found for rollback: {endpoint_name}")
            
        except Exception as e:
            logger.error(f"Rollback failed: {e}")
    
    def _get_endpoint_info(self, endpoint_name: str) -> Optional[Dict[str, Any]]:
        """Get endpoint information."""
        try:
            response = self.sagemaker_client.describe_endpoint(EndpointName=endpoint_name)
            return response
        except self.sagemaker_client.exceptions.ClientError:
            return None
    
    def _get_previous_endpoint_config(self, endpoint_name: str) -> Optional[str]:
        """Get previous endpoint configuration name."""
        try:
            # This is a simplified implementation
            # In practice, you'd maintain a history of configurations
            endpoint_info = self._get_endpoint_info(endpoint_name)
            if endpoint_info:
                return endpoint_info.get('EndpointConfigName')
            return None
        except Exception:
            return None
    
    def get_deployment_status(self, deployment_id: str) -> Optional[DeploymentStatus]:
        """Get status of a specific deployment."""
        return self.active_deployments.get(deployment_id)
    
    def list_active_deployments(self) -> List[DeploymentStatus]:
        """List all active deployments."""
        return list(self.active_deployments.values())
    
    def get_deployment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get deployment history."""
        return self.deployment_history[-limit:]
    
    def delete_endpoint(self, endpoint_name: str) -> bool:
        """Delete SageMaker endpoint."""
        try:
            self.sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            logger.info(f"Endpoint deleted: {endpoint_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete endpoint {endpoint_name}: {e}")
            return False
    
    def create_batch_transform_job(self, 
                                  model_name: str,
                                  job_name: str,
                                  input_s3_uri: str,
                                  output_s3_uri: str,
                                  instance_type: str = "ml.m5.large") -> str:
        """Create batch transform job."""
        try:
            transform_config = self.endpoint_config_manager.create_batch_transform_config(
                model_name=model_name,
                job_name=job_name,
                input_s3_uri=input_s3_uri,
                output_s3_uri=output_s3_uri,
                instance_type=instance_type
            )
            
            response = self.sagemaker_client.create_transform_job(**transform_config)
            
            logger.info(f"Batch transform job created: {job_name}")
            return response['TransformJobArn']
            
        except Exception as e:
            logger.error(f"Failed to create batch transform job: {e}")
            raise
