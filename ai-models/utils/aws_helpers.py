"""
BatteryMind - AWS Helpers
Comprehensive AWS cloud service integration utilities for the BatteryMind
autonomous battery management system with IoT, ML, and blockchain capabilities.

Features:
- IoT Core device management and data ingestion
- SageMaker model deployment and inference
- DynamoDB data storage and retrieval
- S3 file storage and lifecycle management
- Kinesis real-time data streaming
- Lambda function integration
- CloudWatch monitoring and alerting
- Secrets Manager integration

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import json
import time
import asyncio
from typing import Dict, List, Optional, Any, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from urllib.parse import quote

# AWS SDK imports
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError, PartialCredentialsError
    from botocore.config import Config
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

# Additional AWS service imports
try:
    import boto3.session
    from boto3.dynamodb.conditions import Key, Attr
    from boto3.s3.transfer import TransferConfig
    BOTO3_EXTENDED = True
except ImportError:
    BOTO3_EXTENDED = False

# Data processing
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)

@dataclass
class AWSConfig:
    """AWS configuration settings."""
    
    region: str = "us-east-1"
    profile: Optional[str] = None
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    
    # Service-specific settings
    iot_endpoint: Optional[str] = None
    sagemaker_execution_role: Optional[str] = None
    lambda_role: Optional[str] = None
    
    # Performance settings
    max_pool_connections: int = 50
    retries: int = 3
    timeout: int = 30
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)

class AWSManager:
    """
    Centralized AWS service manager for BatteryMind.
    """
    
    def __init__(self, config: AWSConfig = None):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 is required for AWS integration")
        
        self.config = config or AWSConfig()
        
        # Initialize session
        self.session = self._create_session()
        
        # Initialize service clients
        self.clients = {}
        self.resources = {}
        
        # Performance monitoring
        self.request_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0.0
        }
        
        logger.info(f"AWS Manager initialized for region: {self.config.region}")
    
    def _create_session(self) -> boto3.Session:
        """Create configured AWS session."""
        try:
            session_kwargs = {}
            
            if self.config.profile:
                session_kwargs['profile_name'] = self.config.profile
            
            if self.config.access_key_id and self.config.secret_access_key:
                session_kwargs.update({
                    'aws_access_key_id': self.config.access_key_id,
                    'aws_secret_access_key': self.config.secret_access_key
                })
                
                if self.config.session_token:
                    session_kwargs['aws_session_token'] = self.config.session_token
            
            session_kwargs['region_name'] = self.config.region
            
            session = boto3.Session(**session_kwargs)
            
            # Test credentials
            sts = session.client('sts')
            identity = sts.get_caller_identity()
            logger.info(f"AWS session created for account: {identity.get('Account')}")
            
            return session
            
        except Exception as e:
            logger.error(f"Error creating AWS session: {e}")
            raise
    
    def get_client(self, service_name: str, **kwargs) -> Any:
        """Get or create AWS service client."""
        try:
            if service_name not in self.clients:
                client_config = Config(
                    max_pool_connections=self.config.max_pool_connections,
                    retries={'max_attempts': self.config.retries},
                    read_timeout=self.config.timeout,
                    connect_timeout=self.config.timeout
                )
                
                self.clients[service_name] = self.session.client(
                    service_name,
                    config=client_config,
                    **kwargs
                )
                
                logger.debug(f"Created AWS {service_name} client")
            
            return self.clients[service_name]
            
        except Exception as e:
            logger.error(f"Error creating {service_name} client: {e}")
            raise
    
    def get_resource(self, service_name: str, **kwargs) -> Any:
        """Get or create AWS service resource."""
        try:
            if service_name not in self.resources:
                self.resources[service_name] = self.session.resource(
                    service_name,
                    **kwargs
                )
                
                logger.debug(f"Created AWS {service_name} resource")
            
            return self.resources[service_name]
            
        except Exception as e:
            logger.error(f"Error creating {service_name} resource: {e}")
            raise

class IoTCoreManager:
    """
    AWS IoT Core integration for battery device management.
    """
    
    def __init__(self, aws_manager: AWSManager):
        self.aws_manager = aws_manager
        self.iot_client = aws_manager.get_client('iot')
        self.iot_data_client = None
        
        # Device registry
        self.registered_devices = {}
        
        # Message processing
        self.message_handlers = {}
        
        logger.info("IoT Core Manager initialized")
    
    def _get_iot_data_client(self):
        """Get IoT Data client with endpoint."""
        if not self.iot_data_client:
            if not self.aws_manager.config.iot_endpoint:
                # Get endpoint from IoT Core
                response = self.iot_client.describe_endpoint(endpointType='iot:Data-ATS')
                endpoint = response['endpointAddress']
                self.aws_manager.config.iot_endpoint = f"https://{endpoint}"
            
            self.iot_data_client = self.aws_manager.get_client(
                'iot-data',
                endpoint_url=self.aws_manager.config.iot_endpoint
            )
        
        return self.iot_data_client
    
    def register_battery_device(self, 
                              device_id: str,
                              device_attributes: Dict[str, str] = None) -> Dict[str, Any]:
        """Register a battery device in IoT Core."""
        try:
            # Create thing
            thing_response = self.iot_client.create_thing(
                thingName=device_id,
                attributePayload={
                    'attributes': device_attributes or {},
                    'merge': False
                }
            )
            
            # Create certificate
            cert_response = self.iot_client.create_keys_and_certificate(
                setAsActive=True
            )
            
            # Create policy
            policy_name = f"{device_id}_policy"
            policy_document = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Action": [
                            "iot:Connect",
                            "iot:Publish",
                            "iot:Subscribe",
                            "iot:Receive"
                        ],
                        "Resource": [
                            f"arn:aws:iot:{self.aws_manager.config.region}:*:client/{device_id}",
                            f"arn:aws:iot:{self.aws_manager.config.region}:*:topic/battery/{device_id}/*",
                            f"arn:aws:iot:{self.aws_manager.config.region}:*:topicfilter/battery/{device_id}/*"
                        ]
                    }
                ]
            }
            
            self.iot_client.create_policy(
                policyName=policy_name,
                policyDocument=json.dumps(policy_document)
            )
            
            # Attach policy to certificate
            self.iot_client.attach_policy(
                policyName=policy_name,
                target=cert_response['certificateArn']
            )
            
            # Attach certificate to thing
            self.iot_client.attach_thing_principal(
                thingName=device_id,
                principal=cert_response['certificateArn']
            )
            
            device_info = {
                'device_id': device_id,
                'thing_arn': thing_response['thingArn'],
                'certificate_arn': cert_response['certificateArn'],
                'certificate_pem': cert_response['certificatePem'],
                'private_key': cert_response['keyPair']['PrivateKey'],
                'public_key': cert_response['keyPair']['PublicKey'],
                'policy_name': policy_name,
                'created_at': datetime.now().isoformat()
            }
            
            self.registered_devices[device_id] = device_info
            
            logger.info(f"Battery device registered: {device_id}")
            
            return device_info
            
        except Exception as e:
            logger.error(f"Error registering battery device: {e}")
            raise
    
    def publish_battery_data(self, 
                           device_id: str,
                           telemetry_data: Dict[str, Any],
                           topic_suffix: str = "telemetry") -> bool:
        """Publish battery telemetry data to IoT Core."""
        try:
            iot_data_client = self._get_iot_data_client()
            
            topic = f"battery/{device_id}/{topic_suffix}"
            
            # Add metadata
            message = {
                'device_id': device_id,
                'timestamp': datetime.now().isoformat(),
                'data': telemetry_data
            }
            
            iot_data_client.publish(
                topic=topic,
                qos=1,
                payload=json.dumps(message)
            )
            
            logger.debug(f"Published data for device {device_id} to topic {topic}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error publishing battery data: {e}")
            return False
    
    def create_battery_fleet_rule(self, 
                                fleet_id: str,
                                target_action: Dict[str, Any]) -> str:
        """Create IoT rule for battery fleet data processing."""
        try:
            rule_name = f"battery_fleet_{fleet_id}_rule"
            
            sql_statement = f"""
                SELECT * FROM 'topic/battery/+/telemetry' 
                WHERE fleet_id = '{fleet_id}'
            """
            
            rule_payload = {
                'sql': sql_statement,
                'description': f'Battery fleet {fleet_id} data processing rule',
                'actions': [target_action],
                'ruleDisabled': False
            }
            
            self.iot_client.create_topic_rule(
                ruleName=rule_name,
                topicRulePayload=rule_payload
            )
            
            logger.info(f"Created IoT rule for fleet {fleet_id}: {rule_name}")
            
            return rule_name
            
        except Exception as e:
            logger.error(f"Error creating fleet rule: {e}")
            raise
    
    def get_device_shadow(self, device_id: str) -> Dict[str, Any]:
        """Get device shadow state."""
        try:
            iot_data_client = self._get_iot_data_client()
            
            response = iot_data_client.get_thing_shadow(thingName=device_id)
            
            shadow_data = json.loads(response['payload'].read())
            
            return shadow_data
            
        except Exception as e:
            logger.error(f"Error getting device shadow: {e}")
            return {}
    
    def update_device_shadow(self, 
                           device_id: str,
                           desired_state: Dict[str, Any]) -> bool:
        """Update device shadow desired state."""
        try:
            iot_data_client = self._get_iot_data_client()
            
            shadow_update = {
                'state': {
                    'desired': desired_state
                }
            }
            
            iot_data_client.update_thing_shadow(
                thingName=device_id,
                payload=json.dumps(shadow_update)
            )
            
            logger.info(f"Updated shadow for device {device_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating device shadow: {e}")
            return False

class SageMakerManager:
    """
    AWS SageMaker integration for ML model deployment and inference.
    """
    
    def __init__(self, aws_manager: AWSManager):
        self.aws_manager = aws_manager
        self.sagemaker_client = aws_manager.get_client('sagemaker')
        self.runtime_client = aws_manager.get_client('sagemaker-runtime')
        
        # Model registry
        self.deployed_models = {}
        
        logger.info("SageMaker Manager initialized")
    
    def deploy_battery_model(self, 
                           model_name: str,
                           model_s3_path: str,
                           instance_type: str = "ml.t2.medium",
                           initial_instance_count: int = 1) -> Dict[str, Any]:
        """Deploy battery health prediction model to SageMaker."""
        try:
            # Create model
            model_response = self.sagemaker_client.create_model(
                ModelName=model_name,
                PrimaryContainer={
                    'Image': self._get_inference_image(),
                    'ModelDataUrl': model_s3_path,
                    'Environment': {
                        'SAGEMAKER_PROGRAM': 'inference.py',
                        'SAGEMAKER_SUBMIT_DIRECTORY': model_s3_path
                    }
                },
                ExecutionRoleArn=self.aws_manager.config.sagemaker_execution_role
            )
            
            # Create endpoint configuration
            endpoint_config_name = f"{model_name}-config"
            
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[
                    {
                        'VariantName': 'primary',
                        'ModelName': model_name,
                        'InitialInstanceCount': initial_instance_count,
                        'InstanceType': instance_type,
                        'InitialVariantWeight': 1.0
                    }
                ]
            )
            
            # Create endpoint
            endpoint_name = f"{model_name}-endpoint"
            
            self.sagemaker_client.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            # Wait for endpoint to be in service
            self._wait_for_endpoint(endpoint_name)
            
            deployment_info = {
                'model_name': model_name,
                'endpoint_name': endpoint_name,
                'endpoint_config_name': endpoint_config_name,
                'instance_type': instance_type,
                'instance_count': initial_instance_count,
                'status': 'InService',
                'created_at': datetime.now().isoformat()
            }
            
            self.deployed_models[model_name] = deployment_info
            
            logger.info(f"Model deployed successfully: {model_name}")
            
            return deployment_info
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            raise
    
    def _get_inference_image(self) -> str:
        """Get the appropriate inference container image."""
        # Use the scikit-learn inference container for battery models
        account_id = {
            'us-east-1': '683313688378',
            'us-east-2': '257758044811',
            'us-west-2': '246618743249',
            'eu-west-1': '141502667606'
        }.get(self.aws_manager.config.region, '683313688378')
        
        return f"{account_id}.dkr.ecr.{self.aws_manager.config.region}.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3"
    
    def _wait_for_endpoint(self, endpoint_name: str, timeout: int = 600):
        """Wait for endpoint to be in service."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = self.sagemaker_client.describe_endpoint(
                    EndpointName=endpoint_name
                )
                
                status = response['EndpointStatus']
                
                if status == 'InService':
                    logger.info(f"Endpoint {endpoint_name} is in service")
                    return
                elif status == 'Failed':
                    failure_reason = response.get('FailureReason', 'Unknown')
                    raise RuntimeError(f"Endpoint creation failed: {failure_reason}")
                
                logger.info(f"Endpoint {endpoint_name} status: {status}")
                time.sleep(30)
                
            except Exception as e:
                logger.error(f"Error checking endpoint status: {e}")
                raise
        
        raise TimeoutError(f"Endpoint {endpoint_name} did not become ready within {timeout} seconds")
    
    def predict_battery_health(self, 
                             model_name: str,
                             input_data: Union[Dict[str, Any], List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Get battery health prediction from deployed model."""
        try:
            if model_name not in self.deployed_models:
                raise ValueError(f"Model {model_name} is not deployed")
            
            endpoint_name = self.deployed_models[model_name]['endpoint_name']
            
            # Prepare input data
            if isinstance(input_data, dict):
                input_data = [input_data]
            
            payload = json.dumps(input_data)
            
            # Make inference request
            response = self.runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            prediction_result = {
                'model_name': model_name,
                'endpoint_name': endpoint_name,
                'predictions': result,
                'timestamp': datetime.now().isoformat(),
                'latency_ms': response['ResponseMetadata']['HTTPHeaders'].get('x-amzn-invoked-production-variant-time', 0)
            }
            
            logger.debug(f"Battery health prediction completed for model {model_name}")
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def update_endpoint(self, 
                       model_name: str,
                       new_instance_count: int = None,
                       new_instance_type: str = None) -> bool:
        """Update endpoint configuration."""
        try:
            if model_name not in self.deployed_models:
                raise ValueError(f"Model {model_name} is not deployed")
            
            deployment_info = self.deployed_models[model_name]
            endpoint_name = deployment_info['endpoint_name']
            
            # Create new endpoint configuration
            new_config_name = f"{model_name}-config-{int(time.time())}"
            
            variant_config = {
                'VariantName': 'primary',
                'ModelName': model_name,
                'InitialInstanceCount': new_instance_count or deployment_info['instance_count'],
                'InstanceType': new_instance_type or deployment_info['instance_type'],
                'InitialVariantWeight': 1.0
            }
            
            self.sagemaker_client.create_endpoint_config(
                EndpointConfigName=new_config_name,
                ProductionVariants=[variant_config]
            )
            
            # Update endpoint
            self.sagemaker_client.update_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=new_config_name
            )
            
            # Wait for update to complete
            self._wait_for_endpoint(endpoint_name)
            
            # Update deployment info
            if new_instance_count:
                deployment_info['instance_count'] = new_instance_count
            if new_instance_type:
                deployment_info['instance_type'] = new_instance_type
            
            logger.info(f"Endpoint updated successfully: {endpoint_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating endpoint: {e}")
            return False

class DynamoDBManager:
    """
    AWS DynamoDB integration for battery data storage.
    """
    
    def __init__(self, aws_manager: AWSManager):
        self.aws_manager = aws_manager
        self.dynamodb = aws_manager.get_resource('dynamodb')
        self.dynamodb_client = aws_manager.get_client('dynamodb')
        
        # Table references
        self.tables = {}
        
        logger.info("DynamoDB Manager initialized")
    
    def create_battery_table(self, 
                           table_name: str = "BatteryData",
                           read_capacity: int = 5,
                           write_capacity: int = 5) -> str:
        """Create DynamoDB table for battery data."""
        try:
            table = self.dynamodb.create_table(
                TableName=table_name,
                KeySchema=[
                    {
                        'AttributeName': 'battery_id',
                        'KeyType': 'HASH'
                    },
                    {
                        'AttributeName': 'timestamp',
                        'KeyType': 'RANGE'
                    }
                ],
                AttributeDefinitions=[
                    {
                        'AttributeName': 'battery_id',
                        'AttributeType': 'S'
                    },
                    {
                        'AttributeName': 'timestamp',
                        'AttributeType': 'S'
                    }
                ],
                BillingMode='PROVISIONED',
                ProvisionedThroughput={
                    'ReadCapacityUnits': read_capacity,
                    'WriteCapacityUnits': write_capacity
                }
            )
            
            # Wait for table to be created
            table.wait_until_exists()
            
            self.tables[table_name] = table
            
            logger.info(f"DynamoDB table created: {table_name}")
            
            return table_name
            
        except Exception as e:
            logger.error(f"Error creating DynamoDB table: {e}")
            raise
    
    def store_battery_data(self, 
                         table_name: str,
                         battery_id: str,
                         telemetry_data: Dict[str, Any]) -> bool:
        """Store battery telemetry data in DynamoDB."""
        try:
            if table_name not in self.tables:
                self.tables[table_name] = self.dynamodb.Table(table_name)
            
            table = self.tables[table_name]
            
            # Prepare item
            item = {
                'battery_id': battery_id,
                'timestamp': datetime.now().isoformat(),
                **telemetry_data
            }
            
            # Store item
            table.put_item(Item=item)
            
            logger.debug(f"Stored battery data for {battery_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing battery data: {e}")
            return False
    
    def get_battery_data(self, 
                       table_name: str,
                       battery_id: str,
                       start_time: str = None,
                       end_time: str = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Retrieve battery data from DynamoDB."""
        try:
            if table_name not in self.tables:
                self.tables[table_name] = self.dynamodb.Table(table_name)
            
            table = self.tables[table_name]
            
            # Build query expression
            key_condition = Key('battery_id').eq(battery_id)
            
            if start_time and end_time:
                key_condition = key_condition & Key('timestamp').between(start_time, end_time)
            elif start_time:
                key_condition = key_condition & Key('timestamp').gte(start_time)
            elif end_time:
                key_condition = key_condition & Key('timestamp').lte(end_time)
            
            # Execute query
            response = table.query(
                KeyConditionExpression=key_condition,
                Limit=limit,
                ScanIndexForward=False  # Most recent first
            )
            
            items = response.get('Items', [])
            
            logger.debug(f"Retrieved {len(items)} records for battery {battery_id}")
            
            return items
            
        except Exception as e:
            logger.error(f"Error retrieving battery data: {e}")
            return []
    
    def batch_write_battery_data(self, 
                               table_name: str,
                               items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Batch write battery data to DynamoDB."""
        try:
            if table_name not in self.tables:
                self.tables[table_name] = self.dynamodb.Table(table_name)
            
            table = self.tables[table_name]
            
            # Process items in batches of 25 (DynamoDB limit)
            batch_size = 25
            successful_items = 0
            failed_items = 0
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                with table.batch_writer() as batch_writer:
                    for item in batch:
                        try:
                            batch_writer.put_item(Item=item)
                            successful_items += 1
                        except Exception as e:
                            logger.error(f"Error writing item to batch: {e}")
                            failed_items += 1
            
            result = {
                'total_items': len(items),
                'successful_items': successful_items,
                'failed_items': failed_items,
                'success_rate': successful_items / len(items) if items else 0
            }
            
            logger.info(f"Batch write completed: {successful_items}/{len(items)} items written")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in batch write: {e}")
            return {'total_items': len(items), 'successful_items': 0, 'failed_items': len(items)}

# Factory functions and utilities
def create_aws_manager(config: AWSConfig = None) -> AWSManager:
    """Create AWS manager instance."""
    return AWSManager(config)

def create_iot_manager(aws_manager: AWSManager = None) -> IoTCoreManager:
    """Create IoT Core manager instance."""
    aws_manager = aws_manager or create_aws_manager()
    return IoTCoreManager(aws_manager)

def create_sagemaker_manager(aws_manager: AWSManager = None) -> SageMakerManager:
    """Create SageMaker manager instance."""
    aws_manager = aws_manager or create_aws_manager()
    return SageMakerManager(aws_manager)

def create_dynamodb_manager(aws_manager: AWSManager = None) -> DynamoDBManager:
    """Create DynamoDB manager instance."""
    aws_manager = aws_manager or create_aws_manager()
    return DynamoDBManager(aws_manager)

def validate_aws_credentials() -> bool:
    """Validate AWS credentials."""
    try:
        session = boto3.Session()
        sts = session.client('sts')
        sts.get_caller_identity()
        return True
    except (NoCredentialsError, PartialCredentialsError):
        return False
    except Exception as e:
        logger.error(f"Error validating AWS credentials: {e}")
        return False

def get_aws_regions() -> List[str]:
    """Get list of available AWS regions."""
    try:
        session = boto3.Session()
        ec2 = session.client('ec2')
        response = ec2.describe_regions()
        return [region['RegionName'] for region in response['Regions']]
    except Exception as e:
        logger.error(f"Error getting AWS regions: {e}")
        return []

def estimate_aws_costs(service_usage: Dict[str, Any]) -> Dict[str, float]:
    """Estimate AWS service costs (simplified calculation)."""
    try:
        # Simplified cost estimation (actual costs may vary)
        cost_estimates = {
            'iot_core': {
                'messages_per_million': 1.0,
                'device_shadow_operations_per_million': 1.25
            },
            'sagemaker': {
                'ml_t2_medium_per_hour': 0.0464,
                'ml_m5_large_per_hour': 0.1152
            },
            'dynamodb': {
                'read_capacity_unit_per_month': 0.25,
                'write_capacity_unit_per_month': 1.25
            },
            's3': {
                'storage_per_gb_per_month': 0.023,
                'requests_per_1000': 0.0004
            }
        }
        
        total_cost = 0.0
        cost_breakdown = {}
        
        for service, usage in service_usage.items():
            if service in cost_estimates:
                service_cost = 0.0
                service_costs = cost_estimates[service]
                
                for metric, quantity in usage.items():
                    if metric in service_costs:
                        service_cost += quantity * service_costs[metric]
                
                cost_breakdown[service] = service_cost
                total_cost += service_cost
        
        return {
            'total_estimated_cost': total_cost,
            'service_breakdown': cost_breakdown,
            'currency': 'USD',
            'period': 'monthly'
        }
        
    except Exception as e:
        logger.error(f"Error estimating AWS costs: {e}")
        return {}

# Log module initialization
logger.info("BatteryMind AWS Helpers Module v1.0.0 loaded successfully")
