"""
BatteryMind - Real-Time Inference Pipeline

Production-ready real-time inference pipeline for battery health prediction,
degradation forecasting, and optimization recommendations. This module handles
streaming data processing, model inference, and real-time decision making.

Features:
- Real-time streaming data processing
- Multi-model inference orchestration
- Dynamic model selection and routing
- Performance monitoring and alerting
- Fault tolerance and error handling
- Scalable architecture for high throughput

Author: BatteryMind Development Team
Version: 1.0.0
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue, Empty
import pickle
import torch
import warnings
warnings.filterwarnings('ignore')

# AWS and streaming imports
import boto3
from kafka import KafkaConsumer, KafkaProducer
import redis
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# BatteryMind imports
from ..predictors.battery_health_predictor import BatteryHealthPredictor
from ..predictors.degradation_predictor import DegradationPredictor
from ..predictors.optimization_predictor import OptimizationPredictor
from ..predictors.ensemble_predictor import EnsemblePredictor
from ...training_data.preprocessing_scripts.normalization import BatteryDataNormalizer
from ...training_data.preprocessing_scripts.feature_extractor import BatteryFeatureExtractor
from ...utils.logging_utils import setup_logger
from ...utils.aws_helpers import AWSHelper
from ...monitoring.alerts.alert_manager import AlertManager

# Configure logging
logger = setup_logger(__name__)

# Metrics
INFERENCE_COUNTER = Counter('inference_requests_total', 'Total inference requests', ['model_type', 'status'])
INFERENCE_DURATION = Histogram('inference_duration_seconds', 'Inference duration', ['model_type'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Number of active connections')
QUEUE_SIZE = Gauge('queue_size', 'Current queue size')
ERROR_RATE = Counter('inference_errors_total', 'Total inference errors', ['error_type'])

@dataclass
class RealTimeInferenceConfig:
    """Configuration for real-time inference pipeline."""
    
    # Streaming configuration
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_input_topic: str = "battery_telemetry"
    kafka_output_topic: str = "battery_predictions"
    kafka_consumer_group: str = "batterymind_inference"
    
    # Redis configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    redis_db: int = 0
    
    # AWS configuration
    aws_region: str = "us-west-2"
    s3_bucket: str = "batterymind-models"
    kinesis_stream: str = "battery-data-stream"
    
    # Model configuration
    model_cache_size: int = 10
    model_refresh_interval: int = 300  # seconds
    ensemble_enabled: bool = True
    
    # Performance configuration
    max_workers: int = 4
    queue_size: int = 1000
    batch_size: int = 32
    timeout_seconds: int = 30
    
    # Monitoring configuration
    metrics_port: int = 8080
    alert_enabled: bool = True
    performance_threshold: float = 0.1  # seconds
    
    # Feature engineering
    sequence_length: int = 144  # 24 hours at 10-minute intervals
    feature_extraction_enabled: bool = True
    
    # Safety and validation
    safety_checks_enabled: bool = True
    anomaly_detection_enabled: bool = True
    confidence_threshold: float = 0.8

class ModelRegistry:
    """Registry for managing multiple ML models."""
    
    def __init__(self, config: RealTimeInferenceConfig):
        self.config = config
        self.models = {}
        self.model_metadata = {}
        self.aws_helper = AWSHelper()
        self.last_refresh = {}
        self._lock = threading.Lock()
        
        # Initialize models
        self._load_models()
        
        # Start model refresh thread
        self._start_model_refresh_thread()
    
    def _load_models(self):
        """Load all available models."""
        try:
            # Load individual predictors
            self.models['health'] = BatteryHealthPredictor()
            self.models['degradation'] = DegradationPredictor()
            self.models['optimization'] = OptimizationPredictor()
            
            # Load ensemble model if enabled
            if self.config.ensemble_enabled:
                self.models['ensemble'] = EnsemblePredictor()
            
            # Load model metadata
            self._load_model_metadata()
            
            logger.info(f"Loaded {len(self.models)} models successfully")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def _load_model_metadata(self):
        """Load model metadata from S3."""
        try:
            for model_name in self.models.keys():
                metadata_key = f"model-artifacts/trained_models/{model_name}_v1.0/model_metadata.yaml"
                metadata = self.aws_helper.download_from_s3(self.config.s3_bucket, metadata_key)
                if metadata:
                    self.model_metadata[model_name] = metadata
                    
        except Exception as e:
            logger.warning(f"Failed to load model metadata: {e}")
    
    def _start_model_refresh_thread(self):
        """Start background thread for model refresh."""
        def refresh_worker():
            while True:
                try:
                    time.sleep(self.config.model_refresh_interval)
                    self._refresh_models()
                except Exception as e:
                    logger.error(f"Model refresh error: {e}")
        
        refresh_thread = threading.Thread(target=refresh_worker, daemon=True)
        refresh_thread.start()
    
    def _refresh_models(self):
        """Refresh models from remote storage."""
        with self._lock:
            try:
                # Check for model updates
                for model_name in self.models.keys():
                    if self._should_refresh_model(model_name):
                        self._reload_model(model_name)
                        
            except Exception as e:
                logger.error(f"Model refresh failed: {e}")
    
    def _should_refresh_model(self, model_name: str) -> bool:
        """Check if model should be refreshed."""
        last_refresh = self.last_refresh.get(model_name, 0)
        return (time.time() - last_refresh) > self.config.model_refresh_interval
    
    def _reload_model(self, model_name: str):
        """Reload specific model."""
        try:
            # Download latest model from S3
            model_key = f"model-artifacts/trained_models/{model_name}_v1.0/model.pkl"
            model_data = self.aws_helper.download_from_s3(self.config.s3_bucket, model_key)
            
            if model_data:
                # Reload model
                if model_name == 'health':
                    self.models[model_name] = BatteryHealthPredictor()
                elif model_name == 'degradation':
                    self.models[model_name] = DegradationPredictor()
                elif model_name == 'optimization':
                    self.models[model_name] = OptimizationPredictor()
                elif model_name == 'ensemble':
                    self.models[model_name] = EnsemblePredictor()
                
                self.last_refresh[model_name] = time.time()
                logger.info(f"Refreshed model: {model_name}")
                
        except Exception as e:
            logger.error(f"Failed to reload model {model_name}: {e}")
    
    def get_model(self, model_name: str):
        """Get model instance."""
        with self._lock:
            return self.models.get(model_name)
    
    def get_available_models(self) -> List[str]:
        """Get list of available models."""
        return list(self.models.keys())

class DataProcessor:
    """Handles data preprocessing and feature extraction."""
    
    def __init__(self, config: RealTimeInferenceConfig):
        self.config = config
        self.normalizer = BatteryDataNormalizer()
        self.feature_extractor = BatteryFeatureExtractor()
        self.data_cache = {}
        self._lock = threading.Lock()
        
        # Load preprocessing artifacts
        self._load_preprocessing_artifacts()
    
    def _load_preprocessing_artifacts(self):
        """Load preprocessing artifacts from storage."""
        try:
            aws_helper = AWSHelper()
            
            # Load normalizer
            normalizer_key = "model-artifacts/preprocessing/normalizer.pkl"
            normalizer_data = aws_helper.download_from_s3(self.config.s3_bucket, normalizer_key)
            if normalizer_data:
                self.normalizer = pickle.loads(normalizer_data)
            
            # Load feature extractor
            extractor_key = "model-artifacts/preprocessing/feature_extractor.pkl"
            extractor_data = aws_helper.download_from_s3(self.config.s3_bucket, extractor_key)
            if extractor_data:
                self.feature_extractor = pickle.loads(extractor_data)
                
            logger.info("Loaded preprocessing artifacts successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load preprocessing artifacts: {e}")
    
    def process_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process raw sensor data for inference."""
        try:
            battery_id = raw_data.get('battery_id')
            timestamp = raw_data.get('timestamp')
            
            # Validate input data
            if not self._validate_input_data(raw_data):
                raise ValueError("Invalid input data format")
            
            # Extract sensor readings
            sensor_data = self._extract_sensor_data(raw_data)
            
            # Update data cache
            self._update_data_cache(battery_id, sensor_data, timestamp)
            
            # Create sequence if enough data
            if self._has_sufficient_data(battery_id):
                sequence = self._create_sequence(battery_id)
                
                # Extract features
                if self.config.feature_extraction_enabled:
                    features = self._extract_features(sequence)
                else:
                    features = sequence
                
                # Normalize features
                normalized_features = self._normalize_features(features)
                
                return {
                    'battery_id': battery_id,
                    'timestamp': timestamp,
                    'features': normalized_features,
                    'raw_data': sensor_data,
                    'sequence_length': len(sequence)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Data processing error: {e}")
            ERROR_RATE.labels(error_type='data_processing').inc()
            raise
    
    def _validate_input_data(self, data: Dict[str, Any]) -> bool:
        """Validate input data format."""
        required_fields = ['battery_id', 'timestamp', 'voltage', 'current', 'temperature']
        return all(field in data for field in required_fields)
    
    def _extract_sensor_data(self, raw_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract sensor readings from raw data."""
        sensor_fields = [
            'voltage', 'current', 'temperature', 'soc', 'power',
            'energy', 'internal_resistance', 'capacity'
        ]
        
        return {field: raw_data.get(field, 0.0) for field in sensor_fields}
    
    def _update_data_cache(self, battery_id: str, sensor_data: Dict[str, float], timestamp: str):
        """Update data cache with new sensor reading."""
        with self._lock:
            if battery_id not in self.data_cache:
                self.data_cache[battery_id] = []
            
            # Add new reading
            reading = {**sensor_data, 'timestamp': timestamp}
            self.data_cache[battery_id].append(reading)
            
            # Maintain cache size
            max_cache_size = self.config.sequence_length * 2
            if len(self.data_cache[battery_id]) > max_cache_size:
                self.data_cache[battery_id] = self.data_cache[battery_id][-max_cache_size:]
    
    def _has_sufficient_data(self, battery_id: str) -> bool:
        """Check if sufficient data for sequence creation."""
        with self._lock:
            return (battery_id in self.data_cache and 
                   len(self.data_cache[battery_id]) >= self.config.sequence_length)
    
    def _create_sequence(self, battery_id: str) -> np.ndarray:
        """Create sequence from cached data."""
        with self._lock:
            recent_data = self.data_cache[battery_id][-self.config.sequence_length:]
            
            # Convert to numpy array
            sequence = []
            for reading in recent_data:
                values = [reading[field] for field in ['voltage', 'current', 'temperature', 
                                                      'soc', 'power', 'energy', 'internal_resistance', 'capacity']]
                sequence.append(values)
            
            return np.array(sequence)
    
    def _extract_features(self, sequence: np.ndarray) -> np.ndarray:
        """Extract features from sequence."""
        return self.feature_extractor.extract_features(sequence)
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features."""
        return self.normalizer.normalize(features)

class SafetyValidator:
    """Validates inference results for safety compliance."""
    
    def __init__(self, config: RealTimeInferenceConfig):
        self.config = config
        self.safety_rules = self._load_safety_rules()
        self.anomaly_detector = self._load_anomaly_detector()
    
    def _load_safety_rules(self) -> Dict[str, Any]:
        """Load safety rules and constraints."""
        return {
            'voltage_limits': {'min': 2.5, 'max': 4.5},
            'temperature_limits': {'min': -40, 'max': 85},
            'current_limits': {'min': -500, 'max': 500},
            'soc_limits': {'min': 0.0, 'max': 1.0},
            'soh_limits': {'min': 0.0, 'max': 1.0},
            'confidence_threshold': 0.8
        }
    
    def _load_anomaly_detector(self):
        """Load anomaly detection model."""
        # Placeholder for anomaly detection model
        return None
    
    def validate_prediction(self, prediction: Dict[str, Any], 
                          input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate prediction results for safety compliance."""
        validation_result = {
            'is_valid': True,
            'safety_violations': [],
            'warnings': [],
            'confidence_score': 1.0
        }
        
        try:
            # Check prediction confidence
            if not self._check_prediction_confidence(prediction):
                validation_result['is_valid'] = False
                validation_result['warnings'].append('Low prediction confidence')
            
            # Check safety constraints
            violations = self._check_safety_constraints(prediction, input_data)
            if violations:
                validation_result['safety_violations'].extend(violations)
                validation_result['is_valid'] = False
            
            # Check for anomalies
            if self.config.anomaly_detection_enabled:
                anomalies = self._detect_anomalies(prediction, input_data)
                if anomalies:
                    validation_result['warnings'].extend(anomalies)
            
            # Calculate overall confidence
            validation_result['confidence_score'] = self._calculate_confidence_score(
                prediction, validation_result
            )
            
        except Exception as e:
            logger.error(f"Safety validation error: {e}")
            validation_result['is_valid'] = False
            validation_result['warnings'].append(f"Validation error: {e}")
        
        return validation_result
    
    def _check_prediction_confidence(self, prediction: Dict[str, Any]) -> bool:
        """Check if prediction confidence meets threshold."""
        confidence = prediction.get('confidence', 0.0)
        return confidence >= self.config.confidence_threshold
    
    def _check_safety_constraints(self, prediction: Dict[str, Any], 
                                input_data: Dict[str, Any]) -> List[str]:
        """Check safety constraints."""
        violations = []
        
        # Check voltage limits
        voltage = input_data.get('voltage', 0)
        if not (self.safety_rules['voltage_limits']['min'] <= voltage <= 
                self.safety_rules['voltage_limits']['max']):
            violations.append(f"Voltage out of range: {voltage}V")
        
        # Check temperature limits
        temperature = input_data.get('temperature', 0)
        if not (self.safety_rules['temperature_limits']['min'] <= temperature <= 
                self.safety_rules['temperature_limits']['max']):
            violations.append(f"Temperature out of range: {temperature}°C")
        
        # Check SoH prediction
        soh = prediction.get('soh', 1.0)
        if soh < 0.2:  # Critical SoH threshold
            violations.append(f"Critical SoH detected: {soh}")
        
        return violations
    
    def _detect_anomalies(self, prediction: Dict[str, Any], 
                         input_data: Dict[str, Any]) -> List[str]:
        """Detect anomalies in prediction or input data."""
        anomalies = []
        
        # Simple anomaly detection rules
        if input_data.get('voltage', 0) < 2.0:
            anomalies.append("Abnormally low voltage detected")
        
        if input_data.get('temperature', 25) > 60:
            anomalies.append("High temperature detected")
        
        return anomalies
    
    def _calculate_confidence_score(self, prediction: Dict[str, Any], 
                                   validation_result: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        base_confidence = prediction.get('confidence', 0.0)
        
        # Reduce confidence for safety violations
        if validation_result['safety_violations']:
            base_confidence *= 0.5
        
        # Reduce confidence for warnings
        if validation_result['warnings']:
            base_confidence *= 0.8
        
        return max(0.0, min(1.0, base_confidence))

class RealTimeInferencePipeline:
    """Main real-time inference pipeline."""
    
    def __init__(self, config: RealTimeInferenceConfig):
        self.config = config
        self.model_registry = ModelRegistry(config)
        self.data_processor = DataProcessor(config)
        self.safety_validator = SafetyValidator(config)
        self.alert_manager = AlertManager()
        
        # Initialize components
        self.kafka_consumer = None
        self.kafka_producer = None
        self.redis_client = None
        self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
        
        # Processing queues
        self.input_queue = Queue(maxsize=config.queue_size)
        self.output_queue = Queue(maxsize=config.queue_size)
        
        # Statistics
        self.processed_count = 0
        self.error_count = 0
        self.start_time = time.time()
        
        # Initialize connections
        self._initialize_connections()
        
        # Start processing threads
        self._start_processing_threads()
        
        # Start metrics server
        if config.metrics_port:
            start_http_server(config.metrics_port)
        
        logger.info("Real-time inference pipeline initialized")
    
    def _initialize_connections(self):
        """Initialize external connections."""
        try:
            # Initialize Kafka consumer
            self.kafka_consumer = KafkaConsumer(
                self.config.kafka_input_topic,
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                group_id=self.config.kafka_consumer_group,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                enable_auto_commit=True,
                auto_offset_reset='latest'
            )
            
            # Initialize Kafka producer
            self.kafka_producer = KafkaProducer(
                bootstrap_servers=self.config.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
            )
            
            # Initialize Redis client
            self.redis_client = redis.Redis(
                host=self.config.redis_host,
                port=self.config.redis_port,
                password=self.config.redis_password,
                db=self.config.redis_db,
                decode_responses=True
            )
            
            logger.info("External connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connections: {e}")
            raise
    
    def _start_processing_threads(self):
        """Start background processing threads."""
        # Start data ingestion thread
        ingestion_thread = threading.Thread(target=self._data_ingestion_worker, daemon=True)
        ingestion_thread.start()
        
        # Start inference workers
        for i in range(self.config.max_workers):
            worker_thread = threading.Thread(target=self._inference_worker, daemon=True)
            worker_thread.start()
        
        # Start result publishing thread
        publish_thread = threading.Thread(target=self._result_publishing_worker, daemon=True)
        publish_thread.start()
        
        logger.info(f"Started {self.config.max_workers + 2} processing threads")
    
    def _data_ingestion_worker(self):
        """Worker thread for data ingestion from Kafka."""
        while True:
            try:
                message_batch = self.kafka_consumer.poll(timeout_ms=1000)
                
                for topic_partition, messages in message_batch.items():
                    for message in messages:
                        try:
                            # Add to processing queue
                            self.input_queue.put(message.value, timeout=5)
                            QUEUE_SIZE.set(self.input_queue.qsize())
                            
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                            ERROR_RATE.labels(error_type='ingestion').inc()
                
            except Exception as e:
                logger.error(f"Data ingestion error: {e}")
                ERROR_RATE.labels(error_type='ingestion').inc()
                time.sleep(1)
    
    def _inference_worker(self):
        """Worker thread for model inference."""
        while True:
            try:
                # Get data from queue
                raw_data = self.input_queue.get(timeout=5)
                
                # Process inference
                result = self._process_inference(raw_data)
                
                if result:
                    # Add to output queue
                    self.output_queue.put(result, timeout=5)
                
                # Mark task as done
                self.input_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Inference worker error: {e}")
                ERROR_RATE.labels(error_type='inference').inc()
    
    def _process_inference(self, raw_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single inference request."""
        start_time = time.time()
        
        try:
            # Process input data
            processed_data = self.data_processor.process_data(raw_data)
            
            if not processed_data:
                return None
            
            # Select model based on request or use ensemble
            model_name = raw_data.get('model_type', 'ensemble')
            model = self.model_registry.get_model(model_name)
            
            if not model:
                logger.warning(f"Model not found: {model_name}")
                return None
            
            # Perform inference
            prediction = model.predict(processed_data['features'])
            
            # Add metadata
            prediction_result = {
                'battery_id': processed_data['battery_id'],
                'timestamp': processed_data['timestamp'],
                'model_type': model_name,
                'prediction': prediction,
                'inference_time': time.time() - start_time,
                'sequence_length': processed_data['sequence_length']
            }
            
            # Safety validation
            if self.config.safety_checks_enabled:
                validation_result = self.safety_validator.validate_prediction(
                    prediction, processed_data['raw_data']
                )
                prediction_result['validation'] = validation_result
                
                # Send alerts for safety violations
                if validation_result['safety_violations']:
                    self._send_safety_alert(prediction_result)
            
            # Update metrics
            INFERENCE_COUNTER.labels(model_type=model_name, status='success').inc()
            INFERENCE_DURATION.labels(model_type=model_name).observe(time.time() - start_time)
            
            self.processed_count += 1
            
            return prediction_result
            
        except Exception as e:
            logger.error(f"Inference processing error: {e}")
            ERROR_RATE.labels(error_type='inference').inc()
            self.error_count += 1
            return None
    
    def _result_publishing_worker(self):
        """Worker thread for publishing results."""
        while True:
            try:
                # Get result from queue
                result = self.output_queue.get(timeout=5)
                
                # Publish to Kafka
                self.kafka_producer.send(
                    self.config.kafka_output_topic,
                    value=result
                )
                
                # Cache result in Redis
                self._cache_result(result)
                
                # Mark task as done
                self.output_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                logger.error(f"Result publishing error: {e}")
                ERROR_RATE.labels(error_type='publishing').inc()
    
    def _cache_result(self, result: Dict[str, Any]):
        """Cache inference result in Redis."""
        try:
            battery_id = result['battery_id']
            timestamp = result['timestamp']
            
            # Create cache key
            cache_key = f"inference:{battery_id}:{timestamp}"
            
            # Cache for 1 hour
            self.redis_client.setex(
                cache_key,
                3600,
                json.dumps(result)
            )
            
        except Exception as e:
            logger.error(f"Result caching error: {e}")
    
    def _send_safety_alert(self, prediction_result: Dict[str, Any]):
        """Send safety alert for violations."""
        try:
            alert_data = {
                'type': 'safety_violation',
                'battery_id': prediction_result['battery_id'],
                'timestamp': prediction_result['timestamp'],
                'violations': prediction_result['validation']['safety_violations'],
                'severity': 'high'
            }
            
            self.alert_manager.send_alert(alert_data)
            
        except Exception as e:
            logger.error(f"Safety alert error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        uptime = time.time() - self.start_time
        
        return {
            'processed_count': self.processed_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.processed_count, 1),
            'uptime_seconds': uptime,
            'throughput_per_second': self.processed_count / max(uptime, 1),
            'queue_sizes': {
                'input': self.input_queue.qsize(),
                'output': self.output_queue.qsize()
            },
            'available_models': self.model_registry.get_available_models()
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        health_status = {
            'status': 'healthy',
            'components': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Check Kafka connection
        try:
            self.kafka_consumer.poll(timeout_ms=100)
            health_status['components']['kafka'] = 'healthy'
        except Exception as e:
            health_status['components']['kafka'] = f'unhealthy: {e}'
            health_status['status'] = 'unhealthy'
        
        # Check Redis connection
        try:
            self.redis_client.ping()
            health_status['components']['redis'] = 'healthy'
        except Exception as e:
            health_status['components']['redis'] = f'unhealthy: {e}'
            health_status['status'] = 'unhealthy'
        
        # Check models
        try:
            models = self.model_registry.get_available_models()
            health_status['components']['models'] = f'healthy: {len(models)} models'
        except Exception as e:
            health_status['components']['models'] = f'unhealthy: {e}'
            health_status['status'] = 'unhealthy'
        
        return health_status
    
    def shutdown(self):
        """Graceful shutdown of the pipeline."""
        logger.info("Shutting down real-time inference pipeline...")
        
        # Close Kafka connections
        if self.kafka_consumer:
            self.kafka_consumer.close()
        
        if self.kafka_producer:
            self.kafka_producer.close()
        
        # Close Redis connection
        if self.redis_client:
            self.redis_client.close()
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Real-time inference pipeline shut down successfully")

# Factory function
def create_real_time_pipeline(config: Optional[RealTimeInferenceConfig] = None) -> RealTimeInferencePipeline:
    """
    Factory function to create real-time inference pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        RealTimeInferencePipeline: Configured pipeline instance
    """
    if config is None:
        config = RealTimeInferenceConfig()
    
    return RealTimeInferencePipeline(config)

# Example usage
if __name__ == "__main__":
    # Create and start pipeline
    config = RealTimeInferenceConfig()
    pipeline = create_real_time_pipeline(config)
    
    try:
        # Keep pipeline running
        while True:
            time.sleep(10)
            stats = pipeline.get_statistics()
            logger.info(f"Pipeline stats: {stats}")
    
    except KeyboardInterrupt:
        logger.info("Shutting down pipeline...")
        pipeline.shutdown()
