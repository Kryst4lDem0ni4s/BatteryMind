"""
BatteryMind - Model Utilities
Comprehensive model management, deployment, and optimization utilities
for AI/ML models in the BatteryMind autonomous battery management system.

Features:
- Model lifecycle management and versioning
- Performance monitoring and optimization
- Deployment and inference management
- Model compression and quantization
- A/B testing and comparison frameworks
- Automated hyperparameter tuning
- Model interpretability and explainability

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import pickle
import joblib
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import uuid
import threading
from concurrent.futures import ThreadPoolExecutor
import time
import warnings

# ML/AI Framework imports
try:
    import torch
    import torch.nn as nn
    import tensorflow as tf
    from transformers import AutoModel, AutoTokenizer
    PYTORCH_AVAILABLE = True
    TENSORFLOW_AVAILABLE = True
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    TENSORFLOW_AVAILABLE = False
    TRANSFORMERS_AVAILABLE = False

# Model optimization imports
try:
    import optuna
    from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
    from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False

# Configure logging
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Types of AI/ML models in BatteryMind."""
    TRANSFORMER = "transformer"
    LSTM = "lstm"
    CNN = "cnn"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    ENSEMBLE = "ensemble"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    FEDERATED = "federated"

class ModelStatus(Enum):
    """Model deployment status."""
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"

class OptimizationTarget(Enum):
    """Model optimization targets."""
    ACCURACY = "accuracy"
    LATENCY = "latency"
    MEMORY = "memory"
    THROUGHPUT = "throughput"
    F1_SCORE = "f1_score"
    AUC_ROC = "auc_roc"
    RMSE = "rmse"
    MAE = "mae"

@dataclass
class ModelMetadata:
    """Metadata for AI/ML models."""
    
    model_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    version: str = "1.0.0"
    model_type: ModelType = ModelType.NEURAL_NETWORK
    status: ModelStatus = ModelStatus.TRAINING
    
    # Training information
    training_data_hash: str = ""
    training_start_time: Optional[datetime] = None
    training_end_time: Optional[datetime] = None
    training_duration_seconds: float = 0.0
    
    # Performance metrics
    accuracy: float = 0.0
    loss: float = float('inf')
    validation_score: float = 0.0
    test_score: float = 0.0
    
    # Model characteristics
    parameter_count: int = 0
    model_size_mb: float = 0.0
    input_shape: Tuple = ()
    output_shape: Tuple = ()
    
    # Deployment information
    deployment_target: str = ""
    inference_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    throughput_qps: float = 0.0
    
    # Versioning and tracking
    parent_model_id: Optional[str] = None
    experiment_id: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat() if value else None
            elif isinstance(value, Enum):
                data[key] = value.value
        return data

@dataclass
class ModelPerformanceReport:
    """Comprehensive model performance report."""
    
    model_id: str
    evaluation_timestamp: datetime = field(default_factory=datetime.now)
    
    # Accuracy metrics
    accuracy_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    latency_metrics: Dict[str, float] = field(default_factory=dict)
    memory_metrics: Dict[str, float] = field(default_factory=dict)
    throughput_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Business metrics
    business_impact: Dict[str, float] = field(default_factory=dict)
    
    # Recommendations
    optimization_recommendations: List[str] = field(default_factory=list)
    deployment_readiness: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        if isinstance(data['evaluation_timestamp'], datetime):
            data['evaluation_timestamp'] = data['evaluation_timestamp'].isoformat()
        return data

class ModelManager:
    """
    Comprehensive model management system for BatteryMind AI models.
    """
    
    def __init__(self, 
                 model_registry_path: str = "./model_registry",
                 enable_versioning: bool = True,
                 enable_monitoring: bool = True):
        
        self.model_registry_path = Path(model_registry_path)
        self.enable_versioning = enable_versioning
        self.enable_monitoring = enable_monitoring
        
        # Create registry directory
        self.model_registry_path.mkdir(parents=True, exist_ok=True)
        
        # Model registry
        self.registered_models: Dict[str, ModelMetadata] = {}
        self.model_artifacts: Dict[str, Any] = {}
        
        # Performance tracking
        self.performance_history: Dict[str, List[ModelPerformanceReport]] = {}
        
        # Thread safety
        self.registry_lock = threading.RLock()
        
        # Load existing registry
        self._load_registry()
        
        logger.info(f"Model Manager initialized with registry at: {model_registry_path}")
    
    def register_model(self, 
                      model: Any, 
                      metadata: ModelMetadata,
                      save_artifacts: bool = True) -> str:
        """
        Register a new model in the registry.
        
        Args:
            model: The trained model object
            metadata: Model metadata
            save_artifacts: Whether to save model artifacts to disk
            
        Returns:
            Model ID
        """
        try:
            with self.registry_lock:
                # Generate model ID if not provided
                if not metadata.model_id:
                    metadata.model_id = str(uuid.uuid4())
                
                # Calculate model characteristics
                metadata = self._calculate_model_characteristics(model, metadata)
                
                # Save model artifacts if requested
                if save_artifacts:
                    artifact_path = self._save_model_artifacts(model, metadata)
                    logger.info(f"Model artifacts saved to: {artifact_path}")
                
                # Store in registry
                self.registered_models[metadata.model_id] = metadata
                self.model_artifacts[metadata.model_id] = model
                
                # Initialize performance history
                self.performance_history[metadata.model_id] = []
                
                # Save registry
                self._save_registry()
                
                logger.info(f"Model registered: {metadata.name} (ID: {metadata.model_id})")
                
                return metadata.model_id
                
        except Exception as e:
            logger.error(f"Error registering model: {e}")
            raise
    
    def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """
        Load a model from the registry.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model, metadata)
        """
        try:
            with self.registry_lock:
                if model_id not in self.registered_models:
                    raise ValueError(f"Model {model_id} not found in registry")
                
                metadata = self.registered_models[model_id]
                
                # Try to get from memory cache first
                if model_id in self.model_artifacts:
                    model = self.model_artifacts[model_id]
                    logger.info(f"Model loaded from cache: {model_id}")
                    return model, metadata
                
                # Load from disk
                model = self._load_model_artifacts(model_id, metadata)
                
                # Cache in memory
                self.model_artifacts[model_id] = model
                
                logger.info(f"Model loaded from disk: {model_id}")
                
                return model, metadata
                
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            raise
    
    def evaluate_model(self, 
                      model_id: str,
                      test_data: Union[np.ndarray, pd.DataFrame],
                      test_labels: Union[np.ndarray, pd.Series] = None,
                      metrics: List[str] = None) -> ModelPerformanceReport:
        """
        Evaluate model performance and generate comprehensive report.
        
        Args:
            model_id: Model identifier
            test_data: Test dataset
            test_labels: Test labels (for supervised learning)
            metrics: List of metrics to calculate
            
        Returns:
            Performance report
        """
        try:
            model, metadata = self.load_model(model_id)
            
            if metrics is None:
                metrics = ['accuracy', 'latency', 'memory', 'throughput']
            
            report = ModelPerformanceReport(model_id=model_id)
            
            # Accuracy evaluation
            if 'accuracy' in metrics and test_labels is not None:
                accuracy_metrics = self._evaluate_accuracy(model, test_data, test_labels, metadata)
                report.accuracy_metrics = accuracy_metrics
            
            # Performance evaluation
            if 'latency' in metrics:
                latency_metrics = self._evaluate_latency(model, test_data)
                report.latency_metrics = latency_metrics
            
            if 'memory' in metrics:
                memory_metrics = self._evaluate_memory_usage(model, test_data)
                report.memory_metrics = memory_metrics
            
            if 'throughput' in metrics:
                throughput_metrics = self._evaluate_throughput(model, test_data)
                report.throughput_metrics = throughput_metrics
            
            # Generate recommendations
            report.optimization_recommendations = self._generate_optimization_recommendations(report)
            report.deployment_readiness = self._assess_deployment_readiness(report)
            
            # Store performance history
            with self.registry_lock:
                self.performance_history[model_id].append(report)
            
            logger.info(f"Model evaluation completed for: {model_id}")
            
            return report
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_id}: {e}")
            raise
    
    def _calculate_model_characteristics(self, model: Any, metadata: ModelMetadata) -> ModelMetadata:
        """Calculate model characteristics like parameter count and size."""
        try:
            # PyTorch model
            if PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                metadata.parameter_count = sum(p.numel() for p in model.parameters())
                
                # Calculate model size
                temp_path = "/tmp/temp_model.pt"
                torch.save(model.state_dict(), temp_path)
                metadata.model_size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
                Path(temp_path).unlink()
            
            # TensorFlow model
            elif TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
                metadata.parameter_count = model.count_params()
                
                # Calculate model size
                temp_path = "/tmp/temp_model.h5"
                model.save(temp_path)
                metadata.model_size_mb = Path(temp_path).stat().st_size / (1024 * 1024)
                Path(temp_path).unlink()
            
            # Scikit-learn or other models
            else:
                # Estimate size using pickle
                import sys
                metadata.model_size_mb = sys.getsizeof(pickle.dumps(model)) / (1024 * 1024)
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Could not calculate model characteristics: {e}")
            return metadata
    
    def _save_model_artifacts(self, model: Any, metadata: ModelMetadata) -> Path:
        """Save model artifacts to disk."""
        try:
            # Create model directory
            model_dir = self.model_registry_path / metadata.model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model based on type
            if PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                model_path = model_dir / "model.pt"
                torch.save(model.state_dict(), model_path)
                
                # Save full model
                full_model_path = model_dir / "full_model.pt"
                torch.save(model, full_model_path)
            
            elif TENSORFLOW_AVAILABLE and isinstance(model, tf.keras.Model):
                model_path = model_dir / "model.h5"
                model.save(model_path)
                
                # Save weights separately
                weights_path = model_dir / "weights.h5"
                model.save_weights(weights_path)
            
            else:
                # Use joblib for scikit-learn models
                model_path = model_dir / "model.pkl"
                joblib.dump(model, model_path)
            
            # Save metadata
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            return model_dir
            
        except Exception as e:
            logger.error(f"Error saving model artifacts: {e}")
            raise
    
    def _load_model_artifacts(self, model_id: str, metadata: ModelMetadata) -> Any:
        """Load model artifacts from disk."""
        try:
            model_dir = self.model_registry_path / model_id
            
            if not model_dir.exists():
                raise FileNotFoundError(f"Model directory not found: {model_dir}")
            
            # Load based on model type
            if metadata.model_type == ModelType.TRANSFORMER:
                if PYTORCH_AVAILABLE:
                    model_path = model_dir / "full_model.pt"
                    if model_path.exists():
                        return torch.load(model_path)
                    else:
                        # Load state dict (requires model architecture)
                        state_dict_path = model_dir / "model.pt"
                        return torch.load(state_dict_path)
            
            elif metadata.model_type in [ModelType.LSTM, ModelType.CNN, ModelType.NEURAL_NETWORK]:
                # Try TensorFlow first
                h5_path = model_dir / "model.h5"
                if TENSORFLOW_AVAILABLE and h5_path.exists():
                    return tf.keras.models.load_model(h5_path)
                
                # Try PyTorch
                pt_path = model_dir / "full_model.pt"
                if PYTORCH_AVAILABLE and pt_path.exists():
                    return torch.load(pt_path)
            
            # Default to pickle/joblib
            pkl_path = model_dir / "model.pkl"
            if pkl_path.exists():
                return joblib.load(pkl_path)
            
            raise FileNotFoundError(f"No compatible model file found in {model_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model artifacts: {e}")
            raise
    
    def _evaluate_accuracy(self, model: Any, test_data: Any, test_labels: Any, 
                          metadata: ModelMetadata) -> Dict[str, float]:
        """Evaluate model accuracy metrics."""
        try:
            accuracy_metrics = {}
            
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(test_data)
            elif hasattr(model, 'forward'):
                # PyTorch model
                model.eval()
                with torch.no_grad():
                    if isinstance(test_data, np.ndarray):
                        test_data = torch.FloatTensor(test_data)
                    predictions = model(test_data).numpy()
            else:
                logger.warning("Model doesn't have predict or forward method")
                return accuracy_metrics
            
            # Handle different prediction formats
            if len(predictions.shape) > 1 and predictions.shape[1] > 1:
                # Multi-class classification
                predictions = np.argmax(predictions, axis=1)
            
            # Calculate metrics based on problem type
            if metadata.model_type in [ModelType.TRANSFORMER, ModelType.LSTM, ModelType.NEURAL_NETWORK]:
                # Check if regression or classification
                if len(np.unique(test_labels)) > 10:  # Assume regression
                    accuracy_metrics['rmse'] = np.sqrt(mean_squared_error(test_labels, predictions))
                    accuracy_metrics['mae'] = np.mean(np.abs(test_labels - predictions))
                    accuracy_metrics['r2_score'] = r2_score(test_labels, predictions)
                else:  # Classification
                    accuracy_metrics['accuracy'] = accuracy_score(test_labels, predictions)
                    
                    # Calculate precision, recall, f1 if available
                    try:
                        from sklearn.metrics import precision_score, recall_score, f1_score
                        accuracy_metrics['precision'] = precision_score(test_labels, predictions, average='weighted')
                        accuracy_metrics['recall'] = recall_score(test_labels, predictions, average='weighted')
                        accuracy_metrics['f1_score'] = f1_score(test_labels, predictions, average='weighted')
                    except ImportError:
                        pass
            
            return accuracy_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating accuracy: {e}")
            return {}
    
    def _evaluate_latency(self, model: Any, test_data: Any) -> Dict[str, float]:
        """Evaluate model inference latency."""
        try:
            latency_metrics = {}
            
            # Single prediction latency
            sample_data = test_data[:1] if hasattr(test_data, '__getitem__') else test_data
            
            # Warm up
            for _ in range(5):
                if hasattr(model, 'predict'):
                    _ = model.predict(sample_data)
                elif hasattr(model, 'forward'):
                    model.eval()
                    with torch.no_grad():
                        if isinstance(sample_data, np.ndarray):
                            sample_data = torch.FloatTensor(sample_data)
                        _ = model(sample_data)
            
            # Measure latency
            latencies = []
            for _ in range(100):
                start_time = time.time()
                
                if hasattr(model, 'predict'):
                    _ = model.predict(sample_data)
                elif hasattr(model, 'forward'):
                    model.eval()
                    with torch.no_grad():
                        _ = model(sample_data)
                
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
            
            latency_metrics['mean_latency_ms'] = np.mean(latencies)
            latency_metrics['median_latency_ms'] = np.median(latencies)
            latency_metrics['p95_latency_ms'] = np.percentile(latencies, 95)
            latency_metrics['p99_latency_ms'] = np.percentile(latencies, 99)
            latency_metrics['min_latency_ms'] = np.min(latencies)
            latency_metrics['max_latency_ms'] = np.max(latencies)
            
            return latency_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating latency: {e}")
            return {}
    
    def _evaluate_memory_usage(self, model: Any, test_data: Any) -> Dict[str, float]:
        """Evaluate model memory usage."""
        try:
            import psutil
            import os
            
            memory_metrics = {}
            process = psutil.Process(os.getpid())
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            # Memory during inference
            if hasattr(model, 'predict'):
                _ = model.predict(test_data[:100] if hasattr(test_data, '__getitem__') else test_data)
            elif hasattr(model, 'forward'):
                model.eval()
                with torch.no_grad():
                    sample_data = test_data[:100] if hasattr(test_data, '__getitem__') else test_data
                    if isinstance(sample_data, np.ndarray):
                        sample_data = torch.FloatTensor(sample_data)
                    _ = model(sample_data)
            
            inference_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            memory_metrics['baseline_memory_mb'] = baseline_memory
            memory_metrics['inference_memory_mb'] = inference_memory
            memory_metrics['memory_increase_mb'] = inference_memory - baseline_memory
            
            # Model size (already calculated in metadata)
            memory_metrics['model_size_mb'] = getattr(model, 'model_size_mb', 0)
            
            return memory_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating memory usage: {e}")
            return {}
    
    def _evaluate_throughput(self, model: Any, test_data: Any) -> Dict[str, float]:
        """Evaluate model throughput (requests per second)."""
        try:
            throughput_metrics = {}
            
            # Batch sizes to test
            batch_sizes = [1, 8, 16, 32, 64]
            
            for batch_size in batch_sizes:
                if hasattr(test_data, '__len__') and len(test_data) < batch_size:
                    continue
                
                batch_data = test_data[:batch_size] if hasattr(test_data, '__getitem__') else test_data
                
                # Warm up
                for _ in range(5):
                    if hasattr(model, 'predict'):
                        _ = model.predict(batch_data)
                    elif hasattr(model, 'forward'):
                        model.eval()
                        with torch.no_grad():
                            if isinstance(batch_data, np.ndarray):
                                batch_data = torch.FloatTensor(batch_data)
                            _ = model(batch_data)
                
                # Measure throughput
                start_time = time.time()
                num_iterations = 50
                
                for _ in range(num_iterations):
                    if hasattr(model, 'predict'):
                        _ = model.predict(batch_data)
                    elif hasattr(model, 'forward'):
                        model.eval()
                        with torch.no_grad():
                            _ = model(batch_data)
                
                end_time = time.time()
                
                total_predictions = num_iterations * batch_size
                total_time = end_time - start_time
                throughput = total_predictions / total_time
                
                throughput_metrics[f'throughput_batch_{batch_size}_qps'] = throughput
            
            # Overall throughput (best performing batch size)
            if throughput_metrics:
                throughput_metrics['max_throughput_qps'] = max(throughput_metrics.values())
            
            return throughput_metrics
            
        except Exception as e:
            logger.error(f"Error evaluating throughput: {e}")
            return {}
    
    def _generate_optimization_recommendations(self, report: ModelPerformanceReport) -> List[str]:
        """Generate optimization recommendations based on performance report."""
        recommendations = []
        
        try:
            # Latency recommendations
            if report.latency_metrics.get('mean_latency_ms', 0) > 100:
                recommendations.append("Consider model quantization to reduce inference latency")
                recommendations.append("Evaluate model pruning to remove unnecessary parameters")
            
            # Memory recommendations
            if report.memory_metrics.get('memory_increase_mb', 0) > 500:
                recommendations.append("Consider using smaller batch sizes to reduce memory usage")
                recommendations.append("Implement model compression techniques")
            
            # Accuracy recommendations
            if report.accuracy_metrics.get('accuracy', 0) < 0.9:
                recommendations.append("Consider collecting more training data")
                recommendations.append("Evaluate ensemble methods to improve accuracy")
                recommendations.append("Perform hyperparameter tuning")
            
            # Throughput recommendations
            max_throughput = report.throughput_metrics.get('max_throughput_qps', 0)
            if max_throughput < 10:
                recommendations.append("Consider optimizing model architecture for better throughput")
                recommendations.append("Evaluate model parallelization strategies")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _assess_deployment_readiness(self, report: ModelPerformanceReport) -> bool:
        """Assess if model is ready for deployment."""
        try:
            # Define deployment criteria
            min_accuracy = 0.85
            max_latency_ms = 200
            max_memory_mb = 1000
            min_throughput_qps = 1
            
            # Check criteria
            accuracy_ok = report.accuracy_metrics.get('accuracy', 0) >= min_accuracy
            latency_ok = report.latency_metrics.get('mean_latency_ms', float('inf')) <= max_latency_ms
            memory_ok = report.memory_metrics.get('inference_memory_mb', float('inf')) <= max_memory_mb
            throughput_ok = report.throughput_metrics.get('max_throughput_qps', 0) >= min_throughput_qps
            
            return accuracy_ok and latency_ok and memory_ok and throughput_ok
            
        except Exception as e:
            logger.error(f"Error assessing deployment readiness: {e}")
            return False
    
    def _save_registry(self):
        """Save model registry to disk."""
        try:
            registry_file = self.model_registry_path / "registry.json"
            
            registry_data = {
                'models': {
                    model_id: metadata.to_dict() 
                    for model_id, metadata in self.registered_models.items()
                },
                'last_updated': datetime.now().isoformat()
            }
            
            with open(registry_file, 'w') as f:
                json.dump(registry_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def _load_registry(self):
        """Load model registry from disk."""
        try:
            registry_file = self.model_registry_path / "registry.json"
            
            if not registry_file.exists():
                logger.info("No existing registry found, starting fresh")
                return
            
            with open(registry_file, 'r') as f:
                registry_data = json.load(f)
            
            # Load models
            for model_id, metadata_dict in registry_data.get('models', {}).items():
                # Convert datetime strings back to datetime objects
                for key, value in metadata_dict.items():
                    if key.endswith('_time') or key.endswith('_at'):
                        if value:
                            metadata_dict[key] = datetime.fromisoformat(value)
                
                # Convert enums
                if 'model_type' in metadata_dict:
                    metadata_dict['model_type'] = ModelType(metadata_dict['model_type'])
                if 'status' in metadata_dict:
                    metadata_dict['status'] = ModelStatus(metadata_dict['status'])
                
                metadata = ModelMetadata(**metadata_dict)
                self.registered_models[model_id] = metadata
            
            logger.info(f"Loaded {len(self.registered_models)} models from registry")
            
        except Exception as e:
            logger.error(f"Error loading registry: {e}")
    
    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List all registered models."""
        with self.registry_lock:
            models = list(self.registered_models.values())
            
            if status:
                models = [m for m in models if m.status == status]
            
            return sorted(models, key=lambda x: x.created_at, reverse=True)
    
    def get_model_performance_history(self, model_id: str) -> List[ModelPerformanceReport]:
        """Get performance history for a model."""
        return self.performance_history.get(model_id, [])
    
    def compare_models(self, model_ids: List[str], 
                      metric: str = 'accuracy') -> Dict[str, float]:
        """Compare models based on a specific metric."""
        try:
            comparison = {}
            
            for model_id in model_ids:
                if model_id not in self.registered_models:
                    continue
                
                # Get latest performance report
                history = self.performance_history.get(model_id, [])
                if not history:
                    continue
                
                latest_report = history[-1]
                
                # Extract metric value
                if metric in latest_report.accuracy_metrics:
                    comparison[model_id] = latest_report.accuracy_metrics[metric]
                elif metric in latest_report.latency_metrics:
                    comparison[model_id] = latest_report.latency_metrics[metric]
                elif metric in latest_report.memory_metrics:
                    comparison[model_id] = latest_report.memory_metrics[metric]
                elif metric in latest_report.throughput_metrics:
                    comparison[model_id] = latest_report.throughput_metrics[metric]
            
            return comparison
            
        except Exception as e:
            logger.error(f"Error comparing models: {e}")
            return {}
    
    def delete_model(self, model_id: str, remove_artifacts: bool = True):
        """Delete a model from the registry."""
        try:
            with self.registry_lock:
                if model_id not in self.registered_models:
                    raise ValueError(f"Model {model_id} not found")
                
                # Remove from registry
                del self.registered_models[model_id]
                
                # Remove from memory cache
                if model_id in self.model_artifacts:
                    del self.model_artifacts[model_id]
                
                # Remove performance history
                if model_id in self.performance_history:
                    del self.performance_history[model_id]
                
                # Remove artifacts from disk
                if remove_artifacts:
                    model_dir = self.model_registry_path / model_id
                    if model_dir.exists():
                        import shutil
                        shutil.rmtree(model_dir)
                
                # Save updated registry
                self._save_registry()
                
                logger.info(f"Model deleted: {model_id}")
                
        except Exception as e:
            logger.error(f"Error deleting model {model_id}: {e}")
            raise

class HyperparameterTuner:
    """
    Automated hyperparameter tuning for BatteryMind models.
    """
    
    def __init__(self, optimization_target: OptimizationTarget = OptimizationTarget.ACCURACY):
        self.optimization_target = optimization_target
        self.study = None
        
        if not OPTIMIZATION_AVAILABLE:
            logger.warning("Optuna not available. Hyperparameter tuning will be limited.")
        
        logger.info(f"Hyperparameter Tuner initialized with target: {optimization_target.value}")
    
    def optimize(self, 
                objective_function: Callable,
                search_space: Dict[str, Any],
                n_trials: int = 100,
                timeout: int = 3600) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Args:
            objective_function: Function to optimize
            search_space: Parameter search space
            n_trials: Number of optimization trials
            timeout: Timeout in seconds
            
        Returns:
            Best hyperparameters
        """
        try:
            if not OPTIMIZATION_AVAILABLE:
                logger.error("Optuna not available for hyperparameter optimization")
                return {}
            
            # Create study
            direction = "maximize" if self.optimization_target in [
                OptimizationTarget.ACCURACY, OptimizationTarget.F1_SCORE, OptimizationTarget.AUC_ROC
            ] else "minimize"
            
            self.study = optuna.create_study(direction=direction)
            
            # Define objective wrapper
            def objective_wrapper(trial):
                # Sample hyperparameters
                params = {}
                for param_name, param_config in search_space.items():
                    if param_config['type'] == 'float':
                        params[param_name] = trial.suggest_float(
                            param_name, 
                            param_config['low'], 
                            param_config['high'],
                            log=param_config.get('log', False)
                        )
                    elif param_config['type'] == 'int':
                        params[param_name] = trial.suggest_int(
                            param_name,
                            param_config['low'],
                            param_config['high']
                        )
                    elif param_config['type'] == 'categorical':
                        params[param_name] = trial.suggest_categorical(
                            param_name,
                            param_config['choices']
                        )
                
                # Call objective function
                return objective_function(params)
            
            # Optimize
            self.study.optimize(
                objective_wrapper,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            logger.info(f"Optimization completed. Best value: {best_value}")
            logger.info(f"Best parameters: {best_params}")
            
            return {
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(self.study.trials),
                'study': self.study
            }
            
        except Exception as e:
            logger.error(f"Error in hyperparameter optimization: {e}")
            return {}
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history."""
        if not self.study:
            return []
        
        history = []
        for trial in self.study.trials:
            history.append({
                'trial_number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'datetime_start': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
            })
        
        return history

class ModelValidator:
    """
    Model validation utilities for BatteryMind.
    """
    
    def __init__(self):
        logger.info("Model Validator initialized")
    
    def validate_battery_health_model(self, 
                                    model: Any,
                                    test_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """
        Validate battery health prediction model.
        
        Args:
            model: Trained model
            test_data: Test dataset with features and labels
            
        Returns:
            Validation results
        """
        try:
            validation_results = {
                'overall_score': 0.0,
                'accuracy_metrics': {},
                'domain_specific_tests': {},
                'safety_checks': {},
                'recommendations': []
            }
            
            X_test = test_data['features']
            y_test = test_data['labels']
            
            # Make predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(X_test)
            else:
                raise ValueError("Model must have predict method")
            
            # Basic accuracy metrics
            if len(np.unique(y_test)) > 10:  # Regression
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                validation_results['accuracy_metrics'] = {
                    'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
                    'mae': mean_absolute_error(y_test, predictions),
                    'r2_score': r2_score(y_test, predictions)
                }
            else:  # Classification
                from sklearn.metrics import accuracy_score, precision_score, recall_score
                validation_results['accuracy_metrics'] = {
                    'accuracy': accuracy_score(y_test, predictions),
                    'precision': precision_score(y_test, predictions, average='weighted'),
                    'recall': recall_score(y_test, predictions, average='weighted')
                }
            
            # Domain-specific tests for battery health
            domain_tests = self._validate_battery_domain_logic(predictions, y_test)
            validation_results['domain_specific_tests'] = domain_tests
            
            # Safety checks
            safety_checks = self._validate_safety_constraints(predictions)
            validation_results['safety_checks'] = safety_checks
            
            # Calculate overall score
            accuracy_score = validation_results['accuracy_metrics'].get('accuracy', 
                                                                       validation_results['accuracy_metrics'].get('r2_score', 0))
            domain_score = np.mean(list(domain_tests.values())) if domain_tests else 0
            safety_score = np.mean(list(safety_checks.values())) if safety_checks else 1
            
            validation_results['overall_score'] = (accuracy_score * 0.5 + domain_score * 0.3 + safety_score * 0.2)
            
            # Generate recommendations
            if validation_results['overall_score'] < 0.8:
                validation_results['recommendations'].append("Model performance below recommended threshold")
            
            if not all(safety_checks.values()):
                validation_results['recommendations'].append("Safety constraints violated - model needs revision")
            
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating battery health model: {e}")
            return {'error': str(e)}
    
    def _validate_battery_domain_logic(self, predictions: np.ndarray, 
                                     actual: np.ndarray) -> Dict[str, float]:
        """Validate battery domain-specific logic."""
        try:
            domain_tests = {}
            
            # Test 1: SoH should not increase significantly over time
            if len(predictions) > 1:
                soh_increases = np.sum(np.diff(predictions) > 5)  # More than 5% increase
                domain_tests['soh_monotonicity'] = 1.0 - (soh_increases / len(predictions))
            
            # Test 2: Predictions should be within realistic bounds
            valid_range = np.sum((predictions >= 0) & (predictions <= 100))
            domain_tests['realistic_bounds'] = valid_range / len(predictions)
            
            # Test 3: Prediction variance should be reasonable
            prediction_std = np.std(predictions)
            domain_tests['reasonable_variance'] = 1.0 if prediction_std < 30 else 0.5
            
            return domain_tests
            
        except Exception as e:
            logger.error(f"Error in domain validation: {e}")
            return {}
    
    def _validate_safety_constraints(self, predictions: np.ndarray) -> Dict[str, bool]:
        """Validate safety constraints for battery predictions."""
        try:
            safety_checks = {}
            
            # Safety check 1: No predictions below critical threshold without warning
            critical_threshold = 20  # 20% SoH
            below_critical = np.sum(predictions < critical_threshold)
            safety_checks['critical_threshold_handling'] = below_critical < len(predictions) * 0.1
            
            # Safety check 2: No sudden drops that could indicate model instability
            if len(predictions) > 1:
                max_drop = np.max(np.abs(np.diff(predictions)))
                safety_checks['stability_check'] = max_drop < 20  # No more than 20% sudden drop
            
            # Safety check 3: Predictions within physically possible bounds
            safety_checks['physical_bounds'] = np.all((predictions >= 0) & (predictions <= 100))
            
            return safety_checks
            
        except Exception as e:
            logger.error(f"Error in safety validation: {e}")
            return {}

# Factory functions and utilities
def create_model_manager(registry_path: str = "./model_registry") -> ModelManager:
    """Create a model manager instance."""
    return ModelManager(model_registry_path=registry_path)

def create_hyperparameter_tuner(target: OptimizationTarget = OptimizationTarget.ACCURACY) -> HyperparameterTuner:
    """Create a hyperparameter tuner instance."""
    return HyperparameterTuner(optimization_target=target)

def create_model_validator() -> ModelValidator:
    """Create a model validator instance."""
    return ModelValidator()

def calculate_model_hash(model: Any) -> str:
    """Calculate hash of model for versioning."""
    try:
        if hasattr(model, 'state_dict'):
            # PyTorch model
            model_bytes = pickle.dumps(model.state_dict())
        elif hasattr(model, 'get_weights'):
            # TensorFlow model
            model_bytes = pickle.dumps(model.get_weights())
        else:
            # Generic model
            model_bytes = pickle.dumps(model)
        
        return hashlib.md5(model_bytes).hexdigest()
        
    except Exception as e:
        logger.error(f"Error calculating model hash: {e}")
        return str(uuid.uuid4())

def compress_model(model: Any, compression_type: str = 'quantization') -> Any:
    """
    Compress model using specified technique.
    
    Args:
        model: Model to compress
        compression_type: Type of compression ('quantization', 'pruning', 'distillation')
        
    Returns:
        Compressed model
    """
    try:
        if compression_type == 'quantization':
            if PYTORCH_AVAILABLE and isinstance(model, torch.nn.Module):
                # PyTorch quantization
                model.eval()
                quantized_model = torch.quantization.quantize_dynamic(
                    model, 
                    {torch.nn.Linear}, 
                    dtype=torch.qint8
                )
                return quantized_model
            
            elif TENSORFLOW_AVAILABLE and hasattr(model, 'save'):
                # TensorFlow Lite quantization
                converter = tf.lite.TFLiteConverter.from_keras_model(model)
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                quantized_tflite_model = converter.convert()
                return quantized_tflite_model
        
        logger.warning(f"Compression type {compression_type} not implemented")
        return model
        
    except Exception as e:
        logger.error(f"Error compressing model: {e}")
        return model

# Log module initialization
logger.info("BatteryMind Model Utils Module v1.0.0 loaded successfully")
