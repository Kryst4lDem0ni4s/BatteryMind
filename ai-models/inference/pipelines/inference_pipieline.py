"""
BatteryMind - Core Inference Pipeline

Production-ready inference pipeline for battery health prediction, degradation
forecasting, and optimization recommendations. Supports real-time and batch
inference with comprehensive monitoring and error handling.

Features:
- Multi-model ensemble inference orchestration
- Real-time data preprocessing and validation
- Prediction confidence scoring and uncertainty quantification
- Comprehensive logging and monitoring
- Graceful error handling and fallback mechanisms
- Model versioning and A/B testing support

Author: BatteryMind Development Team
Version: 1.0.0
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Model loading and inference
import joblib
import torch
import tensorflow as tf
from transformers import AutoTokenizer, AutoModel

# Data processing and validation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import pydantic
from pydantic import BaseModel, validator

# Monitoring and observability
from prometheus_client import Counter, Histogram, Gauge
import structlog

# Internal imports
from ..predictors.battery_health_predictor import BatteryHealthPredictor
from ..predictors.degradation_predictor import DegradationPredictor
from ..predictors.optimization_predictor import OptimizationPredictor
from ..predictors.ensemble_predictor import EnsemblePredictor
from ...training_data.preprocessing_scripts.data_cleaner import BatteryDataCleaner
from ...training_data.preprocessing_scripts.normalization import BatteryDataNormalizer
from ...training_data.preprocessing_scripts.feature_extractor import BatteryFeatureExtractor

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
INFERENCE_REQUESTS = Counter('inference_requests_total', 'Total inference requests', ['model_type', 'status'])
INFERENCE_LATENCY = Histogram('inference_latency_seconds', 'Inference latency', ['model_type'])
ACTIVE_MODELS = Gauge('active_models_count', 'Number of active models', ['model_type'])
PREDICTION_CONFIDENCE = Histogram('prediction_confidence_score', 'Prediction confidence scores', ['model_type'])

@dataclass
class InferenceConfig:
    """Configuration for inference pipeline."""
    
    # Model configuration
    model_base_path: str = "../../model-artifacts/trained_models"
    enable_ensemble: bool = True
    enable_uncertainty_quantification: bool = True
    
    # Performance settings
    batch_size: int = 32
    max_sequence_length: int = 100
    prediction_timeout_seconds: float = 30.0
    
    # Quality thresholds
    min_confidence_threshold: float = 0.7
    max_prediction_age_hours: float = 1.0
    
    # Monitoring and logging
    enable_detailed_logging: bool = True
    log_predictions: bool = True
    enable_metrics: bool = True
    
    # Fallback and error handling
    enable_fallback_models: bool = True
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    
    # Model versioning
    model_version: str = "1.0.0"
    enable_ab_testing: bool = False
    ab_test_ratio: float = 0.1

class BatteryDataInput(BaseModel):
    """Input data model for battery inference."""
    
    battery_id: str
    timestamp: datetime
    voltage: float
    current: float
    temperature: float
    state_of_charge: float
    cycle_count: int
    age_days: int
    
    # Optional fields
    state_of_health: Optional[float] = None
    capacity: Optional[float] = None
    internal_resistance: Optional[float] = None
    
    # Metadata
    battery_type: str = "li_ion"
    manufacturer: str = "unknown"
    model_name: str = "unknown"
    
    @validator('voltage')
    def validate_voltage(cls, v):
        if not 2.0 <= v <= 5.0:
            raise ValueError('Voltage must be between 2.0V and 5.0V')
        return v
    
    @validator('current')
    def validate_current(cls, v):
        if not -500.0 <= v <= 500.0:
            raise ValueError('Current must be between -500A and 500A')
        return v
    
    @validator('temperature')
    def validate_temperature(cls, v):
        if not -50.0 <= v <= 100.0:
            raise ValueError('Temperature must be between -50°C and 100°C')
        return v
    
    @validator('state_of_charge')
    def validate_soc(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError('SOC must be between 0.0 and 1.0')
        return v

class InferenceResult(BaseModel):
    """Output model for inference results."""
    
    # Prediction results
    battery_id: str
    timestamp: datetime
    prediction_timestamp: datetime
    
    # Health predictions
    state_of_health: float
    state_of_health_confidence: float
    degradation_rate: float
    remaining_useful_life_days: int
    
    # Forecasting results
    capacity_forecast_30d: float
    capacity_forecast_90d: float
    capacity_forecast_365d: float
    
    # Optimization recommendations
    optimal_charging_current: float
    optimal_charging_voltage: float
    recommended_actions: List[str]
    
    # Uncertainty and confidence
    prediction_uncertainty: float
    model_confidence: float
    
    # Metadata
    model_version: str
    inference_time_ms: float
    models_used: List[str]
    
    # Quality indicators
    data_quality_score: float
    prediction_quality_score: float
    
    # Alerts and warnings
    alerts: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

class BatteryInferencePipeline:
    """
    Main inference pipeline for battery health prediction and optimization.
    """
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.models = {}
        self.preprocessors = {}
        self.model_metadata = {}
        self.inference_cache = {}
        
        # Initialize components
        self._initialize_models()
        self._initialize_preprocessors()
        self._setup_monitoring()
        
        logger.info("BatteryInferencePipeline initialized", config=config.__dict__)
    
    def _initialize_models(self):
        """Initialize all prediction models."""
        try:
            # Load individual predictors
            self.models['health_predictor'] = BatteryHealthPredictor(
                model_path=f"{self.config.model_base_path}/transformer_v1.0"
            )
            
            self.models['degradation_predictor'] = DegradationPredictor(
                model_path=f"{self.config.model_base_path}/transformer_v1.0"
            )
            
            self.models['optimization_predictor'] = OptimizationPredictor(
                model_path=f"{self.config.model_base_path}/rl_agent_v1.0"
            )
            
            # Load ensemble model if enabled
            if self.config.enable_ensemble:
                self.models['ensemble_predictor'] = EnsemblePredictor(
                    model_path=f"{self.config.model_base_path}/ensemble_v1.0"
                )
            
            # Load model metadata
            self._load_model_metadata()
            
            # Update metrics
            for model_type in self.models.keys():
                ACTIVE_MODELS.labels(model_type=model_type).set(1)
            
            logger.info("Models initialized successfully", models=list(self.models.keys()))
            
        except Exception as e:
            logger.error("Failed to initialize models", error=str(e))
            raise
    
    def _initialize_preprocessors(self):
        """Initialize data preprocessing components."""
        try:
            # Data cleaner
            self.preprocessors['cleaner'] = BatteryDataCleaner()
            
            # Data normalizer
            self.preprocessors['normalizer'] = BatteryDataNormalizer()
            
            # Feature extractor
            self.preprocessors['feature_extractor'] = BatteryFeatureExtractor()
            
            logger.info("Preprocessors initialized successfully")
            
        except Exception as e:
            logger.error("Failed to initialize preprocessors", error=str(e))
            raise
    
    def _load_model_metadata(self):
        """Load model metadata and configuration."""
        for model_name, model in self.models.items():
            try:
                metadata_path = Path(f"{self.config.model_base_path}/{model_name}/model_metadata.yaml")
                if metadata_path.exists():
                    with open(metadata_path, 'r') as f:
                        self.model_metadata[model_name] = json.load(f)
                else:
                    self.model_metadata[model_name] = {
                        "version": self.config.model_version,
                        "created_at": datetime.now().isoformat(),
                        "performance_metrics": {}
                    }
            except Exception as e:
                logger.warning(f"Failed to load metadata for {model_name}", error=str(e))
                self.model_metadata[model_name] = {}
    
    def _setup_monitoring(self):
        """Setup monitoring and alerting."""
        if self.config.enable_metrics:
            # Initialize metrics collectors
            self.metrics_enabled = True
            logger.info("Monitoring enabled")
        else:
            self.metrics_enabled = False
    
    async def predict(self, input_data: BatteryDataInput) -> InferenceResult:
        """
        Main prediction method for single battery input.
        
        Args:
            input_data: Battery sensor data and metadata
            
        Returns:
            InferenceResult: Comprehensive prediction results
        """
        start_time = time.time()
        
        try:
            # Validate input data
            self._validate_input_data(input_data)
            
            # Check cache for recent predictions
            cached_result = self._check_prediction_cache(input_data)
            if cached_result:
                logger.info("Returning cached prediction", battery_id=input_data.battery_id)
                return cached_result
            
            # Preprocess input data
            processed_data = await self._preprocess_data(input_data)
            
            # Run predictions
            predictions = await self._run_predictions(processed_data)
            
            # Post-process and validate results
            result = self._postprocess_predictions(predictions, input_data, start_time)
            
            # Cache result
            self._cache_prediction(input_data, result)
            
            # Record metrics
            if self.metrics_enabled:
                INFERENCE_REQUESTS.labels(model_type='ensemble', status='success').inc()
                INFERENCE_LATENCY.labels(model_type='ensemble').observe(time.time() - start_time)
                PREDICTION_CONFIDENCE.labels(model_type='ensemble').observe(result.model_confidence)
            
            logger.info("Prediction completed successfully", 
                       battery_id=input_data.battery_id,
                       inference_time_ms=result.inference_time_ms)
            
            return result
            
        except Exception as e:
            # Handle errors and fallback
            error_result = await self._handle_prediction_error(input_data, e, start_time)
            
            if self.metrics_enabled:
                INFERENCE_REQUESTS.labels(model_type='ensemble', status='error').inc()
            
            logger.error("Prediction failed", 
                        battery_id=input_data.battery_id,
                        error=str(e))
            
            return error_result
    
    async def predict_batch(self, input_batch: List[BatteryDataInput]) -> List[InferenceResult]:
        """
        Batch prediction for multiple battery inputs.
        
        Args:
            input_batch: List of battery sensor data
            
        Returns:
            List[InferenceResult]: Batch prediction results
        """
        if not input_batch:
            return []
        
        logger.info("Starting batch prediction", batch_size=len(input_batch))
        
        # Process in batches for memory efficiency
        results = []
        batch_size = self.config.batch_size
        
        for i in range(0, len(input_batch), batch_size):
            batch = input_batch[i:i + batch_size]
            
            # Process batch concurrently
            batch_tasks = [self.predict(item) for item in batch]
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            # Handle exceptions in batch results
            for result in batch_results:
                if isinstance(result, Exception):
                    logger.error("Batch prediction item failed", error=str(result))
                    # Create error result
                    error_result = InferenceResult(
                        battery_id="unknown",
                        timestamp=datetime.now(),
                        prediction_timestamp=datetime.now(),
                        state_of_health=0.0,
                        state_of_health_confidence=0.0,
                        degradation_rate=0.0,
                        remaining_useful_life_days=0,
                        capacity_forecast_30d=0.0,
                        capacity_forecast_90d=0.0,
                        capacity_forecast_365d=0.0,
                        optimal_charging_current=0.0,
                        optimal_charging_voltage=0.0,
                        recommended_actions=[],
                        prediction_uncertainty=1.0,
                        model_confidence=0.0,
                        model_version=self.config.model_version,
                        inference_time_ms=0.0,
                        models_used=[],
                        data_quality_score=0.0,
                        prediction_quality_score=0.0,
                        alerts=["Prediction failed"],
                        warnings=["Error in batch processing"]
                    )
                    results.append(error_result)
                else:
                    results.append(result)
        
        logger.info("Batch prediction completed", 
                   batch_size=len(input_batch),
                   successful_predictions=len([r for r in results if not r.alerts]))
        
        return results
    
    def _validate_input_data(self, input_data: BatteryDataInput):
        """Validate input data quality and completeness."""
        # Check data freshness
        age = datetime.now() - input_data.timestamp
        if age > timedelta(hours=self.config.max_prediction_age_hours):
            raise ValueError(f"Input data is too old: {age}")
        
        # Check for required fields
        required_fields = ['voltage', 'current', 'temperature', 'state_of_charge']
        for field in required_fields:
            if getattr(input_data, field) is None:
                raise ValueError(f"Missing required field: {field}")
        
        # Physics-based validation
        if input_data.voltage <= 0:
            raise ValueError("Voltage must be positive")
        
        if input_data.state_of_charge < 0 or input_data.state_of_charge > 1:
            raise ValueError("SOC must be between 0 and 1")
    
    def _check_prediction_cache(self, input_data: BatteryDataInput) -> Optional[InferenceResult]:
        """Check if recent prediction exists in cache."""
        cache_key = f"{input_data.battery_id}_{input_data.timestamp.isoformat()}"
        
        if cache_key in self.inference_cache:
            cached_result, cache_time = self.inference_cache[cache_key]
            
            # Check if cache is still valid
            if datetime.now() - cache_time < timedelta(minutes=5):
                return cached_result
            else:
                # Remove expired cache
                del self.inference_cache[cache_key]
        
        return None
    
    def _cache_prediction(self, input_data: BatteryDataInput, result: InferenceResult):
        """Cache prediction result."""
        cache_key = f"{input_data.battery_id}_{input_data.timestamp.isoformat()}"
        self.inference_cache[cache_key] = (result, datetime.now())
        
        # Limit cache size
        if len(self.inference_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.inference_cache.keys())[:100]
            for key in oldest_keys:
                del self.inference_cache[key]
    
    async def _preprocess_data(self, input_data: BatteryDataInput) -> Dict[str, Any]:
        """Preprocess input data for model inference."""
        # Convert to DataFrame
        data_dict = {
            'voltage': input_data.voltage,
            'current': input_data.current,
            'temperature': input_data.temperature,
            'state_of_charge': input_data.state_of_charge,
            'cycle_count': input_data.cycle_count,
            'age_days': input_data.age_days
        }
        
        # Optional fields
        if input_data.state_of_health is not None:
            data_dict['state_of_health'] = input_data.state_of_health
        if input_data.capacity is not None:
            data_dict['capacity'] = input_data.capacity
        if input_data.internal_resistance is not None:
            data_dict['internal_resistance'] = input_data.internal_resistance
        
        df = pd.DataFrame([data_dict])
        
        # Clean data
        cleaned_data = self.preprocessors['cleaner'].clean(df)
        
        # Normalize data
        normalized_data = self.preprocessors['normalizer'].transform(cleaned_data)
        
        # Extract features
        features = self.preprocessors['feature_extractor'].extract_features(normalized_data)
        
        return {
            'raw_data': df,
            'cleaned_data': cleaned_data,
            'normalized_data': normalized_data,
            'features': features,
            'input_metadata': {
                'battery_id': input_data.battery_id,
                'timestamp': input_data.timestamp,
                'battery_type': input_data.battery_type,
                'manufacturer': input_data.manufacturer
            }
        }
    
    async def _run_predictions(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run predictions using all available models."""
        predictions = {}
        
        # Run individual model predictions
        prediction_tasks = []
        
        for model_name, model in self.models.items():
            if model_name != 'ensemble_predictor':
                task = self._run_single_model_prediction(model_name, model, processed_data)
                prediction_tasks.append(task)
        
        # Execute predictions concurrently
        individual_results = await asyncio.gather(*prediction_tasks, return_exceptions=True)
        
        # Process individual results
        for i, (model_name, _) in enumerate(self.models.items()):
            if model_name != 'ensemble_predictor':
                result = individual_results[i]
                if isinstance(result, Exception):
                    logger.warning(f"Model {model_name} failed", error=str(result))
                    predictions[model_name] = None
                else:
                    predictions[model_name] = result
        
        # Run ensemble prediction if enabled
        if self.config.enable_ensemble and 'ensemble_predictor' in self.models:
            try:
                ensemble_result = await self._run_ensemble_prediction(processed_data, predictions)
                predictions['ensemble'] = ensemble_result
            except Exception as e:
                logger.warning("Ensemble prediction failed", error=str(e))
                predictions['ensemble'] = None
        
        return predictions
    
    async def _run_single_model_prediction(self, model_name: str, model: Any, 
                                         processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run prediction for a single model."""
        try:
            if model_name == 'health_predictor':
                result = await model.predict_health(processed_data['features'])
            elif model_name == 'degradation_predictor':
                result = await model.predict_degradation(processed_data['features'])
            elif model_name == 'optimization_predictor':
                result = await model.predict_optimization(processed_data['features'])
            else:
                result = await model.predict(processed_data['features'])
            
            return result
            
        except Exception as e:
            logger.error(f"Single model prediction failed for {model_name}", error=str(e))
            raise
    
    async def _run_ensemble_prediction(self, processed_data: Dict[str, Any], 
                                     individual_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Run ensemble prediction combining individual model results."""
        try:
            ensemble_model = self.models['ensemble_predictor']
            
            # Prepare ensemble input
            ensemble_input = {
                'features': processed_data['features'],
                'individual_predictions': individual_predictions
            }
            
            result = await ensemble_model.predict_ensemble(ensemble_input)
            return result
            
        except Exception as e:
            logger.error("Ensemble prediction failed", error=str(e))
            raise
    
    def _postprocess_predictions(self, predictions: Dict[str, Any], 
                               input_data: BatteryDataInput, 
                               start_time: float) -> InferenceResult:
        """Post-process predictions and create final result."""
        
        # Use ensemble result if available, otherwise combine individual results
        if 'ensemble' in predictions and predictions['ensemble'] is not None:
            main_prediction = predictions['ensemble']
        else:
            main_prediction = self._combine_individual_predictions(predictions)
        
        # Calculate confidence and uncertainty
        model_confidence = self._calculate_model_confidence(predictions)
        prediction_uncertainty = self._calculate_prediction_uncertainty(predictions)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(main_prediction, input_data)
        
        # Assess data and prediction quality
        data_quality_score = self._assess_data_quality(input_data)
        prediction_quality_score = self._assess_prediction_quality(main_prediction, model_confidence)
        
        # Generate alerts and warnings
        alerts, warnings = self._generate_alerts_and_warnings(main_prediction, input_data)
        
        # Create final result
        result = InferenceResult(
            battery_id=input_data.battery_id,
            timestamp=input_data.timestamp,
            prediction_timestamp=datetime.now(),
            
            # Health predictions
            state_of_health=main_prediction.get('state_of_health', 0.0),
            state_of_health_confidence=main_prediction.get('soh_confidence', 0.0),
            degradation_rate=main_prediction.get('degradation_rate', 0.0),
            remaining_useful_life_days=main_prediction.get('remaining_useful_life_days', 0),
            
            # Forecasting results
            capacity_forecast_30d=main_prediction.get('capacity_forecast_30d', 0.0),
            capacity_forecast_90d=main_prediction.get('capacity_forecast_90d', 0.0),
            capacity_forecast_365d=main_prediction.get('capacity_forecast_365d', 0.0),
            
            # Optimization recommendations
            optimal_charging_current=main_prediction.get('optimal_charging_current', 0.0),
            optimal_charging_voltage=main_prediction.get('optimal_charging_voltage', 0.0),
            recommended_actions=recommendations,
            
            # Uncertainty and confidence
            prediction_uncertainty=prediction_uncertainty,
            model_confidence=model_confidence,
            
            # Metadata
            model_version=self.config.model_version,
            inference_time_ms=(time.time() - start_time) * 1000,
            models_used=list(predictions.keys()),
            
            # Quality indicators
            data_quality_score=data_quality_score,
            prediction_quality_score=prediction_quality_score,
            
            # Alerts and warnings
            alerts=alerts,
            warnings=warnings
        )
        
        return result
    
    def _combine_individual_predictions(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Combine individual model predictions when ensemble is not available."""
        combined = {}
        
        # Health prediction
        if 'health_predictor' in predictions and predictions['health_predictor']:
            health_pred = predictions['health_predictor']
            combined['state_of_health'] = health_pred.get('state_of_health', 0.0)
            combined['soh_confidence'] = health_pred.get('confidence', 0.0)
        
        # Degradation prediction
        if 'degradation_predictor' in predictions and predictions['degradation_predictor']:
            deg_pred = predictions['degradation_predictor']
            combined['degradation_rate'] = deg_pred.get('degradation_rate', 0.0)
            combined['remaining_useful_life_days'] = deg_pred.get('remaining_useful_life_days', 0)
            combined['capacity_forecast_30d'] = deg_pred.get('capacity_forecast_30d', 0.0)
            combined['capacity_forecast_90d'] = deg_pred.get('capacity_forecast_90d', 0.0)
            combined['capacity_forecast_365d'] = deg_pred.get('capacity_forecast_365d', 0.0)
        
        # Optimization prediction
        if 'optimization_predictor' in predictions and predictions['optimization_predictor']:
            opt_pred = predictions['optimization_predictor']
            combined['optimal_charging_current'] = opt_pred.get('optimal_charging_current', 0.0)
            combined['optimal_charging_voltage'] = opt_pred.get('optimal_charging_voltage', 0.0)
        
        return combined
    
    def _calculate_model_confidence(self, predictions: Dict[str, Any]) -> float:
        """Calculate overall model confidence score."""
        confidences = []
        
        for model_name, prediction in predictions.items():
            if prediction is not None:
                if 'confidence' in prediction:
                    confidences.append(prediction['confidence'])
                elif 'soh_confidence' in prediction:
                    confidences.append(prediction['soh_confidence'])
        
        if confidences:
            return np.mean(confidences)
        else:
            return 0.0
    
    def _calculate_prediction_uncertainty(self, predictions: Dict[str, Any]) -> float:
        """Calculate prediction uncertainty based on model agreement."""
        if len(predictions) < 2:
            return 0.0
        
        # Calculate variance in predictions
        soh_predictions = []
        for prediction in predictions.values():
            if prediction is not None and 'state_of_health' in prediction:
                soh_predictions.append(prediction['state_of_health'])
        
        if len(soh_predictions) > 1:
            return np.std(soh_predictions)
        else:
            return 0.0
    
    def _generate_recommendations(self, prediction: Dict[str, Any], 
                                input_data: BatteryDataInput) -> List[str]:
        """Generate actionable recommendations based on predictions."""
        recommendations = []
        
        soh = prediction.get('state_of_health', 1.0)
        temp = input_data.temperature
        
        # Health-based recommendations
        if soh < 0.8:
            recommendations.append("Consider battery replacement - SOH below 80%")
        elif soh < 0.9:
            recommendations.append("Schedule preventive maintenance - SOH declining")
        
        # Temperature-based recommendations
        if temp > 45:
            recommendations.append("Reduce charging rate - high temperature detected")
        elif temp < 0:
            recommendations.append("Warm battery before charging - low temperature")
        
        # SOC-based recommendations
        if input_data.state_of_charge < 0.1:
            recommendations.append("Charge battery soon - SOC critically low")
        elif input_data.state_of_charge > 0.9:
            recommendations.append("Consider reducing charging rate - SOC high")
        
        return recommendations
    
    def _assess_data_quality(self, input_data: BatteryDataInput) -> float:
        """Assess quality of input data."""
        quality_score = 1.0
        
        # Check for extreme values
        if input_data.temperature > 50 or input_data.temperature < -10:
            quality_score -= 0.1
        
        if input_data.voltage < 2.5 or input_data.voltage > 4.5:
            quality_score -= 0.1
        
        if abs(input_data.current) > 100:
            quality_score -= 0.1
        
        return max(0.0, quality_score)
    
    def _assess_prediction_quality(self, prediction: Dict[str, Any], 
                                 confidence: float) -> float:
        """Assess quality of prediction results."""
        quality_score = confidence
        
        # Check for reasonable prediction values
        soh = prediction.get('state_of_health', 1.0)
        if soh < 0.0 or soh > 1.0:
            quality_score -= 0.2
        
        degradation_rate = prediction.get('degradation_rate', 0.0)
        if degradation_rate < 0.0 or degradation_rate > 0.1:
            quality_score -= 0.1
        
        return max(0.0, quality_score)
    
    def _generate_alerts_and_warnings(self, prediction: Dict[str, Any], 
                                    input_data: BatteryDataInput) -> Tuple[List[str], List[str]]:
        """Generate alerts and warnings based on predictions and input data."""
        alerts = []
        warnings = []
        
        soh = prediction.get('state_of_health', 1.0)
        temp = input_data.temperature
        
        # Critical alerts
        if soh < 0.7:
            alerts.append("CRITICAL: Battery health critically low")
        
        if temp > 60:
            alerts.append("CRITICAL: Battery temperature too high")
        
        if input_data.state_of_charge < 0.05:
            alerts.append("CRITICAL: Battery charge critically low")
        
        # Warnings
        if soh < 0.8:
            warnings.append("Battery health below recommended threshold")
        
        if temp > 45:
            warnings.append("Battery temperature elevated")
        
        if input_data.state_of_charge < 0.1:
            warnings.append("Battery charge low")
        
        return alerts, warnings
    
    async def _handle_prediction_error(self, input_data: BatteryDataInput, 
                                     error: Exception, 
                                     start_time: float) -> InferenceResult:
        """Handle prediction errors and provide fallback result."""
        logger.error("Prediction error occurred", 
                    battery_id=input_data.battery_id,
                    error=str(error))
        
        # Create fallback result
        fallback_result = InferenceResult(
            battery_id=input_data.battery_id,
            timestamp=input_data.timestamp,
            prediction_timestamp=datetime.now(),
            
            # Conservative fallback values
            state_of_health=0.8,  # Assume moderate health
            state_of_health_confidence=0.0,
            degradation_rate=0.001,  # Conservative degradation rate
            remaining_useful_life_days=365,  # Conservative estimate
            
            # Forecasting results
            capacity_forecast_30d=0.0,
            capacity_forecast_90d=0.0,
            capacity_forecast_365d=0.0,
            
            # Safe optimization recommendations
            optimal_charging_current=min(10.0, abs(input_data.current)),
            optimal_charging_voltage=min(4.0, input_data.voltage),
            recommended_actions=["Use conservative charging parameters"],
            
            # Uncertainty and confidence
            prediction_uncertainty=1.0,
            model_confidence=0.0,
            
            # Metadata
            model_version=self.config.model_version,
            inference_time_ms=(time.time() - start_time) * 1000,
            models_used=["fallback"],
            
            # Quality indicators
            data_quality_score=0.0,
            prediction_quality_score=0.0,
            
            # Alerts and warnings
            alerts=[f"Prediction failed: {str(error)}"],
            warnings=["Using fallback prediction values"]
        )
        
        return fallback_result
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of the inference pipeline."""
        return {
            "status": "healthy",
            "models_loaded": len(self.models),
            "active_models": list(self.models.keys()),
            "cache_size": len(self.inference_cache),
            "config": self.config.__dict__,
            "model_metadata": self.model_metadata
        }
