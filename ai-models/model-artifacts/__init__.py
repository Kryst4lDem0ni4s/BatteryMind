"""
BatteryMind - Model Artifacts Management

Comprehensive model artifact management system for tracking, versioning,
and deploying trained AI/ML models across the BatteryMind ecosystem.

This module provides centralized management of:
- Trained model artifacts and weights
- Model configurations and hyperparameters
- Training history and performance metrics
- Model metadata and versioning information
- Export formats for different deployment targets

Key Components:
- Model registry and versioning system
- Artifact storage and retrieval utilities
- Model performance tracking and comparison
- Export format management (ONNX, TensorFlow Lite, TensorRT)
- Deployment configuration management

Features:
- Automated model versioning with semantic versioning
- Comprehensive metadata tracking for reproducibility
- Multi-format model exports for different deployment scenarios
- Performance benchmarking and comparison tools
- Integration with MLOps pipelines and CI/CD systems
- Model lineage tracking and experiment management

Author: BatteryMind Development Team
Version: 1.0.0
License: Proprietary - Tata Technologies InnoVent 2025
"""

import os
import json
import yaml
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
import hashlib
import shutil

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"
__email__ = "batterymind@tatatechnologies.com"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model artifact types
MODEL_TYPES = {
    "transformer": {
        "description": "Transformer-based battery health prediction models",
        "supported_formats": ["pkl", "h5", "onnx", "tflite"],
        "base_path": "trained_models/transformer_v{version}",
        "required_files": ["model.pkl", "config.json", "model_metadata.yaml"]
    },
    "federated": {
        "description": "Federated learning global models and configurations",
        "supported_formats": ["pkl", "npy", "json"],
        "base_path": "trained_models/federated_v{version}",
        "required_files": ["global_model.pkl", "federation_history.json", "privacy_params.yaml"]
    },
    "rl_agent": {
        "description": "Reinforcement learning agents for battery optimization",
        "supported_formats": ["pt", "pkl", "json"],
        "base_path": "trained_models/rl_agent_v{version}",
        "required_files": ["policy_network.pt", "value_network.pt", "environment_config.yaml"]
    },
    "ensemble": {
        "description": "Ensemble models combining multiple prediction approaches",
        "supported_formats": ["pkl", "tar.gz", "npy"],
        "base_path": "trained_models/ensemble_v{version}",
        "required_files": ["ensemble_model.pkl", "base_models.tar.gz", "ensemble_config.json"]
    }
}

# Export format configurations
EXPORT_FORMATS = {
    "onnx": {
        "description": "ONNX format for cross-platform deployment",
        "file_extension": ".onnx",
        "target_path": "exports/onnx_models",
        "optimization_levels": ["basic", "extended", "all"]
    },
    "tensorflow_lite": {
        "description": "TensorFlow Lite for mobile and edge deployment",
        "file_extension": ".tflite",
        "target_path": "exports/tensorflow_lite",
        "quantization_options": ["float16", "int8", "dynamic"]
    },
    "tensorrt": {
        "description": "TensorRT optimized models for NVIDIA GPUs",
        "file_extension": ".trt",
        "target_path": "exports/tensorrt_optimized",
        "precision_modes": ["fp32", "fp16", "int8"]
    },
    "edge": {
        "description": "Edge-optimized models for IoT deployment",
        "file_extension": ".pkl",
        "target_path": "exports/edge_models",
        "optimization_techniques": ["quantization", "pruning", "compression"]
    }
}

@dataclass
class ModelMetadata:
    """
    Comprehensive metadata for trained models.
    
    Attributes:
        model_id (str): Unique model identifier
        model_type (str): Type of model (transformer, federated, rl_agent, ensemble)
        version (str): Model version following semantic versioning
        name (str): Human-readable model name
        description (str): Model description and purpose
        
        # Training information
        training_start_time (str): Training start timestamp
        training_end_time (str): Training completion timestamp
        training_duration_hours (float): Total training duration
        training_dataset (str): Training dataset identifier
        validation_dataset (str): Validation dataset identifier
        
        # Model architecture
        architecture (Dict[str, Any]): Model architecture details
        hyperparameters (Dict[str, Any]): Training hyperparameters
        model_size_mb (float): Model size in megabytes
        parameter_count (int): Total number of model parameters
        
        # Performance metrics
        training_metrics (Dict[str, float]): Training performance metrics
        validation_metrics (Dict[str, float]): Validation performance metrics
        test_metrics (Dict[str, float]): Test performance metrics
        benchmark_scores (Dict[str, float]): Benchmark comparison scores
        
        # Deployment information
        target_platforms (List[str]): Supported deployment platforms
        hardware_requirements (Dict[str, Any]): Hardware requirements
        inference_latency_ms (float): Average inference latency
        memory_usage_mb (float): Memory usage during inference
        
        # Lineage and dependencies
        parent_models (List[str]): Parent model identifiers
        derived_models (List[str]): Models derived from this model
        dependencies (List[str]): External dependencies
        framework_version (str): ML framework version used
        
        # Quality and validation
        validation_status (str): Model validation status
        quality_score (float): Overall quality score
        safety_validation (bool): Safety validation passed
        regulatory_compliance (bool): Regulatory compliance status
        
        # Metadata
        created_by (str): Model creator
        tags (List[str]): Model tags for categorization
        notes (str): Additional notes and comments
        checksum (str): Model file checksum for integrity
    """
    model_id: str
    model_type: str
    version: str
    name: str
    description: str
    
    # Training information
    training_start_time: str = ""
    training_end_time: str = ""
    training_duration_hours: float = 0.0
    training_dataset: str = ""
    validation_dataset: str = ""
    
    # Model architecture
    architecture: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    model_size_mb: float = 0.0
    parameter_count: int = 0
    
    # Performance metrics
    training_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    test_metrics: Dict[str, float] = field(default_factory=dict)
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    
    # Deployment information
    target_platforms: List[str] = field(default_factory=list)
    hardware_requirements: Dict[str, Any] = field(default_factory=dict)
    inference_latency_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    # Lineage and dependencies
    parent_models: List[str] = field(default_factory=list)
    derived_models: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    framework_version: str = ""
    
    # Quality and validation
    validation_status: str = "pending"
    quality_score: float = 0.0
    safety_validation: bool = False
    regulatory_compliance: bool = False
    
    # Metadata
    created_by: str = ""
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    checksum: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return asdict(self)
    
    def to_yaml(self) -> str:
        """Convert metadata to YAML string."""
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelMetadata':
        """Create metadata from dictionary."""
        return cls(**data)
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'ModelMetadata':
        """Create metadata from YAML string."""
        data = yaml.safe_load(yaml_str)
        return cls.from_dict(data)

class ModelArtifactManager:
    """
    Centralized manager for model artifacts and metadata.
    """
    
    def __init__(self, base_path: str = "./model-artifacts"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize directory structure
        self._initialize_directory_structure()
        
        # Load model registry
        self.registry_path = self.base_path / "version_control" / "model_registry.json"
        self.model_registry = self._load_model_registry()
        
        logger.info(f"ModelArtifactManager initialized at {self.base_path}")
    
    def _initialize_directory_structure(self):
        """Initialize the directory structure for model artifacts."""
        directories = [
            "trained_models",
            "checkpoints",
            "performance_metrics",
            "version_control",
            "exports/onnx_models",
            "exports/tensorflow_lite",
            "exports/tensorrt_optimized",
            "exports/edge_models"
        ]
        
        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)
    
    def _load_model_registry(self) -> Dict[str, Any]:
        """Load model registry from file."""
        if self.registry_path.exists():
            with open(self.registry_path, 'r') as f:
                return json.load(f)
        else:
            return {"models": {}, "versions": {}, "latest": {}}
    
    def _save_model_registry(self):
        """Save model registry to file."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.registry_path, 'w') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def register_model(self, metadata: ModelMetadata, model_files: Dict[str, str]) -> str:
        """
        Register a new model with the artifact manager.
        
        Args:
            metadata (ModelMetadata): Model metadata
            model_files (Dict[str, str]): Dictionary of file types to file paths
            
        Returns:
            str: Model artifact path
        """
        # Generate model directory path
        model_dir = self.base_path / MODEL_TYPES[metadata.model_type]["base_path"].format(
            version=metadata.version
        )
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files to artifact directory
        artifact_files = {}
        for file_type, source_path in model_files.items():
            if os.path.exists(source_path):
                dest_path = model_dir / os.path.basename(source_path)
                shutil.copy2(source_path, dest_path)
                artifact_files[file_type] = str(dest_path)
                
                # Calculate checksum for integrity
                with open(dest_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    if not metadata.checksum:
                        metadata.checksum = file_hash
        
        # Save metadata
        metadata_path = model_dir / "model_metadata.yaml"
        with open(metadata_path, 'w') as f:
            f.write(metadata.to_yaml())
        
        # Update model registry
        self.model_registry["models"][metadata.model_id] = {
            "metadata": metadata.to_dict(),
            "artifact_path": str(model_dir),
            "files": artifact_files,
            "registered_at": datetime.now().isoformat()
        }
        
        # Update version tracking
        if metadata.model_type not in self.model_registry["versions"]:
            self.model_registry["versions"][metadata.model_type] = []
        
        self.model_registry["versions"][metadata.model_type].append({
            "version": metadata.version,
            "model_id": metadata.model_id,
            "created_at": datetime.now().isoformat()
        })
        
        # Update latest version
        self.model_registry["latest"][metadata.model_type] = metadata.model_id
        
        # Save registry
        self._save_model_registry()
        
        logger.info(f"Registered model {metadata.model_id} v{metadata.version}")
        return str(model_dir)
    
    def get_model_metadata(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        if model_id in self.model_registry["models"]:
            metadata_dict = self.model_registry["models"][model_id]["metadata"]
            return ModelMetadata.from_dict(metadata_dict)
        return None
    
    def get_latest_model(self, model_type: str) -> Optional[str]:
        """Get the latest model ID for a given type."""
        return self.model_registry["latest"].get(model_type)
    
    def list_models(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by type."""
        models = []
        for model_id, model_info in self.model_registry["models"].items():
            metadata = model_info["metadata"]
            if model_type is None or metadata["model_type"] == model_type:
                models.append({
                    "model_id": model_id,
                    "model_type": metadata["model_type"],
                    "version": metadata["version"],
                    "name": metadata["name"],
                    "created_at": model_info["registered_at"]
                })
        
        return sorted(models, key=lambda x: x["created_at"], reverse=True)
    
    def export_model(self, model_id: str, export_format: str, 
                    optimization_config: Optional[Dict[str, Any]] = None) -> str:
        """
        Export model to specified format.
        
        Args:
            model_id (str): Model identifier
            export_format (str): Target export format
            optimization_config (Dict[str, Any], optional): Optimization configuration
            
        Returns:
            str: Path to exported model
        """
        if model_id not in self.model_registry["models"]:
            raise ValueError(f"Model {model_id} not found in registry")
        
        if export_format not in EXPORT_FORMATS:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        model_info = self.model_registry["models"][model_id]
        metadata = ModelMetadata.from_dict(model_info["metadata"])
        
        # Create export directory
        export_config = EXPORT_FORMATS[export_format]
        export_dir = self.base_path / export_config["target_path"]
        export_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate export filename
        export_filename = f"{metadata.model_type}_{metadata.name.lower().replace(' ', '_')}{export_config['file_extension']}"
        export_path = export_dir / export_filename
        
        # Perform format-specific export
        if export_format == "onnx":
            self._export_to_onnx(model_info, export_path, optimization_config)
        elif export_format == "tensorflow_lite":
            self._export_to_tflite(model_info, export_path, optimization_config)
        elif export_format == "tensorrt":
            self._export_to_tensorrt(model_info, export_path, optimization_config)
        elif export_format == "edge":
            self._export_to_edge(model_info, export_path, optimization_config)
        
        logger.info(f"Exported model {model_id} to {export_format} format at {export_path}")
        return str(export_path)
    
    def _export_to_onnx(self, model_info: Dict[str, Any], export_path: Path, 
                       config: Optional[Dict[str, Any]]):
        """Export model to ONNX format."""
        # Implementation would depend on the specific model framework
        # This is a placeholder for the actual ONNX export logic
        logger.info(f"ONNX export to {export_path} (placeholder implementation)")
    
    def _export_to_tflite(self, model_info: Dict[str, Any], export_path: Path,
                         config: Optional[Dict[str, Any]]):
        """Export model to TensorFlow Lite format."""
        # Implementation would depend on the specific model framework
        logger.info(f"TensorFlow Lite export to {export_path} (placeholder implementation)")
    
    def _export_to_tensorrt(self, model_info: Dict[str, Any], export_path: Path,
                           config: Optional[Dict[str, Any]]):
        """Export model to TensorRT format."""
        # Implementation would depend on the specific model framework
        logger.info(f"TensorRT export to {export_path} (placeholder implementation)")
    
    def _export_to_edge(self, model_info: Dict[str, Any], export_path: Path,
                       config: Optional[Dict[str, Any]]):
        """Export model to edge-optimized format."""
        # Implementation would include quantization, pruning, and compression
        logger.info(f"Edge export to {export_path} (placeholder implementation)")
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple models across various metrics."""
        comparison_data = {
            "models": [],
            "metrics_comparison": {},
            "recommendations": []
        }
        
        for model_id in model_ids:
            if model_id in self.model_registry["models"]:
                model_info = self.model_registry["models"][model_id]
                metadata = model_info["metadata"]
                
                comparison_data["models"].append({
                    "model_id": model_id,
                    "name": metadata["name"],
                    "version": metadata["version"],
                    "model_type": metadata["model_type"],
                    "training_metrics": metadata.get("training_metrics", {}),
                    "validation_metrics": metadata.get("validation_metrics", {}),
                    "model_size_mb": metadata.get("model_size_mb", 0),
                    "inference_latency_ms": metadata.get("inference_latency_ms", 0)
                })
        
        return comparison_data
    
    def get_model_lineage(self, model_id: str) -> Dict[str, Any]:
        """Get model lineage and dependency graph."""
        if model_id not in self.model_registry["models"]:
            return {}
        
        metadata = self.model_registry["models"][model_id]["metadata"]
        
        lineage = {
            "model_id": model_id,
            "parents": metadata.get("parent_models", []),
            "children": metadata.get("derived_models", []),
            "dependencies": metadata.get("dependencies", []),
            "lineage_depth": 0
        }
        
        # Calculate lineage depth (simplified)
        depth = 0
        current_parents = lineage["parents"]
        while current_parents:
            depth += 1
            next_parents = []
            for parent_id in current_parents:
                if parent_id in self.model_registry["models"]:
                    parent_metadata = self.model_registry["models"][parent_id]["metadata"]
                    next_parents.extend(parent_metadata.get("parent_models", []))
            current_parents = next_parents
            if depth > 10:  # Prevent infinite loops
                break
        
        lineage["lineage_depth"] = depth
        return lineage

# Global model artifact manager instance
_model_manager = None

def get_model_manager(base_path: str = "./model-artifacts") -> ModelArtifactManager:
    """Get the global model artifact manager instance."""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelArtifactManager(base_path)
    return _model_manager

# Utility functions
def create_model_metadata(model_id: str, model_type: str, version: str, 
                         name: str, description: str, **kwargs) -> ModelMetadata:
    """Create model metadata with default values."""
    return ModelMetadata(
        model_id=model_id,
        model_type=model_type,
        version=version,
        name=name,
        description=description,
        **kwargs
    )

def validate_model_artifacts(model_path: str, model_type: str) -> Dict[str, bool]:
    """Validate model artifacts for completeness."""
    validation_results = {}
    required_files = MODEL_TYPES[model_type]["required_files"]
    
    model_dir = Path(model_path)
    for required_file in required_files:
        file_path = model_dir / required_file
        validation_results[required_file] = file_path.exists()
    
    return validation_results

def calculate_model_checksum(model_path: str) -> str:
    """Calculate checksum for model integrity verification."""
    model_dir = Path(model_path)
    checksums = []
    
    for file_path in model_dir.rglob("*"):
        if file_path.is_file():
            with open(file_path, 'rb') as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
                checksums.append(f"{file_path.name}:{file_hash}")
    
    # Create combined checksum
    combined_hash = hashlib.sha256("|".join(sorted(checksums)).encode()).hexdigest()
    return combined_hash

# Module exports
__all__ = [
    "ModelMetadata",
    "ModelArtifactManager", 
    "get_model_manager",
    "create_model_metadata",
    "validate_model_artifacts",
    "calculate_model_checksum",
    "MODEL_TYPES",
    "EXPORT_FORMATS"
]

# Module initialization
logger.info(f"BatteryMind Model Artifacts v{__version__} initialized")
logger.info(f"Supported model types: {list(MODEL_TYPES.keys())}")
logger.info(f"Supported export formats: {list(EXPORT_FORMATS.keys())}")
