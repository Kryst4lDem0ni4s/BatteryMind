"""
BatteryMind - Schema Validator

Comprehensive schema validation system for battery data streams, ensuring data quality
and conformance to expected formats across all data ingestion points. Provides
real-time validation, schema evolution management, and detailed validation reporting
for battery telemetry, sensor data, and operational metrics.

Features:
- Real-time schema validation for streaming data
- Multi-format support (JSON, CSV, Avro, Parquet)
- Dynamic schema evolution and versioning
- Custom validation rules for battery-specific constraints
- Performance-optimized validation pipelines
- Detailed validation reporting and alerting
- Integration with data quality monitoring systems

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import json
import time
import warnings
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
from enum import Enum
import re
import hashlib

# Schema validation libraries
try:
    import jsonschema
    from jsonschema import validate, ValidationError, Draft7Validator
    JSONSCHEMA_AVAILABLE = True
except ImportError:
    JSONSCHEMA_AVAILABLE = False
    warnings.warn("jsonschema not available - JSON validation will be limited")

try:
    import cerberus
    CERBERUS_AVAILABLE = True
except ImportError:
    CERBERUS_AVAILABLE = False
    warnings.warn("cerberus not available - advanced validation rules will be limited")

# BatteryMind imports
from ...utils.logging_utils import get_logger
from ...utils.data_utils import DataProcessor
from ...utils.config_parser import ConfigParser

# Configure logging
logger = get_logger(__name__)

class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"          # Fail on any validation error
    MODERATE = "moderate"      # Warn on non-critical errors
    LENIENT = "lenient"        # Only fail on critical errors
    MONITORING = "monitoring"  # Log all issues but don't fail

class DataFormat(Enum):
    """Supported data formats."""
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"
    AVRO = "avro"
    STREAMING = "streaming"

class ValidationSeverity(Enum):
    """Validation error severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

@dataclass
class ValidationRule:
    """
    Individual validation rule definition.
    
    Attributes:
        rule_id (str): Unique identifier for the rule
        field_name (str): Name of the field to validate
        rule_type (str): Type of validation rule
        parameters (Dict[str, Any]): Rule parameters
        severity (ValidationSeverity): Severity level
        description (str): Human-readable description
        custom_validator (Callable): Custom validation function
    """
    rule_id: str
    field_name: str
    rule_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    severity: ValidationSeverity = ValidationSeverity.MEDIUM
    description: str = ""
    custom_validator: Optional[Callable] = None

@dataclass
class ValidationResult:
    """
    Result of a validation operation.
    
    Attributes:
        is_valid (bool): Whether validation passed
        errors (List[Dict]): List of validation errors
        warnings (List[Dict]): List of validation warnings
        field_validations (Dict[str, bool]): Per-field validation status
        validation_time (float): Time taken for validation
        data_hash (str): Hash of validated data
        schema_version (str): Schema version used
        metadata (Dict[str, Any]): Additional metadata
    """
    is_valid: bool
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)
    field_validations: Dict[str, bool] = field(default_factory=dict)
    validation_time: float = 0.0
    data_hash: str = ""
    schema_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SchemaDefinition:
    """
    Complete schema definition for a data type.
    
    Attributes:
        schema_id (str): Unique schema identifier
        schema_version (str): Schema version
        data_format (DataFormat): Expected data format
        json_schema (Dict): JSON Schema definition
        validation_rules (List[ValidationRule]): Custom validation rules
        required_fields (List[str]): Required field names
        field_types (Dict[str, str]): Expected field types
        field_constraints (Dict[str, Dict]): Field-specific constraints
        metadata (Dict[str, Any]): Schema metadata
    """
    schema_id: str
    schema_version: str
    data_format: DataFormat
    json_schema: Dict[str, Any] = field(default_factory=dict)
    validation_rules: List[ValidationRule] = field(default_factory=list)
    required_fields: List[str] = field(default_factory=list)
    field_types: Dict[str, str] = field(default_factory=dict)
    field_constraints: Dict[str, Dict] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BatteryDataSchemas:
    """Pre-defined schemas for battery data types."""
    
    @staticmethod
    def get_telemetry_schema() -> SchemaDefinition:
        """Get battery telemetry data schema."""
        return SchemaDefinition(
            schema_id="battery_telemetry",
            schema_version="1.0",
            data_format=DataFormat.JSON,
            json_schema={
                "type": "object",
                "properties": {
                    "battery_id": {"type": "string", "pattern": "^[A-Z0-9]{6,12}$"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "voltage": {"type": "number", "minimum": 0, "maximum": 5.0},
                    "current": {"type": "number", "minimum": -1000, "maximum": 1000},
                    "temperature": {"type": "number", "minimum": -50, "maximum": 100},
                    "soc": {"type": "number", "minimum": 0, "maximum": 1},
                    "soh": {"type": "number", "minimum": 0, "maximum": 1},
                    "internal_resistance": {"type": "number", "minimum": 0, "maximum": 10},
                    "cycle_count": {"type": "integer", "minimum": 0},
                    "capacity": {"type": "number", "minimum": 0}
                },
                "required": ["battery_id", "timestamp", "voltage", "current", "temperature"],
                "additionalProperties": False
            },
            required_fields=["battery_id", "timestamp", "voltage", "current", "temperature"],
            field_types={
                "battery_id": "string",
                "timestamp": "datetime",
                "voltage": "float",
                "current": "float",
                "temperature": "float",
                "soc": "float",
                "soh": "float",
                "internal_resistance": "float",
                "cycle_count": "integer",
                "capacity": "float"
            },
            field_constraints={
                "voltage": {"min": 0.0, "max": 5.0, "unit": "V"},
                "current": {"min": -1000.0, "max": 1000.0, "unit": "A"},
                "temperature": {"min": -50.0, "max": 100.0, "unit": "°C"},
                "soc": {"min": 0.0, "max": 1.0, "unit": "fraction"},
                "soh": {"min": 0.0, "max": 1.0, "unit": "fraction"}
            },
            validation_rules=[
                ValidationRule(
                    rule_id="voltage_range_check",
                    field_name="voltage",
                    rule_type="range",
                    parameters={"min": 2.5, "max": 4.2},
                    severity=ValidationSeverity.HIGH,
                    description="Voltage should be within normal battery operating range"
                ),
                ValidationRule(
                    rule_id="soc_soh_consistency",
                    field_name="soc",
                    rule_type="custom",
                    parameters={"related_field": "soh"},
                    severity=ValidationSeverity.MEDIUM,
                    description="SOC and SOH should be consistent",
                    custom_validator=lambda data: data.get("soc", 0) <= data.get("soh", 1)
                )
            ]
        )
    
    @staticmethod
    def get_fleet_data_schema() -> SchemaDefinition:
        """Get fleet management data schema."""
        return SchemaDefinition(
            schema_id="fleet_data",
            schema_version="1.0",
            data_format=DataFormat.JSON,
            json_schema={
                "type": "object",
                "properties": {
                    "vehicle_id": {"type": "string", "pattern": "^VH[A-Z0-9]{8}$"},
                    "battery_pack_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "location": {
                        "type": "object",
                        "properties": {
                            "latitude": {"type": "number", "minimum": -90, "maximum": 90},
                            "longitude": {"type": "number", "minimum": -180, "maximum": 180}
                        },
                        "required": ["latitude", "longitude"]
                    },
                    "speed": {"type": "number", "minimum": 0, "maximum": 300},
                    "distance_traveled": {"type": "number", "minimum": 0},
                    "energy_consumed": {"type": "number", "minimum": 0},
                    "charging_status": {"type": "string", "enum": ["charging", "discharging", "idle"]},
                    "driver_id": {"type": "string"}
                },
                "required": ["vehicle_id", "battery_pack_id", "timestamp", "location"],
                "additionalProperties": False
            },
            required_fields=["vehicle_id", "battery_pack_id", "timestamp", "location"],
            field_types={
                "vehicle_id": "string",
                "battery_pack_id": "string",
                "timestamp": "datetime",
                "location": "object",
                "speed": "float",
                "distance_traveled": "float",
                "energy_consumed": "float",
                "charging_status": "string",
                "driver_id": "string"
            }
        )
    
    @staticmethod
    def get_sensor_data_schema() -> SchemaDefinition:
        """Get multi-modal sensor data schema."""
        return SchemaDefinition(
            schema_id="sensor_data",
            schema_version="1.0",
            data_format=DataFormat.JSON,
            json_schema={
                "type": "object",
                "properties": {
                    "sensor_id": {"type": "string"},
                    "battery_id": {"type": "string"},
                    "timestamp": {"type": "string", "format": "date-time"},
                    "sensor_type": {"type": "string", "enum": ["electrical", "thermal", "acoustic", "chemical", "mechanical"]},
                    "measurements": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "number"},
                            "unit": {"type": "string"},
                            "quality_score": {"type": "number", "minimum": 0, "maximum": 1}
                        },
                        "required": ["value", "unit"]
                    },
                    "calibration_date": {"type": "string", "format": "date"},
                    "sampling_rate": {"type": "number", "minimum": 0}
                },
                "required": ["sensor_id", "battery_id", "timestamp", "sensor_type", "measurements"],
                "additionalProperties": False
            }
        )

class SchemaValidator:
    """
    Comprehensive schema validation engine for battery data.
    """
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.MODERATE):
        self.validation_level = validation_level
        self.schemas = {}
        self.validation_cache = {}
        self.performance_stats = {
            "total_validations": 0,
            "successful_validations": 0,
            "failed_validations": 0,
            "average_validation_time": 0.0
        }
        
        # Load pre-defined schemas
        self._load_battery_schemas()
        
        logger.info("SchemaValidator initialized")
    
    def _load_battery_schemas(self):
        """Load pre-defined battery data schemas."""
        battery_schemas = BatteryDataSchemas()
        
        self.schemas["battery_telemetry"] = battery_schemas.get_telemetry_schema()
        self.schemas["fleet_data"] = battery_schemas.get_fleet_data_schema()
        self.schemas["sensor_data"] = battery_schemas.get_sensor_data_schema()
        
        logger.info(f"Loaded {len(self.schemas)} pre-defined schemas")
    
    def register_schema(self, schema: SchemaDefinition):
        """Register a new schema definition."""
        self.schemas[schema.schema_id] = schema
        logger.info(f"Registered schema: {schema.schema_id} v{schema.schema_version}")
    
    def validate_data(self, data: Union[Dict, pd.DataFrame, List], 
                     schema_id: str) -> ValidationResult:
        """
        Validate data against a registered schema.
        
        Args:
            data: Data to validate
            schema_id: ID of schema to validate against
            
        Returns:
            ValidationResult: Comprehensive validation results
        """
        start_time = time.time()
        
        if schema_id not in self.schemas:
            raise ValueError(f"Schema {schema_id} not found")
        
        schema = self.schemas[schema_id]
        
        try:
            # Convert data to appropriate format for validation
            validation_data = self._prepare_data_for_validation(data, schema.data_format)
            
            # Perform validation
            result = self._perform_validation(validation_data, schema)
            
            # Calculate validation time
            result.validation_time = time.time() - start_time
            
            # Generate data hash
            result.data_hash = self._generate_data_hash(validation_data)
            result.schema_version = schema.schema_version
            
            # Update performance statistics
            self._update_performance_stats(result)
            
            logger.debug(f"Validation completed for {schema_id}: {result.is_valid}")
            return result
            
        except Exception as e:
            logger.error(f"Validation failed for {schema_id}: {e}")
            return ValidationResult(
                is_valid=False,
                errors=[{
                    "type": "validation_error",
                    "message": str(e),
                    "severity": ValidationSeverity.CRITICAL.value
                }],
                validation_time=time.time() - start_time
            )
    
    def _prepare_data_for_validation(self, data: Union[Dict, pd.DataFrame, List], 
                                   data_format: DataFormat) -> Dict:
        """Prepare data for validation based on format."""
        if isinstance(data, dict):
            return data
        elif isinstance(data, pd.DataFrame):
            if len(data) == 1:
                return data.iloc[0].to_dict()
            else:
                raise ValueError("DataFrame validation requires single row")
        elif isinstance(data, list):
            if len(data) == 1 and isinstance(data[0], dict):
                return data[0]
            else:
                raise ValueError("List validation requires single dictionary element")
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")
    
    def _perform_validation(self, data: Dict, schema: SchemaDefinition) -> ValidationResult:
        """Perform comprehensive validation."""
        result = ValidationResult(is_valid=True)
        
        # JSON Schema validation
        if JSONSCHEMA_AVAILABLE and schema.json_schema:
            json_result = self._validate_json_schema(data, schema.json_schema)
            result.errors.extend(json_result["errors"])
            result.warnings.extend(json_result["warnings"])
            if json_result["errors"]:
                result.is_valid = False
        
        # Field type validation
        type_result = self._validate_field_types(data, schema.field_types)
        result.field_validations.update(type_result["field_validations"])
        result.errors.extend(type_result["errors"])
        if type_result["errors"]:
            result.is_valid = False
        
        # Required fields validation
        required_result = self._validate_required_fields(data, schema.required_fields)
        result.errors.extend(required_result["errors"])
        if required_result["errors"]:
            result.is_valid = False
        
        # Field constraints validation
        constraints_result = self._validate_field_constraints(data, schema.field_constraints)
        result.errors.extend(constraints_result["errors"])
        result.warnings.extend(constraints_result["warnings"])
        if constraints_result["errors"]:
            result.is_valid = False
        
        # Custom validation rules
        custom_result = self._validate_custom_rules(data, schema.validation_rules)
        result.errors.extend(custom_result["errors"])
        result.warnings.extend(custom_result["warnings"])
        if custom_result["errors"]:
            result.is_valid = False
        
        # Apply validation level filtering
        result = self._apply_validation_level_filtering(result)
        
        return result
    
    def _validate_json_schema(self, data: Dict, json_schema: Dict) -> Dict:
        """Validate data against JSON schema."""
        errors = []
        warnings = []
        
        try:
            validate(instance=data, schema=json_schema)
        except ValidationError as e:
            errors.append({
                "type": "json_schema_error",
                "field": e.path[-1] if e.path else "root",
                "message": e.message,
                "severity": ValidationSeverity.HIGH.value,
                "value": e.instance if hasattr(e, 'instance') else None
            })
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_field_types(self, data: Dict, field_types: Dict[str, str]) -> Dict:
        """Validate field data types."""
        errors = []
        field_validations = {}
        
        for field_name, expected_type in field_types.items():
            if field_name in data:
                value = data[field_name]
                is_valid = self._check_field_type(value, expected_type)
                field_validations[field_name] = is_valid
                
                if not is_valid:
                    errors.append({
                        "type": "type_error",
                        "field": field_name,
                        "message": f"Expected {expected_type}, got {type(value).__name__}",
                        "severity": ValidationSeverity.HIGH.value,
                        "value": value
                    })
        
        return {"errors": errors, "field_validations": field_validations}
    
    def _check_field_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected type."""
        type_checkers = {
            "string": lambda x: isinstance(x, str),
            "integer": lambda x: isinstance(x, int),
            "float": lambda x: isinstance(x, (int, float)),
            "boolean": lambda x: isinstance(x, bool),
            "datetime": lambda x: self._is_datetime_string(x),
            "object": lambda x: isinstance(x, dict),
            "array": lambda x: isinstance(x, list)
        }
        
        checker = type_checkers.get(expected_type.lower())
        return checker(value) if checker else True
    
    def _is_datetime_string(self, value: str) -> bool:
        """Check if string is a valid datetime."""
        if not isinstance(value, str):
            return False
        
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',
            r'\d{4}-\d{2}-\d{2}'
        ]
        
        return any(re.match(pattern, value) for pattern in datetime_patterns)
    
    def _validate_required_fields(self, data: Dict, required_fields: List[str]) -> Dict:
        """Validate required fields are present."""
        errors = []
        
        for field in required_fields:
            if field not in data:
                errors.append({
                    "type": "missing_field",
                    "field": field,
                    "message": f"Required field '{field}' is missing",
                    "severity": ValidationSeverity.CRITICAL.value
                })
            elif data[field] is None:
                errors.append({
                    "type": "null_field",
                    "field": field,
                    "message": f"Required field '{field}' is null",
                    "severity": ValidationSeverity.HIGH.value
                })
        
        return {"errors": errors}
    
    def _validate_field_constraints(self, data: Dict, field_constraints: Dict[str, Dict]) -> Dict:
        """Validate field-specific constraints."""
        errors = []
        warnings = []
        
        for field_name, constraints in field_constraints.items():
            if field_name in data:
                value = data[field_name]
                
                # Range constraints
                if "min" in constraints and value < constraints["min"]:
                    errors.append({
                        "type": "range_error",
                        "field": field_name,
                        "message": f"Value {value} below minimum {constraints['min']}",
                        "severity": ValidationSeverity.HIGH.value,
                        "value": value
                    })
                
                if "max" in constraints and value > constraints["max"]:
                    errors.append({
                        "type": "range_error",
                        "field": field_name,
                        "message": f"Value {value} above maximum {constraints['max']}",
                        "severity": ValidationSeverity.HIGH.value,
                        "value": value
                    })
                
                # Pattern constraints
                if "pattern" in constraints and isinstance(value, str):
                    if not re.match(constraints["pattern"], value):
                        errors.append({
                            "type": "pattern_error",
                            "field": field_name,
                            "message": f"Value '{value}' does not match required pattern",
                            "severity": ValidationSeverity.MEDIUM.value,
                            "value": value
                        })
        
        return {"errors": errors, "warnings": warnings}
    
    def _validate_custom_rules(self, data: Dict, validation_rules: List[ValidationRule]) -> Dict:
        """Validate custom business rules."""
        errors = []
        warnings = []
        
        for rule in validation_rules:
            try:
                if rule.custom_validator:
                    is_valid = rule.custom_validator(data)
                    if not is_valid:
                        error_item = {
                            "type": "custom_rule_error",
                            "rule_id": rule.rule_id,
                            "field": rule.field_name,
                            "message": rule.description,
                            "severity": rule.severity.value
                        }
                        
                        if rule.severity in [ValidationSeverity.CRITICAL, ValidationSeverity.HIGH]:
                            errors.append(error_item)
                        else:
                            warnings.append(error_item)
                            
            except Exception as e:
                errors.append({
                    "type": "custom_rule_error",
                    "rule_id": rule.rule_id,
                    "field": rule.field_name,
                    "message": f"Custom validation failed: {str(e)}",
                    "severity": ValidationSeverity.MEDIUM.value
                })
        
        return {"errors": errors, "warnings": warnings}
    
    def _apply_validation_level_filtering(self, result: ValidationResult) -> ValidationResult:
        """Apply validation level filtering to results."""
        if self.validation_level == ValidationLevel.STRICT:
            # Treat all errors and warnings as failures
            if result.errors or result.warnings:
                result.is_valid = False
        
        elif self.validation_level == ValidationLevel.MODERATE:
            # Only critical and high severity errors cause failure
            critical_errors = [e for e in result.errors 
                             if e.get("severity") in ["critical", "high"]]
            result.is_valid = len(critical_errors) == 0
        
        elif self.validation_level == ValidationLevel.LENIENT:
            # Only critical errors cause failure
            critical_errors = [e for e in result.errors 
                             if e.get("severity") == "critical"]
            result.is_valid = len(critical_errors) == 0
        
        elif self.validation_level == ValidationLevel.MONITORING:
            # Always pass validation, just log issues
            result.is_valid = True
        
        return result
    
    def _generate_data_hash(self, data: Dict) -> str:
        """Generate hash for data integrity checking."""
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()
    
    def _update_performance_stats(self, result: ValidationResult):
        """Update validation performance statistics."""
        self.performance_stats["total_validations"] += 1
        
        if result.is_valid:
            self.performance_stats["successful_validations"] += 1
        else:
            self.performance_stats["failed_validations"] += 1
        
        # Update average validation time
        total = self.performance_stats["total_validations"]
        current_avg = self.performance_stats["average_validation_time"]
        new_avg = (current_avg * (total - 1) + result.validation_time) / total
        self.performance_stats["average_validation_time"] = new_avg
    
    def validate_streaming_data(self, data_stream: List[Dict], 
                              schema_id: str) -> List[ValidationResult]:
        """Validate streaming data in batches."""
        results = []
        
        for data_point in data_stream:
            result = self.validate_data(data_point, schema_id)
            results.append(result)
            
            # Early termination for critical errors in strict mode
            if (self.validation_level == ValidationLevel.STRICT and 
                not result.is_valid):
                break
        
        return results
    
    def get_schema_info(self, schema_id: str) -> Dict[str, Any]:
        """Get information about a registered schema."""
        if schema_id not in self.schemas:
            raise ValueError(f"Schema {schema_id} not found")
        
        schema = self.schemas[schema_id]
        return {
            "schema_id": schema.schema_id,
            "schema_version": schema.schema_version,
            "data_format": schema.data_format.value,
            "required_fields": schema.required_fields,
            "field_types": schema.field_types,
            "validation_rules_count": len(schema.validation_rules),
            "metadata": schema.metadata
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats["total_validations"] > 0:
            stats["success_rate"] = (stats["successful_validations"] / 
                                   stats["total_validations"]) * 100
        else:
            stats["success_rate"] = 0.0
        
        return stats
    
    def export_validation_report(self, results: List[ValidationResult], 
                               output_path: str):
        """Export validation results to a report."""
        report = {
            "validation_summary": {
                "total_validations": len(results),
                "successful_validations": sum(1 for r in results if r.is_valid),
                "failed_validations": sum(1 for r in results if not r.is_valid),
                "report_generated": datetime.now().isoformat()
            },
            "detailed_results": []
        }
        
        for i, result in enumerate(results):
            report["detailed_results"].append({
                "validation_index": i,
                "is_valid": result.is_valid,
                "errors": result.errors,
                "warnings": result.warnings,
                "validation_time": result.validation_time,
                "data_hash": result.data_hash,
                "schema_version": result.schema_version
            })
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Validation report exported to {output_path}")

# Factory functions
def create_schema_validator(validation_level: ValidationLevel = ValidationLevel.MODERATE) -> SchemaValidator:
    """Create a schema validator instance."""
    return SchemaValidator(validation_level)

def validate_battery_data(data: Union[Dict, pd.DataFrame], 
                         data_type: str = "battery_telemetry") -> ValidationResult:
    """Convenience function to validate battery data."""
    validator = create_schema_validator()
    return validator.validate_data(data, data_type)
