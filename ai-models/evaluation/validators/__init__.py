"""
BatteryMind Evaluation Validators Module

This module provides comprehensive validation capabilities for battery management
system AI models, data quality, performance metrics, and safety constraints.

The validators ensure that all components of the BatteryMind system meet
quality standards, safety requirements, and performance expectations.

Available Validators:
- ModelValidator: Validates AI model performance and behavior
- DataValidator: Validates data quality and integrity
- PerformanceValidator: Validates system performance metrics
- SafetyValidator: Validates safety constraints and requirements

Author: BatteryMind Development Team
Version: 1.0.0
"""

from typing import Dict, List, Any, Optional, Union
import logging
import importlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BatteryMind Development Team"

# Validation result types
from enum import Enum

class ValidationStatus(Enum):
    """Enumeration for validation status."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

class ValidationSeverity(Enum):
    """Enumeration for validation severity levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"

# Import validator classes
try:
    from .model_validator import ModelValidator
    logger.info("✓ ModelValidator imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import ModelValidator: {e}")
    ModelValidator = None

try:
    from .data_validator import DataValidator
    logger.info("✓ DataValidator imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import DataValidator: {e}")
    DataValidator = None

try:
    from .performance_validator import PerformanceValidator
    logger.info("✓ PerformanceValidator imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import PerformanceValidator: {e}")
    PerformanceValidator = None

try:
    from .safety_validator import SafetyValidator
    logger.info("✓ SafetyValidator imported successfully")
except ImportError as e:
    logger.warning(f"Failed to import SafetyValidator: {e}")
    SafetyValidator = None

# Validation result container
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class ValidationResult:
    """Container for validation results."""
    validator_name: str
    test_name: str
    status: ValidationStatus
    severity: ValidationSeverity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary format."""
        return {
            "validator_name": self.validator_name,
            "test_name": self.test_name,
            "status": self.status.value,
            "severity": self.severity.value,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp.isoformat(),
            "execution_time_ms": self.execution_time_ms
        }

@dataclass
class ValidationSuite:
    """Container for a suite of validation results."""
    suite_name: str
    results: List[ValidationResult] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    
    def add_result(self, result: ValidationResult) -> None:
        """Add a validation result to the suite."""
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics for the validation suite."""
        if not self.results:
            return {
                "total_tests": 0,
                "passed": 0,
                "failed": 0,
                "warnings": 0,
                "skipped": 0,
                "success_rate": 0.0
            }
        
        status_counts = {
            ValidationStatus.PASSED: 0,
            ValidationStatus.FAILED: 0,
            ValidationStatus.WARNING: 0,
            ValidationStatus.SKIPPED: 0
        }
        
        for result in self.results:
            status_counts[result.status] += 1
        
        total_tests = len(self.results)
        success_rate = (status_counts[ValidationStatus.PASSED] / total_tests) * 100
        
        return {
            "suite_name": self.suite_name,
            "total_tests": total_tests,
            "passed": status_counts[ValidationStatus.PASSED],
            "failed": status_counts[ValidationStatus.FAILED],
            "warnings": status_counts[ValidationStatus.WARNING],
            "skipped": status_counts[ValidationStatus.SKIPPED],
            "success_rate": success_rate,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_seconds": (self.end_time - self.start_time).total_seconds() if self.end_time else None
        }
    
    def finalize(self) -> None:
        """Finalize the validation suite."""
        self.end_time = datetime.now()

class ValidatorManager:
    """
    Manager class for coordinating all validation activities.
    """
    
    def __init__(self):
        self.validators = {}
        self.validation_history = []
        self._initialize_validators()
    
    def _initialize_validators(self) -> None:
        """Initialize all available validators."""
        validator_classes = {
            "model": ModelValidator,
            "data": DataValidator,
            "performance": PerformanceValidator,
            "safety": SafetyValidator
        }
        
        for name, validator_class in validator_classes.items():
            if validator_class is not None:
                try:
                    self.validators[name] = validator_class()
                    logger.info(f"✓ {name.capitalize()}Validator initialized")
                except Exception as e:
                    logger.error(f"Failed to initialize {name.capitalize()}Validator: {e}")
            else:
                logger.warning(f"{name.capitalize()}Validator not available")
    
    def get_available_validators(self) -> List[str]:
        """Get list of available validators."""
        return list(self.validators.keys())
    
    def validate_model(self, model: Any, validation_config: Dict[str, Any]) -> ValidationSuite:
        """
        Validate a machine learning model.
        
        Args:
            model: Model to validate
            validation_config: Configuration for validation
            
        Returns:
            ValidationSuite with results
        """
        suite = ValidationSuite(suite_name="model_validation")
        
        if "model" not in self.validators:
            result = ValidationResult(
                validator_name="ModelValidator",
                test_name="availability_check",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.HIGH,
                message="ModelValidator not available"
            )
            suite.add_result(result)
            suite.finalize()
            return suite
        
        try:
            validator = self.validators["model"]
            results = validator.validate_model(model, validation_config)
            
            for result in results:
                suite.add_result(result)
            
        except Exception as e:
            result = ValidationResult(
                validator_name="ModelValidator",
                test_name="validation_execution",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Model validation failed: {str(e)}"
            )
            suite.add_result(result)
        
        suite.finalize()
        self.validation_history.append(suite)
        return suite
    
    def validate_data(self, data: Any, validation_config: Dict[str, Any]) -> ValidationSuite:
        """
        Validate data quality and integrity.
        
        Args:
            data: Data to validate
            validation_config: Configuration for validation
            
        Returns:
            ValidationSuite with results
        """
        suite = ValidationSuite(suite_name="data_validation")
        
        if "data" not in self.validators:
            result = ValidationResult(
                validator_name="DataValidator",
                test_name="availability_check",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.HIGH,
                message="DataValidator not available"
            )
            suite.add_result(result)
            suite.finalize()
            return suite
        
        try:
            validator = self.validators["data"]
            results = validator.validate_data(data, validation_config)
            
            for result in results:
                suite.add_result(result)
            
        except Exception as e:
            result = ValidationResult(
                validator_name="DataValidator",
                test_name="validation_execution",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Data validation failed: {str(e)}"
            )
            suite.add_result(result)
        
        suite.finalize()
        self.validation_history.append(suite)
        return suite
    
    def validate_performance(self, metrics: Dict[str, Any], 
                           validation_config: Dict[str, Any]) -> ValidationSuite:
        """
        Validate system performance metrics.
        
        Args:
            metrics: Performance metrics to validate
            validation_config: Configuration for validation
            
        Returns:
            ValidationSuite with results
        """
        suite = ValidationSuite(suite_name="performance_validation")
        
        if "performance" not in self.validators:
            result = ValidationResult(
                validator_name="PerformanceValidator",
                test_name="availability_check",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.HIGH,
                message="PerformanceValidator not available"
            )
            suite.add_result(result)
            suite.finalize()
            return suite
        
        try:
            validator = self.validators["performance"]
            results = validator.validate_performance(metrics, validation_config)
            
            for result in results:
                suite.add_result(result)
            
        except Exception as e:
            result = ValidationResult(
                validator_name="PerformanceValidator",
                test_name="validation_execution",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Performance validation failed: {str(e)}"
            )
            suite.add_result(result)
        
        suite.finalize()
        self.validation_history.append(suite)
        return suite
    
    def validate_safety(self, system_state: Dict[str, Any], 
                       validation_config: Dict[str, Any]) -> ValidationSuite:
        """
        Validate safety constraints and requirements.
        
        Args:
            system_state: Current system state to validate
            validation_config: Configuration for validation
            
        Returns:
            ValidationSuite with results
        """
        suite = ValidationSuite(suite_name="safety_validation")
        
        if "safety" not in self.validators:
            result = ValidationResult(
                validator_name="SafetyValidator",
                test_name="availability_check",
                status=ValidationStatus.SKIPPED,
                severity=ValidationSeverity.CRITICAL,
                message="SafetyValidator not available"
            )
            suite.add_result(result)
            suite.finalize()
            return suite
        
        try:
            validator = self.validators["safety"]
            results = validator.validate_safety(system_state, validation_config)
            
            for result in results:
                suite.add_result(result)
            
        except Exception as e:
            result = ValidationResult(
                validator_name="SafetyValidator",
                test_name="validation_execution",
                status=ValidationStatus.FAILED,
                severity=ValidationSeverity.CRITICAL,
                message=f"Safety validation failed: {str(e)}"
            )
            suite.add_result(result)
        
        suite.finalize()
        self.validation_history.append(suite)
        return suite
    
    def run_comprehensive_validation(self, 
                                   model: Any,
                                   data: Any,
                                   metrics: Dict[str, Any],
                                   system_state: Dict[str, Any],
                                   validation_config: Dict[str, Any]) -> Dict[str, ValidationSuite]:
        """
        Run comprehensive validation across all validators.
        
        Args:
            model: Model to validate
            data: Data to validate
            metrics: Performance metrics to validate
            system_state: System state to validate
            validation_config: Configuration for validation
            
        Returns:
            Dictionary of validation suites by validator type
        """
        logger.info("Starting comprehensive validation")
        
        validation_suites = {}
        
        # Run model validation
        logger.info("Running model validation...")
        validation_suites["model"] = self.validate_model(model, validation_config)
        
        # Run data validation
        logger.info("Running data validation...")
        validation_suites["data"] = self.validate_data(data, validation_config)
        
        # Run performance validation
        logger.info("Running performance validation...")
        validation_suites["performance"] = self.validate_performance(metrics, validation_config)
        
        # Run safety validation
        logger.info("Running safety validation...")
        validation_suites["safety"] = self.validate_safety(system_state, validation_config)
        
        logger.info("Comprehensive validation completed")
        return validation_suites
    
    def get_validation_summary(self, validation_suites: Dict[str, ValidationSuite]) -> Dict[str, Any]:
        """
        Get summary of validation results across all suites.
        
        Args:
            validation_suites: Dictionary of validation suites
            
        Returns:
            Comprehensive validation summary
        """
        summary = {
            "overall_status": ValidationStatus.PASSED.value,
            "total_tests": 0,
            "total_passed": 0,
            "total_failed": 0,
            "total_warnings": 0,
            "total_skipped": 0,
            "success_rate": 0.0,
            "suite_summaries": {}
        }
        
        critical_failures = 0
        high_failures = 0
        
        for suite_name, suite in validation_suites.items():
            suite_summary = suite.get_summary()
            summary["suite_summaries"][suite_name] = suite_summary
            
            summary["total_tests"] += suite_summary["total_tests"]
            summary["total_passed"] += suite_summary["passed"]
            summary["total_failed"] += suite_summary["failed"]
            summary["total_warnings"] += suite_summary["warnings"]
            summary["total_skipped"] += suite_summary["skipped"]
            
            # Check for critical failures
            for result in suite.results:
                if result.status == ValidationStatus.FAILED:
                    if result.severity == ValidationSeverity.CRITICAL:
                        critical_failures += 1
                    elif result.severity == ValidationSeverity.HIGH:
                        high_failures += 1
        
        # Calculate overall success rate
        if summary["total_tests"] > 0:
            summary["success_rate"] = (summary["total_passed"] / summary["total_tests"]) * 100
        
        # Determine overall status
        if critical_failures > 0:
            summary["overall_status"] = ValidationStatus.FAILED.value
        elif high_failures > 0 or summary["total_failed"] > 0:
            summary["overall_status"] = ValidationStatus.WARNING.value
        elif summary["total_warnings"] > 0:
            summary["overall_status"] = ValidationStatus.WARNING.value
        else:
            summary["overall_status"] = ValidationStatus.PASSED.value
        
        summary["critical_failures"] = critical_failures
        summary["high_failures"] = high_failures
        
        return summary
    
    def save_validation_results(self, validation_suites: Dict[str, ValidationSuite], 
                               filename: str) -> None:
        """
        Save validation results to file.
        
        Args:
            validation_suites: Dictionary of validation suites
            filename: Output filename
        """
        import json
        
        results_data = {
            "validation_timestamp": datetime.now().isoformat(),
            "summary": self.get_validation_summary(validation_suites),
            "suites": {}
        }
        
        for suite_name, suite in validation_suites.items():
            results_data["suites"][suite_name] = {
                "summary": suite.get_summary(),
                "results": [result.to_dict() for result in suite.results]
            }
        
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Validation results saved to {filename}")

# Create default validator manager instance
validator_manager = ValidatorManager()

# Convenience functions for easy access
def validate_model(model: Any, validation_config: Dict[str, Any]) -> ValidationSuite:
    """Convenience function for model validation."""
    return validator_manager.validate_model(model, validation_config)

def validate_data(data: Any, validation_config: Dict[str, Any]) -> ValidationSuite:
    """Convenience function for data validation."""
    return validator_manager.validate_data(data, validation_config)

def validate_performance(metrics: Dict[str, Any], 
                        validation_config: Dict[str, Any]) -> ValidationSuite:
    """Convenience function for performance validation."""
    return validator_manager.validate_performance(metrics, validation_config)

def validate_safety(system_state: Dict[str, Any], 
                   validation_config: Dict[str, Any]) -> ValidationSuite:
    """Convenience function for safety validation."""
    return validator_manager.validate_safety(system_state, validation_config)

def run_comprehensive_validation(model: Any,
                                data: Any,
                                metrics: Dict[str, Any],
                                system_state: Dict[str, Any],
                                validation_config: Dict[str, Any]) -> Dict[str, ValidationSuite]:
    """Convenience function for comprehensive validation."""
    return validator_manager.run_comprehensive_validation(
        model, data, metrics, system_state, validation_config
    )

# Export all public components
__all__ = [
    # Enums
    "ValidationStatus",
    "ValidationSeverity",
    
    # Data classes
    "ValidationResult",
    "ValidationSuite",
    
    # Validator classes
    "ModelValidator",
    "DataValidator", 
    "PerformanceValidator",
    "SafetyValidator",
    
    # Manager class
    "ValidatorManager",
    "validator_manager",
    
    # Convenience functions
    "validate_model",
    "validate_data",
    "validate_performance",
    "validate_safety",
    "run_comprehensive_validation",
    
    # Version info
    "__version__",
    "__author__"
]

# Module initialization
logger.info(f"BatteryMind Validators module initialized (v{__version__})")
logger.info(f"Available validators: {list(validator_manager.validators.keys())}")
