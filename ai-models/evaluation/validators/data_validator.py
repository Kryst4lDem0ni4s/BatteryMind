"""
BatteryMind - Data Validator

Comprehensive data validation framework for battery management systems.
Validates data quality, consistency, completeness, and domain-specific
constraints for battery telemetry and related datasets.

This module implements:
- Data quality validation (completeness, consistency, accuracy)
- Schema validation and data type checking
- Domain-specific validation for battery data
- Temporal consistency validation
- Outlier detection and anomaly identification
- Data drift detection
- Feature validation and correlation analysis

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
import logging
from datetime import datetime, timedelta
import json
import warnings
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...utils.config_parser import ConfigParser
from ...training_data.generators.physics_simulator import BatteryPhysicsSimulator

logger = setup_logger(__name__)

class DataValidationLevel(Enum):
    """Data validation level enumeration."""
    BASIC = "basic"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"
    PRODUCTION = "production"

class ValidationStatus(Enum):
    """Validation status enumeration."""
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"
    PENDING = "pending"

@dataclass
class DataValidationResult:
    """Data validation result data structure."""
    test_name: str
    status: ValidationStatus
    score: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class DataValidationReport:
    """Comprehensive data validation report."""
    dataset_name: str
    dataset_type: str
    validation_level: DataValidationLevel
    overall_status: ValidationStatus
    overall_score: float
    data_summary: Dict[str, Any] = field(default_factory=dict)
    results: List[DataValidationResult] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

class BaseDataValidator(ABC):
    """Base class for all data validators."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = setup_logger(self.__class__.__name__)
    
    @abstractmethod
    def validate(self, data: pd.DataFrame) -> DataValidationResult:
        """Abstract validation method."""
        pass
    
    def _calculate_score(self, metric_value: float, threshold: float, 
                        higher_is_better: bool = True) -> float:
        """Calculate normalized score based on metric and threshold."""
        if higher_is_better:
            return min(1.0, metric_value / threshold)
        else:
            return max(0.0, 1.0 - metric_value / threshold)

class CompletenessValidator(BaseDataValidator):
    """Validator for data completeness."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.completeness_threshold = config.get('completeness_threshold', 0.95)
        self.critical_columns = config.get('critical_columns', [])
    
    def validate(self, data: pd.DataFrame) -> DataValidationResult:
        """Validate data completeness."""
        try:
            total_cells = data.shape[0] * data.shape[1]
            missing_cells = data.isnull().sum().sum()
            completeness_ratio = 1.0 - (missing_cells / total_cells)
            
            # Check completeness per column
            column_completeness = {}
            critical_failures = []
            
            for column in data.columns:
                col_completeness = 1.0 - (data[column].isnull().sum() / len(data))
                column_completeness[column] = col_completeness
                
                if column in self.critical_columns and col_completeness < self.completeness_threshold:
                    critical_failures.append(column)
            
            # Calculate score
            score = self._calculate_score(completeness_ratio, self.completeness_threshold, True)
            
            # Determine status
            if critical_failures:
                status = ValidationStatus.FAILED
            elif completeness_ratio >= self.completeness_threshold:
                status = ValidationStatus.PASSED
            else:
                status = ValidationStatus.WARNING
            
            # Generate recommendations
            recommendations = []
            if missing_cells > 0:
                recommendations.append(f"Dataset has {missing_cells} missing values ({missing_cells/total_cells:.2%})")
            if critical_failures:
                recommendations.append(f"Critical columns with low completeness: {critical_failures}")
            
            # Find columns with highest missing rates
            missing_rates = data.isnull().sum() / len(data)
            high_missing = missing_rates[missing_rates > 0.1].sort_values(ascending=False)
            if not high_missing.empty:
                recommendations.append(f"Columns with >10% missing: {high_missing.index.tolist()}")
            
            return DataValidationResult(
                test_name="completeness_validation",
                status=status,
                score=score,
                threshold=self.completeness_threshold,
                details={
                    'total_cells': total_cells,
                    'missing_cells': missing_cells,
                    'completeness_ratio': completeness_ratio,
                    'column_completeness': column_completeness,
                    'critical_failures': critical_failures
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Completeness validation failed: {str(e)}")
            return DataValidationResult(
                test_name="completeness_validation",
                status=ValidationStatus.FAILED,
                score=0.0,
                threshold=self.completeness_threshold,
                details={'error': str(e)},
                recommendations=["Fix validation errors and retry"]
            )

class SchemaValidator(BaseDataValidator):
    """Validator for data schema and types."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.expected_schema = config.get('expected_schema', {})
        self.required_columns = config.get('required_columns', [])
    
    def validate(self, data: pd.DataFrame) -> DataValidationResult:
        """Validate data schema."""
        try:
            schema_issues = []
            
            # Check required columns
            missing_columns = set(self.required_columns) - set(data.columns)
            if missing_columns:
                schema_issues.append(f"Missing required columns: {missing_columns}")
            
            # Check extra columns
            extra_columns = set(data.columns) - set(self.expected_schema.keys())
            if extra_columns:
                schema_issues.append(f"Unexpected columns: {extra_columns}")
            
            # Check data types
            type_issues = []
            for column, expected_type in self.expected_schema.items():
                if column in data.columns:
                    actual_type = str(data[column].dtype)
                    if not self._is_compatible_type(actual_type, expected_type):
                        type_issues.append(f"{column}: expected {expected_type}, got {actual_type}")
            
            if type_issues:
                schema_issues.extend(type_issues)
            
            # Check for duplicate columns
            if len(data.columns) != len(set(data.columns)):
                duplicate_cols = [col for col in data.columns if list(data.columns).count(col) > 1]
                schema_issues.append(f"Duplicate columns: {list(set(duplicate_cols))}")
            
            # Calculate score
            total_checks = len(self.required_columns) + len(self.expected_schema) + 1  # +1 for duplicates
            failed_checks = len(schema_issues)
            score = 1.0 - (failed_checks / total_checks)
            
            # Determine status
            if missing_columns:
                status = ValidationStatus.FAILED
            elif schema_issues:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.PASSED
            
            # Generate recommendations
            recommendations = []
            if schema_issues:
                recommendations.extend(schema_issues)
            
            return DataValidationResult(
                test_name="schema_validation",
                status=status,
                score=score,
                threshold=1.0,
                details={
                    'expected_schema': self.expected_schema,
                    'actual_columns': list(data.columns),
                    'actual_types': {col: str(data[col].dtype) for col in data.columns},
                    'missing_columns': list(missing_columns),
                    'extra_columns': list(extra_columns),
                    'type_issues': type_issues
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Schema validation failed: {str(e)}")
            return DataValidationResult(
                test_name="schema_validation",
                status=ValidationStatus.FAILED,
                score=0.0,
                threshold=1.0,
                details={'error': str(e)},
                recommendations=["Fix validation errors and retry"]
            )
    
    def _is_compatible_type(self, actual_type: str, expected_type: str) -> bool:
        """Check if actual type is compatible with expected type."""
        type_compatibility = {
            'int': ['int64', 'int32', 'int16', 'int8'],
            'float': ['float64', 'float32', 'float16'],
            'string': ['object', 'string'],
            'datetime': ['datetime64', 'datetime64[ns]'],
            'bool': ['bool'],
            'category': ['category']
        }
        
        if expected_type in type_compatibility:
            return actual_type in type_compatibility[expected_type]
        
        return actual_type == expected_type

class DomainValidator(BaseDataValidator):
    """Validator for domain-specific constraints."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.domain_constraints = config.get('domain_constraints', {})
        self.physics_simulator = BatteryPhysicsSimulator()
    
    def validate(self, data: pd.DataFrame) -> DataValidationResult:
        """Validate domain-specific constraints."""
        try:
            constraint_violations = []
            
            # Battery-specific domain constraints
            battery_constraints = {
                'voltage': {'min': 2.0, 'max': 5.0, 'unit': 'V'},
                'current': {'min': -500.0, 'max': 500.0, 'unit': 'A'},
                'temperature': {'min': -40.0, 'max': 80.0, 'unit': '°C'},
                'soc': {'min': 0.0, 'max': 1.0, 'unit': 'fraction'},
                'soh': {'min': 0.0, 'max': 1.0, 'unit': 'fraction'},
                'capacity': {'min': 0.0, 'max': 1000.0, 'unit': 'Ah'},
                'internal_resistance': {'min': 0.0, 'max': 1.0, 'unit': 'Ohm'}
            }
            
            # Update with custom constraints
            battery_constraints.update(self.domain_constraints)
            
            # Check each constraint
            for column, constraints in battery_constraints.items():
                if column in data.columns:
                    violations = self._check_column_constraints(data[column], column, constraints)
                    constraint_violations.extend(violations)
            
            # Physics-based validation
            physics_violations = self._validate_physics_constraints(data)
            constraint_violations.extend(physics_violations)
            
            # Check temporal consistency
            if 'timestamp' in data.columns:
                temporal_violations = self._check_temporal_consistency(data)
                constraint_violations.extend(temporal_violations)
            
            # Calculate score
            total_values = len(data) * len([col for col in battery_constraints.keys() if col in data.columns])
            violation_count = sum(v.get('count', 1) for v in constraint_violations)
            score = 1.0 - (violation_count / max(total_values, 1))
            
            # Determine status
            critical_violations = [v for v in constraint_violations if v.get('severity', 'medium') == 'critical']
            if critical_violations:
                status = ValidationStatus.FAILED
            elif constraint_violations:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.PASSED
            
            # Generate recommendations
            recommendations = []
            for violation in constraint_violations:
                recommendations.append(f"{violation['type']}: {violation['message']}")
            
            return DataValidationResult(
                test_name="domain_validation",
                status=status,
                score=score,
                threshold=0.95,
                details={
                    'constraint_violations': constraint_violations,
                    'total_values_checked': total_values,
                    'violation_count': violation_count,
                    'physics_violations': physics_violations
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Domain validation failed: {str(e)}")
            return DataValidationResult(
                test_name="domain_validation",
                status=ValidationStatus.FAILED,
                score=0.0,
                threshold=0.95,
                details={'error': str(e)},
                recommendations=["Fix validation errors and retry"]
            )
    
    def _check_column_constraints(self, series: pd.Series, column_name: str, 
                                 constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check constraints for a specific column."""
        violations = []
        
        # Check min/max bounds
        if 'min' in constraints:
            min_violations = (series < constraints['min']).sum()
            if min_violations > 0:
                violations.append({
                    'type': 'range_violation',
                    'column': column_name,
                    'constraint': 'minimum',
                    'expected': constraints['min'],
                    'count': min_violations,
                    'percentage': (min_violations / len(series)) * 100,
                    'message': f"{column_name} has {min_violations} values below minimum {constraints['min']}"
                })
        
        if 'max' in constraints:
            max_violations = (series > constraints['max']).sum()
            if max_violations > 0:
                violations.append({
                    'type': 'range_violation',
                    'column': column_name,
                    'constraint': 'maximum',
                    'expected': constraints['max'],
                    'count': max_violations,
                    'percentage': (max_violations / len(series)) * 100,
                    'message': f"{column_name} has {max_violations} values above maximum {constraints['max']}"
                })
        
        # Check for outliers using IQR method
        if len(series.dropna()) > 10:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((series < (Q1 - 1.5 * IQR)) | (series > (Q3 + 1.5 * IQR))).sum()
            
            if outliers > len(series) * 0.05:  # More than 5% outliers
                violations.append({
                    'type': 'outlier_violation',
                    'column': column_name,
                    'count': outliers,
                    'percentage': (outliers / len(series)) * 100,
                    'message': f"{column_name} has {outliers} statistical outliers ({outliers/len(series)*100:.1f}%)"
                })
        
        return violations
    
    def _validate_physics_constraints(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate physics-based constraints."""
        violations = []
        
        # Check for physically impossible combinations
        if all(col in data.columns for col in ['voltage', 'current', 'temperature']):
            for i, row in data.iterrows():
                if not self.physics_simulator.is_physically_valid({
                    'voltage': row['voltage'],
                    'current': row['current'],
                    'temperature': row['temperature']
                }):
                    violations.append({
                        'type': 'physics_violation',
                        'row': i,
                        'message': f"Row {i} contains physically impossible values"
                    })
        
        # Check power consistency (P = V * I)
        if all(col in data.columns for col in ['voltage', 'current', 'power']):
            calculated_power = data['voltage'] * data['current']
            power_diff = np.abs(data['power'] - calculated_power)
            power_violations = (power_diff > 0.1 * np.abs(data['power'])).sum()  # 10% tolerance
            
            if power_violations > 0:
                violations.append({
                    'type': 'physics_violation',
                    'constraint': 'power_consistency',
                    'count': power_violations,
                    'message': f"Power calculation inconsistency in {power_violations} rows"
                })
        
        return violations
    
    def _check_temporal_consistency(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check temporal consistency in time-series data."""
        violations = []
        
        # Check for duplicate timestamps
        if 'timestamp' in data.columns:
            duplicate_timestamps = data['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                violations.append({
                    'type': 'temporal_violation',
                    'constraint': 'duplicate_timestamps',
                    'count': duplicate_timestamps,
                    'message': f"Found {duplicate_timestamps} duplicate timestamps"
                })
            
            # Check for non-monotonic timestamps
            if not data['timestamp'].is_monotonic_increasing:
                violations.append({
                    'type': 'temporal_violation',
                    'constraint': 'non_monotonic',
                    'message': "Timestamps are not in chronological order"
                })
            
            # Check for unrealistic time gaps
            if len(data) > 1:
                time_diffs = data['timestamp'].diff().dropna()
                large_gaps = (time_diffs > pd.Timedelta(hours=24)).sum()
                if large_gaps > 0:
                    violations.append({
                        'type': 'temporal_violation',
                        'constraint': 'large_time_gaps',
                        'count': large_gaps,
                        'message': f"Found {large_gaps} time gaps larger than 24 hours"
                    })
        
        return violations

class OutlierValidator(BaseDataValidator):
    """Validator for outlier detection."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.outlier_threshold = config.get('outlier_threshold', 0.05)
        self.outlier_methods = config.get('outlier_methods', ['iqr', 'isolation_forest'])
    
    def validate(self, data: pd.DataFrame) -> DataValidationResult:
        """Validate outliers in data."""
        try:
            numeric_columns = data.select_dtypes(include=[np.number]).columns
            outlier_results = {}
            
            for column in numeric_columns:
                column_outliers = self._detect_outliers(data[column], self.outlier_methods)
                outlier_results[column] = column_outliers
            
            # Calculate overall outlier statistics
            total_outliers = sum(result['count'] for result in outlier_results.values())
            total_values = len(data) * len(numeric_columns)
            outlier_rate = total_outliers / total_values
            
            # Calculate score
            score = self._calculate_score(outlier_rate, self.outlier_threshold, False)
            
            # Determine status
            if outlier_rate > self.outlier_threshold * 2:
                status = ValidationStatus.FAILED
            elif outlier_rate > self.outlier_threshold:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.PASSED
            
            # Generate recommendations
            recommendations = []
            high_outlier_columns = [col for col, result in outlier_results.items() 
                                  if result['percentage'] > self.outlier_threshold * 100]
            
            if high_outlier_columns:
                recommendations.append(f"Columns with high outlier rates: {high_outlier_columns}")
            
            if outlier_rate > self.outlier_threshold:
                recommendations.append("Consider outlier treatment methods (removal, capping, transformation)")
            
            return DataValidationResult(
                test_name="outlier_validation",
                status=status,
                score=score,
                threshold=self.outlier_threshold,
                details={
                    'outlier_results': outlier_results,
                    'total_outliers': total_outliers,
                    'total_values': total_values,
                    'outlier_rate': outlier_rate,
                    'high_outlier_columns': high_outlier_columns
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Outlier validation failed: {str(e)}")
            return DataValidationResult(
                test_name="outlier_validation",
                status=ValidationStatus.FAILED,
                score=0.0,
                threshold=self.outlier_threshold,
                details={'error': str(e)},
                recommendations=["Fix validation errors and retry"]
            )
    
    def _detect_outliers(self, series: pd.Series, methods: List[str]) -> Dict[str, Any]:
        """Detect outliers using specified methods."""
        clean_series = series.dropna()
        
        if len(clean_series) < 10:
            return {'count': 0, 'percentage': 0.0, 'method': 'insufficient_data'}
        
        outlier_indices = set()
        
        for method in methods:
            if method == 'iqr':
                Q1 = clean_series.quantile(0.25)
                Q3 = clean_series.quantile(0.75)
                IQR = Q3 - Q1
                outliers = clean_series[(clean_series < (Q1 - 1.5 * IQR)) | 
                                       (clean_series > (Q3 + 1.5 * IQR))].index
                outlier_indices.update(outliers)
            
            elif method == 'isolation_forest':
                if len(clean_series) >= 100:
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(clean_series.values.reshape(-1, 1))
                    outliers = clean_series[outlier_labels == -1].index
                    outlier_indices.update(outliers)
            
            elif method == 'zscore':
                z_scores = np.abs(stats.zscore(clean_series))
                outliers = clean_series[z_scores > 3].index
                outlier_indices.update(outliers)
        
        return {
            'count': len(outlier_indices),
            'percentage': (len(outlier_indices) / len(clean_series)) * 100,
            'method': ','.join(methods),
            'outlier_indices': list(outlier_indices)
        }

class ConsistencyValidator(BaseDataValidator):
    """Validator for data consistency."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.consistency_threshold = config.get('consistency_threshold', 0.95)
    
    def validate(self, data: pd.DataFrame) -> DataValidationResult:
        """Validate data consistency."""
        try:
            consistency_issues = []
            
            # Check for duplicate rows
            duplicate_rows = data.duplicated().sum()
            if duplicate_rows > 0:
                consistency_issues.append({
                    'type': 'duplicate_rows',
                    'count': duplicate_rows,
                    'percentage': (duplicate_rows / len(data)) * 100,
                    'message': f"Found {duplicate_rows} duplicate rows"
                })
            
            # Check for inconsistent data types within columns
            type_inconsistencies = self._check_type_consistency(data)
            consistency_issues.extend(type_inconsistencies)
            
            # Check for logical inconsistencies
            logical_inconsistencies = self._check_logical_consistency(data)
            consistency_issues.extend(logical_inconsistencies)
            
            # Check for format inconsistencies
            format_inconsistencies = self._check_format_consistency(data)
            consistency_issues.extend(format_inconsistencies)
            
            # Calculate score
            total_issues = sum(issue.get('count', 1) for issue in consistency_issues)
            consistency_score = 1.0 - (total_issues / len(data))
            
            # Determine status
            if consistency_score < 0.9:
                status = ValidationStatus.FAILED
            elif consistency_score < self.consistency_threshold:
                status = ValidationStatus.WARNING
            else:
                status = ValidationStatus.PASSED
            
            # Generate recommendations
            recommendations = []
            for issue in consistency_issues:
                recommendations.append(issue['message'])
            
            return DataValidationResult(
                test_name="consistency_validation",
                status=status,
                score=consistency_score,
                threshold=self.consistency_threshold,
                details={
                    'consistency_issues': consistency_issues,
                    'total_issues': total_issues,
                    'consistency_score': consistency_score
                },
                recommendations=recommendations
            )
            
        except Exception as e:
            self.logger.error(f"Consistency validation failed: {str(e)}")
            return DataValidationResult(
                test_name="consistency_validation",
                status=ValidationStatus.FAILED,
                score=0.0,
                threshold=self.consistency_threshold,
                details={'error': str(e)},
                recommendations=["Fix validation errors and retry"]
            )
    
    def _check_type_consistency(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for type inconsistencies within columns."""
        issues = []
        
        for column in data.columns:
            if data[column].dtype == 'object':
                # Check for mixed numeric and string values
                numeric_count = pd.to_numeric(data[column], errors='coerce').notna().sum()
                string_count = len(data[column].dropna()) - numeric_count
                
                if numeric_count > 0 and string_count > 0:
                    issues.append({
                        'type': 'mixed_types',
                        'column': column,
                        'numeric_count': numeric_count,
                        'string_count': string_count,
                        'message': f"Column {column} contains mixed numeric and string values"
                    })
        
        return issues
    
    def _check_logical_consistency(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for logical inconsistencies in data."""
        issues = []
        
        # Battery-specific logical checks
        if 'soc' in data.columns and 'voltage' in data.columns:
            # SOC should generally correlate with voltage
            correlation = data['soc'].corr(data['voltage'])
            if abs(correlation) < 0.3:
                issues.append({
                    'type': 'logical_inconsistency',
                    'constraint': 'soc_voltage_correlation',
                    'correlation': correlation,
                    'message': f"Low correlation between SOC and voltage: {correlation:.3f}"
                })
        
        # Check for negative values where they shouldn't exist
        non_negative_columns = ['capacity', 'energy', 'soc', 'soh']
        for column in non_negative_columns:
            if column in data.columns:
                negative_count = (data[column] < 0).sum()
                if negative_count > 0:
                    issues.append({
                        'type': 'logical_inconsistency',
                        'column': column,
                        'count': negative_count,
                        'message': f"Column {column} has {negative_count} negative values"
                    })
        
        return issues
    
    def _check_format_consistency(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Check for format inconsistencies."""
        issues = []
        
        # Check date format consistency
        date_columns = data.select_dtypes(include=['object']).columns
        for column in date_columns:
            if 'date' in column.lower() or 'time' in column.lower():
                # Try to parse as datetime
                try:
                    pd.to_datetime(data[column], errors='coerce')
                except:
                    issues.append({
                        'type': 'format_inconsistency',
                        'column': column,
                        'message': f"Column {column} has inconsistent date formats"
                                })
        
        return issues

class DataValidator:
    """Main data validator orchestrating all validation tests."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize data validator."""
        self.config = ConfigParser.load_config(config_path) if config_path else {}
        self.logger = setup_logger(__name__)
        
        # Initialize validators
        self.validators = {
            'completeness': CompletenessValidator(self.config.get('completeness', {})),
            'schema': SchemaValidator(self.config.get('schema', {})),
            'domain': DomainValidator(self.config.get('domain', {})),
            'outliers': OutlierValidator(self.config.get('outliers', {})),
            'consistency': ConsistencyValidator(self.config.get('consistency', {}))
        }
    
    def validate_data(self, data: pd.DataFrame, dataset_name: str, 
                     dataset_type: str = "battery_telemetry",
                     validation_level: DataValidationLevel = DataValidationLevel.STANDARD) -> DataValidationReport:
        """
        Validate data comprehensively.
        
        Args:
            data: DataFrame to validate
            dataset_name: Name of the dataset
            dataset_type: Type of dataset
            validation_level: Level of validation to perform
            
        Returns:
            DataValidationReport: Comprehensive validation report
        """
        self.logger.info(f"Starting data validation for {dataset_name} ({dataset_type})")
        
        # Generate data summary
        data_summary = self._generate_data_summary(data)
        
        # Run validation tests based on level
        validation_results = {}
        
        if validation_level in [DataValidationLevel.BASIC, DataValidationLevel.STANDARD, DataValidationLevel.COMPREHENSIVE]:
            # Basic validations
            validation_results['completeness'] = self.validators['completeness'].validate(data)
            validation_results['schema'] = self.validators['schema'].validate(data, dataset_type)
            validation_results['domain'] = self.validators['domain'].validate(data, dataset_type)
        
        if validation_level in [DataValidationLevel.STANDARD, DataValidationLevel.COMPREHENSIVE]:
            # Standard validations
            validation_results['outliers'] = self.validators['outliers'].validate(data)
            validation_results['consistency'] = self.validators['consistency'].validate(data)
        
        if validation_level == DataValidationLevel.COMPREHENSIVE:
            # Comprehensive validations
            validation_results['battery_specific'] = self._validate_battery_specific(data, dataset_type)
            validation_results['temporal'] = self._validate_temporal_patterns(data)
            validation_results['physics'] = self._validate_physics_constraints(data)
        
        # Aggregate results
        overall_status = self._determine_overall_status(validation_results)
        
        # Create validation report
        report = DataValidationReport(
            dataset_name=dataset_name,
            dataset_type=dataset_type,
            validation_timestamp=datetime.now().isoformat(),
            validation_level=validation_level,
            data_summary=data_summary,
            validation_results=validation_results,
            overall_status=overall_status,
            recommendations=self._generate_recommendations(validation_results)
        )
        
        self.logger.info(f"Data validation completed for {dataset_name}. Status: {overall_status}")
        return report
    
    def _generate_data_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive data summary."""
        summary = {
            'shape': data.shape,
            'columns': list(data.columns),
            'missing_values': data.isnull().sum().to_dict(),
            'data_types': data.dtypes.to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'duplicates': data.duplicated().sum(),
            'unique_values': {col: data[col].nunique() for col in data.columns}
        }
        
        # Add statistical summary for numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            summary['statistics'] = data[numeric_cols].describe().to_dict()
        
        return summary
    
    def _validate_battery_specific(self, data: pd.DataFrame, dataset_type: str) -> Dict[str, Any]:
        """Validate battery-specific requirements."""
        issues = []
        
        # Battery-specific validations based on dataset type
        if dataset_type == "battery_telemetry":
            # Validate battery telemetry specific requirements
            issues.extend(self._validate_telemetry_specific(data))
        
        elif dataset_type == "degradation_curves":
            # Validate degradation curve specific requirements
            issues.extend(self._validate_degradation_specific(data))
        
        elif dataset_type == "fleet_patterns":
            # Validate fleet pattern specific requirements
            issues.extend(self._validate_fleet_specific(data))
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    def _validate_telemetry_specific(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate telemetry-specific requirements."""
        issues = []
        
        # Check for required telemetry columns
        required_cols = ['timestamp', 'battery_id', 'voltage', 'current', 'temperature', 'soc']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            issues.append({
                'type': 'missing_columns',
                'severity': 'error',
                'message': f"Missing required telemetry columns: {missing_cols}",
                'affected_columns': missing_cols
            })
        
        # Validate timestamp ordering
        if 'timestamp' in data.columns:
            if not data['timestamp'].is_monotonic_increasing:
                issues.append({
                    'type': 'timestamp_ordering',
                    'severity': 'warning',
                    'message': "Timestamps are not in ascending order",
                    'affected_columns': ['timestamp']
                })
        
        # Check for reasonable sampling rates
        if 'timestamp' in data.columns and len(data) > 1:
            time_diffs = pd.to_datetime(data['timestamp']).diff().dt.total_seconds()
            median_interval = time_diffs.median()
            
            if median_interval < 0.1:  # Too frequent (< 100ms)
                issues.append({
                    'type': 'sampling_rate',
                    'severity': 'warning',
                    'message': f"Very high sampling rate detected: {median_interval:.3f}s",
                    'affected_columns': ['timestamp']
                })
            elif median_interval > 3600:  # Too sparse (> 1 hour)
                issues.append({
                    'type': 'sampling_rate',
                    'severity': 'warning',
                    'message': f"Very low sampling rate detected: {median_interval:.1f}s",
                    'affected_columns': ['timestamp']
                })
        
        return issues
    
    def _validate_degradation_specific(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate degradation curve specific requirements."""
        issues = []
        
        # Check for required degradation columns
        required_cols = ['cycle_count', 'capacity', 'soh', 'battery_id']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            issues.append({
                'type': 'missing_columns',
                'severity': 'error',
                'message': f"Missing required degradation columns: {missing_cols}",
                'affected_columns': missing_cols
            })
        
        # Validate degradation trends
        if 'soh' in data.columns and 'cycle_count' in data.columns:
            for battery_id in data['battery_id'].unique():
                battery_data = data[data['battery_id'] == battery_id].sort_values('cycle_count')
                
                # Check if SoH is generally decreasing
                soh_trend = battery_data['soh'].diff().mean()
                if soh_trend > 0:
                    issues.append({
                        'type': 'degradation_trend',
                        'severity': 'warning',
                        'message': f"SoH appears to be increasing for battery {battery_id}",
                        'affected_columns': ['soh', 'battery_id']
                    })
        
        return issues
    
    def _validate_fleet_specific(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """Validate fleet pattern specific requirements."""
        issues = []
        
        # Check for required fleet columns
        required_cols = ['vehicle_id', 'route_id', 'usage_pattern', 'distance_km']
        missing_cols = [col for col in required_cols if col not in data.columns]
        
        if missing_cols:
            issues.append({
                'type': 'missing_columns',
                'severity': 'error',
                'message': f"Missing required fleet columns: {missing_cols}",
                'affected_columns': missing_cols
            })
        
        # Validate usage patterns
        if 'usage_pattern' in data.columns:
            valid_patterns = ['urban', 'highway', 'mixed', 'commercial', 'heavy_duty']
            invalid_patterns = set(data['usage_pattern'].dropna().unique()) - set(valid_patterns)
            
            if invalid_patterns:
                issues.append({
                    'type': 'invalid_usage_patterns',
                    'severity': 'warning',
                    'message': f"Invalid usage patterns detected: {invalid_patterns}",
                    'affected_columns': ['usage_pattern']
                })
        
        return issues
    
    def _validate_temporal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal patterns in data."""
        issues = []
        
        if 'timestamp' in data.columns:
            # Convert to datetime
            timestamps = pd.to_datetime(data['timestamp'])
            
            # Check for gaps in time series
            if len(timestamps) > 1:
                time_diffs = timestamps.diff().dt.total_seconds()
                median_interval = time_diffs.median()
                
                # Find large gaps (> 10x median interval)
                large_gaps = time_diffs > (median_interval * 10)
                if large_gaps.sum() > 0:
                    issues.append({
                        'type': 'temporal_gaps',
                        'severity': 'warning',
                        'message': f"Found {large_gaps.sum()} large gaps in time series",
                        'affected_columns': ['timestamp']
                    })
            
            # Check for duplicate timestamps
            duplicate_timestamps = timestamps.duplicated().sum()
            if duplicate_timestamps > 0:
                issues.append({
                    'type': 'duplicate_timestamps',
                    'severity': 'error',
                    'message': f"Found {duplicate_timestamps} duplicate timestamps",
                    'affected_columns': ['timestamp']
                })
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    def _validate_physics_constraints(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Validate physics-based constraints."""
        issues = []
        
        # Voltage constraints
        if 'voltage' in data.columns:
            voltage_violations = ((data['voltage'] < 0) | (data['voltage'] > 5)).sum()
            if voltage_violations > 0:
                issues.append({
                    'type': 'voltage_physics_violation',
                    'severity': 'error',
                    'message': f"Found {voltage_violations} voltage values outside physical range [0, 5]V",
                    'affected_columns': ['voltage']
                })
        
        # Current constraints
        if 'current' in data.columns:
            current_violations = (abs(data['current']) > 1000).sum()
            if current_violations > 0:
                issues.append({
                    'type': 'current_physics_violation',
                    'severity': 'error',
                    'message': f"Found {current_violations} current values outside physical range [-1000, 1000]A",
                    'affected_columns': ['current']
                })
        
        # Temperature constraints
        if 'temperature' in data.columns:
            temp_violations = ((data['temperature'] < -50) | (data['temperature'] > 100)).sum()
            if temp_violations > 0:
                issues.append({
                    'type': 'temperature_physics_violation',
                    'severity': 'error',
                    'message': f"Found {temp_violations} temperature values outside physical range [-50, 100]°C",
                    'affected_columns': ['temperature']
                })
        
        # State of Charge constraints
        if 'soc' in data.columns:
            soc_violations = ((data['soc'] < 0) | (data['soc'] > 1)).sum()
            if soc_violations > 0:
                issues.append({
                    'type': 'soc_physics_violation',
                    'severity': 'error',
                    'message': f"Found {soc_violations} SoC values outside physical range [0, 1]",
                    'affected_columns': ['soc']
                })
        
        # State of Health constraints
        if 'soh' in data.columns:
            soh_violations = ((data['soh'] < 0) | (data['soh'] > 1)).sum()
            if soh_violations > 0:
                issues.append({
                    'type': 'soh_physics_violation',
                    'severity': 'error',
                    'message': f"Found {soh_violations} SoH values outside physical range [0, 1]",
                    'affected_columns': ['soh']
                })
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    def _determine_overall_status(self, validation_results: Dict[str, Dict[str, Any]]) -> str:
        """Determine overall validation status."""
        has_errors = False
        has_warnings = False
        
        for validator_name, results in validation_results.items():
            if 'issues' in results:
                for issue in results['issues']:
                    if issue['severity'] == 'error':
                        has_errors = True
                    elif issue['severity'] == 'warning':
                        has_warnings = True
        
        if has_errors:
            return 'failed'
        elif has_warnings:
            return 'warning'
        else:
            return 'passed'
    
    def _generate_recommendations(self, validation_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for validator_name, results in validation_results.items():
            if 'issues' in results:
                for issue in results['issues']:
                    if issue['type'] == 'missing_data':
                        recommendations.append(f"Consider data imputation for missing values in {issue['affected_columns']}")
                    elif issue['type'] == 'outliers':
                        recommendations.append(f"Review outlier detection method for {issue['affected_columns']}")
                    elif issue['type'] == 'physics_violation':
                        recommendations.append(f"Implement data cleaning for physics violations in {issue['affected_columns']}")
                    elif issue['type'] == 'timestamp_ordering':
                        recommendations.append("Sort data by timestamp before processing")
                    elif issue['type'] == 'temporal_gaps':
                        recommendations.append("Consider interpolation for temporal gaps")
                    elif issue['type'] == 'duplicate_timestamps':
                        recommendations.append("Remove or aggregate duplicate timestamp entries")
        
        # Add general recommendations
        if len(recommendations) == 0:
            recommendations.append("Data quality appears good. Consider periodic validation.")
        
        return list(set(recommendations))  # Remove duplicates
    
    def validate_streaming_data(self, data: pd.DataFrame, 
                              reference_stats: Dict[str, Any]) -> Dict[str, Any]:
        """Validate streaming data against reference statistics."""
        issues = []
        
        # Check data distribution drift
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_stats:
                current_mean = data[col].mean()
                reference_mean = reference_stats[col]['mean']
                
                # Check for significant drift (> 20% change)
                if abs(current_mean - reference_mean) / reference_mean > 0.2:
                    issues.append({
                        'type': 'distribution_drift',
                        'severity': 'warning',
                        'message': f"Significant drift detected in {col}: {current_mean:.3f} vs {reference_mean:.3f}",
                        'affected_columns': [col]
                    })
        
        return {
            'passed': len(issues) == 0,
            'issues': issues,
            'total_issues': len(issues)
        }
    
    def export_validation_report(self, report: DataValidationReport, 
                                output_path: str, format: str = 'json'):
        """Export validation report to file."""
        if format == 'json':
            with open(output_path, 'w') as f:
                json.dump(report.__dict__, f, indent=2, default=str)
        elif format == 'html':
            self._export_html_report(report, output_path)
        elif format == 'csv':
            self._export_csv_report(report, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Validation report exported to {output_path}")
    
    def _export_html_report(self, report: DataValidationReport, output_path: str):
        """Export validation report as HTML."""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>BatteryMind Data Validation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; }}
                .section {{ margin: 20px 0; }}
                .error {{ color: red; font-weight: bold; }}
                .warning {{ color: orange; font-weight: bold; }}
                .success {{ color: green; font-weight: bold; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>BatteryMind Data Validation Report</h1>
                <p>Dataset: {report.dataset_name}</p>
                <p>Type: {report.dataset_type}</p>
                <p>Validation Time: {report.validation_timestamp}</p>
                <p>Status: <span class="{report.overall_status}">{report.overall_status.upper()}</span></p>
            </div>
            
            <div class="section">
                <h2>Data Summary</h2>
                <p>Shape: {report.data_summary['shape']}</p>
                <p>Columns: {len(report.data_summary['columns'])}</p>
                <p>Missing Values: {sum(report.data_summary['missing_values'].values())}</p>
                <p>Duplicates: {report.data_summary['duplicates']}</p>
            </div>
            
            <div class="section">
                <h2>Validation Results</h2>
                <!-- Add detailed validation results here -->
            </div>
            
            <div class="section">
                <h2>Recommendations</h2>
                <ul>
                    {''.join([f'<li>{rec}</li>' for rec in report.recommendations])}
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(output_path, 'w') as f:
            f.write(html_content)
    
    def _export_csv_report(self, report: DataValidationReport, output_path: str):
        """Export validation report summary as CSV."""
        # Create summary data
        summary_data = []
        
        for validator_name, results in report.validation_results.items():
            if 'issues' in results:
                for issue in results['issues']:
                    summary_data.append({
                        'validator': validator_name,
                        'issue_type': issue['type'],
                        'severity': issue['severity'],
                        'message': issue['message'],
                        'affected_columns': ', '.join(issue['affected_columns'])
                    })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(summary_data)
        df.to_csv(output_path, index=False)

# Factory function for creating validators
def create_data_validator(config_path: Optional[str] = None) -> DataValidator:
    """Create a data validator instance."""
    return DataValidator(config_path)

# Utility functions for common validation tasks
def validate_battery_telemetry(data: pd.DataFrame) -> DataValidationReport:
    """Quick validation for battery telemetry data."""
    validator = create_data_validator()
    return validator.validate_data(data, "telemetry", "battery_telemetry")

def validate_degradation_data(data: pd.DataFrame) -> DataValidationReport:
    """Quick validation for degradation data."""
    validator = create_data_validator()
    return validator.validate_data(data, "degradation", "degradation_curves")

def validate_fleet_data(data: pd.DataFrame) -> DataValidationReport:
    """Quick validation for fleet data."""
    validator = create_data_validator()
    return validator.validate_data(data, "fleet", "fleet_patterns")

# Example usage and testing
if __name__ == "__main__":
    # Example usage
    import pandas as pd
    import numpy as np
    
    # Create sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=1000, freq='1min'),
        'battery_id': ['BAT001'] * 1000,
        'voltage': np.random.normal(3.7, 0.2, 1000),
        'current': np.random.normal(0, 10, 1000),
        'temperature': np.random.normal(25, 5, 1000),
        'soc': np.random.uniform(0.2, 0.9, 1000),
        'soh': np.random.uniform(0.8, 1.0, 1000)
    })
    
    # Validate data
    validator = create_data_validator()
    report = validator.validate_data(sample_data, "test_data", "battery_telemetry")
    
    # Print results
    print(f"Validation Status: {report.overall_status}")
    print(f"Total Issues: {sum(len(r.get('issues', [])) for r in report.validation_results.values())}")
    
    # Export report
    validator.export_validation_report(report, "validation_report.json")
    
    print("Data validation completed successfully!")
