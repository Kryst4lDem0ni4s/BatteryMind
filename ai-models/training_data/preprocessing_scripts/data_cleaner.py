"""
BatteryMind - Data Cleaner

Comprehensive data cleaning and validation module for battery sensor data.
Handles missing data, outlier detection, data validation, and quality assessment.

Features:
- Multi-modal sensor data validation
- Advanced outlier detection algorithms
- Missing data imputation strategies
- Data quality metrics and reporting
- Automated data repair capabilities
- Cross-battery consistency checks

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.impute import KNNImputer
from scipy import stats
from scipy.interpolate import interp1d
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CleaningConfiguration:
    """
    Configuration for data cleaning operations.
    
    Attributes:
        # Outlier detection
        remove_outliers (bool): Whether to remove outliers
        outlier_method (str): Method for outlier detection
        outlier_threshold (float): Threshold for outlier detection
        outlier_contamination (float): Expected contamination ratio
        
        # Missing data handling
        handle_missing (bool): Whether to handle missing data
        missing_strategy (str): Strategy for missing data imputation
        max_missing_ratio (float): Maximum allowed missing data ratio
        interpolation_method (str): Method for interpolation
        
        # Data validation
        validate_ranges (bool): Whether to validate data ranges
        voltage_range (Tuple[float, float]): Valid voltage range
        current_range (Tuple[float, float]): Valid current range
        temperature_range (Tuple[float, float]): Valid temperature range
        soc_range (Tuple[float, float]): Valid SoC range
        
        # Quality thresholds
        min_sequence_length (int): Minimum sequence length
        max_gap_duration (float): Maximum gap duration in seconds
        sampling_rate_tolerance (float): Tolerance for sampling rate variation
        
        # Advanced options
        cross_battery_validation (bool): Validate across batteries
        temporal_consistency_check (bool): Check temporal consistency
        physics_based_validation (bool): Apply physics-based validation
    """
    # Outlier detection
    remove_outliers: bool = True
    outlier_method: str = "isolation_forest"  # "isolation_forest", "z_score", "iqr", "lof"
    outlier_threshold: float = 0.1
    outlier_contamination: float = 0.05
    
    # Missing data handling
    handle_missing: bool = True
    missing_strategy: str = "interpolation"  # "interpolation", "knn", "forward_fill", "drop"
    max_missing_ratio: float = 0.05
    interpolation_method: str = "linear"  # "linear", "cubic", "nearest"
    
    # Data validation
    validate_ranges: bool = True
    voltage_range: Tuple[float, float] = (2.0, 4.5)
    current_range: Tuple[float, float] = (-300.0, 300.0)
    temperature_range: Tuple[float, float] = (-50.0, 100.0)
    soc_range: Tuple[float, float] = (0.0, 1.0)
    
    # Quality thresholds
    min_sequence_length: int = 50
    max_gap_duration: float = 300.0  # 5 minutes
    sampling_rate_tolerance: float = 0.1
    
    # Advanced options
    cross_battery_validation: bool = True
    temporal_consistency_check: bool = True
    physics_based_validation: bool = True

@dataclass
class DataQualityMetrics:
    """
    Comprehensive data quality metrics.
    
    Attributes:
        total_samples (int): Total number of samples
        valid_samples (int): Number of valid samples
        missing_samples (int): Number of missing samples
        outlier_samples (int): Number of outlier samples
        out_of_range_samples (int): Number of out-of-range samples
        duplicate_samples (int): Number of duplicate samples
        quality_score (float): Overall quality score (0-1)
        completeness (float): Data completeness ratio
        consistency (float): Data consistency score
        accuracy (float): Data accuracy score
        issues (List[str]): List of identified issues
        recommendations (List[str]): Recommendations for improvement
    """
    total_samples: int = 0
    valid_samples: int = 0
    missing_samples: int = 0
    outlier_samples: int = 0
    out_of_range_samples: int = 0
    duplicate_samples: int = 0
    quality_score: float = 0.0
    completeness: float = 0.0
    consistency: float = 0.0
    accuracy: float = 0.0
    issues: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

class OutlierDetector:
    """
    Advanced outlier detection for battery sensor data.
    """
    
    def __init__(self, method: str = "isolation_forest", **kwargs):
        self.method = method
        self.kwargs = kwargs
        self.detector = None
        self._setup_detector()
    
    def _setup_detector(self):
        """Setup the outlier detection algorithm."""
        if self.method == "isolation_forest":
            self.detector = IsolationForest(
                contamination=self.kwargs.get("contamination", 0.05),
                random_state=42,
                n_estimators=100
            )
        elif self.method == "z_score":
            self.threshold = self.kwargs.get("threshold", 3.0)
        elif self.method == "iqr":
            self.iqr_factor = self.kwargs.get("iqr_factor", 1.5)
        elif self.method == "lof":
            from sklearn.neighbors import LocalOutlierFactor
            self.detector = LocalOutlierFactor(
                contamination=self.kwargs.get("contamination", 0.05),
                n_neighbors=20
            )
    
    def detect_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """
        Detect outliers in the data.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            np.ndarray: Boolean array indicating outliers
        """
        if self.method == "isolation_forest":
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) == 0:
                return np.zeros(len(data), dtype=bool)
            
            outlier_labels = self.detector.fit_predict(numeric_data.fillna(0))
            return outlier_labels == -1
        
        elif self.method == "z_score":
            return self._detect_zscore_outliers(data)
        
        elif self.method == "iqr":
            return self._detect_iqr_outliers(data)
        
        elif self.method == "lof":
            numeric_data = data.select_dtypes(include=[np.number])
            if len(numeric_data.columns) == 0:
                return np.zeros(len(data), dtype=bool)
            
            outlier_labels = self.detector.fit_predict(numeric_data.fillna(0))
            return outlier_labels == -1
        
        else:
            raise ValueError(f"Unknown outlier detection method: {self.method}")
    
    def _detect_zscore_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """Detect outliers using Z-score method."""
        numeric_data = data.select_dtypes(include=[np.number])
        outliers = np.zeros(len(data), dtype=bool)
        
        for column in numeric_data.columns:
            z_scores = np.abs(stats.zscore(numeric_data[column].fillna(0)))
            column_outliers = z_scores > self.threshold
            outliers |= column_outliers
        
        return outliers
    
    def _detect_iqr_outliers(self, data: pd.DataFrame) -> np.ndarray:
        """Detect outliers using IQR method."""
        numeric_data = data.select_dtypes(include=[np.number])
        outliers = np.zeros(len(data), dtype=bool)
        
        for column in numeric_data.columns:
            Q1 = numeric_data[column].quantile(0.25)
            Q3 = numeric_data[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - self.iqr_factor * IQR
            upper_bound = Q3 + self.iqr_factor * IQR
            
            column_outliers = ((numeric_data[column] < lower_bound) | 
                             (numeric_data[column] > upper_bound))
            outliers |= column_outliers.fillna(False)
        
        return outliers

class MissingDataHandler:
    """
    Advanced missing data handling for battery sensor data.
    """
    
    def __init__(self, strategy: str = "interpolation", **kwargs):
        self.strategy = strategy
        self.kwargs = kwargs
        self.imputer = None
        self._setup_imputer()
    
    def _setup_imputer(self):
        """Setup the missing data imputation algorithm."""
        if self.strategy == "knn":
            self.imputer = KNNImputer(
                n_neighbors=self.kwargs.get("n_neighbors", 5),
                weights=self.kwargs.get("weights", "uniform")
            )
    
    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in the dataset.
        
        Args:
            data (pd.DataFrame): Input data with missing values
            
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        if self.strategy == "interpolation":
            return self._interpolate_missing(data)
        elif self.strategy == "knn":
            return self._knn_impute(data)
        elif self.strategy == "forward_fill":
            return data.fillna(method='ffill')
        elif self.strategy == "backward_fill":
            return data.fillna(method='bfill')
        elif self.strategy == "drop":
            return data.dropna()
        else:
            raise ValueError(f"Unknown missing data strategy: {self.strategy}")
    
    def _interpolate_missing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Interpolate missing values using specified method."""
        method = self.kwargs.get("method", "linear")
        
        # Handle time-series interpolation
        if "timestamp" in data.columns:
            data = data.set_index("timestamp")
            data = data.interpolate(method=method, limit_direction="both")
            data = data.reset_index()
        else:
            data = data.interpolate(method=method, limit_direction="both")
        
        return data
    
    def _knn_impute(self, data: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using KNN."""
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            data_imputed = data.copy()
            data_imputed[numeric_columns] = self.imputer.fit_transform(data[numeric_columns])
            return data_imputed
        
        return data

class DataValidator:
    """
    Comprehensive data validation for battery sensor data.
    """
    
    def __init__(self, config: CleaningConfiguration):
        self.config = config
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup validation rules based on configuration."""
        return {
            "voltage": {
                "min": self.config.voltage_range[0],
                "max": self.config.voltage_range[1],
                "type": "numeric"
            },
            "current": {
                "min": self.config.current_range[0],
                "max": self.config.current_range[1],
                "type": "numeric"
            },
            "temperature": {
                "min": self.config.temperature_range[0],
                "max": self.config.temperature_range[1],
                "type": "numeric"
            },
            "soc": {
                "min": self.config.soc_range[0],
                "max": self.config.soc_range[1],
                "type": "numeric"
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Validate data against defined rules.
        
        Args:
            data (pd.DataFrame): Input data to validate
            
        Returns:
            Tuple[pd.DataFrame, List[str]]: Validated data and list of issues
        """
        validated_data = data.copy()
        issues = []
        
        # Range validation
        if self.config.validate_ranges:
            validated_data, range_issues = self._validate_ranges(validated_data)
            issues.extend(range_issues)
        
        # Temporal consistency validation
        if self.config.temporal_consistency_check:
            temporal_issues = self._validate_temporal_consistency(validated_data)
            issues.extend(temporal_issues)
        
        # Physics-based validation
        if self.config.physics_based_validation:
            physics_issues = self._validate_physics_constraints(validated_data)
            issues.extend(physics_issues)
        
        return validated_data, issues
    
    def _validate_ranges(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Validate data ranges."""
        issues = []
        validated_data = data.copy()
        
        for column, rules in self.validation_rules.items():
            if column in data.columns:
                out_of_range = ((data[column] < rules["min"]) | 
                               (data[column] > rules["max"]))
                
                if out_of_range.any():
                    count = out_of_range.sum()
                    issues.append(f"{column}: {count} values out of range [{rules['min']}, {rules['max']}]")
                    
                    # Clip values to valid range
                    validated_data[column] = data[column].clip(rules["min"], rules["max"])
        
        return validated_data, issues
    
    def _validate_temporal_consistency(self, data: pd.DataFrame) -> List[str]:
        """Validate temporal consistency."""
        issues = []
        
        if "timestamp" in data.columns:
            # Check for duplicate timestamps
            duplicates = data["timestamp"].duplicated().sum()
            if duplicates > 0:
                issues.append(f"Found {duplicates} duplicate timestamps")
            
            # Check for temporal ordering
            if not data["timestamp"].is_monotonic_increasing:
                issues.append("Timestamps are not in chronological order")
            
            # Check sampling rate consistency
            if len(data) > 1:
                time_diffs = data["timestamp"].diff().dropna()
                median_diff = time_diffs.median()
                
                # Check for large gaps
                large_gaps = (time_diffs > median_diff * (1 + self.config.sampling_rate_tolerance)).sum()
                if large_gaps > 0:
                    issues.append(f"Found {large_gaps} large time gaps in data")
        
        return issues
    
    def _validate_physics_constraints(self, data: pd.DataFrame) -> List[str]:
        """Validate physics-based constraints."""
        issues = []
        
        # Power calculation consistency
        if all(col in data.columns for col in ["voltage", "current"]):
            calculated_power = data["voltage"] * data["current"]
            if "power" in data.columns:
                power_diff = np.abs(calculated_power - data["power"])
                inconsistent = (power_diff > 0.1 * np.abs(calculated_power)).sum()
                if inconsistent > 0:
                    issues.append(f"Power calculation inconsistency in {inconsistent} samples")
        
        # Energy conservation check
        if all(col in data.columns for col in ["soc", "current", "timestamp"]):
            # Check if SoC changes are consistent with current integration
            if len(data) > 1:
                soc_diff = data["soc"].diff()
                time_diff = data["timestamp"].diff().dt.total_seconds() / 3600  # Convert to hours
                
                # Simplified energy balance check
                expected_soc_change = data["current"] * time_diff / 100  # Assuming 100Ah capacity
                actual_vs_expected = np.abs(soc_diff - expected_soc_change).dropna()
                
                large_discrepancies = (actual_vs_expected > 0.05).sum()
                if large_discrepancies > 0:
                    issues.append(f"Energy balance discrepancy in {large_discrepancies} samples")
        
        return issues

class BatteryDataCleaner:
    """
    Main data cleaning class for battery sensor data.
    """
    
    def __init__(self, config: CleaningConfiguration = None):
        self.config = config or CleaningConfiguration()
        self.outlier_detector = OutlierDetector(
            method=self.config.outlier_method,
            contamination=self.config.outlier_contamination,
            threshold=self.config.outlier_threshold
        )
        self.missing_handler = MissingDataHandler(
            strategy=self.config.missing_strategy,
            method=self.config.interpolation_method
        )
        self.validator = DataValidator(self.config)
        
        # Statistics tracking
        self.cleaning_stats = {}
        
    def clean(self, data: Union[pd.DataFrame, Dict[str, pd.DataFrame]]) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Clean battery sensor data.
        
        Args:
            data: Input data (single DataFrame or dict of DataFrames)
            
        Returns:
            Cleaned data in the same format as input
        """
        if isinstance(data, dict):
            return self._clean_multiple_batteries(data)
        else:
            return self._clean_single_battery(data)
    
    def _clean_single_battery(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean data for a single battery."""
        logger.info(f"Cleaning battery data with {len(data)} samples")
        
        original_size = len(data)
        cleaned_data = data.copy()
        
        # Step 1: Basic validation and filtering
        cleaned_data = self._basic_filtering(cleaned_data)
        
        # Step 2: Handle missing data
        if self.config.handle_missing:
            cleaned_data = self.missing_handler.handle_missing_data(cleaned_data)
        
        # Step 3: Outlier detection and removal
        if self.config.remove_outliers:
            outliers = self.outlier_detector.detect_outliers(cleaned_data)
            cleaned_data = cleaned_data[~outliers]
            logger.info(f"Removed {outliers.sum()} outliers")
        
        # Step 4: Data validation
        if self.config.validate_ranges:
            cleaned_data, validation_issues = self.validator.validate_data(cleaned_data)
            if validation_issues:
                logger.warning(f"Validation issues: {validation_issues}")
        
        # Step 5: Final quality check
        if len(cleaned_data) < self.config.min_sequence_length:
            logger.warning(f"Cleaned data too short: {len(cleaned_data)} < {self.config.min_sequence_length}")
        
        # Update statistics
        self.cleaning_stats = {
            "original_samples": original_size,
            "cleaned_samples": len(cleaned_data),
            "removed_samples": original_size - len(cleaned_data),
            "removal_ratio": (original_size - len(cleaned_data)) / original_size
        }
        
        logger.info(f"Cleaning completed: {len(cleaned_data)} samples remaining")
        return cleaned_data
    
    def _clean_multiple_batteries(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Clean data for multiple batteries."""
        cleaned_data = {}
        
        for battery_id, battery_data in data.items():
            logger.info(f"Cleaning data for battery {battery_id}")
            cleaned_data[battery_id] = self._clean_single_battery(battery_data)
        
        # Cross-battery validation if enabled
        if self.config.cross_battery_validation:
            cleaned_data = self._cross_battery_validation(cleaned_data)
        
        return cleaned_data
    
    def _basic_filtering(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply basic filtering operations."""
        filtered_data = data.copy()
        
        # Remove completely empty rows
        filtered_data = filtered_data.dropna(how='all')
        
        # Remove duplicate rows
        duplicates_before = len(filtered_data)
        filtered_data = filtered_data.drop_duplicates()
        duplicates_removed = duplicates_before - len(filtered_data)
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        # Sort by timestamp if available
        if "timestamp" in filtered_data.columns:
            filtered_data = filtered_data.sort_values("timestamp")
        
        return filtered_data
    
    def _cross_battery_validation(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Perform cross-battery validation and consistency checks."""
        logger.info("Performing cross-battery validation")
        
        # Check for consistent column names
        all_columns = set()
        for battery_data in data.values():
            all_columns.update(battery_data.columns)
        
        # Ensure all batteries have the same columns
        validated_data = {}
        for battery_id, battery_data in data.items():
            missing_columns = all_columns - set(battery_data.columns)
            if missing_columns:
                logger.warning(f"Battery {battery_id} missing columns: {missing_columns}")
                # Add missing columns with NaN values
                for col in missing_columns:
                    battery_data[col] = np.nan
            
            validated_data[battery_id] = battery_data
        
        return validated_data
    
    def get_quality_metrics(self, data: pd.DataFrame) -> DataQualityMetrics:
        """
        Calculate comprehensive data quality metrics.
        
        Args:
            data (pd.DataFrame): Data to analyze
            
        Returns:
            DataQualityMetrics: Quality metrics
        """
        metrics = DataQualityMetrics()
        
        # Basic counts
        metrics.total_samples = len(data)
        metrics.missing_samples = data.isnull().sum().sum()
        metrics.duplicate_samples = data.duplicated().sum()
        
        # Outlier detection
        if metrics.total_samples > 0:
            outliers = self.outlier_detector.detect_outliers(data)
            metrics.outlier_samples = outliers.sum()
        
        # Range validation
        out_of_range_count = 0
        for column, rules in self.validator.validation_rules.items():
            if column in data.columns:
                out_of_range = ((data[column] < rules["min"]) | 
                               (data[column] > rules["max"]))
                out_of_range_count += out_of_range.sum()
        metrics.out_of_range_samples = out_of_range_count
        
        # Calculate derived metrics
        metrics.valid_samples = (metrics.total_samples - metrics.missing_samples - 
                               metrics.outlier_samples - metrics.out_of_range_samples)
        
        if metrics.total_samples > 0:
            metrics.completeness = 1.0 - (metrics.missing_samples / (metrics.total_samples * len(data.columns)))
            metrics.accuracy = metrics.valid_samples / metrics.total_samples
            metrics.consistency = 1.0 - (metrics.duplicate_samples / metrics.total_samples)
        
        # Overall quality score
        metrics.quality_score = (metrics.completeness + metrics.accuracy + metrics.consistency) / 3
        
        # Generate issues and recommendations
        if metrics.completeness < 0.95:
            metrics.issues.append("High missing data rate")
            metrics.recommendations.append("Improve data collection or imputation strategy")
        
        if metrics.accuracy < 0.9:
            metrics.issues.append("High out-of-range or outlier rate")
            metrics.recommendations.append("Review sensor calibration and data validation")
        
        if metrics.consistency < 0.95:
            metrics.issues.append("High duplicate rate")
            metrics.recommendations.append("Implement deduplication in data collection")
        
        return metrics

# Utility functions
def clean_battery_dataset(data: Union[pd.DataFrame, Dict[str, pd.DataFrame]], 
                         config: CleaningConfiguration = None) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Convenience function to clean battery dataset.
    
    Args:
        data: Input data
        config: Cleaning configuration
        
    Returns:
        Cleaned data
    """
    cleaner = BatteryDataCleaner(config)
    return cleaner.clean(data)

def validate_sensor_data(data: pd.DataFrame, 
                        config: CleaningConfiguration = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validate sensor data against defined constraints.
    
    Args:
        data: Input sensor data
        config: Validation configuration
        
    Returns:
        Tuple of validated data and issues list
    """
    validator = DataValidator(config or CleaningConfiguration())
    return validator.validate_data(data)

def detect_anomalies(data: pd.DataFrame, method: str = "isolation_forest", **kwargs) -> np.ndarray:
    """
    Detect anomalies in battery sensor data.
    
    Args:
        data: Input data
        method: Detection method
        **kwargs: Additional parameters
        
    Returns:
        Boolean array indicating anomalies
    """
    detector = OutlierDetector(method, **kwargs)
    return detector.detect_outliers(data)

def repair_missing_data(data: pd.DataFrame, strategy: str = "interpolation", **kwargs) -> pd.DataFrame:
    """
    Repair missing data in battery sensor data.
    
    Args:
        data: Input data with missing values
        strategy: Repair strategy
        **kwargs: Additional parameters
        
    Returns:
        Data with missing values repaired
    """
    handler = MissingDataHandler(strategy, **kwargs)
    return handler.handle_missing_data(data)
