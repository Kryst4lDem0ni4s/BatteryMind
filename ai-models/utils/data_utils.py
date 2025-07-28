"""
BatteryMind - Data Utilities
Comprehensive data processing, validation, and transformation utilities
specifically designed for battery management AI systems.

Features:
- Battery telemetry data processing and validation
- Fleet data aggregation and analysis
- Time series data preprocessing and feature engineering
- Real-time data streaming and buffering
- Data quality monitoring and anomaly detection
- Multi-modal sensor data fusion
- Privacy-preserving data transformations

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings

# Configure logging
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Types of data in the BatteryMind system."""
    BATTERY_TELEMETRY = "battery_telemetry"
    FLEET_DATA = "fleet_data"
    SENSOR_DATA = "sensor_data"
    PREDICTION_DATA = "prediction_data"
    AUTONOMOUS_DECISIONS = "autonomous_decisions"
    BLOCKCHAIN_DATA = "blockchain_data"
    CIRCULAR_ECONOMY = "circular_economy"

class DataQuality(Enum):
    """Data quality levels."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"

@dataclass
class DataValidationResult:
    """Result of data validation."""
    
    is_valid: bool = True
    quality_score: float = 1.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'is_valid': self.is_valid,
            'quality_score': self.quality_score,
            'errors': self.errors,
            'warnings': self.warnings,
            'anomalies': self.anomalies,
            'statistics': self.statistics
        }

class DataProcessor:
    """
    Core data processing engine for BatteryMind system.
    """
    
    def __init__(self, enable_caching: bool = True):
        self.enable_caching = enable_caching
        self.cache = {} if enable_caching else None
        self.processing_stats = {
            'records_processed': 0,
            'errors_encountered': 0,
            'processing_time_ms': 0,
            'cache_hits': 0
        }
        
        logger.info("Data Processor initialized")
    
    def process_battery_telemetry(self, 
                                 data: Union[pd.DataFrame, List[Dict[str, Any]]], 
                                 battery_id: str = None,
                                 validate: bool = True) -> Tuple[pd.DataFrame, DataValidationResult]:
        """
        Process battery telemetry data with validation and preprocessing.
        
        Args:
            data: Raw battery telemetry data
            battery_id: Optional battery identifier for filtering
            validate: Whether to perform data validation
            
        Returns:
            Tuple of processed DataFrame and validation result
        """
        try:
            start_time = datetime.now()
            
            # Convert to DataFrame if needed
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = data.copy()
            
            # Filter by battery ID if specified
            if battery_id and 'battery_id' in df.columns:
                df = df[df['battery_id'] == battery_id]
            
            # Initialize validation result
            validation_result = DataValidationResult()
            
            if validate:
                validation_result = self.validate_battery_data(df)
                if not validation_result.is_valid:
                    logger.warning(f"Battery data validation failed: {validation_result.errors}")
            
            # Preprocessing steps
            df = self._preprocess_battery_telemetry(df)
            
            # Update processing statistics
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            self.processing_stats['records_processed'] += len(df)
            self.processing_stats['processing_time_ms'] += processing_time
            
            logger.info(f"Processed {len(df)} battery telemetry records in {processing_time:.2f}ms")
            
            return df, validation_result
            
        except Exception as e:
            self.processing_stats['errors_encountered'] += 1
            logger.error(f"Error processing battery telemetry: {e}")
            raise
    
    def _preprocess_battery_telemetry(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess battery telemetry data.
        
        Args:
            df: Raw battery telemetry DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        try:
            # Ensure timestamp column is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
            
            # Handle missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if col in ['soh', 'soc', 'voltage', 'current', 'temperature']:
                    # Forward fill for battery metrics
                    df[col] = df[col].fillna(method='ffill').fillna(method='bfill')
            
            # Remove outliers using IQR method
            for col in ['voltage', 'current', 'temperature']:
                if col in df.columns:
                    df = self._remove_outliers_iqr(df, col)
            
            # Calculate derived metrics
            if 'voltage' in df.columns and 'current' in df.columns:
                df['power'] = df['voltage'] * df['current']
            
            if 'soh' in df.columns:
                df['health_category'] = pd.cut(df['soh'], 
                                             bins=[0, 70, 85, 95, 100], 
                                             labels=['Critical', 'Degraded', 'Good', 'Excellent'])
            
            # Add time-based features
            if 'timestamp' in df.columns:
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['month'] = df['timestamp'].dt.month
            
            return df
            
        except Exception as e:
            logger.error(f"Error in battery telemetry preprocessing: {e}")
            return df
    
    def validate_battery_data(self, df: pd.DataFrame) -> DataValidationResult:
        """
        Validate battery data quality and consistency.
        
        Args:
            df: Battery data DataFrame
            
        Returns:
            Validation result
        """
        result = DataValidationResult()
        
        try:
            # Check required columns
            required_columns = ['battery_id', 'timestamp', 'soh', 'soc', 'voltage', 'current', 'temperature']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                result.errors.append(f"Missing required columns: {missing_columns}")
                result.is_valid = False
            
            # Data type validation
            if 'timestamp' in df.columns:
                try:
                    pd.to_datetime(df['timestamp'])
                except:
                    result.errors.append("Invalid timestamp format")
                    result.is_valid = False
            
            # Range validation
            validation_rules = {
                'soh': (0, 100),
                'soc': (0, 100),
                'voltage': (2.0, 5.0),
                'current': (-500, 500),
                'temperature': (-50, 100)
            }
            
            for column, (min_val, max_val) in validation_rules.items():
                if column in df.columns:
                    out_of_range = ((df[column] < min_val) | (df[column] > max_val)).sum()
                    if out_of_range > 0:
                        result.warnings.append(f"Column {column}: {out_of_range} values out of range [{min_val}, {max_val}]")
                        if out_of_range > len(df) * 0.1:  # More than 10% out of range
                            result.is_valid = False
            
            # Missing data analysis
            missing_data = df.isnull().sum()
            if missing_data.sum() > 0:
                result.warnings.append(f"Missing data detected: {missing_data.to_dict()}")
                if missing_data.sum() > len(df) * 0.2:  # More than 20% missing
                    result.quality_score *= 0.8
            
            # Duplicate detection
            if 'timestamp' in df.columns and 'battery_id' in df.columns:
                duplicates = df.duplicated(subset=['timestamp', 'battery_id']).sum()
                if duplicates > 0:
                    result.warnings.append(f"Found {duplicates} duplicate records")
                    result.quality_score *= 0.9
            
            # Anomaly detection
            result.anomalies = self._detect_anomalies(df)
            if len(result.anomalies) > len(df) * 0.05:  # More than 5% anomalies
                result.quality_score *= 0.85
            
            # Calculate statistics
            result.statistics = self._calculate_data_statistics(df)
            
            # Overall quality score
            if result.is_valid:
                if len(result.warnings) == 0:
                    result.quality_score = min(result.quality_score, 1.0)
                else:
                    result.quality_score = min(result.quality_score, 0.9)
            else:
                result.quality_score = min(result.quality_score, 0.5)
            
            return result
            
        except Exception as e:
            logger.error(f"Error validating battery data: {e}")
            result.is_valid = False
            result.errors.append(f"Validation error: {str(e)}")
            return result
    
    def _remove_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Remove outliers using Interquartile Range method.
        
        Args:
            df: DataFrame
            column: Column to process
            
        Returns:
            DataFrame with outliers removed
        """
        try:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_count = ((df[column] < lower_bound) | (df[column] > upper_bound)).sum()
            if outliers_count > 0:
                logger.info(f"Removing {outliers_count} outliers from column {column}")
                df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
            
            return df
            
        except Exception as e:
            logger.error(f"Error removing outliers from {column}: {e}")
            return df
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies in battery data.
        
        Args:
            df: Battery data DataFrame
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        try:
            # Voltage anomalies
            if 'voltage' in df.columns:
                # Sudden voltage drops
                voltage_diff = df['voltage'].diff().abs()
                sudden_drops = voltage_diff > 0.5  # More than 0.5V sudden change
                for idx in df[sudden_drops].index:
                    anomalies.append({
                        'type': 'sudden_voltage_change',
                        'index': idx,
                        'value': df.loc[idx, 'voltage'],
                        'change': voltage_diff.loc[idx] if idx in voltage_diff.index else None
                    })
            
            # Temperature anomalies
            if 'temperature' in df.columns:
                # Extreme temperatures
                extreme_temp = (df['temperature'] > 60) | (df['temperature'] < -20)
                for idx in df[extreme_temp].index:
                    anomalies.append({
                        'type': 'extreme_temperature',
                        'index': idx,
                        'value': df.loc[idx, 'temperature']
                    })
            
            # SoH anomalies
            if 'soh' in df.columns:
                # Impossible SoH increases
                soh_diff = df['soh'].diff()
                impossible_increases = soh_diff > 5  # SoH shouldn't increase by more than 5%
                for idx in df[impossible_increases].index:
                    anomalies.append({
                        'type': 'impossible_soh_increase',
                        'index': idx,
                        'value': df.loc[idx, 'soh'],
                        'increase': soh_diff.loc[idx] if idx in soh_diff.index else None
                    })
            
            # Current anomalies
            if 'current' in df.columns:
                # Impossible current values (considering typical battery systems)
                extreme_current = df['current'].abs() > 200  # More than 200A
                for idx in df[extreme_current].index:
                    anomalies.append({
                        'type': 'extreme_current',
                        'index': idx,
                        'value': df.loc[idx, 'current']
                    })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _calculate_data_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive statistics for battery data.
        
        Args:
            df: Battery data DataFrame
            
        Returns:
            Dictionary of statistics
        """
        try:
            stats = {
                'total_records': len(df),
                'unique_batteries': df['battery_id'].nunique() if 'battery_id' in df.columns else 0,
                'time_range': {},
                'numeric_stats': {},
                'data_quality': {}
            }
            
            # Time range statistics
            if 'timestamp' in df.columns:
                stats['time_range'] = {
                    'start': df['timestamp'].min().isoformat() if not df['timestamp'].isna().all() else None,
                    'end': df['timestamp'].max().isoformat() if not df['timestamp'].isna().all() else None,
                    'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600 if not df['timestamp'].isna().all() else 0
                }
            
            # Numeric statistics
            numeric_columns = ['soh', 'soc', 'voltage', 'current', 'temperature']
            for col in numeric_columns:
                if col in df.columns:
                    stats['numeric_stats'][col] = {
                        'mean': float(df[col].mean()) if not df[col].isna().all() else None,
                        'std': float(df[col].std()) if not df[col].isna().all() else None,
                        'min': float(df[col].min()) if not df[col].isna().all() else None,
                        'max': float(df[col].max()) if not df[col].isna().all() else None,
                        'median': float(df[col].median()) if not df[col].isna().all() else None
                    }
            
            # Data quality statistics
            stats['data_quality'] = {
                'completeness': (1 - df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
                'missing_values': df.isnull().sum().to_dict(),
                'duplicate_records': df.duplicated().sum() if len(df) > 0 else 0
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return {}

class TimeSeriesProcessor:
    """
    Specialized processor for time series battery data.
    """
    
    def __init__(self, window_size: int = 10, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap
        
        logger.info(f"Time Series Processor initialized with window_size={window_size}")
    
    def create_sliding_windows(self, 
                              df: pd.DataFrame, 
                              features: List[str],
                              target: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sliding window sequences for time series modeling.
        
        Args:
            df: Time series DataFrame
            features: List of feature columns
            target: Target column for prediction
            
        Returns:
            Tuple of (X, y) arrays for model training
        """
        try:
            # Sort by timestamp if available
            if 'timestamp' in df.columns:
                df = df.sort_values('timestamp')
            
            # Extract feature data
            feature_data = df[features].values
            
            # Create sliding windows
            X, y = [], []
            step_size = max(1, int(self.window_size * (1 - self.overlap)))
            
            for i in range(0, len(feature_data) - self.window_size + 1, step_size):
                window = feature_data[i:i + self.window_size]
                X.append(window)
                
                if target and target in df.columns:
                    target_value = df[target].iloc[i + self.window_size - 1]
                    y.append(target_value)
            
            X = np.array(X)
            y = np.array(y) if target else None
            
            logger.info(f"Created {len(X)} sliding windows from {len(df)} records")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating sliding windows: {e}")
            return np.array([]), np.array([])
    
    def extract_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract temporal features from time series data.
        
        Args:
            df: Time series DataFrame with timestamp column
            
        Returns:
            DataFrame with additional temporal features
        """
        try:
            if 'timestamp' not in df.columns:
                logger.warning("No timestamp column found for temporal feature extraction")
                return df
            
            df = df.copy()
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Basic temporal features
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['month'] = df['timestamp'].dt.month
            df['quarter'] = df['timestamp'].dt.quarter
            df['year'] = df['timestamp'].dt.year
            
            # Cyclical encoding for temporal features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            
            # Time-based flags
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
            
            logger.info(f"Extracted temporal features for {len(df)} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting temporal features: {e}")
            return df
    
    def detect_seasonal_patterns(self, 
                                df: pd.DataFrame, 
                                value_column: str,
                                period: int = 24) -> Dict[str, Any]:
        """
        Detect seasonal patterns in time series data.
        
        Args:
            df: Time series DataFrame
            value_column: Column to analyze for patterns
            period: Expected period (e.g., 24 for daily patterns)
            
        Returns:
            Dictionary with pattern analysis results
        """
        try:
            if value_column not in df.columns:
                return {}
            
            from scipy import signal
            
            # Prepare data
            values = df[value_column].dropna().values
            
            if len(values) < period * 2:
                logger.warning(f"Insufficient data for seasonal analysis: {len(values)} < {period * 2}")
                return {}
            
            # Autocorrelation analysis
            autocorr = np.correlate(values, values, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            autocorr = autocorr / autocorr[0]  # Normalize
            
            # Find peaks in autocorrelation
            peaks, properties = signal.find_peaks(autocorr[1:period*3], height=0.1, distance=period//2)
            
            # Seasonal decomposition (if enough data)
            seasonal_strength = 0
            if len(values) > period * 4:
                try:
                    from statsmodels.tsa.seasonal import seasonal_decompose
                    decomposition = seasonal_decompose(values[:period*4], model='additive', period=period)
                    seasonal_var = np.var(decomposition.seasonal)
                    residual_var = np.var(decomposition.resid[~np.isnan(decomposition.resid)])
                    seasonal_strength = seasonal_var / (seasonal_var + residual_var) if (seasonal_var + residual_var) > 0 else 0
                except:
                    logger.warning("Could not perform seasonal decomposition")
            
            pattern_analysis = {
                'period': period,
                'seasonal_strength': float(seasonal_strength),
                'autocorr_peaks': peaks.tolist() if len(peaks) > 0 else [],
                'max_autocorr': float(np.max(autocorr[1:period*2])) if len(autocorr) > period*2 else 0,
                'trend_detected': bool(np.corrcoef(np.arange(len(values)), values)[0,1] > 0.1),
                'data_points': len(values)
            }
            
            logger.info(f"Seasonal pattern analysis completed for {value_column}")
            
            return pattern_analysis
            
        except Exception as e:
            logger.error(f"Error detecting seasonal patterns: {e}")
            return {}

class FeatureEngineer:
    """
    Advanced feature engineering for battery data.
    """
    
    def __init__(self):
        self.feature_transformers = {}
        
        logger.info("Feature Engineer initialized")
    
    def engineer_battery_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer advanced features for battery data.
        
        Args:
            df: Battery DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            df = df.copy()
            
            # Rolling statistics
            window_sizes = [5, 10, 20]
            rolling_columns = ['soh', 'soc', 'voltage', 'current', 'temperature']
            
            for col in rolling_columns:
                if col in df.columns:
                    for window in window_sizes:
                        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                        df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()
                        df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window).max()
                        df[f'{col}_rolling_min_{window}'] = df[col].rolling(window=window).min()
            
            # Lag features
            lag_periods = [1, 3, 6, 12]
            for col in rolling_columns:
                if col in df.columns:
                    for lag in lag_periods:
                        df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Rate of change features
            for col in rolling_columns:
                if col in df.columns:
                    df[f'{col}_rate_of_change'] = df[col].pct_change()
                    df[f'{col}_diff'] = df[col].diff()
            
            # Complex engineered features
            if all(col in df.columns for col in ['voltage', 'current']):
                df['power'] = df['voltage'] * df['current']
                df['energy_efficiency'] = df['voltage'] / (df['current'].abs() + 1e-6)
                df['impedance'] = df['voltage'] / (df['current'].abs() + 1e-6)
            
            if all(col in df.columns for col in ['soh', 'soc']):
                df['health_utilization'] = df['soh'] * df['soc'] / 10000  # Normalized
                df['degradation_risk'] = (100 - df['soh']) * df['soc'] / 10000
            
            # Temperature-based features
            if 'temperature' in df.columns:
                df['temp_extreme'] = ((df['temperature'] > 40) | (df['temperature'] < 0)).astype(int)
                df['temp_optimal'] = ((df['temperature'] >= 15) & (df['temperature'] <= 35)).astype(int)
            
            # Interaction features
            if all(col in df.columns for col in ['soh', 'temperature']):
                df['soh_temp_interaction'] = df['soh'] * (df['temperature'] - 25) / 25
            
            # Statistical features over time windows
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp')
                
                # Features within time windows
                for hours in [1, 6, 24]:
                    for col in ['soh', 'soc', 'voltage']:
                        if col in df.columns:
                            df[f'{col}_mean_{hours}h'] = df[col].rolling(
                                window=f'{hours}H', on='timestamp').mean()
                            df[f'{col}_volatility_{hours}h'] = df[col].rolling(
                                window=f'{hours}H', on='timestamp').std()
            
            logger.info(f"Engineered {len([c for c in df.columns if any(x in c for x in ['rolling', 'lag', 'rate', 'diff'])])} new features")
            
            return df
            
        except Exception as e:
            logger.error(f"Error engineering battery features: {e}")
            return df
    
    def create_health_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced health indicators from battery data.
        
        Args:
            df: Battery DataFrame
            
        Returns:
            DataFrame with health indicators
        """
        try:
            df = df.copy()
            
            # State of Health trend
            if 'soh' in df.columns:
                df['soh_trend'] = df['soh'].rolling(window=10).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
                )
                df['soh_acceleration'] = df['soh_trend'].diff()
                df['soh_stability'] = df['soh'].rolling(window=20).std()
            
            # Voltage health indicators
            if 'voltage' in df.columns:
                df['voltage_stability'] = df['voltage'].rolling(window=10).std()
                df['voltage_trend'] = df['voltage'].rolling(window=10).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
                )
            
            # Temperature health indicators
            if 'temperature' in df.columns:
                df['temp_stress_indicator'] = np.maximum(
                    0, df['temperature'] - 40
                ) + np.maximum(0, -df['temperature'])
                df['temp_cycling_stress'] = df['temperature'].rolling(window=5).apply(
                    lambda x: (x.max() - x.min()) if len(x) == 5 else 0
                )
            
            # Current utilization indicators
            if 'current' in df.columns:
                df['current_utilization'] = df['current'].abs().rolling(window=10).mean()
                df['current_stress'] = (df['current'].abs() > 100).astype(int).rolling(window=20).mean()
            
            # Combined health score
            health_columns = ['soh', 'voltage_stability', 'temp_stress_indicator']
            available_health_cols = [col for col in health_columns if col in df.columns]
            
            if available_health_cols:
                # Normalize each component
                normalized_scores = []
                
                if 'soh' in available_health_cols:
                    normalized_scores.append(df['soh'] / 100)
                
                if 'voltage_stability' in available_health_cols:
                    voltage_score = 1 - np.clip(df['voltage_stability'] / 0.1, 0, 1)
                    normalized_scores.append(voltage_score)
                
                if 'temp_stress_indicator' in available_health_cols:
                    temp_score = 1 - np.clip(df['temp_stress_indicator'] / 20, 0, 1)
                    normalized_scores.append(temp_score)
                
                # Calculate weighted average
                if normalized_scores:
                    df['composite_health_score'] = np.mean(normalized_scores, axis=0) * 100
            
            logger.info("Created advanced health indicators")
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating health indicators: {e}")
            return df

class DataAggregator:
    """
    Data aggregation utilities for fleet and multi-battery analysis.
    """
    
    def __init__(self):
        self.aggregation_cache = {}
        
        logger.info("Data Aggregator initialized")
    
    def aggregate_fleet_data(self, 
                           df: pd.DataFrame,
                           groupby_columns: List[str],
                           aggregation_functions: Dict[str, List[str]] = None) -> pd.DataFrame:
        """
        Aggregate battery data at fleet level.
        
        Args:
            df: Battery data DataFrame
            groupby_columns: Columns to group by
            aggregation_functions: Custom aggregation functions
            
        Returns:
            Aggregated DataFrame
        """
        try:
            if aggregation_functions is None:
                aggregation_functions = {
                    'soh': ['mean', 'std', 'min', 'max', 'count'],
                    'soc': ['mean', 'std'],
                    'voltage': ['mean', 'std'],
                    'current': ['mean', 'std', 'max'],
                    'temperature': ['mean', 'max', 'min'],
                    'power': ['mean', 'max']
                }
            
            # Filter aggregation functions to available columns
            available_agg_funcs = {
                col: funcs for col, funcs in aggregation_functions.items()
                if col in df.columns
            }
            
            if not available_agg_funcs:
                logger.warning("No valid columns for aggregation")
                return pd.DataFrame()
            
            # Perform aggregation
            aggregated = df.groupby(groupby_columns).agg(available_agg_funcs)
            
            # Flatten column names
            aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
            aggregated = aggregated.reset_index()
            
            # Add derived metrics
            if 'soh_mean' in aggregated.columns and 'soh_count' in aggregated.columns:
                aggregated['fleet_health_score'] = (
                    aggregated['soh_mean'] * 0.7 + 
                    (100 - aggregated['soh_std'].fillna(0)) * 0.2 +
                    np.minimum(aggregated['soh_count'] / 10, 10) * 0.1
                )
            
            logger.info(f"Aggregated data from {len(df)} records to {len(aggregated)} groups")
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Error aggregating fleet data: {e}")
            return pd.DataFrame()
    
    def create_time_series_aggregations(self, 
                                      df: pd.DataFrame,
                                      time_column: str = 'timestamp',
                                      frequency: str = '1H') -> pd.DataFrame:
        """
        Create time-based aggregations of battery data.
        
        Args:
            df: Battery data DataFrame
            time_column: Name of timestamp column
            frequency: Aggregation frequency (e.g., '1H', '1D')
            
        Returns:
            Time-aggregated DataFrame
        """
        try:
            if time_column not in df.columns:
                logger.error(f"Time column {time_column} not found")
                return pd.DataFrame()
            
            df = df.copy()
            df[time_column] = pd.to_datetime(df[time_column])
            df = df.set_index(time_column)
            
            # Define aggregation functions
            agg_funcs = {
                'soh': ['mean', 'std', 'min', 'max'],
                'soc': ['mean', 'std'],
                'voltage': ['mean', 'std'],
                'current': ['mean', 'std', 'abs_max'],
                'temperature': ['mean', 'max', 'min'],
                'power': ['mean', 'max']
            }
            
            # Filter to available columns
            available_agg_funcs = {
                col: funcs for col, funcs in agg_funcs.items()
                if col in df.columns
            }
            
            # Custom aggregation function for absolute maximum
            def abs_max(series):
                return series.abs().max()
            
            # Register custom function
            pd.core.groupby.SeriesGroupBy.abs_max = abs_max
            
            # Perform resampling
            resampled = df.resample(frequency).agg(available_agg_funcs)
            
            # Flatten column names
            resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
            resampled = resampled.reset_index()
            
            # Add time-based features
            resampled['hour'] = resampled[time_column].dt.hour
            resampled['day_of_week'] = resampled[time_column].dt.dayofweek
            resampled['is_weekend'] = resampled['day_of_week'].isin([5, 6]).astype(int)
            
            logger.info(f"Created time series aggregations with frequency {frequency}")
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error creating time series aggregations: {e}")
            return pd.DataFrame()

# Factory functions for easy instantiation
def create_data_processor(enable_caching: bool = True) -> DataProcessor:
    """Create a configured data processor."""
    return DataProcessor(enable_caching=enable_caching)

def create_time_series_processor(window_size: int = 10, overlap: float = 0.5) -> TimeSeriesProcessor:
    """Create a configured time series processor."""
    return TimeSeriesProcessor(window_size=window_size, overlap=overlap)

def create_feature_engineer() -> FeatureEngineer:
    """Create a feature engineer."""
    return FeatureEngineer()

def create_data_aggregator() -> DataAggregator:
    """Create a data aggregator."""
    return DataAggregator()

# Convenience functions
def process_battery_data(data: Union[pd.DataFrame, List[Dict[str, Any]]], 
                        battery_id: str = None) -> Tuple[pd.DataFrame, DataValidationResult]:
    """
    Convenience function to process battery data.
    
    Args:
        data: Battery data
        battery_id: Optional battery ID filter
        
    Returns:
        Processed data and validation result
    """
    processor = create_data_processor()
    return processor.process_battery_telemetry(data, battery_id)

def validate_battery_telemetry(df: pd.DataFrame) -> DataValidationResult:
    """
    Convenience function to validate battery telemetry.
    
    Args:
        df: Battery telemetry DataFrame
        
    Returns:
        Validation result
    """
    processor = create_data_processor()
    return processor.validate_battery_data(df)

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convenience function to engineer features.
    
    Args:
        df: Battery DataFrame
        
    Returns:
        DataFrame with engineered features
    """
    engineer = create_feature_engineer()
    return engineer.engineer_battery_features(df)

# Log module initialization
logger.info("BatteryMind Data Utils Module v1.0.0 loaded successfully")
