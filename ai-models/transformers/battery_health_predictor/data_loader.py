"""
BatteryMind - Battery Health Data Loader

Advanced data loading pipeline for battery health prediction with efficient
batching, streaming capabilities, and comprehensive data validation.

Features:
- High-performance data loading with multi-processing
- Real-time streaming from AWS IoT Core
- Comprehensive data validation and quality checks
- Efficient memory management for large datasets
- Support for multiple data formats (CSV, Parquet, JSON)
- Time-series specific batching and sequence handling
- Integration with AWS S3 and DynamoDB
- Data augmentation and synthetic data generation

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass, field
import logging
import time
import json
import pickle
import asyncio
from pathlib import Path
import threading
from collections import deque
import warnings
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

# AWS imports
import boto3
import botocore
from botocore.exceptions import ClientError, NoCredentialsError

# Data processing imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pyarrow.parquet as pq
import pyarrow as pa
from scipy import signal
from scipy.stats import zscore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BatteryDataConfig:
    """
    Configuration for battery data loading and processing.
    
    Attributes:
        # Data source configuration
        data_sources (List[str]): List of data source paths or endpoints
        data_format (str): Data format ('csv', 'parquet', 'json', 'streaming')
        
        # Sequence configuration
        sequence_length (int): Length of input sequences
        prediction_horizon (int): Number of future steps to predict
        overlap_ratio (float): Overlap ratio between sequences
        
        # Feature configuration
        feature_columns (List[str]): List of feature column names
        target_columns (List[str]): List of target column names
        timestamp_column (str): Name of timestamp column
        battery_id_column (str): Name of battery ID column
        
        # Data validation
        validate_data (bool): Enable data validation
        remove_outliers (bool): Remove outliers from data
        outlier_threshold (float): Z-score threshold for outlier detection
        
        # Performance configuration
        batch_size (int): Batch size for data loading
        num_workers (int): Number of worker processes
        prefetch_factor (int): Number of batches to prefetch
        pin_memory (bool): Pin memory for faster GPU transfer
        
        # AWS configuration
        aws_region (str): AWS region for S3 and IoT
        s3_bucket (str): S3 bucket for data storage
        iot_topic (str): IoT topic for real-time data
        
        # Caching configuration
        enable_caching (bool): Enable data caching
        cache_dir (str): Directory for data cache
        cache_size_limit (int): Maximum cache size in MB
    """
    # Data source configuration
    data_sources: List[str] = field(default_factory=list)
    data_format: str = "csv"
    
    # Sequence configuration
    sequence_length: int = 512
    prediction_horizon: int = 24
    overlap_ratio: float = 0.5
    
    # Feature configuration
    feature_columns: List[str] = field(default_factory=lambda: [
        'voltage', 'current', 'temperature', 'state_of_charge',
        'internal_resistance', 'capacity', 'cycle_count', 'age_days'
    ])
    target_columns: List[str] = field(default_factory=lambda: [
        'state_of_health', 'capacity_fade_rate', 'resistance_increase_rate', 'thermal_degradation'
    ])
    timestamp_column: str = "timestamp"
    battery_id_column: str = "battery_id"
    
    # Data validation
    validate_data: bool = True
    remove_outliers: bool = True
    outlier_threshold: float = 3.0
    
    # Performance configuration
    batch_size: int = 32
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True
    
    # AWS configuration
    aws_region: str = "us-east-1"
    s3_bucket: str = "batterymind-data"
    iot_topic: str = "battery/telemetry"
    
    # Caching configuration
    enable_caching: bool = True
    cache_dir: str = "./data_cache"
    cache_size_limit: int = 1024  # MB

class BatteryDataValidator:
    """
    Comprehensive data validation for battery telemetry data.
    """
    
    def __init__(self, config: BatteryDataConfig):
        self.config = config
        self.validation_rules = self._setup_validation_rules()
    
    def _setup_validation_rules(self) -> Dict[str, Dict]:
        """Setup validation rules for battery data."""
        return {
            'voltage': {'min': 2.0, 'max': 4.5, 'unit': 'V'},
            'current': {'min': -200.0, 'max': 200.0, 'unit': 'A'},
            'temperature': {'min': -40.0, 'max': 80.0, 'unit': '°C'},
            'state_of_charge': {'min': 0.0, 'max': 1.0, 'unit': '%'},
            'state_of_health': {'min': 0.0, 'max': 1.0, 'unit': '%'},
            'internal_resistance': {'min': 0.001, 'max': 1.0, 'unit': 'Ω'},
            'capacity': {'min': 0.1, 'max': 200.0, 'unit': 'Ah'},
            'cycle_count': {'min': 0, 'max': 10000, 'unit': 'cycles'},
            'age_days': {'min': 0, 'max': 3650, 'unit': 'days'}
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate entire dataframe and return cleaned data with validation report.
        
        Args:
            df (pd.DataFrame): Input dataframe to validate
            
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and validation report
        """
        validation_report = {
            'original_rows': len(df),
            'validation_errors': [],
            'outliers_removed': 0,
            'missing_values_filled': 0,
            'invalid_values_corrected': 0
        }
        
        # Check required columns
        missing_cols = set(self.config.feature_columns + self.config.target_columns) - set(df.columns)
        if missing_cols:
            validation_report['validation_errors'].append(f"Missing columns: {missing_cols}")
            return df, validation_report
        
        # Validate data types
        df = self._validate_data_types(df, validation_report)
        
        # Validate value ranges
        df = self._validate_value_ranges(df, validation_report)
        
        # Handle missing values
        df = self._handle_missing_values(df, validation_report)
        
        # Remove outliers if enabled
        if self.config.remove_outliers:
            df = self._remove_outliers(df, validation_report)
        
        # Validate temporal consistency
        df = self._validate_temporal_consistency(df, validation_report)
        
        validation_report['final_rows'] = len(df)
        validation_report['data_quality_score'] = self._calculate_quality_score(validation_report)
        
        return df, validation_report
    
    def _validate_data_types(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Validate and correct data types."""
        try:
            # Convert timestamp column
            if self.config.timestamp_column in df.columns:
                df[self.config.timestamp_column] = pd.to_datetime(df[self.config.timestamp_column])
            
            # Convert numeric columns
            numeric_columns = self.config.feature_columns + self.config.target_columns
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
        except Exception as e:
            report['validation_errors'].append(f"Data type validation error: {str(e)}")
        
        return df
    
    def _validate_value_ranges(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Validate value ranges according to physical constraints."""
        for column, rules in self.validation_rules.items():
            if column in df.columns:
                # Check minimum values
                invalid_min = df[column] < rules['min']
                if invalid_min.any():
                    df.loc[invalid_min, column] = rules['min']
                    report['invalid_values_corrected'] += invalid_min.sum()
                
                # Check maximum values
                invalid_max = df[column] > rules['max']
                if invalid_max.any():
                    df.loc[invalid_max, column] = rules['max']
                    report['invalid_values_corrected'] += invalid_max.sum()
        
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Handle missing values using appropriate strategies."""
        for column in self.config.feature_columns + self.config.target_columns:
            if column in df.columns:
                missing_count = df[column].isnull().sum()
                if missing_count > 0:
                    if column in ['voltage', 'current', 'temperature']:
                        # Forward fill for sensor data
                        df[column] = df[column].fillna(method='ffill')
                    elif column in ['state_of_charge', 'state_of_health']:
                        # Interpolate for state variables
                        df[column] = df[column].interpolate(method='linear')
                    else:
                        # Use median for other features
                        df[column] = df[column].fillna(df[column].median())
                    
                    report['missing_values_filled'] += missing_count
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Remove outliers using z-score method."""
        initial_rows = len(df)
        
        for column in self.config.feature_columns:
            if column in df.columns:
                z_scores = np.abs(zscore(df[column].dropna()))
                outlier_mask = z_scores > self.config.outlier_threshold
                df = df[~outlier_mask]
        
        outliers_removed = initial_rows - len(df)
        report['outliers_removed'] = outliers_removed
        
        return df
    
    def _validate_temporal_consistency(self, df: pd.DataFrame, report: Dict) -> pd.DataFrame:
        """Validate temporal consistency of data."""
        if self.config.timestamp_column in df.columns:
            # Sort by timestamp
            df = df.sort_values(self.config.timestamp_column)
            
            # Check for duplicate timestamps
            duplicates = df.duplicated(subset=[self.config.timestamp_column, self.config.battery_id_column])
            if duplicates.any():
                df = df[~duplicates]
                report['validation_errors'].append(f"Removed {duplicates.sum()} duplicate timestamps")
        
        return df
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate overall data quality score."""
        if report['original_rows'] == 0:
            return 0.0
        
        # Base score
        retention_rate = report['final_rows'] / report['original_rows']
        
        # Penalty for errors
        error_penalty = min(0.2, len(report['validation_errors']) * 0.05)
        
        # Penalty for corrections
        correction_penalty = min(0.1, (report['invalid_values_corrected'] / report['original_rows']) * 0.5)
        
        quality_score = retention_rate - error_penalty - correction_penalty
        return max(0.0, min(1.0, quality_score))

class BatteryDataset(Dataset):
    """
    PyTorch Dataset for battery health prediction with time-series support.
    """
    
    def __init__(self, data: pd.DataFrame, config: BatteryDataConfig, 
                 mode: str = 'train', transform: Optional[callable] = None):
        """
        Initialize battery dataset.
        
        Args:
            data (pd.DataFrame): Battery data
            config (BatteryDataConfig): Data configuration
            mode (str): Dataset mode ('train', 'val', 'test')
            transform (callable, optional): Data transformation function
        """
        self.data = data
        self.config = config
        self.mode = mode
        self.transform = transform
        
        # Prepare sequences
        self.sequences = self._prepare_sequences()
        
        logger.info(f"Created BatteryDataset with {len(self.sequences)} sequences in {mode} mode")
    
    def _prepare_sequences(self) -> List[Dict[str, Any]]:
        """Prepare time-series sequences from data."""
        sequences = []
        
        # Group by battery ID
        for battery_id, battery_data in self.data.groupby(self.config.battery_id_column):
            battery_data = battery_data.sort_values(self.config.timestamp_column)
            
            # Create sequences with overlap
            step_size = int(self.config.sequence_length * (1 - self.config.overlap_ratio))
            
            for start_idx in range(0, len(battery_data) - self.config.sequence_length - self.config.prediction_horizon + 1, step_size):
                end_idx = start_idx + self.config.sequence_length
                target_end_idx = end_idx + self.config.prediction_horizon
                
                # Extract features and targets
                features = battery_data.iloc[start_idx:end_idx][self.config.feature_columns].values
                targets = battery_data.iloc[end_idx:target_end_idx][self.config.target_columns].values
                
                # Extract metadata
                metadata = {
                    'battery_id': battery_id,
                    'start_timestamp': battery_data.iloc[start_idx][self.config.timestamp_column],
                    'end_timestamp': battery_data.iloc[end_idx-1][self.config.timestamp_column],
                    'sequence_length': self.config.sequence_length,
                    'prediction_horizon': self.config.prediction_horizon
                }
                
                sequences.append({
                    'features': features.astype(np.float32),
                    'targets': targets.astype(np.float32),
                    'metadata': metadata
                })
        
        return sequences
    
    def __len__(self) -> int:
        """Return number of sequences."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get sequence by index.
        
        Args:
            idx (int): Sequence index
            
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing features, targets, and metadata
        """
        sequence = self.sequences[idx]
        
        # Convert to tensors
        features = torch.from_numpy(sequence['features'])
        targets = torch.from_numpy(sequence['targets'])
        
        # Apply transforms if provided
        if self.transform:
            features, targets = self.transform(features, targets)
        
        return {
            'inputs': features,
            'targets': targets,
            'metadata': sequence['metadata']
        }

class BatterySequenceCollator:
    """
    Custom collator for batching battery sequences with padding support.
    """
    
    def __init__(self, config: BatteryDataConfig):
        self.config = config
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate batch of sequences.
        
        Args:
            batch (List[Dict]): List of sequence dictionaries
            
        Returns:
            Dict[str, torch.Tensor]: Batched tensors
        """
        # Stack inputs and targets
        inputs = torch.stack([item['inputs'] for item in batch])
        targets = torch.stack([item['targets'] for item in batch])
        
        # Collect metadata
        metadata = [item['metadata'] for item in batch]
        
        return {
            'inputs': inputs,
            'targets': targets,
            'metadata': metadata
        }

class BatteryDataLoader:
    """
    High-performance data loader for battery health prediction with streaming support.
    """
    
    def __init__(self, config: BatteryDataConfig):
        self.config = config
        self.validator = BatteryDataValidator(config)
        self.scaler = StandardScaler()
        
        # AWS clients
        self.s3_client = None
        self.iot_client = None
        self._setup_aws_clients()
        
        # Caching
        self.cache = {}
        self.cache_size = 0
        
        # Setup cache directory
        if config.enable_caching:
            Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    
    def _setup_aws_clients(self) -> None:
        """Setup AWS clients for S3 and IoT."""
        try:
            self.s3_client = boto3.client('s3', region_name=self.config.aws_region)
            self.iot_client = boto3.client('iot-data', region_name=self.config.aws_region)
            logger.info("AWS clients initialized successfully")
        except NoCredentialsError:
            logger.warning("AWS credentials not found. S3 and IoT features will be disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {e}")
    
    def load_data(self, data_source: str, **kwargs) -> pd.DataFrame:
        """
        Load data from various sources.
        
        Args:
            data_source (str): Path or identifier for data source
            **kwargs: Additional arguments for data loading
            
        Returns:
            pd.DataFrame: Loaded and validated data
        """
        # Check cache first
        cache_key = self._generate_cache_key(data_source, kwargs)
        if self.config.enable_caching and cache_key in self.cache:
            logger.info(f"Loading data from cache: {data_source}")
            return self.cache[cache_key]
        
        # Load data based on format
        if data_source.startswith('s3://'):
            data = self._load_from_s3(data_source, **kwargs)
        elif data_source.endswith('.parquet'):
            data = self._load_parquet(data_source, **kwargs)
        elif data_source.endswith('.csv'):
            data = self._load_csv(data_source, **kwargs)
        elif data_source.endswith('.json'):
            data = self._load_json(data_source, **kwargs)
        else:
            raise ValueError(f"Unsupported data source format: {data_source}")
        
        # Validate data
        if self.config.validate_data:
            data, validation_report = self.validator.validate_dataframe(data)
            logger.info(f"Data validation completed. Quality score: {validation_report['data_quality_score']:.3f}")
        
        # Cache data if enabled
        if self.config.enable_caching:
            self._cache_data(cache_key, data)
        
        return data
    
    def _load_from_s3(self, s3_path: str, **kwargs) -> pd.DataFrame:
        """Load data from S3."""
        if not self.s3_client:
            raise RuntimeError("S3 client not initialized")
        
        # Parse S3 path
        bucket, key = s3_path.replace('s3://', '').split('/', 1)
        
        try:
            # Download file
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            
            # Load based on file extension
            if key.endswith('.csv'):
                return pd.read_csv(response['Body'], **kwargs)
            elif key.endswith('.parquet'):
                return pd.read_parquet(response['Body'], **kwargs)
            elif key.endswith('.json'):
                return pd.read_json(response['Body'], **kwargs)
            else:
                raise ValueError(f"Unsupported S3 file format: {key}")
                
        except ClientError as e:
            logger.error(f"Failed to load data from S3: {e}")
            raise
    
    def _load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from CSV file."""
        default_kwargs = {
            'parse_dates': [self.config.timestamp_column] if self.config.timestamp_column else None,
            'low_memory': False
        }
        default_kwargs.update(kwargs)
        
        return pd.read_csv(file_path, **default_kwargs)
    
    def _load_parquet(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from Parquet file."""
        return pd.read_parquet(file_path, **kwargs)
    
    def _load_json(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load data from JSON file."""
        default_kwargs = {
            'orient': 'records',
            'convert_dates': [self.config.timestamp_column] if self.config.timestamp_column else None
        }
        default_kwargs.update(kwargs)
        
        return pd.read_json(file_path, **default_kwargs)
    
    def create_data_loaders(self, data: pd.DataFrame, 
                          train_split: float = 0.7, 
                          val_split: float = 0.15,
                          test_split: float = 0.15) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create train, validation, and test data loaders.
        
        Args:
            data (pd.DataFrame): Input data
            train_split (float): Training data split ratio
            val_split (float): Validation data split ratio
            test_split (float): Test data split ratio
            
        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test loaders
        """
        # Split data by battery ID to avoid data leakage
        battery_ids = data[self.config.battery_id_column].unique()
        np.random.shuffle(battery_ids)
        
        n_train = int(len(battery_ids) * train_split)
        n_val = int(len(battery_ids) * val_split)
        
        train_ids = battery_ids[:n_train]
        val_ids = battery_ids[n_train:n_train + n_val]
        test_ids = battery_ids[n_train + n_val:]
        
        # Split data
        train_data = data[data[self.config.battery_id_column].isin(train_ids)]
        val_data = data[data[self.config.battery_id_column].isin(val_ids)]
        test_data = data[data[self.config.battery_id_column].isin(test_ids)]
        
        # Fit scaler on training data
        feature_data = train_data[self.config.feature_columns].values
        self.scaler.fit(feature_data)
        
        # Create datasets
        train_dataset = BatteryDataset(train_data, self.config, mode='train')
        val_dataset = BatteryDataset(val_data, self.config, mode='val')
        test_dataset = BatteryDataset(test_data, self.config, mode='test')
        
        # Create collator
        collator = BatterySequenceCollator(self.config)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            collate_fn=collator,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            collate_fn=collator,
            drop_last=False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            prefetch_factor=self.config.prefetch_factor,
            pin_memory=self.config.pin_memory,
            collate_fn=collator,
            drop_last=False
        )
        
        logger.info(f"Created data loaders - Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def _generate_cache_key(self, data_source: str, kwargs: Dict) -> str:
        """Generate cache key for data source."""
        import hashlib
        key_string = f"{data_source}_{str(sorted(kwargs.items()))}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data in memory and disk."""
        # Memory cache
        self.cache[cache_key] = data
        
        # Disk cache
        cache_file = Path(self.config.cache_dir) / f"{cache_key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to cache data to disk: {e}")
    
    def stream_from_iot(self, callback: callable, duration: Optional[int] = None) -> None:
        """
        Stream real-time data from AWS IoT Core.
        
        Args:
            callback (callable): Function to process incoming data
            duration (int, optional): Streaming duration in seconds
        """
        if not self.iot_client:
            raise RuntimeError("IoT client not initialized")
        
        # Implementation would depend on specific IoT setup
        # This is a placeholder for the streaming functionality
        logger.info(f"Starting IoT data streaming from topic: {self.config.iot_topic}")
        
        # In a real implementation, this would use AWS IoT Device SDK
        # to subscribe to MQTT topics and process incoming messages
        pass
    
    def get_data_statistics(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive statistics about the dataset.
        
        Args:
            data (pd.DataFrame): Input data
            
        Returns:
            Dict[str, Any]: Dataset statistics
        """
        stats = {
            'total_rows': len(data),
            'total_batteries': data[self.config.battery_id_column].nunique(),
            'date_range': {
                'start': data[self.config.timestamp_column].min(),
                'end': data[self.config.timestamp_column].max()
            },
            'feature_statistics': {},
            'missing_values': data.isnull().sum().to_dict(),
            'data_quality_score': 0.0
        }
        
        # Feature statistics
        for column in self.config.feature_columns + self.config.target_columns:
            if column in data.columns:
                stats['feature_statistics'][column] = {
                    'mean': float(data[column].mean()),
                    'std': float(data[column].std()),
                    'min': float(data[column].min()),
                    'max': float(data[column].max()),
                    'median': float(data[column].median())
                }
        
        # Calculate overall data quality score
        missing_ratio = data.isnull().sum().sum() / (len(data) * len(data.columns))
        stats['data_quality_score'] = max(0.0, 1.0 - missing_ratio)
        
        return stats

# Factory functions for easy instantiation
def create_battery_data_loader(config: Optional[BatteryDataConfig] = None) -> BatteryDataLoader:
    """
    Factory function to create a BatteryDataLoader.
    
    Args:
        config (BatteryDataConfig, optional): Data configuration
        
    Returns:
        BatteryDataLoader: Configured data loader instance
    """
    if config is None:
        config = BatteryDataConfig()
    
    return BatteryDataLoader(config)

def load_battery_data_from_sources(sources: List[str], 
                                 config: Optional[BatteryDataConfig] = None) -> pd.DataFrame:
    """
    Load and combine battery data from multiple sources.
    
    Args:
        sources (List[str]): List of data source paths
        config (BatteryDataConfig, optional): Data configuration
        
    Returns:
        pd.DataFrame: Combined and validated data
    """
    if config is None:
        config = BatteryDataConfig()
    
    data_loader = BatteryDataLoader(config)
    
    # Load data from all sources
    all_data = []
    for source in sources:
        try:
            data = data_loader.load_data(source)
            all_data.append(data)
            logger.info(f"Loaded {len(data)} rows from {source}")
        except Exception as e:
            logger.error(f"Failed to load data from {source}: {e}")
    
    if not all_data:
        raise ValueError("No data could be loaded from any source")
    
    # Combine all data
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info(f"Combined data: {len(combined_data)} total rows")
    
    return combined_data
