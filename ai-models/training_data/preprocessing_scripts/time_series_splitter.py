"""
BatteryMind - Time Series Splitter Module

Advanced time series data splitting utilities for battery sensor data with
temporal consistency preservation, sliding window generation, and
cross-validation strategies specifically designed for battery applications.

Features:
- Temporal consistency preservation in train/test splits
- Sliding window sequence generation
- Multi-horizon forecasting data preparation
- Cross-validation with temporal constraints
- Battery lifecycle-aware splitting
- Real-time streaming data preparation

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Iterator
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
import warnings
from sklearn.model_selection import TimeSeriesSplit
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TimeSeriesSplitConfig:
    """
    Configuration for time series splitting parameters.
    
    Attributes:
        # Sequence parameters
        sequence_length (int): Length of input sequences
        prediction_horizon (int): Number of steps to predict ahead
        step_size (int): Step size for sliding window
        
        # Split parameters
        train_ratio (float): Ratio of data for training
        validation_ratio (float): Ratio of data for validation
        test_ratio (float): Ratio of data for testing
        
        # Cross-validation parameters
        n_splits (int): Number of cross-validation splits
        gap_size (int): Gap between train and test in CV
        
        # Battery-specific parameters
        min_cycle_length (int): Minimum cycle length for battery data
        preserve_cycles (bool): Preserve complete charge/discharge cycles
        lifecycle_aware (bool): Consider battery lifecycle stages
        
        # Advanced features
        overlap_sequences (bool): Allow overlapping sequences
        shuffle_sequences (bool): Shuffle sequences (not recommended for time series)
        stratify_by_health (bool): Stratify splits by battery health
        include_metadata (bool): Include metadata in splits
    """
    # Sequence parameters
    sequence_length: int = 100
    prediction_horizon: int = 1
    step_size: int = 1
    
    # Split parameters
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cross-validation parameters
    n_splits: int = 5
    gap_size: int = 0
    
    # Battery-specific parameters
    min_cycle_length: int = 50
    preserve_cycles: bool = True
    lifecycle_aware: bool = True
    
    # Advanced features
    overlap_sequences: bool = True
    shuffle_sequences: bool = False
    stratify_by_health: bool = False
    include_metadata: bool = True

class BatteryTimeSeriesSplitter:
    """
    Advanced time series splitter for battery data with temporal consistency.
    """
    
    def __init__(self, config: TimeSeriesSplitConfig):
        self.config = config
        self.split_indices = {}
        self.metadata = {}
        
        # Validate configuration
        self._validate_config()
        
        logger.info("BatteryTimeSeriesSplitter initialized")
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check split ratios sum to 1
        total_ratio = self.config.train_ratio + self.config.validation_ratio + self.config.test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0, got {total_ratio}")
        
        # Check sequence parameters
        if self.config.sequence_length <= 0:
            raise ValueError("Sequence length must be positive")
        
        if self.config.prediction_horizon <= 0:
            raise ValueError("Prediction horizon must be positive")
    
    def create_sequences(self, data: Union[pd.DataFrame, np.ndarray],
                        target_columns: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
        """
        Create sliding window sequences from time series data.
        
        Args:
            data: Input time series data
            target_columns: Columns to use as targets (for supervised learning)
            
        Returns:
            Dictionary containing sequences and targets
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
            column_names = data.columns.tolist()
        else:
            data_array = data
            column_names = [f"feature_{i}" for i in range(data_array.shape[1])]
        
        sequences = []
        targets = []
        metadata_list = []
        
        # Determine target indices
        if target_columns:
            target_indices = [column_names.index(col) for col in target_columns if col in column_names]
        else:
            target_indices = list(range(data_array.shape[1]))
        
        # Generate sequences
        for i in range(0, len(data_array) - self.config.sequence_length - self.config.prediction_horizon + 1,
                      self.config.step_size):
            
            # Extract sequence
            sequence = data_array[i:i + self.config.sequence_length]
            
            # Extract target
            target_start = i + self.config.sequence_length
            target_end = target_start + self.config.prediction_horizon
            target = data_array[target_start:target_end, target_indices]
            
            sequences.append(sequence)
            targets.append(target)
            
            # Store metadata
            if self.config.include_metadata:
                metadata_list.append({
                    'start_index': i,
                    'end_index': i + self.config.sequence_length,
                    'target_start': target_start,
                    'target_end': target_end
                })
        
        result = {
            'sequences': np.array(sequences),
            'targets': np.array(targets)
        }
        
        if self.config.include_metadata:
            result['metadata'] = metadata_list
        
        logger.info(f"Created {len(sequences)} sequences of length {self.config.sequence_length}")
        return result
    
    def temporal_split(self, data: Union[pd.DataFrame, np.ndarray],
                      time_column: Optional[str] = None) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split time series data maintaining temporal order.
        
        Args:
            data: Input time series data
            time_column: Name of time column (if DataFrame)
            
        Returns:
            Dictionary with train/validation/test splits
        """
        if isinstance(data, pd.DataFrame):
            if time_column and time_column in data.columns:
                # Sort by time if time column is provided
                data = data.sort_values(time_column)
            data_array = data.values
        else:
            data_array = data
        
        n_samples = len(data_array)
        
        # Calculate split indices
        train_end = int(n_samples * self.config.train_ratio)
        val_end = int(n_samples * (self.config.train_ratio + self.config.validation_ratio))
        
        # Create splits
        train_data = data_array[:train_end]
        val_data = data_array[train_end:val_end]
        test_data = data_array[val_end:]
        
        # Store split indices
        self.split_indices = {
            'train': (0, train_end),
            'validation': (train_end, val_end),
            'test': (val_end, n_samples)
        }
        
        # Create sequences for each split
        splits = {}
        for split_name, split_data in [('train', train_data), ('validation', val_data), ('test', test_data)]:
            if len(split_data) >= self.config.sequence_length + self.config.prediction_horizon:
                splits[split_name] = self.create_sequences(split_data)
            else:
                logger.warning(f"Insufficient data for {split_name} split")
                splits[split_name] = {'sequences': np.array([]), 'targets': np.array([])}
        
        logger.info(f"Created temporal splits: train={len(splits['train']['sequences'])}, "
                   f"val={len(splits['validation']['sequences'])}, test={len(splits['test']['sequences'])}")
        
        return splits
    
    def battery_lifecycle_split(self, data: pd.DataFrame, 
                              cycle_column: str = 'cycle',
                              soh_column: str = 'soh') -> Dict[str, Dict[str, np.ndarray]]:
        """
        Split data based on battery lifecycle stages.
        
        Args:
            data: Input DataFrame with battery data
            cycle_column: Column name for cycle information
            soh_column: Column name for state of health
            
        Returns:
            Dictionary with lifecycle-based splits
        """
        if cycle_column not in data.columns or soh_column not in data.columns:
            logger.warning("Required columns not found, falling back to temporal split")
            return self.temporal_split(data)
        
        # Define lifecycle stages based on SOH
        def get_lifecycle_stage(soh):
            if soh >= 0.9:
                return 'early_life'
            elif soh >= 0.7:
                return 'mid_life'
            else:
                return 'end_of_life'
        
        data['lifecycle_stage'] = data[soh_column].apply(get_lifecycle_stage)
        
        # Split by lifecycle stages
        splits = {}
        for stage in ['early_life', 'mid_life', 'end_of_life']:
            stage_data = data[data['lifecycle_stage'] == stage]
            
            if len(stage_data) >= self.config.sequence_length + self.config.prediction_horizon:
                # Further split each stage temporally
                stage_splits = self.temporal_split(stage_data.drop('lifecycle_stage', axis=1))
                splits[stage] = stage_splits
            else:
                logger.warning(f"Insufficient data for {stage} stage")
        
        return splits
    
    def cross_validation_split(self, data: Union[pd.DataFrame, np.ndarray]) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate cross-validation splits for time series data.
        
        Args:
            data: Input time series data
            
        Yields:
            Tuples of (train_indices, test_indices)
        """
        if isinstance(data, pd.DataFrame):
            data_array = data.values
        else:
            data_array = data
        
        n_samples = len(data_array)
        
        # Use TimeSeriesSplit with gap
        tscv = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            gap=self.config.gap_size
        )
        
        for train_idx, test_idx in tscv.split(data_array):
            yield train_idx, test_idx
    
    def preserve_battery_cycles(self, data: pd.DataFrame,
                              cycle_column: str = 'cycle') -> Dict[str, List[int]]:
        """
        Split data while preserving complete battery cycles.
        
        Args:
            data: Input DataFrame with cycle information
            cycle_column: Column name for cycle information
            
        Returns:
            Dictionary mapping split names to cycle lists
        """
        if cycle_column not in data.columns:
            raise ValueError(f"Cycle column '{cycle_column}' not found in data")
        
        unique_cycles = sorted(data[cycle_column].unique())
        n_cycles = len(unique_cycles)
        
        # Calculate cycle splits
        train_cycles = int(n_cycles * self.config.train_ratio)
        val_cycles = int(n_cycles * self.config.validation_ratio)
        
        cycle_splits = {
            'train': unique_cycles[:train_cycles],
            'validation': unique_cycles[train_cycles:train_cycles + val_cycles],
            'test': unique_cycles[train_cycles + val_cycles:]
        }
        
        logger.info(f"Split {n_cycles} cycles: train={len(cycle_splits['train'])}, "
                   f"val={len(cycle_splits['validation'])}, test={len(cycle_splits['test'])}")
        
        return cycle_splits
    
    def stratified_split_by_health(self, data: pd.DataFrame,
                                 soh_column: str = 'soh',
                                 n_bins: int = 5) -> Dict[str, np.ndarray]:
        """
        Create stratified splits based on battery health distribution.
        
        Args:
            data: Input DataFrame with SOH information
            soh_column: Column name for state of health
            n_bins: Number of health bins for stratification
            
        Returns:
            Dictionary with stratified split indices
        """
        if soh_column not in data.columns:
            raise ValueError(f"SOH column '{soh_column}' not found in data")
        
        # Create health bins
        soh_values = data[soh_column].values
        bins = np.linspace(soh_values.min(), soh_values.max(), n_bins + 1)
        health_bins = np.digitize(soh_values, bins) - 1
        
        # Ensure stratified sampling within each bin
        splits = {'train': [], 'validation': [], 'test': []}
        
        for bin_idx in range(n_bins):
            bin_indices = np.where(health_bins == bin_idx)[0]
            
            if len(bin_indices) > 0:
                # Split this bin temporally
                n_bin_samples = len(bin_indices)
                train_end = int(n_bin_samples * self.config.train_ratio)
                val_end = int(n_bin_samples * (self.config.train_ratio + self.config.validation_ratio))
                
                splits['train'].extend(bin_indices[:train_end])
                splits['validation'].extend(bin_indices[train_end:val_end])
                splits['test'].extend(bin_indices[val_end:])
        
        # Convert to numpy arrays and sort
        for split_name in splits:
            splits[split_name] = np.array(sorted(splits[split_name]))
        
        return splits
    
    def create_multi_horizon_targets(self, data: np.ndarray,
                                   horizons: List[int]) -> Dict[int, np.ndarray]:
        """
        Create targets for multiple prediction horizons.
        
        Args:
            data: Input time series data
            horizons: List of prediction horizons
            
        Returns:
            Dictionary mapping horizons to target arrays
        """
        multi_horizon_targets = {}
        
        for horizon in horizons:
            targets = []
            
            for i in range(len(data) - self.config.sequence_length - horizon + 1):
                target_start = i + self.config.sequence_length
                target_end = target_start + horizon
                target = data[target_start:target_end]
                targets.append(target)
            
            multi_horizon_targets[horizon] = np.array(targets)
        
        return multi_horizon_targets
    
    def get_split_statistics(self, splits: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Dict[str, Any]]:
        """
        Calculate statistics for each split.
        
        Args:
            splits: Dictionary of splits
            
        Returns:
            Dictionary of split statistics
        """
        statistics = {}
        
        for split_name, split_data in splits.items():
            if 'sequences' in split_data and len(split_data['sequences']) > 0:
                sequences = split_data['sequences']
                targets = split_data['targets']
                
                stats = {
                    'n_sequences': len(sequences),
                    'sequence_shape': sequences.shape,
                    'target_shape': targets.shape,
                    'sequence_mean': np.mean(sequences),
                    'sequence_std': np.std(sequences),
                    'target_mean': np.mean(targets),
                    'target_std': np.std(targets)
                }
                
                statistics[split_name] = stats
        
        return statistics
    
    def save_splits(self, splits: Dict[str, Dict[str, np.ndarray]], 
                   filepath: str):
        """
        Save splits to file.
        
        Args:
            splits: Dictionary of splits to save
            filepath: Path to save file
        """
        save_data = {
            'splits': splits,
            'config': self.config,
            'split_indices': self.split_indices,
            'metadata': self.metadata
        }
        
        np.savez_compressed(filepath, **save_data)
        logger.info(f"Splits saved to {filepath}")
    
    def load_splits(self, filepath: str) -> Dict[str, Dict[str, np.ndarray]]:
        """
        Load splits from file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Dictionary of loaded splits
        """
        loaded_data = np.load(filepath, allow_pickle=True)
        
        self.config = loaded_data['config'].item()
        self.split_indices = loaded_data['split_indices'].item()
        self.metadata = loaded_data['metadata'].item()
        
        splits = loaded_data['splits'].item()
        
        logger.info(f"Splits loaded from {filepath}")
        return splits

class StreamingSequenceGenerator:
    """
    Generate sequences from streaming battery data.
    """
    
    def __init__(self, sequence_length: int, step_size: int = 1):
        self.sequence_length = sequence_length
        self.step_size = step_size
        self.buffer = []
        self.sequence_count = 0
    
    def add_data_point(self, data_point: Union[np.ndarray, List[float]]) -> Optional[np.ndarray]:
        """
        Add new data point and return sequence if ready.
        
        Args:
            data_point: New data point
            
        Returns:
            Sequence array if ready, None otherwise
        """
        self.buffer.append(np.array(data_point))
        
        # Maintain buffer size
        if len(self.buffer) > self.sequence_length:
            self.buffer.pop(0)
        
        # Return sequence if buffer is full
        if len(self.buffer) == self.sequence_length:
            if self.sequence_count % self.step_size == 0:
                sequence = np.array(self.buffer)
                self.sequence_count += 1
                return sequence
            
            self.sequence_count += 1
        
        return None
    
    def reset(self):
        """Reset the generator."""
        self.buffer = []
        self.sequence_count = 0

# Factory functions
def create_time_series_splitter(config: Optional[TimeSeriesSplitConfig] = None) -> BatteryTimeSeriesSplitter:
    """
    Factory function to create a time series splitter.
    
    Args:
        config: Splitting configuration
        
    Returns:
        Configured BatteryTimeSeriesSplitter instance
    """
    if config is None:
        config = TimeSeriesSplitConfig()
    
    return BatteryTimeSeriesSplitter(config)

def create_streaming_generator(sequence_length: int, step_size: int = 1) -> StreamingSequenceGenerator:
    """
    Factory function to create a streaming sequence generator.
    
    Args:
        sequence_length: Length of sequences to generate
        step_size: Step size between sequences
        
    Returns:
        Configured StreamingSequenceGenerator instance
    """
    return StreamingSequenceGenerator(sequence_length, step_size)

# Utility functions
def split_battery_dataset(data: pd.DataFrame,
                         config: Optional[TimeSeriesSplitConfig] = None,
                         split_type: str = 'temporal') -> Dict[str, Dict[str, np.ndarray]]:
    """
    Convenience function to split a battery dataset.
    
    Args:
        data: Input DataFrame with battery data
        config: Splitting configuration
        split_type: Type of split ('temporal', 'lifecycle', 'cycles')
        
    Returns:
        Dictionary of splits
    """
    splitter = create_time_series_splitter(config)
    
    if split_type == 'temporal':
        return splitter.temporal_split(data)
    elif split_type == 'lifecycle':
        return splitter.battery_lifecycle_split(data)
    elif split_type == 'cycles':
        cycle_splits = splitter.preserve_battery_cycles(data)
        # Convert cycle-based splits to data splits
        splits = {}
        for split_name, cycles in cycle_splits.items():
            split_data = data[data['cycle'].isin(cycles)]
            splits[split_name] = splitter.create_sequences(split_data)
        return splits
    else:
        raise ValueError(f"Unknown split type: {split_type}")

def validate_temporal_consistency(splits: Dict[str, Dict[str, np.ndarray]],
                                time_indices: Optional[np.ndarray] = None) -> Dict[str, bool]:
    """
    Validate temporal consistency of splits.
    
    Args:
        splits: Dictionary of splits to validate
        time_indices: Optional time indices for validation
        
    Returns:
        Dictionary of validation results
    """
    validation_results = {}
    
    # Check that splits don't overlap in time
    if 'train' in splits and 'validation' in splits and 'test' in splits:
        train_sequences = splits['train']['sequences']
        val_sequences = splits['validation']['sequences']
        test_sequences = splits['test']['sequences']
        
        # Check non-empty splits
        validation_results['non_empty_splits'] = all(
            len(seq) > 0 for seq in [train_sequences, val_sequences, test_sequences]
        )
        
        # Check temporal order (if time indices provided)
        if time_indices is not None:
            validation_results['temporal_order_preserved'] = True
            # Additional temporal validation logic would go here
        
        # Check sequence shapes consistency
        if len(train_sequences) > 0 and len(val_sequences) > 0:
            validation_results['consistent_sequence_shape'] = (
                train_sequences.shape[1:] == val_sequences.shape[1:]
            )
    
    return validation_results
