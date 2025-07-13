"""
BatteryMind - Advanced Noise Generator

Sophisticated noise generation system for creating realistic sensor noise,
measurement uncertainties, and data quality variations in battery datasets.

Features:
- Multiple noise types (Gaussian, Poisson, impulse, colored)
- Sensor-specific noise characteristics
- Temperature-dependent noise modeling
- Aging-related noise increase
- Correlated noise between sensors
- Drift and bias simulation
- Missing data patterns

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
from enum import Enum
import scipy.signal
from scipy import stats
from scipy.interpolate import interp1d
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NoiseType(Enum):
    """Types of noise that can be generated."""
    GAUSSIAN = "gaussian"
    POISSON = "poisson"
    IMPULSE = "impulse"
    COLORED = "colored"
    DRIFT = "drift"
    QUANTIZATION = "quantization"
    OUTLIERS = "outliers"

class SensorType(Enum):
    """Types of sensors with specific noise characteristics."""
    VOLTAGE = "voltage"
    CURRENT = "current"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    HUMIDITY = "humidity"
    VIBRATION = "vibration"
    ACOUSTIC = "acoustic"

@dataclass
class NoiseParameters:
    """
    Parameters for noise generation.
    
    Attributes:
        noise_type (NoiseType): Type of noise to generate
        amplitude (float): Noise amplitude (standard deviation for Gaussian)
        frequency_range (Tuple[float, float]): Frequency range for colored noise
        correlation_length (float): Correlation length for colored noise
        drift_rate (float): Drift rate per unit time
        outlier_probability (float): Probability of outliers
        outlier_magnitude (float): Magnitude of outliers
        temperature_coefficient (float): Temperature dependency of noise
        aging_coefficient (float): Aging dependency of noise
        quantization_bits (int): Number of bits for quantization noise
    """
    noise_type: NoiseType = NoiseType.GAUSSIAN
    amplitude: float = 0.01
    frequency_range: Tuple[float, float] = (0.1, 10.0)
    correlation_length: float = 10.0
    drift_rate: float = 0.001
    outlier_probability: float = 0.01
    outlier_magnitude: float = 5.0
    temperature_coefficient: float = 0.001
    aging_coefficient: float = 0.0001
    quantization_bits: int = 12

@dataclass
class SensorCharacteristics:
    """
    Characteristics of specific sensor types.
    
    Attributes:
        sensor_type (SensorType): Type of sensor
        base_noise_level (float): Base noise level
        resolution (float): Sensor resolution
        range_min (float): Minimum measurement range
        range_max (float): Maximum measurement range
        temperature_drift (float): Temperature drift coefficient
        nonlinearity (float): Nonlinearity factor
        hysteresis (float): Hysteresis factor
        response_time (float): Response time in seconds
        bandwidth (float): Sensor bandwidth in Hz
    """
    sensor_type: SensorType
    base_noise_level: float = 0.001
    resolution: float = 0.001
    range_min: float = 0.0
    range_max: float = 100.0
    temperature_drift: float = 0.001
    nonlinearity: float = 0.0001
    hysteresis: float = 0.0001
    response_time: float = 0.1
    bandwidth: float = 1000.0

class GaussianNoiseGenerator:
    """
    Generator for Gaussian (white) noise.
    """
    
    def __init__(self, parameters: NoiseParameters):
        self.params = parameters
        self.rng = np.random.RandomState()
        
    def generate(self, size: int, signal_level: float = 1.0,
                temperature: float = 25.0, age_factor: float = 1.0) -> np.ndarray:
        """
        Generate Gaussian noise.
        
        Args:
            size (int): Number of samples
            signal_level (float): Signal level for SNR calculation
            temperature (float): Temperature for temperature-dependent noise
            age_factor (float): Aging factor for noise increase
            
        Returns:
            np.ndarray: Generated noise
        """
        # Base noise amplitude
        amplitude = self.params.amplitude
        
        # Temperature dependency
        temp_factor = 1.0 + self.params.temperature_coefficient * (temperature - 25.0)
        
        # Aging dependency
        aging_factor = 1.0 + self.params.aging_coefficient * age_factor
        
        # Signal-dependent noise (proportional to signal level)
        signal_factor = 0.1 + 0.9 * signal_level
        
        # Final amplitude
        final_amplitude = amplitude * temp_factor * aging_factor * signal_factor
        
        return self.rng.normal(0, final_amplitude, size)

class ColoredNoiseGenerator:
    """
    Generator for colored noise (1/f, pink, brown, etc.).
    """
    
    def __init__(self, parameters: NoiseParameters):
        self.params = parameters
        self.rng = np.random.RandomState()
        
    def generate(self, size: int, sampling_rate: float = 1.0,
                color_exponent: float = -1.0) -> np.ndarray:
        """
        Generate colored noise using spectral shaping.
        
        Args:
            size (int): Number of samples
            sampling_rate (float): Sampling rate in Hz
            color_exponent (float): Spectral exponent (-2=brown, -1=pink, 0=white)
            
        Returns:
            np.ndarray: Generated colored noise
        """
        # Generate white noise
        white_noise = self.rng.normal(0, 1, size)
        
        # Apply FFT
        fft_noise = np.fft.fft(white_noise)
        
        # Create frequency array
        freqs = np.fft.fftfreq(size, 1/sampling_rate)
        freqs[0] = 1e-10  # Avoid division by zero
        
        # Apply spectral shaping
        spectral_shape = np.abs(freqs) ** (color_exponent / 2)
        shaped_fft = fft_noise * spectral_shape
        
        # Convert back to time domain
        colored_noise = np.real(np.fft.ifft(shaped_fft))
        
        # Normalize to desired amplitude
        colored_noise = colored_noise / np.std(colored_noise) * self.params.amplitude
        
        return colored_noise

class ImpulseNoiseGenerator:
    """
    Generator for impulse (spike) noise.
    """
    
    def __init__(self, parameters: NoiseParameters):
        self.params = parameters
        self.rng = np.random.RandomState()
        
    def generate(self, size: int, impulse_rate: float = 0.01,
                impulse_amplitude: float = None) -> np.ndarray:
        """
        Generate impulse noise.
        
        Args:
            size (int): Number of samples
            impulse_rate (float): Rate of impulses (probability per sample)
            impulse_amplitude (float): Amplitude of impulses
            
        Returns:
            np.ndarray: Generated impulse noise
        """
        if impulse_amplitude is None:
            impulse_amplitude = self.params.amplitude * 10
        
        # Generate impulse locations
        impulse_mask = self.rng.random(size) < impulse_rate
        
        # Generate impulse amplitudes
        impulse_amplitudes = self.rng.choice(
            [-impulse_amplitude, impulse_amplitude], 
            size=np.sum(impulse_mask)
        )
        
        # Create noise array
        noise = np.zeros(size)
        noise[impulse_mask] = impulse_amplitudes
        
        return noise

class DriftNoiseGenerator:
    """
    Generator for drift and bias noise.
    """
    
    def __init__(self, parameters: NoiseParameters):
        self.params = parameters
        self.rng = np.random.RandomState()
        
    def generate(self, size: int, time_array: np.ndarray = None) -> np.ndarray:
        """
        Generate drift noise.
        
        Args:
            size (int): Number of samples
            time_array (np.ndarray, optional): Time array for drift calculation
            
        Returns:
            np.ndarray: Generated drift noise
        """
        if time_array is None:
            time_array = np.arange(size)
        
        # Linear drift
        linear_drift = self.params.drift_rate * time_array
        
        # Random walk component
        random_walk = np.cumsum(self.rng.normal(0, self.params.amplitude/10, size))
        
        # Sinusoidal drift (thermal cycles, etc.)
        sinusoidal_drift = 0.1 * self.params.amplitude * np.sin(
            2 * np.pi * time_array / (size / 4)
        )
        
        return linear_drift + random_walk + sinusoidal_drift

class QuantizationNoiseGenerator:
    """
    Generator for quantization noise from ADC.
    """
    
    def __init__(self, parameters: NoiseParameters):
        self.params = parameters
        
    def generate(self, signal: np.ndarray, signal_range: float) -> np.ndarray:
        """
        Generate quantization noise by quantizing the signal.
        
        Args:
            signal (np.ndarray): Input signal
            signal_range (float): Full scale range of the signal
            
        Returns:
            np.ndarray: Quantized signal with quantization noise
        """
        # Calculate quantization step
        num_levels = 2 ** self.params.quantization_bits
        q_step = signal_range / num_levels
        
        # Quantize signal
        quantized = np.round(signal / q_step) * q_step
        
        # Quantization noise is the difference
        return quantized - signal

class OutlierGenerator:
    """
    Generator for outliers and anomalous data points.
    """
    
    def __init__(self, parameters: NoiseParameters):
        self.params = parameters
        self.rng = np.random.RandomState()
        
    def generate(self, size: int, signal: np.ndarray = None) -> np.ndarray:
        """
        Generate outliers.
        
        Args:
            size (int): Number of samples
            signal (np.ndarray, optional): Original signal for outlier scaling
            
        Returns:
            np.ndarray: Outlier noise array
        """
        # Generate outlier locations
        outlier_mask = self.rng.random(size) < self.params.outlier_probability
        
        # Generate outlier values
        if signal is not None:
            signal_std = np.std(signal)
            outlier_magnitude = self.params.outlier_magnitude * signal_std
        else:
            outlier_magnitude = self.params.outlier_magnitude
        
        outlier_values = self.rng.normal(0, outlier_magnitude, np.sum(outlier_mask))
        
        # Create outlier array
        outliers = np.zeros(size)
        outliers[outlier_mask] = outlier_values
        
        return outliers

class SensorNoiseModel:
    """
    Comprehensive sensor noise model combining multiple noise sources.
    """
    
    def __init__(self, sensor_characteristics: SensorCharacteristics):
        self.sensor_char = sensor_characteristics
        self.noise_generators = {}
        
        # Initialize noise generators based on sensor type
        self._initialize_noise_generators()
        
    def _initialize_noise_generators(self) -> None:
        """Initialize appropriate noise generators for the sensor type."""
        base_params = NoiseParameters(amplitude=self.sensor_char.base_noise_level)
        
        # Gaussian thermal noise (always present)
        self.noise_generators['gaussian'] = GaussianNoiseGenerator(base_params)
        
        # Sensor-specific noise characteristics
        if self.sensor_char.sensor_type == SensorType.VOLTAGE:
            # Voltage sensors: 1/f noise, quantization
            colored_params = NoiseParameters(
                noise_type=NoiseType.COLORED,
                amplitude=base_params.amplitude * 0.1
            )
            self.noise_generators['colored'] = ColoredNoiseGenerator(colored_params)
            
            quant_params = NoiseParameters(
                noise_type=NoiseType.QUANTIZATION,
                quantization_bits=16
            )
            self.noise_generators['quantization'] = QuantizationNoiseGenerator(quant_params)
            
        elif self.sensor_char.sensor_type == SensorType.CURRENT:
            # Current sensors: shot noise (Poisson), drift
            drift_params = NoiseParameters(
                noise_type=NoiseType.DRIFT,
                drift_rate=0.001,
                amplitude=base_params.amplitude * 0.05
            )
            self.noise_generators['drift'] = DriftNoiseGenerator(drift_params)
            
        elif self.sensor_char.sensor_type == SensorType.TEMPERATURE:
            # Temperature sensors: slow drift, low frequency noise
            colored_params = NoiseParameters(
                noise_type=NoiseType.COLORED,
                amplitude=base_params.amplitude * 0.2,
                frequency_range=(0.001, 1.0)
            )
            self.noise_generators['colored'] = ColoredNoiseGenerator(colored_params)
            
            drift_params = NoiseParameters(
                noise_type=NoiseType.DRIFT,
                drift_rate=0.01,
                amplitude=base_params.amplitude * 0.1
            )
            self.noise_generators['drift'] = DriftNoiseGenerator(drift_params)
            
        # Impulse noise for all sensors (electromagnetic interference)
        impulse_params = NoiseParameters(
            noise_type=NoiseType.IMPULSE,
            amplitude=base_params.amplitude * 20,
            outlier_probability=0.001
        )
        self.noise_generators['impulse'] = ImpulseNoiseGenerator(impulse_params)
        
        # Outliers for all sensors
        outlier_params = NoiseParameters(
            noise_type=NoiseType.OUTLIERS,
            outlier_probability=0.005,
            outlier_magnitude=5.0
        )
        self.noise_generators['outliers'] = OutlierGenerator(outlier_params)
    
    def add_noise(self, signal: np.ndarray, sampling_rate: float = 1.0,
                  temperature: float = 25.0, age_factor: float = 1.0,
                  time_array: np.ndarray = None) -> np.ndarray:
        """
        Add comprehensive noise to a signal.
        
        Args:
            signal (np.ndarray): Clean signal
            sampling_rate (float): Sampling rate in Hz
            temperature (float): Temperature for temperature-dependent effects
            age_factor (float): Aging factor for sensor degradation
            time_array (np.ndarray, optional): Time array for drift calculation
            
        Returns:
            np.ndarray: Noisy signal
        """
        noisy_signal = signal.copy()
        size = len(signal)
        
        if time_array is None:
            time_array = np.arange(size) / sampling_rate
        
        # Add Gaussian noise
        if 'gaussian' in self.noise_generators:
            gaussian_noise = self.noise_generators['gaussian'].generate(
                size, np.mean(np.abs(signal)), temperature, age_factor
            )
            noisy_signal += gaussian_noise
        
        # Add colored noise
        if 'colored' in self.noise_generators:
            colored_noise = self.noise_generators['colored'].generate(
                size, sampling_rate, color_exponent=-1.0
            )
            noisy_signal += colored_noise
        
        # Add drift
        if 'drift' in self.noise_generators:
            drift_noise = self.noise_generators['drift'].generate(size, time_array)
            noisy_signal += drift_noise
        
        # Add impulse noise
        if 'impulse' in self.noise_generators:
            impulse_noise = self.noise_generators['impulse'].generate(size)
            noisy_signal += impulse_noise
        
        # Add outliers
        if 'outliers' in self.noise_generators:
            outlier_noise = self.noise_generators['outliers'].generate(size, signal)
            noisy_signal += outlier_noise
        
        # Apply quantization noise
        if 'quantization' in self.noise_generators:
            signal_range = self.sensor_char.range_max - self.sensor_char.range_min
            quantization_noise = self.noise_generators['quantization'].generate(
                noisy_signal, signal_range
            )
            noisy_signal += quantization_noise
        
        # Apply sensor nonlinearity
        if self.sensor_char.nonlinearity > 0:
            nonlinear_factor = self.sensor_char.nonlinearity
            noisy_signal += nonlinear_factor * noisy_signal**2
        
        # Apply temperature drift
        temp_drift = self.sensor_char.temperature_drift * (temperature - 25.0)
        noisy_signal += temp_drift * np.mean(np.abs(signal))
        
        return noisy_signal

class CorrelatedNoiseGenerator:
    """
    Generator for correlated noise between multiple sensors.
    """
    
    def __init__(self, correlation_matrix: np.ndarray):
        self.correlation_matrix = correlation_matrix
        self.num_sensors = correlation_matrix.shape[0]
        self.rng = np.random.RandomState()
        
        # Compute Cholesky decomposition for correlated noise generation
        try:
            self.cholesky = np.linalg.cholesky(correlation_matrix)
        except np.linalg.LinAlgError:
            # If matrix is not positive definite, use eigenvalue decomposition
            eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
            eigenvals = np.maximum(eigenvals, 1e-10)  # Ensure positive
            self.cholesky = eigenvecs @ np.diag(np.sqrt(eigenvals))
    
    def generate_correlated_noise(self, size: int, 
                                 noise_amplitudes: List[float]) -> np.ndarray:
        """
        Generate correlated noise for multiple sensors.
        
        Args:
            size (int): Number of samples
            noise_amplitudes (List[float]): Noise amplitudes for each sensor
            
        Returns:
            np.ndarray: Correlated noise array (size x num_sensors)
        """
        # Generate independent white noise
        independent_noise = self.rng.normal(0, 1, (size, self.num_sensors))
        
        # Apply correlation
        correlated_noise = independent_noise @ self.cholesky.T
        
        # Scale by individual amplitudes
        for i, amplitude in enumerate(noise_amplitudes):
            correlated_noise[:, i] *= amplitude
        
        return correlated_noise

class MissingDataGenerator:
    """
    Generator for missing data patterns.
    """
    
    def __init__(self, missing_rate: float = 0.05):
        self.missing_rate = missing_rate
        self.rng = np.random.RandomState()
        
    def generate_missing_mask(self, size: int, pattern: str = "random") -> np.ndarray:
        """
        Generate missing data mask.
        
        Args:
            size (int): Number of samples
            pattern (str): Missing data pattern ("random", "burst", "periodic")
            
        Returns:
            np.ndarray: Boolean mask (True = missing)
        """
        if pattern == "random":
            return self.rng.random(size) < self.missing_rate
            
        elif pattern == "burst":
            # Missing data in bursts
            mask = np.zeros(size, dtype=bool)
            num_bursts = int(size * self.missing_rate / 10)  # Average burst length = 10
            
            for _ in range(num_bursts):
                start = self.rng.randint(0, size - 10)
                length = self.rng.randint(5, 15)
                end = min(start + length, size)
                mask[start:end] = True
                
            return mask
            
        elif pattern == "periodic":
            # Periodic missing data
            period = int(1 / self.missing_rate)
            mask = np.zeros(size, dtype=bool)
            mask[::period] = True
            return mask
            
        else:
            return np.zeros(size, dtype=bool)

# Factory functions for common sensor types
def create_voltage_sensor_model() -> SensorNoiseModel:
    """Create noise model for voltage sensor."""
    characteristics = SensorCharacteristics(
        sensor_type=SensorType.VOLTAGE,
        base_noise_level=0.001,  # 1 mV
        resolution=0.0001,       # 0.1 mV
        range_min=0.0,
        range_max=5.0,
        temperature_drift=0.0001,
        nonlinearity=0.00001
    )
    return SensorNoiseModel(characteristics)

def create_current_sensor_model() -> SensorNoiseModel:
    """Create noise model for current sensor."""
    characteristics = SensorCharacteristics(
        sensor_type=SensorType.CURRENT,
        base_noise_level=0.01,   # 10 mA
        resolution=0.001,        # 1 mA
        range_min=-100.0,
        range_max=100.0,
        temperature_drift=0.001,
        nonlinearity=0.0001
    )
    return SensorNoiseModel(characteristics)

def create_temperature_sensor_model() -> SensorNoiseModel:
    """Create noise model for temperature sensor."""
    characteristics = SensorCharacteristics(
        sensor_type=SensorType.TEMPERATURE,
        base_noise_level=0.1,    # 0.1°C
        resolution=0.01,         # 0.01°C
        range_min=-40.0,
        range_max=85.0,
        temperature_drift=0.001,
        response_time=1.0
    )
    return SensorNoiseModel(characteristics)

def add_realistic_sensor_noise(data: pd.DataFrame, 
                             sensor_configs: Dict[str, SensorCharacteristics] = None,
                             sampling_rate: float = 1.0) -> pd.DataFrame:
    """
    Add realistic sensor noise to a DataFrame of battery data.
    
    Args:
        data (pd.DataFrame): Clean battery data
        sensor_configs (Dict[str, SensorCharacteristics], optional): Sensor configurations
        sampling_rate (float): Sampling rate in Hz
        
    Returns:
        pd.DataFrame: Data with realistic sensor noise
    """
    noisy_data = data.copy()
    
    # Default sensor configurations
    if sensor_configs is None:
        sensor_configs = {
            'voltage': create_voltage_sensor_model().sensor_char,
            'current': create_current_sensor_model().sensor_char,
            'temperature': create_temperature_sensor_model().sensor_char
        }
    
    # Add noise to each sensor column
    for column in data.columns:
        if 'voltage' in column.lower():
            sensor_model = SensorNoiseModel(sensor_configs.get('voltage'))
            noisy_data[column] = sensor_model.add_noise(
                data[column].values, sampling_rate
            )
        elif 'current' in column.lower():
            sensor_model = SensorNoiseModel(sensor_configs.get('current'))
            noisy_data[column] = sensor_model.add_noise(
                data[column].values, sampling_rate
            )
        elif 'temperature' in column.lower():
            sensor_model = SensorNoiseModel(sensor_configs.get('temperature'))
            noisy_data[column] = sensor_model.add_noise(
                data[column].values, sampling_rate
            )
    
    return noisy_data
