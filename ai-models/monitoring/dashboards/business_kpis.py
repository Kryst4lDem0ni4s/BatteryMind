"""
BatteryMind - Business KPIs Dashboard
Advanced business intelligence and key performance indicators system for battery
management with ROI tracking, sustainability metrics, and strategic insights.

Features:
- Comprehensive business KPI tracking and visualization
- ROI calculation and financial impact analysis
- Sustainability metrics and ESG reporting
- Strategic performance indicators and benchmarking
- Predictive business analytics and forecasting
- Cost optimization and efficiency metrics
- Circular economy value tracking

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import threading
import time
import json
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import uuid
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class KPICategory(Enum):
    """Categories of business KPIs."""
    FINANCIAL = "financial"
    OPERATIONAL = "operational"
    SUSTAINABILITY = "sustainability"
    CUSTOMER = "customer"
    INNOVATION = "innovation"
    RISK = "risk"
    EFFICIENCY = "efficiency"
    GROWTH = "growth"

class KPIFrequency(Enum):
    """KPI calculation frequencies."""
    REAL_TIME = "real_time"
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"

class TrendDirection(Enum):
    """KPI trend directions."""
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    VOLATILE = "volatile"
    UNKNOWN = "unknown"

class PerformanceStatus(Enum):
    """KPI performance status."""
    EXCELLENT = "excellent"
    GOOD = "good"
    AVERAGE = "average"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class KPIThreshold:
    """KPI threshold configuration."""
    
    excellent: float
    good: float
    average: float
    poor: float
    unit: str = ""
    higher_is_better: bool = True
    
    def evaluate_performance(self, value: float) -> PerformanceStatus:
        """Evaluate performance status based on thresholds."""
        try:
            if self.higher_is_better:
                if value >= self.excellent:
                    return PerformanceStatus.EXCELLENT
                elif value >= self.good:
                    return PerformanceStatus.GOOD
                elif value >= self.average:
                    return PerformanceStatus.AVERAGE
                elif value >= self.poor:
                    return PerformanceStatus.POOR
                else:
                    return PerformanceStatus.CRITICAL
            else:
                if value <= self.excellent:
                    return PerformanceStatus.EXCELLENT
                elif value <= self.good:
                    return PerformanceStatus.GOOD
                elif value <= self.average:
                    return PerformanceStatus.AVERAGE
                elif value <= self.poor:
                    return PerformanceStatus.POOR
                else:
                    return PerformanceStatus.CRITICAL
                    
        except Exception as e:
            logger.error(f"Error evaluating performance: {e}")
            return PerformanceStatus.UNKNOWN

@dataclass
class KPIDefinition:
    """Definition of a business KPI."""
    
    # Basic properties
    kpi_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    category: KPICategory = KPICategory.OPERATIONAL
    frequency: KPIFrequency = KPIFrequency.DAILY
    
    # Calculation properties
    calculation_formula: str = ""
    data_sources: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    
    # Display properties
    unit: str = ""
    display_format: str = "{:.2f}"
    chart_type: str = "line"
    color_scheme: str = "blue"
    
    # Performance tracking
    thresholds: Optional[KPIThreshold] = None
    target_value: Optional[float] = None
    benchmark_value: Optional[float] = None
    
    # Business context
    business_driver: str = ""
    owner: str = ""
    stakeholders: List[str] = field(default_factory=list)
    
    # Technical properties
    enabled: bool = True
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        # Convert datetime objects to ISO strings
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
        return data

@dataclass
class KPIDataPoint:
    """A single KPI data point."""
    
    kpi_id: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    period: str = ""  # e.g., "2024-01", "Q1-2024"
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 1.0  # Data quality indicator
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'kpi_id': self.kpi_id,
            'value': self.value,
            'timestamp': self.timestamp.isoformat(),
            'period': self.period,
            'metadata': self.metadata,
            'quality_score': self.quality_score,
            'confidence_interval': self.confidence_interval
        }

@dataclass
class KPIAnalysis:
    """Analysis results for a KPI."""
    
    kpi_id: str
    current_value: float
    previous_value: Optional[float] = None
    change_absolute: Optional[float] = None
    change_percentage: Optional[float] = None
    trend_direction: TrendDirection = TrendDirection.UNKNOWN
    performance_status: PerformanceStatus = PerformanceStatus.AVERAGE
    
    # Statistical measures
    volatility: float = 0.0
    moving_average_7d: Optional[float] = None
    moving_average_30d: Optional[float] = None
    
    # Insights
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Forecasting
    predicted_value: Optional[float] = None
    prediction_confidence: Optional[float] = None
    
    analysis_timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        data = asdict(self)
        for key, value in data.items():
            if isinstance(value, datetime):
                data[key] = value.isoformat()
            elif isinstance(value, Enum):
                data[key] = value.value
        return data

class KPICalculator:
    """
    Calculates KPI values from various data sources.
    """
    
    def __init__(self):
        self.calculation_functions: Dict[str, callable] = {}
        self.data_cache: Dict[str, Any] = {}
        self.cache_lock = threading.RLock()
        
        # Register default calculation functions
        self._register_default_calculators()
        
        logger.info("KPI Calculator initialized")
    
    def _register_default_calculators(self):
        """Register default KPI calculation functions."""
        
        # Financial KPIs
        self.register_calculator("total_cost_savings", self._calculate_total_cost_savings)
        self.register_calculator("roi_percentage", self._calculate_roi_percentage)
        self.register_calculator("battery_capex_optimization", self._calculate_battery_capex_optimization)
        self.register_calculator("energy_cost_reduction", self._calculate_energy_cost_reduction)
        self.register_calculator("maintenance_cost_savings", self._calculate_maintenance_cost_savings)
        
        # Operational KPIs
        self.register_calculator("battery_life_extension", self._calculate_battery_life_extension)
        self.register_calculator("fleet_utilization_rate", self._calculate_fleet_utilization_rate)
        self.register_calculator("charging_efficiency", self._calculate_charging_efficiency)
        self.register_calculator("system_uptime", self._calculate_system_uptime)
        self.register_calculator("prediction_accuracy", self._calculate_prediction_accuracy)
        
        # Sustainability KPIs
        self.register_calculator("carbon_footprint_reduction", self._calculate_carbon_footprint_reduction)
        self.register_calculator("circular_economy_score", self._calculate_circular_economy_score)
        self.register_calculator("material_recovery_rate", self._calculate_material_recovery_rate)
        self.register_calculator("energy_efficiency_improvement", self._calculate_energy_efficiency_improvement)
        
        # Customer KPIs
        self.register_calculator("customer_satisfaction", self._calculate_customer_satisfaction)
        self.register_calculator("service_availability", self._calculate_service_availability)
        self.register_calculator("response_time", self._calculate_response_time)
        
    def register_calculator(self, kpi_name: str, calculator_func: callable):
        """Register a custom KPI calculator function."""
        self.calculation_functions[kpi_name] = calculator_func
        logger.info(f"Registered calculator for KPI: {kpi_name}")
    
    def calculate_kpi(self, kpi_definition: KPIDefinition, 
                     data_context: Dict[str, Any]) -> Optional[KPIDataPoint]:
        """
        Calculate KPI value based on definition and data context.
        
        Args:
            kpi_definition: KPI definition
            data_context: Data context for calculation
            
        Returns:
            Calculated KPI data point or None if calculation failed
        """
        try:
            calculator_func = self.calculation_functions.get(kpi_definition.name.lower().replace(' ', '_'))
            
            if not calculator_func:
                logger.warning(f"No calculator found for KPI: {kpi_definition.name}")
                return None
            
            # Calculate value
            result = calculator_func(data_context)
            
            if result is None:
                logger.warning(f"Calculator returned None for KPI: {kpi_definition.name}")
                return None
            
            # Create data point
            data_point = KPIDataPoint(
                kpi_id=kpi_definition.kpi_id,
                value=float(result),
                period=self._get_period_string(kpi_definition.frequency),
                metadata=data_context.get('metadata', {})
            )
            
            return data_point
            
        except Exception as e:
            logger.error(f"Error calculating KPI {kpi_definition.name}: {e}")
            return None
    
    def _get_period_string(self, frequency: KPIFrequency) -> str:
        """Get period string based on frequency."""
        current_time = datetime.now()
        
        if frequency == KPIFrequency.DAILY:
            return current_time.strftime("%Y-%m-%d")
        elif frequency == KPIFrequency.WEEKLY:
            return f"{current_time.year}-W{current_time.isocalendar()[1]:02d}"
        elif frequency == KPIFrequency.MONTHLY:
            return current_time.strftime("%Y-%m")
        elif frequency == KPIFrequency.QUARTERLY:
            quarter = (current_time.month - 1) // 3 + 1
            return f"{current_time.year}-Q{quarter}"
        elif frequency == KPIFrequency.ANNUALLY:
            return str(current_time.year)
        else:
            return current_time.isoformat()
    
    # Financial KPI Calculators
    def _calculate_total_cost_savings(self, data_context: Dict[str, Any]) -> float:
        """Calculate total cost savings from battery optimization."""
        try:
            battery_savings = data_context.get('battery_cost_savings', 0)
            energy_savings = data_context.get('energy_cost_savings', 0)
            maintenance_savings = data_context.get('maintenance_cost_savings', 0)
            operational_savings = data_context.get('operational_cost_savings', 0)
            
            total_savings = battery_savings + energy_savings + maintenance_savings + operational_savings
            return total_savings
            
        except Exception as e:
            logger.error(f"Error calculating total cost savings: {e}")
            return 0.0
    
    def _calculate_roi_percentage(self, data_context: Dict[str, Any]) -> float:
        """Calculate return on investment percentage."""
        try:
            total_savings = data_context.get('total_cost_savings', 0)
            total_investment = data_context.get('total_investment', 1)  # Avoid division by zero
            
            if total_investment == 0:
                return 0.0
            
            roi = ((total_savings - total_investment) / total_investment) * 100
            return roi
            
        except Exception as e:
            logger.error(f"Error calculating ROI percentage: {e}")
            return 0.0
    
    def _calculate_battery_capex_optimization(self, data_context: Dict[str, Any]) -> float:
        """Calculate battery capital expenditure optimization."""
        try:
            original_capex = data_context.get('original_battery_capex', 0)
            optimized_capex = data_context.get('optimized_battery_capex', 0)
            
            if original_capex == 0:
                return 0.0
            
            optimization = ((original_capex - optimized_capex) / original_capex) * 100
            return optimization
            
        except Exception as e:
            logger.error(f"Error calculating battery CAPEX optimization: {e}")
            return 0.0
    
    def _calculate_energy_cost_reduction(self, data_context: Dict[str, Any]) -> float:
        """Calculate energy cost reduction percentage."""
        try:
            baseline_energy_cost = data_context.get('baseline_energy_cost', 0)
            current_energy_cost = data_context.get('current_energy_cost', 0)
            
            if baseline_energy_cost == 0:
                return 0.0
            
            reduction = ((baseline_energy_cost - current_energy_cost) / baseline_energy_cost) * 100
            return reduction
            
        except Exception as e:
            logger.error(f"Error calculating energy cost reduction: {e}")
            return 0.0
    
    def _calculate_maintenance_cost_savings(self, data_context: Dict[str, Any]) -> float:
        """Calculate maintenance cost savings."""
        try:
            baseline_maintenance_cost = data_context.get('baseline_maintenance_cost', 0)
            current_maintenance_cost = data_context.get('current_maintenance_cost', 0)
            
            savings = baseline_maintenance_cost - current_maintenance_cost
            return max(0, savings)  # Ensure non-negative
            
        except Exception as e:
            logger.error(f"Error calculating maintenance cost savings: {e}")
            return 0.0
    
    # Operational KPI Calculators
    def _calculate_battery_life_extension(self, data_context: Dict[str, Any]) -> float:
        """Calculate battery life extension percentage."""
        try:
            baseline_life_years = data_context.get('baseline_battery_life_years', 0)
            extended_life_years = data_context.get('extended_battery_life_years', 0)
            
            if baseline_life_years == 0:
                return 0.0
            
            extension = ((extended_life_years - baseline_life_years) / baseline_life_years) * 100
            return extension
            
        except Exception as e:
            logger.error(f"Error calculating battery life extension: {e}")
            return 0.0
    
    def _calculate_fleet_utilization_rate(self, data_context: Dict[str, Any]) -> float:
        """Calculate fleet utilization rate."""
        try:
            active_vehicles = data_context.get('active_vehicles', 0)
            total_vehicles = data_context.get('total_vehicles', 1)
            
            utilization_rate = (active_vehicles / total_vehicles) * 100
            return utilization_rate
            
        except Exception as e:
            logger.error(f"Error calculating fleet utilization rate: {e}")
            return 0.0
    
    def _calculate_charging_efficiency(self, data_context: Dict[str, Any]) -> float:
        """Calculate overall charging efficiency."""
        try:
            energy_delivered = data_context.get('energy_delivered_kwh', 0)
            energy_consumed = data_context.get('energy_consumed_kwh', 1)
            
            efficiency = (energy_delivered / energy_consumed) * 100
            return min(100, efficiency)  # Cap at 100%
            
        except Exception as e:
            logger.error(f"Error calculating charging efficiency: {e}")
            return 0.0
    
    def _calculate_system_uptime(self, data_context: Dict[str, Any]) -> float:
        """Calculate system uptime percentage."""
        try:
            total_time_hours = data_context.get('total_time_hours', 1)
            downtime_hours = data_context.get('downtime_hours', 0)
            
            uptime_hours = total_time_hours - downtime_hours
            uptime_percentage = (uptime_hours / total_time_hours) * 100
            return max(0, uptime_percentage)
            
        except Exception as e:
            logger.error(f"Error calculating system uptime: {e}")
            return 0.0
    
    def _calculate_prediction_accuracy(self, data_context: Dict[str, Any]) -> float:
        """Calculate AI model prediction accuracy."""
        try:
            correct_predictions = data_context.get('correct_predictions', 0)
            total_predictions = data_context.get('total_predictions', 1)
            
            accuracy = (correct_predictions / total_predictions) * 100
            return accuracy
            
        except Exception as e:
            logger.error(f"Error calculating prediction accuracy: {e}")
            return 0.0
    
    # Sustainability KPI Calculators
    def _calculate_carbon_footprint_reduction(self, data_context: Dict[str, Any]) -> float:
        """Calculate carbon footprint reduction."""
        try:
            baseline_co2_kg = data_context.get('baseline_co2_emissions_kg', 0)
            current_co2_kg = data_context.get('current_co2_emissions_kg', 0)
            
            if baseline_co2_kg == 0:
                return 0.0
            
            reduction = ((baseline_co2_kg - current_co2_kg) / baseline_co2_kg) * 100
            return reduction
            
        except Exception as e:
            logger.error(f"Error calculating carbon footprint reduction: {e}")
            return 0.0
    
    def _calculate_circular_economy_score(self, data_context: Dict[str, Any]) -> float:
        """Calculate circular economy score."""
        try:
            reuse_rate = data_context.get('battery_reuse_rate', 0)
            recycling_rate = data_context.get('material_recycling_rate', 0)
            life_extension_rate = data_context.get('life_extension_rate', 0)
            
            # Weighted average of circular economy factors
            circular_score = (reuse_rate * 0.4 + recycling_rate * 0.3 + life_extension_rate * 0.3)
            return circular_score
            
        except Exception as e:
            logger.error(f"Error calculating circular economy score: {e}")
            return 0.0
    
    def _calculate_material_recovery_rate(self, data_context: Dict[str, Any]) -> float:
        """Calculate material recovery rate from recycling."""
        try:
            recovered_materials_kg = data_context.get('recovered_materials_kg', 0)
            total_materials_kg = data_context.get('total_materials_kg', 1)
            
            recovery_rate = (recovered_materials_kg / total_materials_kg) * 100
            return recovery_rate
            
        except Exception as e:
            logger.error(f"Error calculating material recovery rate: {e}")
            return 0.0
    
    def _calculate_energy_efficiency_improvement(self, data_context: Dict[str, Any]) -> float:
        """Calculate energy efficiency improvement."""
        try:
            baseline_efficiency = data_context.get('baseline_energy_efficiency', 0)
            current_efficiency = data_context.get('current_energy_efficiency', 0)
            
            if baseline_efficiency == 0:
                return 0.0
            
            improvement = ((current_efficiency - baseline_efficiency) / baseline_efficiency) * 100
            return improvement
            
        except Exception as e:
            logger.error(f"Error calculating energy efficiency improvement: {e}")
            return 0.0
    
    # Customer KPI Calculators
    def _calculate_customer_satisfaction(self, data_context: Dict[str, Any]) -> float:
        """Calculate customer satisfaction score."""
        try:
            satisfaction_scores = data_context.get('satisfaction_scores', [])
            
            if not satisfaction_scores:
                return 0.0
            
            average_satisfaction = sum(satisfaction_scores) / len(satisfaction_scores)
            return average_satisfaction
            
        except Exception as e:
            logger.error(f"Error calculating customer satisfaction: {e}")
            return 0.0
    
    def _calculate_service_availability(self, data_context: Dict[str, Any]) -> float:
        """Calculate service availability percentage."""
        try:
            total_service_time = data_context.get('total_service_time_hours', 1)
            service_downtime = data_context.get('service_downtime_hours', 0)
            
            availability = ((total_service_time - service_downtime) / total_service_time) * 100
            return availability
            
        except Exception as e:
            logger.error(f"Error calculating service availability: {e}")
            return 0.0
    
    def _calculate_response_time(self, data_context: Dict[str, Any]) -> float:
        """Calculate average response time."""
        try:
            response_times = data_context.get('response_times_seconds', [])
            
            if not response_times:
                return 0.0
            
            average_response_time = sum(response_times) / len(response_times)
            return average_response_time
            
        except Exception as e:
            logger.error(f"Error calculating response time: {e}")
            return 0.0

class PerformanceIndicator:
    """
    Manages individual performance indicators with trend analysis.
    """
    
    def __init__(self, kpi_definition: KPIDefinition):
        self.definition = kpi_definition
        self.data_points: deque = deque(maxlen=1000)  # Store last 1000 data points
        self.data_lock = threading.RLock()
        
        logger.info(f"Performance Indicator initialized for: {kpi_definition.name}")
    
    def add_data_point(self, data_point: KPIDataPoint):
        """Add a new data point to the indicator."""
        with self.data_lock:
            self.data_points.append(data_point)
    
    def get_latest_value(self) -> Optional[float]:
        """Get the latest KPI value."""
        with self.data_lock:
            if self.data_points:
                return self.data_points[-1].value
            return None
    
    def get_trend_analysis(self, lookback_periods: int = 30) -> KPIAnalysis:
        """Perform trend analysis on the KPI data."""
        try:
            with self.data_lock:
                if len(self.data_points) < 2:
                    return KPIAnalysis(
                        kpi_id=self.definition.kpi_id,
                        current_value=self.get_latest_value() or 0.0
                    )
                
                # Get recent data points
                recent_data = list(self.data_points)[-lookback_periods:]
                values = [dp.value for dp in recent_data]
                
                current_value = values[-1]
                previous_value = values[-2] if len(values) > 1 else None
                
                # Calculate changes
                change_absolute = None
                change_percentage = None
                if previous_value is not None:
                    change_absolute = current_value - previous_value
                    if previous_value != 0:
                        change_percentage = (change_absolute / previous_value) * 100
                
                # Calculate trend direction
                trend_direction = self._calculate_trend_direction(values)
                
                # Calculate volatility
                volatility = np.std(values) if len(values) > 1 else 0.0
                
                # Calculate moving averages
                moving_avg_7d = None
                moving_avg_30d = None
                
                if len(values) >= 7:
                    moving_avg_7d = np.mean(values[-7:])
                
                if len(values) >= 30:
                    moving_avg_30d = np.mean(values[-30:])
                
                # Evaluate performance status
                performance_status = PerformanceStatus.AVERAGE
                if self.definition.thresholds:
                    performance_status = self.definition.thresholds.evaluate_performance(current_value)
                
                # Generate insights and recommendations
                insights = self._generate_insights(values, trend_direction, performance_status)
                recommendations = self._generate_recommendations(performance_status, trend_direction)
                
                return KPIAnalysis(
                    kpi_id=self.definition.kpi_id,
                    current_value=current_value,
                    previous_value=previous_value,
                    change_absolute=change_absolute,
                    change_percentage=change_percentage,
                    trend_direction=trend_direction,
                    performance_status=performance_status,
                    volatility=volatility,
                    moving_average_7d=moving_avg_7d,
                    moving_average_30d=moving_avg_30d,
                    insights=insights,
                    recommendations=recommendations
                )
                
        except Exception as e:
            logger.error(f"Error performing trend analysis: {e}")
            return KPIAnalysis(
                kpi_id=self.definition.kpi_id,
                current_value=self.get_latest_value() or 0.0
            )
    
    def _calculate_trend_direction(self, values: List[float]) -> TrendDirection:
        """Calculate trend direction using linear regression."""
        try:
            if len(values) < 3:
                return TrendDirection.UNKNOWN
            
            # Simple linear regression slope
            x = np.arange(len(values))
            slope = np.polyfit(x, values, 1)[0]
            
            # Calculate relative slope (normalize by mean value)
            mean_value = np.mean(values)
            if mean_value != 0:
                relative_slope = abs(slope) / mean_value
            else:
                relative_slope = 0
            
            # Determine trend based on slope and volatility
            if relative_slope < 0.01:
                return TrendDirection.STABLE
            elif slope > 0:
                return TrendDirection.INCREASING
            else:
                return TrendDirection.DECREASING
                
        except Exception as e:
            logger.error(f"Error calculating trend direction: {e}")
            return TrendDirection.UNKNOWN
    
    def _generate_insights(self, values: List[float], trend: TrendDirection, 
                          performance: PerformanceStatus) -> List[str]:
        """Generate insights based on KPI analysis."""
        insights = []
        
        try:
            if performance == PerformanceStatus.EXCELLENT:
                insights.append(f"{self.definition.name} is performing excellently above target thresholds")
            elif performance == PerformanceStatus.CRITICAL:
                insights.append(f"{self.definition.name} requires immediate attention - below critical threshold")
            
            if trend == TrendDirection.INCREASING:
                if self.definition.thresholds and self.definition.thresholds.higher_is_better:
                    insights.append("Positive upward trend indicates improving performance")
                else:
                    insights.append("Upward trend may indicate emerging issues requiring attention")
            elif trend == TrendDirection.DECREASING:
                if self.definition.thresholds and not self.definition.thresholds.higher_is_better:
                    insights.append("Downward trend shows positive improvement")
                else:
                    insights.append("Declining trend suggests performance degradation")
            
            # Volatility insights
            if len(values) > 1:
                volatility = np.std(values) / (np.mean(values) + 0.001)  # Coefficient of variation
                if volatility > 0.2:
                    insights.append("High volatility detected - consider investigating root causes")
                elif volatility < 0.05:
                    insights.append("Stable performance with low volatility")
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
        
        return insights
    
    def _generate_recommendations(self, performance: PerformanceStatus, 
                                trend: TrendDirection) -> List[str]:
        """Generate recommendations based on performance and trend."""
        recommendations = []
        
        try:
            if performance == PerformanceStatus.CRITICAL:
                recommendations.append("Immediate intervention required to address critical performance")
                recommendations.append("Conduct root cause analysis to identify underlying issues")
                recommendations.append("Implement emergency response procedures if available")
            
            elif performance == PerformanceStatus.POOR:
                recommendations.append("Develop improvement plan to enhance performance")
                recommendations.append("Increase monitoring frequency to track progress")
            
            if trend == TrendDirection.DECREASING and self.definition.thresholds and self.definition.thresholds.higher_is_better:
                recommendations.append("Investigate factors contributing to declining performance")
                recommendations.append("Consider implementing corrective measures")
            
            elif trend == TrendDirection.INCREASING and self.definition.thresholds and not self.definition.thresholds.higher_is_better:
                recommendations.append("Monitor increasing trend and identify mitigation strategies")
            
            if performance == PerformanceStatus.EXCELLENT:
                recommendations.append("Document and replicate successful practices")
                recommendations.append("Consider raising performance targets for continuous improvement")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
        
        return recommendations

class ROICalculator:
    """
    Specialized calculator for Return on Investment analysis.
    """
    
    def __init__(self):
        self.investment_tracking: Dict[str, Dict[str, float]] = {}
        self.benefits_tracking: Dict[str, Dict[str, float]] = {}
        
        logger.info("ROI Calculator initialized")
    
    def track_investment(self, category: str, amount: float, description: str = ""):
        """Track an investment amount."""
        if category not in self.investment_tracking:
            self.investment_tracking[category] = {}
        
        investment_id = str(uuid.uuid4())
        self.investment_tracking[category][investment_id] = {
            'amount': amount,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Tracked investment: {category} - ${amount:,.2f}")
    
    def track_benefit(self, category: str, amount: float, description: str = ""):
        """Track a benefit/return amount."""
        if category not in self.benefits_tracking:
            self.benefits_tracking[category] = {}
        
        benefit_id = str(uuid.uuid4())
        self.benefits_tracking[category][benefit_id] = {
            'amount': amount,
            'description': description,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Tracked benefit: {category} - ${amount:,.2f}")
    
    def calculate_roi(self, time_period_months: int = 12) -> Dict[str, Any]:
        """Calculate comprehensive ROI analysis."""
        try:
            # Calculate total investments
            total_investment = 0
            investment_breakdown = {}
            
            for category, investments in self.investment_tracking.items():
                category_total = sum(inv['amount'] for inv in investments.values())
                investment_breakdown[category] = category_total
                total_investment += category_total
            
            # Calculate total benefits
            total_benefits = 0
            benefits_breakdown = {}
            
            for category, benefits in self.benefits_tracking.items():
                category_total = sum(ben['amount'] for ben in benefits.values())
                benefits_breakdown[category] = category_total
                total_benefits += category_total
            
            # Calculate ROI metrics
            net_benefit = total_benefits - total_investment
            roi_percentage = (net_benefit / total_investment * 100) if total_investment > 0 else 0
            
            # Annualized ROI
            annualized_roi = roi_percentage * (12 / time_period_months) if time_period_months > 0 else roi_percentage
            
            # Payback period (in months)
            monthly_benefit = total_benefits / time_period_months if time_period_months > 0 else 0
            payback_months = total_investment / monthly_benefit if monthly_benefit > 0 else float('inf')
            
            roi_analysis = {
                'total_investment': total_investment,
                'total_benefits': total_benefits,
                'net_benefit': net_benefit,
                'roi_percentage': roi_percentage,
                'annualized_roi': annualized_roi,
                'payback_period_months': payback_months,
                'investment_breakdown': investment_breakdown,
                'benefits_breakdown': benefits_breakdown,
                'analysis_date': datetime.now().isoformat(),
                'time_period_months': time_period_months
            }
            
            return roi_analysis
            
        except Exception as e:
            logger.error(f"Error calculating ROI: {e}")
            return {}

class SustainabilityMetrics:
    """
    Specialized metrics for sustainability and ESG reporting.
    """
    
    def __init__(self):
        self.sustainability_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        logger.info("Sustainability Metrics initialized")
    
    def track_carbon_emission(self, source: str, co2_kg: float, scope: int = 1):
        """Track carbon emissions by source and scope."""
        emission_record = {
            'source': source,
            'co2_kg': co2_kg,
            'scope': scope,
            'timestamp': datetime.now().isoformat()
        }
        
        self.sustainability_data['carbon_emissions'].append(emission_record)
        logger.info(f"Tracked carbon emission: {source} - {co2_kg:.2f} kg CO2")
    
    def track_energy_consumption(self, source: str, kwh: float, renewable_percentage: float = 0):
        """Track energy consumption with renewable energy percentage."""
        energy_record = {
            'source': source,
            'energy_kwh': kwh,
            'renewable_percentage': renewable_percentage,
            'timestamp': datetime.now().isoformat()
        }
        
        self.sustainability_data['energy_consumption'].append(energy_record)
        logger.info(f"Tracked energy consumption: {source} - {kwh:.2f} kWh ({renewable_percentage:.1f}% renewable)")
    
    def track_waste_generation(self, waste_type: str, amount_kg: float, recycled_percentage: float = 0):
        """Track waste generation and recycling rates."""
        waste_record = {
            'waste_type': waste_type,
            'amount_kg': amount_kg,
            'recycled_percentage': recycled_percentage,
            'timestamp': datetime.now().isoformat()
        }
        
        self.sustainability_data['waste_generation'].append(waste_record)
        logger.info(f"Tracked waste: {waste_type} - {amount_kg:.2f} kg ({recycled_percentage:.1f}% recycled)")
    
    def calculate_sustainability_score(self) -> Dict[str, Any]:
        """Calculate comprehensive sustainability score."""
        try:
            # Carbon intensity (lower is better)
            carbon_emissions = self.sustainability_data.get('carbon_emissions', [])
            total_co2 = sum(record['co2_kg'] for record in carbon_emissions)
            
            # Energy efficiency (higher renewable percentage is better)
            energy_consumption = self.sustainability_data.get('energy_consumption', [])
            total_energy = sum(record['energy_kwh'] for record in energy_consumption)
            renewable_energy = sum(
                record['energy_kwh'] * record['renewable_percentage'] / 100
                for record in energy_consumption
            )
            renewable_percentage = (renewable_energy / total_energy * 100) if total_energy > 0 else 0
            
            # Waste management (higher recycling percentage is better)
            waste_generation = self.sustainability_data.get('waste_generation', [])
            total_waste = sum(record['amount_kg'] for record in waste_generation)
            recycled_waste = sum(
                record['amount_kg'] * record['recycled_percentage'] / 100
                for record in waste_generation
            )
            recycling_percentage = (recycled_waste / total_waste * 100) if total_waste > 0 else 0
            
            # Calculate composite sustainability score (0-100)
            # Higher score is better
            carbon_score = max(0, 100 - (total_co2 / 1000))  # Penalize high emissions
            energy_score = renewable_percentage
            waste_score = recycling_percentage
            
            overall_score = (carbon_score * 0.4 + energy_score * 0.3 + waste_score * 0.3)
            
            sustainability_analysis = {
                'overall_score': overall_score,
                'carbon_footprint': {
                    'total_co2_kg': total_co2,
                    'score': carbon_score
                },
                'energy_efficiency': {
                    'total_energy_kwh': total_energy,
                    'renewable_percentage': renewable_percentage,
                    'score': energy_score
                },
                'waste_management': {
                    'total_waste_kg': total_waste,
                    'recycling_percentage': recycling_percentage,
                    'score': waste_score
                },
                'analysis_date': datetime.now().isoformat()
            }
            
            return sustainability_analysis
            
        except Exception as e:
            logger.error(f"Error calculating sustainability score: {e}")
            return {}

class KPITracker:
    """
    Main KPI tracking and dashboard system.
    """
    
    def __init__(self, 
                 calculation_interval_hours: int = 1,
                 retention_days: int = 365,
                 enable_forecasting: bool = True):
        
        self.calculation_interval_hours = calculation_interval_hours
        self.retention_days = retention_days
        self.enable_forecasting = enable_forecasting
        
        # Core components
        self.calculator = KPICalculator()
        self.roi_calculator = ROICalculator()
        self.sustainability_metrics = SustainabilityMetrics()
        
        # KPI management
        self.kpi_definitions: Dict[str, KPIDefinition] = {}
        self.performance_indicators: Dict[str, PerformanceIndicator] = {}
        
        # Processing
        self.is_running = False
        self.processing_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.tracking_stats = {
            'total_kpis': 0,
            'calculations_performed': 0,
            'last_calculation_time': None,
            'data_points_stored': 0
        }
        
        # Load default KPIs
        self._initialize_default_kpis()
        
        logger.info("KPI Tracker initialized")
    
    def _initialize_default_kpis(self):
        """Initialize default business KPIs for BatteryMind."""
        
        default_kpis = [
            # Financial KPIs
            KPIDefinition(
                name="total_cost_savings",
                description="Total cost savings achieved through battery optimization",
                category=KPICategory.FINANCIAL,
                frequency=KPIFrequency.MONTHLY,
                unit="USD",
                thresholds=KPIThreshold(
                    excellent=500000, good=300000, average=150000, poor=50000,
                    unit="USD", higher_is_better=True
                ),
                business_driver="Cost Optimization",
                owner="CFO"
            ),
            
            KPIDefinition(
                name="roi_percentage",
                description="Return on Investment percentage",
                category=KPICategory.FINANCIAL,
                frequency=KPIFrequency.QUARTERLY,
                unit="%",
                thresholds=KPIThreshold(
                    excellent=25, good=15, average=8, poor=3,
                    unit="%", higher_is_better=True
                ),
                business_driver="Investment Returns",
                owner="CFO"
            ),
            
            # Operational KPIs
            KPIDefinition(
                name="battery_life_extension",
                description="Battery life extension achieved through optimization",
                category=KPICategory.OPERATIONAL,
                frequency=KPIFrequency.MONTHLY,
                unit="%",
                thresholds=KPIThreshold(
                    excellent=50, good=30, average=15, poor=5,
                    unit="%", higher_is_better=True
                ),
                business_driver="Asset Optimization",
                owner="CTO"
            ),
            
            KPIDefinition(
                name="fleet_utilization_rate",
                description="Fleet utilization rate across all vehicles",
                category=KPICategory.OPERATIONAL,
                frequency=KPIFrequency.DAILY,
                unit="%",
                thresholds=KPIThreshold(
                    excellent=85, good=75, average=65, poor=50,
                    unit="%", higher_is_better=True
                ),
                business_driver="Operational Efficiency",
                owner="COO"
            ),
            
            KPIDefinition(
                name="prediction_accuracy",
                description="AI model prediction accuracy",
                category=KPICategory.OPERATIONAL,
                frequency=KPIFrequency.DAILY,
                unit="%",
                thresholds=KPIThreshold(
                    excellent=97, good=94, average=90, poor=85,
                    unit="%", higher_is_better=True
                ),
                business_driver="AI Performance",
                owner="CTO"
            ),
            
            # Sustainability KPIs
            KPIDefinition(
                name="carbon_footprint_reduction",
                description="Carbon footprint reduction achieved",
                category=KPICategory.SUSTAINABILITY,
                frequency=KPIFrequency.MONTHLY,
                unit="%",
                thresholds=KPIThreshold(
                    excellent=40, good=25, average=15, poor=5,
                    unit="%", higher_is_better=True
                ),
                business_driver="Environmental Impact",
                owner="Chief Sustainability Officer"
            ),
            
            KPIDefinition(
                name="circular_economy_score",
                description="Circular economy implementation score",
                category=KPICategory.SUSTAINABILITY,
                frequency=KPIFrequency.QUARTERLY,
                unit="Score",
                thresholds=KPIThreshold(
                    excellent=85, good=70, average=55, poor=40,
                    unit="Score", higher_is_better=True
                ),
                business_driver="Sustainability",
                owner="Chief Sustainability Officer"
            ),
            
            # Customer KPIs
            KPIDefinition(
                name="customer_satisfaction",
                description="Customer satisfaction score",
                category=KPICategory.CUSTOMER,
                frequency=KPIFrequency.MONTHLY,
                unit="Score",
                thresholds=KPIThreshold(
                    excellent=4.5, good=4.0, average=3.5, poor=3.0,
                    unit="Score", higher_is_better=True
                ),
                business_driver="Customer Experience",
                owner="Head of Customer Success"
            ),
            
            KPIDefinition(
                name="system_uptime",
                description="System availability and uptime",
                category=KPICategory.OPERATIONAL,
                frequency=KPIFrequency.DAILY,
                unit="%",
                thresholds=KPIThreshold(
                    excellent=99.9, good=99.5, average=99.0, poor=98.0,
                    unit="%", higher_is_better=True
                ),
                business_driver="Service Reliability",
                owner="CTO"
            )
        ]
        
        # Register default KPIs
        for kpi_def in default_kpis:
            self.register_kpi(kpi_def)
    
    def register_kpi(self, kpi_definition: KPIDefinition) -> bool:
        """Register a new KPI definition."""
        try:
            self.kpi_definitions[kpi_definition.kpi_id] = kpi_definition
            self.performance_indicators[kpi_definition.kpi_id] = PerformanceIndicator(kpi_definition)
            
            self.tracking_stats['total_kpis'] += 1
            
            logger.info(f"Registered KPI: {kpi_definition.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering KPI: {e}")
            return False
    
    def start_tracking(self):
        """Start KPI tracking and calculation."""
        if self.is_running:
            logger.warning("KPI tracking already running")
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        self.processing_thread = threading.Thread(
            target=self._tracking_loop,
            name="KPITracker",
            daemon=True
        )
        self.processing_thread.start()
        
        logger.info("KPI tracking started")
    
    def stop_tracking(self, timeout: int = 30):
        """Stop KPI tracking."""
        if not self.is_running:
            return
        
        logger.info("Stopping KPI tracking...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=timeout)
        
        logger.info("KPI tracking stopped")
    
    def _tracking_loop(self):
        """Main KPI tracking loop."""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                self._calculate_all_kpis()
                
                # Wait for next calculation interval
                self.shutdown_event.wait(self.calculation_interval_hours * 3600)
                
            except Exception as e:
                logger.error(f"Error in KPI tracking loop: {e}")
                time.sleep(300)  # Sleep 5 minutes on error
    
    def _calculate_all_kpis(self):
        """Calculate all registered KPIs."""
        try:
            # Mock data context for demonstration
            # In production, this would come from various data sources
            data_context = self._get_mock_data_context()
            
            for kpi_id, kpi_definition in self.kpi_definitions.items():
                try:
                    data_point = self.calculator.calculate_kpi(kpi_definition, data_context)
                    
                    if data_point:
                        performance_indicator = self.performance_indicators[kpi_id]
                        performance_indicator.add_data_point(data_point)
                        
                        self.tracking_stats['calculations_performed'] += 1
                        self.tracking_stats['data_points_stored'] += 1
                    
                except Exception as e:
                    logger.error(f"Error calculating KPI {kpi_definition.name}: {e}")
            
            self.tracking_stats['last_calculation_time'] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in calculate_all_kpis: {e}")
    
    def _get_mock_data_context(self) -> Dict[str, Any]:
        """Get mock data context for KPI calculations."""
        # This is mock data for demonstration
        # In production, this would be fetched from actual data sources
        return {
            'battery_cost_savings': 150000,
            'energy_cost_savings': 75000,
            'maintenance_cost_savings': 50000,
            'operational_cost_savings': 25000,
            'total_investment': 200000,
            'baseline_battery_life_years': 8,
            'extended_battery_life_years': 12,
            'active_vehicles': 85,
            'total_vehicles': 100,
            'correct_predictions': 1850,
            'total_predictions': 2000,
            'baseline_co2_emissions_kg': 10000,
            'current_co2_emissions_kg': 6500,
            'battery_reuse_rate': 75,
            'material_recycling_rate': 85,
            'life_extension_rate': 45,
            'satisfaction_scores': [4.2, 4.5, 4.1, 4.3, 4.4],
            'total_service_time_hours': 720,
            'service_downtime_hours': 8,
            'energy_delivered_kwh': 950,
            'energy_consumed_kwh': 1000,
            'total_time_hours': 720,
            'downtime_hours': 3
        }
    
    def get_kpi_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive KPI dashboard data."""
        try:
            dashboard_data = {
                'summary_stats': self.tracking_stats.copy(),
                'kpis': {},
                'roi_analysis': self.roi_calculator.calculate_roi(),
                'sustainability_metrics': self.sustainability_metrics.calculate_sustainability_score(),
                'generated_at': datetime.now().isoformat()
            }
            
            # Get analysis for each KPI
            for kpi_id, performance_indicator in self.performance_indicators.items():
                kpi_definition = self.kpi_definitions[kpi_id]
                analysis = performance_indicator.get_trend_analysis()
                
                dashboard_data['kpis'][kpi_id] = {
                    'definition': kpi_definition.to_dict(),
                    'analysis': analysis.to_dict(),
                    'latest_value': performance_indicator.get_latest_value()
                }
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error generating KPI dashboard: {e}")
            return {}
    
    def get_business_impact_report(self) -> Dict[str, Any]:
        """Generate comprehensive business impact report."""
        try:
            dashboard = self.get_kpi_dashboard()
            
            # Extract key insights
            financial_impact = {
                'total_cost_savings': 0,
                'roi_percentage': 0,
                'payback_period_months': 0
            }
            
            operational_impact = {
                'battery_life_extension': 0,
                'fleet_utilization': 0,
                'system_uptime': 0
            }
            
            sustainability_impact = {
                'carbon_reduction': 0,
                'circular_economy_score': 0
            }
            
            # Extract values from KPIs
            for kpi_id, kpi_data in dashboard.get('kpis', {}).items():
                kpi_name = kpi_data['definition']['name']
                latest_value = kpi_data['latest_value'] or 0
                
                if kpi_name == 'total_cost_savings':
                    financial_impact['total_cost_savings'] = latest_value
                elif kpi_name == 'roi_percentage':
                    financial_impact['roi_percentage'] = latest_value
                elif kpi_name == 'battery_life_extension':
                    operational_impact['battery_life_extension'] = latest_value
                elif kpi_name == 'fleet_utilization_rate':
                    operational_impact['fleet_utilization'] = latest_value
                elif kpi_name == 'system_uptime':
                    operational_impact['system_uptime'] = latest_value
                elif kpi_name == 'carbon_footprint_reduction':
                    sustainability_impact['carbon_reduction'] = latest_value
                elif kpi_name == 'circular_economy_score':
                    sustainability_impact['circular_economy_score'] = latest_value
            
            # Get ROI analysis
            roi_analysis = dashboard.get('roi_analysis', {})
            if roi_analysis:
                financial_impact['payback_period_months'] = roi_analysis.get('payback_period_months', 0)
            
            impact_report = {
                'executive_summary': {
                    'total_value_created': financial_impact['total_cost_savings'],
                    'roi_achieved': financial_impact['roi_percentage'],
                    'operational_efficiency_gain': operational_impact['fleet_utilization'],
                    'sustainability_improvement': sustainability_impact['carbon_reduction']
                },
                'financial_impact': financial_impact,
                'operational_impact': operational_impact,
                'sustainability_impact': sustainability_impact,
                'full_dashboard': dashboard,
                'report_generated_at': datetime.now().isoformat()
            }
            
            return impact_report
            
        except Exception as e:
            logger.error(f"Error generating business impact report: {e}")
            return {}

class BusinessKPIs:
    """
    Main business KPIs management system integrating all components.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        
        # Initialize core components
        self.kpi_tracker = KPITracker()
        
        # Configuration
        self.config = self._load_config(config_path) if config_path else {}
        
        # Dashboard state
        self.dashboard_cache: Dict[str, Any] = {}
        self.cache_expiry: datetime = datetime.now()
        self.cache_duration_minutes = 5
        
        logger.info("Business KPIs system initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}
    
    def start(self):
        """Start the business KPIs system."""
        self.kpi_tracker.start_tracking()
        logger.info("Business KPIs system started")
    
    def stop(self):
        """Stop the business KPIs system."""
        self.kpi_tracker.stop_tracking()
        logger.info("Business KPIs system stopped")
    
    def get_dashboard_data(self, use_cache: bool = True) -> Dict[str, Any]:
        """Get dashboard data with optional caching."""
        try:
            # Check cache
            if (use_cache and self.dashboard_cache and 
                datetime.now() < self.cache_expiry):
                return self.dashboard_cache
            
            # Generate fresh dashboard data
            dashboard_data = self.kpi_tracker.get_kpi_dashboard()
            
            # Update cache
            if use_cache:
                self.dashboard_cache = dashboard_data
                self.cache_expiry = datetime.now() + timedelta(minutes=self.cache_duration_minutes)
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Error getting dashboard data: {e}")
            return {}
    
    def get_business_report(self) -> Dict[str, Any]:
        """Get comprehensive business impact report."""
        return self.kpi_tracker.get_business_impact_report()
    
    def register_custom_kpi(self, kpi_definition: KPIDefinition) -> bool:
        """Register a custom KPI definition."""
        return self.kpi_tracker.register_kpi(kpi_definition)
    
    def track_investment(self, category: str, amount: float, description: str = ""):
        """Track an investment for ROI calculation."""
        self.kpi_tracker.roi_calculator.track_investment(category, amount, description)
    
    def track_benefit(self, category: str, amount: float, description: str = ""):
        """Track a benefit for ROI calculation."""
        self.kpi_tracker.roi_calculator.track_benefit(category, amount, description)
    
    def track_sustainability_metric(self, metric_type: str, **kwargs):
        """Track sustainability metrics."""
        if metric_type == "carbon_emission":
            self.kpi_tracker.sustainability_metrics.track_carbon_emission(**kwargs)
        elif metric_type == "energy_consumption":
            self.kpi_tracker.sustainability_metrics.track_energy_consumption(**kwargs)
        elif metric_type == "waste_generation":
            self.kpi_tracker.sustainability_metrics.track_waste_generation(**kwargs)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics."""
        return self.kpi_tracker.tracking_stats

# Factory functions
def create_business_kpis_system(config_path: Optional[str] = None) -> BusinessKPIs:
    """Create a fully configured business KPIs system."""
    return BusinessKPIs(config_path)

def create_default_kpi_definitions() -> List[KPIDefinition]:
    """Create default KPI definitions for BatteryMind."""
    kpi_tracker = KPITracker()
    return list(kpi_tracker.kpi_definitions.values())
