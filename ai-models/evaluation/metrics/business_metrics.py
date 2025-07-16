"""
BatteryMind Business Metrics Module

This module provides comprehensive business impact measurement capabilities for battery
management system AI models. It tracks key business metrics including battery life
extension, cost savings, energy efficiency improvements, and overall ROI.

Features:
- Battery life extension calculations
- Cost savings analysis across different scenarios
- Energy efficiency improvements tracking
- ROI and business impact quantification
- Fleet-wide performance aggregation
- Maintenance cost optimization metrics
- Revenue impact assessment

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Enumeration for different business metric types."""
    COST_SAVINGS = "cost_savings"
    BATTERY_LIFE = "battery_life"
    ENERGY_EFFICIENCY = "energy_efficiency"
    MAINTENANCE = "maintenance"
    REVENUE_IMPACT = "revenue_impact"
    OPERATIONAL = "operational"
    ENVIRONMENTAL = "environmental"

@dataclass
class BusinessMetricResult:
    """Container for business metric calculation results."""
    metric_name: str
    value: float
    unit: str
    improvement_percentage: float
    baseline_value: float
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

class BatteryLifeExtensionMetrics:
    """
    Metrics for calculating battery life extension benefits.
    """
    
    def __init__(self):
        self.baseline_cycle_life = 3000  # Standard cycles
        self.baseline_calendar_life = 8  # Years
        self.replacement_cost_per_kwh = 100  # USD per kWh
        
    def calculate_cycle_life_extension(self, 
                                     optimized_cycles: int,
                                     baseline_cycles: Optional[int] = None) -> BusinessMetricResult:
        """
        Calculate cycle life extension benefits.
        
        Args:
            optimized_cycles: Number of cycles with AI optimization
            baseline_cycles: Baseline cycles without optimization
            
        Returns:
            BusinessMetricResult with cycle life extension metrics
        """
        if baseline_cycles is None:
            baseline_cycles = self.baseline_cycle_life
        
        cycle_extension = optimized_cycles - baseline_cycles
        improvement_percentage = (cycle_extension / baseline_cycles) * 100
        
        return BusinessMetricResult(
            metric_name="cycle_life_extension",
            value=cycle_extension,
            unit="cycles",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_cycles,
            metadata={
                "optimized_cycles": optimized_cycles,
                "baseline_cycles": baseline_cycles,
                "extension_factor": optimized_cycles / baseline_cycles
            }
        )
    
    def calculate_calendar_life_extension(self,
                                        optimized_years: float,
                                        baseline_years: Optional[float] = None) -> BusinessMetricResult:
        """
        Calculate calendar life extension benefits.
        
        Args:
            optimized_years: Years of service with AI optimization
            baseline_years: Baseline years without optimization
            
        Returns:
            BusinessMetricResult with calendar life extension metrics
        """
        if baseline_years is None:
            baseline_years = self.baseline_calendar_life
        
        life_extension = optimized_years - baseline_years
        improvement_percentage = (life_extension / baseline_years) * 100
        
        return BusinessMetricResult(
            metric_name="calendar_life_extension",
            value=life_extension,
            unit="years",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_years,
            metadata={
                "optimized_years": optimized_years,
                "baseline_years": baseline_years,
                "extension_factor": optimized_years / baseline_years
            }
        )
    
    def calculate_replacement_cost_avoidance(self,
                                           battery_capacity_kwh: float,
                                           life_extension_years: float,
                                           cost_per_kwh: Optional[float] = None) -> BusinessMetricResult:
        """
        Calculate cost avoidance from delayed battery replacement.
        
        Args:
            battery_capacity_kwh: Battery capacity in kWh
            life_extension_years: Years of extended life
            cost_per_kwh: Replacement cost per kWh
            
        Returns:
            BusinessMetricResult with cost avoidance metrics
        """
        if cost_per_kwh is None:
            cost_per_kwh = self.replacement_cost_per_kwh
        
        total_replacement_cost = battery_capacity_kwh * cost_per_kwh
        cost_avoidance = total_replacement_cost * (life_extension_years / self.baseline_calendar_life)
        
        return BusinessMetricResult(
            metric_name="replacement_cost_avoidance",
            value=cost_avoidance,
            unit="USD",
            improvement_percentage=0,  # This is an absolute cost avoidance
            baseline_value=0,
            metadata={
                "battery_capacity_kwh": battery_capacity_kwh,
                "life_extension_years": life_extension_years,
                "cost_per_kwh": cost_per_kwh,
                "total_replacement_cost": total_replacement_cost
            }
        )

class CostSavingsMetrics:
    """
    Metrics for calculating various cost savings from AI optimization.
    """
    
    def __init__(self):
        self.electricity_cost_per_kwh = 0.12  # USD per kWh
        self.maintenance_cost_per_hour = 75  # USD per hour
        self.downtime_cost_per_hour = 500  # USD per hour
        
    def calculate_energy_cost_savings(self,
                                    baseline_consumption_kwh: float,
                                    optimized_consumption_kwh: float,
                                    electricity_rate: Optional[float] = None) -> BusinessMetricResult:
        """
        Calculate energy cost savings from efficiency improvements.
        
        Args:
            baseline_consumption_kwh: Energy consumption without optimization
            optimized_consumption_kwh: Energy consumption with optimization
            electricity_rate: Cost per kWh
            
        Returns:
            BusinessMetricResult with energy cost savings
        """
        if electricity_rate is None:
            electricity_rate = self.electricity_cost_per_kwh
        
        energy_savings_kwh = baseline_consumption_kwh - optimized_consumption_kwh
        cost_savings = energy_savings_kwh * electricity_rate
        improvement_percentage = (energy_savings_kwh / baseline_consumption_kwh) * 100
        
        return BusinessMetricResult(
            metric_name="energy_cost_savings",
            value=cost_savings,
            unit="USD",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_consumption_kwh * electricity_rate,
            metadata={
                "baseline_consumption_kwh": baseline_consumption_kwh,
                "optimized_consumption_kwh": optimized_consumption_kwh,
                "energy_savings_kwh": energy_savings_kwh,
                "electricity_rate": electricity_rate
            }
        )
    
    def calculate_maintenance_cost_savings(self,
                                         baseline_maintenance_hours: float,
                                         optimized_maintenance_hours: float,
                                         hourly_rate: Optional[float] = None) -> BusinessMetricResult:
        """
        Calculate maintenance cost savings from predictive maintenance.
        
        Args:
            baseline_maintenance_hours: Maintenance hours without optimization
            optimized_maintenance_hours: Maintenance hours with optimization
            hourly_rate: Maintenance cost per hour
            
        Returns:
            BusinessMetricResult with maintenance cost savings
        """
        if hourly_rate is None:
            hourly_rate = self.maintenance_cost_per_hour
        
        maintenance_savings_hours = baseline_maintenance_hours - optimized_maintenance_hours
        cost_savings = maintenance_savings_hours * hourly_rate
        improvement_percentage = (maintenance_savings_hours / baseline_maintenance_hours) * 100
        
        return BusinessMetricResult(
            metric_name="maintenance_cost_savings",
            value=cost_savings,
            unit="USD",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_maintenance_hours * hourly_rate,
            metadata={
                "baseline_maintenance_hours": baseline_maintenance_hours,
                "optimized_maintenance_hours": optimized_maintenance_hours,
                "maintenance_savings_hours": maintenance_savings_hours,
                "hourly_rate": hourly_rate
            }
        )
    
    def calculate_downtime_cost_avoidance(self,
                                        baseline_downtime_hours: float,
                                        optimized_downtime_hours: float,
                                        downtime_cost_rate: Optional[float] = None) -> BusinessMetricResult:
        """
        Calculate cost avoidance from reduced downtime.
        
        Args:
            baseline_downtime_hours: Downtime hours without optimization
            optimized_downtime_hours: Downtime hours with optimization
            downtime_cost_rate: Cost per hour of downtime
            
        Returns:
            BusinessMetricResult with downtime cost avoidance
        """
        if downtime_cost_rate is None:
            downtime_cost_rate = self.downtime_cost_per_hour
        
        downtime_reduction_hours = baseline_downtime_hours - optimized_downtime_hours
        cost_avoidance = downtime_reduction_hours * downtime_cost_rate
        improvement_percentage = (downtime_reduction_hours / baseline_downtime_hours) * 100
        
        return BusinessMetricResult(
            metric_name="downtime_cost_avoidance",
            value=cost_avoidance,
            unit="USD",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_downtime_hours * downtime_cost_rate,
            metadata={
                "baseline_downtime_hours": baseline_downtime_hours,
                "optimized_downtime_hours": optimized_downtime_hours,
                "downtime_reduction_hours": downtime_reduction_hours,
                "downtime_cost_rate": downtime_cost_rate
            }
        )

class EnergyEfficiencyMetrics:
    """
    Metrics for calculating energy efficiency improvements.
    """
    
    def __init__(self):
        self.baseline_efficiency = 0.85  # 85% baseline efficiency
        
    def calculate_charging_efficiency_improvement(self,
                                                baseline_efficiency: float,
                                                optimized_efficiency: float) -> BusinessMetricResult:
        """
        Calculate charging efficiency improvements.
        
        Args:
            baseline_efficiency: Baseline charging efficiency (0-1)
            optimized_efficiency: Optimized charging efficiency (0-1)
            
        Returns:
            BusinessMetricResult with charging efficiency improvements
        """
        efficiency_improvement = optimized_efficiency - baseline_efficiency
        improvement_percentage = (efficiency_improvement / baseline_efficiency) * 100
        
        return BusinessMetricResult(
            metric_name="charging_efficiency_improvement",
            value=efficiency_improvement,
            unit="efficiency_ratio",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_efficiency,
            metadata={
                "baseline_efficiency": baseline_efficiency,
                "optimized_efficiency": optimized_efficiency,
                "efficiency_gain": efficiency_improvement
            }
        )
    
    def calculate_energy_throughput_improvement(self,
                                              baseline_throughput_kwh: float,
                                              optimized_throughput_kwh: float) -> BusinessMetricResult:
        """
        Calculate energy throughput improvements.
        
        Args:
            baseline_throughput_kwh: Baseline energy throughput
            optimized_throughput_kwh: Optimized energy throughput
            
        Returns:
            BusinessMetricResult with energy throughput improvements
        """
        throughput_improvement = optimized_throughput_kwh - baseline_throughput_kwh
        improvement_percentage = (throughput_improvement / baseline_throughput_kwh) * 100
        
        return BusinessMetricResult(
            metric_name="energy_throughput_improvement",
            value=throughput_improvement,
            unit="kWh",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_throughput_kwh,
            metadata={
                "baseline_throughput_kwh": baseline_throughput_kwh,
                "optimized_throughput_kwh": optimized_throughput_kwh,
                "throughput_gain": throughput_improvement
            }
        )
    
    def calculate_round_trip_efficiency(self,
                                      energy_input_kwh: float,
                                      energy_output_kwh: float) -> BusinessMetricResult:
        """
        Calculate round-trip efficiency for battery systems.
        
        Args:
            energy_input_kwh: Energy input during charging
            energy_output_kwh: Energy output during discharging
            
        Returns:
            BusinessMetricResult with round-trip efficiency
        """
        round_trip_efficiency = energy_output_kwh / energy_input_kwh
        improvement_percentage = ((round_trip_efficiency - self.baseline_efficiency) / 
                                self.baseline_efficiency) * 100
        
        return BusinessMetricResult(
            metric_name="round_trip_efficiency",
            value=round_trip_efficiency,
            unit="efficiency_ratio",
            improvement_percentage=improvement_percentage,
            baseline_value=self.baseline_efficiency,
            metadata={
                "energy_input_kwh": energy_input_kwh,
                "energy_output_kwh": energy_output_kwh,
                "energy_loss_kwh": energy_input_kwh - energy_output_kwh,
                "loss_percentage": ((energy_input_kwh - energy_output_kwh) / energy_input_kwh) * 100
            }
        )

class OperationalMetrics:
    """
    Metrics for operational performance improvements.
    """
    
    def calculate_availability_improvement(self,
                                         baseline_availability: float,
                                         optimized_availability: float) -> BusinessMetricResult:
        """
        Calculate system availability improvements.
        
        Args:
            baseline_availability: Baseline system availability (0-1)
            optimized_availability: Optimized system availability (0-1)
            
        Returns:
            BusinessMetricResult with availability improvements
        """
        availability_improvement = optimized_availability - baseline_availability
        improvement_percentage = (availability_improvement / baseline_availability) * 100
        
        return BusinessMetricResult(
            metric_name="availability_improvement",
            value=availability_improvement,
            unit="availability_ratio",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_availability,
            metadata={
                "baseline_availability": baseline_availability,
                "optimized_availability": optimized_availability,
                "availability_gain": availability_improvement
            }
        )
    
    def calculate_response_time_improvement(self,
                                          baseline_response_time: float,
                                          optimized_response_time: float) -> BusinessMetricResult:
        """
        Calculate response time improvements.
        
        Args:
            baseline_response_time: Baseline response time in seconds
            optimized_response_time: Optimized response time in seconds
            
        Returns:
            BusinessMetricResult with response time improvements
        """
        response_time_improvement = baseline_response_time - optimized_response_time
        improvement_percentage = (response_time_improvement / baseline_response_time) * 100
        
        return BusinessMetricResult(
            metric_name="response_time_improvement",
            value=response_time_improvement,
            unit="seconds",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_response_time,
            metadata={
                "baseline_response_time": baseline_response_time,
                "optimized_response_time": optimized_response_time,
                "time_saved": response_time_improvement
            }
        )
    
    def calculate_throughput_improvement(self,
                                       baseline_throughput: float,
                                       optimized_throughput: float,
                                       unit: str = "requests_per_second") -> BusinessMetricResult:
        """
        Calculate system throughput improvements.
        
        Args:
            baseline_throughput: Baseline system throughput
            optimized_throughput: Optimized system throughput
            unit: Unit of measurement for throughput
            
        Returns:
            BusinessMetricResult with throughput improvements
        """
        throughput_improvement = optimized_throughput - baseline_throughput
        improvement_percentage = (throughput_improvement / baseline_throughput) * 100
        
        return BusinessMetricResult(
            metric_name="throughput_improvement",
            value=throughput_improvement,
            unit=unit,
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_throughput,
            metadata={
                "baseline_throughput": baseline_throughput,
                "optimized_throughput": optimized_throughput,
                "throughput_gain": throughput_improvement
            }
        )

class EnvironmentalMetrics:
    """
    Metrics for environmental impact improvements.
    """
    
    def __init__(self):
        self.co2_emission_factor = 0.4  # kg CO2 per kWh (grid average)
        
    def calculate_carbon_footprint_reduction(self,
                                           baseline_energy_kwh: float,
                                           optimized_energy_kwh: float,
                                           emission_factor: Optional[float] = None) -> BusinessMetricResult:
        """
        Calculate carbon footprint reduction from energy efficiency.
        
        Args:
            baseline_energy_kwh: Baseline energy consumption
            optimized_energy_kwh: Optimized energy consumption
            emission_factor: CO2 emission factor (kg CO2 per kWh)
            
        Returns:
            BusinessMetricResult with carbon footprint reduction
        """
        if emission_factor is None:
            emission_factor = self.co2_emission_factor
        
        energy_savings_kwh = baseline_energy_kwh - optimized_energy_kwh
        carbon_reduction_kg = energy_savings_kwh * emission_factor
        improvement_percentage = (energy_savings_kwh / baseline_energy_kwh) * 100
        
        return BusinessMetricResult(
            metric_name="carbon_footprint_reduction",
            value=carbon_reduction_kg,
            unit="kg_CO2",
            improvement_percentage=improvement_percentage,
            baseline_value=baseline_energy_kwh * emission_factor,
            metadata={
                "baseline_energy_kwh": baseline_energy_kwh,
                "optimized_energy_kwh": optimized_energy_kwh,
                "energy_savings_kwh": energy_savings_kwh,
                "emission_factor": emission_factor
            }
        )

class ROICalculator:
    """
    Calculator for Return on Investment (ROI) metrics.
    """
    
    def __init__(self):
        self.default_discount_rate = 0.08  # 8% discount rate
        
    def calculate_net_present_value(self,
                                  cash_flows: List[float],
                                  discount_rate: Optional[float] = None) -> BusinessMetricResult:
        """
        Calculate Net Present Value (NPV) of cash flows.
        
        Args:
            cash_flows: List of cash flows by year (negative for costs, positive for benefits)
            discount_rate: Discount rate for NPV calculation
            
        Returns:
            BusinessMetricResult with NPV calculation
        """
        if discount_rate is None:
            discount_rate = self.default_discount_rate
        
        npv = sum(cf / (1 + discount_rate) ** i for i, cf in enumerate(cash_flows))
        
        return BusinessMetricResult(
            metric_name="net_present_value",
            value=npv,
            unit="USD",
            improvement_percentage=0,  # NPV is an absolute value
            baseline_value=0,
            metadata={
                "cash_flows": cash_flows,
                "discount_rate": discount_rate,
                "project_years": len(cash_flows),
                "total_undiscounted_cash_flow": sum(cash_flows)
            }
        )
    
    def calculate_payback_period(self,
                               initial_investment: float,
                               annual_savings: List[float]) -> BusinessMetricResult:
        """
        Calculate payback period for the investment.
        
        Args:
            initial_investment: Initial investment amount
            annual_savings: List of annual savings
            
        Returns:
            BusinessMetricResult with payback period
        """
        cumulative_savings = 0
        payback_period = 0
        
        for year, savings in enumerate(annual_savings, 1):
            cumulative_savings += savings
            if cumulative_savings >= initial_investment:
                # Linear interpolation for fractional year
                if year > 1:
                    previous_cumulative = cumulative_savings - savings
                    fraction = (initial_investment - previous_cumulative) / savings
                    payback_period = year - 1 + fraction
                else:
                    payback_period = initial_investment / savings
                break
        else:
            payback_period = len(annual_savings)  # Payback not achieved within period
        
        return BusinessMetricResult(
            metric_name="payback_period",
            value=payback_period,
            unit="years",
            improvement_percentage=0,  # Payback period is an absolute value
            baseline_value=0,
            metadata={
                "initial_investment": initial_investment,
                "annual_savings": annual_savings,
                "cumulative_savings": cumulative_savings,
                "payback_achieved": cumulative_savings >= initial_investment
            }
        )
    
    def calculate_roi_percentage(self,
                               total_benefits: float,
                               total_costs: float) -> BusinessMetricResult:
        """
        Calculate Return on Investment as a percentage.
        
        Args:
            total_benefits: Total benefits over the investment period
            total_costs: Total costs including initial investment
            
        Returns:
            BusinessMetricResult with ROI percentage
        """
        roi_percentage = ((total_benefits - total_costs) / total_costs) * 100
        
        return BusinessMetricResult(
            metric_name="roi_percentage",
            value=roi_percentage,
            unit="percent",
            improvement_percentage=0,  # ROI is an absolute percentage
            baseline_value=0,
            metadata={
                "total_benefits": total_benefits,
                "total_costs": total_costs,
                "net_benefit": total_benefits - total_costs,
                "benefit_cost_ratio": total_benefits / total_costs
            }
        )

class FleetMetricsAggregator:
    """
    Aggregator for fleet-wide business metrics.
    """
    
    def __init__(self):
        self.metrics_history = []
        
    def aggregate_fleet_metrics(self,
                               individual_metrics: List[Dict[str, BusinessMetricResult]]) -> Dict[str, BusinessMetricResult]:
        """
        Aggregate metrics across a fleet of batteries.
        
        Args:
            individual_metrics: List of metric dictionaries for each battery
            
        Returns:
            Dictionary of aggregated fleet metrics
        """
        aggregated_metrics = {}
        
        if not individual_metrics:
            return aggregated_metrics
        
        # Get all metric names from first battery
        metric_names = list(individual_metrics[0].keys())
        
        for metric_name in metric_names:
            values = []
            improvements = []
            baseline_values = []
            
            for battery_metrics in individual_metrics:
                if metric_name in battery_metrics:
                    metric = battery_metrics[metric_name]
                    values.append(metric.value)
                    improvements.append(metric.improvement_percentage)
                    baseline_values.append(metric.baseline_value)
            
            if values:
                # Calculate aggregate statistics
                total_value = sum(values)
                avg_improvement = np.mean(improvements)
                total_baseline = sum(baseline_values)
                
                aggregated_metrics[f"fleet_{metric_name}"] = BusinessMetricResult(
                    metric_name=f"fleet_{metric_name}",
                    value=total_value,
                    unit=individual_metrics[0][metric_name].unit,
                    improvement_percentage=avg_improvement,
                    baseline_value=total_baseline,
                    metadata={
                        "fleet_size": len(individual_metrics),
                        "individual_values": values,
                        "individual_improvements": improvements,
                        "min_value": min(values),
                        "max_value": max(values),
                        "std_value": np.std(values)
                    }
                )
        
        return aggregated_metrics
    
    def calculate_fleet_summary(self,
                               fleet_metrics: Dict[str, BusinessMetricResult]) -> Dict[str, float]:
        """
        Calculate fleet-wide summary statistics.
        
        Args:
            fleet_metrics: Dictionary of aggregated fleet metrics
            
        Returns:
            Dictionary of fleet summary statistics
        """
        summary = {
            "total_cost_savings": 0,
            "total_energy_savings": 0,
            "average_life_extension": 0,
            "fleet_roi": 0,
            "total_carbon_reduction": 0
        }
        
        # Map metric names to summary categories
        cost_metrics = ["energy_cost_savings", "maintenance_cost_savings", "downtime_cost_avoidance"]
        energy_metrics = ["energy_throughput_improvement", "charging_efficiency_improvement"]
        life_metrics = ["cycle_life_extension", "calendar_life_extension"]
        
        for metric_name, metric_result in fleet_metrics.items():
            if any(cost_metric in metric_name for cost_metric in cost_metrics):
                summary["total_cost_savings"] += metric_result.value
            elif any(energy_metric in metric_name for energy_metric in energy_metrics):
                summary["total_energy_savings"] += metric_result.value
            elif any(life_metric in metric_name for life_metric in life_metrics):
                summary["average_life_extension"] += metric_result.value / len(life_metrics)
            elif "carbon_footprint_reduction" in metric_name:
                summary["total_carbon_reduction"] += metric_result.value
        
        return summary

class BusinessMetricsReporter:
    """
    Reporter for generating business metrics reports.
    """
    
    def __init__(self):
        self.report_templates = {
            "executive_summary": self._generate_executive_summary,
            "detailed_analysis": self._generate_detailed_analysis,
            "financial_impact": self._generate_financial_impact,
            "operational_impact": self._generate_operational_impact
        }
    
    def generate_report(self,
                       metrics: Dict[str, BusinessMetricResult],
                       report_type: str = "executive_summary") -> str:
        """
        Generate a business metrics report.
        
        Args:
            metrics: Dictionary of business metrics
            report_type: Type of report to generate
            
        Returns:
            Formatted report string
        """
        if report_type not in self.report_templates:
            raise ValueError(f"Unknown report type: {report_type}")
        
        return self.report_templates[report_type](metrics)
    
    def _generate_executive_summary(self, metrics: Dict[str, BusinessMetricResult]) -> str:
        """Generate executive summary report."""
        report_lines = [
            "BatteryMind Business Impact - Executive Summary",
            "=" * 50,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Key metrics summary
        total_savings = sum(m.value for m in metrics.values() if "cost_savings" in m.metric_name)
        avg_improvement = np.mean([m.improvement_percentage for m in metrics.values()])
        
        report_lines.extend([
            "Key Business Impacts:",
            f"• Total Cost Savings: ${total_savings:,.2f}",
            f"• Average Performance Improvement: {avg_improvement:.1f}%",
            f"• Number of Metrics Tracked: {len(metrics)}",
            ""
        ])
        
        # Top performing metrics
        sorted_metrics = sorted(metrics.values(), key=lambda x: x.improvement_percentage, reverse=True)
        report_lines.extend([
            "Top Performance Improvements:",
            ""
        ])
        
        for i, metric in enumerate(sorted_metrics[:5], 1):
            report_lines.append(f"{i}. {metric.metric_name}: {metric.improvement_percentage:.1f}% improvement")
        
        return "\n".join(report_lines)
    
    def _generate_detailed_analysis(self, metrics: Dict[str, BusinessMetricResult]) -> str:
        """Generate detailed analysis report."""
        report_lines = [
            "BatteryMind Business Impact - Detailed Analysis",
            "=" * 50,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        for metric_name, metric in metrics.items():
            report_lines.extend([
                f"Metric: {metric.metric_name}",
                f"Value: {metric.value:.2f} {metric.unit}",
                f"Improvement: {metric.improvement_percentage:.1f}%",
                f"Baseline: {metric.baseline_value:.2f} {metric.unit}",
                f"Calculated: {metric.calculation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ])
        
        return "\n".join(report_lines)
    
    def _generate_financial_impact(self, metrics: Dict[str, BusinessMetricResult]) -> str:
        """Generate financial impact report."""
        report_lines = [
            "BatteryMind Business Impact - Financial Analysis",
            "=" * 50,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Filter financial metrics
        financial_metrics = {k: v for k, v in metrics.items() if "cost" in k or "savings" in k or "roi" in k}
        
        total_savings = sum(m.value for m in financial_metrics.values() if m.unit == "USD")
        
        report_lines.extend([
            f"Total Financial Impact: ${total_savings:,.2f}",
            "",
            "Breakdown by Category:",
            ""
        ])
        
        for metric_name, metric in financial_metrics.items():
            if metric.unit == "USD":
                report_lines.append(f"• {metric.metric_name}: ${metric.value:,.2f}")
        
        return "\n".join(report_lines)
    
    def _generate_operational_impact(self, metrics: Dict[str, BusinessMetricResult]) -> str:
        """Generate operational impact report."""
        report_lines = [
            "BatteryMind Business Impact - Operational Analysis",
            "=" * 50,
            f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Filter operational metrics
        operational_metrics = {k: v for k, v in metrics.items() if 
                             "efficiency" in k or "availability" in k or "throughput" in k}
        
        report_lines.extend([
            "Operational Performance Improvements:",
            ""
        ])
        
        for metric_name, metric in operational_metrics.items():
            report_lines.append(f"• {metric.metric_name}: {metric.improvement_percentage:.1f}% improvement")
        
        return "\n".join(report_lines)
    
    def save_report(self, report_content: str, filename: str) -> None:
        """
        Save report to file.
        
        Args:
            report_content: Report content to save
            filename: Output filename
        """
        with open(filename, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to {filename}")

class BusinessMetricsManager:
    """
    Main manager class for business metrics calculation and reporting.
    """
    
    def __init__(self):
        self.battery_life_metrics = BatteryLifeExtensionMetrics()
        self.cost_savings_metrics = CostSavingsMetrics()
        self.energy_efficiency_metrics = EnergyEfficiencyMetrics()
        self.operational_metrics = OperationalMetrics()
        self.environmental_metrics = EnvironmentalMetrics()
        self.roi_calculator = ROICalculator()
        self.fleet_aggregator = FleetMetricsAggregator()
        self.reporter = BusinessMetricsReporter()
        
    def calculate_comprehensive_metrics(self,
                                      input_data: Dict[str, Any]) -> Dict[str, BusinessMetricResult]:
        """
        Calculate comprehensive business metrics from input data.
        
        Args:
            input_data: Dictionary containing all necessary input data
            
        Returns:
            Dictionary of calculated business metrics
        """
        metrics = {}
        
        # Battery life metrics
        if "cycle_data" in input_data:
            metrics["cycle_life_extension"] = self.battery_life_metrics.calculate_cycle_life_extension(
                input_data["cycle_data"]["optimized_cycles"],
                input_data["cycle_data"].get("baseline_cycles")
            )
        
        if "calendar_data" in input_data:
            metrics["calendar_life_extension"] = self.battery_life_metrics.calculate_calendar_life_extension(
                input_data["calendar_data"]["optimized_years"],
                input_data["calendar_data"].get("baseline_years")
            )
        
        # Cost savings metrics
        if "energy_data" in input_data:
            metrics["energy_cost_savings"] = self.cost_savings_metrics.calculate_energy_cost_savings(
                input_data["energy_data"]["baseline_consumption_kwh"],
                input_data["energy_data"]["optimized_consumption_kwh"],
                input_data["energy_data"].get("electricity_rate")
            )
        
        if "maintenance_data" in input_data:
            metrics["maintenance_cost_savings"] = self.cost_savings_metrics.calculate_maintenance_cost_savings(
                input_data["maintenance_data"]["baseline_maintenance_hours"],
                input_data["maintenance_data"]["optimized_maintenance_hours"],
                input_data["maintenance_data"].get("hourly_rate")
            )
        
        # Energy efficiency metrics
        if "efficiency_data" in input_data:
            metrics["charging_efficiency_improvement"] = self.energy_efficiency_metrics.calculate_charging_efficiency_improvement(
                input_data["efficiency_data"]["baseline_efficiency"],
                input_data["efficiency_data"]["optimized_efficiency"]
            )
        
        # Environmental metrics
        if "environmental_data" in input_data:
            metrics["carbon_footprint_reduction"] = self.environmental_metrics.calculate_carbon_footprint_reduction(
                input_data["environmental_data"]["baseline_energy_kwh"],
                input_data["environmental_data"]["optimized_energy_kwh"],
                input_data["environmental_data"].get("emission_factor")
            )
        
        # ROI calculations
        if "roi_data" in input_data:
            if "cash_flows" in input_data["roi_data"]:
                metrics["net_present_value"] = self.roi_calculator.calculate_net_present_value(
                    input_data["roi_data"]["cash_flows"],
                    input_data["roi_data"].get("discount_rate")
                )
            
            if "payback_data" in input_data["roi_data"]:
                metrics["payback_period"] = self.roi_calculator.calculate_payback_period(
                    input_data["roi_data"]["payback_data"]["initial_investment"],
                    input_data["roi_data"]["payback_data"]["annual_savings"]
                )
        
        return metrics
    
    def generate_comprehensive_report(self,
                                    metrics: Dict[str, BusinessMetricResult],
                                    report_type: str = "executive_summary") -> str:
        """
        Generate a comprehensive business metrics report.
        
        Args:
            metrics: Dictionary of business metrics
            report_type: Type of report to generate
            
        Returns:
            Formatted report string
        """
        return self.reporter.generate_report(metrics, report_type)
    
    def save_metrics_to_json(self,
                           metrics: Dict[str, BusinessMetricResult],
                           filename: str) -> None:
        """
        Save metrics to JSON file.
        
        Args:
            metrics: Dictionary of business metrics
            filename: Output filename
        """
        metrics_dict = {}
        for name, metric in metrics.items():
            metrics_dict[name] = {
                "metric_name": metric.metric_name,
                "value": metric.value,
                "unit": metric.unit,
                "improvement_percentage": metric.improvement_percentage,
                "baseline_value": metric.baseline_value,
                "calculation_timestamp": metric.calculation_timestamp.isoformat(),
                "metadata": metric.metadata
            }
        
        with open(filename, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        logger.info(f"Metrics saved to {filename}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize the business metrics manager
    metrics_manager = BusinessMetricsManager()
    
    # Example input data
    input_data = {
        "cycle_data": {
            "optimized_cycles": 4500,
            "baseline_cycles": 3000
        },
        "calendar_data": {
            "optimized_years": 12,
            "baseline_years": 8
        },
        "energy_data": {
            "baseline_consumption_kwh": 1000,
            "optimized_consumption_kwh": 850,
            "electricity_rate": 0.12
        },
        "maintenance_data": {
            "baseline_maintenance_hours": 100,
            "optimized_maintenance_hours": 60,
            "hourly_rate": 75
        },
        "efficiency_data": {
            "baseline_efficiency": 0.85,
            "optimized_efficiency": 0.92
        },
        "environmental_data": {
            "baseline_energy_kwh": 1000,
            "optimized_energy_kwh": 850,
            "emission_factor": 0.4
        },
        "roi_data": {
            "cash_flows": [-50000, 15000, 20000, 25000, 30000],
            "discount_rate": 0.08,
            "payback_data": {
                "initial_investment": 50000,
                "annual_savings": [15000, 20000, 25000, 30000]
            }
        }
    }
    
    # Calculate metrics
    metrics = metrics_manager.calculate_comprehensive_metrics(input_data)
    
    # Generate reports
    executive_summary = metrics_manager.generate_comprehensive_report(metrics, "executive_summary")
    detailed_analysis = metrics_manager.generate_comprehensive_report(metrics, "detailed_analysis")
    
    # Print results
    print(executive_summary)
    print("\n" + "="*50 + "\n")
    print(detailed_analysis)
    
    # Save metrics
    metrics_manager.save_metrics_to_json(metrics, "business_metrics_results.json")
    
    print("\nBusiness metrics calculation completed successfully!")
