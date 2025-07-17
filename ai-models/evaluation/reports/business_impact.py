"""
BatteryMind - Business Impact Report Module

Comprehensive business impact analysis and ROI calculation framework for
battery management AI/ML implementations. This module quantifies the business
value and economic benefits of AI-driven battery optimization.

Features:
- Financial impact analysis and ROI calculations
- Cost-benefit analysis with sensitivity scenarios
- Battery life extension quantification
- Energy efficiency improvements measurement
- Maintenance cost reduction analysis
- Risk mitigation value assessment
- Long-term business case development

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from datetime import datetime, timedelta
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# Financial analysis imports
from scipy import stats
from sklearn.metrics import mean_absolute_error

# Internal imports
from ..metrics.business_metrics import BusinessMetrics
from ..metrics.efficiency_metrics import EfficiencyMetrics
from ...utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger(__name__)

@dataclass
class BusinessImpactConfig:
    """Configuration for business impact analysis."""
    
    # Financial parameters
    discount_rate: float = 0.08  # 8% annual discount rate
    analysis_period_years: int = 5  # 5-year analysis
    currency: str = "USD"
    
    # Battery economics
    battery_replacement_cost: float = 15000.0  # USD per battery
    maintenance_cost_per_hour: float = 75.0  # USD per maintenance hour
    downtime_cost_per_hour: float = 500.0  # USD per hour of downtime
    energy_cost_per_kwh: float = 0.12  # USD per kWh
    
    # Baseline assumptions
    baseline_battery_life_years: float = 8.0  # Years without AI
    baseline_maintenance_hours_per_year: float = 40.0
    baseline_downtime_hours_per_year: float = 24.0
    baseline_energy_efficiency: float = 0.85  # 85% efficiency
    
    # AI improvement assumptions
    ai_life_extension_factor: float = 1.4  # 40% life extension
    ai_maintenance_reduction_factor: float = 0.7  # 30% reduction
    ai_downtime_reduction_factor: float = 0.5  # 50% reduction
    ai_efficiency_improvement: float = 0.05  # 5% efficiency gain
    
    # Implementation costs
    ai_implementation_cost: float = 50000.0  # One-time implementation
    ai_annual_operating_cost: float = 5000.0  # Annual operating costs
    training_cost: float = 10000.0  # One-time training cost
    
    # Risk factors
    implementation_risk_factor: float = 0.1  # 10% risk adjustment
    technology_obsolescence_risk: float = 0.05  # 5% per year
    
    # Reporting settings
    include_sensitivity_analysis: bool = True
    include_scenario_analysis: bool = True
    confidence_intervals: List[float] = field(default_factory=lambda: [0.8, 0.9, 0.95])

class BusinessImpactAnalyzer:
    """
    Comprehensive business impact analyzer for battery AI implementations.
    """
    
    def __init__(self, config: BusinessImpactConfig):
        self.config = config
        self.analysis_results = {}
        
        # Initialize metrics calculators
        self.business_metrics = BusinessMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        
        logger.info("Business impact analyzer initialized")
    
    def analyze_financial_impact(self, 
                                battery_performance_data: Dict[str, Any],
                                fleet_size: int = 100) -> Dict[str, Any]:
        """
        Analyze comprehensive financial impact of AI implementation.
        
        Args:
            battery_performance_data: Performance data from AI implementation
            fleet_size: Number of batteries in the fleet
            
        Returns:
            Comprehensive financial impact analysis
        """
        logger.info(f"Analyzing financial impact for fleet of {fleet_size} batteries")
        
        # Calculate cost components
        cost_analysis = self._calculate_cost_components(fleet_size)
        
        # Calculate benefit components
        benefit_analysis = self._calculate_benefit_components(battery_performance_data, fleet_size)
        
        # Calculate ROI and payback
        roi_analysis = self._calculate_roi_metrics(cost_analysis, benefit_analysis)
        
        # Perform sensitivity analysis
        sensitivity_analysis = self._perform_sensitivity_analysis(fleet_size) if self.config.include_sensitivity_analysis else {}
        
        # Perform scenario analysis
        scenario_analysis = self._perform_scenario_analysis(fleet_size) if self.config.include_scenario_analysis else {}
        
        # Calculate risk-adjusted returns
        risk_analysis = self._calculate_risk_adjusted_returns(roi_analysis)
        
        # Compile comprehensive analysis
        financial_impact = {
            'analysis_metadata': {
                'analysis_date': datetime.now().isoformat(),
                'fleet_size': fleet_size,
                'analysis_period_years': self.config.analysis_period_years,
                'discount_rate': self.config.discount_rate,
                'currency': self.config.currency
            },
            'cost_analysis': cost_analysis,
            'benefit_analysis': benefit_analysis,
            'roi_analysis': roi_analysis,
            'risk_analysis': risk_analysis,
            'sensitivity_analysis': sensitivity_analysis,
            'scenario_analysis': scenario_analysis,
            'executive_summary': self._generate_executive_summary(roi_analysis, benefit_analysis),
            'recommendations': self._generate_business_recommendations(roi_analysis, risk_analysis)
        }
        
        self.analysis_results = financial_impact
        
        return financial_impact
    
    def _calculate_cost_components(self, fleet_size: int) -> Dict[str, Any]:
        """Calculate all cost components of AI implementation."""
        costs = {}
        
        # One-time implementation costs
        one_time_costs = {
            'ai_implementation': self.config.ai_implementation_cost,
            'training_and_setup': self.config.training_cost,
            'system_integration': self.config.ai_implementation_cost * 0.2,  # 20% of implementation
            'change_management': self.config.training_cost * 0.5  # 50% of training cost
        }
        
        costs['one_time_costs'] = one_time_costs
        costs['total_one_time_cost'] = sum(one_time_costs.values())
        
        # Annual operating costs
        annual_costs = {
            'ai_system_operations': self.config.ai_annual_operating_cost,
            'software_licenses': self.config.ai_annual_operating_cost * 0.3,
            'monitoring_and_maintenance': self.config.ai_annual_operating_cost * 0.2,
            'staff_training_ongoing': self.config.training_cost * 0.1  # 10% annually
        }
        
        costs['annual_operating_costs'] = annual_costs
        costs['total_annual_operating_cost'] = sum(annual_costs.values())
        
        # Total costs over analysis period
        total_costs_by_year = []
        for year in range(self.config.analysis_period_years):
            year_cost = costs['total_annual_operating_cost']
            if year == 0:  # Add one-time costs in first year
                year_cost += costs['total_one_time_cost']
            
            # Apply discount factor
            discount_factor = (1 + self.config.discount_rate) ** year
            discounted_cost = year_cost / discount_factor
            
            total_costs_by_year.append({
                'year': year + 1,
                'nominal_cost': year_cost,
                'discounted_cost': discounted_cost,
                'cumulative_discounted_cost': sum([c['discounted_cost'] for c in total_costs_by_year]) + discounted_cost
            })
        
        costs['costs_by_year'] = total_costs_by_year
        costs['total_discounted_cost'] = sum([c['discounted_cost'] for c in total_costs_by_year])
        
        return costs
    
    def _calculate_benefit_components(self, 
                                    performance_data: Dict[str, Any], 
                                    fleet_size: int) -> Dict[str, Any]:
        """Calculate all benefit components from AI implementation."""
        benefits = {}
        
        # Battery life extension benefits
        baseline_replacement_cost = fleet_size * self.config.battery_replacement_cost
        baseline_replacements_per_year = fleet_size / self.config.baseline_battery_life_years
        
        ai_battery_life = self.config.baseline_battery_life_years * self.config.ai_life_extension_factor
        ai_replacements_per_year = fleet_size / ai_battery_life
        
        annual_replacement_savings = (baseline_replacements_per_year - ai_replacements_per_year) * self.config.battery_replacement_cost
        
        # Maintenance cost reduction
        baseline_maintenance_cost = fleet_size * self.config.baseline_maintenance_hours_per_year * self.config.maintenance_cost_per_hour
        ai_maintenance_cost = baseline_maintenance_cost * self.config.ai_maintenance_reduction_factor
        annual_maintenance_savings = baseline_maintenance_cost - ai_maintenance_cost
        
        # Downtime reduction benefits
        baseline_downtime_cost = fleet_size * self.config.baseline_downtime_hours_per_year * self.config.downtime_cost_per_hour
        ai_downtime_cost = baseline_downtime_cost * self.config.ai_downtime_reduction_factor
        annual_downtime_savings = baseline_downtime_cost - ai_downtime_cost
        
        # Energy efficiency improvements
        annual_energy_usage_kwh = performance_data.get('annual_energy_usage_kwh', fleet_size * 10000)  # Default 10 MWh per battery
        efficiency_improvement = self.config.ai_efficiency_improvement
        annual_energy_savings_kwh = annual_energy_usage_kwh * efficiency_improvement
        annual_energy_cost_savings = annual_energy_savings_kwh * self.config.energy_cost_per_kwh
        
        # Additional operational benefits
        annual_operational_benefits = {
            'improved_predictability': annual_maintenance_savings * 0.1,  # 10% of maintenance savings
            'reduced_insurance_premiums': baseline_replacement_cost * 0.02,  # 2% of asset value
            'improved_asset_utilization': annual_downtime_savings * 0.2,  # 20% of downtime savings
            'enhanced_safety_compliance': 5000.0  # Fixed annual benefit
        }
        
        # Compile annual benefits
        annual_benefits = {
            'battery_life_extension': annual_replacement_savings,
            'maintenance_cost_reduction': annual_maintenance_savings,
            'downtime_reduction': annual_downtime_savings,
            'energy_efficiency_gains': annual_energy_cost_savings,
            'operational_improvements': sum(annual_operational_benefits.values())
        }
        
        benefits['annual_benefit_components'] = annual_benefits
        benefits['total_annual_benefits'] = sum(annual_benefits.values())
        
        # Calculate benefits over analysis period
        benefits_by_year = []
        for year in range(self.config.analysis_period_years):
            year_benefits = benefits['total_annual_benefits']
            
            # Apply growth factor (benefits may increase over time)
            growth_factor = (1 + 0.02) ** year  # 2% annual growth in benefits
            year_benefits *= growth_factor
            
            # Apply discount factor
            discount_factor = (1 + self.config.discount_rate) ** year
            discounted_benefits = year_benefits / discount_factor
            
            benefits_by_year.append({
                'year': year + 1,
                'nominal_benefits': year_benefits,
                'discounted_benefits': discounted_benefits,
                'cumulative_discounted_benefits': sum([b['discounted_benefits'] for b in benefits_by_year]) + discounted_benefits
            })
        
        benefits['benefits_by_year'] = benefits_by_year
        benefits['total_discounted_benefits'] = sum([b['discounted_benefits'] for b in benefits_by_year])
        
        return benefits
    
    def _calculate_roi_metrics(self, 
                             cost_analysis: Dict[str, Any], 
                             benefit_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate ROI and related financial metrics."""
        total_costs = cost_analysis['total_discounted_cost']
        total_benefits = benefit_analysis['total_discounted_benefits']
        
        roi_metrics = {
            'net_present_value': total_benefits - total_costs,
            'roi_percentage': ((total_benefits - total_costs) / total_costs) * 100 if total_costs > 0 else 0,
            'benefit_cost_ratio': total_benefits / total_costs if total_costs > 0 else 0,
            'internal_rate_of_return': self._calculate_irr(cost_analysis, benefit_analysis),
            'payback_period_years': self._calculate_payback_period(cost_analysis, benefit_analysis),
            'profitability_index': total_benefits / total_costs if total_costs > 0 else 0
        }
        
        # Calculate annual ROI
        annual_roi = []
        cumulative_costs = 0
        cumulative_benefits = 0
        
        for year in range(self.config.analysis_period_years):
            year_cost = cost_analysis['costs_by_year'][year]['discounted_cost']
            year_benefit = benefit_analysis['benefits_by_year'][year]['discounted_benefits']
            
            cumulative_costs += year_cost
            cumulative_benefits += year_benefit
            
            annual_roi.append({
                'year': year + 1,
                'annual_roi': ((year_benefit - year_cost) / year_cost) * 100 if year_cost > 0 else 0,
                'cumulative_roi': ((cumulative_benefits - cumulative_costs) / cumulative_costs) * 100 if cumulative_costs > 0 else 0,
                'net_cash_flow': year_benefit - year_cost,
                'cumulative_net_cash_flow': cumulative_benefits - cumulative_costs
            })
        
        roi_metrics['annual_roi_breakdown'] = annual_roi
        
        return roi_metrics
    
    def _calculate_irr(self, cost_analysis: Dict[str, Any], benefit_analysis: Dict[str, Any]) -> float:
        """Calculate Internal Rate of Return (IRR)."""
        cash_flows = []
        
        for year in range(self.config.analysis_period_years):
            cost = cost_analysis['costs_by_year'][year]['nominal_cost']
            benefit = benefit_analysis['benefits_by_year'][year]['nominal_benefits']
            net_cash_flow = benefit - cost
            cash_flows.append(net_cash_flow)
        
        # Simple IRR calculation using numpy
        try:
            irr = np.irr([-cost_analysis['total_one_time_cost']] + cash_flows)
            return float(irr) * 100  # Convert to percentage
        except:
            # Fallback calculation
            return self._approximate_irr(cash_flows, cost_analysis['total_one_time_cost'])
    
    def _approximate_irr(self, cash_flows: List[float], initial_investment: float) -> float:
        """Approximate IRR calculation."""
        total_return = sum(cash_flows)
        average_annual_return = total_return / len(cash_flows)
        approximate_irr = (average_annual_return / initial_investment) * 100
        return max(0, approximate_irr)
    
    def _calculate_payback_period(self, 
                                 cost_analysis: Dict[str, Any], 
                                 benefit_analysis: Dict[str, Any]) -> float:
        """Calculate payback period in years."""
        initial_investment = cost_analysis['total_one_time_cost']
        cumulative_savings = 0
        
        for year in range(self.config.analysis_period_years):
            annual_net_benefit = (benefit_analysis['benefits_by_year'][year]['nominal_benefits'] - 
                                cost_analysis['annual_operating_costs']['total_annual_operating_cost'])
            cumulative_savings += annual_net_benefit
            
            if cumulative_savings >= initial_investment:
                # Linear interpolation for more precise payback period
                previous_cumulative = cumulative_savings - annual_net_benefit
                fraction_of_year = (initial_investment - previous_cumulative) / annual_net_benefit
                return year + fraction_of_year
        
        return self.config.analysis_period_years  # If not paid back within analysis period
    
    def _perform_sensitivity_analysis(self, fleet_size: int) -> Dict[str, Any]:
        """Perform sensitivity analysis on key parameters."""
        sensitivity_results = {}
        
        # Define sensitivity parameters and ranges
        sensitivity_params = {
            'ai_life_extension_factor': [1.2, 1.3, 1.4, 1.5, 1.6],
            'ai_maintenance_reduction_factor': [0.5, 0.6, 0.7, 0.8, 0.9],
            'ai_efficiency_improvement': [0.02, 0.035, 0.05, 0.065, 0.08],
            'battery_replacement_cost': [10000, 12500, 15000, 17500, 20000],
            'energy_cost_per_kwh': [0.08, 0.10, 0.12, 0.14, 0.16]
        }
        
        baseline_performance_data = {'annual_energy_usage_kwh': fleet_size * 10000}
        
        for param_name, param_values in sensitivity_params.items():
            param_results = []
            
            for value in param_values:
                # Temporarily modify configuration
                original_value = getattr(self.config, param_name)
                setattr(self.config, param_name, value)
                
                # Recalculate ROI
                cost_analysis = self._calculate_cost_components(fleet_size)
                benefit_analysis = self._calculate_benefit_components(baseline_performance_data, fleet_size)
                roi_metrics = self._calculate_roi_metrics(cost_analysis, benefit_analysis)
                
                param_results.append({
                    'parameter_value': value,
                    'npv': roi_metrics['net_present_value'],
                    'roi_percentage': roi_metrics['roi_percentage'],
                    'payback_period': roi_metrics['payback_period_years']
                })
                
                # Restore original value
                setattr(self.config, param_name, original_value)
            
            sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def _perform_scenario_analysis(self, fleet_size: int) -> Dict[str, Any]:
        """Perform scenario analysis with optimistic, base, and pessimistic cases."""
        scenarios = {
            'pessimistic': {
                'ai_life_extension_factor': 1.2,
                'ai_maintenance_reduction_factor': 0.8,
                'ai_efficiency_improvement': 0.02,
                'implementation_cost_multiplier': 1.3
            },
            'base': {
                'ai_life_extension_factor': 1.4,
                'ai_maintenance_reduction_factor': 0.7,
                'ai_efficiency_improvement': 0.05,
                'implementation_cost_multiplier': 1.0
            },
            'optimistic': {
                'ai_life_extension_factor': 1.6,
                'ai_maintenance_reduction_factor': 0.5,
                'ai_efficiency_improvement': 0.08,
                'implementation_cost_multiplier': 0.8
            }
        }
        
        scenario_results = {}
        baseline_performance_data = {'annual_energy_usage_kwh': fleet_size * 10000}
        
        for scenario_name, scenario_params in scenarios.items():
            # Store original values
            original_values = {}
            for param, value in scenario_params.items():
                if param != 'implementation_cost_multiplier':
                    original_values[param] = getattr(self.config, param)
                    setattr(self.config, param, value)
            
            # Adjust implementation cost
            original_impl_cost = self.config.ai_implementation_cost
            self.config.ai_implementation_cost *= scenario_params['implementation_cost_multiplier']
            
            # Calculate scenario results
            cost_analysis = self._calculate_cost_components(fleet_size)
            benefit_analysis = self._calculate_benefit_components(baseline_performance_data, fleet_size)
            roi_metrics = self._calculate_roi_metrics(cost_analysis, benefit_analysis)
            
            scenario_results[scenario_name] = {
                'scenario_parameters': scenario_params,
                'npv': roi_metrics['net_present_value'],
                'roi_percentage': roi_metrics['roi_percentage'],
                'payback_period': roi_metrics['payback_period_years'],
                'benefit_cost_ratio': roi_metrics['benefit_cost_ratio']
            }
            
            # Restore original values
            for param, value in original_values.items():
                setattr(self.config, param, value)
            self.config.ai_implementation_cost = original_impl_cost
        
        return scenario_results
    
    def _calculate_risk_adjusted_returns(self, roi_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate risk-adjusted returns and confidence intervals."""
        base_npv = roi_analysis['net_present_value']
        base_roi = roi_analysis['roi_percentage']
        
        # Apply risk adjustments
        implementation_risk_adjustment = base_npv * self.config.implementation_risk_factor
        technology_risk_adjustment = base_npv * self.config.technology_obsolescence_risk * self.config.analysis_period_years
        
        risk_adjusted_npv = base_npv - implementation_risk_adjustment - technology_risk_adjustment
        risk_adjusted_roi = (risk_adjusted_npv / roi_analysis.get('total_investment', 1)) * 100
        
        # Calculate confidence intervals (simplified Monte Carlo approach)
        confidence_intervals = {}
        for confidence_level in self.config.confidence_intervals:
            # Simplified confidence interval calculation
            z_score = stats.norm.ppf((1 + confidence_level) / 2)
            std_dev = base_npv * 0.2  # Assume 20% standard deviation
            
            lower_bound = risk_adjusted_npv - z_score * std_dev
            upper_bound = risk_adjusted_npv + z_score * std_dev
            
            confidence_intervals[f'{int(confidence_level*100)}%'] = {
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        
        return {
            'base_npv': base_npv,
            'risk_adjusted_npv': risk_adjusted_npv,
            'base_roi': base_roi,
            'risk_adjusted_roi': risk_adjusted_roi,
            'implementation_risk_impact': implementation_risk_adjustment,
            'technology_risk_impact': technology_risk_adjustment,
            'confidence_intervals': confidence_intervals,
            'risk_assessment': self._assess_investment_risk(risk_adjusted_npv, risk_adjusted_roi)
        }
    
    def _assess_investment_risk(self, risk_adjusted_npv: float, risk_adjusted_roi: float) -> str:
        """Assess overall investment risk level."""
        if risk_adjusted_npv > 0 and risk_adjusted_roi > 15:
            return "Low Risk - Strong positive returns expected"
        elif risk_adjusted_npv > 0 and risk_adjusted_roi > 8:
            return "Medium Risk - Positive returns likely"
        elif risk_adjusted_npv > 0:
            return "Medium-High Risk - Marginal positive returns"
        else:
            return "High Risk - Negative returns possible"
    
    def _generate_executive_summary(self, 
                                  roi_analysis: Dict[str, Any], 
                                  benefit_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate executive summary of business impact."""
        npv = roi_analysis['net_present_value']
        roi_pct = roi_analysis['roi_percentage']
        payback_period = roi_analysis['payback_period_years']
        annual_benefits = benefit_analysis['total_annual_benefits']
        
        summary = {
            'investment_recommendation': 'APPROVE' if npv > 0 and roi_pct > 10 else 'REVIEW' if npv > 0 else 'REJECT',
            'key_financial_metrics': f"NPV: ${npv:,.0f}, ROI: {roi_pct:.1f}%, Payback: {payback_period:.1f} years",
            'annual_benefits': f"${annual_benefits:,.0f} in annual benefits",
            'primary_value_drivers': "Battery life extension, maintenance reduction, and energy efficiency gains",
            'risk_assessment': "Medium risk with strong upside potential in optimistic scenarios",
            'strategic_importance': "Critical for competitive advantage and operational excellence"
        }
        
        return summary
    
    def _generate_business_recommendations(self, 
                                         roi_analysis: Dict[str, Any], 
                                         risk_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable business recommendations."""
        recommendations = []
        
        npv = roi_analysis['net_present_value']
        roi_pct = roi_analysis['roi_percentage']
        payback_period = roi_analysis['payback_period_years']
        
        # Investment decision recommendations
        if npv > 0 and roi_pct > 15:
            recommendations.append("STRONG RECOMMENDATION: Proceed with AI implementation immediately")
            recommendations.append("Consider accelerating implementation timeline to capture benefits sooner")
        elif npv > 0 and roi_pct > 8:
            recommendations.append("RECOMMENDATION: Proceed with AI implementation with careful risk management")
            recommendations.append("Develop detailed implementation plan with milestone-based go/no-go decisions")
        elif npv > 0:
            recommendations.append("CONDITIONAL RECOMMENDATION: Proceed only if strategic benefits justify risks")
            recommendations.append("Consider pilot implementation to validate assumptions")
        else:
            recommendations.append("NOT RECOMMENDED: ROI does not justify investment at current assumptions")
            recommendations.append("Reassess technology options or wait for cost reductions")
        
        # Operational recommendations
        if payback_period < 2:
            recommendations.append("Fast payback period enables aggressive expansion across fleet")
        elif payback_period < 4:
            recommendations.append("Reasonable payback period supports phased implementation approach")
        else:
            recommendations.append("Long payback period requires careful cash flow management")
        
        # Risk mitigation recommendations
        recommendations.extend([
            "Implement robust change management program to ensure adoption",
            "Establish clear performance metrics and monitoring systems",
            "Develop contingency plans for technology performance shortfalls",
            "Consider staged implementation to minimize risk exposure"
        ])
        
        return recommendations
    
    def generate_business_case_report(self, 
                                    output_path: Optional[str] = None,
                                    include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive business case report.
        
        Args:
            output_path: Optional path to save report
            include_visualizations: Whether to generate visualizations
            
        Returns:
            Complete business case report
        """
        if not self.analysis_results:
            raise ValueError("No analysis results available. Run analyze_financial_impact() first.")
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': 'Business Impact Analysis',
                'analysis_config': self.config.__dict__
            },
            'executive_summary': self.analysis_results['executive_summary'],
            'financial_analysis': {
                'roi_analysis': self.analysis_results['roi_analysis'],
                'cost_benefit_breakdown': {
                    'costs': self.analysis_results['cost_analysis'],
                    'benefits': self.analysis_results['benefit_analysis']
                }
            },
            'risk_assessment': self.analysis_results['risk_analysis'],
            'sensitivity_analysis': self.analysis_results.get('sensitivity_analysis', {}),
            'scenario_analysis': self.analysis_results.get('scenario_analysis', {}),
            'recommendations': self.analysis_results['recommendations'],
            'appendices': {
                'assumptions': self._document_assumptions(),
                'methodology': self._document_methodology()
            }
        }
        
        # Generate visualizations if requested
        if include_visualizations:
            visualizations = self._generate_business_visualizations()
            report['visualizations'] = visualizations
        
        # Save report if path provided
        if output_path:
            self._save_business_report(report, output_path)
        
        return report
    
    def _document_assumptions(self) -> Dict[str, Any]:
        """Document key assumptions used in analysis."""
        return {
            'financial_assumptions': {
                'discount_rate': f"{self.config.discount_rate*100}% annual discount rate",
                'analysis_period': f"{self.config.analysis_period_years} year analysis period",
                'currency': self.config.currency
            },
            'operational_assumptions': {
                'baseline_battery_life': f"{self.config.baseline_battery_life_years} years without AI",
                'ai_life_extension': f"{(self.config.ai_life_extension_factor-1)*100}% life extension with AI",
                'maintenance_reduction': f"{(1-self.config.ai_maintenance_reduction_factor)*100}% maintenance reduction",
                'efficiency_improvement': f"{self.config.ai_efficiency_improvement*100}% energy efficiency gain"
            },
            'cost_assumptions': {
                'battery_replacement_cost': f"${self.config.battery_replacement_cost:,} per battery",
                'energy_cost': f"${self.config.energy_cost_per_kwh}/kWh",
                'maintenance_cost': f"${self.config.maintenance_cost_per_hour}/hour",
                'downtime_cost': f"${self.config.downtime_cost_per_hour}/hour"
            }
        }
    
    def _document_methodology(self) -> Dict[str, str]:
        """Document analysis methodology."""
        return {
            'financial_methodology': "Net Present Value (NPV) and Return on Investment (ROI) calculated using discounted cash flow analysis",
            'risk_adjustment': "Risk factors applied to baseline projections based on implementation and technology risks",
            'sensitivity_analysis': "Key parameters varied across realistic ranges to test robustness of results",
            'scenario_analysis': "Optimistic, base, and pessimistic scenarios developed to bracket potential outcomes"
        }
    
    def _generate_business_visualizations(self) -> List[str]:
        """Generate business impact visualizations."""
        visualizations = []
        
        try:
            # 1. NPV breakdown chart
            self._create_npv_breakdown_chart()
            visualizations.append("npv_breakdown.png")
            
            # 2. Payback period chart
            self._create_payback_chart()
            visualizations.append("payback_analysis.png")
            
            # 3. Sensitivity analysis chart
            self._create_sensitivity_chart()
            visualizations.append("sensitivity_analysis.png")
            
            # 4. Scenario comparison chart
            self._create_scenario_comparison_chart()
            visualizations.append("scenario_comparison.png")
            
        except Exception as e:
            logger.error(f"Error generating business visualizations: {e}")
        
        return visualizations
    
    def _create_npv_breakdown_chart(self):
        """Create NPV breakdown visualization."""
        benefits = self.analysis_results['benefit_analysis']['annual_benefit_components']
        costs = self.analysis_results['cost_analysis']
        
        # Create stacked bar chart of benefits vs costs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Benefits breakdown
        benefit_labels = list(benefits.keys())
        benefit_values = list(benefits.values())
        
        ax1.pie(benefit_values, labels=benefit_labels, autopct='%1.1f%%')
        ax1.set_title('Annual Benefits Breakdown')
        
        # Costs vs Benefits comparison
        total_benefits = sum(benefit_values)
        total_costs = costs['total_annual_operating_cost']
        
        ax2.bar(['Annual Benefits', 'Annual Costs'], [total_benefits, total_costs], 
               color=['green', 'red'], alpha=0.7)
        ax2.set_title('Annual Benefits vs Costs')
        ax2.set_ylabel('Amount ($)')
        
        plt.tight_layout()
        plt.savefig('npv_breakdown.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_payback_chart(self):
        """Create payback period visualization."""
        roi_data = self.analysis_results['roi_analysis']['annual_roi_breakdown']
        
        years = [r['year'] for r in roi_data]
        cumulative_cash_flow = [r['cumulative_net_cash_flow'] for r in roi_data]
        
        plt.figure(figsize=(10, 6))
        plt.plot(years, cumulative_cash_flow, marker='o', linewidth=2, markersize=8)
        plt.axhline(y=0, color='red', linestyle='--', alpha=0.7, label='Breakeven')
        plt.xlabel('Year')
        plt.ylabel('Cumulative Net Cash Flow ($)')
        plt.title('Investment Payback Analysis')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Highlight payback point
        payback_year = self.analysis_results['roi_analysis']['payback_period_years']
        if payback_year <= len(years):
            plt.axvline(x=payback_year, color='green', linestyle=':', alpha=0.7, 
                       label=f'Payback: {payback_year:.1f} years')
            plt.legend()
        
        plt.savefig('payback_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_sensitivity_chart(self):
        """Create sensitivity analysis visualization."""
        sensitivity_data = self.analysis_results.get('sensitivity_analysis', {})
        
        if not sensitivity_data:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, (param_name, param_results) in enumerate(sensitivity_data.items()):
            if i >= len(axes):
                break
                
            param_values = [r['parameter_value'] for r in param_results]
            npv_values = [r['npv'] for r in param_results]
            
            axes[i].plot(param_values, npv_values, marker='o', linewidth=2)
            axes[i].set_xlabel(param_name.replace('_', ' ').title())
            axes[i].set_ylabel('NPV ($)')
            axes[i].set_title(f'Sensitivity to {param_name.replace("_", " ").title()}')
            axes[i].grid(True, alpha=0.3)
            axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Remove unused subplots
        for i in range(len(sensitivity_data), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.savefig('sensitivity_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_scenario_comparison_chart(self):
        """Create scenario comparison visualization."""
        scenario_data = self.analysis_results.get('scenario_analysis', {})
        
        if not scenario_data:
            return
        
        scenarios = list(scenario_data.keys())
        npv_values = [scenario_data[s]['npv'] for s in scenarios]
        roi_values = [scenario_data[s]['roi_percentage'] for s in scenarios]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # NPV comparison
        colors = ['red', 'blue', 'green']
        ax1.bar(scenarios, npv_values, color=colors[:len(scenarios)], alpha=0.7)
        ax1.set_title('NPV by Scenario')
        ax1.set_ylabel('Net Present Value ($)')
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # ROI comparison
        ax2.bar(scenarios, roi_values, color=colors[:len(scenarios)], alpha=0.7)
        ax2.set_title('ROI by Scenario')
        ax2.set_ylabel('ROI (%)')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('scenario_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_business_report(self, report: Dict[str, Any], output_path: str):
        """Save business impact report to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Business impact report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save business report: {e}")

# Factory function
def create_business_impact_analyzer(config: Optional[BusinessImpactConfig] = None) -> BusinessImpactAnalyzer:
    """
    Factory function to create a business impact analyzer.
    
    Args:
        config: Business impact analysis configuration
        
    Returns:
        Configured BusinessImpactAnalyzer instance
    """
    if config is None:
        config = BusinessImpactConfig()
    
    return BusinessImpactAnalyzer(config)
