"""
BatteryMind - Model Comparison Report Module

Comprehensive model comparison and analysis framework for battery management
AI/ML models. This module provides detailed comparative analysis across
different model architectures, performance metrics, and deployment scenarios.

Features:
- Multi-model performance comparison and benchmarking
- Statistical significance testing for model differences
- Visual comparison dashboards and reporting
- Model selection recommendations based on use case
- Performance degradation analysis over time
- Resource utilization and efficiency comparisons
- Cross-validation and holdout test evaluations

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
from pathlib import Path
import json
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical analysis imports
from scipy import stats
from scipy.stats import ttest_ind, wilcoxon, ks_2samp
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm

# Internal imports
from ..metrics.accuracy_metrics import AccuracyMetrics
from ..metrics.performance_metrics import PerformanceMetrics
from ..metrics.efficiency_metrics import EfficiencyMetrics
from ..metrics.business_metrics import BusinessMetrics
from ...utils.visualization import ModelVisualization
from ...utils.logging_utils import setup_logger

# Configure logging
logger = setup_logger(__name__)

@dataclass
class ModelComparisonConfig:
    """Configuration for model comparison analysis."""
    
    # Comparison parameters
    significance_level: float = 0.05
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000
    cross_validation_folds: int = 5
    
    # Metrics to compare
    regression_metrics: List[str] = field(default_factory=lambda: [
        'mse', 'mae', 'rmse', 'r2_score', 'mape'
    ])
    classification_metrics: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1_score', 'auc_roc'
    ])
    performance_metrics: List[str] = field(default_factory=lambda: [
        'inference_time', 'memory_usage', 'cpu_utilization', 'throughput'
    ])
    
    # Visualization settings
    figure_size: Tuple[int, int] = (12, 8)
    color_palette: str = "Set2"
    save_plots: bool = True
    plot_format: str = "png"
    
    # Report settings
    include_statistical_tests: bool = True
    include_confidence_intervals: bool = True
    include_effect_sizes: bool = True
    generate_recommendations: bool = True

class ModelComparator:
    """
    Advanced model comparison and analysis system.
    """
    
    def __init__(self, config: ModelComparisonConfig):
        self.config = config
        self.models_data = {}
        self.comparison_results = {}
        self.statistical_tests = {}
        
        # Initialize metrics calculators
        self.accuracy_metrics = AccuracyMetrics()
        self.performance_metrics = PerformanceMetrics()
        self.efficiency_metrics = EfficiencyMetrics()
        self.business_metrics = BusinessMetrics()
        
        logger.info("Model comparator initialized")
    
    def add_model_results(self, 
                         model_name: str, 
                         predictions: np.ndarray,
                         ground_truth: np.ndarray,
                         performance_data: Dict[str, float],
                         metadata: Optional[Dict[str, Any]] = None):
        """
        Add model results for comparison.
        
        Args:
            model_name: Name/identifier for the model
            predictions: Model predictions
            ground_truth: Ground truth values
            performance_data: Performance metrics (inference time, memory, etc.)
            metadata: Additional model metadata
        """
        self.models_data[model_name] = {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'performance_data': performance_data,
            'metadata': metadata or {},
            'timestamp': datetime.now()
        }
        
        logger.info(f"Added results for model: {model_name}")
    
    def compare_models(self, model_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive model comparison.
        
        Args:
            model_names: List of model names to compare (all if None)
            
        Returns:
            Comprehensive comparison results
        """
        if model_names is None:
            model_names = list(self.models_data.keys())
        
        if len(model_names) < 2:
            raise ValueError("At least 2 models required for comparison")
        
        logger.info(f"Comparing {len(model_names)} models: {model_names}")
        
        # Calculate metrics for all models
        metrics_comparison = self._calculate_metrics_comparison(model_names)
        
        # Perform statistical tests
        statistical_results = self._perform_statistical_tests(model_names)
        
        # Calculate performance comparison
        performance_comparison = self._calculate_performance_comparison(model_names)
        
        # Generate rankings
        model_rankings = self._generate_model_rankings(model_names, metrics_comparison)
        
        # Create comparison summary
        comparison_summary = {
            'models_compared': model_names,
            'comparison_timestamp': datetime.now().isoformat(),
            'metrics_comparison': metrics_comparison,
            'statistical_results': statistical_results,
            'performance_comparison': performance_comparison,
            'model_rankings': model_rankings,
            'recommendations': self._generate_recommendations(model_names, metrics_comparison, statistical_results)
        }
        
        self.comparison_results = comparison_summary
        
        return comparison_summary
    
    def _calculate_metrics_comparison(self, model_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate comparison metrics for all models."""
        metrics_comparison = {}
        
        for model_name in model_names:
            if model_name not in self.models_data:
                logger.warning(f"Model {model_name} not found in data")
                continue
            
            model_data = self.models_data[model_name]
            predictions = model_data['predictions']
            ground_truth = model_data['ground_truth']
            
            # Calculate accuracy metrics
            model_metrics = {}
            
            # Regression metrics
            if len(predictions.shape) == 1 or predictions.shape[1] == 1:
                model_metrics.update({
                    'mse': mean_squared_error(ground_truth, predictions),
                    'mae': mean_absolute_error(ground_truth, predictions),
                    'rmse': np.sqrt(mean_squared_error(ground_truth, predictions)),
                    'r2_score': r2_score(ground_truth, predictions),
                    'mape': np.mean(np.abs((ground_truth - predictions) / ground_truth)) * 100
                })
            
            # Performance metrics
            performance_data = model_data['performance_data']
            model_metrics.update({
                'inference_time_ms': performance_data.get('inference_time_ms', 0.0),
                'memory_usage_mb': performance_data.get('memory_usage_mb', 0.0),
                'cpu_utilization': performance_data.get('cpu_utilization', 0.0),
                'throughput_rps': performance_data.get('throughput_rps', 0.0)
            })
            
            # Custom battery-specific metrics
            model_metrics.update(self._calculate_battery_specific_metrics(predictions, ground_truth))
            
            metrics_comparison[model_name] = model_metrics
        
        return metrics_comparison
    
    def _calculate_battery_specific_metrics(self, predictions: np.ndarray, ground_truth: np.ndarray) -> Dict[str, float]:
        """Calculate battery-specific evaluation metrics."""
        battery_metrics = {}
        
        # State of Health (SoH) specific metrics
        if 'soh' in str(type(predictions)).lower():
            # SoH accuracy within tolerance
            tolerance = 0.05  # 5% tolerance
            within_tolerance = np.abs(predictions - ground_truth) <= tolerance
            battery_metrics['soh_accuracy_5pct'] = np.mean(within_tolerance) * 100
            
            # Critical threshold detection (SoH < 0.8)
            critical_threshold = 0.8
            true_critical = ground_truth < critical_threshold
            pred_critical = predictions < critical_threshold
            
            if np.any(true_critical):
                battery_metrics['critical_detection_precision'] = precision_score(true_critical, pred_critical)
                battery_metrics['critical_detection_recall'] = recall_score(true_critical, pred_critical)
        
        # Remaining Useful Life (RUL) specific metrics
        if 'rul' in str(type(predictions)).lower():
            # RUL prediction horizon accuracy
            horizons = [30, 90, 365]  # days
            for horizon in horizons:
                horizon_mask = ground_truth <= horizon
                if np.any(horizon_mask):
                    horizon_mae = mean_absolute_error(
                        ground_truth[horizon_mask], 
                        predictions[horizon_mask]
                    )
                    battery_metrics[f'rul_mae_{horizon}d'] = horizon_mae
        
        # Thermal management metrics
        if 'temperature' in str(type(predictions)).lower():
            # Temperature safety violations
            safe_temp_range = (5, 45)  # °C
            temp_violations_true = (ground_truth < safe_temp_range[0]) | (ground_truth > safe_temp_range[1])
            temp_violations_pred = (predictions < safe_temp_range[0]) | (predictions > safe_temp_range[1])
            
            if np.any(temp_violations_true):
                battery_metrics['temp_violation_detection'] = f1_score(temp_violations_true, temp_violations_pred)
        
        return battery_metrics
    
    def _perform_statistical_tests(self, model_names: List[str]) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests between models."""
        statistical_results = {}
        
        if not self.config.include_statistical_tests:
            return statistical_results
        
        # Pairwise comparisons
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                comparison_key = f"{model1}_vs_{model2}"
                
                model1_data = self.models_data[model1]
                model2_data = self.models_data[model2]
                
                # Calculate residuals for both models
                residuals1 = model1_data['predictions'] - model1_data['ground_truth']
                residuals2 = model2_data['predictions'] - model2_data['ground_truth']
                
                # Paired t-test (if same test set)
                if len(residuals1) == len(residuals2):
                    try:
                        t_stat, p_value = ttest_ind(residuals1, residuals2)
                        statistical_results[comparison_key] = {
                            'test_type': 'paired_t_test',
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'significant': p_value < self.config.significance_level,
                            'effect_size': self._calculate_cohens_d(residuals1, residuals2)
                        }
                    except Exception as e:
                        logger.warning(f"T-test failed for {comparison_key}: {e}")
                
                # Wilcoxon rank-sum test (non-parametric)
                try:
                    w_stat, w_p_value = wilcoxon(np.abs(residuals1), np.abs(residuals2))
                    statistical_results[f"{comparison_key}_wilcoxon"] = {
                        'test_type': 'wilcoxon_rank_sum',
                        'w_statistic': float(w_stat),
                        'p_value': float(w_p_value),
                        'significant': w_p_value < self.config.significance_level
                    }
                except Exception as e:
                    logger.warning(f"Wilcoxon test failed for {comparison_key}: {e}")
                
                # Kolmogorov-Smirnov test
                try:
                    ks_stat, ks_p_value = ks_2samp(np.abs(residuals1), np.abs(residuals2))
                    statistical_results[f"{comparison_key}_ks"] = {
                        'test_type': 'kolmogorov_smirnov',
                        'ks_statistic': float(ks_stat),
                        'p_value': float(ks_p_value),
                        'significant': ks_p_value < self.config.significance_level
                    }
                except Exception as e:
                    logger.warning(f"KS test failed for {comparison_key}: {e}")
        
        return statistical_results
    
    def _calculate_cohens_d(self, group1: np.ndarray, group2: np.ndarray) -> float:
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _calculate_performance_comparison(self, model_names: List[str]) -> Dict[str, Dict[str, float]]:
        """Calculate performance comparison metrics."""
        performance_comparison = {}
        
        # Extract performance data
        performance_data = {}
        for model_name in model_names:
            if model_name in self.models_data:
                performance_data[model_name] = self.models_data[model_name]['performance_data']
        
        # Calculate relative performance
        for metric in self.config.performance_metrics:
            metric_values = {}
            for model_name in model_names:
                if model_name in performance_data:
                    metric_values[model_name] = performance_data[model_name].get(metric, 0.0)
            
            if metric_values:
                # Find best and worst values
                if metric in ['inference_time', 'memory_usage', 'cpu_utilization']:
                    # Lower is better
                    best_value = min(metric_values.values())
                    worst_value = max(metric_values.values())
                else:
                    # Higher is better
                    best_value = max(metric_values.values())
                    worst_value = min(metric_values.values())
                
                # Calculate relative performance
                for model_name, value in metric_values.items():
                    if model_name not in performance_comparison:
                        performance_comparison[model_name] = {}
                    
                    if worst_value != best_value:
                        if metric in ['inference_time', 'memory_usage', 'cpu_utilization']:
                            # Lower is better - normalize inversely
                            relative_perf = (worst_value - value) / (worst_value - best_value)
                        else:
                            # Higher is better
                            relative_perf = (value - worst_value) / (best_value - worst_value)
                    else:
                        relative_perf = 1.0
                    
                    performance_comparison[model_name][f'{metric}_relative'] = relative_perf
        
        return performance_comparison
    
    def _generate_model_rankings(self, 
                                model_names: List[str], 
                                metrics_comparison: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Generate model rankings based on multiple criteria."""
        rankings = {}
        
        # Accuracy-based ranking
        accuracy_scores = {}
        for model_name in model_names:
            if model_name in metrics_comparison:
                metrics = metrics_comparison[model_name]
                # Composite accuracy score (lower is better for error metrics)
                score = 0.0
                count = 0
                
                if 'mse' in metrics:
                    score += 1.0 / (1.0 + metrics['mse'])  # Inverse for lower-is-better
                    count += 1
                if 'mae' in metrics:
                    score += 1.0 / (1.0 + metrics['mae'])
                    count += 1
                if 'r2_score' in metrics:
                    score += metrics['r2_score']  # Higher is better
                    count += 1
                
                accuracy_scores[model_name] = score / max(count, 1)
        
        rankings['accuracy'] = sorted(accuracy_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Performance-based ranking
        performance_scores = {}
        for model_name in model_names:
            if model_name in metrics_comparison:
                metrics = metrics_comparison[model_name]
                # Composite performance score
                score = 0.0
                count = 0
                
                if 'inference_time_ms' in metrics:
                    score += 1.0 / (1.0 + metrics['inference_time_ms'] / 1000.0)  # Seconds
                    count += 1
                if 'memory_usage_mb' in metrics:
                    score += 1.0 / (1.0 + metrics['memory_usage_mb'] / 1000.0)  # GB
                    count += 1
                if 'throughput_rps' in metrics:
                    score += metrics['throughput_rps'] / 1000.0  # Normalize
                    count += 1
                
                performance_scores[model_name] = score / max(count, 1)
        
        rankings['performance'] = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Overall ranking (weighted combination)
        overall_scores = {}
        accuracy_weight = 0.6
        performance_weight = 0.4
        
        for model_name in model_names:
            accuracy_score = accuracy_scores.get(model_name, 0.0)
            performance_score = performance_scores.get(model_name, 0.0)
            
            overall_score = (accuracy_weight * accuracy_score + 
                           performance_weight * performance_score)
            overall_scores[model_name] = overall_score
        
        rankings['overall'] = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        
        return rankings
    
    def _generate_recommendations(self, 
                                model_names: List[str],
                                metrics_comparison: Dict[str, Dict[str, float]],
                                statistical_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate model selection recommendations."""
        recommendations = []
        
        if not self.config.generate_recommendations:
            return recommendations
        
        # Find best performing model overall
        if self.comparison_results and 'model_rankings' in self.comparison_results:
            rankings = self.comparison_results['model_rankings']
            if 'overall' in rankings and rankings['overall']:
                best_model = rankings['overall'][0][0]
                recommendations.append(f"Overall best performing model: {best_model}")
        
        # Recommendations based on use case
        recommendations.extend([
            "For real-time applications: Consider models with lowest inference time",
            "For edge deployment: Prioritize models with lowest memory usage",
            "For critical safety applications: Choose models with highest accuracy",
            "For large-scale deployment: Balance accuracy and performance requirements"
        ])
        
        # Statistical significance recommendations
        significant_differences = []
        for test_name, result in statistical_results.items():
            if result.get('significant', False):
                significant_differences.append(test_name)
        
        if significant_differences:
            recommendations.append(f"Statistically significant differences found in {len(significant_differences)} comparisons")
        else:
            recommendations.append("No statistically significant differences found between models")
        
        return recommendations
    
    def generate_comparison_report(self, 
                                 output_path: Optional[str] = None,
                                 include_visualizations: bool = True) -> Dict[str, Any]:
        """
        Generate comprehensive model comparison report.
        
        Args:
            output_path: Optional path to save report
            include_visualizations: Whether to generate visualizations
            
        Returns:
            Complete comparison report
        """
        if not self.comparison_results:
            raise ValueError("No comparison results available. Run compare_models() first.")
        
        report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'models_analyzed': len(self.comparison_results['models_compared']),
                'comparison_config': self.config.__dict__
            },
            'executive_summary': self._generate_executive_summary(),
            'detailed_results': self.comparison_results,
            'visualizations': [],
            'conclusions': self._generate_conclusions()
        }
        
        # Generate visualizations if requested
        if include_visualizations:
            visualizations = self._generate_visualizations()
            report['visualizations'] = visualizations
        
        # Save report if path provided
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _generate_executive_summary(self) -> Dict[str, Any]:
        """Generate executive summary of comparison results."""
        models_compared = self.comparison_results['models_compared']
        rankings = self.comparison_results.get('model_rankings', {})
        
        summary = {
            'total_models_compared': len(models_compared),
            'best_overall_model': rankings.get('overall', [('Unknown', 0)])[0][0] if rankings.get('overall') else 'Unknown',
            'best_accuracy_model': rankings.get('accuracy', [('Unknown', 0)])[0][0] if rankings.get('accuracy') else 'Unknown',
            'best_performance_model': rankings.get('performance', [('Unknown', 0)])[0][0] if rankings.get('performance') else 'Unknown',
            'key_findings': self._extract_key_findings(),
            'recommendation': self._get_primary_recommendation()
        }
        
        return summary
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from comparison results."""
        findings = []
        
        metrics_comparison = self.comparison_results.get('metrics_comparison', {})
        
        if metrics_comparison:
            # Find model with best accuracy
            best_r2 = -float('inf')
            best_r2_model = None
            
            for model_name, metrics in metrics_comparison.items():
                if 'r2_score' in metrics and metrics['r2_score'] > best_r2:
                    best_r2 = metrics['r2_score']
                    best_r2_model = model_name
            
            if best_r2_model:
                findings.append(f"{best_r2_model} achieved highest R² score of {best_r2:.4f}")
            
            # Find fastest model
            fastest_time = float('inf')
            fastest_model = None
            
            for model_name, metrics in metrics_comparison.items():
                if 'inference_time_ms' in metrics and metrics['inference_time_ms'] < fastest_time:
                    fastest_time = metrics['inference_time_ms']
                    fastest_model = model_name
            
            if fastest_model:
                findings.append(f"{fastest_model} has fastest inference time of {fastest_time:.2f}ms")
        
        # Statistical significance findings
        statistical_results = self.comparison_results.get('statistical_results', {})
        significant_count = sum(1 for result in statistical_results.values() if result.get('significant', False))
        
        if significant_count > 0:
            findings.append(f"{significant_count} statistically significant differences found")
        else:
            findings.append("No statistically significant performance differences detected")
        
        return findings
    
    def _get_primary_recommendation(self) -> str:
        """Get primary model recommendation."""
        rankings = self.comparison_results.get('model_rankings', {})
        
        if 'overall' in rankings and rankings['overall']:
            best_model = rankings['overall'][0][0]
            return f"Recommend {best_model} for balanced accuracy and performance requirements"
        
        return "Insufficient data for specific recommendation"
    
    def _generate_conclusions(self) -> List[str]:
        """Generate conclusions from comparison analysis."""
        conclusions = []
        
        recommendations = self.comparison_results.get('recommendations', [])
        conclusions.extend(recommendations)
        
        # Add technical conclusions
        conclusions.extend([
            "Model selection should consider specific deployment requirements",
            "Regular retraining may be needed to maintain performance",
            "Monitor model performance in production environment",
            "Consider ensemble approaches for critical applications"
        ])
        
        return conclusions
    
    def _generate_visualizations(self) -> List[str]:
        """Generate comparison visualizations."""
        visualizations = []
        
        try:
            # Set up plotting style
            plt.style.use('default')
            sns.set_palette(self.config.color_palette)
            
            # 1. Metrics comparison bar plot
            self._create_metrics_comparison_plot()
            visualizations.append("metrics_comparison.png")
            
            # 2. Performance scatter plot
            self._create_performance_scatter_plot()
            visualizations.append("performance_comparison.png")
            
            # 3. Statistical significance heatmap
            self._create_significance_heatmap()
            visualizations.append("statistical_significance.png")
            
            # 4. Model rankings radar chart
            self._create_rankings_radar_chart()
            visualizations.append("model_rankings_radar.png")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return visualizations
    
    def _create_metrics_comparison_plot(self):
        """Create metrics comparison bar plot."""
        metrics_data = self.comparison_results.get('metrics_comparison', {})
        
        if not metrics_data:
            return
        
        # Prepare data for plotting
        models = list(metrics_data.keys())
        metrics_to_plot = ['mae', 'rmse', 'r2_score']
        
        fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(15, 5))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics_data[model].get(metric, 0) for model in models]
            
            axes[i].bar(models, values)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        if self.config.save_plots:
            plt.savefig('metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_performance_scatter_plot(self):
        """Create performance vs accuracy scatter plot."""
        metrics_data = self.comparison_results.get('metrics_comparison', {})
        
        if not metrics_data:
            return
        
        models = list(metrics_data.keys())
        accuracy_scores = [metrics_data[model].get('r2_score', 0) for model in models]
        inference_times = [metrics_data[model].get('inference_time_ms', 0) for model in models]
        
        plt.figure(figsize=self.config.figure_size)
        
        for i, model in enumerate(models):
            plt.scatter(inference_times[i], accuracy_scores[i], s=100, label=model)
            plt.annotate(model, (inference_times[i], accuracy_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('R² Score')
        plt.title('Model Performance vs Accuracy Trade-off')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.config.save_plots:
            plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_significance_heatmap(self):
        """Create statistical significance heatmap."""
        statistical_results = self.comparison_results.get('statistical_results', {})
        
        if not statistical_results:
            return
        
        # Extract p-values for heatmap
        models = self.comparison_results['models_compared']
        n_models = len(models)
        p_value_matrix = np.ones((n_models, n_models))
        
        for i, model1 in enumerate(models):
            for j, model2 in enumerate(models):
                if i != j:
                    comparison_key = f"{model1}_vs_{model2}"
                    reverse_key = f"{model2}_vs_{model1}"
                    
                    if comparison_key in statistical_results:
                        p_value_matrix[i, j] = statistical_results[comparison_key]['p_value']
                    elif reverse_key in statistical_results:
                        p_value_matrix[i, j] = statistical_results[reverse_key]['p_value']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(p_value_matrix, annot=True, xticklabels=models, yticklabels=models,
                   cmap='RdYlBu_r', center=0.05, cbar_kws={'label': 'p-value'})
        plt.title('Statistical Significance Heatmap\n(Red = Significant Difference)')
        
        if self.config.save_plots:
            plt.savefig('statistical_significance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_rankings_radar_chart(self):
        """Create radar chart for model rankings."""
        try:
            from math import pi
            
            rankings = self.comparison_results.get('model_rankings', {})
            models = self.comparison_results['models_compared']
            
            # Create radar chart data
            categories = ['Accuracy', 'Performance', 'Overall']
            
            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
            
            angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
            angles += angles[:1]
            
            for model in models:
                values = []
                for category in categories:
                    rank_list = rankings.get(category.lower(), [])
                    rank = next((i for i, (name, _) in enumerate(rank_list) if name == model), len(models))
                    # Invert rank so higher is better
                    values.append(len(models) - rank)
                
                values += values[:1]
                
                ax.plot(angles, values, 'o-', linewidth=2, label=model)
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            ax.set_ylim(0, len(models))
            ax.set_title('Model Rankings Comparison', size=20, y=1.1)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            if self.config.save_plots:
                plt.savefig('model_rankings_radar.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not create radar chart: {e}")
    
    def _save_report(self, report: Dict[str, Any], output_path: str):
        """Save comparison report to file."""
        try:
            # Save as JSON
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Comparison report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save report: {e}")

# Factory function
def create_model_comparator(config: Optional[ModelComparisonConfig] = None) -> ModelComparator:
    """
    Factory function to create a model comparator.
    
    Args:
        config: Model comparison configuration
        
    Returns:
        Configured ModelComparator instance
    """
    if config is None:
        config = ModelComparisonConfig()
    
    return ModelComparator(config)
