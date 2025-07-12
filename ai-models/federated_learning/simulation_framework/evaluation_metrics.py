"""
BatteryMind - Federated Learning Evaluation Metrics

Comprehensive evaluation metrics for federated learning systems in battery
management applications. Provides detailed analysis of model performance,
communication efficiency, privacy preservation, and system scalability.

Features:
- Model convergence and accuracy metrics
- Communication efficiency analysis
- Privacy and security evaluation
- Fairness and bias assessment
- Energy consumption tracking
- Scalability and robustness metrics
- Business impact measurement

Author: BatteryMind Development Team
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
import json
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.preprocessing import StandardScaler
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MetricType(Enum):
    """Types of evaluation metrics."""
    ACCURACY = "accuracy"
    CONVERGENCE = "convergence"
    COMMUNICATION = "communication"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"
    ENERGY = "energy"
    SCALABILITY = "scalability"
    BUSINESS = "business"

@dataclass
class ModelPerformanceMetrics:
    """
    Model performance metrics for federated learning.
    
    Attributes:
        accuracy (float): Model accuracy
        precision (float): Model precision
        recall (float): Model recall
        f1_score (float): F1 score
        mse (float): Mean squared error (for regression)
        mae (float): Mean absolute error (for regression)
        r2_score (float): R-squared score (for regression)
        auc_roc (float): Area under ROC curve
        convergence_rate (float): Rate of convergence
        stability_score (float): Model stability score
        generalization_gap (float): Gap between train and test performance
    """
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mse: float = 0.0
    mae: float = 0.0
    r2_score: float = 0.0
    auc_roc: float = 0.0
    convergence_rate: float = 0.0
    stability_score: float = 0.0
    generalization_gap: float = 0.0

@dataclass
class CommunicationMetrics:
    """
    Communication efficiency metrics.
    
    Attributes:
        total_bytes_transmitted (int): Total bytes transmitted
        total_rounds (int): Total communication rounds
        average_round_time (float): Average time per round
        bandwidth_utilization (float): Average bandwidth utilization
        compression_ratio (float): Data compression ratio achieved
        communication_cost (float): Total communication cost
        network_efficiency (float): Network efficiency score
        latency_statistics (Dict): Latency distribution statistics
    """
    total_bytes_transmitted: int = 0
    total_rounds: int = 0
    average_round_time: float = 0.0
    bandwidth_utilization: float = 0.0
    compression_ratio: float = 1.0
    communication_cost: float = 0.0
    network_efficiency: float = 0.0
    latency_statistics: Dict[str, float] = field(default_factory=dict)

@dataclass
class PrivacyMetrics:
    """
    Privacy preservation metrics.
    
    Attributes:
        differential_privacy_epsilon (float): DP epsilon value
        membership_inference_accuracy (float): Membership inference attack accuracy
        model_inversion_risk (float): Model inversion attack risk
        data_reconstruction_error (float): Data reconstruction error
        privacy_budget_consumed (float): Total privacy budget consumed
        anonymity_score (float): Data anonymity score
        k_anonymity (int): K-anonymity level achieved
    """
    differential_privacy_epsilon: float = 0.0
    membership_inference_accuracy: float = 0.0
    model_inversion_risk: float = 0.0
    data_reconstruction_error: float = 0.0
    privacy_budget_consumed: float = 0.0
    anonymity_score: float = 0.0
    k_anonymity: int = 0

@dataclass
class FairnessMetrics:
    """
    Fairness and bias evaluation metrics.
    
    Attributes:
        demographic_parity (float): Demographic parity score
        equalized_odds (float): Equalized odds score
        individual_fairness (float): Individual fairness score
        group_fairness (float): Group fairness score
        bias_amplification (float): Bias amplification factor
        representation_fairness (float): Representation fairness score
        performance_disparity (Dict): Performance disparity across groups
    """
    demographic_parity: float = 0.0
    equalized_odds: float = 0.0
    individual_fairness: float = 0.0
    group_fairness: float = 0.0
    bias_amplification: float = 0.0
    representation_fairness: float = 0.0
    performance_disparity: Dict[str, float] = field(default_factory=dict)

@dataclass
class EnergyMetrics:
    """
    Energy consumption metrics.
    
    Attributes:
        total_energy_consumed (float): Total energy consumed (kWh)
        energy_per_round (float): Average energy per round
        energy_efficiency (float): Energy efficiency score
        battery_impact (float): Impact on battery life
        carbon_footprint (float): Estimated carbon footprint
        energy_distribution (Dict): Energy consumption distribution
    """
    total_energy_consumed: float = 0.0
    energy_per_round: float = 0.0
    energy_efficiency: float = 0.0
    battery_impact: float = 0.0
    carbon_footprint: float = 0.0
    energy_distribution: Dict[str, float] = field(default_factory=dict)

@dataclass
class ScalabilityMetrics:
    """
    Scalability and robustness metrics.
    
    Attributes:
        max_clients_supported (int): Maximum clients supported
        throughput_scaling (float): Throughput scaling factor
        latency_scaling (float): Latency scaling factor
        fault_tolerance (float): Fault tolerance score
        client_dropout_resilience (float): Resilience to client dropouts
        heterogeneity_handling (float): Ability to handle heterogeneity
        load_balancing_efficiency (float): Load balancing efficiency
    """
    max_clients_supported: int = 0
    throughput_scaling: float = 0.0
    latency_scaling: float = 0.0
    fault_tolerance: float = 0.0
    client_dropout_resilience: float = 0.0
    heterogeneity_handling: float = 0.0
    load_balancing_efficiency: float = 0.0

class FederatedLearningEvaluator:
    """
    Comprehensive evaluator for federated learning systems.
    """
    
    def __init__(self):
        self.evaluation_history: List[Dict] = []
        self.baseline_metrics: Optional[Dict] = None
        
    def evaluate_model_performance(self, predictions: np.ndarray, 
                                 ground_truth: np.ndarray,
                                 task_type: str = "classification") -> ModelPerformanceMetrics:
        """
        Evaluate model performance metrics.
        
        Args:
            predictions (np.ndarray): Model predictions
            ground_truth (np.ndarray): Ground truth labels/values
            task_type (str): Type of task ("classification" or "regression")
            
        Returns:
            ModelPerformanceMetrics: Computed performance metrics
        """
        metrics = ModelPerformanceMetrics()
        
        if task_type == "classification":
            # Classification metrics
            if len(np.unique(ground_truth)) == 2:  # Binary classification
                metrics.accuracy = accuracy_score(ground_truth, predictions)
                metrics.precision = precision_score(ground_truth, predictions, average='binary')
                metrics.recall = recall_score(ground_truth, predictions, average='binary')
                metrics.f1_score = f1_score(ground_truth, predictions, average='binary')
            else:  # Multi-class classification
                metrics.accuracy = accuracy_score(ground_truth, predictions)
                metrics.precision = precision_score(ground_truth, predictions, average='weighted')
                metrics.recall = recall_score(ground_truth, predictions, average='weighted')
                metrics.f1_score = f1_score(ground_truth, predictions, average='weighted')
                
        elif task_type == "regression":
            # Regression metrics
            metrics.mse = mean_squared_error(ground_truth, predictions)
            metrics.mae = mean_absolute_error(ground_truth, predictions)
            metrics.r2_score = r2_score(ground_truth, predictions)
            
            # Calculate accuracy for regression (within tolerance)
            tolerance = 0.1 * np.std(ground_truth)
            metrics.accuracy = np.mean(np.abs(predictions - ground_truth) <= tolerance)
        
        return metrics
    
    def evaluate_convergence(self, training_history: List[Dict]) -> Dict[str, float]:
        """
        Evaluate model convergence characteristics.
        
        Args:
            training_history (List[Dict]): Training history with loss values
            
        Returns:
            Dict[str, float]: Convergence metrics
        """
        if not training_history:
            return {'convergence_rate': 0.0, 'stability_score': 0.0}
        
        losses = [round_data.get('loss', float('inf')) for round_data in training_history]
        
        # Calculate convergence rate
        if len(losses) > 1:
            # Fit exponential decay to loss curve
            rounds = np.arange(len(losses))
            try:
                # Log transform for exponential fit
                log_losses = np.log(np.maximum(losses, 1e-10))
                slope, intercept = np.polyfit(rounds, log_losses, 1)
                convergence_rate = -slope  # Negative slope indicates convergence
            except:
                convergence_rate = 0.0
        else:
            convergence_rate = 0.0
        
        # Calculate stability score (inverse of loss variance in later rounds)
        if len(losses) > 10:
            later_losses = losses[-10:]
            stability_score = 1.0 / (1.0 + np.var(later_losses))
        else:
            stability_score = 0.0
        
        return {
            'convergence_rate': convergence_rate,
            'stability_score': stability_score,
            'final_loss': losses[-1] if losses else float('inf'),
            'loss_reduction': (losses[0] - losses[-1]) / losses[0] if len(losses) > 1 and losses[0] > 0 else 0.0
        }
    
    def evaluate_communication_efficiency(self, communication_log: List[Dict]) -> CommunicationMetrics:
        """
        Evaluate communication efficiency metrics.
        
        Args:
            communication_log (List[Dict]): Communication log data
            
        Returns:
            CommunicationMetrics: Communication efficiency metrics
        """
        metrics = CommunicationMetrics()
        
        if not communication_log:
            return metrics
        
        # Calculate total bytes transmitted
        metrics.total_bytes_transmitted = sum(
            record.get('payload_size_mb', 0) * 1024 * 1024 
            for record in communication_log
        )
        
        # Calculate total rounds
        rounds = set(record.get('round_id', 0) for record in communication_log)
        metrics.total_rounds = len(rounds)
        
        # Calculate average round time
        if metrics.total_rounds > 0:
            round_times = []
            for round_id in rounds:
                round_records = [r for r in communication_log if r.get('round_id') == round_id]
                if round_records:
                    round_start = min(r.get('timestamp', 0) for r in round_records)
                    round_end = max(r.get('timestamp', 0) for r in round_records)
                    round_times.append(round_end - round_start)
            
            metrics.average_round_time = np.mean(round_times) if round_times else 0.0
        
        # Calculate bandwidth utilization
        successful_transmissions = [r for r in communication_log if r.get('success', False)]
        if successful_transmissions:
            total_time = max(r.get('timestamp', 0) for r in communication_log) - \
                        min(r.get('timestamp', 0) for r in communication_log)
            if total_time > 0:
                metrics.bandwidth_utilization = metrics.total_bytes_transmitted / (total_time * 1024 * 1024)  # MB/s
        
        # Calculate network efficiency (successful transmissions ratio)
        if communication_log:
            metrics.network_efficiency = len(successful_transmissions) / len(communication_log)
        
        # Calculate latency statistics
        latencies = [r.get('latency', 0) for r in successful_transmissions]
        if latencies:
            metrics.latency_statistics = {
                'mean': np.mean(latencies),
                'median': np.median(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'percentile_95': np.percentile(latencies, 95),
                'percentile_99': np.percentile(latencies, 99)
            }
        
        return metrics
    
    def evaluate_privacy_preservation(self, privacy_config: Dict, 
                                    attack_results: Optional[Dict] = None) -> PrivacyMetrics:
        """
        Evaluate privacy preservation metrics.
        
        Args:
            privacy_config (Dict): Privacy configuration parameters
            attack_results (Dict, optional): Results from privacy attacks
            
        Returns:
            PrivacyMetrics: Privacy preservation metrics
        """
        metrics = PrivacyMetrics()
        
        # Differential privacy metrics
        if 'epsilon' in privacy_config:
            metrics.differential_privacy_epsilon = privacy_config['epsilon']
        
        if 'privacy_budget_used' in privacy_config:
            metrics.privacy_budget_consumed = privacy_config['privacy_budget_used']
        
        # Attack resistance metrics
        if attack_results:
            metrics.membership_inference_accuracy = attack_results.get('membership_inference_accuracy', 0.0)
            metrics.model_inversion_risk = attack_results.get('model_inversion_risk', 0.0)
            metrics.data_reconstruction_error = attack_results.get('data_reconstruction_error', 0.0)
        
        # Calculate anonymity score based on privacy parameters
        if metrics.differential_privacy_epsilon > 0:
            # Lower epsilon means better privacy
            metrics.anonymity_score = 1.0 / (1.0 + metrics.differential_privacy_epsilon)
        
        return metrics
    
    def evaluate_fairness(self, predictions: np.ndarray, ground_truth: np.ndarray,
                         sensitive_attributes: np.ndarray) -> FairnessMetrics:
        """
        Evaluate fairness and bias metrics.
        
        Args:
            predictions (np.ndarray): Model predictions
            ground_truth (np.ndarray): Ground truth labels
            sensitive_attributes (np.ndarray): Sensitive attribute values
            
        Returns:
            FairnessMetrics: Fairness evaluation metrics
        """
        metrics = FairnessMetrics()
        
        # Get unique groups
        unique_groups = np.unique(sensitive_attributes)
        
        if len(unique_groups) < 2:
            return metrics
        
        # Calculate demographic parity
        group_positive_rates = []
        group_accuracies = []
        
        for group in unique_groups:
            group_mask = sensitive_attributes == group
            group_predictions = predictions[group_mask]
            group_truth = ground_truth[group_mask]
            
            if len(group_predictions) > 0:
                # Positive rate (for binary classification)
                if len(np.unique(ground_truth)) == 2:
                    positive_rate = np.mean(group_predictions == 1)
                    group_positive_rates.append(positive_rate)
                
                # Accuracy for this group
                accuracy = accuracy_score(group_truth, group_predictions)
                group_accuracies.append(accuracy)
        
        # Demographic parity (difference in positive rates)
        if len(group_positive_rates) >= 2:
            metrics.demographic_parity = 1.0 - (max(group_positive_rates) - min(group_positive_rates))
        
        # Performance disparity
        if len(group_accuracies) >= 2:
            metrics.performance_disparity = {
                f'group_{i}': acc for i, acc in enumerate(group_accuracies)
            }
            
            # Group fairness (inverse of accuracy variance)
            metrics.group_fairness = 1.0 / (1.0 + np.var(group_accuracies))
        
        return metrics
    
    def evaluate_energy_consumption(self, energy_log: List[Dict]) -> EnergyMetrics:
        """
        Evaluate energy consumption metrics.
        
        Args:
            energy_log (List[Dict]): Energy consumption log
            
        Returns:
            EnergyMetrics: Energy consumption metrics
        """
        metrics = EnergyMetrics()
        
        if not energy_log:
            return metrics
        
        # Calculate total energy consumed
        metrics.total_energy_consumed = sum(
            record.get('energy_kwh', 0) for record in energy_log
        )
        
        # Calculate energy per round
        rounds = set(record.get('round_id', 0) for record in energy_log)
        if len(rounds) > 0:
            metrics.energy_per_round = metrics.total_energy_consumed / len(rounds)
        
        # Calculate energy distribution by device type
        device_energy = {}
        for record in energy_log:
            device_type = record.get('device_type', 'unknown')
            energy = record.get('energy_kwh', 0)
            device_energy[device_type] = device_energy.get(device_type, 0) + energy
        
        metrics.energy_distribution = device_energy
        
        # Calculate energy efficiency (performance per unit energy)
        if metrics.total_energy_consumed > 0 and hasattr(self, 'latest_accuracy'):
            metrics.energy_efficiency = getattr(self, 'latest_accuracy', 0) / metrics.total_energy_consumed
        
        # Estimate carbon footprint (simplified calculation)
        # Assuming average grid carbon intensity of 0.5 kg CO2/kWh
        metrics.carbon_footprint = metrics.total_energy_consumed * 0.5
        
        return metrics
    
    def evaluate_scalability(self, scalability_tests: List[Dict]) -> ScalabilityMetrics:
        """
        Evaluate scalability metrics.
        
        Args:
            scalability_tests (List[Dict]): Scalability test results
            
        Returns:
            ScalabilityMetrics: Scalability metrics
        """
        metrics = ScalabilityMetrics()
        
        if not scalability_tests:
            return metrics
        
        # Find maximum clients supported
        client_counts = [test.get('num_clients', 0) for test in scalability_tests]
        successful_tests = [test for test in scalability_tests if test.get('success', False)]
        
        if successful_tests:
            metrics.max_clients_supported = max(test.get('num_clients', 0) for test in successful_tests)
        
        # Calculate scaling factors
        if len(scalability_tests) >= 2:
            # Sort by number of clients
            sorted_tests = sorted(scalability_tests, key=lambda x: x.get('num_clients', 0))
            
            # Throughput scaling
            throughputs = [test.get('throughput', 0) for test in sorted_tests]
            client_counts = [test.get('num_clients', 1) for test in sorted_tests]
            
            if len(throughputs) >= 2 and client_counts[-1] > client_counts[0]:
                throughput_ratio = throughputs[-1] / max(throughputs[0], 1e-10)
                client_ratio = client_counts[-1] / client_counts[0]
                metrics.throughput_scaling = throughput_ratio / client_ratio
            
            # Latency scaling
            latencies = [test.get('average_latency', 0) for test in sorted_tests]
            if len(latencies) >= 2:
                latency_ratio = latencies[-1] / max(latencies[0], 1e-10)
                metrics.latency_scaling = latency_ratio / client_ratio
        
        # Calculate fault tolerance
        fault_tolerance_tests = [test for test in scalability_tests if 'fault_tolerance' in test]
        if fault_tolerance_tests:
            metrics.fault_tolerance = np.mean([test['fault_tolerance'] for test in fault_tolerance_tests])
        
        return metrics
    
    def generate_comprehensive_report(self, evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            evaluation_data (Dict[str, Any]): All evaluation data
            
        Returns:
            Dict[str, Any]: Comprehensive evaluation report
        """
        report = {
            'evaluation_timestamp': time.time(),
            'summary': {},
            'detailed_metrics': {},
            'recommendations': [],
            'visualizations': {}
        }
        
        # Model performance evaluation
        if 'model_performance' in evaluation_data:
            perf_data = evaluation_data['model_performance']
            model_metrics = self.evaluate_model_performance(
                perf_data['predictions'], 
                perf_data['ground_truth'],
                perf_data.get('task_type', 'classification')
            )
            report['detailed_metrics']['model_performance'] = model_metrics.__dict__
            report['summary']['model_accuracy'] = model_metrics.accuracy
        
        # Communication efficiency evaluation
        if 'communication_log' in evaluation_data:
            comm_metrics = self.evaluate_communication_efficiency(evaluation_data['communication_log'])
            report['detailed_metrics']['communication'] = comm_metrics.__dict__
            report['summary']['network_efficiency'] = comm_metrics.network_efficiency
        
        # Privacy evaluation
        if 'privacy_config' in evaluation_data:
            privacy_metrics = self.evaluate_privacy_preservation(
                evaluation_data['privacy_config'],
                evaluation_data.get('attack_results')
            )
            report['detailed_metrics']['privacy'] = privacy_metrics.__dict__
            report['summary']['privacy_score'] = privacy_metrics.anonymity_score
        
        # Fairness evaluation
        if 'fairness_data' in evaluation_data:
            fairness_data = evaluation_data['fairness_data']
            fairness_metrics = self.evaluate_fairness(
                fairness_data['predictions'],
                fairness_data['ground_truth'],
                fairness_data['sensitive_attributes']
            )
            report['detailed_metrics']['fairness'] = fairness_metrics.__dict__
            report['summary']['fairness_score'] = fairness_metrics.group_fairness
        
        # Energy evaluation
        if 'energy_log' in evaluation_data:
            energy_metrics = self.evaluate_energy_consumption(evaluation_data['energy_log'])
            report['detailed_metrics']['energy'] = energy_metrics.__dict__
            report['summary']['energy_efficiency'] = energy_metrics.energy_efficiency
        
        # Scalability evaluation
        if 'scalability_tests' in evaluation_data:
            scalability_metrics = self.evaluate_scalability(evaluation_data['scalability_tests'])
            report['detailed_metrics']['scalability'] = scalability_metrics.__dict__
            report['summary']['max_clients'] = scalability_metrics.max_clients_supported
        
        # Generate recommendations
        report['recommendations'] = self._generate_recommendations(report['detailed_metrics'])
        
        # Calculate overall score
        report['summary']['overall_score'] = self._calculate_overall_score(report['summary'])
        
        return report
    
    def _generate_recommendations(self, detailed_metrics: Dict) -> List[str]:
        """Generate recommendations based on evaluation results."""
        recommendations = []
        
        # Model performance recommendations
        if 'model_performance' in detailed_metrics:
            accuracy = detailed_metrics['model_performance'].get('accuracy', 0)
            if accuracy < 0.8:
                recommendations.append("Consider increasing model complexity or training time to improve accuracy")
            if accuracy < 0.6:
                recommendations.append("Model performance is critically low - review data quality and model architecture")
        
        # Communication recommendations
        if 'communication' in detailed_metrics:
            efficiency = detailed_metrics['communication'].get('network_efficiency', 0)
            if efficiency < 0.9:
                recommendations.append("Network efficiency is low - consider implementing better error handling and retry mechanisms")
            
            avg_latency = detailed_metrics['communication'].get('latency_statistics', {}).get('mean', 0)
            if avg_latency > 1.0:  # 1 second
                recommendations.append("High network latency detected - consider edge computing or model compression")
        
        # Privacy recommendations
        if 'privacy' in detailed_metrics:
            epsilon = detailed_metrics['privacy'].get('differential_privacy_epsilon', 0)
            if epsilon > 1.0:
                recommendations.append("Differential privacy epsilon is high - consider stronger privacy protection")
        
        # Energy recommendations
        if 'energy' in detailed_metrics:
            efficiency = detailed_metrics['energy'].get('energy_efficiency', 0)
            if efficiency < 0.1:
                recommendations.append("Energy efficiency is low - consider model optimization and efficient scheduling")
        
        # Scalability recommendations
        if 'scalability' in detailed_metrics:
            max_clients = detailed_metrics['scalability'].get('max_clients_supported', 0)
            if max_clients < 100:
                recommendations.append("Limited scalability - consider hierarchical federated learning or better load balancing")
        
        return recommendations
    
    def _calculate_overall_score(self, summary: Dict) -> float:
        """Calculate overall system score."""
        scores = []
        weights = {
            'model_accuracy': 0.3,
            'network_efficiency': 0.2,
            'privacy_score': 0.2,
            'fairness_score': 0.1,
            'energy_efficiency': 0.1,
            'scalability_score': 0.1
        }
        
        for metric, weight in weights.items():
            if metric in summary:
                scores.append(summary[metric] * weight)
        
        return sum(scores) if scores else 0.0
    
    def export_evaluation_report(self, report: Dict, file_path: str) -> None:
        """Export evaluation report to file."""
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Evaluation report exported to {file_path}")

# Factory functions
def create_federated_evaluator() -> FederatedLearningEvaluator:
    """Create a federated learning evaluator."""
    return FederatedLearningEvaluator()

def evaluate_federated_system(evaluation_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to evaluate a complete federated learning system.
    
    Args:
        evaluation_data (Dict[str, Any]): Complete evaluation data
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation report
    """
    evaluator = FederatedLearningEvaluator()
    return evaluator.generate_comprehensive_report(evaluation_data)
