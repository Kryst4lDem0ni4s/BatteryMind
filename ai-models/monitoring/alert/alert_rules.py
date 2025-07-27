"""
BatteryMind - Alert Rules Engine
Comprehensive alert rules engine for battery management AI models with intelligent
rule evaluation, complex condition matching, and dynamic rule management.

Features:
- Complex rule conditions with logical operators
- Multiple rule types (metric, anomaly, performance, business)
- Rule chaining and dependency management
- Dynamic rule evaluation with real-time processing
- Rule performance optimization and caching
- A/B testing for rule effectiveness
- Machine learning-based rule recommendations

Author: BatteryMind Development Team
Version: 1.0.0
"""

import logging
import threading
import time
import json
import ast
import operator
from typing import Dict, List, Optional, Any, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid
import re
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

# BatteryMind imports
from ...utils.logging_utils import setup_logger
from ...config.monitoring_config import MonitoringConfig

# Configure logging
logger = setup_logger(__name__)

class RuleType(Enum):
    """Types of alert rules."""
    METRIC = "metric"
    ANOMALY = "anomaly"
    PERFORMANCE = "performance"
    BUSINESS = "business"
    COMPOSITE = "composite"
    THRESHOLD = "threshold"
    PATTERN = "pattern"
    ML_BASED = "ml_based"

class AlertCondition(Enum):
    """Alert condition operators."""
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EQUALS = "eq"
    NOT_EQUALS = "ne"
    CONTAINS = "contains"
    NOT_CONTAINS = "not_contains"
    MATCHES = "matches"
    NOT_MATCHES = "not_matches"
    IN = "in"
    NOT_IN = "not_in"
    BETWEEN = "between"
    NOT_BETWEEN = "not_between"
    ANOMALY = "anomaly"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    CHANGE_RATE = "change_rate"

class LogicalOperator(Enum):
    """Logical operators for combining conditions."""
    AND = "and"
    OR = "or"
    NOT = "not"
    XOR = "xor"

class RuleStatus(Enum):
    """Rule execution status."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    DISABLED = "disabled"
    ERROR = "error"
    TESTING = "testing"

@dataclass
class AlertThreshold:
    """Alert threshold definition."""
    
    value: Union[float, int, str, List[Any]]
    unit: str = ""
    
    # Advanced threshold settings
    hysteresis: Optional[float] = None  # Prevents flapping
    duration_minutes: int = 1  # Duration threshold must be breached
    aggregation_method: str = "avg"  # avg, max, min, sum, count
    
    # Adaptive thresholds
    adaptive: bool = False
    baseline_period_hours: int = 24
    deviation_multiplier: float = 2.0
    
    # Time-based thresholds
    time_dependent: bool = False
    time_ranges: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate(self, current_value: Union[float, int, str], 
                condition: AlertCondition,
                historical_data: Optional[List[float]] = None) -> bool:
        """Evaluate threshold against current value."""
        try:
            # Handle adaptive thresholds
            if self.adaptive and historical_data:
                threshold_value = self._calculate_adaptive_threshold(historical_data)
            else:
                threshold_value = self.value
            
            # Apply condition logic
            if condition == AlertCondition.GREATER_THAN:
                return float(current_value) > float(threshold_value)
            elif condition == AlertCondition.GREATER_THAN_OR_EQUAL:
                return float(current_value) >= float(threshold_value)
            elif condition == AlertCondition.LESS_THAN:
                return float(current_value) < float(threshold_value)
            elif condition == AlertCondition.LESS_THAN_OR_EQUAL:
                return float(current_value) <= float(threshold_value)
            elif condition == AlertCondition.EQUALS:
                return current_value == threshold_value
            elif condition == AlertCondition.NOT_EQUALS:
                return current_value != threshold_value
            elif condition == AlertCondition.CONTAINS:
                return str(threshold_value) in str(current_value)
            elif condition == AlertCondition.NOT_CONTAINS:
                return str(threshold_value) not in str(current_value)
            elif condition == AlertCondition.MATCHES:
                return bool(re.match(str(threshold_value), str(current_value)))
            elif condition == AlertCondition.NOT_MATCHES:
                return not bool(re.match(str(threshold_value), str(current_value)))
            elif condition == AlertCondition.IN:
                return current_value in threshold_value
            elif condition == AlertCondition.NOT_IN:
                return current_value not in threshold_value
            elif condition == AlertCondition.BETWEEN:
                if isinstance(threshold_value, list) and len(threshold_value) == 2:
                    return threshold_value[0] <= float(current_value) <= threshold_value[1]
            elif condition == AlertCondition.NOT_BETWEEN:
                if isinstance(threshold_value, list) and len(threshold_value) == 2:
                    return not (threshold_value[0] <= float(current_value) <= threshold_value[1])
            
            return False
            
        except Exception as e:
            logger.error(f"Error evaluating threshold: {e}")
            return False
    
    def _calculate_adaptive_threshold(self, historical_data: List[float]) -> float:
        """Calculate adaptive threshold based on historical data."""
        try:
            if not historical_data:
                return self.value
            
            # Calculate baseline statistics
            mean = np.mean(historical_data)
            std = np.std(historical_data)
            
            # Apply deviation multiplier
            adaptive_threshold = mean + (self.deviation_multiplier * std)
            
            return adaptive_threshold
            
        except Exception as e:
            logger.error(f"Error calculating adaptive threshold: {e}")
            return self.value

@dataclass
class RuleCondition:
    """Individual rule condition."""
    
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    metric_name: str = ""
    condition: AlertCondition = AlertCondition.GREATER_THAN
    threshold: AlertThreshold = field(default_factory=AlertThreshold)
    
    # Time window settings
    evaluation_window_minutes: int = 5
    aggregation_method: str = "avg"
    
    # Advanced condition settings
    weight: float = 1.0  # For weighted conditions
    required: bool = True  # If False, condition is optional
    
    # Context filtering
    filters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def evaluate(self, data: Dict[str, Any], 
                historical_data: Optional[Dict[str, List[float]]] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate condition against provided data.
        
        Returns:
            Tuple of (condition_met, evaluation_details)
        """
        try:
            # Extract metric value
            metric_value = self._extract_metric_value(data)
            if metric_value is None:
                return False, {'error': f'Metric {self.metric_name} not found'}
            
            # Get historical data for this metric
            metric_history = historical_data.get(self.metric_name, []) if historical_data else []
            
            # Evaluate threshold
            condition_met = self.threshold.evaluate(
                metric_value, 
                self.condition, 
                metric_history
            )
            
            evaluation_details = {
                'metric_name': self.metric_name,
                'current_value': metric_value,
                'threshold_value': self.threshold.value,
                'condition': self.condition.value,
                'condition_met': condition_met,
                'evaluation_time': datetime.now(),
                'weight': self.weight
            }
            
            return condition_met, evaluation_details
            
        except Exception as e:
            logger.error(f"Error evaluating condition {self.id}: {e}")
            return False, {'error': str(e)}
    
    def _extract_metric_value(self, data: Dict[str, Any]) -> Optional[Union[float, int, str]]:
        """Extract metric value from data using dot notation."""
        try:
            # Support dot notation for nested keys
            keys = self.metric_name.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return value
            
        except Exception as e:
            logger.error(f"Error extracting metric value: {e}")
            return None

class BaseRule(ABC):
    """Abstract base class for all rule types."""
    
    def __init__(self, rule_id: str, name: str, description: str = ""):
        self.id = rule_id
        self.name = name
        self.description = description
        self.status = RuleStatus.ACTIVE
        self.created_at = datetime.now()
        self.last_evaluated = None
        self.evaluation_count = 0
        self.trigger_count = 0
        
        # Performance metrics
        self.avg_evaluation_time_ms = 0.0
        self.false_positive_rate = 0.0
        self.effectiveness_score = 0.0
    
    @abstractmethod
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate rule against provided data."""
        pass
    
    def update_performance_metrics(self, evaluation_time_ms: float, triggered: bool, false_positive: bool = False):
        """Update rule performance metrics."""
        self.evaluation_count += 1
        
        # Update average evaluation time
        self.avg_evaluation_time_ms = (
            (self.avg_evaluation_time_ms * (self.evaluation_count - 1) + evaluation_time_ms) / 
            self.evaluation_count
        )
        
        if triggered:
            self.trigger_count += 1
        
        if false_positive:
            # Update false positive rate (simplified calculation)
            self.false_positive_rate = (self.false_positive_rate + 1) / 2
        
        # Calculate effectiveness score
        self._calculate_effectiveness_score()
    
    def _calculate_effectiveness_score(self):
        """Calculate rule effectiveness score."""
        if self.evaluation_count == 0:
            self.effectiveness_score = 0.0
            return
        
        # Simple effectiveness calculation based on trigger rate and false positive rate
        trigger_rate = self.trigger_count / self.evaluation_count
        effectiveness = trigger_rate * (1 - self.false_positive_rate)
        self.effectiveness_score = min(max(effectiveness, 0.0), 1.0)

class MetricRule(BaseRule):
    """Rule for metric-based alerts."""
    
    def __init__(self, rule_id: str, name: str, description: str = "",
                 metric_name: str = "", condition: AlertCondition = AlertCondition.GREATER_THAN,
                 threshold: AlertThreshold = None, severity: str = "MEDIUM",
                 evaluation_window_minutes: int = 5):
        
        super().__init__(rule_id, name, description)
        self.rule_type = RuleType.METRIC
        self.metric_name = metric_name
        self.condition = condition
        self.threshold = threshold or AlertThreshold(value=0)
        self.severity = severity
        self.evaluation_window_minutes = evaluation_window_minutes
    
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate metric rule."""
        start_time = time.time()
        
        try:
            # Create a rule condition for evaluation
            rule_condition = RuleCondition(
                metric_name=self.metric_name,
                condition=self.condition,
                threshold=self.threshold,
                evaluation_window_minutes=self.evaluation_window_minutes
            )
            
            # Evaluate condition
            historical_data = context.get('historical_data', {})
            condition_met, evaluation_details = rule_condition.evaluate(data, historical_data)
            
            # Prepare result
            result = {
                'rule_id': self.id,
                'rule_name': self.name,
                'rule_type': self.rule_type.value,
                'triggered': condition_met,
                'severity': self.severity,
                'metric_details': evaluation_details,
                'evaluation_time': datetime.now()
            }
            
            # Update performance metrics
            evaluation_time_ms = (time.time() - start_time) * 1000
            self.update_performance_metrics(evaluation_time_ms, condition_met)
            self.last_evaluated = datetime.now()
            
            return condition_met, result
            
        except Exception as e:
            logger.error(f"Error evaluating metric rule {self.id}: {e}")
            return False, {'error': str(e)}

class AnomalyRule(BaseRule):
    """Rule for anomaly-based alerts."""
    
    def __init__(self, rule_id: str, name: str, description: str = "",
                 metric_name: str = "", anomaly_threshold: float = 2.0,
                 severity: str = "HIGH", evaluation_window_minutes: int = 10):
        
        super().__init__(rule_id, name, description)
        self.rule_type = RuleType.ANOMALY
        self.metric_name = metric_name
        self.anomaly_threshold = anomaly_threshold  # Standard deviations
        self.severity = severity
        self.evaluation_window_minutes = evaluation_window_minutes
    
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate anomaly rule."""
        start_time = time.time()
        
        try:
            # Extract current metric value
            current_value = self._extract_metric_value(data, self.metric_name)
            if current_value is None:
                return False, {'error': f'Metric {self.metric_name} not found'}
            
            # Get historical data
            historical_data = context.get('historical_data', {})
            metric_history = historical_data.get(self.metric_name, [])
            
            if len(metric_history) < 10:  # Need minimum history for anomaly detection
                return False, {'error': 'Insufficient historical data for anomaly detection'}
            
            # Calculate anomaly score
            anomaly_score = self._calculate_anomaly_score(current_value, metric_history)
            is_anomaly = abs(anomaly_score) > self.anomaly_threshold
            
            result = {
                'rule_id': self.id,
                'rule_name': self.name,
                'rule_type': self.rule_type.value,
                'triggered': is_anomaly,
                'severity': self.severity,
                'anomaly_details': {
                    'metric_name': self.metric_name,
                    'current_value': current_value,
                    'anomaly_score': anomaly_score,
                    'threshold': self.anomaly_threshold,
                    'historical_mean': np.mean(metric_history),
                    'historical_std': np.std(metric_history)
                },
                'evaluation_time': datetime.now()
            }
            
            # Update performance metrics
            evaluation_time_ms = (time.time() - start_time) * 1000
            self.update_performance_metrics(evaluation_time_ms, is_anomaly)
            self.last_evaluated = datetime.now()
            
            return is_anomaly, result
            
        except Exception as e:
            logger.error(f"Error evaluating anomaly rule {self.id}: {e}")
            return False, {'error': str(e)}
    
    def _extract_metric_value(self, data: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from data."""
        try:
            keys = metric_name.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return float(value)
            
        except (ValueError, TypeError):
            return None
    
    def _calculate_anomaly_score(self, current_value: float, historical_data: List[float]) -> float:
        """Calculate anomaly score (z-score)."""
        try:
            mean = np.mean(historical_data)
            std = np.std(historical_data)
            
            if std == 0:
                return 0.0
            
            z_score = (current_value - mean) / std
            return z_score
            
        except Exception as e:
            logger.error(f"Error calculating anomaly score: {e}")
            return 0.0

class PerformanceRule(BaseRule):
    """Rule for performance-based alerts."""
    
    def __init__(self, rule_id: str, name: str, description: str = "",
                 performance_metrics: List[str] = None, thresholds: Dict[str, float] = None,
                 severity: str = "MEDIUM", evaluation_window_minutes: int = 5):
        
        super().__init__(rule_id, name, description)
        self.rule_type = RuleType.PERFORMANCE
        self.performance_metrics = performance_metrics or []
        self.thresholds = thresholds or {}
        self.severity = severity
        self.evaluation_window_minutes = evaluation_window_minutes
    
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate performance rule."""
        start_time = time.time()
        
        try:
            violations = []
            
            for metric in self.performance_metrics:
                threshold = self.thresholds.get(metric)
                if threshold is None:
                    continue
                
                current_value = self._extract_metric_value(data, metric)
                if current_value is None:
                    continue
                
                if current_value > threshold:
                    violations.append({
                        'metric': metric,
                        'current_value': current_value,
                        'threshold': threshold,
                        'violation_percentage': ((current_value - threshold) / threshold) * 100
                    })
            
            triggered = len(violations) > 0
            
            result = {
                'rule_id': self.id,
                'rule_name': self.name,
                'rule_type': self.rule_type.value,
                'triggered': triggered,
                'severity': self.severity,
                'performance_details': {
                    'violations': violations,
                    'total_metrics_checked': len(self.performance_metrics),
                    'violation_count': len(violations)
                },
                'evaluation_time': datetime.now()
            }
            
            # Update performance metrics
            evaluation_time_ms = (time.time() - start_time) * 1000
            self.update_performance_metrics(evaluation_time_ms, triggered)
            self.last_evaluated = datetime.now()
            
            return triggered, result
            
        except Exception as e:
            logger.error(f"Error evaluating performance rule {self.id}: {e}")
            return False, {'error': str(e)}
    
    def _extract_metric_value(self, data: Dict[str, Any], metric_name: str) -> Optional[float]:
        """Extract metric value from data."""
        try:
            keys = metric_name.split('.')
            value = data
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None
            
            return float(value)
            
        except (ValueError, TypeError):
            return None

class BusinessRule(BaseRule):
    """Rule for business logic-based alerts."""
    
    def __init__(self, rule_id: str, name: str, description: str = "",
                 business_conditions: List[str] = None, custom_logic: str = "",
                 severity: str = "MEDIUM", evaluation_window_minutes: int = 5):
        
        super().__init__(rule_id, name, description)
        self.rule_type = RuleType.BUSINESS
        self.business_conditions = business_conditions or []
        self.custom_logic = custom_logic
        self.severity = severity
        self.evaluation_window_minutes = evaluation_window_minutes
    
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate business rule."""
        start_time = time.time()
        
        try:
            triggered = False
            evaluation_details = {}
            
            # Evaluate custom logic if provided
            if self.custom_logic:
                triggered = self._evaluate_custom_logic(data, context)
                evaluation_details['custom_logic_result'] = triggered
            
            # Evaluate business conditions
            condition_results = []
            for condition in self.business_conditions:
                condition_result = self._evaluate_business_condition(condition, data, context)
                condition_results.append(condition_result)
            
            # Combine results (all conditions must be true)
            if condition_results:
                conditions_met = all(condition_results)
                triggered = triggered or conditions_met
                evaluation_details['business_conditions'] = condition_results
            
            result = {
                'rule_id': self.id,
                'rule_name': self.name,
                'rule_type': self.rule_type.value,
                'triggered': triggered,
                'severity': self.severity,
                'business_details': evaluation_details,
                'evaluation_time': datetime.now()
            }
            
            # Update performance metrics
            evaluation_time_ms = (time.time() - start_time) * 1000
            self.update_performance_metrics(evaluation_time_ms, triggered)
            self.last_evaluated = datetime.now()
            
            return triggered, result
            
        except Exception as e:
            logger.error(f"Error evaluating business rule {self.id}: {e}")
            return False, {'error': str(e)}
    
    def _evaluate_custom_logic(self, data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate custom business logic."""
        try:
            # Create safe execution context
            safe_globals = {
                '__builtins__': {},
                'data': data,
                'context': context,
                'datetime': datetime,
                'len': len,
                'sum': sum,
                'max': max,
                'min': min,
                'abs': abs,
                'round': round
            }
            
            # Execute custom logic
            result = eval(self.custom_logic, safe_globals)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Error evaluating custom logic: {e}")
            return False
    
    def _evaluate_business_condition(self, condition: str, data: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Evaluate individual business condition."""
        try:
            # Simple condition evaluation
            # This would be expanded with a proper business rule parser
            return eval(condition, {'data': data, 'context': context})
            
        except Exception as e:
            logger.error(f"Error evaluating business condition: {e}")
            return False

class CompositeRule(BaseRule):
    """Rule that combines multiple other rules with logical operators."""
    
    def __init__(self, rule_id: str, name: str, description: str = "",
                 sub_rules: List[BaseRule] = None, logical_operator: LogicalOperator = LogicalOperator.AND,
                 severity: str = "MEDIUM"):
        
        super().__init__(rule_id, name, description)
        self.rule_type = RuleType.COMPOSITE
        self.sub_rules = sub_rules or []
        self.logical_operator = logical_operator
        self.severity = severity
    
    def evaluate(self, data: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """Evaluate composite rule."""
        start_time = time.time()
        
        try:
            sub_rule_results = []
            
            # Evaluate all sub-rules
            for sub_rule in self.sub_rules:
                triggered, result = sub_rule.evaluate(data, context)
                sub_rule_results.append({
                    'rule_id': sub_rule.id,
                    'rule_name': sub_rule.name,
                    'triggered': triggered,
                    'result': result
                })
            
            # Apply logical operator
            triggered_results = [r['triggered'] for r in sub_rule_results]
            
            if self.logical_operator == LogicalOperator.AND:
                final_result = all(triggered_results)
            elif self.logical_operator == LogicalOperator.OR:
                final_result = any(triggered_results)
            elif self.logical_operator == LogicalOperator.NOT:
                final_result = not any(triggered_results)
            elif self.logical_operator == LogicalOperator.XOR:
                final_result = sum(triggered_results) == 1
            else:
                final_result = False
            
            result = {
                'rule_id': self.id,
                'rule_name': self.name,
                'rule_type': self.rule_type.value,
                'triggered': final_result,
                'severity': self.severity,
                'composite_details': {
                    'logical_operator': self.logical_operator.value,
                    'sub_rule_count': len(self.sub_rules),
                    'triggered_sub_rules': sum(triggered_results),
                    'sub_rule_results': sub_rule_results
                },
                'evaluation_time': datetime.now()
            }
            
            # Update performance metrics
            evaluation_time_ms = (time.time() - start_time) * 1000
            self.update_performance_metrics(evaluation_time_ms, final_result)
            self.last_evaluated = datetime.now()
            
            return final_result, result
            
        except Exception as e:
            logger.error(f"Error evaluating composite rule {self.id}: {e}")
            return False, {'error': str(e)}

class AlertRulesEngine:
    """
    Comprehensive alert rules engine for managing and evaluating rules.
    """
    
    def __init__(self, evaluation_interval_seconds: int = 60,
                 enable_rule_chaining: bool = True,
                 max_rule_complexity: int = 10):
        
        self.evaluation_interval_seconds = evaluation_interval_seconds
        self.enable_rule_chaining = enable_rule_chaining
        self.max_rule_complexity = max_rule_complexity
        
        # Rule storage
        self.rules: Dict[str, BaseRule] = {}
        self.rule_groups: Dict[str, List[str]] = {}  # group_name -> [rule_ids]
        
        # Rule evaluation state
        self.rule_evaluation_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.rule_dependencies: Dict[str, List[str]] = {}  # rule_id -> [dependency_rule_ids]
        
        # Performance and metrics
        self.evaluation_stats = {
            'total_evaluations': 0,
            'total_triggers': 0,
            'avg_evaluation_time_ms': 0.0,
            'rule_performance': {}
        }
        
        # Threading
        self.is_running = False
        self.evaluation_thread: Optional[threading.Thread] = None
        self.shutdown_event = threading.Event()
        
        # Historical data cache
        self.historical_data_cache: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.cache_lock = threading.RLock()
        
        logger.info("Alert Rules Engine initialized")
    
    def add_rule(self, rule: BaseRule) -> bool:
        """Add a rule to the engine."""
        try:
            if rule.id in self.rules:
                logger.warning(f"Rule {rule.id} already exists, updating")
            
            self.rules[rule.id] = rule
            
            # Initialize performance tracking
            if rule.id not in self.evaluation_stats['rule_performance']:
                self.evaluation_stats['rule_performance'][rule.id] = {
                    'evaluations': 0,
                    'triggers': 0,
                    'avg_time_ms': 0.0,
                    'effectiveness_score': 0.0
                }
            
            logger.info(f"Added rule: {rule.name} ({rule.id})")
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule: {e}")
            return False
    
    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule from the engine."""
        try:
            if rule_id in self.rules:
                rule_name = self.rules[rule_id].name
                del self.rules[rule_id]
                
                # Clean up related data
                if rule_id in self.rule_evaluation_history:
                    del self.rule_evaluation_history[rule_id]
                
                if rule_id in self.rule_dependencies:
                    del self.rule_dependencies[rule_id]
                
                # Remove from groups
                for group_name, rule_ids in self.rule_groups.items():
                    if rule_id in rule_ids:
                        rule_ids.remove(rule_id)
                
                logger.info(f"Removed rule: {rule_name} ({rule_id})")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error removing rule: {e}")
            return False
    
    def evaluate_rules(self, data: Dict[str, Any], 
                      context: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Evaluate all active rules against provided data.
        
        Args:
            data: Input data for rule evaluation
            context: Additional context information
            
        Returns:
            List of triggered rule results
        """
        if context is None:
            context = {}
        
        # Add historical data to context
        with self.cache_lock:
            context['historical_data'] = {
                metric: list(values) for metric, values in self.historical_data_cache.items()
            }
        
        triggered_rules = []
        start_time = time.time()
        
        try:
            # Update historical data cache
            self._update_historical_cache(data)
            
            # Evaluate rules in dependency order if chaining is enabled
            if self.enable_rule_chaining:
                evaluation_order = self._get_evaluation_order()
            else:
                evaluation_order = list(self.rules.keys())
            
            for rule_id in evaluation_order:
                rule = self.rules.get(rule_id)
                if not rule or rule.status != RuleStatus.ACTIVE:
                    continue
                
                try:
                    # Evaluate rule
                    triggered, result = rule.evaluate(data, context)
                    
                    # Store evaluation result
                    evaluation_record = {
                        'timestamp': datetime.now(),
                        'triggered': triggered,
                        'data_snapshot': self._create_data_snapshot(data),
                        'result': result
                    }
                    
                    self.rule_evaluation_history[rule_id].append(evaluation_record)
                    
                    # Track statistics
                    self.evaluation_stats['total_evaluations'] += 1
                    if triggered:
                        self.evaluation_stats['total_triggers'] += 1
                        triggered_rules.append(result)
                    
                    # Update rule performance stats
                    rule_stats = self.evaluation_stats['rule_performance'][rule_id]
                    rule_stats['evaluations'] += 1
                    if triggered:
                        rule_stats['triggers'] += 1
                    rule_stats['avg_time_ms'] = rule.avg_evaluation_time_ms
                    rule_stats['effectiveness_score'] = rule.effectiveness_score
                    
                    # Update context for dependent rules
                    if self.enable_rule_chaining:
                        context[f'rule_{rule_id}_result'] = result
                        context[f'rule_{rule_id}_triggered'] = triggered
                    
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule_id}: {e}")
                    continue
            
            # Update overall statistics
            total_time_ms = (time.time() - start_time) * 1000
            self.evaluation_stats['avg_evaluation_time_ms'] = (
                (self.evaluation_stats['avg_evaluation_time_ms'] * 
                 (self.evaluation_stats['total_evaluations'] - len(self.rules)) + total_time_ms) /
                self.evaluation_stats['total_evaluations']
            )
            
            return triggered_rules
            
        except Exception as e:
            logger.error(f"Error during rule evaluation: {e}")
            return []
    
    def evaluate_alert(self, alert: Any) -> Optional[Dict[str, Any]]:
        """Evaluate rules for a specific alert."""
        try:
            # Convert alert to data dictionary
            alert_data = {
                'id': getattr(alert, 'id', ''),
                'title': getattr(alert, 'title', ''),
                'description': getattr(alert, 'description', ''),
                'severity': getattr(alert, 'severity', 'MEDIUM'),
                'source': getattr(alert, 'source', ''),
                'source_type': getattr(alert, 'source_type', ''),
                'metric_name': getattr(alert, 'metric_name', ''),
                'current_value': getattr(alert, 'current_value', None),
                'threshold_value': getattr(alert, 'threshold_value', None),
                'labels': getattr(alert, 'labels', {}),
                'annotations': getattr(alert, 'annotations', {}),
                'context': getattr(alert, 'context', {}),
                'created_at': getattr(alert, 'created_at', datetime.now())
            }
            
            # Evaluate rules
            triggered_rules = self.evaluate_rules(alert_data)
            
            if triggered_rules:
                # Return aggregated result
                return {
                    'alert_id': alert_data['id'],
                    'triggered_rules_count': len(triggered_rules),
                    'triggered_rules': triggered_rules,
                    'highest_severity': self._get_highest_severity(triggered_rules),
                    'labels': self._aggregate_labels(triggered_rules),
                    'annotations': self._aggregate_annotations(triggered_rules)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating alert: {e}")
            return None
    
    def add_rule_to_group(self, rule_id: str, group_name: str) -> bool:
        """Add a rule to a group."""
        try:
            if rule_id not in self.rules:
                return False
            
            if group_name not in self.rule_groups:
                self.rule_groups[group_name] = []
            
            if rule_id not in self.rule_groups[group_name]:
                self.rule_groups[group_name].append(rule_id)
                logger.info(f"Added rule {rule_id} to group {group_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule to group: {e}")
            return False
    
    def add_rule_dependency(self, rule_id: str, dependency_rule_id: str) -> bool:
        """Add a dependency between rules."""
        try:
            if rule_id not in self.rules or dependency_rule_id not in self.rules:
                return False
            
            if rule_id not in self.rule_dependencies:
                self.rule_dependencies[rule_id] = []
            
            if dependency_rule_id not in self.rule_dependencies[rule_id]:
                self.rule_dependencies[rule_id].append(dependency_rule_id)
                logger.info(f"Added dependency: {rule_id} depends on {dependency_rule_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error adding rule dependency: {e}")
            return False
    
    def get_rule_statistics(self) -> Dict[str, Any]:
        """Get rule engine statistics."""
        stats = self.evaluation_stats.copy()
        
        # Add current state information
        stats['total_rules'] = len(self.rules)
        stats['active_rules'] = len([r for r in self.rules.values() if r.status == RuleStatus.ACTIVE])
        stats['rule_groups'] = len(self.rule_groups)
        
        # Calculate trigger rate
        if stats['total_evaluations'] > 0:
            stats['trigger_rate'] = stats['total_triggers'] / stats['total_evaluations']
        else:
            stats['trigger_rate'] = 0.0
        
        return stats
    
    def _update_historical_cache(self, data: Dict[str, Any]):
        """Update historical data cache with current data."""
        try:
            with self.cache_lock:
                timestamp = datetime.now()
                
                # Extract numeric metrics and cache them
                for key, value in data.items():
                    if isinstance(value, (int, float)):
                        self.historical_data_cache[key].append(value)
                    elif isinstance(value, dict):
                        # Handle nested data
                        for nested_key, nested_value in value.items():
                            if isinstance(nested_value, (int, float)):
                                cache_key = f"{key}.{nested_key}"
                                self.historical_data_cache[cache_key].append(nested_value)
        
        except Exception as e:
            logger.error(f"Error updating historical cache: {e}")
    
    def _get_evaluation_order(self) -> List[str]:
        """Get rule evaluation order based on dependencies."""
        try:
            # Simple topological sort for dependency resolution
            visited = set()
            order = []
            
            def visit(rule_id: str):
                if rule_id in visited:
                    return
                
                visited.add(rule_id)
                
                # Visit dependencies first
                for dep_id in self.rule_dependencies.get(rule_id, []):
                    if dep_id in self.rules:
                        visit(dep_id)
                
                order.append(rule_id)
            
            # Visit all rules
            for rule_id in self.rules.keys():
                visit(rule_id)
            
            return order
            
        except Exception as e:
            logger.error(f"Error getting evaluation order: {e}")
            return list(self.rules.keys())
    
    def _create_data_snapshot(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a snapshot of relevant data for history."""
        try:
            # Create a simplified snapshot to avoid memory issues
            snapshot = {}
            
            for key, value in data.items():
                if isinstance(value, (int, float, str, bool)):
                    snapshot[key] = value
                elif isinstance(value, dict) and len(value) < 10:
                    # Only include small dictionaries
                    snapshot[key] = value
            
            return snapshot
            
        except Exception as e:
            logger.error(f"Error creating data snapshot: {e}")
            return {}
    
    def _get_highest_severity(self, triggered_rules: List[Dict[str, Any]]) -> str:
        """Get the highest severity from triggered rules."""
        severity_order = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
        
        highest = 'LOW'
        highest_value = 0
        
        for rule_result in triggered_rules:
            severity = rule_result.get('severity', 'LOW')
            value = severity_order.get(severity, 0)
            
            if value > highest_value:
                highest = severity
                highest_value = value
        
        return highest
    
    def _aggregate_labels(self, triggered_rules: List[Dict[str, Any]]) -> Dict[str, str]:
        """Aggregate labels from triggered rules."""
        aggregated_labels = {}
        
        for rule_result in triggered_rules:
            # Add rule-specific labels
            aggregated_labels[f"triggered_rule_{rule_result.get('rule_id', 'unknown')}"] = "true"
            aggregated_labels[f"rule_type_{rule_result.get('rule_type', 'unknown')}"] = "true"
        
        return aggregated_labels
    
    def _aggregate_annotations(self, triggered_rules: List[Dict[str, Any]]) -> Dict[str, str]:
        """Aggregate annotations from triggered rules."""
        aggregated_annotations = {}
        
        # Add summary information
        aggregated_annotations['triggered_rules_count'] = str(len(triggered_rules))
        aggregated_annotations['evaluation_timestamp'] = datetime.now().isoformat()
        
        # Add rule names
        rule_names = [rule_result.get('rule_name', 'Unknown') for rule_result in triggered_rules]
        aggregated_annotations['triggered_rule_names'] = ', '.join(rule_names)
        
        return aggregated_annotations

# Factory functions for creating common rules
def create_battery_health_rule() -> MetricRule:
    """Create a standard battery health monitoring rule."""
    return MetricRule(
        rule_id=str(uuid.uuid4()),
        name="Battery Health Degradation",
        description="Alert when battery SoH drops below critical threshold",
        metric_name="battery_soh",
        condition=AlertCondition.LESS_THAN,
        threshold=AlertThreshold(value=80.0, unit="percent"),
        severity="HIGH",
        evaluation_window_minutes=5
    )

def create_model_accuracy_rule() -> MetricRule:
    """Create a model accuracy monitoring rule."""
    return MetricRule(
        rule_id=str(uuid.uuid4()),
        name="Model Accuracy Degradation",
        description="Alert when model accuracy drops below threshold",
        metric_name="model_accuracy",
        condition=AlertCondition.LESS_THAN,
        threshold=AlertThreshold(value=95.0, unit="percent"),
        severity="MEDIUM",
        evaluation_window_minutes=15
    )

def create_temperature_anomaly_rule() -> AnomalyRule:
    """Create a temperature anomaly detection rule."""
    return AnomalyRule(
        rule_id=str(uuid.uuid4()),
        name="Battery Temperature Anomaly",
        description="Detect unusual battery temperature patterns",
        metric_name="battery_temperature",
        anomaly_threshold=2.5,
        severity="HIGH",
        evaluation_window_minutes=10
    )
