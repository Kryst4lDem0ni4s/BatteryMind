"""
BatteryMind - Voting Classifier

Advanced voting-based ensemble classifier for battery health prediction
with sophisticated voting mechanisms, confidence weighting, and adaptive
model selection based on prediction reliability.

Features:
- Hard and soft voting mechanisms
- Confidence-weighted voting
- Dynamic model selection based on reliability
- Consensus-based decision making
- Uncertainty quantification through voting diversity
- Physics-informed voting constraints

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import logging
from collections import Counter
from scipy import stats
from scipy.special import softmax
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VotingConfig:
    """
    Configuration for voting-based ensemble classification.
    
    Attributes:
        # Voting mechanism
        voting_type (str): Type of voting ('hard', 'soft', 'confidence_weighted')
        consensus_threshold (float): Minimum consensus required for decision
        
        # Model reliability
        enable_reliability_weighting (bool): Weight votes by model reliability
        reliability_window (int): Window for calculating model reliability
        min_reliability_score (float): Minimum reliability score to include model
        
        # Confidence handling
        confidence_threshold (float): Minimum confidence for including vote
        use_prediction_confidence (bool): Use model's own confidence estimates
        
        # Adaptive selection
        enable_adaptive_selection (bool): Dynamically select models for voting
        selection_criteria (str): Criteria for model selection
        min_voters (int): Minimum number of models that must vote
        
        # Classification thresholds
        health_thresholds (Dict[str, float]): Thresholds for health classification
        degradation_thresholds (Dict[str, float]): Thresholds for degradation levels
        
        # Uncertainty quantification
        calculate_voting_entropy (bool): Calculate entropy of voting distribution
        consensus_confidence_mapping (bool): Map consensus to confidence scores
    """
    # Voting mechanism
    voting_type: str = "confidence_weighted"
    consensus_threshold: float = 0.6
    
    # Model reliability
    enable_reliability_weighting: bool = True
    reliability_window: int = 50
    min_reliability_score: float = 0.5
    
    # Confidence handling
    confidence_threshold: float = 0.5
    use_prediction_confidence: bool = True
    
    # Adaptive selection
    enable_adaptive_selection: bool = True
    selection_criteria: str = "reliability_based"
    min_voters: int = 2
    
    # Classification thresholds
    health_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'excellent': 0.95,
        'good': 0.85,
        'fair': 0.75,
        'poor': 0.65,
        'critical': 0.0
    })
    degradation_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'low': 0.001,
        'moderate': 0.005,
        'high': 0.01,
        'severe': 0.02
    })
    
    # Uncertainty quantification
    calculate_voting_entropy: bool = True
    consensus_confidence_mapping: bool = True

@dataclass
class VotingResult:
    """
    Result structure for voting-based classification.
    """
    # Classification results
    predicted_class: str
    class_probabilities: Dict[str, float]
    voting_consensus: float
    
    # Individual votes
    individual_votes: Dict[str, str]
    vote_confidences: Dict[str, float]
    
    # Reliability metrics
    voter_reliabilities: Dict[str, float]
    effective_voters: List[str]
    
    # Uncertainty metrics
    voting_entropy: Optional[float] = None
    consensus_confidence: Optional[float] = None
    prediction_uncertainty: Optional[float] = None
    
    # Metadata
    voting_method: str = "hard"
    models_excluded: List[str] = field(default_factory=list)
    consensus_threshold_met: bool = True

class ModelReliabilityTracker:
    """
    Tracks and manages model reliability scores over time.
    """
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.prediction_history = {}
        self.accuracy_history = {}
        self.reliability_scores = {}
    
    def update_model_performance(self, model_id: str, prediction: Any, 
                                ground_truth: Any, confidence: float = 1.0) -> None:
        """
        Update model performance tracking.
        
        Args:
            model_id (str): Model identifier
            prediction: Model prediction
            ground_truth: Actual ground truth
            confidence (float): Prediction confidence
        """
        if model_id not in self.prediction_history:
            self.prediction_history[model_id] = []
            self.accuracy_history[model_id] = []
        
        # Calculate accuracy (simplified for demonstration)
        if isinstance(prediction, str) and isinstance(ground_truth, str):
            accuracy = 1.0 if prediction == ground_truth else 0.0
        else:
            # For numerical predictions, use relative accuracy
            try:
                error = abs(float(prediction) - float(ground_truth))
                accuracy = max(0.0, 1.0 - error)
            except:
                accuracy = 0.5  # Default for unparseable predictions
        
        # Weight accuracy by confidence
        weighted_accuracy = accuracy * confidence
        
        # Update history
        self.prediction_history[model_id].append((prediction, ground_truth, confidence))
        self.accuracy_history[model_id].append(weighted_accuracy)
        
        # Maintain window size
        if len(self.accuracy_history[model_id]) > self.window_size:
            self.prediction_history[model_id].pop(0)
            self.accuracy_history[model_id].pop(0)
        
        # Update reliability score
        self._calculate_reliability_score(model_id)
    
    def _calculate_reliability_score(self, model_id: str) -> None:
        """Calculate reliability score for a model."""
        if model_id not in self.accuracy_history or not self.accuracy_history[model_id]:
            self.reliability_scores[model_id] = 0.5  # Default score
            return
        
        accuracies = self.accuracy_history[model_id]
        
        # Calculate weighted average with recent bias
        weights = np.exp(np.linspace(0, 1, len(accuracies)))  # Exponential weights favoring recent
        weighted_accuracy = np.average(accuracies, weights=weights)
        
        # Calculate consistency (inverse of variance)
        consistency = 1.0 / (1.0 + np.var(accuracies))
        
        # Combine accuracy and consistency
        reliability = 0.7 * weighted_accuracy + 0.3 * consistency
        
        self.reliability_scores[model_id] = reliability
    
    def get_reliability_score(self, model_id: str) -> float:
        """Get current reliability score for a model."""
        return self.reliability_scores.get(model_id, 0.5)
    
    def get_reliable_models(self, min_score: float = 0.5) -> List[str]:
        """Get list of models above reliability threshold."""
        return [model_id for model_id, score in self.reliability_scores.items() 
                if score >= min_score]

class BatteryVotingClassifier:
    """
    Voting-based ensemble classifier for battery health and degradation prediction.
    """
    
    def __init__(self, config: VotingConfig):
        self.config = config
        self.models = {}
        self.reliability_tracker = ModelReliabilityTracker(config.reliability_window)
        
    def add_model(self, model_id: str, model: Any) -> None:
        """
        Add a model to the voting ensemble.
        
        Args:
            model_id (str): Unique identifier for the model
            model: Model instance with predict method
        """
        self.models[model_id] = model
        logger.info(f"Added model {model_id} to voting classifier")
    
    def _classify_health_state(self, soh_value: float) -> str:
        """Classify battery health state based on SoH value."""
        thresholds = self.config.health_thresholds
        
        if soh_value >= thresholds['excellent']:
            return 'excellent'
        elif soh_value >= thresholds['good']:
            return 'good'
        elif soh_value >= thresholds['fair']:
            return 'fair'
        elif soh_value >= thresholds['poor']:
            return 'poor'
        else:
            return 'critical'
    
    def _classify_degradation_level(self, degradation_rate: float) -> str:
        """Classify degradation level based on degradation rate."""
        thresholds = self.config.degradation_thresholds
        
        if degradation_rate <= thresholds['low']:
            return 'low'
        elif degradation_rate <= thresholds['moderate']:
            return 'moderate'
        elif degradation_rate <= thresholds['high']:
            return 'high'
        else:
            return 'severe'
    
    def _get_model_predictions(self, inputs: Any) -> Dict[str, Dict[str, Any]]:
        """Get predictions from all models."""
        predictions = {}
        
        for model_id, model in self.models.items():
            try:
                # Get prediction based on model type
                if hasattr(model, 'predict_health'):
                    result = model.predict_health(inputs)
                elif hasattr(model, 'predict'):
                    result = model.predict(inputs)
                else:
                    logger.warning(f"Model {model_id} has no predict method")
                    continue
                
                # Extract relevant information
                if hasattr(result, 'state_of_health'):
                    # Battery health predictor result
                    soh = result.state_of_health
                    confidence = getattr(result, 'confidence_score', 0.8)
                    health_class = self._classify_health_state(soh)
                    
                    degradation_rate = result.degradation_patterns.get('capacity_fade_rate', 0.0)
                    degradation_class = self._classify_degradation_level(degradation_rate)
                    
                    predictions[model_id] = {
                        'health_class': health_class,
                        'degradation_class': degradation_class,
                        'soh_value': soh,
                        'degradation_rate': degradation_rate,
                        'confidence': confidence
                    }
                    
                elif isinstance(result, dict) and 'forecasts' in result:
                    # Degradation forecaster result
                    forecasts = result['forecasts']
                    if isinstance(forecasts, torch.Tensor):
                        forecasts = forecasts.cpu().numpy()
                    
                    # Use mean degradation rate for classification
                    mean_degradation = np.mean(forecasts) if len(forecasts.shape) > 0 else forecasts
                    degradation_class = self._classify_degradation_level(mean_degradation)
                    
                    # Estimate health class from degradation (simplified)
                    estimated_soh = max(0.5, 1.0 - mean_degradation * 100)  # Rough estimate
                    health_class = self._classify_health_state(estimated_soh)
                    
                    confidence = result.get('ensemble_confidence', 0.7)
                    
                    predictions[model_id] = {
                        'health_class': health_class,
                        'degradation_class': degradation_class,
                        'soh_value': estimated_soh,
                        'degradation_rate': mean_degradation,
                        'confidence': confidence
                    }
                    
                else:
                    # Generic numerical result
                    if isinstance(result, (int, float)):
                        soh = float(result)
                        health_class = self._classify_health_state(soh)
                        
                        predictions[model_id] = {
                            'health_class': health_class,
                            'degradation_class': 'unknown',
                            'soh_value': soh,
                            'degradation_rate': 0.0,
                            'confidence': 0.6
                        }
                    else:
                        logger.warning(f"Unknown result format from model {model_id}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error getting prediction from model {model_id}: {e}")
                continue
        
        return predictions
    
    def _select_voting_models(self, all_predictions: Dict[str, Dict[str, Any]]) -> List[str]:
        """Select models for voting based on reliability and confidence."""
        if not self.config.enable_adaptive_selection:
            return list(all_predictions.keys())
        
        eligible_models = []
        
        for model_id, prediction in all_predictions.items():
            # Check confidence threshold
            if prediction['confidence'] < self.config.confidence_threshold:
                continue
            
            # Check reliability threshold
            if self.config.enable_reliability_weighting:
                reliability = self.reliability_tracker.get_reliability_score(model_id)
                if reliability < self.config.min_reliability_score:
                    continue
            
            eligible_models.append(model_id)
        
        # Ensure minimum number of voters
        if len(eligible_models) < self.config.min_voters:
            # Add models with highest confidence/reliability
            remaining_models = [m for m in all_predictions.keys() if m not in eligible_models]
            if remaining_models:
                # Sort by confidence * reliability
                def score_model(model_id):
                    pred = all_predictions[model_id]
                    reliability = self.reliability_tracker.get_reliability_score(model_id)
                    return pred['confidence'] * reliability
                
                remaining_models.sort(key=score_model, reverse=True)
                needed = self.config.min_voters - len(eligible_models)
                eligible_models.extend(remaining_models[:needed])
        
        return eligible_models
    
    def _perform_hard_voting(self, predictions: Dict[str, Dict[str, Any]], 
                           voting_models: List[str]) -> Dict[str, Any]:
        """Perform hard voting (majority vote)."""
        health_votes = [predictions[m]['health_class'] for m in voting_models]
        degradation_votes = [predictions[m]['degradation_class'] for m in voting_models 
                           if predictions[m]['degradation_class'] != 'unknown']
        
        # Count votes
        health_counts = Counter(health_votes)
        degradation_counts = Counter(degradation_votes) if degradation_votes else Counter(['unknown'])
        
        # Determine winners
        health_winner = health_counts.most_common(1)[0][0]
        degradation_winner = degradation_counts.most_common(1)[0][0]
        
        # Calculate consensus
        health_consensus = health_counts[health_winner] / len(health_votes)
        degradation_consensus = (degradation_counts[degradation_winner] / len(degradation_votes) 
                               if degradation_votes else 1.0)
        
        # Calculate class probabilities
        health_probs = {cls: count / len(health_votes) for cls, count in health_counts.items()}
        degradation_probs = {cls: count / len(degradation_votes) for cls, count in degradation_counts.items()} if degradation_votes else {'unknown': 1.0}
        
        return {
            'health_class': health_winner,
            'degradation_class': degradation_winner,
            'health_consensus': health_consensus,
            'degradation_consensus': degradation_consensus,
            'health_probabilities': health_probs,
            'degradation_probabilities': degradation_probs,
            'individual_votes': {m: {'health': predictions[m]['health_class'], 
                                   'degradation': predictions[m]['degradation_class']} 
                               for m in voting_models}
        }
    
    def _perform_soft_voting(self, predictions: Dict[str, Dict[str, Any]], 
                           voting_models: List[str]) -> Dict[str, Any]:
        """Perform soft voting using confidence scores."""
        # Aggregate confidence-weighted votes
        health_scores = {}
        degradation_scores = {}
        
        total_confidence = sum(predictions[m]['confidence'] for m in voting_models)
        
        for model_id in voting_models:
            pred = predictions[model_id]
            weight = pred['confidence'] / total_confidence
            
            # Health voting
            health_class = pred['health_class']
            if health_class not in health_scores:
                health_scores[health_class] = 0.0
            health_scores[health_class] += weight
            
            # Degradation voting
            degradation_class = pred['degradation_class']
            if degradation_class != 'unknown':
                if degradation_class not in degradation_scores:
                    degradation_scores[degradation_class] = 0.0
                degradation_scores[degradation_class] += weight
        
        # Normalize scores
        if health_scores:
            total_health = sum(health_scores.values())
            health_scores = {k: v / total_health for k, v in health_scores.items()}
        
        if degradation_scores:
            total_degradation = sum(degradation_scores.values())
            degradation_scores = {k: v / total_degradation for k, v in degradation_scores.items()}
        else:
            degradation_scores = {'unknown': 1.0}
        
        # Determine winners
        health_winner = max(health_scores.items(), key=lambda x: x[1])[0]
        degradation_winner = max(degradation_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'health_class': health_winner,
            'degradation_class': degradation_winner,
            'health_consensus': health_scores[health_winner],
            'degradation_consensus': degradation_scores[degradation_winner],
            'health_probabilities': health_scores,
            'degradation_probabilities': degradation_scores,
            'individual_votes': {m: {'health': predictions[m]['health_class'], 
                                   'degradation': predictions[m]['degradation_class']} 
                               for m in voting_models}
        }
    
    def _perform_confidence_weighted_voting(self, predictions: Dict[str, Dict[str, Any]], 
                                          voting_models: List[str]) -> Dict[str, Any]:
        """Perform confidence-weighted voting with reliability factors."""
        # Calculate combined weights (confidence * reliability)
        weights = {}
        total_weight = 0.0
        
        for model_id in voting_models:
            confidence = predictions[model_id]['confidence']
            reliability = self.reliability_tracker.get_reliability_score(model_id)
            weight = confidence * reliability
            weights[model_id] = weight
            total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            weights = {k: 1.0 / len(voting_models) for k in voting_models}
        
        # Aggregate weighted votes
        health_scores = {}
        degradation_scores = {}
        
        for model_id in voting_models:
            pred = predictions[model_id]
            weight = weights[model_id]
            
            # Health voting
            health_class = pred['health_class']
            if health_class not in health_scores:
                health_scores[health_class] = 0.0
            health_scores[health_class] += weight
            
            # Degradation voting
            degradation_class = pred['degradation_class']
            if degradation_class != 'unknown':
                if degradation_class not in degradation_scores:
                    degradation_scores[degradation_class] = 0.0
                degradation_scores[degradation_class] += weight
        
        # Handle case where no degradation votes
        if not degradation_scores:
            degradation_scores = {'unknown': 1.0}
        
        # Determine winners
        health_winner = max(health_scores.items(), key=lambda x: x[1])[0]
        degradation_winner = max(degradation_scores.items(), key=lambda x: x[1])[0]
        
        return {
            'health_class': health_winner,
            'degradation_class': degradation_winner,
            'health_consensus': health_scores[health_winner],
            'degradation_consensus': degradation_scores[degradation_winner],
            'health_probabilities': health_scores,
            'degradation_probabilities': degradation_scores,
            'individual_votes': {m: {'health': predictions[m]['health_class'], 
                                   'degradation': predictions[m]['degradation_class']} 
                               for m in voting_models},
            'model_weights': weights
        }
    
    def _calculate_voting_entropy(self, probabilities: Dict[str, float]) -> float:
        """Calculate entropy of voting distribution."""
        if not probabilities:
            return 0.0
        
        probs = np.array(list(probabilities.values()))
        probs = probs[probs > 0]  # Remove zero probabilities
        
        if len(probs) == 0:
            return 0.0
        
        return -np.sum(probs * np.log2(probs))
    
    def predict(self, inputs: Any, return_details: bool = False) -> Union[VotingResult, str]:
        """
        Make voting-based prediction.
        
        Args:
            inputs: Input data for prediction
            return_details (bool): Whether to return detailed voting results
            
        Returns:
            Union[VotingResult, str]: Voting result or simple class prediction
        """
        if not self.models:
            raise ValueError("No models available for voting")
        
        # Get predictions from all models
        all_predictions = self._get_model_predictions(inputs)
        
        if not all_predictions:
            raise ValueError("No valid predictions obtained from models")
        
        # Select models for voting
        voting_models = self._select_voting_models(all_predictions)
        excluded_models = [m for m in all_predictions.keys() if m not in voting_models]
        
        if len(voting_models) < self.config.min_voters:
            logger.warning(f"Only {len(voting_models)} models available for voting (minimum: {self.config.min_voters})")
        
        # Perform voting based on configured method
        if self.config.voting_type == "hard":
            voting_result = self._perform_hard_voting(all_predictions, voting_models)
        elif self.config.voting_type == "soft":
            voting_result = self._perform_soft_voting(all_predictions, voting_models)
        elif self.config.voting_type == "confidence_weighted":
            voting_result = self._perform_confidence_weighted_voting(all_predictions, voting_models)
        else:
            raise ValueError(f"Unknown voting type: {self.config.voting_type}")
        
        # Calculate uncertainty metrics
        voting_entropy = None
        if self.config.calculate_voting_entropy:
            health_entropy = self._calculate_voting_entropy(voting_result['health_probabilities'])
            degradation_entropy = self._calculate_voting_entropy(voting_result['degradation_probabilities'])
            voting_entropy = (health_entropy + degradation_entropy) / 2
        
        # Map consensus to confidence
        consensus_confidence = None
        if self.config.consensus_confidence_mapping:
            avg_consensus = (voting_result['health_consensus'] + voting_result['degradation_consensus']) / 2
            consensus_confidence = avg_consensus
        
        # Check consensus threshold
        consensus_met = (voting_result['health_consensus'] >= self.config.consensus_threshold and
                        voting_result['degradation_consensus'] >= self.config.consensus_threshold)
        
        # Create result object
        result = VotingResult(
            predicted_class=f"{voting_result['health_class']}_{voting_result['degradation_class']}",
            class_probabilities={
                'health': voting_result['health_probabilities'],
                'degradation': voting_result['degradation_probabilities']
            },
            voting_consensus=(voting_result['health_consensus'] + voting_result['degradation_consensus']) / 2,
            individual_votes=voting_result['individual_votes'],
            vote_confidences={m: all_predictions[m]['confidence'] for m in voting_models},
            voter_reliabilities={m: self.reliability_tracker.get_reliability_score(m) for m in voting_models},
            effective_voters=voting_models,
            voting_entropy=voting_entropy,
            consensus_confidence=consensus_confidence,
            voting_method=self.config.voting_type,
            models_excluded=excluded_models,
            consensus_threshold_met=consensus_met
        )
        
        if return_details:
            return result
        else:
            return result.predicted_class
    
    def update_model_performance(self, model_id: str, prediction: Any, 
                               ground_truth: Any, confidence: float = 1.0) -> None:
        """Update model performance for reliability tracking."""
        self.reliability_tracker.update_model_performance(model_id, prediction, ground_truth, confidence)
    
    def get_model_reliabilities(self) -> Dict[str, float]:
        """Get current reliability scores for all models."""
        return {model_id: self.reliability_tracker.get_reliability_score(model_id) 
                for model_id in self.models.keys()}
    
    def get_voting_statistics(self) -> Dict[str, Any]:
        """Get comprehensive voting statistics."""
        reliabilities = self.get_model_reliabilities()
        
        return {
            'total_models': len(self.models),
            'reliable_models': len(self.reliability_tracker.get_reliable_models(self.config.min_reliability_score)),
            'average_reliability': np.mean(list(reliabilities.values())) if reliabilities else 0.0,
            'model_reliabilities': reliabilities,
            'voting_config': self.config.__dict__
        }

# Factory function
def create_battery_voting_classifier(config: Optional[VotingConfig] = None) -> BatteryVotingClassifier:
    """
    Factory function to create a battery voting classifier.
    
    Args:
        config (VotingConfig, optional): Voting configuration
        
    Returns:
        BatteryVotingClassifier: Configured voting classifier
    """
    if config is None:
        config = VotingConfig()
    
    return BatteryVotingClassifier(config)
