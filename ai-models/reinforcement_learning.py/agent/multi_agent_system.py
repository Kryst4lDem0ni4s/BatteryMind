"""
BatteryMind - Multi-Agent System

Coordinated multi-agent reinforcement learning system for complex battery
management scenarios involving multiple specialized agents working together
to optimize different aspects of battery performance and safety.

Features:
- Hierarchical multi-agent coordination with specialized roles
- Decentralized decision making with centralized coordination
- Communication protocols between agents for information sharing
- Conflict resolution mechanisms for competing objectives
- Scalable architecture for large battery fleets
- Integration with federated learning for distributed intelligence

Author: BatteryMind Development Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
import logging
import time
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Local imports
from .charging_agent import ChargingAgent, ChargingConfig
from .thermal_agent import ThermalAgent, ThermalConfig
from .load_balancing_agent import LoadBalancingAgent, LoadBalancingConfig
from ..environments.battery_env import BatteryEnvironment
from ..algorithms.ppo import PPOAlgorithm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MultiAgentConfig:
    """
    Configuration for multi-agent system.
    
    Attributes:
        # System architecture
        coordination_strategy (str): Strategy for agent coordination
        communication_protocol (str): Protocol for inter-agent communication
        conflict_resolution (str): Method for resolving agent conflicts
        
        # Agent configuration
        enable_charging_agent (bool): Enable charging optimization agent
        enable_thermal_agent (bool): Enable thermal management agent
        enable_load_balancing_agent (bool): Enable load balancing agent
        
        # Coordination parameters
        coordination_frequency (float): Frequency of coordination updates (Hz)
        communication_range (float): Range for agent communication
        consensus_threshold (float): Threshold for reaching consensus
        
        # Learning parameters
        shared_experience (bool): Share experiences between agents
        centralized_critic (bool): Use centralized critic for training
        decentralized_execution (bool): Use decentralized execution
        
        # Performance optimization
        parallel_execution (bool): Execute agents in parallel
        asynchronous_updates (bool): Use asynchronous agent updates
        load_balancing (bool): Balance computational load across agents
        
        # Safety and constraints
        global_safety_constraints (Dict[str, float]): Global safety constraints
        priority_hierarchy (List[str]): Agent priority hierarchy
        emergency_protocols (Dict[str, Any]): Emergency response protocols
    """
    # System architecture
    coordination_strategy: str = "hierarchical"
    communication_protocol: str = "broadcast"
    conflict_resolution: str = "priority_based"
    
    # Agent configuration
    enable_charging_agent: bool = True
    enable_thermal_agent: bool = True
    enable_load_balancing_agent: bool = True
    
    # Coordination parameters
    coordination_frequency: float = 1.0
    communication_range: float = 100.0
    consensus_threshold: float = 0.8
    
    # Learning parameters
    shared_experience: bool = True
    centralized_critic: bool = True
    decentralized_execution: bool = True
    
    # Performance optimization
    parallel_execution: bool = True
    asynchronous_updates: bool = True
    load_balancing: bool = True
    
    # Safety and constraints
    global_safety_constraints: Dict[str, float] = field(default_factory=lambda: {
        'max_temperature': 50.0,
        'max_voltage': 4.2,
        'min_voltage': 2.8,
        'max_current': 100.0
    })
    priority_hierarchy: List[str] = field(default_factory=lambda: [
        'thermal_agent', 'charging_agent', 'load_balancing_agent'
    ])
    emergency_protocols: Dict[str, Any] = field(default_factory=dict)

class AgentMessage:
    """Message structure for inter-agent communication."""
    
    def __init__(self, sender_id: str, receiver_id: str, message_type: str, 
                 content: Dict[str, Any], timestamp: float = None):
        self.sender_id = sender_id
        self.receiver_id = receiver_id
        self.message_type = message_type
        self.content = content
        self.timestamp = timestamp or time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        return {
            'sender_id': self.sender_id,
            'receiver_id': self.receiver_id,
            'message_type': self.message_type,
            'content': self.content,
            'timestamp': self.timestamp
        }

class CommunicationProtocol:
    """Communication protocol for multi-agent coordination."""
    
    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.message_queue = deque(maxlen=10000)
        self.agent_registry = {}
        self.communication_history = defaultdict(list)
        
    def register_agent(self, agent_id: str, agent_instance):
        """Register an agent in the communication system."""
        self.agent_registry[agent_id] = agent_instance
        logger.info(f"Agent {agent_id} registered in communication system")
    
    def send_message(self, message: AgentMessage):
        """Send message through the communication protocol."""
        if self.config.communication_protocol == "broadcast":
            self._broadcast_message(message)
        elif self.config.communication_protocol == "direct":
            self._direct_message(message)
        elif self.config.communication_protocol == "hierarchical":
            self._hierarchical_message(message)
        
        # Store in history
        self.communication_history[message.sender_id].append(message)
    
    def _broadcast_message(self, message: AgentMessage):
        """Broadcast message to all agents."""
        for agent_id, agent in self.agent_registry.items():
            if agent_id != message.sender_id:
                self._deliver_message(agent_id, message)
    
    def _direct_message(self, message: AgentMessage):
        """Send direct message to specific agent."""
        if message.receiver_id in self.agent_registry:
            self._deliver_message(message.receiver_id, message)
    
    def _hierarchical_message(self, message: AgentMessage):
        """Send message through hierarchical structure."""
        # Implement hierarchical routing based on agent priorities
        sender_priority = self._get_agent_priority(message.sender_id)
        
        for agent_id, agent in self.agent_registry.items():
            if agent_id != message.sender_id:
                receiver_priority = self._get_agent_priority(agent_id)
                
                # Send to agents with lower or equal priority
                if receiver_priority >= sender_priority:
                    self._deliver_message(agent_id, message)
    
    def _deliver_message(self, agent_id: str, message: AgentMessage):
        """Deliver message to specific agent."""
        if hasattr(self.agent_registry[agent_id], 'receive_message'):
            self.agent_registry[agent_id].receive_message(message)
    
    def _get_agent_priority(self, agent_id: str) -> int:
        """Get agent priority from hierarchy."""
        try:
            return self.config.priority_hierarchy.index(agent_id)
        except ValueError:
            return len(self.config.priority_hierarchy)  # Lowest priority
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics."""
        total_messages = sum(len(messages) for messages in self.communication_history.values())
        
        return {
            'total_messages': total_messages,
            'active_agents': len(self.agent_registry),
            'messages_per_agent': {
                agent_id: len(messages) 
                for agent_id, messages in self.communication_history.items()
            },
            'avg_messages_per_agent': total_messages / len(self.agent_registry) if self.agent_registry else 0
        }

class ConflictResolver:
    """Conflict resolution system for multi-agent decisions."""
    
    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.conflict_history = []
        
    def resolve_conflict(self, agent_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Resolve conflicts between agent decisions.
        
        Args:
            agent_decisions (Dict[str, Dict[str, Any]]): Decisions from each agent
            
        Returns:
            Dict[str, Any]: Resolved decision
        """
        if self.config.conflict_resolution == "priority_based":
            return self._priority_based_resolution(agent_decisions)
        elif self.config.conflict_resolution == "voting":
            return self._voting_based_resolution(agent_decisions)
        elif self.config.conflict_resolution == "optimization":
            return self._optimization_based_resolution(agent_decisions)
        else:
            return self._consensus_based_resolution(agent_decisions)
    
    def _priority_based_resolution(self, agent_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts based on agent priority hierarchy."""
        resolved_decision = {}
        
        # Sort agents by priority
        sorted_agents = sorted(
            agent_decisions.keys(),
            key=lambda x: self._get_agent_priority(x)
        )
        
        # Apply decisions in priority order
        for agent_id in sorted_agents:
            decision = agent_decisions[agent_id]
            
            # Check for conflicts with existing decisions
            conflicts = self._detect_conflicts(resolved_decision, decision)
            
            if not conflicts:
                # No conflicts, apply decision
                resolved_decision.update(decision)
            else:
                # Resolve conflicts based on priority
                for param, value in decision.items():
                    if param not in resolved_decision or self._should_override(agent_id, param):
                        resolved_decision[param] = value
        
        # Record conflict resolution
        self.conflict_history.append({
            'timestamp': time.time(),
            'agent_decisions': agent_decisions,
            'resolved_decision': resolved_decision,
            'resolution_method': 'priority_based'
        })
        
        return resolved_decision
    
    def _voting_based_resolution(self, agent_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts through voting mechanism."""
        resolved_decision = {}
        
        # Collect all parameters
        all_params = set()
        for decision in agent_decisions.values():
            all_params.update(decision.keys())
        
        # Vote on each parameter
        for param in all_params:
            votes = []
            for agent_id, decision in agent_decisions.items():
                if param in decision:
                    votes.append(decision[param])
            
            if votes:
                # Use median for numerical values, mode for categorical
                if isinstance(votes[0], (int, float)):
                    resolved_decision[param] = np.median(votes)
                else:
                    # Use most common value
                    from collections import Counter
                    resolved_decision[param] = Counter(votes).most_common(1)[0][0]
        
        return resolved_decision
    
    def _optimization_based_resolution(self, agent_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts through optimization."""
        # Implement multi-objective optimization to find compromise solution
        # This is a simplified implementation
        
        resolved_decision = {}
        
        # Weight decisions by agent importance
        agent_weights = {
            'thermal_agent': 0.4,
            'charging_agent': 0.35,
            'load_balancing_agent': 0.25
        }
        
        # Collect all parameters
        all_params = set()
        for decision in agent_decisions.values():
            all_params.update(decision.keys())
        
        # Weighted average for numerical parameters
        for param in all_params:
            weighted_values = []
            total_weight = 0.0
            
            for agent_id, decision in agent_decisions.items():
                if param in decision and isinstance(decision[param], (int, float)):
                    weight = agent_weights.get(agent_id, 0.1)
                    weighted_values.append(decision[param] * weight)
                    total_weight += weight
            
            if weighted_values and total_weight > 0:
                resolved_decision[param] = sum(weighted_values) / total_weight
        
        return resolved_decision
    
    def _consensus_based_resolution(self, agent_decisions: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve conflicts through consensus building."""
        resolved_decision = {}
        
        # Find parameters where agents agree
        all_params = set()
        for decision in agent_decisions.values():
            all_params.update(decision.keys())
        
        for param in all_params:
            values = []
            for decision in agent_decisions.values():
                if param in decision:
                    values.append(decision[param])
            
            if values:
                # Check for consensus (similar values)
                if isinstance(values[0], (int, float)):
                    if np.std(values) / np.mean(values) < 0.1:  # Low coefficient of variation
                        resolved_decision[param] = np.mean(values)
                    else:
                        # No consensus, use median
                        resolved_decision[param] = np.median(values)
                else:
                    # For categorical values, use most common if consensus exists
                    from collections import Counter
                    counts = Counter(values)
                    most_common_count = counts.most_common(1)[0][1]
                    
                    if most_common_count / len(values) >= self.config.consensus_threshold:
                        resolved_decision[param] = counts.most_common(1)[0][0]
        
        return resolved_decision
    
    def _detect_conflicts(self, existing_decision: Dict[str, Any], 
                         new_decision: Dict[str, Any]) -> List[str]:
        """Detect conflicts between decisions."""
        conflicts = []
        
        for param, value in new_decision.items():
            if param in existing_decision:
                existing_value = existing_decision[param]
                
                # Check for significant differences
                if isinstance(value, (int, float)) and isinstance(existing_value, (int, float)):
                    if abs(value - existing_value) / max(abs(value), abs(existing_value), 1e-6) > 0.1:
                        conflicts.append(param)
                elif value != existing_value:
                    conflicts.append(param)
        
        return conflicts
    
    def _get_agent_priority(self, agent_id: str) -> int:
        """Get agent priority from hierarchy."""
        try:
            return self.config.priority_hierarchy.index(agent_id)
        except ValueError:
            return len(self.config.priority_hierarchy)
    
    def _should_override(self, agent_id: str, param: str) -> bool:
        """Determine if agent should override existing decision for parameter."""
        # Higher priority agents can override lower priority decisions
        # Safety-critical parameters have special handling
        
        safety_critical_params = ['temperature_limit', 'voltage_limit', 'current_limit']
        
        if param in safety_critical_params and agent_id == 'thermal_agent':
            return True
        
        return False

class MultiAgentSystem:
    """
    Coordinated multi-agent system for comprehensive battery management.
    """
    
    def __init__(self, config: MultiAgentConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize communication system
        self.communication = CommunicationProtocol(config)
        self.conflict_resolver = ConflictResolver(config)
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Performance tracking
        self.system_metrics = {}
        self.coordination_history = []
        
        # Execution management
        self.executor = ThreadPoolExecutor(max_workers=4) if config.parallel_execution else None
        
        logger.info(f"MultiAgentSystem initialized with {len(self.agents)} agents")
    
    def _initialize_agents(self):
        """Initialize all configured agents."""
        if self.config.enable_charging_agent:
            charging_config = ChargingConfig()
            self.agents['charging_agent'] = ChargingAgent(charging_config)
            self.communication.register_agent('charging_agent', self.agents['charging_agent'])
        
        if self.config.enable_thermal_agent:
            thermal_config = ThermalConfig()
            self.agents['thermal_agent'] = ThermalAgent(thermal_config)
            self.communication.register_agent('thermal_agent', self.agents['thermal_agent'])
        
        if self.config.enable_load_balancing_agent:
            load_config = LoadBalancingConfig()
            self.agents['load_balancing_agent'] = LoadBalancingAgent(load_config)
            self.communication.register_agent('load_balancing_agent', self.agents['load_balancing_agent'])
    
    def coordinate_agents(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate agents to make collective decisions.
        
        Args:
            system_state (Dict[str, Any]): Current system state
            
        Returns:
            Dict[str, Any]: Coordinated system actions
        """
        start_time = time.time()
        
        # Get decisions from all agents
        if self.config.parallel_execution:
            agent_decisions = self._get_parallel_decisions(system_state)
        else:
            agent_decisions = self._get_sequential_decisions(system_state)
        
        # Resolve conflicts between agent decisions
        resolved_decision = self.conflict_resolver.resolve_conflict(agent_decisions)
        
        # Apply global safety constraints
        safe_decision = self._apply_safety_constraints(resolved_decision, system_state)
        
        # Broadcast final decision to all agents
        self._broadcast_final_decision(safe_decision)
        
        # Record coordination
        coordination_time = time.time() - start_time
        self.coordination_history.append({
            'timestamp': time.time(),
            'system_state': system_state,
            'agent_decisions': agent_decisions,
            'resolved_decision': resolved_decision,
            'final_decision': safe_decision,
            'coordination_time': coordination_time
        })
        
        return safe_decision
    
    def _get_parallel_decisions(self, system_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get decisions from agents in parallel."""
        if not self.executor:
            return self._get_sequential_decisions(system_state)
        
        # Submit tasks to thread pool
        futures = {}
        for agent_id, agent in self.agents.items():
            future = self.executor.submit(self._get_agent_decision, agent_id, agent, system_state)
            futures[agent_id] = future
        
        # Collect results
        agent_decisions = {}
        for agent_id, future in futures.items():
            try:
                agent_decisions[agent_id] = future.result(timeout=5.0)  # 5 second timeout
            except Exception as e:
                logger.warning(f"Agent {agent_id} decision failed: {e}")
                agent_decisions[agent_id] = {}
        
        return agent_decisions
    
    def _get_sequential_decisions(self, system_state: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Get decisions from agents sequentially."""
        agent_decisions = {}
        
        for agent_id, agent in self.agents.items():
            try:
                agent_decisions[agent_id] = self._get_agent_decision(agent_id, agent, system_state)
            except Exception as e:
                logger.warning(f"Agent {agent_id} decision failed: {e}")
                agent_decisions[agent_id] = {}
        
        return agent_decisions
    
    def _get_agent_decision(self, agent_id: str, agent, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Get decision from a specific agent."""
        if hasattr(agent, 'make_decision'):
            return agent.make_decision(system_state)
        elif hasattr(agent, 'predict'):
            # For agents with predict method (like load balancing)
            if agent_id == 'load_balancing_agent':
                battery_states = system_state.get('battery_states', {})
                total_load = system_state.get('total_load', 0.0)
                load_distribution = agent.predict_load_distribution(battery_states, total_load)
                return {'load_distribution': load_distribution}
        
        return {}
    
    def _apply_safety_constraints(self, decision: Dict[str, Any], 
                                 system_state: Dict[str, Any]) -> Dict[str, Any]:
        """Apply global safety constraints to the decision."""
        safe_decision = decision.copy()
        
        # Temperature constraints
        if 'temperature_setpoint' in safe_decision:
            max_temp = self.config.global_safety_constraints.get('max_temperature', 50.0)
            safe_decision['temperature_setpoint'] = min(safe_decision['temperature_setpoint'], max_temp)
        
        # Voltage constraints
        if 'voltage_setpoint' in safe_decision:
            max_voltage = self.config.global_safety_constraints.get('max_voltage', 4.2)
            min_voltage = self.config.global_safety_constraints.get('min_voltage', 2.8)
            safe_decision['voltage_setpoint'] = np.clip(
                safe_decision['voltage_setpoint'], min_voltage, max_voltage
            )
        
        # Current constraints
        if 'current_setpoint' in safe_decision:
            max_current = self.config.global_safety_constraints.get('max_current', 100.0)
            safe_decision['current_setpoint'] = min(safe_decision['current_setpoint'], max_current)
        
        # Load distribution constraints
        if 'load_distribution' in safe_decision:
            load_dist = safe_decision['load_distribution']
            if isinstance(load_dist, np.ndarray):
                # Ensure no individual load exceeds safety limits
                max_individual_load = self.config.global_safety_constraints.get('max_current', 100.0)
                safe_decision['load_distribution'] = np.clip(load_dist, 0, max_individual_load)
        
        return safe_decision
    
    def _broadcast_final_decision(self, decision: Dict[str, Any]):
        """Broadcast final decision to all agents."""
        message = AgentMessage(
            sender_id='coordinator',
            receiver_id='all',
            message_type='final_decision',
            content=decision
        )
        
        self.communication.send_message(message)
    
    def train_system(self, num_episodes: int = 1000, 
                    environment_config: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Train the multi-agent system.
        
        Args:
            num_episodes (int): Number of training episodes
            environment_config (Dict, optional): Environment configuration
            
        Returns:
            Dict[str, Any]: Training results
        """
        logger.info(f"Starting multi-agent system training for {num_episodes} episodes")
        
        # Create shared environment
        if environment_config is None:
            environment_config = {}
        
        env = BatteryEnvironment(environment_config)
        
        # Training loop
        episode_rewards = []
        coordination_times = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            episode_coordination_time = 0.0
            
            while True:
                # Coordinate agents to get action
                start_time = time.time()
                action = self.coordinate_agents(state)
                coordination_time = time.time() - start_time
                episode_coordination_time += coordination_time
                
                # Execute action in environment
                next_state, reward, done, info = env.step(action)
                
                # Train individual agents
                self._train_individual_agents(state, action, reward, next_state, done)
                
                episode_reward += reward
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            coordination_times.append(episode_coordination_time)
            
            if episode % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:])
                avg_coord_time = np.mean(coordination_times[-100:])
                logger.info(f"Episode {episode}: Avg Reward={avg_reward:.2f}, "
                           f"Avg Coordination Time={avg_coord_time:.3f}s")
        
        training_results = {
            'episode_rewards': episode_rewards,
            'coordination_times': coordination_times,
            'final_performance': self.evaluate_system(),
            'communication_stats': self.communication.get_communication_stats()
        }
        
        logger.info("Multi-agent system training completed")
        return training_results
    
    def _train_individual_agents(self, state: Dict[str, Any], action: Dict[str, Any],
                                reward: float, next_state: Dict[str, Any], done: bool):
        """Train individual agents with shared experience."""
        if not self.config.shared_experience:
            return
        
        # Extract agent-specific experiences and train
        for agent_id, agent in self.agents.items():
            if hasattr(agent, 'train') and hasattr(agent, 'algorithm'):
                # Create agent-specific state and action
                agent_state = self._extract_agent_state(state, agent_id)
                agent_action = action.get(f'{agent_id}_action', np.zeros(agent.config.action_dim))
                agent_next_state = self._extract_agent_state(next_state, agent_id)
                
                # Store experience and train
                agent.algorithm.store_experience(agent_state, agent_action, reward, agent_next_state, done)
                
                if len(agent.algorithm.memory) > agent.config.batch_size:
                    agent.algorithm.train()
    
    def _extract_agent_state(self, system_state: Dict[str, Any], agent_id: str) -> np.ndarray:
        """Extract agent-specific state from system state."""
        # This is a simplified extraction - in practice, each agent would have
        # specific state features relevant to its task
        
        if agent_id == 'thermal_agent':
            # Extract temperature-related features
            features = [
                system_state.get('temperature', 25.0) / 60.0,
                system_state.get('ambient_temperature', 25.0) / 60.0,
                system_state.get('cooling_power', 0.0) / 100.0
            ]
        elif agent_id == 'charging_agent':
            # Extract charging-related features
            features = [
                system_state.get('voltage', 3.7) / 4.2,
                system_state.get('current', 0.0) / 100.0,
                system_state.get('soc', 0.5),
                system_state.get('charging_rate', 0.0) / 2.0
            ]
        elif agent_id == 'load_balancing_agent':
            # Extract load balancing features
            battery_states = system_state.get('battery_states', {})
            features = []
            for i in range(8):  # Assuming 8 batteries
                if i in battery_states:
                    battery = battery_states[i]
                    features.extend([
                        battery.get('voltage', 3.7) / 4.2,
                        battery.get('current', 0.0) / 100.0,
                        battery.get('temperature', 25.0) / 60.0,
                        battery.get('soc', 0.5)
                    ])
                else:
                    features.extend([0.5, 0.0, 0.4, 0.5])
        else:
            features = [0.5] * 16  # Default features
        
        # Pad or truncate to expected size
        while len(features) < 32:
            features.append(0.0)
        
        return np.array(features[:32], dtype=np.float32)
    
    def evaluate_system(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate the multi-agent system performance."""
        # Create evaluation environment
        env = BatteryEnvironment({})
        
        total_reward = 0.0
        total_coordination_time = 0.0
        safety_violations = 0
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0.0
            
            while True:
                start_time = time.time()
                action = self.coordinate_agents(state)
                coordination_time = time.time() - start_time
                total_coordination_time += coordination_time
                
                next_state, reward, done, info = env.step(action)
                episode_reward += reward
                
                # Check for safety violations
                if info.get('safety_violation', False):
                    safety_violations += 1
                
                state = next_state
                if done:
                    break
            
            total_reward += episode_reward
        
        return {
            'average_reward': total_reward / num_episodes,
            'average_coordination_time': total_coordination_time / num_episodes,
            'safety_violation_rate': safety_violations / num_episodes,
            'communication_efficiency': len(self.communication.communication_history) / num_episodes
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        return {
            'active_agents': list(self.agents.keys()),
            'coordination_strategy': self.config.coordination_strategy,
            'communication_stats': self.communication.get_communication_stats(),
            'recent_coordination_time': self.coordination_history[-1]['coordination_time'] if self.coordination_history else 0,
            'system_health': self._assess_system_health(),
            'performance_metrics': self.system_metrics
        }
    
    def _assess_system_health(self) -> str:
        """Assess overall system health."""
        if not self.coordination_history:
            return "unknown"
        
        recent_coords = self.coordination_history[-10:]
        avg_coord_time = np.mean([coord['coordination_time'] for coord in recent_coords])
        
        if avg_coord_time < 0.1:
            return "excellent"
        elif avg_coord_time < 0.5:
            return "good"
        elif avg_coord_time < 1.0:
            return "fair"
        else:
            return "poor"
    
    def save_system(self, filepath: str):
        """Save the multi-agent system."""
        save_data = {
            'config': self.config,
            'agents': {agent_id: agent.get_state() if hasattr(agent, 'get_state') else None 
                      for agent_id, agent in self.agents.items()},
            'coordination_history': self.coordination_history[-1000:],  # Save last 1000 entries
            'system_metrics': self.system_metrics
        }
        
        torch.save(save_data, filepath)
        logger.info(f"Multi-agent system saved to {filepath}")
    
    def load_system(self, filepath: str):
        """Load the multi-agent system."""
        save_data = torch.load(filepath, map_location=self.device)
        
        self.config = save_data['config']
        self.coordination_history = save_data.get('coordination_history', [])
        self.system_metrics = save_data.get('system_metrics', {})
        
        # Load individual agents
        agent_states = save_data.get('agents', {})
        for agent_id, agent_state in agent_states.items():
            if agent_id in self.agents and agent_state and hasattr(self.agents[agent_id], 'load_state'):
                self.agents[agent_id].load_state(agent_state)
        
        logger.info(f"Multi-agent system loaded from {filepath}")

# Factory function
def create_multi_agent_system(config: Optional[MultiAgentConfig] = None) -> MultiAgentSystem:
    """
    Factory function to create a multi-agent system.
    
    Args:
        config (MultiAgentConfig, optional): System configuration
        
    Returns:
        MultiAgentSystem: Configured multi-agent system
    """
    if config is None:
        config = MultiAgentConfig()
    
    return MultiAgentSystem(config)
