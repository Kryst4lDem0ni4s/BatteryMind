"""
BatteryMind - Federated Learning Network Simulator

Advanced network simulation framework for federated learning environments.
Simulates realistic network conditions, latency, bandwidth limitations,
and communication patterns for distributed battery management systems.

Features:
- Realistic network topology simulation
- Dynamic bandwidth and latency modeling
- Network failure and recovery simulation
- Communication cost tracking
- Hierarchical network structures
- Mobile network simulation for vehicle fleets
- Network security and privacy modeling

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
import random
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import networkx as nx
from scipy.stats import norm, expon, gamma
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkType(Enum):
    """Types of network connections."""
    ETHERNET = "ethernet"
    WIFI = "wifi"
    LTE = "lte"
    FIVE_G = "5g"
    SATELLITE = "satellite"
    EDGE = "edge"

class NodeType(Enum):
    """Types of network nodes."""
    CLIENT = "client"
    EDGE_SERVER = "edge_server"
    CENTRAL_SERVER = "central_server"
    AGGREGATOR = "aggregator"
    COORDINATOR = "coordinator"

@dataclass
class NetworkConditions:
    """
    Network conditions for simulation.
    
    Attributes:
        bandwidth_mbps (float): Available bandwidth in Mbps
        latency_ms (float): Network latency in milliseconds
        packet_loss_rate (float): Packet loss rate (0-1)
        jitter_ms (float): Network jitter in milliseconds
        reliability (float): Connection reliability (0-1)
        congestion_level (float): Network congestion level (0-1)
        is_mobile (bool): Whether connection is mobile
        signal_strength (float): Signal strength for mobile connections
    """
    bandwidth_mbps: float = 100.0
    latency_ms: float = 10.0
    packet_loss_rate: float = 0.001
    jitter_ms: float = 2.0
    reliability: float = 0.99
    congestion_level: float = 0.1
    is_mobile: bool = False
    signal_strength: float = 1.0

@dataclass
class NetworkNode:
    """
    Network node in the federated learning system.
    
    Attributes:
        node_id (str): Unique node identifier
        node_type (NodeType): Type of network node
        location (Tuple[float, float]): Geographic coordinates
        network_type (NetworkType): Type of network connection
        conditions (NetworkConditions): Current network conditions
        computational_capacity (float): Computational capacity
        storage_capacity (float): Storage capacity in GB
        energy_level (float): Energy level (0-1) for mobile devices
        is_active (bool): Whether node is currently active
        last_seen (float): Timestamp of last activity
        metadata (Dict): Additional node metadata
    """
    node_id: str
    node_type: NodeType
    location: Tuple[float, float]
    network_type: NetworkType
    conditions: NetworkConditions = field(default_factory=NetworkConditions)
    computational_capacity: float = 1.0
    storage_capacity: float = 10.0
    energy_level: float = 1.0
    is_active: bool = True
    last_seen: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class NetworkMessage:
    """
    Network message for federated learning communication.
    
    Attributes:
        message_id (str): Unique message identifier
        sender_id (str): Sender node ID
        receiver_id (str): Receiver node ID
        message_type (str): Type of message
        payload_size_mb (float): Message payload size in MB
        priority (int): Message priority (1-10)
        timestamp (float): Message creation timestamp
        deadline (Optional[float]): Message deadline
        encrypted (bool): Whether message is encrypted
        compressed (bool): Whether message is compressed
        metadata (Dict): Additional message metadata
    """
    message_id: str
    sender_id: str
    receiver_id: str
    message_type: str
    payload_size_mb: float
    priority: int = 5
    timestamp: float = field(default_factory=time.time)
    deadline: Optional[float] = None
    encrypted: bool = True
    compressed: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

class NetworkTopology:
    """
    Network topology manager for federated learning simulation.
    """
    
    def __init__(self, topology_type: str = "hierarchical"):
        self.topology_type = topology_type
        self.graph = nx.Graph()
        self.nodes: Dict[str, NetworkNode] = {}
        self.edges: Dict[Tuple[str, str], Dict] = {}
        self.routing_table: Dict[str, Dict[str, List[str]]] = {}
        
    def add_node(self, node: NetworkNode) -> None:
        """Add a node to the network topology."""
        self.nodes[node.node_id] = node
        self.graph.add_node(node.node_id, **node.__dict__)
        logger.debug(f"Added node {node.node_id} to topology")
    
    def add_edge(self, node1_id: str, node2_id: str, 
                 conditions: Optional[NetworkConditions] = None) -> None:
        """Add an edge between two nodes."""
        if conditions is None:
            conditions = NetworkConditions()
        
        edge_key = (node1_id, node2_id)
        self.edges[edge_key] = {
            'conditions': conditions,
            'traffic_history': [],
            'last_updated': time.time()
        }
        
        self.graph.add_edge(node1_id, node2_id, **conditions.__dict__)
        logger.debug(f"Added edge between {node1_id} and {node2_id}")
    
    def create_hierarchical_topology(self, num_clients: int, num_edge_servers: int = 3) -> None:
        """Create a hierarchical network topology."""
        # Add central server
        central_server = NetworkNode(
            node_id="central_server",
            node_type=NodeType.CENTRAL_SERVER,
            location=(0.0, 0.0),
            network_type=NetworkType.ETHERNET,
            computational_capacity=10.0,
            storage_capacity=1000.0
        )
        self.add_node(central_server)
        
        # Add edge servers
        edge_servers = []
        for i in range(num_edge_servers):
            edge_server = NetworkNode(
                node_id=f"edge_server_{i}",
                node_type=NodeType.EDGE_SERVER,
                location=(random.uniform(-10, 10), random.uniform(-10, 10)),
                network_type=NetworkType.ETHERNET,
                computational_capacity=5.0,
                storage_capacity=100.0
            )
            self.add_node(edge_server)
            edge_servers.append(edge_server)
            
            # Connect to central server
            self.add_edge("central_server", edge_server.node_id, 
                         NetworkConditions(bandwidth_mbps=1000, latency_ms=5))
        
        # Add client nodes
        clients_per_edge = num_clients // num_edge_servers
        for i in range(num_clients):
            edge_idx = i // clients_per_edge
            if edge_idx >= len(edge_servers):
                edge_idx = len(edge_servers) - 1
            
            client = NetworkNode(
                node_id=f"client_{i}",
                node_type=NodeType.CLIENT,
                location=(random.uniform(-50, 50), random.uniform(-50, 50)),
                network_type=random.choice([NetworkType.WIFI, NetworkType.LTE, NetworkType.FIVE_G]),
                computational_capacity=random.uniform(0.5, 2.0),
                storage_capacity=random.uniform(1.0, 10.0),
                energy_level=random.uniform(0.3, 1.0),
                is_mobile=random.choice([True, False])
            )
            self.add_node(client)
            
            # Connect to assigned edge server
            edge_server_id = edge_servers[edge_idx].node_id
            client_conditions = self._generate_client_conditions(client)
            self.add_edge(client.node_id, edge_server_id, client_conditions)
    
    def _generate_client_conditions(self, client: NetworkNode) -> NetworkConditions:
        """Generate realistic network conditions for a client."""
        if client.network_type == NetworkType.ETHERNET:
            return NetworkConditions(
                bandwidth_mbps=random.uniform(50, 1000),
                latency_ms=random.uniform(1, 10),
                packet_loss_rate=random.uniform(0.0001, 0.001),
                reliability=random.uniform(0.95, 0.99)
            )
        elif client.network_type == NetworkType.WIFI:
            return NetworkConditions(
                bandwidth_mbps=random.uniform(10, 100),
                latency_ms=random.uniform(5, 30),
                packet_loss_rate=random.uniform(0.001, 0.01),
                reliability=random.uniform(0.9, 0.98)
            )
        elif client.network_type == NetworkType.LTE:
            return NetworkConditions(
                bandwidth_mbps=random.uniform(5, 50),
                latency_ms=random.uniform(20, 100),
                packet_loss_rate=random.uniform(0.01, 0.05),
                reliability=random.uniform(0.85, 0.95),
                is_mobile=True,
                signal_strength=random.uniform(0.3, 1.0)
            )
        elif client.network_type == NetworkType.FIVE_G:
            return NetworkConditions(
                bandwidth_mbps=random.uniform(100, 1000),
                latency_ms=random.uniform(1, 20),
                packet_loss_rate=random.uniform(0.001, 0.01),
                reliability=random.uniform(0.9, 0.99),
                is_mobile=True,
                signal_strength=random.uniform(0.5, 1.0)
            )
        else:
            return NetworkConditions()
    
    def update_routing_table(self) -> None:
        """Update routing table using shortest path algorithm."""
        self.routing_table = {}
        
        for source in self.nodes:
            self.routing_table[source] = {}
            
            for target in self.nodes:
                if source != target:
                    try:
                        path = nx.shortest_path(self.graph, source, target, weight='latency_ms')
                        self.routing_table[source][target] = path
                    except nx.NetworkXNoPath:
                        self.routing_table[source][target] = []
    
    def get_route(self, source_id: str, target_id: str) -> List[str]:
        """Get route between two nodes."""
        if source_id in self.routing_table and target_id in self.routing_table[source_id]:
            return self.routing_table[source_id][target_id]
        return []
    
    def visualize_topology(self, save_path: Optional[str] = None) -> None:
        """Visualize the network topology."""
        plt.figure(figsize=(12, 8))
        
        # Create layout
        pos = nx.spring_layout(self.graph, k=1, iterations=50)
        
        # Draw nodes by type
        node_colors = {
            NodeType.CENTRAL_SERVER: 'red',
            NodeType.EDGE_SERVER: 'blue',
            NodeType.CLIENT: 'green',
            NodeType.AGGREGATOR: 'orange'
        }
        
        for node_type, color in node_colors.items():
            nodes_of_type = [node_id for node_id, node in self.nodes.items() 
                           if node.node_type == node_type]
            if nodes_of_type:
                nx.draw_networkx_nodes(self.graph, pos, nodelist=nodes_of_type,
                                     node_color=color, node_size=300, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, alpha=0.5)
        
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        
        plt.title("Federated Learning Network Topology")
        plt.legend(handles=[plt.Line2D([0], [0], marker='o', color='w', 
                                     markerfacecolor=color, markersize=10, label=node_type.value)
                          for node_type, color in node_colors.items()])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

class NetworkSimulator:
    """
    Advanced network simulator for federated learning environments.
    """
    
    def __init__(self, topology: NetworkTopology):
        self.topology = topology
        self.message_queue: List[NetworkMessage] = []
        self.transmission_log: List[Dict] = []
        self.network_stats: Dict[str, Any] = {
            'total_messages': 0,
            'total_bytes': 0,
            'failed_transmissions': 0,
            'average_latency': 0.0,
            'network_utilization': {}
        }
        self.simulation_time = 0.0
        self.time_step = 0.1  # seconds
        
    def send_message(self, message: NetworkMessage) -> bool:
        """
        Send a message through the network.
        
        Args:
            message (NetworkMessage): Message to send
            
        Returns:
            bool: True if message was successfully queued
        """
        # Validate sender and receiver
        if message.sender_id not in self.topology.nodes:
            logger.error(f"Sender {message.sender_id} not found in topology")
            return False
        
        if message.receiver_id not in self.topology.nodes:
            logger.error(f"Receiver {message.receiver_id} not found in topology")
            return False
        
        # Check if sender is active
        sender = self.topology.nodes[message.sender_id]
        if not sender.is_active:
            logger.warning(f"Sender {message.sender_id} is not active")
            return False
        
        # Add to message queue
        self.message_queue.append(message)
        logger.debug(f"Message {message.message_id} queued for transmission")
        return True
    
    def simulate_transmission(self, message: NetworkMessage) -> Dict[str, Any]:
        """
        Simulate message transmission through the network.
        
        Args:
            message (NetworkMessage): Message to transmit
            
        Returns:
            Dict[str, Any]: Transmission result
        """
        # Get route
        route = self.topology.get_route(message.sender_id, message.receiver_id)
        if not route:
            return {
                'success': False,
                'error': 'No route available',
                'latency': 0,
                'bandwidth_used': 0
            }
        
        # Calculate transmission metrics
        total_latency = 0.0
        total_bandwidth_used = 0.0
        transmission_successful = True
        
        # Simulate transmission along route
        for i in range(len(route) - 1):
            hop_from = route[i]
            hop_to = route[i + 1]
            
            # Get edge conditions
            edge_key = (hop_from, hop_to)
            if edge_key not in self.topology.edges:
                edge_key = (hop_to, hop_from)
            
            if edge_key not in self.topology.edges:
                transmission_successful = False
                break
            
            conditions = self.topology.edges[edge_key]['conditions']
            
            # Simulate hop transmission
            hop_result = self._simulate_hop_transmission(message, conditions)
            
            total_latency += hop_result['latency']
            total_bandwidth_used += hop_result['bandwidth_used']
            
            if not hop_result['success']:
                transmission_successful = False
                break
        
        # Apply compression and encryption overhead
        if message.compressed:
            total_latency *= 1.1  # Compression overhead
        if message.encrypted:
            total_latency *= 1.05  # Encryption overhead
        
        # Record transmission
        transmission_record = {
            'message_id': message.message_id,
            'sender_id': message.sender_id,
            'receiver_id': message.receiver_id,
            'route': route,
            'success': transmission_successful,
            'latency': total_latency,
            'bandwidth_used': total_bandwidth_used,
            'timestamp': self.simulation_time,
            'payload_size_mb': message.payload_size_mb
        }
        
        self.transmission_log.append(transmission_record)
        self._update_network_stats(transmission_record)
        
        return transmission_record
    
    def _simulate_hop_transmission(self, message: NetworkMessage, 
                                 conditions: NetworkConditions) -> Dict[str, Any]:
        """Simulate transmission over a single network hop."""
        # Calculate base transmission time
        transmission_time = (message.payload_size_mb * 8) / conditions.bandwidth_mbps  # seconds
        
        # Add latency
        base_latency = conditions.latency_ms / 1000.0  # Convert to seconds
        
        # Add jitter
        jitter = np.random.normal(0, conditions.jitter_ms / 1000.0)
        total_latency = base_latency + jitter + transmission_time
        
        # Simulate packet loss
        packet_loss_occurred = np.random.random() < conditions.packet_loss_rate
        
        # Simulate network congestion
        congestion_factor = 1.0 + conditions.congestion_level * np.random.random()
        total_latency *= congestion_factor
        
        # Simulate mobile network variability
        if conditions.is_mobile:
            signal_factor = conditions.signal_strength
            total_latency /= signal_factor
            
            # Mobile networks have higher variability
            mobile_variability = np.random.normal(1.0, 0.2)
            total_latency *= max(0.5, mobile_variability)
        
        # Check reliability
        transmission_successful = (np.random.random() < conditions.reliability and 
                                 not packet_loss_occurred)
        
        return {
            'success': transmission_successful,
            'latency': total_latency,
            'bandwidth_used': message.payload_size_mb,
            'congestion_factor': congestion_factor
        }
    
    def _update_network_stats(self, transmission_record: Dict) -> None:
        """Update network statistics."""
        self.network_stats['total_messages'] += 1
        self.network_stats['total_bytes'] += transmission_record['payload_size_mb'] * 1024 * 1024
        
        if not transmission_record['success']:
            self.network_stats['failed_transmissions'] += 1
        
        # Update average latency
        current_avg = self.network_stats['average_latency']
        total_msgs = self.network_stats['total_messages']
        new_latency = transmission_record['latency']
        
        self.network_stats['average_latency'] = (
            (current_avg * (total_msgs - 1) + new_latency) / total_msgs
        )
        
        # Update network utilization
        for node_id in transmission_record['route']:
            if node_id not in self.network_stats['network_utilization']:
                self.network_stats['network_utilization'][node_id] = 0
            self.network_stats['network_utilization'][node_id] += transmission_record['bandwidth_used']
    
    def simulate_network_dynamics(self, duration: float) -> None:
        """
        Simulate dynamic network conditions over time.
        
        Args:
            duration (float): Simulation duration in seconds
        """
        end_time = self.simulation_time + duration
        
        while self.simulation_time < end_time:
            # Update node conditions
            self._update_node_conditions()
            
            # Update network conditions
            self._update_network_conditions()
            
            # Process message queue
            self._process_message_queue()
            
            # Advance simulation time
            self.simulation_time += self.time_step
            
            # Log periodic statistics
            if int(self.simulation_time) % 60 == 0:  # Every minute
                logger.info(f"Simulation time: {self.simulation_time:.1f}s, "
                           f"Messages processed: {self.network_stats['total_messages']}")
    
    def _update_node_conditions(self) -> None:
        """Update dynamic node conditions."""
        for node in self.topology.nodes.values():
            if node.node_type == NodeType.CLIENT and node.is_mobile:
                # Simulate mobile device movement and energy consumption
                node.energy_level = max(0.0, node.energy_level - 0.001)
                
                # Device becomes inactive if energy is too low
                if node.energy_level < 0.1:
                    node.is_active = False
                
                # Simulate location updates for mobile devices
                if node.is_active:
                    dx = np.random.normal(0, 0.1)
                    dy = np.random.normal(0, 0.1)
                    node.location = (node.location[0] + dx, node.location[1] + dy)
    
    def _update_network_conditions(self) -> None:
        """Update dynamic network conditions."""
        for edge_key, edge_data in self.topology.edges.items():
            conditions = edge_data['conditions']
            
            # Simulate bandwidth fluctuations
            bandwidth_variation = np.random.normal(1.0, 0.1)
            conditions.bandwidth_mbps *= max(0.1, bandwidth_variation)
            
            # Simulate latency variations
            latency_variation = np.random.normal(1.0, 0.05)
            conditions.latency_ms *= max(0.5, latency_variation)
            
            # Simulate congestion changes
            congestion_change = np.random.normal(0, 0.01)
            conditions.congestion_level = np.clip(
                conditions.congestion_level + congestion_change, 0.0, 1.0
            )
            
            # Update timestamp
            edge_data['last_updated'] = self.simulation_time
    
    def _process_message_queue(self) -> None:
        """Process messages in the queue."""
        messages_to_process = [msg for msg in self.message_queue 
                             if msg.timestamp <= self.simulation_time]
        
        for message in messages_to_process:
            # Check if message has expired
            if message.deadline and self.simulation_time > message.deadline:
                logger.warning(f"Message {message.message_id} expired")
                self.message_queue.remove(message)
                continue
            
            # Simulate transmission
            result = self.simulate_transmission(message)
            
            if result['success']:
                logger.debug(f"Message {message.message_id} transmitted successfully")
            else:
                logger.warning(f"Message {message.message_id} transmission failed")
            
            # Remove from queue
            self.message_queue.remove(message)
    
    def get_network_statistics(self) -> Dict[str, Any]:
        """Get comprehensive network statistics."""
        if not self.transmission_log:
            return self.network_stats
        
        # Calculate additional statistics
        latencies = [record['latency'] for record in self.transmission_log 
                    if record['success']]
        
        if latencies:
            stats = {
                **self.network_stats,
                'latency_statistics': {
                    'min': min(latencies),
                    'max': max(latencies),
                    'mean': np.mean(latencies),
                    'median': np.median(latencies),
                    'std': np.std(latencies),
                    'percentile_95': np.percentile(latencies, 95),
                    'percentile_99': np.percentile(latencies, 99)
                },
                'success_rate': (self.network_stats['total_messages'] - 
                               self.network_stats['failed_transmissions']) / 
                               max(1, self.network_stats['total_messages']),
                'throughput_mbps': self.network_stats['total_bytes'] / 
                                 (1024 * 1024 * max(1, self.simulation_time)),
                'simulation_duration': self.simulation_time
            }
        else:
            stats = self.network_stats
        
        return stats
    
    def export_simulation_data(self, file_path: str) -> None:
        """Export simulation data for analysis."""
        simulation_data = {
            'topology': {
                'nodes': {node_id: {
                    'node_type': node.node_type.value,
                    'location': node.location,
                    'network_type': node.network_type.value,
                    'is_active': node.is_active
                } for node_id, node in self.topology.nodes.items()},
                'edges': {f"{edge[0]}-{edge[1]}": {
                    'bandwidth_mbps': data['conditions'].bandwidth_mbps,
                    'latency_ms': data['conditions'].latency_ms,
                    'reliability': data['conditions'].reliability
                } for edge, data in self.topology.edges.items()}
            },
            'transmission_log': self.transmission_log,
            'network_statistics': self.get_network_statistics()
        }
        
        with open(file_path, 'w') as f:
            json.dump(simulation_data, f, indent=2, default=str)
        
        logger.info(f"Simulation data exported to {file_path}")

# Factory functions
def create_network_simulator(num_clients: int = 100, num_edge_servers: int = 5) -> NetworkSimulator:
    """
    Create a network simulator with default hierarchical topology.
    
    Args:
        num_clients (int): Number of client nodes
        num_edge_servers (int): Number of edge servers
        
    Returns:
        NetworkSimulator: Configured network simulator
    """
    topology = NetworkTopology("hierarchical")
    topology.create_hierarchical_topology(num_clients, num_edge_servers)
    topology.update_routing_table()
    
    simulator = NetworkSimulator(topology)
    return simulator

def simulate_federated_learning_round(simulator: NetworkSimulator, 
                                    round_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simulate a complete federated learning round.
    
    Args:
        simulator (NetworkSimulator): Network simulator
        round_data (Dict[str, Any]): Round configuration data
        
    Returns:
        Dict[str, Any]: Round simulation results
    """
    round_start_time = simulator.simulation_time
    
    # Phase 1: Model distribution
    model_size_mb = round_data.get('model_size_mb', 10.0)
    clients = [node_id for node_id, node in simulator.topology.nodes.items() 
              if node.node_type == NodeType.CLIENT and node.is_active]
    
    distribution_messages = []
    for client_id in clients:
        message = NetworkMessage(
            message_id=f"model_dist_{client_id}_{round_start_time}",
            sender_id="central_server",
            receiver_id=client_id,
            message_type="model_distribution",
            payload_size_mb=model_size_mb,
            priority=8
        )
        distribution_messages.append(message)
        simulator.send_message(message)
    
    # Simulate distribution phase
    simulator.simulate_network_dynamics(30.0)  # 30 seconds for distribution
    
    # Phase 2: Local training (simulated delay)
    training_time = round_data.get('training_time_seconds', 60.0)
    simulator.simulate_network_dynamics(training_time)
    
    # Phase 3: Model aggregation
    update_size_mb = round_data.get('update_size_mb', 5.0)
    aggregation_messages = []
    
    for client_id in clients:
        # Only send updates from clients that received the model
        if any(msg.receiver_id == client_id and msg.message_type == "model_distribution" 
               for msg in distribution_messages):
            message = NetworkMessage(
                message_id=f"model_update_{client_id}_{simulator.simulation_time}",
                sender_id=client_id,
                receiver_id="central_server",
                message_type="model_update",
                payload_size_mb=update_size_mb,
                priority=9
            )
            aggregation_messages.append(message)
            simulator.send_message(message)
    
    # Simulate aggregation phase
    simulator.simulate_network_dynamics(30.0)  # 30 seconds for aggregation
    
    round_end_time = simulator.simulation_time
    
    # Calculate round statistics
    round_stats = {
        'round_duration': round_end_time - round_start_time,
        'participating_clients': len(clients),
        'successful_distributions': len([msg for msg in distribution_messages 
                                       if any(log['message_id'] == msg.message_id and log['success'] 
                                             for log in simulator.transmission_log)]),
        'successful_updates': len([msg for msg in aggregation_messages 
                                 if any(log['message_id'] == msg.message_id and log['success'] 
                                       for log in simulator.transmission_log)]),
        'total_data_transferred_mb': (len(distribution_messages) * model_size_mb + 
                                    len(aggregation_messages) * update_size_mb),
        'network_statistics': simulator.get_network_statistics()
    }
    
    return round_stats
