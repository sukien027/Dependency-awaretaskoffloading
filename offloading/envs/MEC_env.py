#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) su_kien. All Rights Reserved 
#
# @Time    : 04/08/2024 09:15
# @Author  : su_kien
# @Email   : sukien027@gmail.com
# @File    : MEC_env.py
# @IDE     : PyCharm

import numpy as np
from typing import List, Tuple, Dict, Any
import yaml
from task_graph import TaskDAG, TaskNode  # Import DAG modeling module


class VehicleTerminal:
    """Vehicle terminal with computing and communication capabilities"""
    
    def __init__(self, vehicle_id: int, config: Dict, channel_model):
        self.id = vehicle_id
        self.position = np.array(config['position'])
        self.velocity = np.array(config['velocity'])
        self.max_power = config['max_power']
        self.local_compute_cap = config['local_compute_cap']  # CPU GHz
        self.energy_budget = config['energy_budget']
        
        # Channel related
        self.channel_model = channel_model
        self.channel_state = None
        self.sinr = 0.0
        
        # Task queue and DAG
        self.task_queue: List[TaskDAG] = []
        self.current_dag: TaskDAG = None
        
        # State variables
        self.buffer_occupancy = 0.0
        self.remaining_energy = self.energy_budget
        
        # Physical constants
        self.k = 1e-27  # Energy coefficient
        self.time_slot = 0.01  # 10ms time slot
        
    def update_position(self, time_step: float = 0.1):
        """Update vehicle position based on velocity"""
        self.position += self.velocity * time_step
        
    def get_state(self) -> np.ndarray:
        """Get vehicle state vector"""
        if self.current_dag:
            dag_state = self.current_dag.get_state()
            ready_nodes = len(self.current_dag.get_ready_nodes())
        else:
            dag_state = np.zeros(10)  # Placeholder
            ready_nodes = 0
            
        state = np.array([
            self.buffer_occupancy,
            self.remaining_energy,
            self.sinr,
            len(self.task_queue),
            ready_nodes,
            np.linalg.norm(self.velocity),
            np.linalg.norm(self.channel_state) if self.channel_state is not None else 0.0
        ])
        
        return state
        
    def local_compute(self, node: TaskNode, power_ratio: float) -> Tuple[float, float]:
        """Execute task node locally"""
        allocated_power = power_ratio * self.max_power
        
        # Execution time based on CPU capability
        exec_time = node.compute_load / (self.local_compute_cap * allocated_power)
        
        # Energy consumption
        energy = allocated_power * exec_time
        
        self.remaining_energy -= energy
        return exec_time, energy
        
    def offload_compute(self, node: TaskNode, server_id: int) -> Tuple[float, float]:
        """Offload task node to MEC server"""
        # Transmission delay based on channel state
        tx_rate = np.log2(1 + self.sinr) if self.sinr > 0 else 0.001
        tx_time = node.data_size / tx_rate
        
        # Transmission energy
        tx_energy = self.max_power * 0.5 * tx_time  # Assume 50% power for transmission
        
        # Remote execution time (estimated)
        exec_time = node.compute_load / 10e9  # Assume 10GHz server
        
        total_time = tx_time + exec_time
        total_energy = tx_energy
        
        return total_time, total_energy
        
    def reset(self):
        """Reset vehicle state"""
        self.buffer_occupancy = 0.0
        self.remaining_energy = self.energy_budget
        self.task_queue.clear()
        self.current_dag = None


class MECServer:
    """MEC server with computing resources"""
    
    def __init__(self, server_id: int, config: Dict):
        self.id = server_id
        self.position = np.array(config['position'])
        self.compute_cap = config['compute_cap']  # Total CPU GHz
        self.bandwidth = config['bandwidth']
        self.num_antennas = config['num_antennas']
        
        # Task queue
        self.task_queue: List[Tuple[int, TaskNode]] = []  # (vehicle_id, task_node)
        
    def process_task(self, node: TaskNode, compute_share: float) -> float:
        """Process task node on server"""
        allocated_cap = compute_share * self.compute_cap
        exec_time = node.compute_load / allocated_cap
        return exec_time


class ChannelModel:
    """MIMO channel model for V2I communications"""
    
    def __init__(self, config: Dict):
        self.path_loss_exp = config['path_loss_exp']
        self.shadowing_std = config['shadowing_std']
        self.noise_power = config['noise_power']
        self.fast_fading = config.get('fast_fading', True)
        self.seed = config.get('seed', 42)
        
        np.random.seed(self.seed)
        
    def get_channel(self, distance: float, num_antennas: int = 8) -> np.ndarray:
        """Generate MIMO channel vector"""
        # Path loss
        pl = 10 ** (-self.path_loss_exp * np.log10(distance) / 10)
        
        # Shadowing
        shadowing = 10 ** (np.random.normal(0, self.shadowing_std) / 10)
        
        # Fast fading (Rayleigh)
        if self.fast_fading:
            fading = (np.random.randn(num_antennas, 2) * 
                     np.sqrt(0.5)).view(np.complex128).flatten()
        else:
            fading = np.ones(num_antennas, dtype=np.complex128)
            
        # Combined channel
        channel = np.sqrt(pl * shadowing) * fading
        return channel
        
    def calculate_sinr(self, channels: np.ndarray, powers: np.ndarray) -> np.ndarray:
        """Calculate SINR using Zero-Forcing beamforming"""
        if len(channels.shape) == 2:
            # Multiple vehicles
            H = channels.T  # Shape: (num_antennas, num_vehicles)
            H_pinv = np.linalg.pinv(H)
            effective_noise = np.power(np.linalg.norm(H_pinv, axis=1), 2) * self.noise_power
            effective_noise = np.maximum(effective_noise, 1e-12)
            sinr = powers / effective_noise
        else:
            # Single vehicle
            channel_norm = np.linalg.norm(channels)
            sinr = powers * channel_norm**2 / self.noise_power
            
        return sinr


class MECEnvironment:
    """Main MEC environment for DAG-based task offloading"""
    
    def __init__(self, config_path: str = 'configs/env.yaml'):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Environment parameters
        self.num_vehicles = self.config['num_vehicles']
        self.num_servers = self.config['num_servers']
        self.max_time_slots = self.config['max_time_slots']
        
        # Reward weights
        reward_config = self.config.get('reward', {})
        self.lambda_d = reward_config.get('lambda_d', 0.5)  # Delay weight
        self.lambda_e = reward_config.get('lambda_e', 0.5)  # Energy weight
        
        # Initialize components
        self.channel_model = ChannelModel(self.config['channel'])
        self._init_vehicles()
        self._init_servers()
        
        # State tracking
        self.current_time_slot = 0
        self.total_delay = 0.0
        self.total_energy = 0.0
        
    def _init_vehicles(self):
        """Initialize vehicles"""
        self.vehicles: List[VehicleTerminal] = []
        vehicle_config = self.config['vehicle']
        
        for i in range(self.num_vehicles):
            config = {
                'position': np.random.uniform(0, 1000, 2),
                'velocity': np.random.uniform(-10, 10, 2),
                'max_power': vehicle_config['max_power'],
                'local_compute_cap': vehicle_config['local_compute_cap'],
                'energy_budget': vehicle_config['energy_budget']
            }
            
            vehicle = VehicleTerminal(i, config, self.channel_model)
            self.vehicles.append(vehicle)
            
    def _init_servers(self):
        """Initialize MEC servers"""
        self.servers: List[MECServer] = []
        server_config = self.config['server']
        
        for i in range(self.num_servers):
            config = {
                'position': np.array([500.0, 500.0]),  # Fixed position
                'compute_cap': server_config['compute_cap'],
                'bandwidth': server_config['bandwidth'],
                'num_antennas': server_config['num_antennas']
            }
            
            server = MECServer(i, config)
            self.servers.append(server)
            
    def reset(self, seed: int = None) -> Dict[str, np.ndarray]:
        """Reset environment and return initial state"""
        if seed is not None:
            np.random.seed(seed)
            
        self.current_time_slot = 0
        self.total_delay = 0.0
        self.total_energy = 0.0
        
        # Reset vehicles
        for vehicle in self.vehicles:
            vehicle.reset()
            
            # Generate initial DAG task
            from task_graph import generate_random_dag  # Import here to avoid circular imports
            vehicle.current_dag = generate_random_dag(max_nodes=10)
            
        # Get initial observation
        observation = self._get_observation()
        return observation
        
    def step(self, action: Dict[str, np.ndarray]) -> Tuple[Dict, float, bool, Dict]:
        """Execute one time slot"""
        self.current_time_slot += 1
        
        # Parse actions
        exec_locations = action['execution_location']  # Shape: (num_vehicles, max_nodes)
        power_allocations = action['power_allocation']  # Shape: (num_vehicles, max_nodes)
        
        # Update vehicle positions and channels
        self._update_channels()
        
        # Calculate SINR
        sinr_list = self._calculate_sinr(power_allocations)
        
        # Process tasks based on decisions
        step_delay = 0.0
        step_energy = 0.0
        
        for i, vehicle in enumerate(self.vehicles):
            vehicle.sinr = sinr_list[i] if i < len(sinr_list) else 0.0
            
            if vehicle.current_dag:
                ready_nodes = vehicle.current_dag.get_ready_nodes()
                
                for node_id in ready_nodes:
                    if node_id < exec_locations.shape[1]:
                        exec_loc = exec_locations[i, node_id]
                        power_alloc = power_allocations[i, node_id]
                        
                        node = vehicle.current_dag.nodes[node_id]
                        
                        if exec_loc < 0.5:  # Local execution
                            delay, energy = vehicle.local_compute(node, power_alloc)
                        else:  # Remote execution
                            delay, energy = vehicle.offload_compute(node, 0)
                            # Add to server queue
                            self.servers[0].task_queue.append((i, node))
                            
                        step_delay += delay
                        step_energy += energy
                        
                        # Mark node as completed
                        node.status = 'completed'
                        
        # Process server tasks
        server_delay = 0.0
        for vehicle_id, node in self.servers[0].task_queue:
            delay = self.servers[0].process_task(node, 1.0/len(self.servers[0].task_queue))
            server_delay += delay
            
        step_delay += server_delay
        self.servers[0].task_queue.clear()
        
        # Update totals
        self.total_delay += step_delay
        self.total_energy += step_energy
        
        # Calculate reward
        reward = self._calculate_reward(step_delay, step_energy)
        
        # Check termination
        done = self.current_time_slot >= self.max_time_slots
        
        # Get next observation
        next_observation = self._get_observation()
        info = self._get_info()
        
        return next_observation, reward, done, info
        
    def _update_channels(self):
        """Update channel states for all vehicles"""
        for vehicle in self.vehicles:
            distance = np.linalg.norm(vehicle.position - self.servers[0].position)
            vehicle.channel_state = self.channel_model.get_channel(
                distance, self.servers[0].num_antennas
            )
            
    def _calculate_sinr(self, power_allocations: np.ndarray) -> np.ndarray:
        """Calculate SINR for all vehicles"""
        channels = []
        powers = []
        
        for i, vehicle in enumerate(self.vehicles):
            channels.append(vehicle.channel_state)
            # Average power for this vehicle
            avg_power = np.mean(power_allocations[i]) * vehicle.max_power
            powers.append(avg_power)
            
        channels = np.array(channels)
        powers = np.array(powers)
        
        sinr = self.channel_model.calculate_sinr(channels, powers)
        return sinr
        
    def _calculate_reward(self, delay: float, energy: float) -> float:
        """Calculate reward based on delay and energy"""
        # Normalize and weight
        norm_delay = delay / (0.1 * self.num_vehicles)  # Normalize by time slot
        norm_energy = energy / (self.num_vehicles * 10)  # Normalize
        
        reward = - (self.lambda_d * norm_delay + self.lambda_e * norm_energy)
        return reward
        
    def _get_observation(self) -> Dict[str, np.ndarray]:
        """Get current observation for all vehicles"""
        vehicle_states = []
        dag_features = []
        dag_adjacency = []
        channel_states = []
        
        for vehicle in self.vehicles:
            # Vehicle state
            state = vehicle.get_state()
            vehicle_states.append(state)
            
            # DAG features
            if vehicle.current_dag:
                features, adjacency = vehicle.current_dag.get_graph_representation()
                dag_features.append(features)
                dag_adjacency.append(adjacency)
            else:
                # Placeholder for empty DAG
                dag_features.append(np.zeros((10, 3)))  # max_nodes x features
                dag_adjacency.append(np.zeros((10, 10)))
                
            # Channel state (real and imaginary parts)
            if vehicle.channel_state is not None:
                channel_real_imag = np.stack(
                    [vehicle.channel_state.real, vehicle.channel_state.imag], axis=1
                )
            else:
                channel_real_imag = np.zeros((8, 2))  # 8 antennas
            channel_states.append(channel_real_imag)
            
        return {
            'vehicle_states': np.array(vehicle_states, dtype=np.float32),
            'dag_features': np.array(dag_features, dtype=np.float32),
            'dag_adjacency': np.array(dag_adjacency, dtype=np.float32),
            'channel_states': np.array(channel_states, dtype=np.float32)
        }
        
    def _get_info(self) -> Dict[str, Any]:
        """Get additional environment information"""
        completed_tasks = sum(
            1 for v in self.vehicles 
            if v.current_dag and all(n.status == 'completed' for n in v.current_dag.nodes.values())
        )
        
        avg_buffer = np.mean([v.buffer_occupancy for v in self.vehicles])
        avg_energy = np.mean([v.remaining_energy for v in self.vehicles])
        
        return {
            'time_slot': self.current_time_slot,
            'total_delay': self.total_delay,
            'total_energy': self.total_energy,
            'completed_tasks': completed_tasks,
            'avg_buffer': avg_buffer,
            'avg_energy': avg_energy
        }
        
    def render(self):
        """Render current environment state"""
        print(f"Time Slot: {self.current_time_slot}")
        print(f"Total Delay: {self.total_delay:.2f}s")
        print(f"Total Energy: {self.total_energy:.2f}J")
        print(f"Completed Tasks: {sum(1 for v in self.vehicles if not v.task_queue)}")
        print("-" * 50)


if __name__ == "__main__":
    env = MECEnvironment('configs/env.yaml')
    obs = env.reset()
    
    action = {
        'execution_location': np.random.rand(env.num_vehicles, 10),
        'power_allocation': np.random.rand(env.num_vehicles, 10)
    }
    
    next_obs, reward, done, info = env.step(action)
    print(f"Reward: {reward:.4f}")
    print(f"Done: {done}")
