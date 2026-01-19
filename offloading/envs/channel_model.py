#!/usr/bin/python3
# _*_ coding: utf-8 _*_
#
# Copyright (C) su_kien. All Rights Reserved 
#
# @Time    : 01/08/2024 14:04
# @Author  : su_kien
# @Email   : sukien027@gmail.com
# @File    : channel_model.py
# @IDE     : PyCharm
import numpy as np
from typing import Tuple, Optional, Union, List, Dict
import warnings
from scipy.special import j0  


class MarkovChannelModel:
    """
    Finite-state Markov channel model for time-varying wireless channels
    
    This model simulates channel states using a discrete Markov chain with 
    11 states representing different channel gain levels.
    
    Attributes:
        distance (float): Distance between user and base station (meters)
        path_loss_exponent (float): Path loss exponent (default: 3.0 for urban)
        reference_loss (float): Path loss at reference distance
        state (int): Current channel state (0-10)
        state_history (list): History of visited states
        gain_history (list): History of channel gains
    """
    
    # Markov chain parameters from empirical measurements
    STATE_AVERAGES = np.array([0.031, 0.153, 0.399, 0.772, 1.274, 1.911, 2.694, 
                               3.630, 4.730, 6.021, 7.902])
    
    TRANSITION_PROBABILITIES = np.array([
        [0.514, 0.514, 1.0],
        [0.513, 0.696, 1.0],
        [0.513, 0.745, 1.0],
        [0.515, 0.776, 1.0],
        [0.513, 0.799, 1.0],
        [0.514, 0.821, 1.0],
        [0.516, 0.842, 1.0],
        [0.511, 0.858, 1.0],
        [0.516, 0.880, 1.0],
        [0.512, 0.897, 1.0],
        [0.671, 1.000, 1.0]
    ])
    
    def __init__(self, 
                 distance: float,
                 path_loss_exponent: float = 3.0,
                 reference_loss: float = 0.001,
                 seed: Optional[int] = None) -> None:
        """
        Initialize Markov channel model
        
        Args:
            distance: Distance between user and base station (meters)
            path_loss_exponent: Path loss exponent (default: 3.0 for urban environment)
            reference_loss: Path loss at reference distance (1 meter)
            seed: Random seed for reproducibility
        """
        self.distance = distance
        self.path_loss_exponent = path_loss_exponent
        self.reference_loss = reference_loss
        
        # Calculate path loss using log-distance model
        self.path_loss = self.reference_loss * np.power(1.0 / distance, self.path_loss_exponent)
        
        # Set random seed for reproducibility
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state randomly (uniform distribution over states)
        self.state = np.random.randint(0, len(self.STATE_AVERAGES))
        self.num_states = len(self.STATE_AVERAGES)
        
        # History tracking for analysis
        self.state_history = [self.state]
        self.gain_history = [self._calculate_current_gain()]
    
    def _calculate_current_gain(self) -> float:
        """
        Calculate channel gain for current state
        
        Returns:
            Current channel gain including path loss
        """
        if not (0 <= self.state < self.num_states):
            raise ValueError(f"Invalid state: {self.state}. Must be between 0 and {self.num_states-1}")
        
        # Channel gain = sqrt(path_loss × state_gain)
        return np.sqrt(self.path_loss * self.STATE_AVERAGES[self.state])
    
    def get_channel_gain(self) -> float:
        """
        Get current channel gain
        
        Returns:
            Current channel gain
        """
        return self._calculate_current_gain()
    
    def sample_next_state(self) -> float:
        """
        Sample next channel state according to transition probabilities
        
        Returns:
            New channel gain after state transition
        """
        p_down, p_up, _ = self.TRANSITION_PROBABILITIES[self.state]
        
        rand_val = np.random.rand()
        
        if rand_val >= p_up:
            self.state = min(self.state + 1, self.num_states - 1)
        elif rand_val <= p_down:
            self.state = max(self.state - 1, 0)

        
        self.state_history.append(self.state)
        current_gain = self._calculate_current_gain()
        self.gain_history.append(current_gain)
        
        return current_gain
    
    def get_state_statistics(self) -> Dict:
        """
        Calculate statistics of visited states
        
        Returns:
            Dictionary containing state distribution statistics
        """
        if len(self.state_history) < 2:
            return {'warning': 'Insufficient state history'}
        
        unique_states, counts = np.unique(self.state_history, return_counts=True)
        state_probabilities = counts / len(self.state_history)
        
        return {
            'state_distribution': dict(zip(unique_states.astype(int), state_probabilities)),
            'stationary_distribution': self._calculate_stationary_distribution(),
            'average_gain': np.mean(self.gain_history),
            'gain_variance': np.var(self.gain_history),
            'gain_std': np.std(self.gain_history),
            'state_entropy': self._calculate_state_entropy(state_probabilities)
        }
    
    def _calculate_stationary_distribution(self) -> np.ndarray:
        """
        Calculate stationary distribution of Markov chain
        
        Returns:
            Stationary probability distribution vector
        """
        # Construct full transition probability matrix
        P = np.zeros((self.num_states, self.num_states))
        
        for i in range(self.num_states):
            p_down, p_up, _ = self.TRANSITION_PROBABILITIES[i]
            p_stay = 1.0  
            
            if i > 0:
                P[i, i-1] = p_down
            if i < self.num_states - 1:
                P[i, i+1] = 1.0 - p_up

            P[i, i] = 1.0 - np.sum(P[i, :])
        
        eigenvalues, eigenvectors = np.linalg.eig(P.T)
        
        stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary_dist = np.real(eigenvectors[:, stationary_idx])

        stationary_dist = np.abs(stationary_dist)
        stationary_dist = stationary_dist / np.sum(stationary_dist)
        
        return stationary_dist
    
    def _calculate_state_entropy(self, probabilities: np.ndarray) -> float:
        """
        Calculate entropy of state distribution
        
        Args:
            probabilities: State probability distribution
            
        Returns:
            Entropy value in bits
        """
        # Remove zero probabilities for log calculation
        non_zero_probs = probabilities[probabilities > 0]
        return -np.sum(non_zero_probs * np.log2(non_zero_probs))


class ARChannelModel:
    """
    Autoregressive channel model for time-correlated Rayleigh fading
    
    Implements AR(1) model: h[t] = ρ·h[t-1] + √(1-ρ²)·w[t]
    where w[t] ~ CN(0,1) and ρ is the temporal correlation coefficient
    
    Attributes:
        distance (float): Distance between transmitter and receiver
        n_tx (int): Number of transmit antennas
        n_rx (int): Number of receive antennas
        rho (float): Temporal correlation coefficient
        channel_matrix (np.ndarray): Current channel matrix
    """
    
    def __init__(self,
                 distance: float,
                 num_tx_antennas: int = 1,
                 num_rx_antennas: int = 1,
                 correlation_coefficient: float = 0.95,
                 doppler_frequency: Optional[float] = None,
                 sampling_period: float = 1e-3,
                 seed: Optional[int] = None) -> None:
        """
        Initialize AR channel model
        
        Args:
            distance: Distance between user and base station (meters)
            num_tx_antennas: Number of transmit antennas
            num_rx_antennas: Number of receive antennas
            correlation_coefficient: AR coefficient ρ (0 ≤ ρ ≤ 1)
            doppler_frequency: Doppler frequency (Hz) for Jakes model
            sampling_period: Time between samples (seconds)
            seed: Random seed for reproducibility
        """
        self.distance = distance
        self.n_tx = num_tx_antennas
        self.n_rx = num_rx_antennas
        
        self.path_loss_exponent = 3.0
        self.reference_loss = 0.001
        self.path_loss = self.reference_loss * np.power(1.0 / distance, self.path_loss_exponent)
        
        if seed is not None:
            np.random.seed(seed)
        
        if doppler_frequency is not None:
            self.rho = j0(2 * np.pi * doppler_frequency * sampling_period)
        else:
            self.rho = correlation_coefficient
        
        # Initialize channel matrix with complex Gaussian
        self.channel_matrix = self._generate_complex_gaussian(self.n_tx, self.n_rx)
        
        # History tracking
        self.channel_history = [self.channel_matrix.copy()]
    
    def _generate_complex_gaussian(self, 
                                   rows: int, 
                                   cols: int, 
                                   variance: float = 1.0) -> np.ndarray:
        """
        Generate complex Gaussian random matrix
        
        Args:
            rows: Number of rows
            cols: Number of columns
            variance: Variance of real/imaginary parts
            
        Returns:
            Complex Gaussian matrix CN(0, variance·I)
        """
        real_part = np.random.normal(0, np.sqrt(variance/2), size=(rows, cols))
        imag_part = np.random.normal(0, np.sqrt(variance/2), size=(rows, cols))
        return real_part + 1j * imag_part
    
    def get_channel_matrix(self, include_path_loss: bool = True) -> np.ndarray:
        """
        Get current channel matrix
        
        Args:
            include_path_loss: Whether to include path loss scaling
            
        Returns:
            Channel matrix H
        """
        if include_path_loss:
            return self.channel_matrix * np.sqrt(self.path_loss)
        else:
            return self.channel_matrix
    
    def sample_next_channel(self) -> np.ndarray:
        """
        Generate next channel matrix according to AR(1) model
        
        Returns:
            New channel matrix including path loss
        """
        # Generate innovation term
        innovation = self._generate_complex_gaussian(self.n_tx, self.n_rx, 1.0)
        
        # AR(1) update: H[t] = ρ·H[t-1] + √(1-ρ²)·W[t]
        self.channel_matrix = (self.rho * self.channel_matrix + 
                              np.sqrt(1 - self.rho**2) * innovation)
        
        # Record history
        self.channel_history.append(self.channel_matrix.copy())
        
        return self.get_channel_matrix()
    
    def get_channel_statistics(self) -> Dict:
        """
        Calculate channel statistics
        
        Returns:
            Dictionary containing channel statistics
        """
        if len(self.channel_history) < 2:
            return {'warning': 'Insufficient channel history'}
        
        # Flatten channel matrices for statistics
        channel_magnitudes = np.abs(np.array(self.channel_history))
        magnitudes_flat = channel_magnitudes.flatten()
        
        return {
            'mean_magnitude': np.mean(magnitudes_flat),
            'magnitude_variance': np.var(magnitudes_flat),
            'mean_power': np.mean(magnitudes_flat**2),
            'temporal_correlation': self._estimate_temporal_correlation(),
            'fading_distribution': self._test_rayleigh_distribution(magnitudes_flat)
        }
    
    def _estimate_temporal_correlation(self) -> float:
        """
        Estimate temporal correlation from channel history
        
        Returns:
            Estimated correlation coefficient
        """
        if len(self.channel_history) < 2:
            return 0.0
        
        # Convert to array for easier processing
        H_array = np.array(self.channel_history)
        
        # Calculate correlation between consecutive time steps
        correlations = []
        for i in range(H_array.shape[0] - 1):
            # Flatten matrices and calculate correlation
            h_current = H_array[i].flatten()
            h_next = H_array[i+1].flatten()
            
            # Complex correlation coefficient
            corr = np.abs(np.corrcoef(h_current.real, h_next.real)[0, 1])
            correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _test_rayleigh_distribution(self, magnitudes: np.ndarray) -> Dict:
        """
        Test if magnitudes follow Rayleigh distribution
        
        Args:
            magnitudes: Array of channel magnitudes
            
        Returns:
            Dictionary with distribution test results
        """
        from scipy.stats import rayleigh, kstest
        
        # Fit Rayleigh distribution to data
        scale_param = np.sqrt(2 / np.pi) * np.mean(magnitudes)
        
        # Kolmogorov-Smirnov test
        ks_statistic, p_value = kstest(magnitudes, 'rayleigh', args=(scale_param,))
        
        return {
            'rayleigh_scale': scale_param,
            'ks_statistic': ks_statistic,
            'p_value': p_value,
            'is_rayleigh': p_value > 0.05  # 5% significance level
        }


class RayleighFadingChannel:
    """
    Rayleigh fading channel model for multipath propagation environments
    
    Rayleigh fading assumes no line-of-sight component, appropriate for
    urban environments with rich scattering.
    
    Attributes:
        distance (float): Distance between transmitter and receiver
        n_tx (int): Number of transmit antennas
        n_rx (int): Number of receive antennas
        k_factor (float): Rician K-factor (0 for pure Rayleigh)
        path_loss (float): Calculated path loss
    """
    
    def __init__(self,
                 distance: float,
                 num_tx_antennas: int = 1,
                 num_rx_antennas: int = 1,
                 k_factor: float = 0.0,  # K=0 for pure Rayleigh
                 path_loss_exponent: float = 3.0,
                 reference_loss: float = 0.001,
                 seed: Optional[int] = None) -> None:
        """
        Initialize Rayleigh fading channel
        
        Args:
            distance: Distance between transmitter and receiver (meters)
            num_tx_antennas: Number of transmit antennas
            num_rx_antennas: Number of receive antennas
            k_factor: Rician K-factor (0 for pure Rayleigh fading)
            path_loss_exponent: Path loss exponent
            reference_loss: Path loss at reference distance
            seed: Random seed for reproducibility
        """
        self.distance = distance
        self.n_tx = num_tx_antennas
        self.n_rx = num_rx_antennas
        self.k_factor = k_factor
        
        # Path loss calculation
        self.path_loss_exponent = path_loss_exponent
        self.reference_loss = reference_loss
        self.path_loss = self.reference_loss * np.power(1.0 / distance, self.path_loss_exponent)
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_channel_matrix(self,
                                include_path_loss: bool = True,
                                normalized: bool = False) -> np.ndarray:
        """
        Generate Rayleigh fading channel matrix
        
        Args:
            include_path_loss: Whether to include path loss in channel gain
            normalized: Whether to normalize channel matrix (unit average power)
            
        Returns:
            Rayleigh fading channel matrix
        """
        # Generate complex Gaussian matrix for Rayleigh fading
        # Real and imaginary parts each have variance 0.5
        real_part = np.random.normal(0, np.sqrt(0.5), size=(self.n_tx, self.n_rx))
        imag_part = np.random.normal(0, np.sqrt(0.5), size=(self.n_tx, self.n_rx))
        H_rayleigh = real_part + 1j * imag_part
        
        # Add line-of-sight component if K > 0 (Rician fading)
        if self.k_factor > 0:
            # LOS component (unit amplitude, random phase)
            phase = np.random.uniform(0, 2*np.pi, size=(self.n_tx, self.n_rx))
            H_los = np.exp(1j * phase)
            
            # Combine LOS and NLOS components according to K-factor
            H = (np.sqrt(self.k_factor/(self.k_factor + 1)) * H_los +
                 np.sqrt(1/(self.k_factor + 1)) * H_rayleigh)
        else:
            H = H_rayleigh
        
        # Normalize if requested (unit average power)
        if normalized:
            H = H / np.sqrt(np.mean(np.abs(H)**2))
        
        # Apply path loss
        if include_path_loss:
            H = H * np.sqrt(self.path_loss)
        
        return H
    
    def calculate_channel_capacity(self,
                                   snr_db: float,
                                   channel_matrix: Optional[np.ndarray] = None,
                                   power_allocation: str = 'equal') -> float:
        """
        Calculate Shannon capacity of fading channel
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            channel_matrix: Channel matrix (generated if None)
            power_allocation: Power allocation strategy ('equal' or 'waterfilling')
            
        Returns:
            Channel capacity in bits per second per Hz (bps/Hz)
        """
        if channel_matrix is None:
            channel_matrix = self.generate_channel_matrix(include_path_loss=False)
        
        # Convert SNR from dB to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        n_tx, n_rx = channel_matrix.shape
        
        if power_allocation == 'equal':
            # Equal power allocation across antennas
            if n_rx >= n_tx:
                # Use singular value decomposition
                _, singular_values, _ = np.linalg.svd(channel_matrix)
                capacity = np.sum(np.log2(1 + snr_linear * (singular_values**2) / n_tx))
            else:
                # For non-square matrices
                H_H = channel_matrix.conj().T
                eigenvalues = np.linalg.eigvals(channel_matrix @ H_H)
                capacity = np.sum(np.log2(1 + snr_linear * eigenvalues / n_tx))
        
        elif power_allocation == 'waterfilling':
            # Water-filling power allocation
            _, singular_values, _ = np.linalg.svd(channel_matrix)
            num_streams = len(singular_values)
            allocated_power = np.zeros(num_streams)
            
            # Water-filling algorithm
            mu = 1.0  # Water level
            for _ in range(100):  # Maximum iterations
                allocated_power = np.maximum(0, mu - 1/(singular_values**2 + 1e-10))
                total_power = np.sum(allocated_power)
                
                if np.abs(total_power - snr_linear) < 1e-6:
                    break
                
                mu += (snr_linear - total_power) / num_streams
            
            capacity = np.sum(np.log2(1 + allocated_power * singular_values**2))
        
        else:
            raise ValueError(f"Unknown power allocation strategy: {power_allocation}")
        
        return float(np.real(capacity))
    
    def get_fading_statistics(self, num_samples: int = 1000) -> Dict:
        """
        Generate fading statistics
        
        Args:
            num_samples: Number of channel samples for statistics
            
        Returns:
            Dictionary with fading statistics
        """
        # Generate multiple channel realizations
        channel_magnitudes = []
        for _ in range(num_samples):
            H = self.generate_channel_matrix(include_path_loss=False)
            magnitudes = np.abs(H).flatten()
            channel_magnitudes.extend(magnitudes)
        
        channel_magnitudes = np.array(channel_magnitudes)
        
        # Calculate statistics
        from scipy.stats import rayleigh
        
        # Fit Rayleigh distribution
        scale_param = np.sqrt(2 / np.pi) * np.mean(channel_magnitudes)
        rayleigh_dist = rayleigh(scale=scale_param)
        
        return {
            'mean_magnitude': np.mean(channel_magnitudes),
            'magnitude_variance': np.var(channel_magnitudes),
            'mean_power': np.mean(channel_magnitudes**2),
            'rayleigh_scale': scale_param,
            'theoretical_mean': scale_param * np.sqrt(np.pi / 2),
            'theoretical_variance': (4 - np.pi) / 2 * scale_param**2,
            'outage_probability': self._calculate_outage_probability(channel_magnitudes)
        }
    
    def _calculate_outage_probability(self, magnitudes: np.ndarray, 
                                      threshold: float = 0.1) -> float:
        """
        Calculate outage probability (P(|h| < threshold))
        
        Args:
            magnitudes: Array of channel magnitudes
            threshold: Outage threshold
            
        Returns:
            Outage probability
        """
        return np.sum(magnitudes < threshold) / len(magnitudes)


class MIMOChannelModel:
    """
    MIMO channel model with spatial correlation
    
    Supports Rayleigh, Rician, and LoS channel types with antenna correlation
    using Kronecker model: H = R_rx^(1/2) · H_iid · R_tx^(1/2)^T
    
    Attributes:
        n_tx (int): Number of transmit antennas
        n_rx (int): Number of receive antennas
        channel_type (str): Type of channel ('rayleigh', 'rician', or 'los')
        correlation_tx (np.ndarray): Transmit antenna correlation matrix
        correlation_rx (np.ndarray): Receive antenna correlation matrix
    """
    
    def __init__(self,
                 num_tx_antennas: int,
                 num_rx_antennas: int,
                 channel_type: str = 'rayleigh',
                 correlation_tx: Optional[np.ndarray] = None,
                 correlation_rx: Optional[np.ndarray] = None,
                 seed: Optional[int] = None) -> None:
        """
        Initialize MIMO channel model
        
        Args:
            num_tx_antennas: Number of transmit antennas
            num_rx_antennas: Number of receive antennas
            channel_type: Type of channel ('rayleigh', 'rician', or 'los')
            correlation_tx: Transmit antenna correlation matrix
            correlation_rx: Receive antenna correlation matrix
            seed: Random seed for reproducibility
        """
        self.n_tx = num_tx_antennas
        self.n_rx = num_rx_antennas
        self.channel_type = channel_type.lower()
        
        # Validate channel type
        valid_types = ['rayleigh', 'rician', 'los']
        if self.channel_type not in valid_types:
            warnings.warn(f"Invalid channel type '{channel_type}'. Using 'rayleigh'.")
            self.channel_type = 'rayleigh'
        
        # Set antenna correlation matrices (default: uncorrelated)
        if correlation_tx is None:
            self.correlation_tx = np.eye(num_tx_antennas)
        else:
            self.correlation_tx = correlation_tx
        
        if correlation_rx is None:
            self.correlation_rx = np.eye(num_rx_antennas)
        else:
            self.correlation_rx = correlation_rx
        
        # Validate correlation matrix dimensions
        if self.correlation_tx.shape != (num_tx_antennas, num_tx_antennas):
            raise ValueError(f"TX correlation matrix must be {num_tx_antennas}x{num_tx_antennas}")
        if self.correlation_rx.shape != (num_rx_antennas, num_rx_antennas):
            raise ValueError(f"RX correlation matrix must be {num_rx_antennas}x{num_rx_antennas}")
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
    
    def generate_channel_matrix(self,
                                k_factor: float = 0.0,
                                include_correlation: bool = True) -> np.ndarray:
        """
        Generate MIMO channel matrix
        
        Args:
            k_factor: Rician K-factor (for 'rician' channel type)
            include_correlation: Whether to include antenna correlation
            
        Returns:
            MIMO channel matrix H of shape (n_rx, n_tx)
        """
        # Generate IID channel matrix based on channel type
        if self.channel_type == 'los':
            # Line-of-sight channel (deterministic)
            H_iid = np.ones((self.n_rx, self.n_tx), dtype=complex)
        elif self.channel_type == 'rician':
            # Rician fading: combination of LOS and Rayleigh components
            # Rayleigh component
            real_part = np.random.normal(0, np.sqrt(0.5), size=(self.n_rx, self.n_tx))
            imag_part = np.random.normal(0, np.sqrt(0.5), size=(self.n_rx, self.n_tx))
            H_rayleigh = real_part + 1j * imag_part
            
            # LOS component (unit amplitude, random phase)
            phase = np.random.uniform(0, 2*np.pi, size=(self.n_rx, self.n_tx))
            H_los = np.exp(1j * phase)
            
            # Combine according to K-factor
            H_iid = (np.sqrt(k_factor/(k_factor + 1)) * H_los +
                     np.sqrt(1/(k_factor + 1)) * H_rayleigh)
        else:  # 'rayleigh'
            # Pure Rayleigh fading
            real_part = np.random.normal(0, np.sqrt(0.5), size=(self.n_rx, self.n_tx))
            imag_part = np.random.normal(0, np.sqrt(0.5), size=(self.n_rx, self.n_tx))
            H_iid = real_part + 1j * imag_part
        
        # Apply antenna correlation using Kronecker model
        if include_correlation:
            # Cholesky decomposition of correlation matrices
            try:
                R_tx_sqrt = np.linalg.cholesky(self.correlation_tx)
                R_rx_sqrt = np.linalg.cholesky(self.correlation_rx)
            except np.linalg.LinAlgError:
                # If not positive definite, use eigenvalue decomposition
                eigvals_tx, eigvecs_tx = np.linalg.eig(self.correlation_tx)
                eigvals_rx, eigvecs_rx = np.linalg.eig(self.correlation_rx)
                R_tx_sqrt = eigvecs_tx @ np.diag(np.sqrt(np.maximum(eigvals_tx, 0))) @ eigvecs_tx.T
                R_rx_sqrt = eigvecs_rx @ np.diag(np.sqrt(np.maximum(eigvals_rx, 0))) @ eigvecs_rx.T
            
            # Apply correlation: H = R_rx^(1/2) · H_iid · R_tx^(1/2)^T
            H = R_rx_sqrt @ H_iid @ R_tx_sqrt.T
        else:
            H = H_iid
        
        return H
    
    def calculate_mimo_capacity(self,
                                snr_db: float,
                                channel_matrix: Optional[np.ndarray] = None,
                                power_allocation: str = 'waterfilling') -> Tuple[float, np.ndarray]:
        """
        Calculate MIMO channel capacity with optimal precoding
        
        Args:
            snr_db: Signal-to-noise ratio in dB
            channel_matrix: MIMO channel matrix (generated if None)
            power_allocation: Power allocation strategy ('equal' or 'waterfilling')
            
        Returns:
            capacity: Channel capacity in bps/Hz
            precoding_matrix: Optimal precoding matrix
        """
        if channel_matrix is None:
            channel_matrix = self.generate_channel_matrix()
        
        # Convert SNR to linear scale
        snr_linear = 10 ** (snr_db / 10)
        
        # Singular Value Decomposition of channel matrix
        U, S, Vh = np.linalg.svd(channel_matrix, full_mat_indexes=False)
        num_streams = len(S)
        singular_values_sq = S ** 2
        
        if power_allocation == 'equal':
            # Equal power allocation across antennas
            power_per_stream = snr_linear / self.n_tx
            capacity = np.sum(np.log2(1 + power_per_stream * singular_values_sq))
            precoding_matrix = Vh.conj().T  # Right singular vectors
            
        elif power_allocation == 'waterfilling':
            # Water-filling power allocation (optimal for parallel channels)
            # Solve: ∑ max(0, μ - 1/λ_i) = P_total
            
            # Initial water level
            mu = 1.0
            allocated_power = np.zeros(num_streams)
            
            # Iterative water-filling algorithm
            for iteration in range(100):
                # Calculate power allocation for current water level
                allocated_power = np.maximum(0, mu - 1/(singular_values_sq + 1e-10))
                total_power = np.sum(allocated_power)
                
                # Check convergence
                if np.abs(total_power - snr_linear) < 1e-6:
                    break
                
                # Adjust water level
                mu += (snr_linear - total_power) / num_streams
            
            # Calculate capacity with optimal power allocation
            capacity = np.sum(np.log2(1 + allocated_power * singular_values_sq))
            
            # Construct precoding matrix: V * sqrt(P)
            precoding_matrix = Vh.conj().T @ np.diag(np.sqrt(allocated_power))
        
        else:
            raise ValueError(f"Unknown power allocation strategy: {power_allocation}")
        
        return float(np.real(capacity)), precoding_matrix
    
    def get_channel_rank(self, 
                         channel_matrix: Optional[np.ndarray] = None,
                         threshold: float = 1e-6) -> int:
        """
        Calculate effective rank of MIMO channel
        
        Args:
            channel_matrix: MIMO channel matrix
            threshold: Singular value threshold for effective rank
            
        Returns:
            Effective rank of channel matrix
        """
        if channel_matrix is None:
            channel_matrix = self.generate_channel_matrix()
        
        singular_values = np.linalg.svd(channel_matrix, compute_uv=False)
        
        effective_rank = np.sum(singular_values > threshold)
        
        return effective_rank
    
    def calculate_spatial_correlation(self) -> Dict:
        """
        Calculate spatial correlation metrics
        
        Returns:
            Dictionary with spatial correlation statistics
        """
        # Generate multiple channel realizations
        num_realizations = 100
        channel_matrices = []
        
        for _ in range(num_realizations):
            H = self.generate_channel_matrix(include_correlation=True)
            channel_matrices.append(H.flatten())
        
        channel_array = np.array(channel_matrices)  # shape: (num_realizations, n_rx*n_tx)
        
        correlation_matrix = np.corrcoef(channel_array.T)
        
        eigenvalues = np.linalg.eigvals(correlation_matrix)
        
        return {
            'correlation_matrix': correlation_matrix,
            'eigenvalues': eigenvalues,
            'effective_degrees': np.sum(eigenvalues) ** 2 / np.sum(eigenvalues ** 2),
            'correlation_strength': 1 - 1 / np.mean(eigenvalues ** 2)
        }



def generate_correlated_rayleigh_channels(num_channels: int,
                                          correlation_matrix: np.ndarray,
                                          num_samples: int = 1000,
                                          seed: Optional[int] = None) -> np.ndarray:
    """
    Generate correlated Rayleigh fading channels
    
    Args:
        num_channels: Number of correlated channels
        correlation_matrix: Desired correlation matrix (num_channels × num_channels)
        num_samples: Number of time samples
        seed: Random seed
        
    Returns:
        Correlated Rayleigh fading channels (num_channels × num_samples)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate independent complex Gaussian channels
    H_iid = (np.random.normal(0, np.sqrt(0.5), size=(num_channels, num_samples)) +
             1j * np.random.normal(0, np.sqrt(0.5), size=(num_channels, num_samples)))
    
    # Apply correlation using Cholesky decomposition
    try:
        L = np.linalg.cholesky(correlation_matrix)
    except np.linalg.LinAlgError:
        # If not positive definite, use eigenvalue decomposition
        eigvals, eigvecs = np.linalg.eig(correlation_matrix)
        L = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 0))) @ eigvecs.T
    
    # Correlate the channels: H_correlated = L @ H_iid
    H_correlated = L @ H_iid
    
    return H_correlated


def calculate_channel_quality_indicator(snr_db: float,
                                        channel_gain: float,
                                        noise_power: float = 1.0) -> float:
    """
    Calculate Channel Quality Indicator (CQI) based on SNR
    
    Args:
        snr_db: Signal-to-noise ratio in dB
        channel_gain: Current channel gain
        noise_power: Noise power spectral density
        
    Returns:
        CQI value (0-15 typically)
    """
    # Calculate effective SNR
    effective_snr_db = snr_db + 20 * np.log10(channel_gain / np.sqrt(noise_power))
    
    # Map to CQI (simplified mapping for LTE)
    # Real implementations use more complex mappings based on BLER targets
    cqi_mapping = [
        (0.0, 0), (-6.0, 1), (-4.0, 2), (-2.0, 3),
        (0.0, 4), (2.0, 5), (4.0, 6), (6.0, 7),
        (8.0, 8), (10.0, 9), (12.0, 10), (14.0, 11),
        (16.0, 12), (18.0, 13), (20.0, 14), (22.0, 15)
    ]
    
    # Find appropriate CQI
    cqi = 0
    for threshold, value in cqi_mapping:
        if effective_snr_db >= threshold:
            cqi = value
        else:
            break
    
    return cqi


def main():
    """Test function for channel models"""
    print("Testing Channel Models for MEC Task Offloading")
    print("=" * 50)
    
    # Test Markov Model
    print("\n1. Testing Markov Channel Model")
    markov_model = MarkovChannelModel(distance=100, seed=42)
    
    # Generate a few samples
    for i in range(5):
        gain = markov_model.sample_next_state()
        print(f"  Time {i}: State={markov_model.state}, Gain={gain:.6f}")
    
    # Get statistics
    stats = markov_model.get_state_statistics()
    print(f"  Average gain: {stats['average_gain']:.6f}")
    
    # Test AR Model
    print("\n2. Testing AR Channel Model")
    ar_model = ARChannelModel(distance=150, num_tx_antennas=2, num_rx_antennas=2, seed=42)
    
    for i in range(3):
        H = ar_model.sample_next_channel()
        print(f"  Time {i}: Channel matrix shape: {H.shape}")
    
    # Test Rayleigh Model
    print("\n3. Testing Rayleigh Fading Channel")
    rayleigh_channel = RayleighFadingChannel(distance=200, num_tx_antennas=4, 
                                             num_rx_antennas=2, seed=42)
    
    H = rayleigh_channel.generate_channel_matrix()
    print(f"  Generated channel matrix: {H.shape}")
    
    capacity = rayleigh_channel.calculate_channel_capacity(snr_db=20, channel_matrix=H)
    print(f"  Channel capacity: {capacity:.2f} bps/Hz")
    
    # Test MIMO Model
    print("\n4. Testing MIMO Channel Model")
    mimo_model = MIMOChannelModel(num_tx_antennas=4, num_rx_antennas=4, 
                                  channel_type='rayleigh', seed=42)
    
    H_mimo = mimo_model.generate_channel_matrix()
    capacity_mimo, precoder = mimo_model.calculate_mimo_capacity(snr_db=20, 
                                                                  channel_matrix=H_mimo)
    
    print(f"  MIMO channel shape: {H_mimo.shape}")
    print(f"  MIMO capacity: {capacity_mimo:.2f} bps/Hz")
    print(f"  Channel rank: {mimo_model.get_channel_rank(H_mimo)}")
    
    print("\n✓ All channel models tested successfully!")


if __name__ == "__main__":
    main()
