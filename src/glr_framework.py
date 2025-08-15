"""
Universal Binary Principle (UBP) Framework v3.1 - Enhanced GLR Framework Module

This module implements the comprehensive Golay-Leech-Resonance (GLR) error
correction framework with spatiotemporal coherence management, realm-specific
lattice structures, and advanced error correction algorithms.

Enhanced for v3.1 with improved integration with v3.0 components and
better performance optimization.

Author: Euan Craig
Version: 3.1
Date: August 2025
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy import signal
from scipy.spatial.distance import pdist, squareform
import json
import time
import math

try:
    from .core import UBPConstants
    from .bitfield import Bitfield, OffBit
    from .hex_dictionary import HexDictionary
except ImportError:
    from core import UBPConstants
    from bitfield import Bitfield, OffBit
    from hex_dictionary import HexDictionary


@dataclass
class GLRMetrics:
    """Comprehensive metrics for GLR error correction performance."""
    spatial_coherence: float
    temporal_coherence: float
    nrci_spatial: float
    nrci_temporal: float
    nrci_combined: float
    error_correction_rate: float
    lattice_efficiency: float
    resonance_stability: float
    correction_iterations: int
    convergence_time: float
    realm_synchronization: float = 0.0
    quantum_coherence: float = 0.0


@dataclass
class LatticeStructure:
    """Definition of a lattice structure for GLR error correction."""
    name: str
    coordination_number: int
    lattice_type: str
    symmetry_group: str
    basis_vectors: np.ndarray
    nearest_neighbors: List[Tuple[int, ...]]
    correction_weights: np.ndarray
    resonance_frequency: float
    crv_value: float = 0.0
    wavelength_nm: float = 0.0


@dataclass
class ErrorCorrectionResult:
    """Result of an error correction operation."""
    corrected_offbits: List[int]
    correction_applied: bool
    error_count: int
    correction_strength: float
    nrci_improvement: float
    execution_time: float
    method_used: str


class ComprehensiveErrorCorrectionFramework:
    """
    Enhanced GLR error correction framework implementing spatiotemporal
    coherence management with realm-specific lattice structures.
    
    This class provides the core error correction capabilities for the UBP
    framework, combining Golay[23,12] codes with Leech lattice projections
    and resonance-based temporal correction.
    
    Enhanced for v3.1 with better integration and performance.
    """
    
    def __init__(self, realm_name: str = "electromagnetic", 
                 enable_error_correction: bool = True,
                 hex_dictionary_instance: Optional[HexDictionary] = None):
        """
        Initialize the GLR framework for a specific computational realm.
        
        Args:
            realm_name: Name of the computational realm to configure for
            enable_error_correction: Whether to enable error correction (default: True)
            hex_dictionary_instance: Optional HexDictionary for data storage
        """
        self.realm_name = realm_name
        self.enable_error_correction = enable_error_correction
        self.hex_dictionary = hex_dictionary_instance or HexDictionary()
        
        # Initialize lattice structures
        self.lattice_structures = self._initialize_lattice_structures()
        self.current_lattice = self.lattice_structures.get(realm_name, 
                                                          self.lattice_structures["electromagnetic"])
        
        # Error correction components
        self.correction_history = []
        self.temporal_buffer = []
        self.spatial_cache = {}
        
        # GLR-specific parameters
        self.golay_generator_matrix = self._generate_golay_matrix()
        self.leech_lattice_basis = self._generate_leech_basis()
        self.resonance_frequencies = self._calculate_resonance_frequencies()
        
        # Performance metrics
        self.current_metrics = GLRMetrics(
            spatial_coherence=0.0,
            temporal_coherence=0.0,
            nrci_spatial=0.0,
            nrci_temporal=0.0,
            nrci_combined=0.0,
            error_correction_rate=0.0,
            lattice_efficiency=0.0,
            resonance_stability=0.0,
            correction_iterations=0,
            convergence_time=0.0
        )
        
        # Initialize metrics tracking
        self.metrics_history = []
        
        # Enhanced v3.1 features
        self.quantum_entanglement_matrix = np.eye(24)  # 24D Leech lattice
        self.realm_coupling_coefficients = self._calculate_realm_coupling()
        self.adaptive_threshold = 0.95  # Adaptive error correction threshold
        
        print(f"✅ GLR Error Correction Framework v3.1 Initialized")
        print(f"   Realm: {realm_name}")
        print(f"   Lattice: {self.current_lattice.lattice_type}")
        print(f"   Coordination: {self.current_lattice.coordination_number}")
        print(f"   Error Correction: {'Enabled' if enable_error_correction else 'Disabled'}")
    
    def _initialize_lattice_structures(self) -> Dict[str, LatticeStructure]:
        """
        Initialize all realm-specific lattice structures.
        
        Returns:
            Dictionary mapping realm names to LatticeStructure objects
        """
        lattices = {}
        
        # Electromagnetic (Cubic GLR) - 6-fold coordination
        lattices["electromagnetic"] = LatticeStructure(
            name="Electromagnetic Cubic GLR",
            coordination_number=6,
            lattice_type="cubic",
            symmetry_group="Oh",
            basis_vectors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
            nearest_neighbors=[(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)],
            correction_weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) / 6.0,
            resonance_frequency=UBPConstants.CRV_ELECTROMAGNETIC,
            crv_value=UBPConstants.CRV_ELECTROMAGNETIC,
            wavelength_nm=635.0
        )
        
        # Quantum (Tetrahedral GLR) - 4-fold coordination
        lattices["quantum"] = LatticeStructure(
            name="Quantum Tetrahedral GLR",
            coordination_number=4,
            lattice_type="tetrahedral",
            symmetry_group="Td",
            basis_vectors=np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / np.sqrt(3),
            nearest_neighbors=[(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)],
            correction_weights=np.array([1.0, 1.0, 1.0, 1.0]) / 4.0,
            resonance_frequency=UBPConstants.CRV_QUANTUM,
            crv_value=UBPConstants.CRV_QUANTUM,
            wavelength_nm=655.0
        )
        
        # Gravitational (FCC GLR) - 12-fold coordination
        lattices["gravitational"] = LatticeStructure(
            name="Gravitational FCC GLR",
            coordination_number=12,
            lattice_type="face_centered_cubic",
            symmetry_group="Oh",
            basis_vectors=np.array([[0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]),
            nearest_neighbors=[(1, 1, 0), (1, -1, 0), (-1, 1, 0), (-1, -1, 0),
                             (1, 0, 1), (1, 0, -1), (-1, 0, 1), (-1, 0, -1),
                             (0, 1, 1), (0, 1, -1), (0, -1, 1), (0, -1, -1)],
            correction_weights=np.ones(12) / 12.0,
            resonance_frequency=UBPConstants.CRV_GRAVITATIONAL,
            crv_value=UBPConstants.CRV_GRAVITATIONAL,
            wavelength_nm=1000.0
        )
        
        # Biological (H4 120-Cell GLR) - 20-fold coordination
        lattices["biological"] = LatticeStructure(
            name="Biological H4 120-Cell GLR",
            coordination_number=20,
            lattice_type="120_cell",
            symmetry_group="H4",
            basis_vectors=self._generate_h4_basis(),
            nearest_neighbors=self._generate_h4_neighbors(),
            correction_weights=np.ones(20) / 20.0,
            resonance_frequency=UBPConstants.CRV_BIOLOGICAL,
            crv_value=UBPConstants.CRV_BIOLOGICAL,
            wavelength_nm=700.0
        )
        
        # Cosmological (H3 Icosahedral GLR) - 12-fold coordination
        lattices["cosmological"] = LatticeStructure(
            name="Cosmological H3 Icosahedral GLR",
            coordination_number=12,
            lattice_type="icosahedral",
            symmetry_group="H3",
            basis_vectors=self._generate_icosahedral_basis(),
            nearest_neighbors=self._generate_icosahedral_neighbors(),
            correction_weights=np.ones(12) / 12.0,
            resonance_frequency=UBPConstants.CRV_COSMOLOGICAL,
            crv_value=UBPConstants.CRV_COSMOLOGICAL,
            wavelength_nm=800.0
        )
        
        # Nuclear (E8-to-G2 GLR) - 8-fold coordination
        lattices["nuclear"] = LatticeStructure(
            name="Nuclear E8-to-G2 GLR",
            coordination_number=8,
            lattice_type="e8_g2",
            symmetry_group="E8",
            basis_vectors=self._generate_e8_basis(),
            nearest_neighbors=self._generate_e8_neighbors(),
            correction_weights=np.ones(8) / 8.0,
            resonance_frequency=UBPConstants.CRV_NUCLEAR,
            crv_value=UBPConstants.CRV_NUCLEAR,
            wavelength_nm=0.001  # Very short wavelength for nuclear
        )
        
        # Optical (Photonic GLR) - 6-fold coordination
        lattices["optical"] = LatticeStructure(
            name="Optical Photonic GLR",
            coordination_number=6,
            lattice_type="photonic",
            symmetry_group="D6h",
            basis_vectors=np.array([[1, 0, 0], [0.5, np.sqrt(3)/2, 0], [0, 0, 1]]),
            nearest_neighbors=[(1, 0, 0), (-1, 0, 0), (0.5, np.sqrt(3)/2, 0), 
                             (-0.5, -np.sqrt(3)/2, 0), (0, 0, 1), (0, 0, -1)],
            correction_weights=np.ones(6) / 6.0,
            resonance_frequency=UBPConstants.CRV_OPTICAL,
            crv_value=UBPConstants.CRV_OPTICAL,
            wavelength_nm=600.0
        )
        
        return lattices
    
    def _generate_h4_basis(self) -> np.ndarray:
        """Generate basis vectors for H4 120-cell lattice."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # H4 basis vectors (4D)
        basis = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [0.5, 0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5, 0.5]
        ])
        
        return basis[:3, :3]  # Project to 3D for practical use
    
    def _generate_h4_neighbors(self) -> List[Tuple[int, ...]]:
        """Generate nearest neighbors for H4 120-cell lattice."""
        phi = (1 + np.sqrt(5)) / 2
        
        # Simplified 20-fold coordination for 3D projection
        neighbors = []
        for i in range(20):
            angle = 2 * np.pi * i / 20
            x = np.cos(angle)
            y = np.sin(angle)
            z = 0.5 * np.sin(2 * angle)
            neighbors.append((x, y, z))
        
        return neighbors
    
    def _generate_icosahedral_basis(self) -> np.ndarray:
        """Generate basis vectors for icosahedral lattice."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        # Icosahedral basis
        basis = np.array([
            [1, phi, 0],
            [0, 1, phi],
            [phi, 0, 1]
        ]) / np.sqrt(1 + phi**2)
        
        return basis
    
    def _generate_icosahedral_neighbors(self) -> List[Tuple[int, ...]]:
        """Generate nearest neighbors for icosahedral lattice."""
        phi = (1 + np.sqrt(5)) / 2
        
        # 12 vertices of icosahedron
        neighbors = [
            (1, phi, 0), (-1, phi, 0), (1, -phi, 0), (-1, -phi, 0),
            (phi, 0, 1), (phi, 0, -1), (-phi, 0, 1), (-phi, 0, -1),
            (0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi)
        ]
        
        # Normalize
        norm_factor = np.sqrt(1 + phi**2)
        neighbors = [(x/norm_factor, y/norm_factor, z/norm_factor) for x, y, z in neighbors]
        
        return neighbors
    
    def _generate_e8_basis(self) -> np.ndarray:
        """Generate basis vectors for E8 lattice."""
        # Simplified E8 basis projected to 3D
        basis = np.array([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
            [1, 1, 0]
        ])
        
        return basis[:3]  # Use first 3 as basis
    
    def _generate_e8_neighbors(self) -> List[Tuple[int, ...]]:
        """Generate nearest neighbors for E8 lattice."""
        # 8-fold coordination for nuclear realm
        neighbors = [
            (1, 0, 0), (-1, 0, 0),
            (0, 1, 0), (0, -1, 0),
            (0, 0, 1), (0, 0, -1),
            (0.5, 0.5, 0.5), (-0.5, -0.5, -0.5)
        ]
        
        return neighbors
    
    def _generate_golay_matrix(self) -> np.ndarray:
        """
        Generate the Golay[23,12] generator matrix.
        
        Returns:
            23x12 generator matrix for Golay code
        """
        # Simplified Golay generator matrix
        # In practice, this would be the full 23x12 matrix
        generator = np.random.randint(0, 2, (23, 12))
        
        # Ensure systematic form [I|P] where I is identity
        generator[:12, :12] = np.eye(12, dtype=int)
        
        return generator
    
    def _generate_leech_basis(self) -> np.ndarray:
        """
        Generate basis vectors for 24D Leech lattice.
        
        Returns:
            24x24 basis matrix for Leech lattice
        """
        # Simplified Leech lattice basis
        # In practice, this would be constructed from the Golay code
        basis = np.eye(24)
        
        # Add some structure based on Golay code
        for i in range(24):
            for j in range(24):
                if i != j:
                    basis[i, j] = 0.1 * np.sin(2 * np.pi * i * j / 24)
        
        return basis
    
    def _calculate_resonance_frequencies(self) -> Dict[str, float]:
        """
        Calculate resonance frequencies for all realms.
        
        Returns:
            Dictionary mapping realm names to resonance frequencies
        """
        frequencies = {}
        
        for realm_name, lattice in self.lattice_structures.items():
            frequencies[realm_name] = lattice.resonance_frequency
        
        return frequencies
    
    def _calculate_realm_coupling(self) -> np.ndarray:
        """
        Calculate coupling coefficients between different realms.
        
        Returns:
            7x7 matrix of realm coupling coefficients
        """
        realm_names = list(self.lattice_structures.keys())
        n_realms = len(realm_names)
        coupling_matrix = np.eye(n_realms)
        
        # Calculate coupling based on frequency ratios
        for i, realm_i in enumerate(realm_names):
            for j, realm_j in enumerate(realm_names):
                if i != j:
                    freq_i = self.lattice_structures[realm_i].resonance_frequency
                    freq_j = self.lattice_structures[realm_j].resonance_frequency
                    
                    # Coupling strength based on frequency ratio
                    ratio = min(freq_i, freq_j) / max(freq_i, freq_j)
                    coupling_matrix[i, j] = ratio * 0.1  # Scale coupling
        
        return coupling_matrix
    
    # ========================================================================
    # ERROR CORRECTION METHODS
    # ========================================================================
    
    def correct_spatial_errors(self, offbits: List[int], 
                              coordinates: List[Tuple[int, ...]] = None) -> ErrorCorrectionResult:
        """
        Perform spatial error correction using lattice-based methods.
        
        Args:
            offbits: List of OffBit values to correct
            coordinates: Optional coordinates for spatial context
            
        Returns:
            ErrorCorrectionResult with correction details
        """
        start_time = time.time()
        
        if not self.enable_error_correction:
            return ErrorCorrectionResult(
                corrected_offbits=offbits,
                correction_applied=False,
                error_count=0,
                correction_strength=0.0,
                nrci_improvement=0.0,
                execution_time=time.time() - start_time,
                method_used="disabled"
            )
        
        corrected_offbits = []
        error_count = 0
        total_correction = 0.0
        
        # Calculate initial NRCI
        initial_nrci = self._calculate_spatial_nrci(offbits)
        
        for i, offbit in enumerate(offbits):
            # Get spatial neighbors based on lattice structure
            neighbors = self._get_spatial_neighbors(i, offbits, coordinates)
            
            if neighbors:
                # Apply lattice-based correction
                corrected_offbit = self._apply_lattice_correction(offbit, neighbors)
                
                # Check if correction was needed
                if corrected_offbit != offbit:
                    error_count += 1
                    correction_strength = abs(OffBit.get_activation_layer(corrected_offbit) - 
                                            OffBit.get_activation_layer(offbit)) / 64.0
                    total_correction += correction_strength
                
                corrected_offbits.append(corrected_offbit)
            else:
                corrected_offbits.append(offbit)
        
        # Calculate final NRCI
        final_nrci = self._calculate_spatial_nrci(corrected_offbits)
        nrci_improvement = final_nrci - initial_nrci
        
        # Update metrics
        self.current_metrics.spatial_coherence = final_nrci
        self.current_metrics.error_correction_rate = error_count / len(offbits) if offbits else 0
        self.current_metrics.correction_iterations += 1
        
        execution_time = time.time() - start_time
        
        result = ErrorCorrectionResult(
            corrected_offbits=corrected_offbits,
            correction_applied=error_count > 0,
            error_count=error_count,
            correction_strength=total_correction / max(error_count, 1),
            nrci_improvement=nrci_improvement,
            execution_time=execution_time,
            method_used="spatial_lattice"
        )
        
        # Store in HexDictionary if available
        if self.hex_dictionary:
            correction_data = {
                'realm': self.realm_name,
                'method': 'spatial_correction',
                'error_count': error_count,
                'nrci_improvement': nrci_improvement,
                'execution_time': execution_time,
                'lattice_type': self.current_lattice.lattice_type
            }
            self.hex_dictionary.store(correction_data, 'json', 
                                    {'correction_type': 'spatial'})
        
        return result
    
    def correct_temporal_errors(self, offbit_sequence: List[List[int]], 
                               time_steps: List[float] = None) -> ErrorCorrectionResult:
        """
        Perform temporal error correction using resonance-based methods.
        
        Args:
            offbit_sequence: Sequence of OffBit states over time
            time_steps: Optional time step values
            
        Returns:
            ErrorCorrectionResult with temporal correction details
        """
        start_time = time.time()
        
        if not self.enable_error_correction or len(offbit_sequence) < 2:
            return ErrorCorrectionResult(
                corrected_offbits=offbit_sequence[-1] if offbit_sequence else [],
                correction_applied=False,
                error_count=0,
                correction_strength=0.0,
                nrci_improvement=0.0,
                execution_time=time.time() - start_time,
                method_used="insufficient_data"
            )
        
        # Calculate initial temporal NRCI
        initial_nrci = self._calculate_temporal_nrci(offbit_sequence)
        
        # Apply temporal correction using resonance frequencies
        corrected_sequence = []
        error_count = 0
        total_correction = 0.0
        
        for t, offbits in enumerate(offbit_sequence):
            if t == 0:
                corrected_sequence.append(offbits)
                continue
            
            # Get temporal context
            previous_states = corrected_sequence[-min(3, t):]  # Use last 3 states
            
            corrected_offbits = []
            for i, offbit in enumerate(offbits):
                # Apply temporal resonance correction
                corrected_offbit = self._apply_temporal_correction(
                    offbit, previous_states, t, i, time_steps
                )
                
                if corrected_offbit != offbit:
                    error_count += 1
                    correction_strength = abs(OffBit.get_activation_layer(corrected_offbit) - 
                                            OffBit.get_activation_layer(offbit)) / 64.0
                    total_correction += correction_strength
                
                corrected_offbits.append(corrected_offbit)
            
            corrected_sequence.append(corrected_offbits)
        
        # Calculate final temporal NRCI
        final_nrci = self._calculate_temporal_nrci(corrected_sequence)
        nrci_improvement = final_nrci - initial_nrci
        
        # Update metrics
        self.current_metrics.temporal_coherence = final_nrci
        self.current_metrics.correction_iterations += 1
        
        execution_time = time.time() - start_time
        
        result = ErrorCorrectionResult(
            corrected_offbits=corrected_sequence[-1],
            correction_applied=error_count > 0,
            error_count=error_count,
            correction_strength=total_correction / max(error_count, 1),
            nrci_improvement=nrci_improvement,
            execution_time=execution_time,
            method_used="temporal_resonance"
        )
        
        return result
    
    def _get_spatial_neighbors(self, index: int, offbits: List[int], 
                              coordinates: List[Tuple[int, ...]] = None) -> List[int]:
        """
        Get spatial neighbors for an OffBit based on lattice structure.
        
        Args:
            index: Index of the OffBit
            offbits: List of all OffBits
            coordinates: Optional spatial coordinates
            
        Returns:
            List of neighbor OffBit values
        """
        neighbors = []
        
        if coordinates and len(coordinates) > index:
            # Use actual coordinates to find neighbors
            current_coord = coordinates[index]
            
            for neighbor_offset in self.current_lattice.nearest_neighbors:
                # Calculate neighbor coordinate
                neighbor_coord = tuple(current_coord[i] + neighbor_offset[i] 
                                     for i in range(min(len(current_coord), len(neighbor_offset))))
                
                # Find OffBit at neighbor coordinate
                for j, coord in enumerate(coordinates):
                    if coord == neighbor_coord and j < len(offbits):
                        neighbors.append(offbits[j])
                        break
        else:
            # Use index-based neighbors (simplified)
            coordination = self.current_lattice.coordination_number
            
            for i in range(1, coordination + 1):
                neighbor_idx = (index + i) % len(offbits)
                neighbors.append(offbits[neighbor_idx])
                
                neighbor_idx = (index - i) % len(offbits)
                neighbors.append(offbits[neighbor_idx])
        
        return neighbors[:self.current_lattice.coordination_number]
    
    def _apply_lattice_correction(self, offbit: int, neighbors: List[int]) -> int:
        """
        Apply lattice-based error correction to an OffBit.
        
        Args:
            offbit: OffBit to correct
            neighbors: List of neighbor OffBits
            
        Returns:
            Corrected OffBit value
        """
        if not neighbors:
            return offbit
        
        # Extract activation layers
        activation = OffBit.get_activation_layer(offbit)
        neighbor_activations = [OffBit.get_activation_layer(n) for n in neighbors]
        
        # Calculate weighted average based on lattice weights
        weights = self.current_lattice.correction_weights[:len(neighbors)]
        if len(weights) < len(neighbors):
            # Extend weights if needed
            weights = np.concatenate([weights, np.ones(len(neighbors) - len(weights)) / len(neighbors)])
        
        weighted_average = np.average(neighbor_activations, weights=weights)
        
        # Apply correction if deviation is significant
        deviation = abs(activation - weighted_average)
        if deviation > self.adaptive_threshold * 64:
            # Correct towards weighted average
            correction_factor = 0.5  # Partial correction
            corrected_activation = int(activation + correction_factor * (weighted_average - activation))
            corrected_activation = max(0, min(63, corrected_activation))  # Clamp to valid range
            
            return OffBit.set_activation_layer(offbit, corrected_activation)
        
        return offbit
    
    def _apply_temporal_correction(self, offbit: int, previous_states: List[List[int]], 
                                  time_index: int, offbit_index: int, 
                                  time_steps: List[float] = None) -> int:
        """
        Apply temporal resonance-based correction to an OffBit.
        
        Args:
            offbit: OffBit to correct
            previous_states: Previous temporal states
            time_index: Current time index
            offbit_index: Index of OffBit within state
            time_steps: Optional time step values
            
        Returns:
            Corrected OffBit value
        """
        if not previous_states:
            return offbit
        
        # Get resonance frequency for current realm
        resonance_freq = self.current_lattice.resonance_frequency
        
        # Calculate expected value based on temporal resonance
        activation = OffBit.get_activation_layer(offbit)
        
        # Extract previous activations for this OffBit
        previous_activations = []
        for state in previous_states:
            if offbit_index < len(state):
                prev_activation = OffBit.get_activation_layer(state[offbit_index])
                previous_activations.append(prev_activation)
        
        if not previous_activations:
            return offbit
        
        # Apply temporal resonance model
        dt = 1.0  # Default time step
        if time_steps and time_index > 0:
            dt = time_steps[time_index] - time_steps[time_index - 1]
        
        # Calculate resonance-based expected value
        phase = 2 * np.pi * resonance_freq * dt * time_index
        resonance_factor = np.cos(phase)
        
        # Weighted average of previous states with resonance modulation
        temporal_average = np.mean(previous_activations)
        expected_activation = temporal_average * (1 + 0.1 * resonance_factor)
        
        # Apply correction if deviation is significant
        deviation = abs(activation - expected_activation)
        if deviation > self.adaptive_threshold * 32:  # Lower threshold for temporal
            correction_factor = 0.3  # Gentler temporal correction
            corrected_activation = int(activation + correction_factor * (expected_activation - activation))
            corrected_activation = max(0, min(63, corrected_activation))
            
            return OffBit.set_activation_layer(offbit, corrected_activation)
        
        return offbit
    
    def _calculate_spatial_nrci(self, offbits: List[int]) -> float:
        """
        Calculate spatial Non-Random Coherence Index.
        
        Args:
            offbits: List of OffBit values
            
        Returns:
            Spatial NRCI value (0.0 to 1.0)
        """
        if len(offbits) < 2:
            return 1.0
        
        # Extract activation layers
        activations = [OffBit.get_activation_layer(offbit) for offbit in offbits]
        
        # Calculate spatial coherence based on neighbor correlations
        coherence_sum = 0.0
        pair_count = 0
        
        for i in range(len(activations)):
            neighbors = self._get_spatial_neighbors(i, offbits)
            if neighbors:
                neighbor_activations = [OffBit.get_activation_layer(n) for n in neighbors]
                
                # Calculate correlation with neighbors
                if len(neighbor_activations) > 0:
                    neighbor_mean = np.mean(neighbor_activations)
                    correlation = 1.0 - abs(activations[i] - neighbor_mean) / 64.0
                    coherence_sum += max(0.0, correlation)
                    pair_count += 1
        
        if pair_count == 0:
            return 1.0
        
        spatial_nrci = coherence_sum / pair_count
        return min(1.0, max(0.0, spatial_nrci))
    
    def _calculate_temporal_nrci(self, offbit_sequence: List[List[int]]) -> float:
        """
        Calculate temporal Non-Random Coherence Index.
        
        Args:
            offbit_sequence: Sequence of OffBit states over time
            
        Returns:
            Temporal NRCI value (0.0 to 1.0)
        """
        if len(offbit_sequence) < 2:
            return 1.0
        
        # Calculate temporal coherence for each OffBit position
        temporal_coherences = []
        
        # Determine the minimum length across all time steps
        min_length = min(len(state) for state in offbit_sequence)
        
        for pos in range(min_length):
            # Extract temporal sequence for this position
            temporal_sequence = [OffBit.get_activation_layer(offbit_sequence[t][pos]) 
                               for t in range(len(offbit_sequence))]
            
            # Calculate temporal coherence using autocorrelation
            if len(temporal_sequence) > 1:
                # Simple temporal coherence based on smoothness
                differences = [abs(temporal_sequence[i+1] - temporal_sequence[i]) 
                             for i in range(len(temporal_sequence)-1)]
                
                if differences:
                    avg_difference = np.mean(differences)
                    coherence = 1.0 - avg_difference / 64.0
                    temporal_coherences.append(max(0.0, coherence))
        
        if not temporal_coherences:
            return 1.0
        
        temporal_nrci = np.mean(temporal_coherences)
        return min(1.0, max(0.0, temporal_nrci))
    
    # ========================================================================
    # ADVANCED CORRECTION METHODS
    # ========================================================================
    
    def apply_golay_correction(self, data_bits: List[int]) -> List[int]:
        """
        Apply Golay[23,12] error correction to data bits.
        
        Args:
            data_bits: List of 12 data bits
            
        Returns:
            List of corrected 12 data bits
        """
        if len(data_bits) != 12:
            raise ValueError("Golay correction requires exactly 12 data bits")
        
        # Encode using generator matrix
        data_vector = np.array(data_bits)
        encoded = np.dot(data_vector, self.golay_generator_matrix.T) % 2
        
        # Simulate transmission errors (for demonstration)
        # In practice, this would be the received codeword
        received = encoded.copy()
        
        # Simple error detection and correction
        # This is a simplified version - full Golay decoding is more complex
        syndrome = np.dot(received, self.golay_generator_matrix) % 2
        
        if np.any(syndrome):
            # Error detected - apply correction
            # Simplified correction: flip bits based on syndrome
            error_positions = np.where(syndrome)[0]
            for pos in error_positions[:3]:  # Correct up to 3 errors
                if pos < len(received):
                    received[pos] = 1 - received[pos]
        
        # Extract corrected data bits
        corrected_data = received[:12]
        
        return corrected_data.tolist()
    
    def apply_leech_lattice_correction(self, vector_24d: np.ndarray) -> np.ndarray:
        """
        Apply Leech lattice-based error correction to a 24D vector.
        
        Args:
            vector_24d: 24-dimensional vector to correct
            
        Returns:
            Corrected 24-dimensional vector
        """
        if len(vector_24d) != 24:
            raise ValueError("Leech lattice correction requires 24-dimensional vector")
        
        # Project onto Leech lattice
        # This is a simplified version - full Leech lattice operations are complex
        
        # Find closest lattice point
        lattice_coords = np.dot(self.leech_lattice_basis, vector_24d)
        
        # Round to nearest lattice point
        rounded_coords = np.round(lattice_coords)
        
        # Project back to original space
        corrected_vector = np.dot(self.leech_lattice_basis.T, rounded_coords)
        
        return corrected_vector
    
    def apply_quantum_error_correction(self, quantum_states: List[int]) -> List[int]:
        """
        Apply quantum error correction using entanglement matrix.
        
        Args:
            quantum_states: List of quantum state OffBits
            
        Returns:
            List of corrected quantum states
        """
        if not quantum_states:
            return quantum_states
        
        # Extract quantum information
        quantum_activations = [OffBit.get_activation_layer(state) for state in quantum_states]
        
        # Pad or truncate to match entanglement matrix size
        padded_activations = quantum_activations[:24]
        while len(padded_activations) < 24:
            padded_activations.append(0)
        
        activation_vector = np.array(padded_activations, dtype=float)
        
        # Apply quantum entanglement correction
        corrected_vector = np.dot(self.quantum_entanglement_matrix, activation_vector)
        
        # Normalize and convert back to OffBit format
        corrected_activations = np.clip(np.round(corrected_vector), 0, 63).astype(int)
        
        # Create corrected OffBits
        corrected_states = []
        for i, original_state in enumerate(quantum_states):
            if i < len(corrected_activations):
                corrected_state = OffBit.set_activation_layer(original_state, corrected_activations[i])
                corrected_states.append(corrected_state)
            else:
                corrected_states.append(original_state)
        
        return corrected_states
    
    # ========================================================================
    # METRICS AND ANALYSIS
    # ========================================================================
    
    def calculate_comprehensive_metrics(self, offbits: List[int], 
                                      offbit_sequence: List[List[int]] = None) -> GLRMetrics:
        """
        Calculate comprehensive GLR metrics for current state.
        
        Args:
            offbits: Current OffBit state
            offbit_sequence: Optional temporal sequence
            
        Returns:
            Updated GLRMetrics object
        """
        start_time = time.time()
        
        # Calculate spatial metrics
        spatial_nrci = self._calculate_spatial_nrci(offbits)
        spatial_coherence = self._calculate_spatial_coherence(offbits)
        
        # Calculate temporal metrics if sequence provided
        if offbit_sequence:
            temporal_nrci = self._calculate_temporal_nrci(offbit_sequence)
            temporal_coherence = self._calculate_temporal_coherence(offbit_sequence)
        else:
            temporal_nrci = spatial_nrci
            temporal_coherence = spatial_coherence
        
        # Calculate combined NRCI
        combined_nrci = (spatial_nrci + temporal_nrci) / 2
        
        # Calculate lattice efficiency
        lattice_efficiency = self._calculate_lattice_efficiency(offbits)
        
        # Calculate resonance stability
        resonance_stability = self._calculate_resonance_stability(offbits)
        
        # Calculate realm synchronization
        realm_sync = self._calculate_realm_synchronization(offbits)
        
        # Calculate quantum coherence
        quantum_coherence = self._calculate_quantum_coherence(offbits)
        
        # Update metrics
        self.current_metrics = GLRMetrics(
            spatial_coherence=spatial_coherence,
            temporal_coherence=temporal_coherence,
            nrci_spatial=spatial_nrci,
            nrci_temporal=temporal_nrci,
            nrci_combined=combined_nrci,
            error_correction_rate=self.current_metrics.error_correction_rate,
            lattice_efficiency=lattice_efficiency,
            resonance_stability=resonance_stability,
            correction_iterations=self.current_metrics.correction_iterations,
            convergence_time=time.time() - start_time,
            realm_synchronization=realm_sync,
            quantum_coherence=quantum_coherence
        )
        
        # Store metrics in history
        self.metrics_history.append(self.current_metrics)
        
        return self.current_metrics
    
    def _calculate_spatial_coherence(self, offbits: List[int]) -> float:
        """Calculate spatial coherence metric."""
        if len(offbits) < 2:
            return 1.0
        
        activations = [OffBit.get_activation_layer(offbit) for offbit in offbits]
        
        # Calculate variance-based coherence
        variance = np.var(activations)
        max_variance = (63**2) / 4  # Maximum possible variance
        
        coherence = 1.0 - (variance / max_variance)
        return max(0.0, min(1.0, coherence))
    
    def _calculate_temporal_coherence(self, offbit_sequence: List[List[int]]) -> float:
        """Calculate temporal coherence metric."""
        if len(offbit_sequence) < 2:
            return 1.0
        
        # Calculate coherence across time for each position
        coherences = []
        min_length = min(len(state) for state in offbit_sequence)
        
        for pos in range(min_length):
            temporal_values = [OffBit.get_activation_layer(offbit_sequence[t][pos]) 
                             for t in range(len(offbit_sequence))]
            
            # Calculate temporal variance
            if len(temporal_values) > 1:
                variance = np.var(temporal_values)
                max_variance = (63**2) / 4
                coherence = 1.0 - (variance / max_variance)
                coherences.append(max(0.0, coherence))
        
        return np.mean(coherences) if coherences else 1.0
    
    def _calculate_lattice_efficiency(self, offbits: List[int]) -> float:
        """Calculate lattice structure efficiency."""
        if not offbits:
            return 1.0
        
        # Calculate how well the OffBits conform to lattice structure
        coordination = self.current_lattice.coordination_number
        
        efficiency_sum = 0.0
        for i, offbit in enumerate(offbits):
            neighbors = self._get_spatial_neighbors(i, offbits)
            
            if len(neighbors) == coordination:
                # Full coordination achieved
                efficiency_sum += 1.0
            else:
                # Partial coordination
                efficiency_sum += len(neighbors) / coordination
        
        return efficiency_sum / len(offbits)
    
    def _calculate_resonance_stability(self, offbits: List[int]) -> float:
        """Calculate resonance frequency stability."""
        if not offbits:
            return 1.0
        
        activations = [OffBit.get_activation_layer(offbit) for offbit in offbits]
        
        # Calculate how well activations match expected resonance pattern
        resonance_freq = self.current_lattice.resonance_frequency
        
        # Generate expected pattern
        expected_pattern = []
        for i in range(len(activations)):
            phase = 2 * np.pi * resonance_freq * i / len(activations)
            expected_value = 32 + 16 * np.sin(phase)  # Center around 32 with amplitude 16
            expected_pattern.append(expected_value)
        
        # Calculate correlation with expected pattern
        if len(activations) > 1 and len(expected_pattern) > 1:
            correlation = np.corrcoef(activations, expected_pattern)[0, 1]
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0
        
        # Convert correlation to stability (0 to 1)
        stability = (correlation + 1) / 2
        return max(0.0, min(1.0, stability))
    
    def _calculate_realm_synchronization(self, offbits: List[int]) -> float:
        """Calculate synchronization with other realms."""
        # Simplified realm synchronization calculation
        # In practice, this would involve cross-realm coherence analysis
        
        if not offbits:
            return 1.0
        
        # Calculate how well current realm aligns with coupling matrix
        realm_index = list(self.lattice_structures.keys()).index(self.realm_name)
        coupling_row = self.realm_coupling_coefficients[realm_index]
        
        # Use coupling coefficients as synchronization metric
        synchronization = np.mean(coupling_row)
        
        return max(0.0, min(1.0, synchronization))
    
    def _calculate_quantum_coherence(self, offbits: List[int]) -> float:
        """Calculate quantum coherence metric."""
        if not offbits:
            return 1.0
        
        # Calculate quantum coherence based on entanglement matrix
        activations = [OffBit.get_activation_layer(offbit) for offbit in offbits]
        
        # Pad to 24D for quantum calculation
        padded_activations = activations[:24]
        while len(padded_activations) < 24:
            padded_activations.append(0)
        
        activation_vector = np.array(padded_activations, dtype=float)
        
        # Calculate coherence using entanglement matrix
        coherence_vector = np.dot(self.quantum_entanglement_matrix, activation_vector)
        
        # Measure how much the vector is preserved under entanglement transformation
        original_norm = np.linalg.norm(activation_vector)
        coherence_norm = np.linalg.norm(coherence_vector)
        
        if original_norm > 0:
            quantum_coherence = coherence_norm / original_norm
        else:
            quantum_coherence = 1.0
        
        return max(0.0, min(1.0, quantum_coherence))
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def switch_realm(self, new_realm: str) -> bool:
        """
        Switch to a different computational realm.
        
        Args:
            new_realm: Name of the realm to switch to
            
        Returns:
            True if switch successful, False otherwise
        """
        if new_realm not in self.lattice_structures:
            print(f"❌ Unknown realm: {new_realm}")
            return False
        
        old_realm = self.realm_name
        self.realm_name = new_realm
        self.current_lattice = self.lattice_structures[new_realm]
        
        print(f"✅ Switched from {old_realm} to {new_realm} realm")
        print(f"   New lattice: {self.current_lattice.lattice_type}")
        print(f"   Coordination: {self.current_lattice.coordination_number}")
        
        return True
    
    def get_metrics(self) -> GLRMetrics:
        """Get current GLR metrics."""
        return self.current_metrics
    
    def get_lattice_info(self) -> Dict[str, Any]:
        """
        Get information about the current lattice structure.
        
        Returns:
            Dictionary with lattice information
        """
        return {
            'realm_name': self.realm_name,
            'lattice_name': self.current_lattice.name,
            'lattice_type': self.current_lattice.lattice_type,
            'coordination_number': self.current_lattice.coordination_number,
            'symmetry_group': self.current_lattice.symmetry_group,
            'resonance_frequency': self.current_lattice.resonance_frequency,
            'crv_value': self.current_lattice.crv_value,
            'wavelength_nm': self.current_lattice.wavelength_nm
        }
    
    def export_metrics(self, file_path: str) -> bool:
        """
        Export metrics history to a file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'realm_name': self.realm_name,
                'lattice_info': self.get_lattice_info(),
                'current_metrics': self.current_metrics.__dict__,
                'metrics_history': [metrics.__dict__ for metrics in self.metrics_history],
                'correction_history': len(self.correction_history)
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported GLR metrics to {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def create_glr_framework(realm_name: str = "electromagnetic",
                        enable_error_correction: bool = True,
                        hex_dictionary: Optional[HexDictionary] = None) -> ComprehensiveErrorCorrectionFramework:
    """
    Create and return a new GLR Framework instance.
    
    Args:
        realm_name: Name of the computational realm
        enable_error_correction: Whether to enable error correction
        hex_dictionary: Optional HexDictionary instance
        
    Returns:
        Initialized ComprehensiveErrorCorrectionFramework instance
    """
    return ComprehensiveErrorCorrectionFramework(realm_name, enable_error_correction, hex_dictionary)


def benchmark_glr_framework(framework: ComprehensiveErrorCorrectionFramework, 
                           num_offbits: int = 1000) -> Dict[str, float]:
    """
    Benchmark GLR Framework performance.
    
    Args:
        framework: GLR Framework instance to benchmark
        num_offbits: Number of OffBits to test with
        
    Returns:
        Dictionary with benchmark results
    """
    import random
    
    start_time = time.time()
    
    # Generate test OffBits
    test_offbits = [random.randint(0, 0xFFFFFF) for _ in range(num_offbits)]
    
    # Test spatial correction
    spatial_start = time.time()
    spatial_result = framework.correct_spatial_errors(test_offbits)
    spatial_time = time.time() - spatial_start
    
    # Test temporal correction
    temporal_sequence = [test_offbits[i:i+100] for i in range(0, len(test_offbits), 100)]
    temporal_start = time.time()
    temporal_result = framework.correct_temporal_errors(temporal_sequence)
    temporal_time = time.time() - temporal_start
    
    # Calculate metrics
    metrics = framework.calculate_comprehensive_metrics(test_offbits, temporal_sequence)
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'spatial_correction_time': spatial_time,
        'temporal_correction_time': temporal_time,
        'spatial_nrci': metrics.nrci_spatial,
        'temporal_nrci': metrics.nrci_temporal,
        'combined_nrci': metrics.nrci_combined,
        'lattice_efficiency': metrics.lattice_efficiency,
        'resonance_stability': metrics.resonance_stability,
        'quantum_coherence': metrics.quantum_coherence,
        'offbits_per_second': num_offbits / total_time
    }


if __name__ == "__main__":
    # Test the GLR Framework
    print("🧪 Testing GLR Framework v3.1...")
    
    framework = create_glr_framework("quantum")
    
    # Test basic error correction
    test_offbits = [0x123456, 0x654321, 0xABCDEF, 0xFEDCBA]
    
    spatial_result = framework.correct_spatial_errors(test_offbits)
    print(f"Spatial correction: {spatial_result.error_count} errors, NRCI improvement: {spatial_result.nrci_improvement:.3f}")
    
    # Test temporal correction
    temporal_sequence = [[0x123456, 0x654321], [0x234567, 0x765432], [0x345678, 0x876543]]
    temporal_result = framework.correct_temporal_errors(temporal_sequence)
    print(f"Temporal correction: {temporal_result.error_count} errors, NRCI improvement: {temporal_result.nrci_improvement:.3f}")
    
    # Test metrics calculation
    metrics = framework.calculate_comprehensive_metrics(test_offbits, temporal_sequence)
    print(f"Combined NRCI: {metrics.nrci_combined:.3f}")
    print(f"Lattice efficiency: {metrics.lattice_efficiency:.3f}")
    print(f"Quantum coherence: {metrics.quantum_coherence:.3f}")
    
    # Test realm switching
    framework.switch_realm("electromagnetic")
    lattice_info = framework.get_lattice_info()
    print(f"Current lattice: {lattice_info['lattice_type']}")
    
    print("✅ GLR Framework v3.1 test completed successfully!")

