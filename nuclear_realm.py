```python
"""
Universal Binary Principle (UBP) Framework v3.1.1 - Nuclear Realm Module
Author: Euan Craig, New Zealand
Date: 18 August 2025

This module implements the complete Nuclear Realm with E8-to-G2 symmetry lattice,
Zitterbewegung modeling, CARFE integration, and NMR validation capabilities.

The Nuclear realm operates at frequencies from 10^16 to 10^20 Hz, with special
focus on Zitterbewegung frequency (1.2356×10^20 Hz) and NMR validation at 600 MHz.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import math
from scipy.special import factorial, gamma
from scipy.linalg import expm
import json

# Import configuration for consistency
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from ubp_config import get_config, UBPConfig, RealmConfig

# Import UBPConstants directly for raw values
from system_constants import UBPConstants # Corrected import

# Import Bitfield and OffBit from the bits module
from bits import Bitfield, OffBit 


@dataclass
class NuclearRealmMetrics:
    """Comprehensive metrics for Nuclear Realm operations."""
    zitterbewegung_frequency: float
    e8_g2_coherence: float
    carfe_stability: float
    nmr_validation_score: float
    nuclear_binding_energy: float
    spin_orbit_coupling: float
    magnetic_moment: float
    quadrupole_moment: float
    hyperfine_splitting: float
    isotope_shift: float


@dataclass
class E8G2LatticeStructure:
    """E8-to-G2 lattice structure for nuclear realm operations.
    
    The 'simple_roots' here represent a set of basis vectors used for projection
    in the E8-to-G2 coherence calculation. While they are a common choice for
    E8 simple roots in R^8, their specific ordering and interpretation should
    be considered in the context of the coherence calculation's heuristic nature.
    The Cartan matrix, however, is intended to be the mathematically correct
    Cartan matrix for E8.
    """
    simple_roots: np.ndarray  # Used for projection in coherence calculation
    cartan_matrix: np.ndarray
    fundamental_weights: np.ndarray
    killing_form_signature: Tuple[int, int]
    e8_dimension: int = 248  # E8 Lie algebra dimension
    g2_dimension: int = 14   # G2 Lie algebra dimension
    weyl_group_order: int = 696729600


@dataclass
class ZitterbewegungState:
    """State representation for Zitterbewegung modeling."""
    spin_state: complex
    position_uncertainty: float
    momentum_uncertainty: float
    frequency: float # Populated from config/UBPConstants
    amplitude: float = 1.0
    phase: float = 0.0
    compton_wavelength: float # Populated from config/UBPConstants


@dataclass
class CARFEParameters:
    """Parameters for Cykloid Adelic Recursive Expansive Field Equation."""
    adelic_prime_base: List[int]
    recursion_depth: int = 10
    expansion_coefficient: float = 1.618034  # Golden ratio
    field_strength: float = 1.0
    temporal_coupling: float = 0.318309886  # 1/π
    convergence_threshold: float = 1e-12


class NuclearRealm:
    """
    Complete Nuclear Realm implementation for the UBP Framework.
    
    This class provides nuclear physics modeling with E8-to-G2 symmetry,
    Zitterbewegung dynamics, CARFE field equations, and NMR validation.
    """
    
    def __init__(self, bitfield: Optional[Bitfield] = None):
        """
        Initialize the Nuclear Realm.
        
        Args:
            bitfield: Optional Bitfield instance for nuclear operations
        """
        self.bitfield = bitfield
        self.config = get_config() # Get the global UBPConfig instance
        
        # Nuclear realm parameters from config
        nuclear_realm_cfg = self.config.get_realm_config('nuclear')
        self.frequency_range = nuclear_realm_cfg.frequency_range if nuclear_realm_cfg else (1e16, 1e20)
        self.zitterbewegung_freq = nuclear_realm_cfg.main_crv if nuclear_realm_cfg else UBPConstants.CRV_NUCLEAR_BASE
        self.nmr_validation_freq = 600e6     # Hz (600 MHz) - Hardcoded as a specific test frequency
        self.nmr_field_strength = 0.5        # Tesla - Hardcoded as a specific test value
        
        # Initialize E8-to-G2 lattice structure
        self.e8_g2_lattice = self._initialize_e8_g2_lattice()
        
        # Initialize constants from UBPConfig for consistency
        self.nuclear_constants = {
            'fine_structure': self.config.constants.FINE_STRUCTURE_CONSTANT,
            'nuclear_magneton': self.config.constants.NUCLEAR_MAGNETON,
            'proton_gyromagnetic': self.config.constants.PROTON_GYROMAGNETIC,
            'neutron_gyromagnetic': self.config.constants.NEUTRON_GYROMAGNETIC,
            'deuteron_binding': self.config.constants.DEUTERON_BINDING_ENERGY,
            'planck_reduced': self.config.constants.PLANCK_REDUCED,
            'electron_mass': self.config.constants.ELECTRON_MASS,
            'proton_mass': self.config.constants.PROTON_MASS,
            'neutron_mass': self.config.constants.NEUTRON_MASS,
            'speed_of_light': self.config.constants.SPEED_OF_LIGHT
        }

        # Initialize Zitterbewegung modeling with values from constants
        hbar = self.nuclear_constants['planck_reduced']
        m_e = self.nuclear_constants['electron_mass']
        c = self.nuclear_constants['speed_of_light']
        electron_compton_wavelength = hbar / (m_e * c)

        self.zitterbewegung_state = ZitterbewegungState(
            spin_state=1+0j,
            frequency=self.zitterbewegung_freq,
            position_uncertainty=electron_compton_wavelength,
            momentum_uncertainty=hbar / (2 * electron_compton_wavelength),
            compton_wavelength=electron_compton_wavelength
        )
        
        # Initialize CARFE parameters
        self.carfe_params = CARFEParameters(
            adelic_prime_base=[2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        )
        
        # Performance metrics
        self.metrics = NuclearRealmMetrics(
            zitterbewegung_frequency=self.zitterbewegung_freq,
            e8_g2_coherence=0.0,
            carfe_stability=0.0,
            nmr_validation_score=0.0,
            nuclear_binding_energy=0.0,
            spin_orbit_coupling=0.0,
            magnetic_moment=0.0,
            quadrupole_moment=0.0,
            hyperfine_splitting=0.0,
            isotope_shift=0.0
        )
        
        print(f"🔬 Nuclear Realm Initialized")
        print(f"   Frequency Range: {self.frequency_range[0]:.1e} - {self.frequency_range[1]:.1e} Hz")
        print(f"   Zitterbewegung: {self.zitterbewegung_freq:.4e} Hz")
        print(f"   E8 Dimension: {self.e8_g2_lattice.e8_dimension}")
        print(f"   G2 Dimension: {self.e8_g2_lattice.g2_dimension}")
    
    def _initialize_e8_g2_lattice(self) -> E8G2LatticeStructure:
        """Initialize the E8-to-G2 lattice structure."""
        
        # Canonical E8 simple roots (vectors for projection in R^8).
        # These are a commonly used set of simple roots for E8 (derived from specific constructions).
        e8_simple_roots_vectors = np.array([
            [0, 0, 0, 0, 0, 0, 1, -1],   # alpha_1
            [0, 0, 0, 0, 0, 0, 1, 1],    # alpha_2
            [0, 0, 0, 0, 0, 1, -1, 0],   # alpha_3
            [0, 0, 0, 0, 1, -1, 0, 0],   # alpha_4
            [0, 0, 0, 1, -1, 0, 0, 0],   # alpha_5
            [0, 0, 1, -1, 0, 0, 0, 0],   # alpha_6
            [0, 1, -1, 0, 0, 0, 0, 0],   # alpha_7
            [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, 0.5] # alpha_8, connecting root
        ])
        
        # The Cartan matrix for E8. This is the canonical, correct matrix for E8.
        # Node ordering (simplified representation of Dynkin diagram):
        #   1-2-3-4-5-6-7 (A7 sub-diagram)
        #       |
        #       8 (node 8 branches off from node 4)
        e8_cartan = np.array([
            [ 2, -1,  0,  0,  0,  0,  0,  0],
            [-1,  2, -1,  0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0,  0,  0,  0],
            [ 0,  0, -1,  2, -1,  0,  0, -1], # Node 4 connects to node 8
            [ 0,  0,  0, -1,  2, -1,  0,  0],
            [ 0,  0,  0,  0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0, -1,  2,  0],
            [ 0,  0,  0, -1,  0,  0,  0,  2]  # Node 8 connects to node 4
        ])
        
        # Fundamental weights for E8 (calculated from inverse of Cartan matrix)
        e8_fundamental_weights = np.linalg.inv(e8_cartan.T)
        
        return E8G2LatticeStructure(
            e8_dimension=248,
            g2_dimension=14,
            simple_roots=e8_simple_roots_vectors, 
            cartan_matrix=e8_cartan,
            weyl_group_order=696729600,  # |W(E8)|
            fundamental_weights=e8_fundamental_weights,
            killing_form_signature=(8, 0)  # E8 is positive definite
        )
    
    def calculate_zitterbewegung_dynamics(self, time_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Calculate Zitterbewegung dynamics for given time array.
        
        Args:
            time_array: Array of time values (seconds)
            
        Returns:
            Dictionary containing position, velocity, and spin dynamics
        """
        freq = self.zitterbewegung_state.frequency
        omega = 2 * np.pi * freq
        
        # Zitterbewegung position oscillation
        # x(t) = x₀ + (ħ/2mc) * sin(2mct/ħ)
        hbar = self.nuclear_constants['planck_reduced']
        compton_wavelength = self.zitterbewegung_state.compton_wavelength
        zitter_amplitude = compton_wavelength / 2
        
        position = zitter_amplitude * np.sin(omega * time_array)
        velocity = zitter_amplitude * omega * np.cos(omega * time_array)
        
        # Spin dynamics (Pauli matrices evolution) - simplified
        spin_x = np.cos(omega * time_array / 2)
        spin_y = np.sin(omega * time_array / 2)
        spin_z = np.cos(omega * time_array)
        
        # Energy oscillation
        energy = hbar * omega * (1 + np.cos(omega * time_array)) / 2
        
        return {
            'position': position,
            'velocity': velocity,
            'spin_x': spin_x,
            'spin_y': spin_y,
            'spin_z': spin_z,
            'energy': energy,
            'frequency': freq,
            'amplitude': zitter_amplitude
        }
    
    def solve_carfe_equation(self, initial_field: np.ndarray, time_steps: int = 100) -> Dict[str, Any]:
        """
        Solve the Cykloid Adelic Recursive Expansive Field Equation (CARFE).
        
        Args:
            initial_field: Initial field configuration
            time_steps: Number of temporal evolution steps
            
        Returns:
            Dictionary containing field evolution and stability metrics
        """
        params = self.carfe_params
        field_evolution = [initial_field.copy()]
        stability_metrics = []
        
        dt = params.temporal_coupling / time_steps
        
        for step in range(time_steps):
            current_field = field_evolution[-1]
            
            # CARFE recursive expansion
            # F(t+dt) = F(t) + φ * ∇²F(t) + Σ(p-adic corrections)
            
            # Laplacian operator (simplified for 1D field)
            if len(current_field) > 2:
                laplacian = (current_field[2:] - 2*current_field[1:-1] + current_field[:-2])
                # Pad laplacian to match current_field shape, typically with zeros at boundaries
                padded_laplacian = np.pad(laplacian, (1, 1), 'constant')
            else:
                padded_laplacian = np.zeros_like(current_field) # Completed this line
            
            # P-adic corrections (simplified heuristic)
            p_adic_correction = np.zeros_like(current_field)
            for p in params.adelic_prime_base:
                # Simple p-adic norm inspired correction
                # For a real number x, |x|_p = p^(-v_p(x)) where v_p(x) is the p-adic valuation.
                # Here, we'll use a heuristic for real fields.
                # Example: |x|_p = 1 / (p^k) if x is divisible by p^k but not p^(k+1)
                # This is hard for arbitrary floats. A simpler approach is needed for simulation.
                # Heuristic: apply small perturbation based on p-adic prime and field value.
                p_adic_correction += (current_field % p) / p * 0.01 * params.field_strength # Very simplified
            
            # Combine terms
            next_field = current_field + \
                         params.expansion_coefficient * padded_laplacian * dt + \
                         p_adic_correction * dt * params.temporal_coupling
            
            field_evolution.append(next_field)
            
            # Check stability (e.g., L2 norm convergence)
            current_norm = np.linalg.norm(current_field)
            next_norm = np.linalg.norm(next_field)
            
            if current_norm > 0:
                stability = abs(next_norm - current_norm) / current_norm
            else:
                stability = next_norm # If current is zero, stability depends on next
            
            stability_metrics.append(stability)
            
            # Check for convergence
            if stability < params.convergence_threshold and step > 0:
                # print(f"CARFE converged at step {step}")
                break
        
        # Calculate overall CARFE stability
        overall_stability = np.mean(stability_metrics) if stability_metrics else 0.0
        
        return {
            'field_evolution': [f.tolist() for f in field_evolution],
            'stability_metrics': stability_metrics,
            'overall_stability': overall_stability,
            'final_field': field_evolution[-1].tolist()
        }

    def calculate_e8_g2_coherence(self, nuclear_states: List[np.ndarray]) -> float:
        """
        Calculate E8-to-G2 coherence for nuclear states.
        
        Args:
            nuclear_states: List of nuclear state vectors (e.g., from OffBits)
            
        Returns:
            Coherence score (0 to 1)
        """
        if not nuclear_states:
            return 0.0
        
        # Project states onto E8 simple roots (simplified)
        projected_vectors = []
        for state in nuclear_states:
            # Pad or truncate state to match E8 simple roots dimension (8)
            padded_state = np.pad(state, (0, max(0, 8 - len(state))), 'constant')[:8]
            
            # Project onto basis
            proj_e8 = np.dot(padded_state, self.e8_g2_lattice.simple_roots.T)
            projected_vectors.append(proj_e8.flatten()) # Ensure it's flat after projection
            
        if not projected_vectors:
            return 0.0

        # Calculate coherence based on variance of projected vectors
        # A more complex method would involve distances in the lattice.
        all_projected_values = np.concatenate(projected_vectors)
        if len(all_projected_values) == 0:
            return 0.0

        variance = np.var(all_projected_values)
        mean_abs_value = np.mean(np.abs(all_projected_values))

        # Coherence is inversely proportional to normalized variance
        if mean_abs_value > 1e-10:
            coherence = 1.0 / (1.0 + variance / mean_abs_value)
        else:
            coherence = 1.0 # If all values are zero, it's perfectly coherent conceptually
        
        return min(1.0, max(0.0, coherence))

    def simulate_nmr_response(self, nucleus_spin: float, external_field: float, 
                             frequency_array: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Simulate Nuclear Magnetic Resonance (NMR) response.
        
        Args:
            nucleus_spin: Spin of the nucleus (e.g., 0.5 for proton)
            external_field: External magnetic field strength (Tesla)
            frequency_array: Array of frequencies for simulation (Hz)
            
        Returns:
            Dictionary with NMR spectrum and related parameters
        """
        # Gyromagnetic ratio depends on nucleus type
        if nucleus_spin == 0.5: # Proton
            gamma = self.nuclear_constants['proton_gyromagnetic']
            nucleus_name = "proton"
        elif nucleus_spin == 1.0: # Deuteron
            gamma = self.nuclear_constants['proton_gyromagnetic'] / 6.514 # Approx deuteron ratio
            nucleus_name = "deuteron"
        else:
            gamma = 0.0 # No resonance if spin is 0 or unknown
            nucleus_name = "unknown"

        # Larmor frequency
        larmor_frequency = (gamma * external_field) / (2 * np.pi)
        
        # Simulate Lorentzian line shape for resonance
        # A(f) = A_0 / (1 + ((f - f_0) / (width/2))^2)
        line_width = 100.0 # Hz (example)
        amplitude = 1.0
        
        # Ensure division by zero is handled if line_width is too small
        denom = (1 + ((frequency_array - larmor_frequency) / (line_width / 2))**2)
        if np.any(denom == 0):
            # Add a small epsilon to denominator to prevent division by zero
            denom = denom + np.finfo(float).eps
        
        spectrum = amplitude / denom
        
        # Calculate signal-to-noise ratio (simplified)
        snr = np.max(spectrum) / (np.mean(np.abs(spectrum - np.mean(spectrum))) + 1e-10) # Approx noise level
        
        return {
            'nucleus': nucleus_name,
            'larmor_frequency': larmor_frequency,
            'spectrum': spectrum,
            'frequency_array': frequency_array,
            'signal_to_noise_ratio': snr,
            'external_field': external_field
        }
    
    def calculate_nuclear_properties(self, atomic_number: int, mass_number: int, 
                                    excitation_energy: float = 0.0) -> Dict[str, float]:
        """
        Calculate various nuclear properties.
        
        Args:
            atomic_number: Z (number of protons)
            mass_number: A (number of nucleons)
            excitation_energy: Excitation energy of the nucleus (MeV)
            
        Returns:
            Dictionary of nuclear properties
        """
        # Calculate binding energy using Weizsäcker formula (semi-empirical mass formula)
        # B(A, Z) = a_v A - a_s A^(2/3) - a_c Z(Z-1)/A^(1/3) - a_a (A-2Z)^2/A + a_p A^(-1/2)
        
        # Coefficients (MeV)
        a_v = 15.67   # Volume term
        a_s = 17.23   # Surface term
        a_c = 0.714   # Coulomb term
        a_a = 23.285  # Asymmetry term
        
        a_p = 12.0 # Pairing term (positive for even-even, negative for odd-odd, zero for odd-even)
        if mass_number % 2 == 0 and atomic_number % 2 == 0: # Even-Even
            pairing_term = a_p * mass_number**(-0.5)
        elif mass_number % 2 != 0 and atomic_number % 2 != 0: # Odd-Odd
            pairing_term = -a_p * mass_number**(-0.5)
        else: # Odd-Even or Even-Odd
            pairing_term = 0.0
            
        binding_energy = (
            a_v * mass_number - 
            a_s * mass_number**(2/3) - 
            a_c * atomic_number * (atomic_number - 1) / mass_number**(1/3) -
            a_a * (mass_number - 2 * atomic_number)**2 / mass_number +
            pairing_term
        )
        
        # Spin-orbit coupling (simplified heuristic)
        # Directly proportional to excitation_energy and nuclear spin properties
        spin_orbit_coupling = excitation_energy * 0.1 * atomic_number / mass_number
        
        # Magnetic dipole moment (simplified, proportional to spin and gyromagnetic ratio)
        # Assume proton-like behavior dominates for simplicity
        magnetic_moment = self.nuclear_constants['nuclear_magneton'] * atomic_number * nucleus_spin / 0.5 
        
        # Electric quadrupole moment (simplified, related to deformation)
        quadrupole_moment = 0.05 * mass_number**(2/3) * np.sin(excitation_energy / 10) # Heuristic
        
        # Hyperfine splitting (simplified, related to magnetic moment and electron density)
        hyperfine_splitting = magnetic_moment * self.nuclear_constants['fine_structure'] * 1e-6 # MHz
        
        # Isotope shift (simplified, related to mass difference and nuclear radius)
        isotope_shift = (mass_number - atomic_number) * 1e-12 # nm
        
        return {
            'binding_energy_per_nucleon_MeV': binding_energy / mass_number,
            'total_binding_energy_MeV': binding_energy,
            'spin_orbit_coupling_MeV': spin_orbit_coupling,
            'magnetic_moment_nuclear_magnetons': magnetic_moment / self.nuclear_constants['nuclear_magneton'],
            'quadrupole_moment_barns': quadrupole_moment,
            'hyperfine_splitting_MHz': hyperfine_splitting,
            'isotope_shift_nm': isotope_shift
        }
    
    def run_nuclear_computation(self, nuclear_data: List[Dict[str, Any]], 
                               computation_type: str = 'full') -> Dict[str, Any]:
        """
        Run comprehensive nuclear realm computation, updating internal metrics.
        
        Args:
            nuclear_data: List of dictionaries, each containing 'atomic_number', 'mass_number', 'spin', 'field' etc.
            computation_type: Type of computation ('zitter', 'carfe', 'nmr', 'properties', 'full')
            
        Returns:
            Dictionary containing computation results and updated metrics
        """
        results = {
            'computation_type': computation_type,
            'nuclear_entries_processed': len(nuclear_data)
        }
        
        # Default/Initial values for metrics update
        e8_g2_coherence_avg = 0.0
        carfe_stability_avg = 0.0
        nmr_validation_score_avg = 0.0
        nuclear_binding_energy_avg = 0.0
        spin_orbit_coupling_avg = 0.0

        num_valid_entries = 0

        for entry in nuclear_data:
            atomic_num = entry.get('atomic_number')
            mass_num = entry.get('mass_number')
            nucleus_spin = entry.get('nucleus_spin')
            initial_field = entry.get('initial_field')
            
            if atomic_num is None or mass_num is None:
                continue # Skip invalid entries
            
            num_valid_entries += 1

            if computation_type in ['zitter', 'full']:
                # Zitterbewegung dynamics
                time_array = np.linspace(0, 1e-20, 100) # Short time scale
                zitter_results = self.calculate_zitterbewegung_dynamics(time_array)
                # results.setdefault('zitterbewegung_dynamics', []).append(zitter_results)
                # No direct metric for this, mostly for visualization/simulation
            
            if computation_type in ['carfe', 'full'] and initial_field is not None:
                # CARFE equation
                carfe_results = self.solve_carfe_equation(np.array(initial_field), time_steps=50)
                # results.setdefault('carfe_results', []).append(carfe_results)
                carfe_stability_avg += carfe_results['overall_stability']
            
            if computation_type in ['nmr', 'full'] and nucleus_spin is not None and entry.get('nmr_field') is not None:
                # NMR response
                nmr_freq_array = np.linspace(self.nmr_validation_freq * 0.99, self.nmr_validation_freq * 1.01, 200)
                nmr_results = self.simulate_nmr_response(nucleus_spin, entry['nmr_field'], nmr_freq_array)
                # results.setdefault('nmr_results', []).append(nmr_results)
                # Validate NMR based on peak presence
                if nmr_results['signal_to_noise_ratio'] > 10: # Simple threshold
                    nmr_validation_score_avg += 1.0
                else:
                    nmr_validation_score_avg += 0.0 # No clear peak detected
            
            if computation_type in ['properties', 'full']:
                # Nuclear properties
                excitation_energy = entry.get('excitation_energy', 0.0)
                nuclear_props = self.calculate_nuclear_properties(atomic_num, mass_num, excitation_energy)
                # results.setdefault('nuclear_properties', []).append(nuclear_props)
                nuclear_binding_energy_avg += nuclear_props.get('binding_energy_per_nucleon_MeV', 0.0)
                spin_orbit_coupling_avg += nuclear_props.get('spin_orbit_coupling_MeV', 0.0)

            # E8-G2 coherence needs a set of states, not just one.
            # For simplicity, calculate per entry if each entry is a 'state'.
            # A more robust approach would collect states and run this once.
            # Assuming `entry` contains `state_vector`
            state_vector_for_e8g2 = entry.get('state_vector', np.random.rand(8)) # Generate random if not provided
            e8_g2_coherence_avg += self.calculate_e8_g2_coherence([state_vector_for_e8g2])

        # Update aggregated metrics
        if num_valid_entries > 0:
            self.metrics.e8_g2_coherence = e8_g2_coherence_avg / num_valid_entries
            self.metrics.carfe_stability = carfe_stability_avg / num_valid_entries
            self.metrics.nmr_validation_score = nmr_validation_score_avg / num_valid_entries
            self.metrics.nuclear_binding_energy = nuclear_binding_energy_avg / num_valid_entries
            self.metrics.spin_orbit_coupling = spin_orbit_coupling_avg / num_valid_entries
        
        results['metrics_snapshot'] = self.metrics.__dict__
        return results
    
    def get_nuclear_metrics(self) -> NuclearRealmMetrics:
        """Get current nuclear realm metrics."""
        return self.metrics
    
    def validate_nuclear_realm(self) -> Dict[str, Any]:
        """
        Comprehensive validation of nuclear realm implementation.
        
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'realm_name': 'Nuclear',
            'zitterbewegung_freq': self.zitterbewegung_freq,
            'e8_dimension': self.e8_g2_lattice.e8_dimension,
            'g2_dimension': self.e8_g2_lattice.g2_dimension
        }
        
        # Example test data: Deuterium, Helium-4, Oxygen-16
        test_nuclear_data = [
            {'atomic_number': 1, 'mass_number': 2, 'nucleus_spin': 1.0, 
             'initial_field': [0.1, 0.2, 0.3], 'nmr_field': 0.5, 'excitation_energy': 0.0}, # Deuterium
            {'atomic_number': 2, 'mass_number': 4, 'nucleus_spin': 0.0, 
             'initial_field': [0.5, 0.4, 0.6], 'nmr_field': 0.0, 'excitation_energy': 0.0}, # Helium-4
            {'atomic_number': 8, 'mass_number': 16, 'nucleus_spin': 0.0, 
             'initial_field': [0.8, 0.7, 0.9], 'nmr_field': 0.0, 'excitation_energy': 0.1} # Oxygen-16
        ]
        
        computation_results = self.run_nuclear_computation(test_nuclear_data, 'full')
        validation_results.update(computation_results)
        
        # Validation criteria
        metrics = self.metrics
        validation_criteria = {
            'e8_g2_coherence_positive': metrics.e8_g2_coherence > 0.1,
            'carfe_stability_achieved': metrics.carfe_stability < 0.1, # Expect low value for stability
            'nmr_validation_possible': metrics.nmr_validation_score > 0.0, # At least one validation successful
            'binding_energy_realistic': metrics.nuclear_binding_energy > 5.0 # MeV/nucleon for stable nuclei
        }
        
        validation_results['validation_criteria'] = validation_criteria
        validation_results['overall_valid'] = all(validation_criteria.values())
        
        return validation_results
