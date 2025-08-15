"""
Universal Binary Principle (UBP) Framework v2.0 - Realms Module

This module implements the Platonic Computational Realms system, providing
specialized configurations and behaviors for each of the five core realms
based on Platonic solids.

Author: Euan Craig
Version: 2.0
Date: August 2025
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, asdict
import json
from core import UBPConstants, TriangularProjectionConfig


@dataclass
class RealmMetrics:
    """Comprehensive metrics for a computational realm."""
    spatial_coherence: float
    temporal_coherence: float
    nrci_current: float
    nrci_target: float
    optimization_factor: float
    resonance_strength: float
    lattice_efficiency: float
    error_correction_rate: float


class PlatonicRealm:
    """
    Represents a single Platonic computational realm with its specific
    geometric, resonance, and error correction properties.
    """
    
    def __init__(self, config: TriangularProjectionConfig):
        """
        Initialize a Platonic realm with its configuration.
        
        Args:
            config: TriangularProjectionConfig defining the realm properties
        """
        self.config = config
        self.current_metrics = self._initialize_metrics()
        self.interaction_history = []
        self.last_computation_time = 0.0
        
        print(f"🔷 Initialized {config.name} Realm ({config.platonic_solid})")
    
    def _initialize_metrics(self) -> RealmMetrics:
        """Initialize the realm metrics based on configuration."""
        return RealmMetrics(
            spatial_coherence=self.config.spatial_coherence,
            temporal_coherence=self.config.temporal_coherence,
            nrci_current=self.config.nrci_baseline,
            nrci_target=UBPConstants.NRCI_TARGET,
            optimization_factor=self.config.optimization_factor,
            resonance_strength=self._calculate_resonance_strength(),
            lattice_efficiency=self._calculate_lattice_efficiency(),
            error_correction_rate=0.0  # Will be updated during operations
        )
    
    def _calculate_resonance_strength(self) -> float:
        """
        Calculate the resonance strength based on CRV frequency and wavelength.
        
        Returns:
            Resonance strength factor (0.0 to 1.0)
        """
        # Resonance strength based on frequency-wavelength relationship
        # Higher frequencies with appropriate wavelengths have stronger resonance
        frequency_factor = min(1.0, self.config.crv_frequency / 1e15)  # Normalize to optical range
        wavelength_factor = 1.0 / (1.0 + abs(self.config.wavelength - 650) / 650)  # Optimal around 650nm
        
        return (frequency_factor + wavelength_factor) / 2.0
    
    def _calculate_lattice_efficiency(self) -> float:
        """
        Calculate lattice efficiency based on coordination number and geometry.
        
        Returns:
            Lattice efficiency factor (0.0 to 1.0)
        """
        # Higher coordination numbers generally provide better connectivity
        max_coordination = 20  # Dodecahedron has highest coordination
        coordination_factor = self.config.coordination_number / max_coordination
        
        # Geometric efficiency based on Platonic solid properties
        geometric_factors = {
            "Tetrahedron": 0.6,   # Simple but limited connectivity
            "Cube": 0.8,          # Good balance of simplicity and connectivity
            "Octahedron": 0.9,    # Excellent geometric properties
            "Dodecahedron": 1.0,  # Maximum complexity and connectivity
            "Icosahedron": 0.85   # High connectivity but complex
        }
        
        geometric_factor = geometric_factors.get(self.config.platonic_solid, 0.5)
        
        return (coordination_factor + geometric_factor) / 2.0
    
    def calculate_realm_energy(self, active_offbits: int, time_delta: float, 
                              observer_factor: float = 1.0) -> float:
        """
        Calculate the emergent energy for this realm using the full UBP energy equation.
        
        Args:
            active_offbits: Number of active OffBits in the computation
            time_delta: Time elapsed for the computation step
            observer_factor: Observer intent factor
            
        Returns:
            Calculated emergent energy for this realm
        """
        # UBP Energy Equation Components
        M = active_offbits  # Toggle count
        C = UBPConstants.LIGHT_SPEED  # Processing rate
        
        # Resonance efficiency: R = R₀ · (1 - Hₜ / ln(4))
        R_0 = 0.95
        H_t = 0.05  # Tonal entropy
        R = R_0 * (1 - H_t / np.log(4))
        
        # Structural optimization from realm configuration
        S_opt = self.config.optimization_factor
        
        # Global Coherence Invariant: P_GCI = cos(2π f_avg Δt)
        f_avg = self.config.crv_frequency
        P_GCI = np.cos(2 * UBPConstants.PI * f_avg * time_delta) ** 2
        
        # Observer factor (passed in)
        O_observer = observer_factor
        
        # Zeta-zero alignment constant
        c_infinity = UBPConstants.C_INFINITY
        
        # Spin entropy: I_spin = Σ p_s · log(1/p_s)
        p_s = min(1.0, self.config.crv_frequency / 1e15)  # Normalize frequency as probability
        if p_s <= 0:
            p_s = 1e-10  # Prevent log(0)
        I_spin = p_s * np.log(1 / p_s)
        
        # Matrix interaction term (simplified)
        w_ij = 0.1
        M_ij = 1.0  # XOR toggle operation result
        
        # Full UBP Energy Equation
        E = M * C * (R * S_opt) * P_GCI * O_observer * c_infinity * I_spin * w_ij * M_ij
        
        # Normalize to prevent astronomical numbers
        E_normalized = E / 1e30
        
        return E_normalized
    
    def calculate_nrci(self, signal_data: np.ndarray, target_data: np.ndarray) -> float:
        """
        Calculate the Non-Random Coherence Index (NRCI) for this realm.
        
        Args:
            signal_data: Observed signal array
            target_data: Expected/target signal array
            
        Returns:
            NRCI value between 0.0 and 1.0
        """
        if len(signal_data) != len(target_data):
            raise ValueError("Signal and target data must have the same length")
        
        if len(signal_data) == 0:
            return 0.0
        
        # Calculate RMSE between signal and target
        rmse = np.sqrt(np.mean((signal_data - target_data) ** 2))
        
        # Calculate standard deviation of target
        target_std = np.std(target_data)
        
        if target_std == 0:
            return 1.0 if rmse == 0 else 0.0
        
        # NRCI formula: 1 - (RMSE / σ(target))
        nrci = 1.0 - (rmse / target_std)
        
        # Clamp to [0, 1] range
        nrci = max(0.0, min(1.0, nrci))
        
        # Update current metrics
        self.current_metrics.nrci_current = nrci
        
        return nrci
    
    def apply_glr_correction(self, input_data: np.ndarray, 
                           correction_strength: float = 1.0) -> np.ndarray:
        """
        Apply Golay-Leech-Resonance (GLR) error correction to input data.
        
        Args:
            input_data: Input data array to correct
            correction_strength: Strength of correction (0.0 to 1.0)
            
        Returns:
            Corrected data array
        """
        if len(input_data) == 0:
            return input_data.copy()
        
        # Spatial GLR correction based on lattice structure
        spatial_correction = self._apply_spatial_glr(input_data)
        
        # Temporal GLR correction based on resonance frequency
        temporal_correction = self._apply_temporal_glr(input_data)
        
        # Weighted combination based on realm's spatial/temporal coherence
        spatial_weight = self.config.spatial_coherence
        temporal_weight = self.config.temporal_coherence
        total_weight = spatial_weight + temporal_weight
        
        if total_weight > 0:
            spatial_weight /= total_weight
            temporal_weight /= total_weight
        else:
            spatial_weight = temporal_weight = 0.5
        
        # Combine corrections
        corrected_data = (spatial_weight * spatial_correction + 
                         temporal_weight * temporal_correction)
        
        # Apply correction strength
        final_data = ((1.0 - correction_strength) * input_data + 
                     correction_strength * corrected_data)
        
        # Update error correction rate
        error_reduction = np.mean(np.abs(input_data - final_data))
        self.current_metrics.error_correction_rate = error_reduction
        
        return final_data
    
    def _apply_spatial_glr(self, data: np.ndarray) -> np.ndarray:
        """Apply spatial GLR correction based on lattice geometry."""
        # Spatial correction using coordination number and lattice type
        coordination_factor = self.config.coordination_number / 20.0  # Normalize
        
        # Apply smoothing based on coordination (higher coordination = more smoothing)
        if len(data) > 1:
            smoothed = np.convolve(data, np.ones(3)/3, mode='same')
            return coordination_factor * smoothed + (1 - coordination_factor) * data
        else:
            return data.copy()
    
    def _apply_temporal_glr(self, data: np.ndarray) -> np.ndarray:
        """Apply temporal GLR correction based on resonance frequency."""
        # Temporal correction using CRV frequency
        frequency_factor = min(1.0, self.config.crv_frequency / 1e15)
        
        # Apply frequency-based filtering
        if len(data) > 2:
            # Simple high-pass filter for high frequencies, low-pass for low frequencies
            if frequency_factor > 0.5:
                # High frequency realm - emphasize rapid changes
                filtered = np.gradient(data)
                return data + 0.1 * frequency_factor * filtered
            else:
                # Low frequency realm - smooth out rapid changes
                filtered = np.convolve(data, np.ones(5)/5, mode='same')
                return frequency_factor * filtered + (1 - frequency_factor) * data
        else:
            return data.copy()
    
    def update_metrics(self, computation_time: float, nrci_value: float) -> None:
        """
        Update realm metrics after a computation.
        
        Args:
            computation_time: Time taken for the computation
            nrci_value: Achieved NRCI value
        """
        self.last_computation_time = computation_time
        self.current_metrics.nrci_current = nrci_value
        
        # Update resonance strength based on performance
        if nrci_value > self.current_metrics.nrci_target * 0.9:
            self.current_metrics.resonance_strength *= 1.01  # Slight improvement
        else:
            self.current_metrics.resonance_strength *= 0.99  # Slight degradation
        
        # Clamp resonance strength
        self.current_metrics.resonance_strength = max(0.1, min(1.0, 
            self.current_metrics.resonance_strength))
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information for this realm."""
        return {
            "name": self.config.name,
            "platonic_solid": self.config.platonic_solid,
            "configuration": asdict(self.config),
            "current_metrics": asdict(self.current_metrics),
            "last_computation_time": self.last_computation_time,
            "interaction_count": len(self.interaction_history)
        }


class RealmManager:
    """
    Manages all Platonic computational realms and their interactions.
    
    This class provides the high-level interface for realm operations,
    cross-realm synchronization, and system-wide coherence management.
    """
    
    def __init__(self):
        """Initialize the RealmManager with all five Platonic realms."""
        self.realms = {}
        self.active_realm = None
        self.cross_realm_coherence = {}
        self.global_metrics = {}
        
        self._initialize_realms()
        self._initialize_cross_realm_coherence()
        
        print("🌟 UBP Realm Manager Initialized")
        print(f"   Available Realms: {list(self.realms.keys())}")
    
    def _initialize_realms(self) -> None:
        """Initialize all five Platonic computational realms."""
        # Initialize realms with default configurations to avoid circular dependency
        
        # Electromagnetic Realm (Cubic GLR)
        em_config = TriangularProjectionConfig(
            name="Electromagnetic",
            platonic_solid="Cube",
            coordination_number=6,
            crv_frequency=UBPConstants.CRV_ELECTROMAGNETIC,
            wavelength=UBPConstants.WAVELENGTH_ELECTROMAGNETIC,
            spatial_coherence=0.95,
            temporal_coherence=0.90,
            nrci_baseline=0.85,
            lattice_type="Cubic",
            optimization_factor=1.2
        )
        self.realms["electromagnetic"] = PlatonicRealm(em_config)
        
        # Quantum Realm (Tetrahedral GLR)
        quantum_config = TriangularProjectionConfig(
            name="Quantum",
            platonic_solid="Tetrahedron",
            coordination_number=4,
            crv_frequency=UBPConstants.CRV_QUANTUM,
            wavelength=UBPConstants.WAVELENGTH_QUANTUM,
            spatial_coherence=0.88,
            temporal_coherence=0.85,
            nrci_baseline=0.80,
            lattice_type="Tetrahedral",
            optimization_factor=1.1
        )
        self.realms["quantum"] = PlatonicRealm(quantum_config)
        
        # Gravitational Realm (FCC GLR)
        grav_config = TriangularProjectionConfig(
            name="Gravitational",
            platonic_solid="Octahedron",
            coordination_number=12,
            crv_frequency=UBPConstants.CRV_GRAVITATIONAL,
            wavelength=UBPConstants.WAVELENGTH_GRAVITATIONAL,
            spatial_coherence=0.92,
            temporal_coherence=0.88,
            nrci_baseline=0.82,
            lattice_type="FCC",
            optimization_factor=1.15
        )
        self.realms["gravitational"] = PlatonicRealm(grav_config)
        
        # Biological Realm (H4 120-Cell GLR)
        bio_config = TriangularProjectionConfig(
            name="Biological",
            platonic_solid="Dodecahedron",
            coordination_number=20,
            crv_frequency=UBPConstants.CRV_BIOLOGICAL,
            wavelength=UBPConstants.WAVELENGTH_BIOLOGICAL,
            spatial_coherence=0.85,
            temporal_coherence=0.92,
            nrci_baseline=0.78,
            lattice_type="H4_120_Cell",
            optimization_factor=1.05
        )
        self.realms["biological"] = PlatonicRealm(bio_config)
        
        # Cosmological Realm (Icosahedral GLR)
        cosmo_config = TriangularProjectionConfig(
            name="Cosmological",
            platonic_solid="Icosahedron",
            coordination_number=12,
            crv_frequency=UBPConstants.CRV_COSMOLOGICAL,
            wavelength=UBPConstants.WAVELENGTH_COSMOLOGICAL,
            spatial_coherence=0.75,
            temporal_coherence=0.85,
            nrci_baseline=0.75,
            lattice_type="H3_Icosahedral",
            optimization_factor=1.0
        )
        self.realms["cosmological"] = PlatonicRealm(cosmo_config)
        
        # Nuclear Realm (E8-to-G2 GLR)
        nuclear_config = TriangularProjectionConfig(
            name="Nuclear",
            platonic_solid="E8_Lattice",
            coordination_number=240,  # E8 has 240 roots
            crv_frequency=1.2356e20,  # Zitterbewegung frequency
            wavelength=2.426e-12,     # Compton wavelength
            spatial_coherence=0.95,
            temporal_coherence=0.98,
            nrci_baseline=0.95,
            lattice_type="E8_G2_Lattice",
            optimization_factor=1.2
        )
        self.realms["nuclear"] = PlatonicRealm(nuclear_config)
        
        # Optical Realm (Photonic GLR)
        optical_config = TriangularProjectionConfig(
            name="Optical",
            platonic_solid="Photonic_Crystal",
            coordination_number=6,  # Hexagonal photonic crystal
            crv_frequency=5e14,     # 5×10^14 Hz (600 nm)
            wavelength=600e-9,      # 600 nm
            spatial_coherence=0.99,
            temporal_coherence=0.995,
            nrci_baseline=0.98,
            lattice_type="Hexagonal_Photonic",
            optimization_factor=1.5
        )
        self.realms["optical"] = PlatonicRealm(optical_config)
        
        # Set default active realm
        self.active_realm = "electromagnetic"
    
    def _initialize_cross_realm_coherence(self) -> None:
        """Initialize cross-realm coherence matrix."""
        realm_names = list(self.realms.keys())
        
        for realm1 in realm_names:
            self.cross_realm_coherence[realm1] = {}
            for realm2 in realm_names:
                if realm1 == realm2:
                    self.cross_realm_coherence[realm1][realm2] = 1.0
                else:
                    # Initialize with baseline coherence values
                    self.cross_realm_coherence[realm1][realm2] = 0.5
    
    def get_realm(self, realm_name: str) -> PlatonicRealm:
        """
        Get a specific realm instance.
        
        Args:
            realm_name: Name of the realm to retrieve
            
        Returns:
            PlatonicRealm instance
            
        Raises:
            KeyError: If realm_name is not found
        """
        if realm_name not in self.realms:
            available = list(self.realms.keys())
            raise KeyError(f"Unknown realm '{realm_name}'. Available: {available}")
        
        return self.realms[realm_name]
    
    def set_active_realm(self, realm_name: str) -> None:
        """
        Set the active computational realm.
        
        Args:
            realm_name: Name of the realm to activate
        """
        if realm_name not in self.realms:
            available = list(self.realms.keys())
            raise KeyError(f"Unknown realm '{realm_name}'. Available: {available}")
        
        self.active_realm = realm_name
        print(f"🔄 Active realm set to: {realm_name}")
    
    def get_active_realm(self) -> PlatonicRealm:
        """Get the currently active realm instance."""
        return self.get_realm(self.active_realm)
    
    def get_available_realms(self) -> List[str]:
        """
        Get list of available realm names.
        
        Returns:
            List of realm names
        """
        return list(self.realms.keys())
    
    def is_realm_available(self, realm_name: str) -> bool:
        """
        Check if a realm is available.
        
        Args:
            realm_name: Name of the realm to check
            
        Returns:
            True if realm exists, False otherwise
        """
        return realm_name in self.realms
    
    def get_realm_config(self, realm_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific realm.
        
        Args:
            realm_name: Name of the realm
            
        Returns:
            Dictionary containing realm configuration
        """
        if realm_name not in self.realms:
            raise ValueError(f"Realm '{realm_name}' not found")
        
        realm = self.realms[realm_name]
        return {
            'name': realm.config.name,
            'platonic_solid': realm.config.platonic_solid,
            'coordination_number': realm.config.coordination_number,
            'crv_frequency': realm.config.crv_frequency,
            'wavelength': realm.config.wavelength,
            'spatial_coherence': realm.config.spatial_coherence,
            'temporal_coherence': realm.config.temporal_coherence,
            'nrci_baseline': realm.config.nrci_baseline,
            'lattice_type': realm.config.lattice_type,
            'optimization_factor': realm.config.optimization_factor
        }
    
    def get_cross_realm_coherence(self, realm1: str, realm2: str,
                                      data1: np.ndarray, data2: np.ndarray) -> float:
        """
        Calculate coherence between two realms based on their data.
        
        Args:
            realm1_name: Name of the first realm
            realm2_name: Name of the second realm
            data1: Data array from first realm
            data2: Data array from second realm
            
        Returns:
            Cross-realm coherence value (0.0 to 1.0)
        """
        if len(data1) != len(data2) or len(data1) == 0:
            return 0.0
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(data1, data2)[0, 1]
        
        # Handle NaN case
        if np.isnan(correlation):
            correlation = 0.0
        
        # Convert correlation to coherence (absolute value, scaled to [0,1])
        coherence = abs(correlation)
        
        # Update cross-realm coherence matrix
        self.cross_realm_coherence[realm1_name][realm2_name] = coherence
        self.cross_realm_coherence[realm2_name][realm1_name] = coherence
        
        return coherence
    
    def get_global_coherence(self) -> float:
        """
        Calculate the global coherence across all realms.
        
        Returns:
            Global coherence value (0.0 to 1.0)
        """
        total_coherence = 0.0
        pair_count = 0
        
        realm_names = list(self.realms.keys())
        
        for i, realm1 in enumerate(realm_names):
            for j, realm2 in enumerate(realm_names):
                if i < j:  # Avoid double counting
                    coherence = self.cross_realm_coherence[realm1][realm2]
                    total_coherence += coherence
                    pair_count += 1
        
        if pair_count == 0:
            return 0.0
        
        return total_coherence / pair_count
    
    def synchronize_realms(self, target_coherence: float = 0.95) -> Dict[str, float]:
        """
        Attempt to synchronize all realms to achieve target coherence.
        
        Args:
            target_coherence: Target coherence level to achieve
            
        Returns:
            Dictionary of achieved coherence values between realm pairs
        """
        print(f"🔄 Synchronizing realms (target coherence: {target_coherence:.3f})")
        
        realm_names = list(self.realms.keys())
        achieved_coherence = {}
        
        # Generate synthetic synchronization signals for each realm
        sync_signals = {}
        for realm_name in realm_names:
            realm = self.realms[realm_name]
            frequency = realm.config.crv_frequency
            
            # Create a synchronization signal based on realm's CRV
            t = np.linspace(0, UBPConstants.CSC_PERIOD, 1000)
            signal = np.cos(2 * np.pi * frequency * t)
            sync_signals[realm_name] = signal
        
        # Calculate cross-realm coherence for all pairs
        for i, realm1 in enumerate(realm_names):
            for j, realm2 in enumerate(realm_names):
                if i < j:
                    coherence = self.calculate_cross_realm_coherence(
                        realm1, realm2, 
                        sync_signals[realm1], sync_signals[realm2]
                    )
                    achieved_coherence[f"{realm1}-{realm2}"] = coherence
        
        global_coherence = self.get_global_coherence()
        achieved_coherence["global"] = global_coherence
        
        print(f"   Global coherence achieved: {global_coherence:.6f}")
        
        return achieved_coherence
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive status for the entire realm system."""
        realm_statuses = {}
        for realm_name, realm in self.realms.items():
            realm_statuses[realm_name] = realm.get_status()
        
        return {
            "active_realm": self.active_realm,
            "global_coherence": self.get_global_coherence(),
            "cross_realm_coherence": self.cross_realm_coherence,
            "realm_statuses": realm_statuses,
            "total_realms": len(self.realms)
        }


if __name__ == "__main__":
    # Test the Realms module
    print("="*60)
    print("UBP REALMS MODULE TEST")
    print("="*60)
    
    # Create realm manager
    rm = RealmManager()
    
    # Test realm operations
    quantum_realm = rm.get_realm("quantum")
    print(f"\nQuantum Realm Status:")
    print(f"  Platonic Solid: {quantum_realm.config.platonic_solid}")
    print(f"  CRV Frequency: {quantum_realm.config.crv_frequency:.6f}")
    print(f"  Resonance Strength: {quantum_realm.current_metrics.resonance_strength:.6f}")
    
    # Test NRCI calculation
    test_signal = np.random.normal(0, 1, 100)
    test_target = np.sin(np.linspace(0, 2*np.pi, 100))
    nrci = quantum_realm.calculate_nrci(test_signal, test_target)
    print(f"  Test NRCI: {nrci:.6f}")
    
    # Test GLR correction
    corrected_signal = quantum_realm.apply_glr_correction(test_signal)
    print(f"  GLR Correction Applied: {len(corrected_signal)} points")
    
    # Test realm synchronization
    sync_results = rm.synchronize_realms()
    print(f"\nSynchronization Results:")
    for pair, coherence in sync_results.items():
        print(f"  {pair}: {coherence:.6f}")
    
    # Test energy calculation
    energy = quantum_realm.calculate_realm_energy(
        active_offbits=1000, 
        time_delta=0.001, 
        observer_factor=1.2
    )
    print(f"\nRealm Energy: {energy:.6e}")
    
    print("\n✅ Realms module test completed successfully!")

