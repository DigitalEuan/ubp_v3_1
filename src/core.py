"""
Universal Binary Principle (UBP) Framework v2.0 - Core Module

This module provides the foundational constants, mathematical definitions,
and core framework integration for the UBP computational system.

Author: Euan Craig
Version: 2.0
Date: August 2025
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass


class UBPConstants:
    """
    Core mathematical and physical constants for the UBP framework.
    
    These constants are derived from the fundamental mathematical relationships
    that govern the Universal Binary Principle across all computational realms.
    """
    
    # Mathematical Constants
    PI = np.pi
    E = np.e
    PHI = (1 + np.sqrt(5)) / 2  # Golden ratio
    
    # UBP-Specific Constants
    LIGHT_SPEED = 299792458  # Processing rate (toggles/s)
    SPEED_OF_LIGHT = 299792458  # Alias for compatibility
    C_INFINITY = 24 * (1 + PHI)  # ≈ 38.83281573
    
    # Core Resonance Values (CRVs) - Realm-specific toggle probabilities
    CRV_QUANTUM = E / 12  # ≈ 0.2265234857
    CRV_ELECTROMAGNETIC = PI  # 3.141593
    CRV_GRAVITATIONAL = 100.0
    CRV_BIOLOGICAL = 10.0
    CRV_COSMOLOGICAL = PI ** PHI  # ≈ 0.83203682
    CRV_NUCLEAR = 1.2356e20  # Zitterbewegung frequency
    CRV_OPTICAL = 5e14  # 600 nm frequency
    
    # Wavelengths (nm)
    WAVELENGTH_QUANTUM = 655
    WAVELENGTH_ELECTROMAGNETIC = 635
    WAVELENGTH_GRAVITATIONAL = 1000
    WAVELENGTH_BIOLOGICAL = 700
    WAVELENGTH_COSMOLOGICAL = 800
    WAVELENGTH_OPTICAL = 600
    
    # UBP Energy Equation Constants
    R0 = 0.95  # Base resonance efficiency
    HT = 0.05  # Harmonic threshold
    
    # Temporal Constants
    CSC_PERIOD = 1 / PI  # Coherent Synchronization Cycle ≈ 0.318309886 s
    TAUTFLUENCE_TIME = 2.117e-15  # seconds
    
    # Error Correction Thresholds
    NRCI_TARGET = 0.999999  # Target Non-random Coherence Index
    COHERENCE_THRESHOLD = 0.95  # Minimum coherence for realm interactions
    COHERENCE_PRESSURE_MIN = 0.8  # Minimum coherence pressure


@dataclass
class TriangularProjectionConfig:
    """
    Configuration for a specific computational realm in the UBP framework.
    
    Each realm is defined by its Platonic solid geometry, resonance properties,
    and error correction parameters.
    """
    name: str
    platonic_solid: str
    coordination_number: int
    crv_frequency: float
    wavelength: float
    spatial_coherence: float
    temporal_coherence: float
    nrci_baseline: float
    lattice_type: str
    optimization_factor: float


class UBPFramework:
    """
    Main framework integration class for the Universal Binary Principle.
    
    This class coordinates all UBP subsystems and provides the primary
    interface for computational operations across multiple realms.
    """
    
    def __init__(self, bitfield_dimensions: Tuple[int, ...] = (32, 32, 32, 4, 2, 2)):
        """
        Initialize the UBP Framework with specified Bitfield dimensions.
        
        Args:
            bitfield_dimensions: 6D tuple defining the Bitfield structure
                Default: (32, 32, 32, 4, 2, 2) for notebook testing
                Production: (170, 170, 170, 5, 2, 2) for full system
        """
        self.bitfield_dimensions = bitfield_dimensions
        self.realms = self._initialize_platonic_realms()
        self.current_realm = "electromagnetic"  # Default realm
        self.observer_intent = 1.0  # Neutral observer state
        
        print(f"✅ UBP Framework v2.0 Initialized")
        print(f"   Bitfield Dimensions: {bitfield_dimensions}")
        print(f"   Available Realms: {list(self.realms.keys())}")
    
    def _initialize_platonic_realms(self) -> Dict[str, TriangularProjectionConfig]:
        """
        Initialize the five core Platonic computational realms.
        
        Returns:
            Dictionary mapping realm names to their configurations
        """
        realms = {
            "quantum": TriangularProjectionConfig(
                name="Quantum",
                platonic_solid="Tetrahedron",
                coordination_number=4,
                crv_frequency=UBPConstants.CRV_QUANTUM,
                wavelength=UBPConstants.WAVELENGTH_QUANTUM,
                spatial_coherence=0.7465,
                temporal_coherence=0.433,
                nrci_baseline=0.875,
                lattice_type="Tetrahedral",
                optimization_factor=1.2
            ),
            "electromagnetic": TriangularProjectionConfig(
                name="Electromagnetic",
                platonic_solid="Cube",
                coordination_number=6,
                crv_frequency=UBPConstants.CRV_ELECTROMAGNETIC,
                wavelength=UBPConstants.WAVELENGTH_ELECTROMAGNETIC,
                spatial_coherence=0.7496,
                temporal_coherence=0.910,
                nrci_baseline=1.0,
                lattice_type="Cubic",
                optimization_factor=1.498
            ),
            "gravitational": TriangularProjectionConfig(
                name="Gravitational",
                platonic_solid="Octahedron",
                coordination_number=12,
                crv_frequency=UBPConstants.CRV_GRAVITATIONAL,
                wavelength=UBPConstants.WAVELENGTH_GRAVITATIONAL,
                spatial_coherence=0.8559,
                temporal_coherence=1.081,
                nrci_baseline=0.915,
                lattice_type="FCC",
                optimization_factor=1.35
            ),
            "biological": TriangularProjectionConfig(
                name="Biological",
                platonic_solid="Dodecahedron",
                coordination_number=20,
                crv_frequency=UBPConstants.CRV_BIOLOGICAL,
                wavelength=UBPConstants.WAVELENGTH_BIOLOGICAL,
                spatial_coherence=0.4879,
                temporal_coherence=0.973,
                nrci_baseline=0.911,
                lattice_type="H4_120Cell",
                optimization_factor=1.8
            ),
            "cosmological": TriangularProjectionConfig(
                name="Cosmological",
                platonic_solid="Icosahedron",
                coordination_number=12,
                crv_frequency=UBPConstants.CRV_COSMOLOGICAL,
                wavelength=UBPConstants.WAVELENGTH_COSMOLOGICAL,
                spatial_coherence=0.6222,
                temporal_coherence=1.022,
                nrci_baseline=0.797,
                lattice_type="H3_Icosahedral",
                optimization_factor=1.4
            )
        }
        return realms
    
    def get_realm_config(self, realm_name: str) -> TriangularProjectionConfig:
        """
        Get the configuration for a specific computational realm.
        
        Args:
            realm_name: Name of the realm ("quantum", "electromagnetic", etc.)
            
        Returns:
            TriangularProjectionConfig for the specified realm
            
        Raises:
            KeyError: If realm_name is not recognized
        """
        if realm_name not in self.realms:
            available = list(self.realms.keys())
            raise KeyError(f"Unknown realm '{realm_name}'. Available: {available}")
        
        return self.realms[realm_name]
    
    def set_current_realm(self, realm_name: str) -> None:
        """
        Set the active computational realm for subsequent operations.
        
        Args:
            realm_name: Name of the realm to activate
        """
        if realm_name not in self.realms:
            available = list(self.realms.keys())
            raise KeyError(f"Unknown realm '{realm_name}'. Available: {available}")
        
        self.current_realm = realm_name
        print(f"🔄 Active realm set to: {realm_name}")
    
    def set_observer_intent(self, intent_level: float) -> None:
        """
        Set the observer intent level for computations.
        
        Args:
            intent_level: Float between 0.0 (unfocused) and 2.0 (highly intentional)
                         1.0 = neutral, 1.5 = focused, 0.5 = passive
        """
        self.observer_intent = max(0.0, min(2.0, intent_level))
        print(f"🎯 Observer intent set to: {self.observer_intent:.3f}")
    
    def calculate_observer_factor(self) -> float:
        """
        Calculate the observer factor based on current intent level.
        
        Returns:
            Observer factor for use in energy calculations
        """
        # Purpose tensor calculation: F_μν(ψ)
        if self.observer_intent <= 1.0:
            purpose_tensor = 1.0  # Neutral
        else:
            purpose_tensor = 1.5  # Intentional
        
        # Observer factor: O_observer = 1 + (1/4π) * log(s/s_0) * F_μν(ψ)
        s_ratio = self.observer_intent / 1.0  # s_0 = 1.0 (neutral baseline)
        if s_ratio <= 0:
            s_ratio = 1e-10  # Prevent log(0)
        
        observer_factor = 1.0 + (1.0 / (4 * UBPConstants.PI)) * np.log(s_ratio) * purpose_tensor
        return observer_factor
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the UBP framework.
        
        Returns:
            Dictionary containing current system state and configuration
        """
        current_config = self.get_realm_config(self.current_realm)
        
        return {
            "framework_version": "2.0.0",
            "bitfield_dimensions": self.bitfield_dimensions,
            "current_realm": self.current_realm,
            "observer_intent": self.observer_intent,
            "observer_factor": self.calculate_observer_factor(),
            "realm_config": {
                "name": current_config.name,
                "platonic_solid": current_config.platonic_solid,
                "crv_frequency": current_config.crv_frequency,
                "wavelength": current_config.wavelength,
                "nrci_baseline": current_config.nrci_baseline,
                "optimization_factor": current_config.optimization_factor
            },
            "available_realms": list(self.realms.keys()),
            "constants": {
                "nrci_target": UBPConstants.NRCI_TARGET,
                "coherence_threshold": UBPConstants.COHERENCE_THRESHOLD,
                "csc_period": UBPConstants.CSC_PERIOD
            }
        }


# Global framework instance for easy access
# This will be initialized when the module is imported
_global_framework = None

def get_framework(bitfield_dimensions: Optional[Tuple[int, ...]] = None) -> UBPFramework:
    """
    Get or create the global UBP framework instance.
    
    Args:
        bitfield_dimensions: Optional dimensions for new framework creation
        
    Returns:
        Global UBPFramework instance
    """
    global _global_framework
    
    if _global_framework is None:
        if bitfield_dimensions is None:
            bitfield_dimensions = (32, 32, 32, 4, 2, 2)  # Default for testing
        _global_framework = UBPFramework(bitfield_dimensions)
    
    return _global_framework


if __name__ == "__main__":
    # Test the core framework
    framework = UBPFramework()
    
    print("\n" + "="*60)
    print("UBP FRAMEWORK v2.0 - CORE MODULE TEST")
    print("="*60)
    
    # Test realm switching
    for realm in ["quantum", "electromagnetic", "gravitational"]:
        framework.set_current_realm(realm)
        config = framework.get_realm_config(realm)
        print(f"  {config.name}: {config.platonic_solid}, CRV={config.crv_frequency:.6f}")
    
    # Test observer intent
    framework.set_observer_intent(1.5)
    
    # Display system status
    status = framework.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Current Realm: {status['current_realm']}")
    print(f"  Observer Factor: {status['observer_factor']:.6f}")
    print(f"  NRCI Target: {status['constants']['nrci_target']}")
    
    print("\n✅ Core module test completed successfully!")

