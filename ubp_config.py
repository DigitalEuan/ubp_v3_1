"""
Universal Binary Principle (UBP) Framework v3.2+ - Central Configuration Module
Author: Euan Craig, New Zealand
Date: 20 August 2025

This module centralizes all configurable parameters for the UBP framework,
including system constants, performance thresholds, realm definitions,
observer parameters, and Bitfield dimensions. It ensures a single source
of truth for all framework settings.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass, field
import json

# --- Dataclasses for Configuration Structure ---

@dataclass
class ConstantConfig:
    """Stores fundamental UBP and physical constants."""
    PI: float = np.pi
    SPEED_OF_LIGHT: float = 299792458.0  # meters/second
    PLANCK_CONSTANT: float = 6.62607015e-34  # J.Hz-1
    GRAVITATIONAL_CONSTANT: float = 6.67430e-11  # m3 kg-1 s-2
    ELEMENTARY_CHARGE: float = 1.602176634e-19  # Coulombs
    # UBP-specific constants
    C_INFINITY: float = 1.0e+308 # Conceptual maximum speed/information propagation rate
    OFFBIT_ENERGY_UNIT: float = 1.0e-30 # Base energy unit for a single OffBit operation/state
    UBP_QUANTUM_COHERENCE_UNIT: float = 1.0e-15 # Baseline for quantum coherence
    EPSILON_UBP: float = 1e-18 # Smallest significant UBP value, prevents division by zero in log/etc.
    GOLDEN_RATIO: float = (1 + np.sqrt(5)) / 2 # Phi, important for geometric aspects

@dataclass
class PerformanceConfig:
    """Configures performance-related thresholds and targets."""
    TARGET_NRCI: float = 0.999999 # Target Normalized Resonance Coherence Index (0-1)
    COHERENCE_THRESHOLD: float = 0.95 # Minimum coherence for stable operations
    MIN_STABILITY: float = 0.85 # Minimum stability for system integrity
    MAX_ERROR_TOLERANCE: float = 0.001 # Maximum allowable error rate

@dataclass
class TemporalConfig:
    """Configures time-related parameters."""
    COHERENT_SYNCHRONIZATION_CYCLE_PERIOD: float = 1.0e-9 # Seconds (e.g., 1 nanosecond)
    BITTIME_UNIT_DURATION: float = 1.0e-12 # Seconds (picoseconds)

@dataclass
class ObserverConfig:
    """Configures parameters related to the observer/consciousness model."""
    DEFAULT_INTENT_LEVEL: float = 1.0 # Neutral intent
    MIN_INTENT_LEVEL: float = 0.0 # Unfocused
    MAX_INTENT_LEVEL: float = 2.0 # Highly intentional
    OBSERVER_INFLUENCE_FACTOR: float = 0.1 # Multiplier for observer impact

@dataclass
class RealmConfig:
    """
    Configuration for a specific computational realm in the UBP framework.
    Includes fundamental parameters for various "platonic solids" of reality.
    """
    name: str
    platonic_solid: str
    main_crv: float  # Central Resonance Value (Hz or arbitrary unit)
    wavelength: float  # Associated wavelength (e.g., nm for EM, meters for Grav)
    coordination_number: int = 12 # Default, e.g., for FCC lattice
    spatial_coherence: float = 0.99  # Baseline spatial coherence for the realm
    temporal_coherence: float = 0.99  # Baseline temporal coherence for the realm
    nrci_baseline: float = 0.8  # Default NRCI baseline for this realm
    lattice_type: str = "Resonant manifold" # Generic description
    optimization_factor: float = 1.0 # Multiplier for certain optimizations

@dataclass
class UBPConfig:
    """
    The main UBP Framework Configuration container.
    Initializes all sub-configurations and realm definitions.
    """
    environment: str = "development" # "development", "production", "testing"
    
    # Global constants
    constants: ConstantConfig = field(default_factory=ConstantConfig)
    
    # Performance parameters
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    
    # Temporal parameters
    temporal: TemporalConfig = field(default_factory=TemporalConfig)

    # Observer parameters
    observer: ObserverConfig = field(default_factory=ObserverConfig)

    # Bitfield Dimensions (6D tuple as specified by UBP design)
    BITFIELD_DIMENSIONS: Tuple[int, int, int, int, int, int] = (10, 10, 10, 10, 10, 10)
    
    # Realm configurations - a dictionary for easy access
    realms: Dict[str, RealmConfig] = field(default_factory=dict)

    def __post_init__(self):
        # Define default realms with their specific CRVs and properties.
        # These values are illustrative and would be detailed in a UBP design document.
        self._initialize_default_realms()
        self.apply_environment_settings()

    def _initialize_default_realms(self):
        """Initializes the predefined computational realms."""
        self.realms = {
            "quantum": RealmConfig(
                name="quantum",
                platonic_solid="icosahedron",
                main_crv=4.2e12, # Terahertz range (illustrative Quantum CRV)
                wavelength=700e-9, # Placeholder, depends on specific quantum phenomena
                nrci_baseline=0.9
            ),
            "electromagnetic": RealmConfig(
                name="electromagnetic",
                platonic_solid="octahedron",
                main_crv=2.45e9, # Gigahertz range (e.g., Microwave/Wifi CRV)
                wavelength=0.122, # Meters (for 2.45 GHz)
                nrci_baseline=0.85
            ),
            "gravitational": RealmConfig(
                name="gravitational",
                platonic_solid="dodecahedron",
                main_crv=1.0e-18, # Extremely low frequency (illustrative Gravitational CRV)
                wavelength=3.0e+26, # Meters (for 1e-18 Hz)
                nrci_baseline=0.7
            ),
            "plasma": RealmConfig(
                name="plasma",
                platonic_solid="tetrahedron",
                main_crv=1.0e6, # Megahertz range (illustrative Plasma CRV)
                wavelength=300, # Meters
                nrci_baseline=0.75
            ),
            "nuclear": RealmConfig(
                name="nuclear",
                platonic_solid="star_tetrahedron",
                main_crv=1.0e20, # Very high frequency (illustrative Nuclear CRV)
                wavelength=3.0e-12, # Meters (gamma-ray scale)
                nrci_baseline=0.95
            ),
            "optical": RealmConfig(
                name="optical",
                platonic_solid="cuboctahedron",
                main_crv=5.0e14, # Visible light range (e.g., 500THz for green light)
                wavelength=600e-9, # Meters (for 500 THz)
                nrci_baseline=0.9
            ),
            "biologic": RealmConfig(
                name="biologic",
                platonic_solid="rhombic_dodecahedron",
                main_crv=7.83, # Schumann resonance (illustrative Biologic CRV)
                wavelength=3.82e7, # Meters
                nrci_baseline=0.65
            )
            # Add more realms as needed based on the UBP design document
        }

    def apply_environment_settings(self):
        """Applies environment-specific adjustments to configuration."""
        if self.environment == "development":
            self.performance.TARGET_NRCI = 0.99
            self.performance.COHERENCE_THRESHOLD = 0.90
            self.BITFIELD_DIMENSIONS = (5, 5, 5, 5, 5, 5) # Smaller for dev
            print(f"UBPConfig: Applied DEVELOPMENT environment settings.")
        elif self.environment == "production":
            self.performance.TARGET_NRCI = 0.999999
            self.performance.COHERENCE_THRESHOLD = 0.95
            self.BITFIELD_DIMENSIONS = (100, 100, 100, 100, 100, 100) # Larger for prod
            print(f"UBPConfig: Applied PRODUCTION environment settings.")
        elif self.environment == "testing":
            self.performance.TARGET_NRCI = 0.9
            self.performance.COHERENCE_THRESHOLD = 0.8
            self.BITFIELD_DIMENSIONS = (1, 1, 1, 1, 1, 1) # Minimal for tests
            print(f"UBPConfig: Applied TESTING environment settings.")
        else:
            print(f"UBPConfig: Unknown environment '{self.environment}'. Using default settings.")

    def get_bitfield_dimensions(self) -> Tuple[int, ...]:
        """Returns the configured Bitfield dimensions."""
        return self.BITFIELD_DIMENSIONS

    def get_realm_config(self, realm_name: str) -> Optional[RealmConfig]:
        """Returns the configuration for a specific realm."""
        return self.realms.get(realm_name)

# --- Singleton Instance Management ---
_ubp_config_instance: Optional[UBPConfig] = None

def get_config(environment: Optional[str] = None) -> UBPConfig:
    """
    Returns the singleton UBPConfig instance.
    Initializes it if it doesn't exist. Can set environment on first call.
    """
    global _ubp_config_instance
    if _ubp_config_instance is None:
        if environment:
            _ubp_config_instance = UBPConfig(environment=environment)
        else:
            _ubp_config_instance = UBPConfig()
    elif environment and _ubp_config_instance.environment != environment:
        print(f"⚠️ Warning: UBPConfig already initialized with environment '{_ubp_config_instance.environment}'. "
              f"Ignoring request to set environment to '{environment}'.")
    return _ubp_config_instance

# --- Example Usage (for testing/demonstration) ---
if __name__ == "__main__":
    print("--- Testing ubp_config.py ---")

    # Test default initialization
    config_default = get_config()
    print(f"\nDefault Config Environment: {config_default.environment}")
    print(f"Bitfield Dimensions: {config_default.get_bitfield_dimensions()}")
    print(f"Pi: {config_default.constants.PI}")
    print(f"Target NRCI: {config_default.performance.TARGET_NRCI}")
    print(f"CSC Period: {config_default.temporal.COHERENT_SYNCHRONIZATION_CYCLE_PERIOD} seconds")
    print(f"Default Observer Intent: {config_default.observer.DEFAULT_INTENT_LEVEL}")

    # Test getting a specific realm
    em_realm = config_default.get_realm_config("electromagnetic")
    if em_realm:
        print(f"\nElectromagnetic Realm:")
        print(f"  Platonic Solid: {em_realm.platonic_solid}")
        print(f"  Main CRV: {em_realm.main_crv} Hz")
        print(f"  Wavelength: {em_realm.wavelength} m")
        print(f"  NRCI Baseline: {em_realm.nrci_baseline}")
    else:
        print("Electromagnetic realm not found.")

    # Test setting a different environment (should work only on first call for global instance)
    print("\nAttempting to re-initialize with 'testing' environment...")
    config_test = get_config(environment="testing") # Should print a warning
    print(f"Config after setting to 'testing': {config_test.environment}")
    print(f"Bitfield Dimensions in 'testing': {config_test.get_bitfield_dimensions()}")

    # To truly test different environments, you'd need to reset the global _ubp_config_instance
    # For demonstration, let's simulate by manually setting it to None and re-initializing
    print("\n--- Simulating fresh start for 'production' environment ---")
    _ubp_config_instance = None # Reset global instance for testing purposes
    config_prod = get_config(environment="production")
    print(f"Config Environment: {config_prod.environment}")
    print(f"Bitfield Dimensions: {config_prod.get_bitfield_dimensions()}")
    print(f"Target NRCI: {config_prod.performance.TARGET_NRCI}")

    print("\n✅ ubp_config.py test completed successfully!")
