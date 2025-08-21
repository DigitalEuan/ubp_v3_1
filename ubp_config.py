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

# Import system constants to populate ConstantConfig
from system_constants import UBPConstants as RawUBPConstants

# --- Dataclasses for Configuration Structure ---

@dataclass
class ConstantConfig:
    """Stores fundamental UBP and physical constants."""
    # Populated from system_constants.UBPConstants
    PI: float = RawUBPConstants.PI
    E: float = RawUBPConstants.E # Added missing E constant
    PHI: float = RawUBPConstants.PHI # Added missing PHI (Golden Ratio) constant, previously GOLDEN_RATIO below
    EULER_MASCHERONI: float = RawUBPConstants.EULER_MASCHERONI # Added missing EULER_MASCHERONI constant
    SPEED_OF_LIGHT: float = RawUBPConstants.SPEED_OF_LIGHT
    PLANCK_CONSTANT: float = RawUBPConstants.PLANCK_CONSTANT
    PLANCK_REDUCED: float = RawUBPConstants.PLANCK_REDUCED
    BOLTZMANN_CONSTANT: float = RawUBPConstants.BOLTZMANN_CONSTANT
    FINE_STRUCTURE_CONSTANT: float = RawUBPConstants.FINE_STRUCTURE_CONSTANT
    GRAVITATIONAL_CONSTANT: float = RawUBPConstants.GRAVITATIONAL_CONSTANT
    AVOGADRO_NUMBER: float = RawUBPConstants.AVOGADRO_NUMBER
    ELEMENTARY_CHARGE: float = RawUBPConstants.ELEMENTARY_CHARGE
    VACUUM_PERMITTIVITY: float = RawUBPConstants.VACUUM_PERMITTIVITY
    VACUUM_PERMEABILITY: float = RawUBPConstants.VACUUM_PERMEABILITY
    ELECTRON_MASS: float = RawUBPConstants.ELECTRON_MASS
    PROTON_MASS: float = RawUBPConstants.PROTON_MASS
    NEUTRON_MASS: float = RawUBPConstants.NEUTRON_MASS
    NUCLEAR_MAGNETON: float = RawUBPConstants.NUCLEAR_MAGNETON
    PROTON_GYROMAGNETIC: float = RawUBPConstants.PROTON_GYROMAGNETIC
    NEUTRON_GYROMAGNETIC: float = RawUBPConstants.NEUTRON_GYROMAGNETIC
    DEUTERON_BINDING_ENERGY: float = RawUBPConstants.DEUTERON_BINDING_ENERGY
    RYDBERG_CONSTANT: float = RawUBPConstants.RYDBERG_CONSTANT

    # UBP-specific constants
    C_INFINITY: float = RawUBPConstants.C_INFINITY # Conceptual maximum speed/information propagation rate
    OFFBIT_ENERGY_UNIT: float = RawUBPConstants.OFFBIT_ENERGY_UNIT # Base energy unit for a single OffBit operation/state
    UBP_QUANTUM_COHERENCE_UNIT: float = 1.0e-15 # Baseline for quantum coherence
    EPSILON_UBP: float = 1e-18 # Smallest significant UBP value, prevents division by zero in log/etc.
    GOLDEN_RATIO: float = RawUBPConstants.PHI # Phi, important for geometric aspects (kept for backward compatibility, but PHI is preferred)


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
    PLANCK_TIME_SECONDS: float = RawUBPConstants.PLANCK_TIME_SECONDS
    COHERENT_SYNCHRONIZATION_CYCLE_PERIOD_DEFAULT: float = RawUBPConstants.COHERENT_SYNCHRONIZATION_CYCLE_SECONDS

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
    sub_crvs: List[float] = field(default_factory=list) # Corrected: Added missing 'sub_crvs' attribute
    frequency_range: Tuple[float, float] = (0.0, 0.0) # Corrected: Added missing 'frequency_range' attribute

@dataclass
class MoleculeConfig: # Corrected: Added missing MoleculeConfig dataclass
    """Configuration for molecular simulation in HTR."""
    name: str
    nodes: int
    bond_length: float  # L_0 in meters
    bond_energy: float  # eV
    geometry_type: str
    smiles: Optional[str] = None


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

    # HTR Molecule configurations
    molecules: Dict[str, MoleculeConfig] = field(default_factory=dict) # Corrected: Added missing 'molecules' field

    # Simplified config for CRV, Error Correction, and Bitfield sizing within UBPConfig
    @dataclass
    class CRVConfig:
        prediction_base_computation_time: float = 0.00001 # Base time in seconds
        prediction_complexity_factor: float = 0.1 # Factor for complexity adjustment
        prediction_noise_factor: float = 0.05 # Factor for noise adjustment
        score_weights_frequency: float = 0.4 # Weight for frequency matching in CRV selection
        score_weights_complexity: float = 0.3 # Weight for complexity matching
        score_weights_noise: float = 0.2 # Weight for noise tolerance
        score_weights_performance: float = 0.1 # Weight for performance
        crv_match_tolerance: float = 0.05 # Tolerance for CRV frequency matching
        confidence_freq_boost: float = 0.2 # Confidence boost for frequency match
        confidence_noise_boost: float = 0.1 # Confidence boost for low noise
        confidence_historical_perf_boost: float = 0.1 # Confidence boost for good historical perf
        harmonic_ratio_tolerance: float = 0.02 # Tolerance for detecting harmonic ratios
        harmonic_fraction_denominator_limit: int = 4 # Max denominator for simple fractional harmonics

    @dataclass
    class ErrorCorrectionConfig:
        error_threshold: float = 0.05 # General error threshold for correction
        golay_code: str = "23,12" # (n,k) for Golay code
        bch_code: str = "31,21" # (n,k) for BCH code
        hamming_code: str = "7,4" # (n,k) for Hamming code
        padic_prime: int = 7 # P-adic prime for certain error models
        fibonacci_depth: int = 5 # Depth for Fibonacci encoding/decoding
        nrci_base_score: float = 0.9 # Base NRCI for error correction
    
    @dataclass
    class BitfieldConfig:
        size_mobile: Tuple[int, int, int, int, int, int] = (10, 10, 10, 1, 1, 1)
        size_raspberry_pi: Tuple[int, int, int, int, int, int] = (20, 20, 20, 2, 2, 1)
        size_local: Tuple[int, int, int, int, int, int] = (50, 50, 50, 5, 2, 2)
        size_colab: Tuple[int, int, int, int, int, int] = (70, 70, 70, 5, 3, 2)
        size_kaggle: Tuple[int, int, int, int, int, int] = (60, 60, 60, 5, 3, 2)
        size_production: Tuple[int, int, int, int, int, int] = (100, 100, 100, 10, 5, 5)

    crv: CRVConfig = field(default_factory=CRVConfig)
    error_correction: ErrorCorrectionConfig = field(default_factory=ErrorCorrectionConfig)
    bitfield: BitfieldConfig = field(default_factory=BitfieldConfig)
    
    default_realm: str = "electromagnetic" # Corrected: Added default_realm attribute for CRVSelector fallback


    def __post_init__(self):
        # Define default realms with their specific CRVs and properties.
        self._initialize_default_realms()
        self._initialize_default_molecules() # Corrected: Added call to _initialize_default_molecules
        self.apply_environment_settings()

    def _initialize_default_realms(self):
        """Initializes the predefined computational realms."""
        self.realms = {
            "quantum": RealmConfig(
                name="quantum",
                platonic_solid="icosahedron",
                main_crv=4.2e12, # Terahertz range (illustrative Quantum CRV)
                wavelength=700e-9, # Placeholder, depends on specific quantum phenomena
                nrci_baseline=0.9,
                sub_crvs=[4.0e12, 4.4e12], # Example sub_crvs
                frequency_range=(1e12, 1e15)
            ),
            "electromagnetic": RealmConfig(
                name="electromagnetic",
                platonic_solid="octahedron",
                main_crv=2.45e9, # Gigahertz range (e.g., Microwave/Wifi CRV)
                wavelength=0.122, # Meters (for 2.45 GHz)
                nrci_baseline=0.85,
                sub_crvs=[2.4e9, 2.5e9],
                frequency_range=(1e9, 1e11)
            ),
            "gravitational": RealmConfig(
                name="gravitational",
                platonic_solid="dodecahedron",
                main_crv=1.0e-18, # Extremely low frequency (illustrative Gravitational CRV)
                wavelength=3.0e+26, # Meters (for 1e-18 Hz)
                nrci_baseline=0.7,
                sub_crvs=[0.5e-18, 1.5e-18],
                frequency_range=(1e-20, 1e-15)
            ),
            "plasma": RealmConfig(
                name="plasma",
                platonic_solid="tetrahedron",
                main_crv=1.0e6, # Megahertz range (illustrative Plasma CRV)
                wavelength=300, # Meters
                nrci_baseline=0.75,
                sub_crvs=[0.9e6, 1.1e6],
                frequency_range=(1e5, 1e8)
            ),
            "nuclear": RealmConfig(
                name="nuclear",
                platonic_solid="star_tetrahedron",
                main_crv=1.0e20, # Very high frequency (illustrative Nuclear CRV)
                wavelength=3.0e-12, # Meters (gamma-ray scale)
                nrci_baseline=0.95,
                sub_crvs=[0.9e20, 1.1e20],
                frequency_range=(1e19, 1e21)
            ),
            "optical": RealmConfig(
                name="optical",
                platonic_solid="cuboctahedron",
                main_crv=5.0e14, # Visible light range (e.g., 500THz for green light)
                wavelength=600e-9, # Meters (for 500 THz)
                nrci_baseline=0.9,
                sub_crvs=[4.8e14, 5.2e14],
                frequency_range=(4e14, 8e14)
            ),
            "biologic": RealmConfig(
                name="biologic",
                platonic_solid="rhombic_dodecahedron",
                main_crv=7.83, # Schumann resonance (illustrative Biologic CRV)
                wavelength=3.82e7, # Meters
                nrci_baseline=0.65,
                sub_crvs=[7.5, 8.0],
                frequency_range=(1e0, 1e3)
            )
        }

    def _initialize_default_molecules(self): # Corrected: Added missing _initialize_default_molecules method
        """Initializes predefined molecular configurations for HTR."""
        self.molecules = {
            'propane': MoleculeConfig('propane', 10, 0.154e-9, 4.8, 'alkane', 'CCC'),
            'benzene': MoleculeConfig('benzene', 6, 0.14e-9, 5.0, 'aromatic', 'c1ccccc1'),
            'methane': MoleculeConfig('methane', 5, 0.109e-9, 4.5, 'tetrahedral', 'C'),
            'butane': MoleculeConfig('butane', 13, 0.154e-9, 4.8, 'alkane', 'CCCC')
        }


    def apply_environment_settings(self):
        """Applies environment-specific adjustments to configuration."""
        if self.environment == "development":
            self.performance.TARGET_NRCI = 0.99
            self.performance.COHERENCE_THRESHOLD = 0.90
            self.BITFIELD_DIMENSIONS = self.bitfield.size_mobile # Smaller for dev
            print(f"UBPConfig: Applied DEVELOPMENT environment settings.")
        elif self.environment == "production":
            self.performance.TARGET_NRCI = 0.999999
            self.performance.COHERENCE_THRESHOLD = 0.95
            self.BITFIELD_DIMENSIONS = self.bitfield.size_production # Larger for prod
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
        return self.realms.get(realm_name.lower())
    
    def get_molecule_config(self, molecule_name: str) -> Optional[MoleculeConfig]: # Corrected: Added missing get_molecule_config method
        """Returns the configuration for a specific molecule."""
        return self.molecules.get(molecule_name.lower())
    
    def get_summary(self) -> Dict[str, Any]:
        """Returns a summary of the current configuration."""
        return {
            "environment": self.environment,
            "bitfield_dimensions": self.BITFIELD_DIMENSIONS,
            "target_nrci": self.performance.TARGET_NRCI,
            "num_realms_configured": len(self.realms),
            "num_molecules_configured": len(self.molecules), # Corrected: Added molecules count to summary
            "example_quantum_crv": self.realms.get("quantum").main_crv if "quantum" in self.realms else None,
            "epsilon_ubp": self.constants.EPSILON_UBP,
        }

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
    print(f"E: {config_default.constants.E}") # Test E
    print(f"Golden Ratio: {config_default.constants.PHI}") # Test PHI
    print(f"Euler-Mascheroni: {config_default.constants.EULER_MASCHERONI}") # Test EULER_MASCHERONI
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
        print(f"  Sub CRVs: {em_realm.sub_crvs}")
        print(f"  Frequency Range: {em_realm.frequency_range}")
    else:
        print("Electromagnetic realm not found.")

    # Test getting a specific molecule
    propane_mol = config_default.get_molecule_config("propane")
    if propane_mol:
        print(f"\nPropane Molecule:")
        print(f"  Nodes: {propane_mol.nodes}")
        print(f"  Bond Length: {propane_mol.bond_length} m")
    else:
        print("Propane molecule not found.")

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
