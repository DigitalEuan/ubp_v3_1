"""
UBP Framework v3.0 - System Constants
Author: Euan Craig, New Zealand
Date: 13 August 2025

System Constants provides all mathematical and physical constants used throughout
the UBP Framework v3.0. This is the single source of truth for all constant values.
"""

import numpy as np
from typing import Dict, Any
import math

class UBPConstants:
    """
    Universal Binary Principle Constants.
    
    Contains all mathematical, physical, and system constants used throughout
    the UBP Framework v3.0. This class serves as the single source of truth
    for all constant values.
    """
    
    # ========================================================================
    # FUNDAMENTAL PHYSICAL CONSTANTS
    # ========================================================================
    
    # Speed of light (m/s)
    SPEED_OF_LIGHT = 299792458.0
    C = SPEED_OF_LIGHT  # Alias
    
    # Planck constant (J⋅s)
    PLANCK_CONSTANT = 6.62607015e-34
    H = PLANCK_CONSTANT  # Alias
    
    # Reduced Planck constant (J⋅s)
    HBAR = PLANCK_CONSTANT / (2 * np.pi)
    
    # Elementary charge (C)
    ELEMENTARY_CHARGE = 1.602176634e-19
    E = ELEMENTARY_CHARGE  # Alias
    
    # Gravitational constant (m³⋅kg⁻¹⋅s⁻²)
    GRAVITATIONAL_CONSTANT = 6.67430e-11
    G = GRAVITATIONAL_CONSTANT  # Alias
    
    # Fine structure constant (dimensionless)
    FINE_STRUCTURE_CONSTANT = 0.0072973525693
    ALPHA = FINE_STRUCTURE_CONSTANT  # Alias
    
    # Electron mass (kg)
    ELECTRON_MASS = 9.1093837015e-31
    ME = ELECTRON_MASS  # Alias
    
    # Proton mass (kg)
    PROTON_MASS = 1.67262192369e-27
    MP = PROTON_MASS  # Alias
    
    # Boltzmann constant (J/K)
    BOLTZMANN_CONSTANT = 1.380649e-23
    KB = BOLTZMANN_CONSTANT  # Alias
    
    # ========================================================================
    # PLANCK UNITS
    # ========================================================================
    
    # Planck time (s)
    PLANCK_TIME = 5.391247e-44
    
    # Planck length (m)
    PLANCK_LENGTH = 1.616255e-35
    
    # Planck energy (J)
    PLANCK_ENERGY = 1.956082e9
    
    # Planck mass (kg)
    PLANCK_MASS = 2.176434e-8
    
    # Planck temperature (K)
    PLANCK_TEMPERATURE = 1.416784e32
    
    # ========================================================================
    # MATHEMATICAL CONSTANTS
    # ========================================================================
    
    # Pi
    PI = np.pi
    
    # Euler's number
    E_EULER = np.e
    
    # Golden ratio
    PHI = (1 + np.sqrt(5)) / 2
    GOLDEN_RATIO = PHI  # Alias
    
    # Natural logarithm of 2
    LN2 = np.log(2)
    
    # Square root of 2
    SQRT2 = np.sqrt(2)
    
    # Square root of 3
    SQRT3 = np.sqrt(3)
    
    # Square root of 5
    SQRT5 = np.sqrt(5)
    
    # Euler-Mascheroni constant
    EULER_MASCHERONI = 0.5772156649015329
    GAMMA = EULER_MASCHERONI  # Alias
    
    # ========================================================================
    # UBP SPECIFIC CONSTANTS
    # ========================================================================
    
    # Core toggle bias probabilities
    QUANTUM_TOGGLE_BIAS = E_EULER / 12  # ≈ 0.2265234857
    COSMOLOGICAL_TOGGLE_BIAS = PI ** PHI  # ≈ 0.83203682
    
    # Zitterbewegung frequency (Hz)
    ZITTERBEWEGUNG_FREQUENCY = 1.2356e20
    
    # Coherent Synchronization Cycle period (s)
    CSC_PERIOD = 1.0 / PI  # ≈ 0.318309886
    
    # Tautfluence parameters
    TAUTFLUENCE_WAVELENGTH = 635e-9  # 635 nm in meters
    TAUTFLUENCE_TIME = 2.117e-15  # seconds
    
    # Observer intent tensor parameters
    OBSERVER_NEUTRAL = 1.0
    OBSERVER_INTENTIONAL = 1.5
    
    # Coherence infinity constant
    C_INFINITY = 24 * (1 + PHI)  # ≈ 38.83281573
    
    # Energy equation parameters
    R0 = 0.95
    HT = 0.05
    R_ENERGY = R0 * (1 - HT / np.log(4))  # ≈ 0.9658855
    
    # Structural optimization default
    S_OPT_DEFAULT = 0.98
    
    # Global Coherence Index parameters
    DELTA_T_GCI = CSC_PERIOD  # 0.318309886 s
    P_GCI_DEFAULT = 0.927046
    
    # Toggle operation weights
    W_IJ_DEFAULT = 0.1
    
    # ========================================================================
    # NRCI AND COHERENCE TARGETS
    # ========================================================================
    
    # Target NRCI values
    NRCI_TARGET_STANDARD = 0.999999
    NRCI_TARGET_ULTRA_HIGH = 0.9999999
    NRCI_TARGET_PHOTONICS = 0.999999999
    
    # Coherence thresholds
    COHERENCE_THRESHOLD_MINIMUM = 0.95
    COHERENCE_THRESHOLD_HIGH = 0.99
    COHERENCE_THRESHOLD_ULTRA = 0.999
    
    # Coherence pressure parameters
    COHERENCE_PRESSURE_TARGET = 0.8
    COHERENCE_PRESSURE_MAXIMUM = 1.0
    COHERENCE_PRESSURE_MITIGATION_THRESHOLD = 0.8
    
    # ========================================================================
    # GEOMETRIC CONSTANTS
    # ========================================================================
    
    # Platonic solid coordination numbers
    TETRAHEDRON_COORDINATION = 4
    CUBE_COORDINATION = 6
    OCTAHEDRON_COORDINATION = 8
    DODECAHEDRON_COORDINATION = 12
    ICOSAHEDRON_COORDINATION = 20
    
    # Lattice structure parameters
    FCC_COORDINATION = 12  # Face-centered cubic
    H4_120_CELL_COORDINATION = 20  # 4D dodecahedral
    H3_ICOSAHEDRAL_COORDINATION = 12  # 3D icosahedral
    E8_G2_COORDINATION = 248  # E8 to G2 projection
    
    # Fractal dimension target
    FRACTAL_DIMENSION_TARGET = 2.3
    
    # ========================================================================
    # WAVELENGTH AND FREQUENCY CONSTANTS
    # ========================================================================
    
    # Standard wavelengths (meters)
    WAVELENGTH_635NM = 635e-9  # Electromagnetic
    WAVELENGTH_655NM = 655e-9  # Quantum
    WAVELENGTH_700NM = 700e-9  # Biological
    WAVELENGTH_800NM = 800e-9  # Cosmological
    WAVELENGTH_1000NM = 1000e-9  # Gravitational
    WAVELENGTH_600NM = 600e-9  # Optical
    
    # Corresponding frequencies (Hz)
    FREQUENCY_635NM = SPEED_OF_LIGHT / WAVELENGTH_635NM  # ≈ 4.72e14 Hz
    FREQUENCY_655NM = SPEED_OF_LIGHT / WAVELENGTH_655NM  # ≈ 4.58e14 Hz
    FREQUENCY_700NM = SPEED_OF_LIGHT / WAVELENGTH_700NM  # ≈ 4.28e14 Hz
    FREQUENCY_800NM = SPEED_OF_LIGHT / WAVELENGTH_800NM  # ≈ 3.75e14 Hz
    FREQUENCY_1000NM = SPEED_OF_LIGHT / WAVELENGTH_1000NM  # ≈ 3.00e14 Hz
    FREQUENCY_600NM = SPEED_OF_LIGHT / WAVELENGTH_600NM  # ≈ 5.00e14 Hz
    
    # ========================================================================
    # ERROR CORRECTION CONSTANTS
    # ========================================================================
    
    # Golay code parameters
    GOLAY_N = 23  # Code length
    GOLAY_K = 12  # Information bits
    GOLAY_T = 3   # Error correction capability
    
    # Hamming code parameters
    HAMMING_N = 7  # Code length
    HAMMING_K = 4  # Information bits
    HAMMING_T = 1  # Error correction capability
    
    # BCH code parameters
    BCH_N = 31  # Code length
    BCH_K = 21  # Information bits
    BCH_T = 2   # Error correction capability
    
    # Reed-Solomon compression ratio
    REED_SOLOMON_COMPRESSION = 0.3  # 30% compression
    
    # p-adic encoding parameters
    PADIC_DEFAULT_PRIME = 2
    PADIC_DEFAULT_PRECISION = 20
    
    # Fibonacci encoding parameters
    FIBONACCI_MAX_INDEX = 50
    FIBONACCI_DEFAULT_REDUNDANCY = 0.3
    
    # ========================================================================
    # HARDWARE CONFIGURATION CONSTANTS
    # ========================================================================
    
    # Memory limits (bytes)
    MEMORY_8GB_IMAC = 8 * 1024**3
    MEMORY_4GB_MOBILE = 4 * 1024**3
    MEMORY_RASPBERRY_PI5 = 8 * 1024**3  # Assuming 8GB model
    
    # OffBit counts for different hardware
    OFFBITS_8GB_IMAC = 1000000
    OFFBITS_RASPBERRY_PI5 = 100000
    OFFBITS_4GB_MOBILE = 10000
    
    # Bitfield dimensions for different configurations
    BITFIELD_6D_FULL = (170, 170, 170, 5, 2, 2)  # ~2.3M cells
    BITFIELD_6D_MEDIUM = (100, 100, 100, 5, 2, 2)  # ~1M cells
    BITFIELD_6D_SMALL = (50, 50, 50, 5, 2, 2)  # ~250K cells
    
    # Performance targets
    TARGET_OPERATIONS_PER_SECOND = 5000
    TARGET_OPERATION_TIME_RASPBERRY_PI = 2.0  # seconds
    
    # ========================================================================
    # VALIDATION CONSTANTS
    # ========================================================================
    
    # Statistical significance threshold
    P_VALUE_THRESHOLD = 0.01
    
    # Validation iteration counts
    VALIDATION_ITERATIONS_STANDARD = 1000
    VALIDATION_ITERATIONS_EXTENSIVE = 10000
    
    # Test field dimensions
    TEST_FIELD_3X3X10 = (3, 3, 10)
    
    # Sparsity parameters
    SPARSITY_DEFAULT = 0.01
    SPARSITY_DENSE = 0.1
    SPARSITY_SPARSE = 0.001
    
    # ========================================================================
    # REALM-SPECIFIC CONSTANTS
    # ========================================================================
    
    # Realm frequency ranges (Hz)
    NUCLEAR_FREQ_MIN = 1e16
    NUCLEAR_FREQ_MAX = 1e20
    OPTICAL_FREQ_MIN = 1e14
    OPTICAL_FREQ_MAX = 1e15
    QUANTUM_FREQ_MIN = 1e13
    QUANTUM_FREQ_MAX = 1e16
    ELECTROMAGNETIC_FREQ_MIN = 1e6
    ELECTROMAGNETIC_FREQ_MAX = 1e12
    GRAVITATIONAL_FREQ_MIN = 1e-4
    GRAVITATIONAL_FREQ_MAX = 1e4
    BIOLOGICAL_FREQ_MIN = 1e-2
    BIOLOGICAL_FREQ_MAX = 1e3
    COSMOLOGICAL_FREQ_MIN = 1e-18
    COSMOLOGICAL_FREQ_MAX = 1e-10
    
    # Realm time scales (seconds)
    NUCLEAR_TIMESCALE = 1e-23
    OPTICAL_TIMESCALE = 1e-15
    QUANTUM_TIMESCALE = 1e-18
    ELECTROMAGNETIC_TIMESCALE = 1e-12
    GRAVITATIONAL_TIMESCALE = 1e-3
    BIOLOGICAL_TIMESCALE = 1e-3
    COSMOLOGICAL_TIMESCALE = 1e6
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    @classmethod
    def get_frequency_from_wavelength(cls, wavelength_meters: float) -> float:
        """Convert wavelength to frequency."""
        return cls.SPEED_OF_LIGHT / wavelength_meters
    
    @classmethod
    def get_wavelength_from_frequency(cls, frequency_hz: float) -> float:
        """Convert frequency to wavelength."""
        return cls.SPEED_OF_LIGHT / frequency_hz
    
    @classmethod
    def get_photon_energy(cls, frequency_hz: float) -> float:
        """Calculate photon energy from frequency."""
        return cls.PLANCK_CONSTANT * frequency_hz
    
    @classmethod
    def get_photon_energy_from_wavelength(cls, wavelength_meters: float) -> float:
        """Calculate photon energy from wavelength."""
        frequency = cls.get_frequency_from_wavelength(wavelength_meters)
        return cls.get_photon_energy(frequency)
    
    @classmethod
    def get_quantum_toggle_bias(cls) -> float:
        """Get quantum realm toggle bias probability."""
        return cls.QUANTUM_TOGGLE_BIAS
    
    @classmethod
    def get_cosmological_toggle_bias(cls) -> float:
        """Get cosmological realm toggle bias probability."""
        return cls.COSMOLOGICAL_TOGGLE_BIAS
    
    @classmethod
    def get_csc_period(cls) -> float:
        """Get Coherent Synchronization Cycle period."""
        return cls.CSC_PERIOD
    
    @classmethod
    def get_nrci_target(cls, precision_level: str = "standard") -> float:
        """
        Get NRCI target based on precision level.
        
        Args:
            precision_level: "standard", "ultra_high", or "photonics"
            
        Returns:
            NRCI target value
        """
        targets = {
            "standard": cls.NRCI_TARGET_STANDARD,
            "ultra_high": cls.NRCI_TARGET_ULTRA_HIGH,
            "photonics": cls.NRCI_TARGET_PHOTONICS
        }
        return targets.get(precision_level, cls.NRCI_TARGET_STANDARD)
    
    @classmethod
    def get_hardware_offbit_count(cls, hardware_type: str) -> int:
        """
        Get recommended OffBit count for hardware type.
        
        Args:
            hardware_type: "8gb_imac", "raspberry_pi5", or "4gb_mobile"
            
        Returns:
            Recommended OffBit count
        """
        counts = {
            "8gb_imac": cls.OFFBITS_8GB_IMAC,
            "raspberry_pi5": cls.OFFBITS_RASPBERRY_PI5,
            "4gb_mobile": cls.OFFBITS_4GB_MOBILE
        }
        return counts.get(hardware_type, cls.OFFBITS_4GB_MOBILE)
    
    @classmethod
    def get_bitfield_dimensions(cls, size_category: str) -> tuple:
        """
        Get Bitfield dimensions for size category.
        
        Args:
            size_category: "full", "medium", or "small"
            
        Returns:
            Bitfield dimensions tuple
        """
        dimensions = {
            "full": cls.BITFIELD_6D_FULL,
            "medium": cls.BITFIELD_6D_MEDIUM,
            "small": cls.BITFIELD_6D_SMALL
        }
        return dimensions.get(size_category, cls.BITFIELD_6D_SMALL)
    
    @classmethod
    def get_realm_frequency_range(cls, realm: str) -> tuple:
        """
        Get frequency range for a specific realm.
        
        Args:
            realm: Realm name
            
        Returns:
            Tuple of (min_frequency, max_frequency) in Hz
        """
        ranges = {
            "nuclear": (cls.NUCLEAR_FREQ_MIN, cls.NUCLEAR_FREQ_MAX),
            "optical": (cls.OPTICAL_FREQ_MIN, cls.OPTICAL_FREQ_MAX),
            "quantum": (cls.QUANTUM_FREQ_MIN, cls.QUANTUM_FREQ_MAX),
            "electromagnetic": (cls.ELECTROMAGNETIC_FREQ_MIN, cls.ELECTROMAGNETIC_FREQ_MAX),
            "gravitational": (cls.GRAVITATIONAL_FREQ_MIN, cls.GRAVITATIONAL_FREQ_MAX),
            "biological": (cls.BIOLOGICAL_FREQ_MIN, cls.BIOLOGICAL_FREQ_MAX),
            "cosmological": (cls.COSMOLOGICAL_FREQ_MIN, cls.COSMOLOGICAL_FREQ_MAX)
        }
        return ranges.get(realm, (1e12, 1e13))  # Default range
    
    @classmethod
    def get_realm_timescale(cls, realm: str) -> float:
        """
        Get characteristic timescale for a specific realm.
        
        Args:
            realm: Realm name
            
        Returns:
            Characteristic timescale in seconds
        """
        timescales = {
            "nuclear": cls.NUCLEAR_TIMESCALE,
            "optical": cls.OPTICAL_TIMESCALE,
            "quantum": cls.QUANTUM_TIMESCALE,
            "electromagnetic": cls.ELECTROMAGNETIC_TIMESCALE,
            "gravitational": cls.GRAVITATIONAL_TIMESCALE,
            "biological": cls.BIOLOGICAL_TIMESCALE,
            "cosmological": cls.COSMOLOGICAL_TIMESCALE
        }
        return timescales.get(realm, 1e-12)  # Default timescale
    
    @classmethod
    def get_all_constants(cls) -> Dict[str, Any]:
        """
        Get all constants as a dictionary.
        
        Returns:
            Dictionary of all constants
        """
        constants = {}
        
        # Get all class attributes that are constants (uppercase)
        for attr_name in dir(cls):
            if attr_name.isupper() and not attr_name.startswith('_'):
                constants[attr_name] = getattr(cls, attr_name)
        
        return constants
    
    @classmethod
    def validate_constants(cls) -> Dict[str, bool]:
        """
        Validate that all constants have reasonable values.
        
        Returns:
            Dictionary of validation results
        """
        validations = {}
        
        # Physical constants validation
        validations['speed_of_light'] = 2.9e8 < cls.SPEED_OF_LIGHT < 3.1e8
        validations['planck_constant'] = 6e-34 < cls.PLANCK_CONSTANT < 7e-34
        validations['fine_structure'] = 0.007 < cls.FINE_STRUCTURE_CONSTANT < 0.008
        
        # Mathematical constants validation
        validations['pi'] = 3.14 < cls.PI < 3.15
        validations['e'] = 2.71 < cls.E_EULER < 2.72
        validations['golden_ratio'] = 1.61 < cls.PHI < 1.62
        
        # UBP constants validation
        validations['quantum_bias'] = 0.2 < cls.QUANTUM_TOGGLE_BIAS < 0.3
        validations['cosmological_bias'] = 0.8 < cls.COSMOLOGICAL_TOGGLE_BIAS < 0.9
        validations['csc_period'] = 0.3 < cls.CSC_PERIOD < 0.4
        
        # NRCI targets validation
        validations['nrci_standard'] = 0.999 < cls.NRCI_TARGET_STANDARD < 1.0
        validations['nrci_ultra_high'] = 0.9999 < cls.NRCI_TARGET_ULTRA_HIGH < 1.0
        
        return validations

# Create a global instance for easy access
UBP_CONSTANTS = UBPConstants()

# Export commonly used constants for convenience
PI = UBPConstants.PI
E = UBPConstants.E_EULER
PHI = UBPConstants.PHI
C = UBPConstants.SPEED_OF_LIGHT
HBAR = UBPConstants.HBAR
ALPHA = UBPConstants.FINE_STRUCTURE_CONSTANT

