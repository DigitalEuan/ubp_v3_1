"""
UBP Framework v3.1.1 - System Constants
Author: Euan Craig, New Zealand
Date: 18 August 2025

This module defines all fundamental constants used across the UBP Framework.
This ensures a single, consistent source of truth for physical, mathematical,
and UBP-specific parameters.
"""

import numpy as np

class UBPConstants:
    """
    Collection of universal, mathematical, and UBP-specific constants.
    All values are defined here for consistency across the framework.
    """

    # --- Universal Physical Constants ---
    SPEED_OF_LIGHT = 299792458  # meters per second (m/s)
    PLANCK_CONSTANT = 6.62607015e-34  # Joule-seconds (J⋅s)
    BOLTZMANN_CONSTANT = 1.380649e-23  # Joules per Kelvin (J/K)
    FINE_STRUCTURE_CONSTANT = 0.0072973525693  # Dimensionless
    GRAVITATIONAL_CONSTANT = 6.67430e-11  # m³⋅kg⁻¹⋅s⁻²
    AVOGADRO_NUMBER = 6.02214076e23  # mol⁻¹
    ELEMENTARY_CHARGE = 1.602176634e-19  # Coulombs (C)

    # --- Mathematical Constants ---
    PI = np.pi  # π (Pi)
    E = np.e  # e (Euler's number)
    PHI = (1 + np.sqrt(5)) / 2  # φ (Golden Ratio)
    EULER_MASCHERONI = 0.5772156649  # γ (Euler-Mascheroni constant)

    # --- UBP-Specific Core Values ---
    # Core Resonance Values (CRVs) - Reference only, actual values might be in ubp_config.py
    # or crv_database.py for dynamic management.
    CRV_ELECTROMAGNETIC_BASE = PI  # Base for EM realm
    CRV_QUANTUM_BASE = E / 12  # Base for Quantum realm
    CRV_GRAVITATIONAL_BASE = 160.19  # Empirical, derived from gravitational wave research
    CRV_BIOLOGICAL_BASE = 10.0  # Empirical, related to neural frequencies
    CRV_COSMOLOGICAL_BASE = PI ** PHI # Empirical, π^φ
    CRV_NUCLEAR_BASE = 1.2356e20 # Zitterbewegung frequency
    CRV_OPTICAL_BASE = 5.0e14 # 600 nm light frequency

    # Toggle Algebra & Bitfield Parameters
    OFFBIT_DEFAULT_SIZE_BYTES = 4  # Each OffBit is typically 32 bits
    BITFIELD_DEFAULT_SPARSITY = 0.01
    MAX_BITFIELD_DIMENSIONS = 6 # 6D operational space

    # OffBit counts for different hardware profiles (used by hardware_profiles.py)
    # These values are aligned with memory limitations and performance expectations.
    OFFBITS_4GB_MOBILE = 10000       # Memory optimized for mobile
    OFFBITS_RASPBERRY_PI5 = 100000   # Balanced for RPi5
    OFFBITS_8GB_IMAC = 1000000       # High performance desktop
    OFFBITS_GOOGLE_COLAB = 2500000   # Optimized for Colab's typical resources
    OFFBITS_KAGGLE = 2000000         # Optimized for Kaggle's typical resources
    OFFBITS_HPC = 10000000           # High-Performance Computing
    OFFBITS_DEVELOPMENT = 10000      # Small for quick testing

    # Bitfield dimension configurations (used by hardware_profiles.py)
    # Dimensions are (X, Y, Z, A, B, C) where X,Y,Z are spatial/primary, A,B,C are conceptual/secondary.
    BITFIELD_6D_FULL = (150, 150, 150, 5, 2, 2)    # Large configuration for high-end systems
    BITFIELD_6D_MEDIUM = (80, 80, 80, 5, 2, 2)     # Medium configuration for balanced systems
    BITFIELD_6D_SMALL = (30, 30, 30, 5, 2, 2)      # Small configuration for memory-constrained systems

    # Harmonic Toggle Resonance (HTR) Parameters
    HTR_DEFAULT_THRESHOLD = 0.05  # Threshold for harmonic resonance detection
    HTR_MAX_ITERATIONS = 1000
    HTR_GENETIC_POPULATION_SIZE = 50
    HTR_GENETIC_GENERATIONS = 100

    # Error Correction Parameters
    NRCI_TARGET_HIGH_COHERENCE = 0.999999  # Target NRCI for optimal coherence
    NRCI_TARGET_STANDARD = 0.9999  # Standard NRCI target
    COHERENCE_THRESHOLD = 0.95  # Minimum coherence for stable operations
    GOLAY_CODE_PARAMS = (23, 12)  # (n, k) for Golay[23,12]
    HAMMING_CODE_PARAMS = (7, 4)  # (n, k) for Hamming[7,4]
    BCH_CODE_PARAMS = (31, 21)  # (n, k) for BCH[31,21]
    REED_SOLOMON_DEFAULT_COMPRESSION_RATIO = 0.30

    # Temporal Mechanics (BitTime)
    BIT_TIME_UNIT_SECONDS = 1e-12  # Base unit of BitTime (picoseconds)
    PLANCK_TIME_SECONDS = 5.391247e-44  # Smallest unit of time
    COHERENT_SYNCHRONIZATION_CYCLE_SECONDS = 1 / PI  # CSC period
    TAUTFLUENCE_TIME_SECONDS = 2.117e-15 # Tautfluence period (empirical)

    # Realm Specific Frequencies / Baselines
    # These are illustrative and should be harmonized with CRVRegistry in ubp_reference_sheet.py
    REALM_FREQ_NUCLEAR = 1.2356e20
    REALM_FREQ_OPTICAL = 5.0e14
    REALM_FREQ_QUANTUM = 4.58e14
    REALM_FREQ_ELECTROMAGNETIC = 3.141593
    REALM_FREQ_GRAVITATIONAL = 100.0
    REALM_FREQ_BIOLOGICAL = 10.0
    REALM_FREQ_COSMOLOGICAL = 1e-11

    # Default performance targets
    DEFAULT_TARGET_OPS_PER_SECOND = 5000
    DEFAULT_MAX_OPERATION_TIME_SECONDS = 1.0
    DEFAULT_VALIDATION_ITERATIONS = 1000

    # Directory Naming
    DATA_DIR_NAME = "data"
    OUTPUT_DIR_NAME = "output"
    TEMP_DIR_NAME = "temp"
    CACHE_DIR_NAME = "cache"
    LOGS_DIR_NAME = "logs"

    # Configuration Defaults for UBPConfig
    UBP_CONFIG_DEFAULT_MEMORY_LIMIT_MB = 1000
    UBP_CONFIG_DEFAULT_PARALLEL_PROCESSING = True
    UBP_CONFIG_DEFAULT_GPU_ACCELERATION = False
    UBP_CONFIG_DEFAULT_CACHE_ENABLED = True
