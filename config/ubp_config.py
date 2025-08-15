"""
UBP Framework v3.0 - Centralized Configuration System
Author: Euan Craig, New Zealand
Date: 13 August 2025

Single-point configuration management for all UBP Framework parameters.
All adjustable values are managed here to ensure consistency across modules.
"""

import os
import json
import numpy as np
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

@dataclass
class BitfieldConfig:
    """Bitfield configuration parameters."""
    # Size configuration (environment-dependent)
    size_local: int = 1000000      # Local computer (full performance)
    size_colab: int = 500000       # Google Colab (cloud optimized)
    size_kaggle: int = 300000      # Kaggle (competition ready)
    size_mobile: int = 100000      # Mobile/edge devices
    
    # Structure parameters
    dimensions: int = 6            # 6D operational space
    layers: int = 4               # Reality, Information, Activation, Unactivated
    offbit_size: int = 24         # 24-bit OffBits
    sparsity: float = 0.01        # Sparse matrix optimization
    
    # Memory optimization
    use_sparse_matrix: bool = True
    compression_enabled: bool = True
    compression_ratio: float = 0.30  # Reed-Solomon 30% compression

@dataclass
class RealmConfig:
    """Individual realm configuration."""
    name: str
    main_crv: float
    sub_crvs: List[float]
    wavelength: float
    geometry: str
    coordination_number: int
    nrci_baseline: float
    frequency_range: List[float]  # [min_freq, max_freq]

@dataclass
class CRVConfig:
    """Core Resonance Value configuration."""
    # Main CRVs (refined from research)
    electromagnetic: float = 3.141593        # π-resonance
    quantum: float = 0.2265234857           # e/12
    gravitational: float = 160.19           # Research-validated
    biological: float = 10.0                # 10 Hz base
    cosmological: float = 0.832037          # π^φ
    nuclear: float = 1.2356e20              # Zitterbewegung
    optical: float = 5.0e14                 # 600 nm frequency
    
    # CRV selection parameters
    adaptive_selection: bool = True
    fallback_enabled: bool = True
    performance_monitoring: bool = True
    optimization_enabled: bool = True

@dataclass
class HTRConfig:
    """Harmonic Toggle Resonance configuration."""
    # Molecular simulation
    default_molecule: str = 'propane'
    default_realm: str = 'quantum'
    
    # CRV optimization
    genetic_optimization: bool = True
    optimization_generations: int = 50
    optimization_population: int = 20
    target_bond_energies: Dict[str, float] = None
    
    # Sensitivity analysis
    monte_carlo_runs: int = 500
    noise_level: float = 0.01
    
    # Performance targets
    target_nrci: float = 0.9999999
    max_reconstruction_error: float = 0.05e-9  # 0.05 nm
    
    def __post_init__(self):
        if self.target_bond_energies is None:
            self.target_bond_energies = {
                'propane': 4.8,
                'benzene': 5.0,
                'methane': 4.5,
                'butane': 4.8
            }

@dataclass
class ErrorCorrectionConfig:
    """Error correction system configuration."""
    # GLR Framework
    glr_enabled: bool = True
    golay_code: str = "23,12"     # Golay[23,12]
    bch_code: str = "31,21"       # BCH[31,21]
    hamming_code: str = "7,4"     # Hamming[7,4]
    
    # Advanced encodings
    p_adic_encoding: bool = True
    fibonacci_encoding: bool = True
    reed_solomon_enabled: bool = True
    
    # Correction thresholds
    nrci_threshold: float = 0.999999
    coherence_threshold: float = 0.95
    coherence_pressure_min: float = 0.8

@dataclass
class PerformanceConfig:
    """Performance and optimization configuration."""
    # Computation targets
    target_nrci: float = 0.999999
    max_computation_time: float = 0.1      # seconds per operation
    memory_limit_mb: int = 1000            # Memory usage limit
    
    # Optimization settings
    parallel_processing: bool = True
    gpu_acceleration: bool = False         # Enable if available
    cache_enabled: bool = True
    
    # Monitoring
    real_time_monitoring: bool = True
    performance_logging: bool = True
    benchmark_enabled: bool = True

@dataclass
class ObserverConfig:
    """Observer effect configuration."""
    # Observer intent processing
    intent_enabled: bool = True
    intent_range: List[float] = None       # [min, max] intent values
    purpose_tensor_enabled: bool = True
    
    # Observer effect quantification
    statistical_significance: float = 0.01  # p < 0.01
    observer_factor_base: float = 1.0
    intent_scaling: float = 0.1
    
    def __post_init__(self):
        if self.intent_range is None:
            self.intent_range = [0.0, 2.0]

@dataclass
class TemporalConfig:
    """Temporal coordination configuration."""
    # BitTime mechanics
    planck_time: float = 5.391e-44         # Planck time in seconds
    bit_time: float = 1e-12                # Toggle time scale
    
    # Coherent Synchronization Cycle
    csc_period: float = 0.318309886        # 1/π seconds
    tautfluence_time: float = 2.117e-15    # Tautfluence period
    
    # Temporal alignment
    nist_sync_enabled: bool = True
    temporal_precision: float = 1e-12      # Δt < 10^-12 s

class UBPConfig:
    """
    Centralized UBP Framework v3.0 Configuration System
    
    Single source of truth for all adjustable parameters across the entire system.
    Provides environment detection, parameter validation, and dynamic updates.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize UBP configuration.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration sections
        self.bitfield = BitfieldConfig()
        self.crv = CRVConfig()
        self.htr = HTRConfig()
        self.error_correction = ErrorCorrectionConfig()
        self.performance = PerformanceConfig()
        self.observer = ObserverConfig()
        self.temporal = TemporalConfig()
        
        # Environment detection
        self.environment = self._detect_environment()
        self.working_dir = self._setup_directories()
        
        # Realm configurations
        self.realms = self._initialize_realm_configs()
        
        # Physical constants
        self.constants = self._initialize_constants()
        
        # Load custom configuration if provided
        if config_file and os.path.exists(config_file):
            self.load_config(config_file)
        
        # Apply environment-specific optimizations
        self._apply_environment_optimizations()
        
        self.logger.info(f"UBP Config initialized for {self.environment} environment")
    
    def _detect_environment(self) -> str:
        """Detect execution environment."""
        import sys
        
        if "google.colab" in sys.modules:
            return "colab"
        elif "kaggle" in os.environ.get("KAGGLE_URL_BASE", ""):
            return "kaggle"
        elif os.path.exists("/proc/device-tree/model"):
            # Check for Raspberry Pi
            try:
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                    if "raspberry pi" in model:
                        return "raspberry_pi"
            except:
                pass
        
        return "local"
    
    def _setup_directories(self) -> Dict[str, str]:
        """Setup working directories based on environment."""
        if self.environment == "colab":
            base_dir = "/content"
        elif self.environment == "kaggle":
            base_dir = "/kaggle/working"
        else:
            base_dir = os.getcwd()
        
        directories = {
            'base': base_dir,
            'data': os.path.join(base_dir, 'data'),
            'output': os.path.join(base_dir, 'output'),
            'cache': os.path.join(base_dir, 'cache'),
            'logs': os.path.join(base_dir, 'logs')
        }
        
        # Create directories if they don't exist
        for dir_path in directories.values():
            os.makedirs(dir_path, exist_ok=True)
        
        return directories
    
    def _initialize_realm_configs(self) -> Dict[str, RealmConfig]:
        """Initialize realm configurations with CRVs and Sub-CRVs."""
        realms = {}
        
        # Electromagnetic Realm
        realms['electromagnetic'] = RealmConfig(
            name='electromagnetic',
            main_crv=self.crv.electromagnetic,
            sub_crvs=[2.28e7, 1.570796, 6.283185, 9.424778],
            wavelength=635.0,
            geometry='cube',
            coordination_number=6,
            nrci_baseline=1.0,
            frequency_range=[1e6, 1e12]
        )
        
        # Quantum Realm
        realms['quantum'] = RealmConfig(
            name='quantum',
            main_crv=self.crv.quantum,
            sub_crvs=[6.4444e13, 0.1132617, 0.4530470, 0.6795704],
            wavelength=655.0,
            geometry='tetrahedron',
            coordination_number=4,
            nrci_baseline=0.875,
            frequency_range=[1e13, 1e16]
        )
        
        # Gravitational Realm
        realms['gravitational'] = RealmConfig(
            name='gravitational',
            main_crv=self.crv.gravitational,
            sub_crvs=[11.266, 40.812, 176.09, 0.43693, 0.11748],
            wavelength=1000.0,
            geometry='octahedron',
            coordination_number=8,
            nrci_baseline=0.915,
            frequency_range=[1e-4, 1e4]
        )
        
        # Biological Realm
        realms['biological'] = RealmConfig(
            name='biological',
            main_crv=self.crv.biological,
            sub_crvs=[49.931, 5.0, 20.0, 40.0, 8.0],
            wavelength=700.0,
            geometry='dodecahedron',
            coordination_number=20,
            nrci_baseline=0.911,
            frequency_range=[1e-2, 1e3]
        )
        
        # Cosmological Realm
        realms['cosmological'] = RealmConfig(
            name='cosmological',
            main_crv=self.crv.cosmological,
            sub_crvs=[1.1128e-18, 0.416018, 1.664074, 2.496111],
            wavelength=800.0,
            geometry='icosahedron',
            coordination_number=12,
            nrci_baseline=0.797,
            frequency_range=[1e-18, 1e-10]
        )
        
        # Nuclear Realm
        realms['nuclear'] = RealmConfig(
            name='nuclear',
            main_crv=self.crv.nuclear,
            sub_crvs=[1.6249e16, 6.178e19, 2.4712e20, 3.7068e20],
            wavelength=2.4e-12,
            geometry='e8_g2',
            coordination_number=248,
            nrci_baseline=0.950,
            frequency_range=[1e16, 1e20]
        )
        
        # Optical Realm
        realms['optical'] = RealmConfig(
            name='optical',
            main_crv=self.crv.optical,
            sub_crvs=[1.4398e14, 2.5e14, 1.0e15, 1.5e15],
            wavelength=600.0,
            geometry='hexagonal',
            coordination_number=6,
            nrci_baseline=0.999999,
            frequency_range=[1e14, 1e15]
        )
        
        return realms
    
    def _initialize_constants(self) -> Dict[str, float]:
        """Initialize physical and mathematical constants."""
        return {
            # Mathematical constants
            'PI': np.pi,
            'E': np.e,
            'PHI': (1 + np.sqrt(5)) / 2,  # Golden ratio
            
            # Physical constants
            'LIGHT_SPEED': 299792458,      # m/s
            'PLANCK_CONSTANT': 6.62607015e-34,  # J⋅s
            'FINE_STRUCTURE': 0.0072973525693,  # α
            
            # UBP-specific constants
            'C_INFINITY': 24 * (1 + (1 + np.sqrt(5)) / 2),  # ≈ 38.83
            'R0': 0.95,                    # Base resonance efficiency
            'HT': 0.05,                    # Harmonic threshold
            
            # Temporal constants
            'CSC_PERIOD': 1 / np.pi,       # Coherent Synchronization Cycle
            'TAUTFLUENCE_TIME': 2.117e-15, # Tautfluence period
            'BIT_TIME': 1e-12,             # Toggle time scale
            
            # Error correction thresholds
            'NRCI_TARGET': 0.999999,
            'COHERENCE_THRESHOLD': 0.95,
            'COHERENCE_PRESSURE_MIN': 0.8
        }
    
    def _apply_environment_optimizations(self):
        """Apply environment-specific optimizations."""
        if self.environment == "colab":
            self.bitfield.size_local = self.bitfield.size_colab
            self.performance.memory_limit_mb = 500
            self.performance.gpu_acceleration = True  # Colab has GPU access
            
        elif self.environment == "kaggle":
            self.bitfield.size_local = self.bitfield.size_kaggle
            self.performance.memory_limit_mb = 300
            self.performance.parallel_processing = True
            
        elif self.environment == "raspberry_pi":
            self.bitfield.size_local = self.bitfield.size_mobile
            self.performance.memory_limit_mb = 100
            self.performance.parallel_processing = False
            self.bitfield.compression_enabled = True
            
        else:  # local
            # Use full performance settings
            self.performance.memory_limit_mb = 1000
            self.performance.parallel_processing = True
    
    def get_bitfield_size(self) -> int:
        """Get appropriate bitfield size for current environment."""
        return self.bitfield.size_local
    
    def get_realm_config(self, realm_name: str) -> Optional[RealmConfig]:
        """Get configuration for a specific realm."""
        return self.realms.get(realm_name.lower())
    
    def get_crv(self, realm_name: str, use_main: bool = True) -> float:
        """Get CRV for a realm (main or first sub-CRV)."""
        realm_config = self.get_realm_config(realm_name)
        if not realm_config:
            return self.crv.electromagnetic  # Default fallback
        
        if use_main:
            return realm_config.main_crv
        else:
            return realm_config.sub_crvs[0] if realm_config.sub_crvs else realm_config.main_crv
    
    def get_sub_crvs(self, realm_name: str) -> List[float]:
        """Get all Sub-CRVs for a realm."""
        realm_config = self.get_realm_config(realm_name)
        return realm_config.sub_crvs if realm_config else []
    
    def update_parameter(self, section: str, parameter: str, value: Any):
        """Update a configuration parameter dynamically."""
        if hasattr(self, section):
            section_obj = getattr(self, section)
            if hasattr(section_obj, parameter):
                setattr(section_obj, parameter, value)
                self.logger.info(f"Updated {section}.{parameter} = {value}")
            else:
                self.logger.warning(f"Parameter {parameter} not found in section {section}")
        else:
            self.logger.warning(f"Configuration section {section} not found")
    
    def save_config(self, filepath: str):
        """Save current configuration to file."""
        config_dict = {
            'bitfield': asdict(self.bitfield),
            'crv': asdict(self.crv),
            'htr': asdict(self.htr),
            'error_correction': asdict(self.error_correction),
            'performance': asdict(self.performance),
            'observer': asdict(self.observer),
            'temporal': asdict(self.temporal),
            'environment': self.environment,
            'constants': self.constants
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
        
        self.logger.info(f"Configuration saved to {filepath}")
    
    def load_config(self, filepath: str):
        """Load configuration from file."""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
            
            # Update configuration sections
            for section_name, section_data in config_dict.items():
                if hasattr(self, section_name) and isinstance(section_data, dict):
                    section_obj = getattr(self, section_name)
                    for param, value in section_data.items():
                        if hasattr(section_obj, param):
                            setattr(section_obj, param, value)
            
            self.logger.info(f"Configuration loaded from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
    
    def validate_config(self) -> bool:
        """Validate configuration parameters."""
        valid = True
        
        # Validate bitfield size
        if self.bitfield.size_local <= 0:
            self.logger.error("Bitfield size must be positive")
            valid = False
        
        # Validate CRVs
        for realm_name, realm_config in self.realms.items():
            if realm_config.main_crv <= 0:
                self.logger.error(f"Main CRV for {realm_name} must be positive")
                valid = False
        
        # Validate performance targets
        if self.performance.target_nrci < 0 or self.performance.target_nrci > 1:
            self.logger.error("Target NRCI must be between 0 and 1")
            valid = False
        
        # Validate HTR configuration
        if self.htr.target_nrci < 0 or self.htr.target_nrci > 1:
            self.logger.error("HTR target NRCI must be between 0 and 1")
            valid = False
        
        return valid
    
    def get_summary(self) -> Dict[str, Any]:
        """Get configuration summary."""
        return {
            'environment': self.environment,
            'bitfield_size': self.get_bitfield_size(),
            'realms_count': len(self.realms),
            'crv_adaptive': self.crv.adaptive_selection,
            'htr_enabled': True,
            'error_correction': self.error_correction.glr_enabled,
            'performance_monitoring': self.performance.real_time_monitoring,
            'target_nrci': self.performance.target_nrci,
            'working_directory': self.working_dir['base']
        }

# Global configuration instance
_global_config = None

def get_config() -> UBPConfig:
    """Get global UBP configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = UBPConfig()
    return _global_config

def set_config(config: UBPConfig):
    """Set global UBP configuration instance."""
    global _global_config
    _global_config = config

