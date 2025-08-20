"""
Universal Binary Principle (UBP) Framework v3.2+ - Core Module (Refactored)
Author: Euan Craig, New Zealand
Date: 20 August 2025

This module provides the foundational constants, mathematical definitions,
and core framework integration for the UBP computational system.
It now dynamically loads all constants and realm configurations from
ubp_config.py via the centralized get_config() function, ensuring
a single source of truth and reducing code duplication.
"""

import numpy as np
import math
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

# Import the centralized configuration
from ubp_config import get_config, RealmConfig # Import RealmConfig for type hinting

# Module version, consistent with docstring
__version__ = "2.1.1" # Updated version to reflect changes


# TriangularProjectionConfig class is removed as realms are now exclusively handled by ubp_config.RealmConfig.


class UBPFramework:
    """
    Main framework integration class for the Universal Binary Principle.
    
    This class coordinates all UBP subsystems and provides the primary
    interface for computational operations across multiple realms. It now
    initializes and uses constants and realm configurations from the
    centralized UBPConfig.
    """
    
    def __init__(self, bitfield_dimensions: Optional[Tuple[int, ...]] = None):
        """
        Initialize the UBP Framework with specified Bitfield dimensions.
        
        Args:
            bitfield_dimensions: 6D tuple defining the Bitfield structure.
                                 If None, dimensions are pulled from UBPConfig.
        """
        # Load the centralized UBP configuration
        self.config = get_config()

        # Set bitfield dimensions based on config or provided argument
        self.bitfield_dimensions = bitfield_dimensions if bitfield_dimensions else self.config.get_bitfield_dimensions()
        
        # Realms are now managed by the config instance
        self.realms: Dict[str, RealmConfig] = self.config.realms
        
        self.current_realm = "electromagnetic"  # Default realm, ensure it exists in config
        if self.current_realm not in self.realms:
            # Fallback if default realm isn't in config, pick the first available
            if self.realms:
                self.current_realm = next(iter(self.realms.keys())) 
            else:
                raise RuntimeError("No realms defined in UBPConfig. Framework cannot initialize.")
        
        self.observer_intent = self.config.observer.DEFAULT_INTENT_LEVEL
        
        print(f"✅ UBP Framework v{__version__} Initialized")
        print(f"   Bitfield Dimensions: {self.bitfield_dimensions}")
        print(f"   Active UBPConfig Environment: {self.config.environment}")
        print(f"   Available Realms: {list(self.realms.keys())}")
    
    # Removed _initialize_platonic_realms as realms are loaded from UBPConfig
    # The UBPConfig class already handles this initialization and provides RealmConfig objects.
    
    def get_realm_config(self, realm_name: str) -> RealmConfig:
        """
        Get the configuration for a specific computational realm.
        
        Args:
            realm_name: Name of the realm ("quantum", "electromagnetic", etc.)
            
        Returns:
            RealmConfig for the specified realm
            
        Raises:
            KeyError: If realm_name is not recognized
        """
        # Directly use the config's realm access method
        realm_cfg = self.config.get_realm_config(realm_name)
        if realm_cfg is None:
            available = list(self.realms.keys())
            raise KeyError(f"Unknown realm '{realm_name}'. Available: {available}")
        
        return realm_cfg
    
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
        self.observer_intent = max(self.config.observer.MIN_INTENT_LEVEL, 
                                   min(self.config.observer.MAX_INTENT_LEVEL, intent_level))
        print(f"🎯 Observer intent set to: {self.observer_intent:.3f}")
    
    def calculate_observer_factor(self) -> float:
        """
        Calculate the observer factor based on current intent level.
        
        Returns:
            Observer factor for use in energy calculations
        """
        # Purpose tensor calculation: F_μν(ψ) - Simplified conceptual representation
        # Based on the observer's intent, it influences the "purpose" tensor.
        default_intent = self.config.observer.DEFAULT_INTENT_LEVEL
        if self.observer_intent <= default_intent:
            purpose_tensor = 1.0  # Neutral or passive observation
        else:
            purpose_tensor = 1.0 + (self.observer_intent - default_intent) * self.config.observer.OBSERVER_INFLUENCE_FACTOR # Amplified for intentional
        
        # Observer factor: O_observer = 1 + (1/4π) * log(s/s_0) * F_μν(ψ)
        s_ratio = self.observer_intent / default_intent
        
        # Use EPSILON_UBP from config for robustness
        epsilon_ubp = self.config.constants.EPSILON_UBP
        if s_ratio < epsilon_ubp: # Use < for safety margin if intent is very low
            s_ratio = epsilon_ubp  # Prevent log(0) or log of very small number issues
        
        observer_factor = 1.0 + (1.0 / (4 * self.config.constants.PI)) * np.log(s_ratio) * purpose_tensor
        return observer_factor
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the UBP framework.
        
        Returns:
            Dictionary containing current system state and configuration
        """
        current_config = self.get_realm_config(self.current_realm)
        
        return {
            "framework_version": __version__,
            "bitfield_dimensions": self.bitfield_dimensions,
            "current_realm": self.current_realm,
            "observer_intent": self.observer_intent,
            "observer_factor": self.calculate_observer_factor(),
            "realm_config": {
                "name": current_config.name,
                "platonic_solid": current_config.platonic_solid,
                "main_crv": current_config.main_crv,
                "wavelength": current_config.wavelength,
                "nrci_baseline": current_config.nrci_baseline,
                "optimization_factor": current_config.optimization_factor
            },
            "available_realms": list(self.realms.keys()),
            "constants_snapshot": { # Provide a snapshot of relevant constants from config
                "nrci_target": self.config.performance.TARGET_NRCI,
                "coherence_threshold": self.config.performance.COHERENCE_THRESHOLD,
                "csc_period": self.config.temporal.COHERENT_SYNCHRONIZATION_CYCLE_PERIOD,
                "epsilon_ubp": self.config.constants.EPSILON_UBP,
                "min_stability": self.config.performance.MIN_STABILITY,
                "speed_of_light": self.config.constants.SPEED_OF_LIGHT,
                "planck_constant": self.config.constants.PLANCK_CONSTANT,
                "gravitational_constant": self.config.constants.GRAVITATIONAL_CONSTANT,
                "elementary_charge": self.config.constants.ELEMENTARY_CHARGE,
                "c_infinity": self.config.constants.C_INFINITY,
                "offbit_energy_unit": self.config.constants.OFFBIT_ENERGY_UNIT,
            },
            "ubp_config_environment": self.config.environment
        }


# Global framework instance for easy access
# This will be initialized when the module is imported
_global_framework = None

def get_framework(bitfield_dimensions: Optional[Tuple[int, ...]] = None) -> UBPFramework:
    """
    Get or create the global UBP framework instance.
    
    Args:
        bitfield_dimensions: Optional dimensions for new framework creation.
                             If a framework already exists, this argument is ignored.
        
    Returns:
        Global UBPFramework instance
    """
    global _global_framework
    
    if _global_framework is None:
        _global_framework = UBPFramework(bitfield_dimensions)
    elif bitfield_dimensions is not None and _global_framework.bitfield_dimensions != bitfield_dimensions:
        print(f"⚠️ Warning: UBP Framework already initialized with dimensions {_global_framework.bitfield_dimensions}. "
              f"Ignoring requested dimensions {bitfield_dimensions} for global instance.")
    
    return _global_framework


if __name__ == "__main__":
    # Test the core framework
    # Explicitly get the config first to control environment for testing
    ubp_conf = get_config(environment="development")
    
    # Initialize framework, it will use the global config instance
    framework = UBPFramework()
    
    print("\n" + "="*60)
    print("UBP FRAMEWORK v2.1.1 - REFACATORED CORE MODULE TEST")
    print("="*60)
    
    # Test realm switching
    test_realms = ["quantum", "electromagnetic", "gravitational", "plasma", "nuclear", "optical", "biologic"]
    for realm in test_realms:
        try:
            framework.set_current_realm(realm)
            config = framework.get_realm_config(realm)
            print(f"  {config.name}: {config.platonic_solid}, Main CRV={config.main_crv:.6e}, Wavelength={config.wavelength}m")
        except KeyError as e:
            print(f"  Skipping realm '{realm}': {e}")
    
    # Test observer intent
    framework.set_observer_intent(1.5)
    
    # Display system status
    status = framework.get_system_status()
    print(f"\nSystem Status:")
    print(f"  Current Realm: {status['current_realm']}")
    print(f"  Observer Factor: {status['observer_factor']:.6f}")
    print(f"  NRCI Target: {status['constants_snapshot']['nrci_target']}")
    print(f"  Framework Version: {status['framework_version']}")
    print(f"  Epsilon UBP: {status['constants_snapshot']['epsilon_ubp']:.2e}")
    print(f"  Min Stability: {status['constants_snapshot']['min_stability']}")
    print(f"  C_INFINITY: {status['constants_snapshot']['c_infinity']:.6e}")
    print(f"  OffBit Energy Unit: {status['constants_snapshot']['offbit_energy_unit']:.2e}")
    print(f"  UBPConfig Environment: {status['ubp_config_environment']}")

    # Verify a constant loaded directly from config
    print(f"  Verified Planck Constant: {framework.config.constants.PLANCK_CONSTANT:.2e}")
    
    print("\n✅ Refactored Core module test completed successfully!")
