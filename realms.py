"""
Universal Binary Principle (UBP) Framework v3.2+ - Realms Module
Author: Euan Craig, New Zealand
Date: 20 August 2025

This module defines the concept of 'Realms' within the UBP framework,
which are distinct computational environments or domains of reality, each
governed by specific UBP constants, CRVs, and computational geometries.
Realms provide contextual filters and operational parameters for OffBits
and Bitfields.
"""

from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass, field

# Corrected import: UBPConstants is in system_constants, not core_v2
from system_constants import UBPConstants
from ubp_config import get_config, RealmConfig

@dataclass
class Realm:
    """
    Represents a specific computational realm within the UBP framework.
    Each realm has its own set of base parameters, CRVs, and rules.
    """
    name: str
    config: RealmConfig
    is_active: bool = True
    current_crv: float = field(init=False)
    current_wavelength: float = field(init=False)
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Initialize current_crv and current_wavelength from config
        self.current_crv = self.config.main_crv
        self.current_wavelength = self.config.wavelength
        self.meta.update({
            "platonic_solid": self.config.platonic_solid,
            "coordination_number": self.config.coordination_number,
            "spatial_coherence_baseline": self.config.spatial_coherence,
            "temporal_coherence_baseline": self.config.temporal_coherence,
            "nrci_baseline": self.config.nrci_baseline,
            "lattice_type": self.config.lattice_type,
            "optimization_factor": self.config.optimization_factor
        })
        print(f"Realm '{self.name}' initialized. Main CRV: {self.current_crv:.2e} Hz")

    def activate(self):
        """Activates the realm, making it available for computations."""
        self.is_active = True
        print(f"Realm '{self.name}' activated.")

    def deactivate(self):
        """Deactivates the realm."""
        self.is_active = False
        print(f"Realm '{self.name}' deactivated.")

    def update_crv(self, new_crv: float, reason: str = "manual_update"):
        """
        Updates the current CRV for the realm. This could be dynamic based on
        CRV database selections or system conditions.
        """
        self.current_crv = new_crv
        # Re-calculate wavelength if relevant (assuming c = lambda * nu)
        if new_crv > UBPConstants.EPSILON_UBP: # Avoid division by zero
            self.current_wavelength = UBPConstants.SPEED_OF_LIGHT / new_crv
        else:
            self.current_wavelength = float('inf') # Or some other appropriate value
        self.meta["last_crv_update_reason"] = reason
        print(f"Realm '{self.name}' CRV updated to {self.current_crv:.2e} Hz (Reason: {reason})")

    def get_realm_parameters(self) -> Dict[str, Any]:
        """Returns a dictionary of current operational parameters for the realm."""
        params = {
            "name": self.name,
            "is_active": self.is_active,
            "current_crv": self.current_crv,
            "current_wavelength": self.current_wavelength,
        }
        params.update(self.meta)
        return params

    def __str__(self):
        return (f"Realm(name='{self.name}', active={self.is_active}, "
                f"CRV={self.current_crv:.2e} Hz, Wavelength={self.current_wavelength:.2e} m)")

class RealmManager:
    """
    Manages the collection of available realms, their activation status,
    and provides methods for selecting the most appropriate realm for
    a given computational task.
    """
    def __init__(self):
        self.realms: Dict[str, Realm] = {}
        self.ubp_config = get_config()
        self._initialize_realms_from_config()
        print(f"RealmManager initialized with {len(self.realms)} realms.")

    def _initialize_realms_from_config(self):
        """Initializes realms based on the UBPConfig."""
        for realm_name, realm_cfg in self.ubp_config.realms.items():
            self.add_realm(realm_cfg)

    def add_realm(self, realm_config: RealmConfig):
        """Adds a new realm to the manager using its configuration."""
        if realm_config.name.lower() in self.realms:
            print(f"Warning: Realm '{realm_config.name}' already exists. Overwriting.")
        self.realms[realm_config.name.lower()] = Realm(name=realm_config.name, config=realm_config)
        print(f"Added realm: {realm_config.name}")

    def get_realm(self, name: str) -> Optional[Realm]:
        """Retrieves a realm by its name."""
        return self.realms.get(name.lower())

    def get_active_realms(self) -> List[Realm]:
        """Returns a list of all currently active realms."""
        return [realm for realm in self.realms.values() if realm.is_active]

    def select_optimal_realm(self, data_characteristics: Dict[str, Any]) -> Optional[Realm]:
        """
        Selects the most optimal realm based on data characteristics.
        This is a placeholder for a more sophisticated selection algorithm
        that would consider CRV resonance, computational load, etc.
        """
        # For now, a simplified selection based on frequency, prioritizing active realms
        target_freq = data_characteristics.get('frequency', 0.0)
        
        best_realm = None
        min_freq_diff = float('inf')

        active_realms = self.get_active_realms()
        if not active_realms:
            print("No active realms available for selection.")
            return None

        # Simple greedy selection based on closest main_crv to target_freq
        for realm in active_realms:
            diff = abs(realm.current_crv - target_freq)
            if diff < min_freq_diff:
                min_freq_diff = diff
                best_realm = realm
        
        if best_realm:
            print(f"Selected optimal realm: {best_realm.name} (closest CRV to {target_freq:.2e} Hz)")
        else:
            print("Could not select an optimal realm.")

        return best_realm

    def __str__(self):
        return f"RealmManager(num_realms={len(self.realms)}, active_realms={len(self.get_active_realms())})"

if __name__ == "__main__":
    print("--- Testing Realms Module ---")
    
    # Ensure config is initialized (e.g., in development mode for smaller dimensions)
    from ubp_config import get_config
    get_config(environment="development")

    realm_manager = RealmManager()
    print(realm_manager)

    quantum_realm = realm_manager.get_realm("quantum")
    if quantum_realm:
        print(f"\nQuantum Realm details: {quantum_realm}")
        print(f"  CRV: {quantum_realm.current_crv}")
        print(f"  Wavelength: {quantum_realm.current_wavelength}")
        print(f"  Platonic Solid: {quantum_realm.meta['platonic_solid']}")
        
        # Test updating CRV
        quantum_realm.update_crv(1.0e13, "dynamic_adjustment")
        print(f"Updated Quantum Realm: {quantum_realm}")

    em_realm = realm_manager.get_realm("electromagnetic")
    if em_realm:
        print(f"\nEM Realm details: {em_realm}")
        em_realm.deactivate()
        print(f"EM Realm deactivated: {em_realm.is_active}")

    print(f"\nActive realms: {[r.name for r in realm_manager.get_active_realms()]}")

    # Test optimal realm selection
    print("\n--- Testing Realm Selection ---")
    
    # Scenario 1: Target frequency matches Quantum realm
    data_chars_1 = {"frequency": 1.0e13, "complexity": 0.7}
    selected_realm_1 = realm_manager.select_optimal_realm(data_chars_1)
    if selected_realm_1:
        print(f"Selected realm for data_chars_1: {selected_realm_1.name}")
        assert selected_realm_1.name == "quantum"

    # Scenario 2: Target frequency matches Optical realm
    data_chars_2 = {"frequency": 5.0e14, "noise_level": 0.05}
    selected_realm_2 = realm_manager.select_optimal_realm(data_chars_2)
    if selected_realm_2:
        print(f"Selected realm for data_chars_2: {selected_realm_2.name}")
        assert selected_realm_2.name == "optical"

    # Scenario 3: No active realms matching
    if em_realm:
        em_realm.deactivate() # Ensure it's deactivated
    
    # Deactivate all active realms for this test
    for r in realm_manager.get_active_realms():
        r.deactivate()

    data_chars_3 = {"frequency": 1.0e9} # Example for a deactivated realm
    selected_realm_3 = realm_manager.select_optimal_realm(data_chars_3)
    print(f"Selected realm for data_chars_3 (all realms deactivated): {selected_realm_3}")
    assert selected_realm_3 is None

    # Re-activate a realm for further testing
    if quantum_realm:
        quantum_realm.activate()
    
    print("\n✅ Realms module test completed successfully!")
