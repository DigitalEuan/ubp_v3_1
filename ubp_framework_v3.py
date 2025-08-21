"""
Universal Binary Principle (UBP) Framework v3.2+ - Main Orchestration Module
Author: Euan Craig, New Zealand
Date: 20 August 2025

This module serves as the primary entry point for the UBP Framework, integrating
the core UBP logic, HexDictionary data layer, and GLR error correction subsystem.
It orchestrates interactions between these components and provides a high-level
interface for computational operations.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import json
import time

# Import core UBP framework logic (realm management, observer, config access)
from core_v2 import UBPFramework as CoreUBPFramework # Renamed to avoid conflict with local UBPFramework dataclass
# Import Bitfield and OffBit (assuming they are in bits.py)
from bits import Bitfield, OffBit 
# Import HexDictionary
from hex_dictionary import HexDictionary
# Import GLR Error Correction Framework (now its own module)
from glr_error_correction import ComprehensiveErrorCorrectionFramework, GLRMetrics, ErrorCorrectionResult
# Import UBPConfig for overall system configuration
from ubp_config import get_config


@dataclass
class UBPFramework: # This is the main orchestrator/container dataclass
    """
    A comprehensive UBP Framework instance, holding all core components.
    This acts as the orchestrator for the entire system.
    """
    core_ubp: CoreUBPFramework # The core logic (realm management, observer)
    bitfield: Bitfield
    hex_dictionary: HexDictionary
    glr_framework: ComprehensiveErrorCorrectionFramework
    
    # Additional components/states managed by the orchestrator
    config: Any = field(init=False) # Will be initialized from core_ubp
    
    def __post_init__(self):
        self.config = self.core_ubp.config # Link to the shared UBPConfig instance
        print(f"✅ UBPFramework Orchestrator v3.2 Initialized.")
        print(f"   Core UBP Framework Version: {self.core_ubp.get_system_status()['framework_version']}")
        print(f"   Bitfield Dimensions: {self.bitfield.dimensions}")
        print(f"   HexDictionary Entries: {len(self.hex_dictionary.entries)}")
        print(f"   GLR Realm: {self.glr_framework.realm_name}")

    def run_computation(self, operation_type: str, input_data: List[Union[int, OffBit]], 
                       realm: Optional[str] = None, observer_intent: float = 1.0,
                       enable_htr: bool = False, enable_error_correction: bool = True) -> Dict[str, Any]:
        """
        Executes a computational operation within the UBP framework.
        
        Args:
            operation_type: Type of operation (e.g., 'process_offbits', 'correct_data').
            input_data: List of OffBit objects or raw OffBit values (integers).
            realm: Optional specific realm to use. If None, uses current_realm.
            observer_intent: Observer intent level (0.0 to 2.0).
            enable_htr: Flag to enable Harmonic Toggle Resonance (HTR).
            enable_error_correction: Flag to enable GLR error correction.
            
        Returns:
            Dictionary containing computation results and metrics.
        """
        start_time = time.time()
        results = {"status": "success", "message": "Computation completed."}
        
        # 1. Set realm and observer intent on the core framework
        current_realm_name = realm if realm else self.core_ubp.current_realm
        if realm and current_realm_name != self.core_ubp.current_realm:
            self.core_ubp.set_current_realm(current_realm_name)
            self.glr_framework.switch_realm(current_realm_name) # Sync GLR framework's realm
            
        self.core_ubp.set_observer_intent(observer_intent)
        
        results['initial_system_status'] = self.core_ubp.get_system_status()

        # Convert input_data to a list of OffBit values (np.uint32) for processing
        processed_data_values: List[np.uint32] = []
        for item in input_data:
            if isinstance(item, OffBit):
                processed_data_values.append(item.value)
            elif isinstance(item, int):
                # Ensure it's converted to np.uint32 as expected by OffBit or Bitfield
                processed_data_values.append(np.uint32(item))
            else:
                raise TypeError(f"Input data must be a list of OffBit objects or integers, got {type(item)}")
        
        # Use a copy for processing that might modify
        processed_data_values_mutable = list(processed_data_values) 

        # 2. Apply HTR (conceptual for this module, requires htr_engine integration)
        if enable_htr:
            print("Applying conceptual HTR transformation...")
            # Placeholder for HTR integration:
            # if self.htr_engine:
            #     processed_data_values_mutable = self.htr_engine.process(processed_data_values_mutable, current_realm_name)
            results['htr_applied'] = True
        else:
            results['htr_applied'] = False

        # 3. Apply GLR Error Correction
        if enable_error_correction:
            print(f"Applying GLR Error Correction in {self.glr_framework.realm_name} realm...")
            # `correct_spatial_errors` expects a list of OffBit values (integers)
            spatial_correction_result = self.glr_framework.correct_spatial_errors(processed_data_values_mutable)
            processed_data_values_mutable = spatial_correction_result.corrected_offbits
            results['glr_spatial_correction'] = {
                'applied': spatial_correction_result.correction_applied,
                'errors_corrected': spatial_correction_result.error_count,
                'nrci_improvement': spatial_correction_result.nrci_improvement
            }
            # Calculate and store overall GLR metrics
            glr_metrics = self.glr_framework.calculate_comprehensive_metrics(processed_data_values_mutable, None) # No temporal sequence in this example
            results['glr_current_metrics'] = glr_metrics.__dict__
        else:
            results['glr_applied'] = False

        # 4. Store processed data in HexDictionary
        if processed_data_values_mutable:
            # HexDictionary now uses SHA256 hashing; the 'store' method handles this.
            # Convert np.uint32 list to a regular Python list for JSON/array storage consistency if needed.
            # Using 'array' type for numpy array serialization in HexDictionary.
            data_to_store_for_hex = np.array(processed_data_values_mutable, dtype=np.uint32)

            hex_key = self.hex_dictionary.store(data_to_store_for_hex, 'array', metadata={'operation': operation_type, 'realm': current_realm_name})
            results['hex_dictionary_key'] = hex_key
            results['hex_dictionary_stored_count'] = len(processed_data_values_mutable)
            print(f"Processed data stored in HexDictionary with key: {hex_key}")
        else:
            print("No processed data to store in HexDictionary.")


        # 5. Final system status and computation metrics
        results['final_system_status'] = self.core_ubp.get_system_status()
        # For snapshot, convert back to OffBit objects for display if useful, or just show values
        results['processed_data_snapshot'] = [int(val) for val in processed_data_values_mutable[:5]] + ['...'] if len(processed_data_values_mutable) > 5 else [int(val) for val in processed_data_values_mutable]
        results['total_execution_time_s'] = time.time() - start_time
        
        return results

# ========================================================================
# MAIN UBP FRAMEWORK CREATION FUNCTION
# ========================================================================

def create_ubp_system(
    bitfield_size: int = 5000, # This argument is largely symbolic here, Bitfield dimensions now come from config.
    default_realm: str = "electromagnetic", 
    enable_error_correction: bool = True,
    enable_htr: bool = False, # HTREngine is not instantiated here yet
    enable_rgdl: bool = False # RGDLEngine is not instantiated here yet
) -> UBPFramework:
    """
    Creates and initializes a comprehensive UBP Framework v3.2 instance,
    integrating key components like Core UBP, Bitfield, HexDictionary, and GLR error correction.
    
    Args:
        bitfield_size: The logical size for the Bitfield. (Note: Actual dimensions from UBPConfig).
        default_realm: The initial default realm for the system.
        enable_error_correction: Whether to enable GLR error correction.
        enable_htr: Flag to indicate if HTR should be conceptually enabled (HTR engine not instantiated here).
        enable_rgdl: Flag to indicate if RGDL should be conceptually enabled (RGDLEngine not instantiated here).
        
    Returns:
        A UBPFramework instance, serving as the orchestrator.
    """
    print(f"Initializing UBP System v3.2 with bitfield_size={bitfield_size} (config-driven), default_realm={default_realm}...")

    # Ensure UBPConfig is loaded before any components that depend on it
    # Use 'development' environment for tests by default, or you can pass 'production'/'testing'
    config = get_config(environment="development") 
    print(f"  UBPConfig loaded (Environment: {config.environment}).")

    # 1. Initialize Core UBP Framework (manages realms, observer, global config access)
    core_ubp_instance = CoreUBPFramework()
    print("  Core UBP Framework initialized.")
    
    # Set the initial realm for the core framework
    core_ubp_instance.set_current_realm(default_realm)

    # 2. Initialize Bitfield
    bitfield_dimensions = core_ubp_instance.bitfield_dimensions
    bitfield = Bitfield(dimensions=bitfield_dimensions) # Bitfield uses dimensions tuple
    print(f"  Bitfield initialized with dimensions: {bitfield.dimensions}")

    # 3. Initialize HexDictionary
    hex_dict = HexDictionary()
    print("  HexDictionary initialized.")

    # 4. Initialize GLR Error Correction Framework
    # Need to pass the config object or relevant parts to GLR if it depends on them.
    # For now, it will pull config via its own get_config()
    glr = ComprehensiveErrorCorrectionFramework(
        realm_name=default_realm,
        enable_error_correction=enable_error_correction,
        hex_dictionary_instance=hex_dict # Pass hex_dictionary instance
    )
    print("  GLR Error Correction Framework initialized.")

    # Note: HTR and RGDL engines would be instantiated here if they were separate
    # modules and part of this top-level orchestration. Their enablement flags
    # are processed conceptually for now.
    if enable_htr:
        print("  HTR Engine enablement requested (integration point for future HTREngine instance).")
    if enable_rgdl:
        print("  RGDL Engine enablement requested (integration point for future RGDLEngine instance).")

    # 5. Create and return the comprehensive UBPFramework orchestrator instance
    framework_instance = UBPFramework(
        core_ubp=core_ubp_instance,
        bitfield=bitfield,
        hex_dictionary=hex_dict,
        glr_framework=glr
    )
    print("UBP System v3.2 Orchestrator instance created successfully.")
    return framework_instance


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🧪 Testing UBP System v3.2 Orchestrator Module")
    print("="*60)
    
    # Create the UBP system
    # For testing, ensure hex_dictionary is clean
    HexDictionary().clear_all() 

    ubp_system = create_ubp_system(
        bitfield_size=1000, # This argument is illustrative, actual dimensions from config.
        default_realm="quantum",
        enable_error_correction=True
    )
    
    # Example usage: Run a computation
    # Create some dummy OffBit values (or raw integers)
    # Let's create some with deliberate "errors" for GLR to potentially correct
    test_offbits_input = [
        OffBit.create_offbit(reality_state=1, information_payload=2, coherence_phase=3, observer_context=4), # Good
        OffBit.create_offbit(reality_state=5, information_payload=10, coherence_phase=60, observer_context=15), # High coherence
        OffBit.create_offbit(reality_state=2, information_payload=8, coherence_phase=4, observer_context=7), # Good
        OffBit.create_offbit(reality_state=0, information_payload=0, coherence_phase=0, observer_context=0), # Empty
        OffBit.create_offbit(reality_state=63, information_payload=63, coherence_phase=63, observer_context=63), # Maxed
        OffBit.create_offbit(reality_state=10, information_payload=10, coherence_phase=10, observer_context=10), # Another good
        OffBit.create_offbit(reality_state=1, information_payload=1, coherence_phase=1, observer_context=1) # Another good
    ]
    # Add a few raw integer values to test the type handling in run_computation
    test_offbits_input.append(int(OffBit.create_offbit(reality_state=10, information_payload=20, coherence_phase=30, observer_context=40).value))
    test_offbits_input.append(int(OffBit.create_offbit(reality_state=11, information_payload=21, coherence_phase=31, observer_context=41).value))


    # Simulate running a computation
    print("\n--- Running a sample computation ---")
    results = ubp_system.run_computation(
        operation_type="offbit_processing",
        input_data=test_offbits_input,
        realm="quantum",
        observer_intent=1.5,
        enable_htr=False,
        enable_error_correction=True
    )
    
    print("\nComputation Results Summary:")
    print(f"  Status: {results['status']}")
    print(f"  Total Execution Time: {results['total_execution_time_s']:.4f} seconds")
    print(f"  GLR Spatial Correction Applied: {results.get('glr_spatial_correction', {}).get('applied')}")
    print(f"  Errors Corrected (Spatial): {results.get('glr_spatial_correction', {}).get('errors_corrected')}")
    print(f"  HexDictionary Key for Processed Data: {results.get('hex_dictionary_key')}")
    print(f"  Final Combined NRCI: {results.get('glr_current_metrics', {}).get('nrci_combined'):.3f}")
    
    # Demonstrate retrieving from HexDictionary
    if 'hex_dictionary_key' in results and results['hex_dictionary_key']:
        retrieved_data_np = ubp_system.hex_dictionary.retrieve(results['hex_dictionary_key'])
        if retrieved_data_np is not None:
            # retrieved_data_np is a numpy array (dtype=uint32)
            retrieved_offbits_count = len(retrieved_data_np)
            print(f"  Retrieved {retrieved_offbits_count} items from HexDictionary.")
            # Convert a few back to OffBit objects for verification
            if retrieved_offbits_count > 0:
                first_retrieved_offbit = OffBit(retrieved_data_np[0])
                print(f"  First retrieved OffBit value (from HexDict): {first_retrieved_offbit.value}")
                print(f"  First retrieved OffBit meta (from HexDict): {first_retrieved_offbit.meta}")
        
    print("\n--- Checking System Status ---")
    system_status = ubp_system.core_ubp.get_system_status()
    print(f"  Current Realm: {system_status['current_realm']}")
    print(f"  Observer Factor: {system_status['observer_factor']:.3f}")
    print(f"  GLR Framework's Current Realm: {ubp_system.glr_framework.realm_name}")

    # Export GLR metrics
    ubp_system.glr_framework.export_metrics("orchestrator_glr_metrics.json")
    
    print("\n✅ UBP System Orchestrator test completed successfully!")
