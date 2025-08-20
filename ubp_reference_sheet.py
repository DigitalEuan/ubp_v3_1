"""
UBP Framework v3.1.1 - Comprehensive Reference Sheet
Author: Euan Craig, New Zealand
Date: 18 August 2025

This is the single source of truth for all UBP Framework v3.0 components:
- Class names and import paths
- CRVs and Sub-CRVs with cross-realm relationships (dynamically loaded from ubp_config)
- System constants and configuration values
- Method signatures and API contracts

This reference sheet prevents integration issues and serves as the central
configuration point for the entire UBP system, dynamically reflecting the
current UBPConfig.
"""

from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import numpy as np
import json # For serialization later

# Import the centralized UBPConfig
from ubp_config import get_config, UBPConfig, RealmConfig

# ============================================================================
# SYSTEM METADATA
# ============================================================================

UBP_VERSION = "3.1" # Updated version to reflect changes
UBP_BUILD_DATE = "2025-08-14" # Updated build date
UBP_AUTHOR = "Euan Craig, New Zealand"

# ============================================================================
# CLASS REGISTRY - Single Source of Truth for All Class Names
# ============================================================================

class UBPClassRegistry:
    """Registry of all UBP Framework class names and their import paths."""
    
    # Core v2.0 Classes (Proven Foundation)
    CORE_CLASSES = {
        'UBPFramework': 'core_v2.UBPFramework', # This is the core operational framework
        'Bitfield': 'bits.Bitfield',
        'OffBit': 'bits.OffBit',
        'RealmManager': 'realms.RealmManager',
        'PlatonicRealm': 'realms.PlatonicRealm',
        'ToggleAlgebra': 'toggle_algebra.ToggleAlgebra',
        'ToggleOperationResult': 'toggle_algebra.ToggleOperationResult',
        'HexDictionary': 'hex_dictionary.HexDictionary',
        'RGDLEngine': 'rgdl_engine.RGDLEngine',
        'NuclearRealm': 'nuclear_realm.NuclearRealm',
        'OpticalRealm': 'optical_realm.OpticalRealm'
    }
    
    # v3.0 Enhancement Classes
    V3_ENHANCEMENT_CLASSES = {
        'EnhancedCRVDatabase': 'crv_database.EnhancedCRVDatabase',
        'CRVProfile': 'crv_database.CRVProfile',
        'SubCRV': 'crv_database.SubCRV',
        'AdaptiveCRVSelector': 'enhanced_crv_selector.AdaptiveCRVSelector',
        'CRVPerformanceMonitor': 'enhanced_crv_selector.CRVPerformanceMonitor',
        'HarmonicPatternAnalyzer': 'enhanced_crv_selector.HarmonicPatternAnalyzer',
        'CRVSelectionResult': 'enhanced_crv_selector.CRVSelectionResult',
        'HTREngine': 'htr_engine.HTREngine',
        'BitTimeMechanics': 'bittime_mechanics.BitTimeMechanics',
        'RuneProtocol': 'rune_protocol.RuneProtocol',
        'ComprehensiveErrorCorrectionFramework': 'glr_error_correction.ComprehensiveErrorCorrectionFramework', # New path
        'GLRMetrics': 'glr_error_correction.GLRMetrics', # New path
        'ErrorCorrectionResult': 'glr_error_correction.ErrorCorrectionResult', # New path
        'LatticeStructure': 'glr_error_correction.LatticeStructure', # New path
        'AutomaticRealmSelector': 'realms.AutomaticRealmSelector', # NEW: Moved here from realm_selector.py
        'DataCharacteristics': 'realms.DataCharacteristics', # NEW: Moved here from realm_selector.py
        'RealmScore': 'realms.RealmScore', # NEW: Moved here from realm_selector.py
        'RealmSelectionResult': 'realms.RealmSelectionResult' # NEW: Moved here from realm_selector.py
    }
    
    # Configuration Classes
    CONFIG_CLASSES = {
        'HardwareProfile': 'hardware_profiles.HardwareProfile',
        'HardwareManager': 'hardware_profiles.HardwareManager',
        'UBPConfig': 'ubp_config.UBPConfig',
        'BitfieldConfig': 'ubp_config.BitfieldConfig',
        'RealmConfig': 'ubp_config.RealmConfig',
        'CRVConfig': 'ubp_config.CRVConfig',
        'HTRConfig': 'ubp_config.HTRConfig',
        'ErrorCorrectionConfig': 'ubp_config.ErrorCorrectionConfig',
        'PerformanceConfig': 'ubp_config.PerformanceConfig',
        'ObserverConfig': 'ubp_config.ObserverConfig',
        'TemporalConfig': 'ubp_config.TemporalConfig',
    }
    
    # Integration Classes (the main orchestrator)
    INTEGRATION_CLASSES = {
        'UBPFrameworkV31Orchestrator': 'ubp_framework_v3.UBPFramework', # Renamed from UBPFrameworkV3 to UBPFramework in module
        'create_ubp_system_function': 'ubp_framework_v3.create_ubp_system' # Reference the factory function
    }
    
    @classmethod
    def get_all_classes(cls) -> Dict[str, str]:
        """Get all registered classes."""
        all_classes = {}
        all_classes.update(cls.CORE_CLASSES)
        all_classes.update(cls.V3_ENHANCEMENT_CLASSES)
        all_classes.update(cls.CONFIG_CLASSES)
        all_classes.update(cls.INTEGRATION_CLASSES)
        return all_classes
    
    @classmethod
    def get_import_path(cls, class_name: str) -> Optional[str]:
        """Get import path for a class name."""
        all_classes = cls.get_all_classes()
        return all_classes.get(class_name)

# ============================================================================
# CRV REGISTRY - Dynamically Loaded from UBPConfig
# ============================================================================

@dataclass
class CRVDefinitionSummary:
    """Summary of CRV definition for the reference sheet."""
    name: str
    frequency: float  # Hz
    wavelength: float  # nm
    realm: str
    geometry: str
    coordination_number: int
    nrci_baseline: float
    sub_crv_frequencies: List[float]
    notes: str

class CRVRegistry:
    """Registry of all CRVs and Sub-CRVs, dynamically loaded from UBPConfig."""
    
    @classmethod
    def get_crv_summary(cls, realm: str) -> Optional[CRVDefinitionSummary]:
        """Get CRV summary for a realm from UBPConfig."""
        config_instance = get_config()
        realm_cfg = config_instance.get_realm_config(realm)
        
        if realm_cfg:
            return CRVDefinitionSummary(
                name=realm_cfg.name + "_main",
                frequency=realm_cfg.main_crv,
                wavelength=realm_cfg.wavelength,
                realm=realm_cfg.name,
                geometry=realm_cfg.platonic_solid, # Use platonic_solid for geometry
                coordination_number=realm_cfg.coordination_number,
                nrci_baseline=realm_cfg.nrci_baseline,
                sub_crv_frequencies=realm_cfg.sub_crvs, # Directly from config
                notes=f"Dynamically loaded from UBPConfig for {realm_cfg.name} realm."
            )
        return None
    
    @classmethod
    def get_all_main_crvs_summary(cls) -> Dict[str, float]:
        """Get all main CRV frequencies from UBPConfig."""
        config_instance = get_config()
        return {realm_name: realm_cfg.main_crv for realm_name, realm_cfg in config_instance.realms.items()}
    
    @classmethod
    def get_cross_realm_frequencies_summary(cls) -> Dict[str, float]:
        """Get cross-realm frequencies (from UBPConfig.constants)."""
        config_instance = get_config()
        constants = config_instance.constants
        cross_realms = {
            'PI_UNIVERSAL': constants['PI'],
            'E_OVER_12': constants['E'] / 12, # Derived from E
            'GOLDEN_RATIO': constants['PHI'],
            'PI_PHI': constants['PI'] ** constants['PHI'], # Derived
            'FINE_STRUCTURE_CONSTANT': constants['FINE_STRUCTURE']
        }
        return cross_realms

# ============================================================================
# METHOD SIGNATURES - API Contracts for All UBP Components
# ============================================================================

class UBPMethodSignatures:
    """Standard method signatures for UBP Framework components."""
    
    # Core computation method signature (example, assuming it's in ubp_framework_v3.py)
    RUN_COMPUTATION_SIGNATURE = {
        'method_name': 'run_computation',
        'parameters': [
            ('operation_type', 'str', 'Type of operation to perform'),
            ('input_data', 'Any', 'Input data array (can be np.ndarray or other type)'),
            ('realm', 'Optional[str]', 'Specific realm to use (None for auto-selection)'),
            ('observer_intent', 'float', 'Observer intent parameter (default: 1.0)'),
            ('enable_htr', 'bool', 'Enable Harmonic Toggle Resonance (default: True)'),
            ('enable_error_correction', 'bool', 'Enable advanced error correction (default: True)')
        ],
        'return_type': 'Dict[str, Any]'
    }
    
    # CRV selection method signature (from enhanced_crv_selector.py)
    SELECT_OPTIMAL_CRV_SIGNATURE = {
        'method_name': 'select_optimal_crv',
        'parameters': [
            ('realm', 'str', 'Target realm for CRV selection'),
            ('data_characteristics', 'Dict[str, Any]', 'Input data characteristics for analysis'),
            ('performance_history', 'Optional[Dict[str, Any]]', 'Historical performance data')
        ],
        'return_type': 'CRVSelectionResult'
    }
    
    # HTR processing method signature (from htr_engine.py)
    HTR_PROCESS_SIGNATURE = {
        'method_name': 'process_with_htr',
        'parameters': [
            ('input_data', 'Any', 'Input data for HTR processing'),
            ('realm', 'str', 'Target realm'),
            ('crv_frequency', 'float', 'CRV frequency to use')
        ],
        'return_type': 'Dict[str, Any]'
    }

# ============================================================================
# SYSTEM CONSTANTS - All UBP Mathematical and Physical Constants
# ============================================================================

class UBPSystemConstants:
    """All UBP system constants in one place, mostly referring to UBPConfig."""
    
    @classmethod
    def get_universal_constants(cls) -> Dict[str, float]:
        config_instance = get_config()
        constants = config_instance.constants
        return {
            'speed_of_light': constants['SPEED_OF_LIGHT'],
            'planck_constant': constants['PLANCK_CONSTANT'],
            'fine_structure_constant': constants['FINE_STRUCTURE'],
            'gravitational_constant': constants['GRAVITATIONAL_CONSTANT'],
            'elementary_charge': constants['ELEMENTARY_CHARGE']
        }
    
    @classmethod
    def get_mathematical_constants(cls) -> Dict[str, float]:
        config_instance = get_config()
        constants = config_instance.constants
        return {
            'pi': constants['PI'],
            'e': constants['E'],
            'golden_ratio': constants['PHI'],
            'pi_phi': constants['PI'] ** constants['PHI'],
            'e_over_12': constants['E'] / 12,
            'golden_ratio_reciprocal': constants['GOLDEN_RATIO_RECIPROCAL'],
            'harmonic_toggle': constants['HARMONIC_TOGGLE']
        }
    
    @classmethod
    def get_ubp_specific_constants(cls) -> Dict[str, float]:
        config_instance = get_config()
        constants = config_instance.constants
        return {
            'c_infinity': constants['C_INFINITY'],
            'offbit_energy_unit': constants['OFFBIT_ENERGY_UNIT'],
            'epsilon_ubp': constants['EPSILON_UBP'],
            'coherent_sync_cycle': constants['CSC_PERIOD'],
            'tautfluence_time': config_instance.temporal.tautfluence_time,
            'zitterbewegung_frequency': config_instance.realms.get('nuclear', RealmConfig(name='dummy', platonic_solid='', coordination_number=0, main_crv=0.0, sub_crvs=[], wavelength=0.0, spatial_coherence=0.0, temporal_coherence=0.0, nrci_baseline=0.0, lattice_type='', optimization_factor=0.0)).main_crv # Pull from nuclear realm CRV
        }
    
    @classmethod
    def get_nrci_targets(cls) -> Dict[str, float]:
        config_instance = get_config()
        return {
            'standard': config_instance.performance.target_nrci,
            'coherence_threshold': config_instance.performance.coherence_threshold,
            'coherence_pressure_min': config_instance.performance.coherence_pressure_min,
            'minimum_stability': config_instance.performance.min_stability,
            'photonics': config_instance.realms.get('optical').nrci_baseline if 'optical' in config_instance.realms else 0.9999999,
            'error_correction_base_score': config_instance.error_correction.nrci_base_score
        }
    
    @classmethod
    def get_realm_frequency_ranges(cls) -> Dict[str, Tuple[float, float]]:
        config_instance = get_config()
        ranges = {}
        for realm_name, realm_cfg in config_instance.realms.items():
            if realm_cfg.frequency_range and len(realm_cfg.frequency_range) == 2:
                ranges[realm_name] = tuple(realm_cfg.frequency_range)
            else:
                ranges[realm_name] = (0.0, 0.0)
        return ranges

    @classmethod
    def get_hardware_offbits_defaults(cls) -> Dict[str, Tuple[int, ...]]: # Changed return type to tuple for dimensions
        config_instance = get_config()
        return {
            'mobile': config_instance.bitfield.size_mobile,
            'colab': config_instance.bitfield.size_colab,
            'kaggle': config_instance.bitfield.size_kaggle,
            'raspberry_pi': config_instance.bitfield.size_mobile, # Assuming mobile config for Pi
            'desktop': config_instance.bitfield.size_local,
            'production': config_instance.bitfield.size_production,
            'current_environment': config_instance.get_bitfield_dimensions() # Get dimensions for current environment
        }

    @classmethod
    def get_error_correction_params(cls) -> Dict[str, Any]: # Changed to Any for mixed types
        config_instance = get_config()
        return {
            'golay_code': tuple(map(int, config_instance.error_correction.golay_code.split(','))),
            'bch_code': tuple(map(int, config_instance.error_correction.bch_code.split(','))),
            'hamming_code': tuple(map(int, config_instance.error_correction.hamming_code.split(','))),
            'padic_prime': config_instance.error_correction.padic_prime,
            'fibonacci_depth': config_instance.error_correction.fibonacci_depth
        }

# ============================================================================
# OPERATION TYPES - All Valid UBP Operations
# ============================================================================

class UBPOperationTypes:
    """Registry of all valid UBP operation types."""
    
    BASIC_OPERATIONS = [
        'energy_calculation',
        'coherence_analysis',
        'nrci_computation',
        'toggle_operation',
        'resonance_analysis'
    ]
    
    ADVANCED_OPERATIONS = [
        'htr_transform',
        'molecular_simulation',
        'genetic_crv_optimization',
        'cross_realm_coherence',
        'bittime_precision'
    ]
    
    GLYPH_OPERATIONS = [
        'glyph_quantify',
        'glyph_correlate',
        'glyph_self_reference'
    ]
    
    ERROR_CORRECTION_OPERATIONS = [
        'golay_encode',
        'golay_decode',
        'padic_encode',
        'fibonacci_encode'
    ]
    
    @classmethod
    def get_all_operations(cls) -> List[str]:
        """Get all valid operation types."""
        all_ops = []
        all_ops.extend(cls.BASIC_OPERATIONS)
        all_ops.extend(cls.ADVANCED_OPERATIONS)
        all_ops.extend(cls.GLYPH_OPERATIONS)
        all_ops.extend(cls.ERROR_CORRECTION_OPERATIONS)
        return all_ops
    
    @classmethod
    def is_valid_operation(cls, operation_type: str) -> bool:
        """Check if operation type is valid."""
        return operation_type in cls.get_all_operations()

# ============================================================================
# IMPORT HELPER FUNCTIONS
# ============================================================================

def get_class_import_statement(class_name: str) -> Optional[str]:
    """Generate import statement for a class."""
    import_path = UBPClassRegistry.get_import_path(class_name)
    if import_path:
        # Special handling for create_ubp_system_function as it's a function not a class
        if class_name == 'create_ubp_system_function':
            module_path, func_name = import_path.rsplit('.', 1)
            return f"from {module_path} import {func_name}"
        else:
            module_path, class_name_in_module = import_path.rsplit('.', 1)
            return f"from {module_path} import {class_name_in_module}"
    return None

# Updated list of available modules reflecting the changes
UBP_AVAILABLE_MODULES = [
    'bittime_mechanics.py', 'hex_dictionary.py', 'nuclear_realm.py',
    'optical_realm.py', 'realms.py', 'rgdl_engine.py', 'rune_protocol.py', # realm_selector.py and automatic_realm_selector.py removed
    'toggle_algebra.py', 'core_v2.py', 
    'ubp_framework_v3.py', # Renamed main orchestrator
    'crv_database.py', 'hardware_profiles.py', 'system_constants.py', 'ubp_config.py',
    'ubp_reference_sheet.py', 'test_ubp_v31_validation.py', 'UBP_Test_Drive_HexDictionary_Element_Storage.py',
    'UBP_Test_Drive_Complete_Periodic_Table_118_Elements.py', 'UBP_Test_Drive_Material_Research_Resonant_Steel.py',
    'install_deps.py', 'htr_engine.py', 'offbit.py', 'bits.py',
    'glr_error_correction.py', # New module for GLR framework
    'enhanced_crv_selector.py' # Added explicitely as it contains new classes
]

def validate_system_integrity() -> Dict[str, bool]:
    """Validate that all registered classes can be "found" (based on available modules)."""
    results = {}
    all_classes = UBPClassRegistry.get_all_classes()
    
    for class_name, import_path in all_classes.items():
        # Special handling for the function, get module path differently
        if class_name == 'create_ubp_system_function':
            module_name_raw = import_path.rsplit('.', 1)[0] # Get module part
        else:
            module_name_raw = import_path.split('.')[0]
        
        # Correctly map module names for validation (e.g., 'core_v2' -> 'core_v2.py')
        module_file_name = f"{module_name_raw}.py"

        if module_file_name in UBP_AVAILABLE_MODULES:
            results[class_name] = True
        else:
            results[class_name] = False
    
    return results

# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

def validate_crv_configuration() -> Dict[str, Any]:
    """Validate CRV configuration integrity, dynamically from UBPConfig."""
    config_instance = get_config()
    validation_results = {
        'total_realms': len(config_instance.realms),
        'total_main_crvs': len(CRVRegistry.get_all_main_crvs_summary()),
        'total_sub_crvs': sum(len(realm_cfg.sub_crvs) for realm_cfg in config_instance.realms.values()),
        'cross_realm_frequencies_count': len(CRVRegistry.get_cross_realm_frequencies_summary()),
        'pi_integration': config_instance.constants['PI'] == np.pi,
        'frequency_ranges_valid': all(
            len(realm_cfg.frequency_range) == 2 and realm_cfg.frequency_range[0] < realm_cfg.frequency_range[1]
            for realm_cfg in config_instance.realms.values()
        )
    }
    
    return validation_results

# ============================================================================
# REFERENCE SHEET DATA EXPORT
# ============================================================================

def get_full_reference_data() -> Dict[str, Any]:
    """
    Gathers all comprehensive reference data into a single dictionary.
    This data can then be serialized and stored in HexDictionary.
    """
    config_instance = get_config()
    
    # 1. System Metadata
    system_metadata = {
        'version': UBP_VERSION,
        'build_date': UBP_BUILD_DATE,
        'author': UBP_AUTHOR
    }
    
    # 2. Class Registry
    class_registry_data = UBPClassRegistry.get_all_classes()
    
    # 3. CRV Registry (summarized dynamically)
    crv_data = {}
    for realm_name in config_instance.realms.keys():
        summary = CRVRegistry.get_crv_summary(realm_name)
        if summary:
            # Convert dataclass to dictionary for JSON serialization
            crv_data[realm_name] = summary.__dict__ 
    
    crv_summary_data = {
        'main_crvs': CRVRegistry.get_all_main_crvs_summary(),
        'cross_realm_frequencies': CRVRegistry.get_cross_realm_frequencies_summary(),
        'realm_details': crv_data
    }
    
    # 4. Method Signatures
    method_signatures_data = {
        'run_computation': UBPMethodSignatures.RUN_COMPUTATION_SIGNATURE,
        'select_optimal_crv': UBPMethodSignatures.SELECT_OPTIMAL_CRV_SIGNATURE,
        'htr_process': UBPMethodSignatures.HTR_PROCESS_SIGNATURE
    }
    
    # 5. System Constants (partially dynamic, partially hardcoded)
    system_constants_data = {
        'universal': UBPSystemConstants.get_universal_constants(),
        'mathematical': UBPSystemConstants.get_mathematical_constants(),
        'ubp_specific': UBPSystemConstants.get_ubp_specific_constants(),
        'nrci_targets': UBPSystemConstants.get_nrci_targets(),
        'realm_frequency_ranges': UBPSystemConstants.get_realm_frequency_ranges(),
        'hardware_offbits_defaults': UBPSystemConstants.get_hardware_offbits_defaults(),
        'error_correction_params': UBPSystemConstants.get_error_correction_params()
    }
    
    # 6. Operation Types
    operation_types_data = {
        'all_operations': UBPOperationTypes.get_all_operations(),
        'basic': UBPOperationTypes.BASIC_OPERATIONS,
        'advanced': UBPOperationTypes.ADVANCED_OPERATIONS,
        'glyph': UBPOperationTypes.GLYPH_OPERATIONS,
        'error_correction': UBPOperationTypes.ERROR_CORRECTION_OPERATIONS
    }
    
    return {
        'system_metadata': system_metadata,
        'class_registry': class_registry_data,
        'crv_registry': crv_summary_data,
        'method_signatures': method_signatures_data,
        'system_constants': system_constants_data,
        'operation_types': operation_types_data,
        'config_snapshot': config_instance.get_summary() # Include a summary of the active config
    }


# ============================================================================
# REFERENCE SHEET SUMMARY
# ============================================================================

def print_reference_sheet_summary():
    """Print a summary of the UBP Reference Sheet."""
    print("🔧 UBP Framework v3.0 Reference Sheet Summary")
    print("=" * 60)
    
    # Get full data to ensure consistency with what's stored
    full_data = get_full_reference_data()

    # System Metadata
    print(f"🚀 System Version: {full_data['system_metadata']['version']}")
    print(f"📅 Build Date: {full_data['system_metadata']['build_date']}")
    print(f"👤 Author: {full_data['system_metadata']['author']}")
    
    # Class Registry Summary
    print(f"\n📚 Total Registered Classes: {len(full_data['class_registry'])}")
    print(f"   - Core v2.0 Classes: {len(UBPClassRegistry.CORE_CLASSES)}")
    print(f"   - v3.0 Enhancement Classes: {len(UBPClassRegistry.V3_ENHANCEMENT_CLASSES)}")
    print(f"   - Configuration Classes: {len(UBPClassRegistry.CONFIG_CLASSES)}")
    print(f"   - Integration Classes: {len(UBPClassRegistry.INTEGRATION_CLASSES)}")
    
    # CRV Registry Summary (dynamic)
    crv_validation = validate_crv_configuration()
    print(f"\n🎯 CRV Registry Summary (Dynamic from UBPConfig):")
    print(f"   - Total Realms: {crv_validation['total_realms']}")
    print(f"   - Main CRVs: {crv_validation['total_main_crvs']}")
    print(f"   - Total Sub-CRVs: {crv_validation['total_sub_crvs']}")
    print(f"   - Cross-Realm Frequencies: {crv_validation['cross_realm_frequencies_count']}")
    print(f"   - π Integration: {'✅' if crv_validation['pi_integration'] else '❌'}")
    print(f"   - Frequency Ranges Valid: {'✅' if crv_validation['frequency_ranges_valid'] else '❌'}")
    
    # Operation Types Summary
    all_operations = UBPOperationTypes.get_all_operations()
    print(f"\n⚙️ Operation Types: {len(all_operations)}")
    print(f"   - Basic Operations: {len(UBPOperationTypes.BASIC_OPERATIONS)}")
    print(f"   - Advanced Operations: {len(UBPOperationTypes.ADVANCED_OPERATIONS)}")
    print(f"   - Glyph Operations: {len(UBPOperationTypes.GLYPH_OPERATIONS)}")
    print(f"   - Error Correction: {len(UBPOperationTypes.ERROR_CORRECTION_OPERATIONS)}")
    
    print("=" * 60)

if __name__ == '__main__':
    # Initialize UBPConfig implicitly by calling get_config before using any dynamic parts
    get_config()
    print_reference_sheet_summary()
