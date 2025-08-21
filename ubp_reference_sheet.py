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
from system_constants import UBPConstants # Import UBPConstants directly for values not in UBPConfig's dataclasses

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
        'OffBit': 'bits.OffBit', # OffBit dataclass now in bits.py
        'OffBitUtils': 'offbit_utils.OffBitUtils', # New utility class for raw int manipulation
        'RealmManager': 'realms.RealmManager',
        'PlatonicRealm': 'realms.PlatonicRealm',
        'ToggleAlgebra': 'toggle_algebra.ToggleAlgebra',
        'ToggleOperationResult': 'toggle_algebra.ToggleAlgebraOperationResult', # Corrected name from ToggleOperationResult
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
    }
    
    # Configuration Classes
    CONFIG_CLASSES = {
        'HardwareProfile': 'hardware_profiles.HardwareProfile', # Re-added based on existence in hardware_profiles.py
        'HardwareProfileManager': 'hardware_profiles.HardwareProfileManager', # Re-added based on existence in hardware_profiles.py
        'UBPConfig': 'ubp_config.UBPConfig',
        'BitfieldConfig': 'ubp_config.BitfieldConfig',
        'RealmConfig': 'ubp_config.RealmConfig',
        'CRVConfig': 'ubp_config.CRVConfig',
        'ErrorCorrectionConfig': 'ubp_config.ErrorCorrectionConfig',
        'PerformanceConfig': 'ubp_config.Performance.PerformanceConfig', # Corrected path
        'ObserverConfig': 'ubp_config.ObserverConfig',
        'TemporalConfig': 'ubp_config.TemporalConfig',
        'ConstantConfig': 'ubp_config.ConstantConfig', # Re-added ConstantConfig
        'MoleculeConfig': 'ubp_config.MoleculeConfig', # Re-added MoleculeConfig
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
                sub_crv_frequencies=getattr(realm_cfg, 'sub_crvs', []), # Directly from config, with fallback
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
            'PI_UNIVERSAL': constants.PI,
            'E_OVER_12': constants.E / 12, # Derived from E
            'GOLDEN_RATIO': constants.GOLDEN_RATIO,
            'PI_PHI': constants.PI ** constants.GOLDEN_RATIO, # Derived
            'FINE_STRUCTURE_CONSTANT': constants.FINE_STRUCTURE_CONSTANT
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
            ('data', 'np.ndarray', 'Input data for HTR processing'), # Changed from input_data to data
            ('realm', 'Optional[str]', 'Realm to use for processing (optional)'), # Changed from str to Optional[str]
            ('optimize', 'bool', 'Whether to run CRV optimization before processing') # Changed from crv_frequency to optimize
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
            'speed_of_light': constants.SPEED_OF_LIGHT,
            'planck_constant': constants.PLANCK_CONSTANT,
            'fine_structure_constant': constants.FINE_STRUCTURE_CONSTANT,
            'gravitational_constant': constants.GRAVITATIONAL_CONSTANT,
            'elementary_charge': constants.ELEMENTARY_CHARGE
        }
    
    @classmethod
    def get_mathematical_constants(cls) -> Dict[str, float]:
        config_instance = get_config()
        constants = config_instance.constants
        return {
            'pi': constants.PI,
            'e': constants.E,
            'golden_ratio': constants.GOLDEN_RATIO,
            'pi_phi': constants.PI ** constants.GOLDEN_RATIO,
            'e_over_12': constants.E / 12,
        }
    
    @classmethod
    def get_ubp_specific_constants(cls) -> Dict[str, float]:
        config_instance = get_config()
        constants = config_instance.constants
        temporal_config = config_instance.temporal
        
        # Safely get zitterbewegung_frequency if nuclear realm exists
        nuclear_realm_crv = config_instance.realms.get('nuclear')
        zitter_freq = nuclear_realm_crv.main_crv if nuclear_realm_crv else 1.2356e20 # Fallback to common value

        return {
            'c_infinity': constants.C_INFINITY,
            'offbit_energy_unit': constants.OFFBIT_ENERGY_UNIT,
            'epsilon_ubp': constants.EPSILON_UBP,
            'coherent_sync_cycle': temporal_config.COHERENT_SYNCHRONIZATION_CYCLE_PERIOD,
            'bittime_unit_duration': temporal_config.BITTIME_UNIT_DURATION, # Added this constant
            'tautfluence_time': UBPConstants.TAUTFLUENCE_TIME_SECONDS, # From system_constants.py directly
            'zitterbewegung_frequency': zitter_freq 
        }
    
    @classmethod
    def get_nrci_targets(cls) -> Dict[str, float]:
        config_instance = get_config()
        performance = config_instance.performance
        
        # Safely get optical NRCI baseline if realm exists
        optical_realm = config_instance.realms.get('optical')
        optical_nrci = optical_realm.nrci_baseline if optical_realm else 0.9999999
        
        return {
            'standard': performance.TARGET_NRCI,
            'coherence_threshold': performance.COHERENCE_THRESHOLD,
            'minimum_stability': performance.MIN_STABILITY,
            'photonics': optical_nrci,
            'error_correction_base_score': config_instance.error_correction.nrci_base_score
        }
    
    @classmethod
    def get_realm_frequency_ranges(cls) -> Dict[str, Tuple[float, float]]:
        config_instance = get_config()
        ranges = {}
        for realm_name, realm_cfg in config_instance.realms.items():
            if hasattr(realm_cfg, 'frequency_range') and isinstance(realm_cfg.frequency_range, tuple) and len(realm_cfg.frequency_range) == 2 and realm_cfg.frequency_range[0] < realm_cfg.frequency_range[1]:
                ranges[realm_name] = realm_cfg.frequency_range
            else:
                ranges[realm_name] = (0.0, 0.0) # Default if not found or malformed
        return ranges

    @classmethod
    def get_hardware_offbits_defaults(cls) -> Dict[str, Tuple[int, ...]]: # Changed return type to tuple for dimensions
        config_instance = get_config()
        bitfield_config = config_instance.bitfield
        return {
            'mobile': bitfield_config.size_mobile,
            'colab': bitfield_config.size_colab,
            'kaggle': bitfield_config.size_kaggle,
            'raspberry_pi': bitfield_config.size_raspberry_pi, # Explicitly use raspberry_pi config
            'local': bitfield_config.size_local,
            'production': bitfield_config.size_production,
            'current_environment': config_instance.get_bitfield_dimensions() # Get dimensions for current environment
        }

    @classmethod
    def get_error_correction_params(cls) -> Dict[str, Any]: # Changed to Any for mixed types
        config_instance = get_config()
        error_correction_config = config_instance.error_correction
        
        def parse_code(code_str: str) -> Tuple[int, int]:
            try:
                n, k = map(int, code_str.split(','))
                return (n, k)
            except (ValueError, AttributeError):
                return (0, 0) # Default invalid
        
        return {
            'golay_code': parse_code(error_correction_config.golay_code),
            'bch_code': parse_code(error_correction_config.bch_code),
            'hamming_code': parse_code(error_correction_config.hamming_code),
            'padic_prime': error_correction_config.padic_prime,
            'fibonacci_depth': error_correction_config.fibonacci_depth
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
    'optical_realm.py', 'realms.py', 'rgdl_engine.py', 'rune_protocol.py',
    'toggle_algebra.py', 'core_v2.py', 
    'ubp_framework_v3.py', # Renamed main orchestrator
    'crv_database.py', 'hardware_profiles.py', 'system_constants.py', 'ubp_config.py',
    'ubp_reference_sheet.py', 'test_ubp_v31_validation.py', 'UBP_Test_Drive_HexDictionary_Element_Storage.py',
    'UBP_Test_Drive_Complete_Periodic_Table_118_Elements.py', 'UBP_Test_Drive_Material_Research_Resonant_Steel.py',
    'install_deps.py', 'htr_engine.py', 'bits.py', # Removed offbit.py
    'offbit_utils.py', # Added offbit_utils.py
    'glr_error_correction.py', # New module for GLR framework
    'enhanced_crv_selector.py', # Added explicitely as it contains new classes
    'hex_dictionary_persistent.py', # Added this for consistency with provided files
    'clear_hex_dictionary.py', # Added this for consistency with provided files
    'verify_storage.py' # Added this for consistency with provided files
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
        'total_sub_crvs': sum(len(getattr(realm_cfg, 'sub_crvs', [])) for realm_cfg in config_instance.realms.values()),
        'cross_realm_frequencies_count': len(CRVRegistry.get_cross_realm_frequencies_summary()),
        'pi_integration': config_instance.constants.PI == np.pi, # Use dot notation
        'frequency_ranges_valid': all(
            hasattr(realm_cfg, 'frequency_range') and isinstance(realm_cfg.frequency_range, tuple) and len(realm_cfg.frequency_range) == 2 and realm_cfg.frequency_range[0] < realm_cfg.frequency_range[1]
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
