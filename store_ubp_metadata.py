# store_ubp_metadata.py
import os
import sys
import json
import importlib # Import importlib for module reloading

from hex_dictionary_persistent import HexDictionary # IMPORTANT: Changed to import from hex_dictionary_persistent
import ubp_config # Import ubp_config
# Explicitly reload ubp_config to ensure the latest version is used
importlib.reload(ubp_config)
from ubp_config import get_config # Now import get_config after reloading ubp_config

# ubp_reference_sheet also imports ubp_config, so it's good practice to ensure it also uses the reloaded version.
# If ubp_reference_sheet was already loaded, reload it too.
import ubp_reference_sheet
importlib.reload(ubp_reference_sheet)


print("--- Starting UBP Metadata Storage to HexDictionary ---")

# 1. Ensure UBPConfig is initialized (e.g., for development environment)
# This will ensure CRVs and constants are loaded correctly for ubp_reference_sheet.
config = get_config(environment="development") 
print(f"UBPConfig initialized in '{config.environment}' environment.")

# 2. Initialize HexDictionary and clear it for a fresh start
hex_dict = HexDictionary() # This HexDictionary instance now refers to the persistent one
# It's good practice to clear the hex_dictionary_storage.json file explicitly if a fresh start is desired.
# The HexDictionary __init__ method ensures the storage directory exists and loads metadata,
# but `clear_all()` actually removes all existing data.
hex_dict.clear_all()
print("HexDictionary cleared for a fresh start.")

stored_keys = {}

# 3. Store the entire UBP Reference Data
print("\nStoring UBP Reference Data (Constants, CRVs, etc.)...")
full_reference_data = ubp_reference_sheet.get_full_reference_data()
ref_data_key = hex_dict.store(full_reference_data, 'json', metadata={'description': 'Full UBP Framework Reference Data'})
stored_keys['ubp_reference_data'] = ref_data_key
print(f"Stored 'Full UBP Framework Reference Data' with key: {ref_data_key}")

# 4. Store the list of all Python module names
print("\nStoring list of all Python module names...")
# This list is obtained from the prompt's context provided by the environment.
all_python_modules = [
    'bittime_mechanics.py', 'hex_dictionary.py', 'nuclear_realm.py',
    'optical_realm.py', 'realms.py', 'rgdl_engine.py', 'rune_protocol.py',
    'toggle_algebra.py', 'core_v2.py', 'crv_database.py', 'hardware_profiles.py',
    'system_constants.py', 'ubp_config.py', 'ubp_reference_sheet.py',
    'test_ubp_v31_validation.py', 'UBP_Test_Drive_HexDictionary_Element_Storage.py',
    'UBP_Test_Drive_Complete_Periodic_Table_118_Elements.py',
    'UBP_Test_Drive_Material_Research_Resonant_Steel.py', 'install_deps.py',
    'htr_engine.py', 'bits.py', 'enhanced_crv_selector.py',
    'glr_error_correction.py', 'ubp_framework_v3.py', 'UBP_Test_Drive_System_State_Storage.py',
    'hex_dictionary_persistent.py', 'clear_hex_dictionary.py', 'verify_storage.py',
    'store_ubp_metadata.py', 'offbit_utils.py' # Added self and offbit_utils.py
]
modules_key = hex_dict.store(all_python_modules, 'json', metadata={'description': 'List of all UBP Framework Python Modules'})
stored_keys['ubp_module_list'] = modules_key
print(f"Stored 'List of all UBP Framework Python Modules' with key: {modules_key}")

# 5. Store the entire content of ubp_reference_sheet.py
print("\nStoring content of 'ubp_reference_sheet.py'...")
try:
    # Use the current file's path to read its content
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    reference_sheet_path = os.path.join(current_script_dir, 'ubp_reference_sheet.py')
    with open(reference_sheet_path, 'r') as f:
        reference_sheet_content = f.read()
    content_key = hex_dict.store(reference_sheet_content, 'str', metadata={'description': 'Raw content of ubp_reference_sheet.py'})
    stored_keys['ubp_reference_sheet_content'] = content_key
    print(f"Stored 'ubp_reference_sheet.py content' with key: {content_key}")
except FileNotFoundError:
    print("Error: 'ubp_reference_sheet.py' not found. Cannot store its content.")
except Exception as e:
    print(f"An unexpected error occurred while reading ubp_reference_sheet.py: {e}")

print("\n--- UBP Metadata Storage Complete ---")
print("Successfully stored the following items in HexDictionary (keys for retrieval):")
for name, key in stored_keys.items():
    print(f"- {name}: {key}")
