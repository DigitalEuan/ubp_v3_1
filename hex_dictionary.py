"""
Universal Binary Principle (UBP) Framework v3.2+ - HexDictionary Module
Author: Euan Craig, New Zealand
Date: 20 August 2025

This module implements the HexDictionary, a persistent, content-addressable
key-value store for the UBP framework. It uses SHA256 hashing for content
integrity and keys, and supports serialization/deserialization of various
data types for persistence.
"""

import hashlib
import json
import numpy as np
import os
import pickle
from typing import Any, Dict, Optional, Union

# Define the directory for persistent storage
HEX_DICT_STORAGE_DIR = "/output/hex_dictionary_storage/"
HEX_DICT_METADATA_FILE = os.path.join(HEX_DICT_STORAGE_DIR, "hex_dict_metadata.json")

class HexDictionary:
    """
    A persistent, content-addressable key-value store.
    Keys are SHA256 hashes of the stored data.
    Supports various data types for serialization.
    """
    def __init__(self, storage_dir: str = HEX_DICT_STORAGE_DIR):
        self.storage_dir = storage_dir
        self.entries: Dict[str, Dict[str, Any]] = {}  # Stores {'hash': {'path': 'file', 'type': 'type', 'meta': {}}}
        self._ensure_storage_dir()
        self._load_metadata()
        print(f"HexDictionary initialized. Loaded {len(self.entries)} entries.")

    def _ensure_storage_dir(self):
        """Ensures the storage directory exists."""
        os.makedirs(self.storage_dir, exist_ok=True)

    def _load_metadata(self):
        """Loads the HexDictionary metadata from file."""
        if os.path.exists(HEX_DICT_METADATA_FILE):
            with open(HEX_DICT_METADATA_FILE, 'r') as f:
                try:
                    self.entries = json.load(f)
                    # Convert keys from str to actual types if needed,
                    # but for hashes, str is fine.
                    # Ensure metadata is dict type
                    for key, value in self.entries.items():
                        if 'meta' not in value or not isinstance(value['meta'], dict):
                            value['meta'] = {}
                except json.JSONDecodeError:
                    print("Warning: HexDictionary metadata file is corrupt. Starting with empty dictionary.")
                    self.entries = {}
        else:
            self.entries = {}

    def _save_metadata(self):
        """Saves the HexDictionary metadata to file."""
        with open(HEX_DICT_METADATA_FILE, 'w') as f:
            json.dump(self.entries, f, indent=4)

    def _serialize_data(self, data: Any, data_type: str) -> bytes:
        """
        Serializes data into bytes based on the specified data_type.
        Supports common Python types and numpy arrays.
        """
        if data_type == 'bytes':
            return data
        elif data_type == 'str':
            return data.encode('utf-8')
        elif data_type == 'int' or data_type == 'float':
            return str(data).encode('utf-8')
        elif data_type == 'json':
            # Assumes data is already a JSON string or a dict/list
            if isinstance(data, (dict, list)):
                return json.dumps(data).encode('utf-8')
            return data.encode('utf-8') # If it's already a JSON string
        elif data_type == 'array' and isinstance(data, np.ndarray):
            # Use pickle for numpy arrays to preserve dtype and shape
            return pickle.dumps(data)
        elif data_type == 'list' or data_type == 'dict':
            return json.dumps(data).encode('utf-8')
        else:
            # Default to pickle for complex or unknown types
            return pickle.dumps(data)

    def _deserialize_data(self, data_bytes: bytes, data_type: str) -> Any:
        """
        Deserializes bytes back into the original data type.
        """
        if data_type == 'bytes':
            return data_bytes
        elif data_type == 'str':
            return data_bytes.decode('utf-8')
        elif data_type == 'int':
            return int(data_bytes.decode('utf-8'))
        elif data_type == 'float':
            return float(data_bytes.decode('utf-8'))
        elif data_type == 'json':
            return json.loads(data_bytes.decode('utf-8'))
        elif data_type == 'array':
            return pickle.loads(data_bytes)
        elif data_type == 'list' or data_type == 'dict':
            return json.loads(data_bytes.decode('utf-8'))
        else:
            return pickle.loads(data_bytes)

    def store(self, data: Any, data_type: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Stores data in the HexDictionary, using its SHA256 hash as the key.
        
        Args:
            data: The data to store.
            data_type: A string indicating the type of data (e.g., 'str', 'int', 'float',
                       'json', 'array' for numpy, 'bytes', 'list', 'dict').
            metadata: Optional dictionary of additional metadata to store with the entry.
            
        Returns:
            The SHA256 hash (hex string) that serves as the key for the stored data.
        """
        serialized_data = self._serialize_data(data, data_type)
        data_hash = hashlib.sha256(serialized_data).hexdigest()
        
        file_path = os.path.join(self.storage_dir, f"{data_hash}.bin")
        
        if data_hash not in self.entries:
            with open(file_path, 'wb') as f:
                f.write(serialized_data)
            
            self.entries[data_hash] = {
                'path': file_path,
                'type': data_type,
                'meta': metadata if metadata is not None else {}
            }
            self._save_metadata()
            print(f"Stored new entry: {data_hash} (Type: {data_type})")
        else:
            # Data already exists, update metadata if new metadata is provided
            if metadata is not None:
                self.entries[data_hash]['meta'].update(metadata)
                self._save_metadata()
                print(f"Data already exists: {data_hash}. Updated metadata.")
            else:
                print(f"Data already exists: {data_hash}. No new metadata provided.")
                
        return data_hash

    def retrieve(self, data_hash: str) -> Optional[Any]:
        """
        Retrieves data from the HexDictionary using its SHA256 hash.
        
        Args:
            data_hash: The SHA256 hash (hex string) key of the data.
            
        Returns:
            The deserialized data, or None if the hash is not found.
        """
        entry_info = self.entries.get(data_hash)
        if not entry_info:
            print(f"Error: Data with hash '{data_hash}' not found in HexDictionary.")
            return None

        file_path = entry_info['path']
        data_type = entry_info['type']
        
        if not os.path.exists(file_path):
            print(f"Error: Data file for hash '{data_hash}' not found on disk at {file_path}. Removing entry.")
            del self.entries[data_hash]
            self._save_metadata()
            return None

        with open(file_path, 'rb') as f:
            serialized_data = f.read()
        
        try:
            data = self._deserialize_data(serialized_data, data_type)
            return data
        except Exception as e:
            print(f"Error deserializing data for hash '{data_hash}': {e}")
            return None

    def get_metadata(self, data_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves the metadata associated with a stored data entry.
        """
        entry_info = self.entries.get(data_hash)
        if entry_info:
            return entry_info.get('meta')
        return None

    def delete(self, data_hash: str) -> bool:
        """
        Deletes a data entry and its associated file from the HexDictionary.
        
        Args:
            data_hash: The SHA256 hash (hex string) key of the data to delete.
            
        Returns:
            True if the entry was successfully deleted, False otherwise.
        """
        entry_info = self.entries.get(data_hash)
        if not entry_info:
            print(f"Warning: Cannot delete. Data with hash '{data_hash}' not found in HexDictionary.")
            return False

        file_path = entry_info['path']
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted file: {file_path}")
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")
                return False
        
        del self.entries[data_hash]
        self._save_metadata()
        print(f"Deleted entry: {data_hash}")
        return True

    def clear_all(self):
        """
        Clears all entries from the HexDictionary and deletes all stored files.
        """
        print("Clearing all HexDictionary entries and files...")
        for data_hash in list(self.entries.keys()): # Iterate over a copy as we modify
            self.delete(data_hash)
        
        if os.path.exists(HEX_DICT_METADATA_FILE):
            os.remove(HEX_DICT_METADATA_FILE)
            print(f"Deleted metadata file: {HEX_DICT_METADATA_FILE}")
        
        print("HexDictionary cleared.")

    def __len__(self):
        return len(self.entries)

    def __contains__(self, data_hash: str) -> bool:
        return data_hash in self.entries

if __name__ == "__main__":
    print("--- Testing HexDictionary ---")
    
    # Ensure a clean slate for testing
    temp_dict = HexDictionary()
    temp_dict.clear_all()
    del temp_dict # Ensure all handles are closed before re-initializing

    # Initialize HexDictionary for testing
    hd = HexDictionary()
    
    # 1. Test storing various data types
    print("\n--- Storing data ---")
    
    # String
    str_data = "Hello, UBP HexDictionary!"
    str_hash = hd.store(str_data, 'str', metadata={'source': 'test_str'})
    print(f"String stored with hash: {str_hash}")

    # Integer
    int_data = 123456789
    int_hash = hd.store(int_data, 'int', metadata={'source': 'test_int'})
    print(f"Integer stored with hash: {int_hash}")

    # Float
    float_data = 3.1415926535
    float_hash = hd.store(float_data, 'float', metadata={'source': 'test_float'})
    print(f"Float stored with hash: {float_hash}")

    # List (as JSON)
    list_data = ["apple", "banana", "cherry", 123]
    list_hash = hd.store(list_data, 'json', metadata={'source': 'test_list'})
    print(f"List stored with hash: {list_hash}")

    # Dictionary (as JSON)
    dict_data = {"name": "OffBit", "version": 3.2, "active": True}
    dict_hash = hd.store(dict_data, 'json', metadata={'source': 'test_dict'})
    print(f"Dict stored with hash: {dict_hash}")
    
    # NumPy Array
    np_array_data = np.array([[1.1, 2.2], [3.3, 4.4]], dtype=np.float32)
    np_array_hash = hd.store(np_array_data, 'array', metadata={'source': 'test_numpy'})
    print(f"NumPy array stored with hash: {np_array_hash}")

    # 2. Test retrieving data
    print("\n--- Retrieving data ---")
    
    retrieved_str = hd.retrieve(str_hash)
    print(f"Retrieved string: '{retrieved_str}' (Matches: {retrieved_str == str_data})")
    assert retrieved_str == str_data

    retrieved_int = hd.retrieve(int_hash)
    print(f"Retrieved int: {retrieved_int} (Matches: {retrieved_int == int_data})")
    assert retrieved_int == int_data

    retrieved_float = hd.retrieve(float_hash)
    print(f"Retrieved float: {retrieved_float} (Matches: {retrieved_float == float_data})")
    assert retrieved_float == float_data

    retrieved_list = hd.retrieve(list_hash)
    print(f"Retrieved list: {retrieved_list} (Matches: {retrieved_list == list_data})")
    assert retrieved_list == list_data

    retrieved_dict = hd.retrieve(dict_hash)
    print(f"Retrieved dict: {retrieved_dict} (Matches: {retrieved_dict == dict_data})")
    assert retrieved_dict == dict_data

    retrieved_np_array = hd.retrieve(np_array_hash)
    print(f"Retrieved NumPy array:\n{retrieved_np_array}\n(Matches: {np.array_equal(retrieved_np_array, np_array_data)})")
    assert np.array_equal(retrieved_np_array, np_array_data)
    assert retrieved_np_array.dtype == np_array_data.dtype

    # Test non-existent hash
    non_existent_hash = hashlib.sha256(b"non_existent").hexdigest()
    assert hd.retrieve(non_existent_hash) is None

    # 3. Test metadata retrieval
    print("\n--- Retrieving metadata ---")
    meta = hd.get_metadata(str_hash)
    print(f"Metadata for string hash: {meta}")
    assert meta == {'source': 'test_str'}

    # 4. Test persistence by re-initializing and checking
    print("\n--- Testing persistence (re-initializing HexDictionary) ---")
    hd_reloaded = HexDictionary()
    print(f"Reloaded HexDictionary has {len(hd_reloaded)} entries.")
    assert len(hd_reloaded) == len(hd) # Should be 6 entries
    
    reloaded_str = hd_reloaded.retrieve(str_hash)
    print(f"Retrieved string from reloaded dict: '{reloaded_str}' (Matches: {reloaded_str == str_data})")
    assert reloaded_str == str_data

    # 5. Test deletion
    print("\n--- Testing deletion ---")
    assert hd.delete(str_hash)
    assert hd.retrieve(str_hash) is None
    print(f"Current entries after deleting string hash: {len(hd)}")
    assert len(hd) == 5

    # Test deleting non-existent hash
    assert not hd.delete(str_hash) # Should fail gracefully

    # Re-initialize and check deleted entry is gone
    print("\n--- Re-initializing after deletion to confirm persistence of deletion ---")
    hd_reloaded_after_delete = HexDictionary()
    print(f"Reloaded HexDictionary has {len(hd_reloaded_after_delete)} entries.")
    assert len(hd_reloaded_after_delete) == 5 # Should be 5 now
    assert hd_reloaded_after_delete.retrieve(str_hash) is None

    # 6. Test clear_all
    print("\n--- Testing clear_all ---")
    hd.clear_all()
    print(f"Entries after clear_all: {len(hd)}")
    assert len(hd) == 0
    assert not os.path.exists(HEX_DICT_METADATA_FILE)
    assert not os.listdir(HEX_DICT_STORAGE_DIR) # Storage directory should be empty

    # Final re-initialization to confirm everything is clear
    print("\n--- Final re-initialization after clear_all ---")
    hd_final = HexDictionary()
    print(f"Final re-initialized HexDictionary has {len(hd_final)} entries.")
    assert len(hd_final) == 0

    print("\n✅ HexDictionary module test completed successfully!")
