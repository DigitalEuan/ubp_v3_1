import hashlib
import json
import numpy as np
import os
import pickle
from typing import Any, Dict, Optional, Union

# Define the default directory for PERSISTENT storage for this version of HexDictionary
DEFAULT_HEX_DICT_STORAGE_DIR = "/persistent_state/hex_dictionary_storage/"
DEFAULT_HEX_DICT_METADATA_FILE = os.path.join(DEFAULT_HEX_DICT_STORAGE_DIR, "hex_dict_metadata.json")

class HexDictionary:
    """
    A persistent, content-addressable key-value store.
    Keys are SHA256 hashes of the stored data.
    Supports various data types for serialization.
    This version is specifically configured for persistent storage.
    """
    def __init__(self, storage_dir: str = DEFAULT_HEX_DICT_STORAGE_DIR, metadata_file: str = DEFAULT_HEX_DICT_METADATA_FILE):
        self.storage_dir = storage_dir
        self.metadata_file = metadata_file
        self.entries: Dict[str, Dict[str, Any]] = {}  # Stores {'hash': {'path': 'file', 'type': 'type', 'meta': {}}}
        self._ensure_storage_dir()
        self._load_metadata()
        print(f"Persistent HexDictionary initialized at {self.storage_dir}. Loaded {len(self.entries)} entries.")

    def _ensure_storage_dir(self):
        """Ensures the storage directory exists."""
        os.makedirs(self.storage_dir, exist_ok=True)

    def _load_metadata(self):
        """Loads the HexDictionary metadata from file."""
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                try:
                    self.entries = json.load(f)
                    # Ensure metadata is dict type
                    for key, value in self.entries.items():
                        if 'meta' not in value or not isinstance(value['meta'], dict):
                            value['meta'] = {}
                except json.JSONDecodeError:
                    print("Warning: Persistent HexDictionary metadata file is corrupt. Starting with empty dictionary.")
                    self.entries = {}
        else:
            self.entries = {}

    def _save_metadata(self):
        """Saves the HexDictionary metadata to file."""
        with open(self.metadata_file, 'w') as f:
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
            if isinstance(data, (dict, list)):
                return json.dumps(data).encode('utf-8')
            return data.encode('utf-8')
        elif data_type == 'array' and isinstance(data, np.ndarray):
            return pickle.dumps(data)
        elif data_type == 'list' or data_type == 'dict':
            return json.dumps(data).encode('utf-8')
        else:
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
            if metadata is not None:
                self.entries[data_hash]['meta'].update(metadata)
                self._save_metadata()
                # print(f"Data already exists: {data_hash}. Updated metadata.") # Removed for less verbose output
                
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
        print(f"Clearing all HexDictionary entries and files from {self.storage_dir}...")
        for data_hash in list(self.entries.keys()): # Iterate over a copy as we modify
            self.delete(data_hash)
        
        if os.path.exists(self.metadata_file):
            os.remove(self.metadata_file)
            print(f"Deleted metadata file: {self.metadata_file}")
        
        print("HexDictionary cleared.")

    def __len__(self):
        return len(self.entries)

    def __contains__(self, data_hash: str) -> bool:
        return data_hash in self.entries

if __name__ == "__main__":
    print("--- Testing Persistent HexDictionary ---")
    
    # Ensure a clean slate for testing with persistent paths
    hd_persistent = HexDictionary()
    hd_persistent.clear_all() 

    # Test with persistent paths
    print(f"Persistent HexDictionary storage: {hd_persistent.storage_dir}")
    print(f"Persistent HexDictionary metadata: {hd_persistent.metadata_file}")

    str_data_persistent = "Hello, persistent HexDictionary!"
    str_hash_persistent = hd_persistent.store(str_data_persistent, 'str', metadata={'source': 'test_str_persistent'})
    print(f"String stored with persistent hash: {str_hash_persistent}")
    
    # Verify retrieval
    retrieved_str_persistent = hd_persistent.retrieve(str_hash_persistent)
    print(f"Retrieved persistent string: '{retrieved_str_persistent}' (Matches: {retrieved_str_persistent == str_data_persistent})")
    assert retrieved_str_persistent == str_data_persistent

    # Test persistence (re-initializing)
    print("\n--- Testing persistence (re-initializing persistent HexDictionary) ---")
    del hd_persistent # Close handle
    hd_persistent_reloaded = HexDictionary() # This will now initialize from /persistent_state/
    print(f"Reloaded persistent HexDictionary has {len(hd_persistent_reloaded)} entries.")
    # This should persist across 'Run Experiment' clicks, unlike the /output/ version.
    assert len(hd_persistent_reloaded) == 1 
    reloaded_str_persistent = hd_persistent_reloaded.retrieve(str_hash_persistent)
    print(f"Retrieved string from reloaded persistent dict: '{reloaded_str_persistent}' (Matches: {reloaded_str_persistent == str_data_persistent})")
    assert reloaded_str_persistent == str_data_persistent

    # Clean up after testing, for next full test run
    print("\n--- Clearing persistent HexDictionary for clean test environment ---")
    hd_persistent_reloaded.clear_all()
    assert len(hd_persistent_reloaded) == 0
    assert not os.path.exists(DEFAULT_HEX_DICT_METADATA_FILE)

    print("✅ HexDictionary (Persistent Folder) module test completed successfully!")
