"""
Universal Binary Principle (UBP) Framework v3.1 - Enhanced HexDictionary Module

This module implements the HexDictionary Universal Data Layer, providing
efficient hexadecimal-based data storage and retrieval that integrates
seamlessly with the UBP framework's OffBit structure.

Enhanced for v3.1 with improved performance, better integration with v3.0
components, and expanded data type support.

Author: Euan Craig
Version: 3.1
Date: August 2025
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Set
from dataclasses import dataclass
import json
import hashlib
import struct
import zlib
from collections import defaultdict
import pickle
import time

try:
    from .core import UBPConstants
    from .bitfield import Bitfield, OffBit
except ImportError:
    from core import UBPConstants
    from bitfield import Bitfield, OffBit


@dataclass
class HexEntry:
    """A single entry in the HexDictionary."""
    hex_key: str
    data_type: str
    raw_data: Any
    compressed_data: bytes
    metadata: Dict[str, Any]
    access_count: int
    creation_timestamp: float
    last_access_timestamp: float


@dataclass
class HexDictionaryStats:
    """Statistics for HexDictionary performance and usage."""
    total_entries: int
    total_size_bytes: int
    compression_ratio: float
    average_access_time: float
    cache_hit_rate: float
    most_accessed_keys: List[str]
    data_type_distribution: Dict[str, int]


class HexDictionary:
    """
    Universal Data Layer using hexadecimal-based efficient storage.
    
    This class provides a high-performance data storage and retrieval system
    that can handle various data types (strings, numbers, OffBits, arrays)
    and compress them efficiently using hexadecimal encoding schemes.
    
    Enhanced for v3.1 with better integration and performance.
    """
    
    def __init__(self, max_cache_size: int = 10000, compression_level: int = 6):
        """
        Initialize the HexDictionary.
        
        Args:
            max_cache_size: Maximum number of entries to keep in memory cache
            compression_level: Compression level for data storage (1-9)
        """
        self.entries: Dict[str, HexEntry] = {}
        self.cache: Dict[str, Any] = {}
        self.max_cache_size = max_cache_size
        self.compression_level = compression_level
        
        # Performance tracking
        self.access_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Enhanced data type handlers for v3.1
        self.type_handlers = {
            'string': self._handle_string,
            'integer': self._handle_integer,
            'float': self._handle_float,
            'offbit': self._handle_offbit,
            'array': self._handle_array,
            'bitfield_coords': self._handle_bitfield_coords,
            'json': self._handle_json,
            'binary': self._handle_binary,
            'crv_data': self._handle_crv_data,  # New for v3.1
            'htr_state': self._handle_htr_state,  # New for v3.1
            'realm_config': self._handle_realm_config,  # New for v3.1
            'nrci_metrics': self._handle_nrci_metrics,  # New for v3.1
        }
        
        # Reverse lookup indices
        self.data_type_index: Dict[str, Set[str]] = defaultdict(set)
        self.metadata_index: Dict[str, Set[str]] = defaultdict(set)
        
        print("✅ UBP HexDictionary v3.1 Universal Data Layer Initialized")
        print(f"   Max Cache Size: {max_cache_size:,}")
        print(f"   Compression Level: {compression_level}")
        print(f"   Supported Data Types: {len(self.type_handlers)}")
    
    def _generate_hex_key(self, data: Any, data_type: str) -> str:
        """
        Generate a unique hexadecimal key for the given data.
        
        Args:
            data: Data to generate key for
            data_type: Type of the data
            
        Returns:
            Hexadecimal string key
        """
        # Create a hash of the data and type
        data_str = str(data) + data_type + str(time.time())
        hash_obj = hashlib.sha256(data_str.encode('utf-8'))
        
        # Take first 16 characters of hex digest for efficiency
        hex_key = hash_obj.hexdigest()[:16]
        
        # Ensure uniqueness by checking existing keys
        counter = 0
        original_key = hex_key
        while hex_key in self.entries:
            counter += 1
            hex_key = f"{original_key}_{counter:04x}"
        
        return hex_key
    
    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using zlib."""
        return zlib.compress(data, level=self.compression_level)
    
    def _decompress_data(self, compressed_data: bytes) -> bytes:
        """Decompress data using zlib."""
        return zlib.decompress(compressed_data)
    
    # ========================================================================
    # DATA TYPE HANDLERS
    # ========================================================================
    
    def _handle_string(self, data: str) -> bytes:
        """Handle string data type."""
        return data.encode('utf-8')
    
    def _handle_integer(self, data: int) -> bytes:
        """Handle integer data type."""
        return struct.pack('>q', data)  # Big-endian 64-bit signed integer
    
    def _handle_float(self, data: float) -> bytes:
        """Handle float data type."""
        return struct.pack('>d', data)  # Big-endian 64-bit double
    
    def _handle_offbit(self, data: int) -> bytes:
        """Handle OffBit data type."""
        # Store OffBit as 32-bit integer with layer information
        try:
            layers = OffBit.get_all_layers(data)
            layer_bytes = struct.pack('>IBBBB', data, 
                                     layers['reality'], layers['information'],
                                     layers['activation'], layers['unactivated'])
        except:
            # Fallback for simple integer OffBit
            layer_bytes = struct.pack('>I', data)
        return layer_bytes
    
    def _handle_array(self, data: Union[List, np.ndarray]) -> bytes:
        """Handle array data type."""
        if isinstance(data, np.ndarray):
            return data.tobytes()
        else:
            # Convert list to numpy array and then to bytes
            array = np.array(data)
            return array.tobytes()
    
    def _handle_bitfield_coords(self, data: Tuple[int, ...]) -> bytes:
        """Handle Bitfield coordinates."""
        if len(data) != 6:
            raise ValueError("Bitfield coordinates must be 6-dimensional")
        return struct.pack('>6I', *data)
    
    def _handle_json(self, data: Dict[str, Any]) -> bytes:
        """Handle JSON-serializable data."""
        json_str = json.dumps(data, sort_keys=True)
        return json_str.encode('utf-8')
    
    def _handle_binary(self, data: bytes) -> bytes:
        """Handle raw binary data."""
        return data
    
    # New v3.1 handlers
    def _handle_crv_data(self, data: Dict[str, float]) -> bytes:
        """Handle Core Resonance Value data."""
        return json.dumps(data).encode('utf-8')
    
    def _handle_htr_state(self, data: Dict[str, Any]) -> bytes:
        """Handle HTR engine state data."""
        return json.dumps(data, default=str).encode('utf-8')
    
    def _handle_realm_config(self, data: Dict[str, Any]) -> bytes:
        """Handle realm configuration data."""
        return json.dumps(data, default=str).encode('utf-8')
    
    def _handle_nrci_metrics(self, data: Dict[str, float]) -> bytes:
        """Handle NRCI metrics data."""
        return json.dumps(data).encode('utf-8')
    
    # ========================================================================
    # REVERSE DATA TYPE HANDLERS
    # ========================================================================
    
    def _restore_string(self, data: bytes) -> str:
        """Restore string from bytes."""
        return data.decode('utf-8')
    
    def _restore_integer(self, data: bytes) -> int:
        """Restore integer from bytes."""
        return struct.unpack('>q', data)[0]
    
    def _restore_float(self, data: bytes) -> float:
        """Restore float from bytes."""
        return struct.unpack('>d', data)[0]
    
    def _restore_offbit(self, data: bytes) -> int:
        """Restore OffBit from bytes."""
        if len(data) == 4:
            # Simple integer OffBit
            return struct.unpack('>I', data)[0]
        else:
            # Full OffBit with layers
            offbit_value, reality, info, activation, unactivated = struct.unpack('>IBBBB', data)
            return offbit_value
    
    def _restore_array(self, data: bytes, metadata: Dict[str, Any]) -> np.ndarray:
        """Restore array from bytes."""
        dtype = metadata.get('dtype', 'float64')
        shape = metadata.get('shape', (-1,))
        return np.frombuffer(data, dtype=dtype).reshape(shape)
    
    def _restore_bitfield_coords(self, data: bytes) -> Tuple[int, ...]:
        """Restore Bitfield coordinates from bytes."""
        return struct.unpack('>6I', data)
    
    def _restore_json(self, data: bytes) -> Dict[str, Any]:
        """Restore JSON data from bytes."""
        json_str = data.decode('utf-8')
        return json.loads(json_str)
    
    def _restore_binary(self, data: bytes) -> bytes:
        """Restore binary data."""
        return data
    
    # New v3.1 restore methods
    def _restore_crv_data(self, data: bytes) -> Dict[str, float]:
        """Restore CRV data from bytes."""
        return json.loads(data.decode('utf-8'))
    
    def _restore_htr_state(self, data: bytes) -> Dict[str, Any]:
        """Restore HTR state from bytes."""
        return json.loads(data.decode('utf-8'))
    
    def _restore_realm_config(self, data: bytes) -> Dict[str, Any]:
        """Restore realm config from bytes."""
        return json.loads(data.decode('utf-8'))
    
    def _restore_nrci_metrics(self, data: bytes) -> Dict[str, float]:
        """Restore NRCI metrics from bytes."""
        return json.loads(data.decode('utf-8'))
    
    # ========================================================================
    # CORE OPERATIONS
    # ========================================================================
    
    def store(self, data: Any, data_type: str, metadata: Optional[Dict[str, Any]] = None,
              custom_key: Optional[str] = None) -> str:
        """
        Store data in the HexDictionary.
        
        Args:
            data: Data to store
            data_type: Type of data (must be in supported types)
            metadata: Optional metadata dictionary
            custom_key: Optional custom hex key (if None, auto-generated)
            
        Returns:
            Hexadecimal key for the stored data
        """
        start_time = time.time()
        
        if data_type not in self.type_handlers:
            raise ValueError(f"Unsupported data type: {data_type}")
        
        # Generate or use custom key
        if custom_key:
            if custom_key in self.entries:
                raise ValueError(f"Key {custom_key} already exists")
            hex_key = custom_key
        else:
            hex_key = self._generate_hex_key(data, data_type)
        
        # Handle the data based on its type
        handler = self.type_handlers[data_type]
        raw_bytes = handler(data)
        
        # Add metadata for arrays
        if metadata is None:
            metadata = {}
        
        if data_type == 'array' and isinstance(data, np.ndarray):
            metadata['dtype'] = str(data.dtype)
            metadata['shape'] = data.shape
        
        # Compress the data
        compressed_bytes = self._compress_data(raw_bytes)
        
        # Create entry
        entry = HexEntry(
            hex_key=hex_key,
            data_type=data_type,
            raw_data=data,  # Keep original for cache
            compressed_data=compressed_bytes,
            metadata=metadata,
            access_count=0,
            creation_timestamp=time.time(),
            last_access_timestamp=time.time()
        )
        
        # Store entry
        self.entries[hex_key] = entry
        
        # Update indices
        self.data_type_index[data_type].add(hex_key)
        for key, value in metadata.items():
            self.metadata_index[f"{key}:{value}"].add(hex_key)
        
        # Add to cache
        self._update_cache(hex_key, data)
        
        # Record access time
        access_time = time.time() - start_time
        self.access_times.append(access_time)
        
        return hex_key
    
    def retrieve(self, hex_key: str) -> Any:
        """
        Retrieve data from the HexDictionary.
        
        Args:
            hex_key: Hexadecimal key of the data
            
        Returns:
            Retrieved data in its original form
        """
        start_time = time.time()
        
        # Check cache first
        if hex_key in self.cache:
            self.cache_hits += 1
            self._update_access_stats(hex_key)
            return self.cache[hex_key]
        
        self.cache_misses += 1
        
        # Retrieve from storage
        if hex_key not in self.entries:
            raise KeyError(f"Key {hex_key} not found in HexDictionary")
        
        entry = self.entries[hex_key]
        
        # Decompress data
        raw_bytes = self._decompress_data(entry.compressed_data)
        
        # Restore data based on type
        restore_method_name = f"_restore_{entry.data_type}"
        if hasattr(self, restore_method_name):
            restore_method = getattr(self, restore_method_name)
            if entry.data_type == 'array':
                data = restore_method(raw_bytes, entry.metadata)
            else:
                data = restore_method(raw_bytes)
        else:
            raise ValueError(f"No restore method for data type: {entry.data_type}")
        
        # Update cache
        self._update_cache(hex_key, data)
        
        # Update access statistics
        self._update_access_stats(hex_key)
        
        # Record access time
        access_time = time.time() - start_time
        self.access_times.append(access_time)
        
        return data
    
    def _update_cache(self, hex_key: str, data: Any) -> None:
        """Update the memory cache with new data."""
        # Remove oldest entries if cache is full
        if len(self.cache) >= self.max_cache_size:
            # Remove least recently used entry
            oldest_key = min(self.cache.keys(), 
                           key=lambda k: self.entries[k].last_access_timestamp)
            del self.cache[oldest_key]
        
        self.cache[hex_key] = data
    
    def _update_access_stats(self, hex_key: str) -> None:
        """Update access statistics for an entry."""
        entry = self.entries[hex_key]
        entry.access_count += 1
        entry.last_access_timestamp = time.time()
    
    def delete(self, hex_key: str) -> bool:
        """
        Delete an entry from the HexDictionary.
        
        Args:
            hex_key: Key to delete
            
        Returns:
            True if deleted, False if key not found
        """
        if hex_key not in self.entries:
            return False
        
        entry = self.entries[hex_key]
        
        # Remove from indices
        self.data_type_index[entry.data_type].discard(hex_key)
        for key, value in entry.metadata.items():
            self.metadata_index[f"{key}:{value}"].discard(hex_key)
        
        # Remove from cache
        if hex_key in self.cache:
            del self.cache[hex_key]
        
        # Remove entry
        del self.entries[hex_key]
        
        return True
    
    def exists(self, hex_key: str) -> bool:
        """Check if a key exists in the dictionary."""
        return hex_key in self.entries
    
    def get_entry_info(self, hex_key: str) -> Dict[str, Any]:
        """
        Get detailed information about an entry.
        
        Args:
            hex_key: Key to get info for
            
        Returns:
            Dictionary with entry information
        """
        if hex_key not in self.entries:
            raise KeyError(f"Key {hex_key} not found")
        
        entry = self.entries[hex_key]
        
        return {
            'hex_key': entry.hex_key,
            'data_type': entry.data_type,
            'metadata': entry.metadata,
            'access_count': entry.access_count,
            'creation_timestamp': entry.creation_timestamp,
            'last_access_timestamp': entry.last_access_timestamp,
            'compressed_size_bytes': len(entry.compressed_data),
            'in_cache': hex_key in self.cache
        }
    
    # ========================================================================
    # SEARCH AND QUERY OPERATIONS
    # ========================================================================
    
    def find_by_type(self, data_type: str) -> List[str]:
        """
        Find all keys of a specific data type.
        
        Args:
            data_type: Data type to search for
            
        Returns:
            List of hex keys
        """
        return list(self.data_type_index.get(data_type, set()))
    
    def find_by_metadata(self, metadata_key: str, metadata_value: Any) -> List[str]:
        """
        Find all keys with specific metadata.
        
        Args:
            metadata_key: Metadata key to search for
            metadata_value: Metadata value to match
            
        Returns:
            List of hex keys
        """
        search_key = f"{metadata_key}:{metadata_value}"
        return list(self.metadata_index.get(search_key, set()))
    
    def search(self, query: Dict[str, Any]) -> List[str]:
        """
        Search for entries matching multiple criteria.
        
        Args:
            query: Dictionary with search criteria
                   Supported keys: 'data_type', 'metadata', 'min_access_count'
                   
        Returns:
            List of hex keys matching all criteria
        """
        result_keys = set(self.entries.keys())
        
        # Filter by data type
        if 'data_type' in query:
            type_keys = set(self.find_by_type(query['data_type']))
            result_keys &= type_keys
        
        # Filter by metadata
        if 'metadata' in query:
            for key, value in query['metadata'].items():
                metadata_keys = set(self.find_by_metadata(key, value))
                result_keys &= metadata_keys
        
        # Filter by access count
        if 'min_access_count' in query:
            min_count = query['min_access_count']
            filtered_keys = {key for key in result_keys 
                           if self.entries[key].access_count >= min_count}
            result_keys &= filtered_keys
        
        return list(result_keys)
    
    # ========================================================================
    # PERFORMANCE AND STATISTICS
    # ========================================================================
    
    def get_statistics(self) -> HexDictionaryStats:
        """
        Get comprehensive statistics about the HexDictionary.
        
        Returns:
            HexDictionaryStats object with performance metrics
        """
        total_size = sum(len(entry.compressed_data) for entry in self.entries.values())
        
        # Calculate compression ratio
        if self.entries:
            original_size = sum(len(pickle.dumps(entry.raw_data)) for entry in self.entries.values())
            compression_ratio = total_size / original_size if original_size > 0 else 1.0
        else:
            compression_ratio = 1.0
        
        # Calculate cache hit rate
        total_accesses = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / total_accesses if total_accesses > 0 else 0.0
        
        # Most accessed keys
        most_accessed = sorted(self.entries.keys(), 
                             key=lambda k: self.entries[k].access_count, 
                             reverse=True)[:10]
        
        # Data type distribution
        type_dist = {}
        for entry in self.entries.values():
            type_dist[entry.data_type] = type_dist.get(entry.data_type, 0) + 1
        
        # Average access time
        avg_access_time = np.mean(self.access_times) if self.access_times else 0.0
        
        return HexDictionaryStats(
            total_entries=len(self.entries),
            total_size_bytes=total_size,
            compression_ratio=compression_ratio,
            average_access_time=avg_access_time,
            cache_hit_rate=cache_hit_rate,
            most_accessed_keys=most_accessed,
            data_type_distribution=type_dist
        )
    
    def clear_cache(self) -> None:
        """Clear the memory cache."""
        self.cache.clear()
        print("✅ HexDictionary cache cleared")
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize storage by recompressing data and cleaning up indices.
        
        Returns:
            Dictionary with optimization results
        """
        start_time = time.time()
        original_size = sum(len(entry.compressed_data) for entry in self.entries.values())
        
        # Recompress all entries with maximum compression
        recompressed_count = 0
        for entry in self.entries.values():
            try:
                # Decompress and recompress with level 9
                raw_data = self._decompress_data(entry.compressed_data)
                new_compressed = zlib.compress(raw_data, level=9)
                if len(new_compressed) < len(entry.compressed_data):
                    entry.compressed_data = new_compressed
                    recompressed_count += 1
            except Exception:
                continue
        
        # Clean up indices
        self.data_type_index.clear()
        self.metadata_index.clear()
        
        for hex_key, entry in self.entries.items():
            self.data_type_index[entry.data_type].add(hex_key)
            for key, value in entry.metadata.items():
                self.metadata_index[f"{key}:{value}"].add(hex_key)
        
        new_size = sum(len(entry.compressed_data) for entry in self.entries.values())
        optimization_time = time.time() - start_time
        
        return {
            'recompressed_entries': recompressed_count,
            'size_reduction_bytes': original_size - new_size,
            'size_reduction_percent': ((original_size - new_size) / original_size * 100) if original_size > 0 else 0,
            'optimization_time': optimization_time,
            'indices_rebuilt': True
        }
    
    def export_data(self, file_path: str) -> bool:
        """
        Export all dictionary data to a file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'entries': {key: {
                    'hex_key': entry.hex_key,
                    'data_type': entry.data_type,
                    'compressed_data': entry.compressed_data.hex(),
                    'metadata': entry.metadata,
                    'access_count': entry.access_count,
                    'creation_timestamp': entry.creation_timestamp,
                    'last_access_timestamp': entry.last_access_timestamp
                } for key, entry in self.entries.items()},
                'statistics': self.get_statistics().__dict__
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return True
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False
    
    def import_data(self, file_path: str) -> bool:
        """
        Import dictionary data from a file.
        
        Args:
            file_path: Path to import file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            # Clear existing data
            self.entries.clear()
            self.cache.clear()
            self.data_type_index.clear()
            self.metadata_index.clear()
            
            # Import entries
            for key, entry_data in import_data['entries'].items():
                entry = HexEntry(
                    hex_key=entry_data['hex_key'],
                    data_type=entry_data['data_type'],
                    raw_data=None,  # Will be loaded on demand
                    compressed_data=bytes.fromhex(entry_data['compressed_data']),
                    metadata=entry_data['metadata'],
                    access_count=entry_data['access_count'],
                    creation_timestamp=entry_data['creation_timestamp'],
                    last_access_timestamp=entry_data['last_access_timestamp']
                )
                
                self.entries[key] = entry
                
                # Rebuild indices
                self.data_type_index[entry.data_type].add(key)
                for meta_key, meta_value in entry.metadata.items():
                    self.metadata_index[f"{meta_key}:{meta_value}"].add(key)
            
            print(f"✅ Imported {len(self.entries)} entries from {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Import failed: {e}")
            return False


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def create_hex_dictionary(max_cache_size: int = 10000, compression_level: int = 6) -> HexDictionary:
    """
    Create and return a new HexDictionary instance.
    
    Args:
        max_cache_size: Maximum cache size
        compression_level: Compression level (1-9)
        
    Returns:
        Initialized HexDictionary instance
    """
    return HexDictionary(max_cache_size=max_cache_size, compression_level=compression_level)


def benchmark_hex_dictionary(hex_dict: HexDictionary, num_operations: int = 1000) -> Dict[str, float]:
    """
    Benchmark HexDictionary performance.
    
    Args:
        hex_dict: HexDictionary instance to benchmark
        num_operations: Number of operations to perform
        
    Returns:
        Dictionary with benchmark results
    """
    import random
    
    start_time = time.time()
    
    # Store operations
    store_times = []
    keys = []
    
    for i in range(num_operations):
        data = f"test_data_{i}_{random.random()}"
        store_start = time.time()
        key = hex_dict.store(data, 'string', {'test_id': i})
        store_times.append(time.time() - store_start)
        keys.append(key)
    
    # Retrieve operations
    retrieve_times = []
    for key in keys:
        retrieve_start = time.time()
        hex_dict.retrieve(key)
        retrieve_times.append(time.time() - retrieve_start)
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'average_store_time': np.mean(store_times),
        'average_retrieve_time': np.mean(retrieve_times),
        'operations_per_second': (num_operations * 2) / total_time,
        'cache_hit_rate': hex_dict.cache_hits / (hex_dict.cache_hits + hex_dict.cache_misses)
    }


if __name__ == "__main__":
    # Test the HexDictionary
    print("🧪 Testing HexDictionary v3.1...")
    
    hex_dict = create_hex_dictionary()
    
    # Test basic operations
    key1 = hex_dict.store("Hello, UBP!", 'string')
    key2 = hex_dict.store(42, 'integer')
    key3 = hex_dict.store([1, 2, 3, 4, 5], 'array')
    
    print(f"Stored string: {hex_dict.retrieve(key1)}")
    print(f"Stored integer: {hex_dict.retrieve(key2)}")
    print(f"Stored array: {hex_dict.retrieve(key3)}")
    
    # Test statistics
    stats = hex_dict.get_statistics()
    print(f"Total entries: {stats.total_entries}")
    print(f"Compression ratio: {stats.compression_ratio:.2f}")
    
    print("✅ HexDictionary v3.1 test completed successfully!")

