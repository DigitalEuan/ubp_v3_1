"""
Universal Binary Principle (UBP) Framework v3.1 - Enhanced Bitfield Module

This module provides a compatibility layer that combines the best of v2.0 and v3.0
Bitfield implementations, ensuring seamless integration across all UBP components.

Author: Euan Craig
Version: 3.1
Date: August 2025
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import json
import time

# Import the v2.0 OffBit class which is excellent
try:
    from .bitfield_v2 import OffBit
except ImportError:
    from bitfield_v2 import OffBit


@dataclass
class BitfieldStats:
    """Statistics for Bitfield performance and state."""
    total_offbits: int
    active_offbits: int
    memory_usage_mb: float
    coherence_level: float
    last_update: float


class Bitfield:
    """
    Enhanced UBP Bitfield v3.1 - Compatible with both v2.0 and v3.0 interfaces.
    
    This class provides a unified interface that supports:
    - v2.0 style initialization with dimensions and sparsity
    - v3.0 style initialization with size parameter
    - Enhanced performance and compatibility
    """
    
    def __init__(self, 
                 dimensions: Optional[Tuple[int, int, int, int, int, int]] = None,
                 sparsity: float = 0.01,
                 size: Optional[int] = None):
        """
        Initialize the Bitfield with flexible parameter support.
        
        Args:
            dimensions: 6D tuple for v2.0 compatibility (optional)
            sparsity: Fraction of OffBits to initialize as active
            size: Total number of OffBits for v3.0 compatibility (optional)
        """
        # Handle different initialization modes
        if size is not None:
            # v3.0 style initialization - convert size to dimensions
            self.size = size
            self.dimensions = self._size_to_dimensions(size)
        elif dimensions is not None:
            # v2.0 style initialization
            self.dimensions = dimensions
            self.size = np.prod(dimensions)
        else:
            # Default initialization
            self.dimensions = (32, 32, 32, 4, 2, 2)  # Default for testing
            self.size = np.prod(self.dimensions)
        
        self.sparsity = sparsity
        
        # Initialize the 6D grid
        self.grid = np.zeros(self.dimensions, dtype=np.uint32)
        
        # Calculate total OffBits
        self.total_offbits = self.size
        
        # Initialize metadata
        self.creation_timestamp = time.time()
        self.modification_count = 0
        self.stats = BitfieldStats(
            total_offbits=self.total_offbits,
            active_offbits=0,
            memory_usage_mb=self.get_memory_usage_mb(),
            coherence_level=0.0,
            last_update=self.creation_timestamp
        )
        
        # Initialize with sparse activation
        self._initialize_sparse_activation()
        
        print(f"✅ UBP Bitfield v3.1 Initialized")
        print(f"   Dimensions: {self.dimensions}")
        print(f"   Total OffBits: {self.total_offbits:,}")
        print(f"   Memory Usage: {self.get_memory_usage_mb():.2f} MB")
        print(f"   Sparsity: {self.sparsity:.3f}")
    
    def _size_to_dimensions(self, size: int) -> Tuple[int, int, int, int, int, int]:
        """
        Convert a total size to 6D dimensions.
        
        Args:
            size: Total number of OffBits desired
            
        Returns:
            6D tuple of dimensions
        """
        # Calculate dimensions that approximate the desired size
        # Using a balanced approach for 6D space
        
        if size <= 1000:
            # Small size - use compact dimensions
            base = int(size ** (1/6)) + 1
            return (base, base, base, 2, 2, 2)
        elif size <= 100000:
            # Medium size
            base = int((size / 8) ** (1/4)) + 1
            return (base, base, base, 4, 2, 2)
        else:
            # Large size - use production-like dimensions
            base = int((size / 40) ** (1/4)) + 1
            return (base, base, base, 5, 2, 2)
    
    def _initialize_sparse_activation(self):
        """Initialize the Bitfield with sparse activation pattern."""
        if self.sparsity <= 0:
            return
        
        # Calculate number of OffBits to activate
        num_active = int(self.total_offbits * self.sparsity)
        
        # Generate random coordinates for activation
        # Ensure we don't exceed the actual grid size
        actual_size = np.prod(self.dimensions)
        safe_num_active = min(num_active, actual_size)
        
        if safe_num_active > 0:
            flat_indices = np.random.choice(actual_size, size=safe_num_active, replace=False)
            
            for flat_idx in flat_indices:
                coords = np.unravel_index(flat_idx, self.dimensions)
                
                # Create a random OffBit with some activation
                reality = np.random.randint(0, 64)
                information = np.random.randint(0, 64)
                activation = np.random.randint(16, 64)  # Ensure some activation
                unactivated = np.random.randint(0, 32)
                
                offbit_value = OffBit.create_offbit(reality, information, activation, unactivated)
                self.grid[coords] = offbit_value
        
        self.stats.active_offbits = safe_num_active
        self.stats.last_update = time.time()
    
    # ========================================================================
    # CORE BITFIELD OPERATIONS
    # ========================================================================
    
    def get_offbit(self, coords: Tuple[int, ...]) -> int:
        """
        Retrieve the 32-bit OffBit value at the given coordinates.
        
        Args:
            coords: 6D coordinates tuple
            
        Returns:
            32-bit integer representing the OffBit
        """
        if len(coords) != 6:
            raise ValueError(f"Expected 6D coordinates, got {len(coords)}D")
        
        # Ensure coordinates are within bounds
        for i, (coord, dim) in enumerate(zip(coords, self.dimensions)):
            if coord < 0 or coord >= dim:
                raise IndexError(f"Coordinate {i} ({coord}) out of bounds for dimension {dim}")
        
        return int(self.grid[coords])
    
    def set_offbit(self, coords: Tuple[int, ...], value: int) -> None:
        """
        Set the OffBit value at the given coordinates.
        
        Args:
            coords: 6D coordinates tuple
            value: 32-bit integer value to set
        """
        if len(coords) != 6:
            raise ValueError(f"Expected 6D coordinates, got {len(coords)}D")
        
        # Ensure coordinates are within bounds
        for i, (coord, dim) in enumerate(zip(coords, self.dimensions)):
            if coord < 0 or coord >= dim:
                raise IndexError(f"Coordinate {i} ({coord}) out of bounds for dimension {dim}")
        
        # Update the grid
        old_value = self.grid[coords]
        self.grid[coords] = np.uint32(value)
        
        # Update statistics
        if old_value == 0 and value != 0:
            self.stats.active_offbits += 1
        elif old_value != 0 and value == 0:
            self.stats.active_offbits -= 1
        
        self.modification_count += 1
        self.stats.last_update = time.time()
    
    def get_offbits_in_region(self, start_coords: Tuple[int, ...], 
                             end_coords: Tuple[int, ...]) -> List[int]:
        """
        Get all OffBits in a specified 6D region.
        
        Args:
            start_coords: Starting coordinates (inclusive)
            end_coords: Ending coordinates (exclusive)
            
        Returns:
            List of OffBit values in the region
        """
        if len(start_coords) != 6 or len(end_coords) != 6:
            raise ValueError("Both coordinate tuples must be 6D")
        
        # Create slice objects for each dimension
        slices = tuple(slice(start, end) for start, end in zip(start_coords, end_coords))
        
        # Extract the region and flatten to list
        region = self.grid[slices]
        return region.flatten().tolist()
    
    def set_offbits_in_region(self, start_coords: Tuple[int, ...], 
                             end_coords: Tuple[int, ...], 
                             values: List[int]) -> None:
        """
        Set all OffBits in a specified 6D region.
        
        Args:
            start_coords: Starting coordinates (inclusive)
            end_coords: Ending coordinates (exclusive)
            values: List of OffBit values to set
        """
        if len(start_coords) != 6 or len(end_coords) != 6:
            raise ValueError("Both coordinate tuples must be 6D")
        
        # Create slice objects for each dimension
        slices = tuple(slice(start, end) for start, end in zip(start_coords, end_coords))
        
        # Calculate expected size
        region_shape = tuple(end - start for start, end in zip(start_coords, end_coords))
        expected_size = np.prod(region_shape)
        
        if len(values) != expected_size:
            raise ValueError(f"Expected {expected_size} values, got {len(values)}")
        
        # Reshape values to match region shape and set
        values_array = np.array(values, dtype=np.uint32).reshape(region_shape)
        self.grid[slices] = values_array
        
        self.modification_count += 1
        self.stats.last_update = time.time()
    
    # ========================================================================
    # COMPATIBILITY METHODS
    # ========================================================================
    
    def get_all_offbits(self) -> List[int]:
        """
        Get all OffBits as a flat list (v3.0 compatibility).
        
        Returns:
            List of all OffBit values
        """
        return self.grid.flatten().tolist()
    
    def set_all_offbits(self, values: List[int]) -> None:
        """
        Set all OffBits from a flat list (v3.0 compatibility).
        
        Args:
            values: List of OffBit values (must match total_offbits)
        """
        if len(values) != self.total_offbits:
            raise ValueError(f"Expected {self.total_offbits} values, got {len(values)}")
        
        values_array = np.array(values, dtype=np.uint32).reshape(self.dimensions)
        self.grid = values_array
        
        # Update statistics
        self.stats.active_offbits = np.count_nonzero(values_array)
        self.modification_count += 1
        self.stats.last_update = time.time()
    
    def get_random_offbits(self, count: int) -> List[int]:
        """
        Get a random sample of OffBits (useful for testing).
        
        Args:
            count: Number of random OffBits to retrieve
            
        Returns:
            List of random OffBit values
        """
        flat_indices = np.random.choice(self.total_offbits, size=min(count, self.total_offbits), replace=False)
        flat_grid = self.grid.flatten()
        return [int(flat_grid[idx]) for idx in flat_indices]
    
    def get_active_offbits(self) -> List[Tuple[Tuple[int, ...], int]]:
        """
        Get all active (non-zero) OffBits with their coordinates.
        
        Returns:
            List of tuples: (coordinates, offbit_value)
        """
        active_offbits = []
        
        # Find all non-zero positions
        nonzero_indices = np.nonzero(self.grid)
        
        for i in range(len(nonzero_indices[0])):
            coords = tuple(nonzero_indices[j][i] for j in range(6))
            value = int(self.grid[coords])
            active_offbits.append((coords, value))
        
        return active_offbits
    
    # ========================================================================
    # ANALYSIS AND STATISTICS
    # ========================================================================
    
    def calculate_coherence(self) -> float:
        """
        Calculate overall Bitfield coherence.
        
        Returns:
            Float between 0.0 and 1.0 representing coherence level
        """
        active_offbits = self.get_active_offbits()
        
        if not active_offbits:
            return 0.0
        
        # Calculate coherence for each active OffBit
        coherence_values = []
        for coords, offbit_value in active_offbits:
            coherence = OffBit.calculate_coherence(offbit_value)
            coherence_values.append(coherence)
        
        # Return average coherence
        overall_coherence = np.mean(coherence_values)
        self.stats.coherence_level = overall_coherence
        
        return overall_coherence
    
    def get_memory_usage_mb(self) -> float:
        """
        Calculate memory usage in megabytes.
        
        Returns:
            Memory usage in MB
        """
        # Each OffBit is stored as uint32 (4 bytes)
        bytes_used = self.total_offbits * 4
        mb_used = bytes_used / (1024 * 1024)
        return mb_used
    
    def get_statistics(self) -> BitfieldStats:
        """
        Get current Bitfield statistics.
        
        Returns:
            BitfieldStats object with current metrics
        """
        # Update statistics
        self.stats.active_offbits = np.count_nonzero(self.grid)
        self.stats.memory_usage_mb = self.get_memory_usage_mb()
        self.stats.coherence_level = self.calculate_coherence()
        self.stats.last_update = time.time()
        
        return self.stats
    
    def get_layer_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for each OffBit layer across the entire Bitfield.
        
        Returns:
            Dictionary with statistics for each layer
        """
        active_offbits = [value for coords, value in self.get_active_offbits()]
        
        if not active_offbits:
            return {
                'reality': {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0},
                'information': {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0},
                'activation': {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0},
                'unactivated': {'mean': 0.0, 'std': 0.0, 'min': 0, 'max': 0}
            }
        
        # Extract layer values
        layers = {
            'reality': [OffBit.get_reality_layer(offbit) for offbit in active_offbits],
            'information': [OffBit.get_information_layer(offbit) for offbit in active_offbits],
            'activation': [OffBit.get_activation_layer(offbit) for offbit in active_offbits],
            'unactivated': [OffBit.get_unactivated_layer(offbit) for offbit in active_offbits]
        }
        
        # Calculate statistics for each layer
        statistics = {}
        for layer_name, values in layers.items():
            statistics[layer_name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': int(np.min(values)),
                'max': int(np.max(values))
            }
        
        return statistics
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def clear(self) -> None:
        """Clear all OffBits (set to zero)."""
        self.grid.fill(0)
        self.stats.active_offbits = 0
        self.modification_count += 1
        self.stats.last_update = time.time()
    
    def copy(self) -> 'Bitfield':
        """
        Create a deep copy of the Bitfield.
        
        Returns:
            New Bitfield instance with copied data
        """
        new_bitfield = Bitfield(dimensions=self.dimensions, sparsity=0.0)
        new_bitfield.grid = self.grid.copy()
        new_bitfield.stats = BitfieldStats(
            total_offbits=self.stats.total_offbits,
            active_offbits=self.stats.active_offbits,
            memory_usage_mb=self.stats.memory_usage_mb,
            coherence_level=self.stats.coherence_level,
            last_update=time.time()
        )
        return new_bitfield
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export Bitfield to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the Bitfield
        """
        return {
            'dimensions': self.dimensions,
            'sparsity': self.sparsity,
            'total_offbits': self.total_offbits,
            'grid_data': self.grid.tolist(),
            'statistics': {
                'active_offbits': self.stats.active_offbits,
                'memory_usage_mb': self.stats.memory_usage_mb,
                'coherence_level': self.stats.coherence_level,
                'creation_timestamp': self.creation_timestamp,
                'modification_count': self.modification_count
            }
        }
    
    def import_from_dict(self, data: Dict[str, Any]) -> None:
        """
        Import Bitfield from a dictionary.
        
        Args:
            data: Dictionary representation of a Bitfield
        """
        self.dimensions = tuple(data['dimensions'])
        self.sparsity = data['sparsity']
        self.total_offbits = data['total_offbits']
        self.size = self.total_offbits
        
        # Restore grid data
        grid_data = np.array(data['grid_data'], dtype=np.uint32)
        self.grid = grid_data.reshape(self.dimensions)
        
        # Restore statistics
        stats_data = data['statistics']
        self.stats = BitfieldStats(
            total_offbits=self.total_offbits,
            active_offbits=stats_data['active_offbits'],
            memory_usage_mb=stats_data['memory_usage_mb'],
            coherence_level=stats_data['coherence_level'],
            last_update=time.time()
        )
        
        self.creation_timestamp = stats_data['creation_timestamp']
        self.modification_count = stats_data['modification_count']
    
    def __str__(self) -> str:
        """String representation of the Bitfield."""
        stats = self.get_statistics()
        return (f"UBP Bitfield v3.1\n"
                f"Dimensions: {self.dimensions}\n"
                f"Total OffBits: {stats.total_offbits:,}\n"
                f"Active OffBits: {stats.active_offbits:,}\n"
                f"Memory Usage: {stats.memory_usage_mb:.2f} MB\n"
                f"Coherence: {stats.coherence_level:.3f}\n"
                f"Modifications: {self.modification_count}")
    
    def __repr__(self) -> str:
        """Detailed representation of the Bitfield."""
        return f"Bitfield(dimensions={self.dimensions}, sparsity={self.sparsity}, size={self.size})"


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def create_bitfield(size: Optional[int] = None,
                   dimensions: Optional[Tuple[int, int, int, int, int, int]] = None,
                   sparsity: float = 0.01) -> Bitfield:
    """
    Create a new Bitfield with flexible parameters.
    
    Args:
        size: Total number of OffBits (v3.0 style)
        dimensions: 6D dimensions tuple (v2.0 style)
        sparsity: Fraction of OffBits to activate
        
    Returns:
        New Bitfield instance
    """
    return Bitfield(dimensions=dimensions, sparsity=sparsity, size=size)


def benchmark_bitfield(bitfield: Bitfield, num_operations: int = 1000) -> Dict[str, float]:
    """
    Benchmark Bitfield performance.
    
    Args:
        bitfield: Bitfield instance to benchmark
        num_operations: Number of operations to perform
        
    Returns:
        Dictionary with benchmark results
    """
    import random
    
    start_time = time.time()
    
    # Test random access operations
    for i in range(num_operations):
        # Generate random coordinates
        coords = tuple(random.randint(0, dim-1) for dim in bitfield.dimensions)
        
        # Read operation
        value = bitfield.get_offbit(coords)
        
        # Write operation (every 10th iteration)
        if i % 10 == 0:
            new_value = random.randint(0, 0xFFFFFF)
            bitfield.set_offbit(coords, new_value)
    
    total_time = time.time() - start_time
    
    return {
        'total_time': total_time,
        'operations_per_second': num_operations / total_time,
        'memory_usage_mb': bitfield.get_memory_usage_mb(),
        'coherence_level': bitfield.calculate_coherence(),
        'active_offbits': bitfield.stats.active_offbits
    }


if __name__ == "__main__":
    # Test the enhanced Bitfield
    print("🧪 Testing UBP Bitfield v3.1...")
    
    # Test v3.0 style initialization
    bitfield_v3 = Bitfield(size=1000)
    print(f"v3.0 style: {bitfield_v3}")
    
    # Test v2.0 style initialization
    bitfield_v2 = Bitfield(dimensions=(10, 10, 10, 2, 2, 2))
    print(f"v2.0 style: {bitfield_v2}")
    
    # Test operations
    coords = (5, 5, 5, 1, 1, 1)
    test_offbit = OffBit.create_offbit(32, 16, 48, 8)
    
    bitfield_v2.set_offbit(coords, test_offbit)
    retrieved = bitfield_v2.get_offbit(coords)
    
    print(f"Set/Get test: {test_offbit} -> {retrieved} ({'PASS' if test_offbit == retrieved else 'FAIL'})")
    
    # Test statistics
    stats = bitfield_v2.get_statistics()
    print(f"Statistics: {stats.active_offbits} active, {stats.coherence_level:.3f} coherence")
    
    print("✅ UBP Bitfield v3.1 test completed successfully!")

