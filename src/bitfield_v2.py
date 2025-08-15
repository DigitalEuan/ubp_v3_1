"""
Universal Binary Principle (UBP) Framework v2.0 - Bitfield Module

This module implements the foundational 6D Bitfield data structure and
OffBit ontology for the UBP computational system. It provides the core
data layer that all UBP operations are built upon.

Author: Euan Craig
Version: 2.0
Date: August 2025
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
import json


class OffBit:
    """
    Helper class for managing operations on 24-bit OffBits within the UBP framework.
    
    Each OffBit contains four 6-bit layers:
    - Reality Layer (bits 0-5): Fundamental state information
    - Information Layer (bits 6-11): Data and computational content
    - Activation Layer (bits 12-17): Current activation state
    - Unactivated Layer (bits 18-23): Potential/dormant states
    """
    
    # Bit masks for each 6-bit layer
    REALITY_MASK = 0b111111  # Bits 0-5
    INFORMATION_MASK = 0b111111 << 6  # Bits 6-11
    ACTIVATION_MASK = 0b111111 << 12  # Bits 12-17
    UNACTIVATED_MASK = 0b111111 << 18  # Bits 18-23
    
    # Layer shift amounts
    REALITY_SHIFT = 0
    INFORMATION_SHIFT = 6
    ACTIVATION_SHIFT = 12
    UNACTIVATED_SHIFT = 18
    
    @staticmethod
    def get_reality_layer(offbit_value: int) -> int:
        """Extract the Reality layer (bits 0-5) from an OffBit."""
        return (offbit_value & OffBit.REALITY_MASK) >> OffBit.REALITY_SHIFT
    
    @staticmethod
    def get_information_layer(offbit_value: int) -> int:
        """Extract the Information layer (bits 6-11) from an OffBit."""
        return (offbit_value & OffBit.INFORMATION_MASK) >> OffBit.INFORMATION_SHIFT
    
    @staticmethod
    def get_activation_layer(offbit_value: int) -> int:
        """Extract the Activation layer (bits 12-17) from an OffBit."""
        return (offbit_value & OffBit.ACTIVATION_MASK) >> OffBit.ACTIVATION_SHIFT
    
    @staticmethod
    def get_unactivated_layer(offbit_value: int) -> int:
        """Extract the Unactivated layer (bits 18-23) from an OffBit."""
        return (offbit_value & OffBit.UNACTIVATED_MASK) >> OffBit.UNACTIVATED_SHIFT
    
    @staticmethod
    def set_reality_layer(offbit_value: int, new_value: int) -> int:
        """Set the Reality layer of an OffBit to a new 6-bit value."""
        cleared = offbit_value & ~OffBit.REALITY_MASK
        return cleared | ((new_value & 0b111111) << OffBit.REALITY_SHIFT)
    
    @staticmethod
    def set_information_layer(offbit_value: int, new_value: int) -> int:
        """Set the Information layer of an OffBit to a new 6-bit value."""
        cleared = offbit_value & ~OffBit.INFORMATION_MASK
        return cleared | ((new_value & 0b111111) << OffBit.INFORMATION_SHIFT)
    
    @staticmethod
    def set_activation_layer(offbit_value: int, new_value: int) -> int:
        """Set the Activation layer of an OffBit to a new 6-bit value."""
        cleared = offbit_value & ~OffBit.ACTIVATION_MASK
        return cleared | ((new_value & 0b111111) << OffBit.ACTIVATION_SHIFT)
    
    @staticmethod
    def set_unactivated_layer(offbit_value: int, new_value: int) -> int:
        """Set the Unactivated layer of an OffBit to a new 6-bit value."""
        cleared = offbit_value & ~OffBit.UNACTIVATED_MASK
        return cleared | ((new_value & 0b111111) << OffBit.UNACTIVATED_SHIFT)
    
    @staticmethod
    def create_offbit(reality: int = 0, information: int = 0, 
                     activation: int = 0, unactivated: int = 0) -> int:
        """
        Create a new OffBit from individual layer values.
        
        Args:
            reality: 6-bit value for Reality layer
            information: 6-bit value for Information layer
            activation: 6-bit value for Activation layer
            unactivated: 6-bit value for Unactivated layer
            
        Returns:
            32-bit integer representing the complete OffBit
        """
        offbit = 0
        offbit |= (reality & 0b111111) << OffBit.REALITY_SHIFT
        offbit |= (information & 0b111111) << OffBit.INFORMATION_SHIFT
        offbit |= (activation & 0b111111) << OffBit.ACTIVATION_SHIFT
        offbit |= (unactivated & 0b111111) << OffBit.UNACTIVATED_SHIFT
        return offbit
    
    @staticmethod
    def get_all_layers(offbit_value: int) -> Dict[str, int]:
        """
        Extract all four layers from an OffBit.
        
        Returns:
            Dictionary with keys: 'reality', 'information', 'activation', 'unactivated'
        """
        return {
            'reality': OffBit.get_reality_layer(offbit_value),
            'information': OffBit.get_information_layer(offbit_value),
            'activation': OffBit.get_activation_layer(offbit_value),
            'unactivated': OffBit.get_unactivated_layer(offbit_value)
        }
    
    @staticmethod
    def set_layer(offbit_value: int, layer_name: str, new_value: int) -> int:
        """
        Set a specific layer of an OffBit to a new value.
        
        Args:
            offbit_value: Current OffBit value
            layer_name: Name of layer ('reality', 'information', 'activation', 'unactivated')
            new_value: New 6-bit value for the layer
            
        Returns:
            Updated OffBit value
        """
        layer_name = layer_name.lower()
        if layer_name == 'reality':
            return OffBit.set_reality_layer(offbit_value, new_value)
        elif layer_name == 'information':
            return OffBit.set_information_layer(offbit_value, new_value)
        elif layer_name == 'activation':
            return OffBit.set_activation_layer(offbit_value, new_value)
        elif layer_name == 'unactivated':
            return OffBit.set_unactivated_layer(offbit_value, new_value)
        else:
            raise ValueError(f"Unknown layer name: {layer_name}. Must be one of: reality, information, activation, unactivated")
    
    @staticmethod
    def get_layer(offbit_value: int, layer_name: str) -> int:
        """
        Get a specific layer value from an OffBit.
        
        Args:
            offbit_value: Current OffBit value
            layer_name: Name of layer ('reality', 'information', 'activation', 'unactivated')
            
        Returns:
            6-bit value of the specified layer
        """
        layer_name = layer_name.lower()
        if layer_name == 'reality':
            return OffBit.get_reality_layer(offbit_value)
        elif layer_name == 'information':
            return OffBit.get_information_layer(offbit_value)
        elif layer_name == 'activation':
            return OffBit.get_activation_layer(offbit_value)
        elif layer_name == 'unactivated':
            return OffBit.get_unactivated_layer(offbit_value)
        else:
            raise ValueError(f"Unknown layer name: {layer_name}. Must be one of: reality, information, activation, unactivated")
    
    @staticmethod
    def is_active(offbit_value: int) -> bool:
        """Check if an OffBit is considered 'active' (has non-zero activation layer)."""
        return OffBit.get_activation_layer(offbit_value) > 0
    
    @staticmethod
    def calculate_coherence(offbit_value: int) -> float:
        """
        Calculate a coherence metric for an individual OffBit.
        
        Coherence is based on the balance and organization of the four layers.
        Higher coherence indicates more organized, purposeful bit patterns.
        
        Returns:
            Float between 0.0 and 1.0 representing coherence level
        """
        layers = OffBit.get_all_layers(offbit_value)
        
        # Calculate entropy across layers (lower entropy = higher coherence)
        layer_values = list(layers.values())
        total = sum(layer_values)
        
        if total == 0:
            return 0.0  # No information = no coherence
        
        # Normalized layer distribution
        probabilities = [v / total for v in layer_values if v > 0]
        
        # Calculate Shannon entropy
        entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
        max_entropy = np.log2(len(probabilities)) if len(probabilities) > 1 else 1.0
        
        # Coherence is inverse of normalized entropy
        coherence = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
        
        return coherence


@dataclass
class BitfieldStats:
    """Statistics and metrics for a Bitfield instance."""
    total_offbits: int
    active_offbits: int
    average_coherence: float
    layer_distributions: Dict[str, Dict[str, int]]
    sparsity: float
    memory_usage_mb: float


class Bitfield:
    """
    The foundational 6D Bitfield data structure for the UBP framework.
    
    This class manages a 6-dimensional array of 32-bit integers, each containing
    a 24-bit OffBit with four 6-bit layers. The Bitfield serves as the primary
    computational space for all UBP operations.
    """
    
    def __init__(self, dimensions: Tuple[int, int, int, int, int, int] = (32, 32, 32, 4, 2, 2), sparsity: float = 0.01):
        """
        Initialize the Bitfield with specified dimensions and sparsity.
        
        Args:
            dimensions: 6D tuple defining the Bitfield structure
                Default: (32, 32, 32, 4, 2, 2) for testing (16,384 OffBits)
                Production: (170, 170, 170, 5, 2, 2) for full system (~2.3M OffBits)
            sparsity: Fraction of OffBits to initialize as active (default: 0.01)
        """
        self.dimensions = dimensions
        self.sparsity = sparsity
        self.grid = np.zeros(dimensions, dtype=np.uint32)
        self.offbit_helper = OffBit()
        
        # Calculate total OffBits
        self.total_offbits = np.prod(dimensions)
        
        # Initialize metadata
        self.creation_timestamp = np.datetime64('now')
        self.modification_count = 0
        
        print(f"✅ UBP Bitfield Initialized")
        print(f"   Dimensions: {dimensions}")
        print(f"   Total OffBits: {self.total_offbits:,}")
        print(f"   Memory Usage: {self.get_memory_usage_mb():.2f} MB")
    
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
        
        self.grid[coords] = np.uint32(value)
        self.modification_count += 1
    
    def get_offbit_layers(self, coords: Tuple[int, ...]) -> Dict[str, int]:
        """
        Get all four layers of an OffBit at the given coordinates.
        
        Args:
            coords: 6D coordinates tuple
            
        Returns:
            Dictionary with layer names and their 6-bit values
        """
        offbit_value = self.get_offbit(coords)
        return OffBit.get_all_layers(offbit_value)
    
    def set_offbit_layer(self, coords: Tuple[int, ...], layer_name: str, value: int) -> None:
        """
        Set a specific layer of an OffBit at the given coordinates.
        
        Args:
            coords: 6D coordinates tuple
            layer_name: One of 'reality', 'information', 'activation', 'unactivated'
            value: 6-bit value to set for the layer
        """
        current_offbit = self.get_offbit(coords)
        
        if layer_name == 'reality':
            new_offbit = OffBit.set_reality_layer(current_offbit, value)
        elif layer_name == 'information':
            new_offbit = OffBit.set_information_layer(current_offbit, value)
        elif layer_name == 'activation':
            new_offbit = OffBit.set_activation_layer(current_offbit, value)
        elif layer_name == 'unactivated':
            new_offbit = OffBit.set_unactivated_layer(current_offbit, value)
        else:
            raise ValueError(f"Unknown layer name: {layer_name}")
        
        self.set_offbit(coords, new_offbit)
    
    def get_active_offbits_count(self) -> int:
        """
        Count the number of OffBits with non-zero activation layers.
        
        Returns:
            Number of active OffBits in the Bitfield
        """
        # Extract activation layers from all OffBits
        activation_values = (self.grid & OffBit.ACTIVATION_MASK) >> OffBit.ACTIVATION_SHIFT
        return int(np.count_nonzero(activation_values))
    
    def get_sparsity(self) -> float:
        """
        Calculate the sparsity of the Bitfield (fraction of zero OffBits).
        
        Returns:
            Float between 0.0 and 1.0 representing sparsity
        """
        zero_count = np.count_nonzero(self.grid == 0)
        return float(zero_count) / self.total_offbits
    
    def get_memory_usage_mb(self) -> float:
        """
        Calculate the memory usage of the Bitfield in megabytes.
        
        Returns:
            Memory usage in MB
        """
        bytes_used = self.grid.nbytes
        return bytes_used / (1024 * 1024)
    
    def calculate_global_coherence(self) -> float:
        """
        Calculate the global coherence across the entire Bitfield.
        
        This is a key UBP metric representing the overall organization
        and purposefulness of the computational space.
        
        Returns:
            Float between 0.0 and 1.0 representing global coherence
        """
        if self.total_offbits == 0:
            return 0.0
        
        # Calculate coherence for each non-zero OffBit
        coherence_sum = 0.0
        active_count = 0
        
        # Flatten the grid for easier iteration
        flat_grid = self.grid.flatten()
        
        for offbit_value in flat_grid:
            if offbit_value != 0:
                coherence_sum += OffBit.calculate_coherence(offbit_value)
                active_count += 1
        
        if active_count == 0:
            return 0.0
        
        return coherence_sum / active_count
    
    def initialize_random_state(self, density: float = 0.01, 
                               realm_crv: float = 0.2265234857) -> None:
        """
        Initialize the Bitfield to a random state with specified density.
        
        This is useful for creating initial chaotic states for simulations.
        
        Args:
            density: Fraction of OffBits to activate (0.0 to 1.0)
            realm_crv: Core Resonance Value to use for toggle probability
        """
        print(f"🎲 Initializing Bitfield to random state (density={density:.3f})")
        
        # Calculate number of OffBits to activate
        num_to_activate = int(self.total_offbits * density)
        
        # Generate random coordinates for activation
        flat_indices = np.random.choice(self.total_offbits, num_to_activate, replace=False)
        
        for flat_idx in flat_indices:
            # Convert flat index to 6D coordinates
            coords = np.unravel_index(flat_idx, self.dimensions)
            
            # Generate random OffBit based on CRV
            reality = int(np.random.exponential(realm_crv * 63)) % 64
            information = int(np.random.exponential(realm_crv * 63)) % 64
            activation = int(np.random.exponential(realm_crv * 63)) % 64
            unactivated = int(np.random.exponential(realm_crv * 63)) % 64
            
            offbit = OffBit.create_offbit(reality, information, activation, unactivated)
            self.set_offbit(coords, offbit)
        
        print(f"   Activated {num_to_activate:,} OffBits")
        print(f"   Global coherence: {self.calculate_global_coherence():.6f}")
    
    def get_statistics(self) -> BitfieldStats:
        """
        Generate comprehensive statistics about the current Bitfield state.
        
        Returns:
            BitfieldStats object with detailed metrics
        """
        active_count = self.get_active_offbits_count()
        global_coherence = self.calculate_global_coherence()
        sparsity = self.get_sparsity()
        memory_usage = self.get_memory_usage_mb()
        
        # Calculate layer distributions
        layer_distributions = {
            'reality': {},
            'information': {},
            'activation': {},
            'unactivated': {}
        }
        
        # Sample a subset for layer analysis (to avoid performance issues)
        sample_size = min(10000, self.total_offbits)
        flat_grid = self.grid.flatten()
        sample_indices = np.random.choice(len(flat_grid), sample_size, replace=False)
        
        for idx in sample_indices:
            offbit_value = flat_grid[idx]
            if offbit_value != 0:
                layers = OffBit.get_all_layers(offbit_value)
                for layer_name, layer_value in layers.items():
                    if layer_value not in layer_distributions[layer_name]:
                        layer_distributions[layer_name][layer_value] = 0
                    layer_distributions[layer_name][layer_value] += 1
        
        return BitfieldStats(
            total_offbits=self.total_offbits,
            active_offbits=active_count,
            average_coherence=global_coherence,
            layer_distributions=layer_distributions,
            sparsity=sparsity,
            memory_usage_mb=memory_usage
        )
    
    def export_to_dict(self) -> Dict[str, Any]:
        """
        Export the Bitfield to a dictionary for serialization.
        
        Returns:
            Dictionary representation of the Bitfield
        """
        return {
            'dimensions': self.dimensions,
            'grid_data': self.grid.tolist(),
            'total_offbits': self.total_offbits,
            'creation_timestamp': str(self.creation_timestamp),
            'modification_count': self.modification_count,
            'statistics': {
                'active_offbits': self.get_active_offbits_count(),
                'global_coherence': self.calculate_global_coherence(),
                'sparsity': self.get_sparsity(),
                'memory_usage_mb': self.get_memory_usage_mb()
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Bitfield':
        """
        Create a Bitfield instance from a dictionary.
        
        Args:
            data: Dictionary representation of a Bitfield
            
        Returns:
            New Bitfield instance
        """
        bitfield = cls(tuple(data['dimensions']))
        bitfield.grid = np.array(data['grid_data'], dtype=np.uint32)
        bitfield.modification_count = data.get('modification_count', 0)
        return bitfield


if __name__ == "__main__":
    # Test the Bitfield implementation
    print("="*60)
    print("UBP BITFIELD MODULE TEST")
    print("="*60)
    
    # Create a test Bitfield
    bf = Bitfield((8, 8, 8, 2, 2, 2))
    
    # Test OffBit operations
    test_coords = (1, 2, 3, 0, 1, 0)
    test_offbit = OffBit.create_offbit(reality=15, information=31, activation=7, unactivated=3)
    
    bf.set_offbit(test_coords, test_offbit)
    retrieved = bf.get_offbit(test_coords)
    layers = bf.get_offbit_layers(test_coords)
    
    print(f"Test OffBit: {test_offbit:032b}")
    print(f"Retrieved:   {retrieved:032b}")
    print(f"Layers: {layers}")
    print(f"Is Active: {OffBit.is_active(retrieved)}")
    print(f"Coherence: {OffBit.calculate_coherence(retrieved):.6f}")
    
    # Test random initialization
    bf.initialize_random_state(density=0.05)
    
    # Get statistics
    stats = bf.get_statistics()
    print(f"\nBitfield Statistics:")
    print(f"  Total OffBits: {stats.total_offbits:,}")
    print(f"  Active OffBits: {stats.active_offbits:,}")
    print(f"  Global Coherence: {stats.average_coherence:.6f}")
    print(f"  Sparsity: {stats.sparsity:.3f}")
    print(f"  Memory Usage: {stats.memory_usage_mb:.2f} MB")
    
    print("\n✅ Bitfield module test completed successfully!")

