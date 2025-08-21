"""
Universal Binary Principle (UBP) Framework v3.2+ - OffBit and Bitfield Module
Author: Euan Craig, New Zealand
Date: 20 August 2025

This module defines the fundamental computational units of the UBP:
the OffBit (a 32-bit binary vector) and the Bitfield (a 6D lattice).
It incorporates the ontological layers for OffBit and ensures Bitfield
dimensions are dynamically loaded from UBPConfig.
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Union, Optional
from dataclasses import dataclass, field

# OffBit Ontological Layers (conceptual mapping for 6-bit segments)
OFFBIT_LAYER_MASKS = {
    "reality": (0b111111 << 0),   # Bits 0-5
    "information": (0b111111 << 6),  # Bits 6-11
    "coherence": (0b111111 << 12), # Bits 12-17
    "observer": (0b111111 << 18)  # Bits 18-23
    # Remaining 8 bits (24-31) are reserved for padding or future extensions
}

@dataclass
class OffBit:
    """
    The fundamental unit of UBP computation, a 32-bit binary vector.
    Value is stored as np.uint32. Metadata allows for richer contextual information.

    Ontological Layers (6-bit segments within the 32-bit value):
    - reality: bits 0-5
    - information: bits 6-11
    - coherence: bits 12-17
    - observer: bits 18-23
    """
    value: np.uint32 = field(default_factory=lambda: np.uint32(0))
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """
        Post-initialization to ensure value is np.uint32 and meta is properly populated.
        """
        if not isinstance(self.value, np.uint32):
            self.value = np.uint32(self.value)
        
        # Populate meta with default values if not provided or incomplete.
        # This ensures consistency for all OffBit instances.
        default_meta_values = self._default_meta()
        for key, default_value in default_meta_values.items():
            if key not in self.meta:
                self.meta[key] = default_value
        
        # Update layer-specific meta fields based on the actual value at initialization
        self.meta["reality_state"] = self.get_layer_value("reality")
        self.meta["information_payload"] = self.get_layer_value("information")
        self.meta["coherence_phase"] = self.get_layer_value("coherence")
        self.meta["observer_context"] = self.get_layer_value("observer")


    def _default_meta(self) -> Dict[str, Any]:
        """
        Generates default metadata based on OffBit's conceptual layers.
        These are dynamically updated by __post_init__ or layer setters.
        """
        return {
            "reality_state": self.get_layer_value("reality"),
            "information_payload": self.get_layer_value("information"),
            "coherence_phase": self.get_layer_value("coherence"),
            "observer_context": self.get_layer_value("observer"),
            "timestamp": None, # Will be set during processing
            "realm": "unassigned",
            "source_coords": None # Useful for tracking origin in a Bitfield
        }

    def get_layer_value(self, layer_name: str) -> int:
        """
        Extracts the 6-bit integer value for a specific ontological layer.

        Args:
            layer_name (str): The name of the ontological layer (e.g., "reality", "information").

        Returns:
            int: The 6-bit value (0-63) of the specified layer.

        Raises:
            ValueError: If an unknown layer name is provided.
        """
        if layer_name not in OFFBIT_LAYER_MASKS:
            raise ValueError(f"Unknown OffBit layer: {layer_name}. Must be one of {list(OFFBIT_LAYER_MASKS.keys())}")
        
        mask = OFFBIT_LAYER_MASKS[layer_name]
        # Calculate right shift dynamically: number of trailing zeros in the mask
        shift = (mask & -mask).bit_length() - 1 
        return int((self.value & mask) >> shift)

    def set_layer_value(self, layer_name: str, layer_value: int) -> None:
        """
        Sets the 6-bit integer value for a specific ontological layer.

        Args:
            layer_name (str): The name of the ontological layer.
            layer_value (int): The 6-bit value (0-63) to set for the layer.

        Raises:
            ValueError: If an unknown layer name or an out-of-range layer_value is provided.
        """
        if layer_name not in OFFBIT_LAYER_MASKS:
            raise ValueError(f"Unknown OffBit layer: {layer_name}. Must be one of {list(OFFBIT_LAYER_MASKS.keys())}")
        
        if not (0 <= layer_value <= 63): # 6-bit value range
            raise ValueError(f"Layer value must be between 0 and 63 for layer {layer_name}")
        
        mask = OFFBIT_LAYER_MASKS[layer_name]
        # Calculate right shift dynamically
        shift = (mask & -mask).bit_length() - 1
        
        # Clear existing bits for the layer and then set new ones
        self.value = (self.value & ~mask) | (np.uint32(layer_value) << shift)
        
        # Update relevant meta field instantly
        meta_key_map = {
            "reality": "reality_state",
            "information": "information_payload",
            "coherence": "coherence_phase",
            "observer": "observer_context"
        }
        self.meta[meta_key_map.get(layer_name, layer_name)] = layer_value

    @classmethod
    def create_offbit(cls, 
                      reality_state: int = 0, 
                      information_payload: int = 0, 
                      coherence_phase: int = 0, 
                      observer_context: int = 0,
                      **kwargs) -> 'OffBit':
        """
        Factory method to create an OffBit by setting its ontological layers directly.
        Each layer expects a 6-bit integer (0-63).

        Args:
            reality_state (int): Value for the 'reality' layer (bits 0-5).
            information_payload (int): Value for the 'information' layer (bits 6-11).
            coherence_phase (int): Value for the 'coherence' layer (bits 12-17).
            observer_context (int): Value for the 'observer' layer (bits 18-23).
            **kwargs: Additional metadata to include in the OffBit's 'meta' dictionary.

        Returns:
            OffBit: A new OffBit instance with the specified layer values and metadata.

        Raises:
            ValueError: If any layer value is outside the 0-63 range.
        """
        if not all(0 <= v <= 63 for v in [reality_state, information_payload, coherence_phase, observer_context]):
            raise ValueError("All OffBit layer values must be between 0 and 63.")

        value = np.uint32(0)
        value |= (np.uint32(reality_state) << 0)
        value |= (np.uint32(information_payload) << 6)
        value |= (np.uint32(coherence_phase) << 12)
        value |= (np.uint32(observer_context) << 18)
        
        # Initialize OffBit, then update meta with provided kwargs
        offbit = cls(value=value)
        offbit.meta.update(kwargs) # Allow overriding default meta values or adding new ones
        
        return offbit

    def to_binary_string(self) -> str:
        """
        Returns the 32-bit binary representation of the OffBit's value.

        Returns:
            str: A 32-character binary string.
        """
        return bin(self.value)[2:].zfill(32)

    def __str__(self) -> str:
        """
        Returns a string representation of the OffBit, including its value,
        binary representation, and metadata.
        """
        # Truncate meta for cleaner display if it's very large
        meta_display = str(self.meta)
        if len(meta_display) > 100:
            meta_display = meta_display[:97] + "..."
        return f"OffBit(value={self.value}, binary='{self.to_binary_string()}', meta={meta_display})"
    
    def __eq__(self, other: Any) -> bool:
        """
        Compares two OffBit instances for equality based on their 'value' attribute.
        """
        if not isinstance(other, OffBit):
            return NotImplemented
        return self.value == other.value

    def __hash__(self) -> int:
        """
        Computes the hash of an OffBit instance based on its 'value'.
        This allows OffBit objects to be used in sets or as dictionary keys.
        """
        return hash(self.value)


class Bitfield:
    """
    The 6D computational lattice for OffBits.
    Dimensions are loaded from the UBP configuration via the framework.
    This is a resonant manifold, not just storage.
    """
    def __init__(self, dimensions: Optional[Tuple[int, ...]] = None):
        """
        Initialize the Bitfield.
        
        Args:
            dimensions (Optional[Tuple[int, ...]]): A tuple defining the 6D shape of the Bitfield.
                                                     If None, it expects dimensions to be set by the UBPConfig
                                                     when the UBPFramework initializes it.
        """
        if dimensions is None or len(dimensions) != 6 or not all(isinstance(d, int) and d > 0 for d in dimensions):
            # Fallback to a default small, valid 6D dimension if not provided or invalid
            self.dimensions = (1, 1, 1, 1, 1, 1) 
            print("⚠️ Bitfield initialized with placeholder/default dimensions. Expected to be updated by UBPConfig and must be a 6D tuple of positive integers.")
        else:
            self.dimensions = dimensions

        # Initialize the grid with zeros as np.uint32.
        # Use np.array for grid to allow direct indexing.
        self.grid = np.zeros(self.dimensions, dtype=np.uint32)
        self.total_size = np.prod(self.dimensions)
        print(f"Bitfield created with dimensions {self.dimensions} (Total Cells: {self.total_size}).")

    def set_offbit(self, coords: Tuple[int, ...], offbit: OffBit) -> None:
        """
        Places an OffBit at specified 6D coordinates in the Bitfield grid.
        
        Args:
            coords (Tuple[int, ...]): A tuple of 6 integers (x, y, z, a, b, c) representing the position.
            offbit (OffBit): The OffBit object to place.
        
        Raises:
            IndexError: If coordinates are out of bounds or not 6-dimensional.
            TypeError: If 'offbit' is not an instance of OffBit.
        """
        if not isinstance(offbit, OffBit):
            raise TypeError("The 'offbit' argument must be an instance of OffBit.")

        if len(coords) != len(self.dimensions) or not all(0 <= coords[i] < self.dimensions[i] for i in range(len(coords))):
            raise IndexError(f"Coordinates {coords} out of bounds for Bitfield with dimensions {self.dimensions}. Expected 6 dimensions.")
        
        self.grid[coords] = offbit.value
        
        # Update OffBit's meta with its source_coords if it's not already set
        # This records where the OffBit was placed in the Bitfield.
        if offbit.meta.get("source_coords") is None:
            offbit.meta["source_coords"] = coords

    def get_offbit(self, coords: Tuple[int, ...]) -> OffBit:
        """
        Retrieves an OffBit from specified 6D coordinates.
        
        Args:
            coords (Tuple[int, ...]): A tuple of 6 integers (x, y, z, a, b, c) representing the position.
        
        Returns:
            OffBit: The OffBit object retrieved from the coordinates.
        
        Raises:
            IndexError: If coordinates are out of bounds or not 6-dimensional.
        """
        if len(coords) != len(self.dimensions) or not all(0 <= coords[i] < self.dimensions[i] for i in range(len(coords))):
            raise IndexError(f"Coordinates {coords} out of bounds for Bitfield with dimensions {self.dimensions}. Expected 6 dimensions.")
        
        value = self.grid[coords]
        
        # Reconstruct OffBit from its raw value.
        # Ensure metadata is correctly populated for the retrieved OffBit.
        offbit = OffBit(value=value)
        # The __post_init__ method of OffBit already populates layer values.
        # Add source coordinates to meta upon retrieval for tracking.
        offbit.meta["source_coords"] = coords 
        
        return offbit

    def get_all_offbits(self) -> List[OffBit]:
        """
        Returns a flat list of all OffBit objects currently stored in the Bitfield.

        Returns:
            List[OffBit]: A list containing all OffBit instances in the grid.
        """
        all_offbits = []
        # Use np.ndenumerate to get both coordinates and value for each element
        for coords, value in np.ndenumerate(self.grid):
            # Only reconstruct OffBit if the cell is not zero (empty/default)
            if value != 0:
                offbit = OffBit(value=value)
                # The __post_init__ of OffBit handles initial meta population for layers.
                # Just ensure source_coords is set.
                offbit.meta["source_coords"] = coords 
                all_offbits.append(offbit)
        return all_offbits

    def __len__(self) -> int:
        """
        Returns the total number of cells (potential OffBit locations) in the Bitfield.
        """
        return self.total_size

    def __str__(self) -> str:
        """
        Returns a string representation of the Bitfield, showing its dimensions and total size.
        """
        return f"Bitfield(dimensions={self.dimensions}, total_cells={self.total_size})"

if __name__ == "__main__":
    print("--- Testing OffBit ---")
    # Create an OffBit using the factory method
    ob1 = OffBit.create_offbit(reality_state=10, information_payload=20, coherence_phase=30, observer_context=40, custom_field="test_custom_meta")
    print(f"Created OffBit: {ob1}")
    print(f"Binary: {ob1.to_binary_string()}")
    print(f"Reality Layer: {ob1.get_layer_value('reality')}")
    print(f"Information Layer: {ob1.get_layer_value('information')}")
    print(f"Coherence Layer: {ob1.get_layer_value('coherence')}")
    print(f"Observer Layer: {ob1.get_layer_value('observer')}")
    print(f"Custom Field in Meta: {ob1.meta.get('custom_field')}")

    # Test setting a layer value
    ob1.set_layer_value("information", 50)
    print(f"\nOffBit after setting information layer to 50: {ob1}")
    print(f"New Information Layer: {ob1.get_layer_value('information')}")
    print(f"Binary: {ob1.to_binary_string()}")

    # Test invalid layer value
    try:
        ob1.set_layer_value("coherence", 64)
    except ValueError as e:
        print(f"Caught expected error: {e}")
    
    # Test unknown layer
    try:
        ob1.get_layer_value("unknown_layer")
    except ValueError as e:
        print(f"Caught expected error: {e}")

    # Test creating a default OffBit
    ob2 = OffBit()
    print(f"\nDefault OffBit: {ob2}")
    print(f"Default Meta: {ob2.meta}")

    # Test OffBit with partial meta (should still populate defaults)
    ob3 = OffBit(meta={"custom_init": True})
    print(f"\nOffBit with partial initial meta: {ob3}")
    assert ob3.meta.get('reality_state') is not None
    assert ob3.meta.get('custom_init') is True
    
    # Test equality and hashing
    ob1_copy = OffBit(ob1.value)
    print(f"\nEquality Test: ob1 == ob1_copy? {ob1 == ob1_copy}")
    assert ob1 == ob1_copy
    s = {ob1, ob1_copy, ob2}
    print(f"Set of OffBits (should only contain unique values): {len(s)} unique objects.")
    assert len(s) == 2 # ob1 and ob1_copy have same value, so only 2 unique if ob1 != ob2.

    print("\n--- Testing Bitfield ---")
    # Using the dimensions from the UBPConfig document for testing
    # Note: Bitfield requires exactly 6 dimensions.
    test_dimensions = (1, 2, 1, 1, 1, 2) # Smaller dimensions for easier testing (Total 4 cells)
    bf = Bitfield(dimensions=test_dimensions)
    print(bf)

    # Place an OffBit
    test_coords_1 = (0, 0, 0, 0, 0, 0)
    test_offbit_1 = OffBit.create_offbit(reality_state=5, information_payload=10)
    bf.set_offbit(test_coords_1, test_offbit_1)
    print(f"Placed OffBit at {test_coords_1}")

    test_coords_2 = (0, 1, 0, 0, 0, 1)
    test_offbit_2 = OffBit.create_offbit(reality_state=7, information_payload=12, coherence_phase=25)
    bf.set_offbit(test_coords_2, test_offbit_2)
    print(f"Placed OffBit at {test_coords_2}")

    # Test placing non-OffBit type
    try:
        bf.set_offbit((0,0,0,0,0,0), 123)
    except TypeError as e:
        print(f"Caught expected error: {e}")

    # Retrieve and verify
    retrieved_offbit_1 = bf.get_offbit(test_coords_1)
    print(f"Retrieved OffBit from {test_coords_1}: {retrieved_offbit_1}")
    assert retrieved_offbit_1.value == test_offbit_1.value
    assert retrieved_offbit_1.get_layer_value('reality') == 5
    assert retrieved_offbit_1.get_layer_value('information') == 10
    assert retrieved_offbit_1.meta['source_coords'] == test_coords_1

    retrieved_offbit_2 = bf.get_offbit(test_coords_2)
    print(f"Retrieved OffBit from {test_coords_2}: {retrieved_offbit_2}")
    assert retrieved_offbit_2.value == test_offbit_2.value
    assert retrieved_offbit_2.get_layer_value('reality') == 7
    assert retrieved_offbit_2.get_layer_value('coherence') == 25
    assert retrieved_offbit_2.meta['source_coords'] == test_coords_2

    # Test out of bounds
    try:
        bf.set_offbit((0, 0, 0, 0, 0, 2), test_offbit_1) # Out of bounds for last dimension
    except IndexError as e:
        print(f"Caught expected error: {e}")
    
    try:
        bf.get_offbit((0, 0, 0, 0, 0, 2)) # Out of bounds for last dimension
    except IndexError as e:
        print(f"Caught expected error: {e}")

    # Test incorrect number of dimensions
    try:
        bf.set_offbit((0, 0, 0), test_offbit_1)
    except IndexError as e:
        print(f"Caught expected error: {e}")

    print(f"Total size of Bitfield: {len(bf)}")

    # Test get_all_offbits
    all_offbits_list = bf.get_all_offbits()
    print(f"\nAll OffBits in Bitfield ({len(all_offbits_list)}):")
    for ob in all_offbits_list:
        print(ob)
    
    # Check that only the two placed offbits are returned (not the empty cells)
    assert len(all_offbits_list) == 2 
    assert all_offbits_list[0].get_layer_value('information') == 10
    assert all_offbits_list[0].meta['source_coords'] == test_coords_1
    assert all_offbits_list[1].get_layer_value('information') == 12
    assert all_offbits_list[1].meta['source_coords'] == test_coords_2
    
    print("\n✅ bits.py module test completed successfully!")
