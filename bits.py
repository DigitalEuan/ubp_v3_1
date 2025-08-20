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
    """
    value: np.uint32 = field(default_factory=lambda: np.uint32(0))
    meta: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Ensure value is np.uint32 upon initialization
        if not isinstance(self.value, np.uint32):
            self.value = np.uint32(self.value)
        
        # Initialize default meta if not provided or incomplete
        if not self.meta:
            self.meta = self._default_meta()
        else:
            # Ensure all default meta keys are present if meta was partially provided
            default_meta_values = self._default_meta()
            for key, default_value in default_meta_values.items():
                if key not in self.meta:
                    self.meta[key] = default_value
            # Update layer-specific meta fields based on the actual value
            self.meta["reality_state"] = self.get_layer_value("reality")
            self.meta["information_payload"] = self.get_layer_value("information")
            self.meta["coherence_phase"] = self.get_layer_value("coherence")
            self.meta["observer_context"] = self.get_layer_value("observer")


    def _default_meta(self) -> Dict[str, Any]:
        """Generates default metadata based on OffBit's conceptual layers."""
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
        """Extracts the 6-bit value for a specific ontological layer."""
        if layer_name not in OFFBIT_LAYER_MASKS:
            raise ValueError(f"Unknown OffBit layer: {layer_name}")
        
        mask = OFFBIT_LAYER_MASKS[layer_name]
        # Calculate right shift dynamically: number of trailing zeros in the mask
        shift = (mask & -mask).bit_length() - 1 
        return int((self.value & mask) >> shift)

    def set_layer_value(self, layer_name: str, layer_value: int) -> None:
        """Sets the 6-bit value for a specific ontological layer."""
        if layer_name not in OFFBIT_LAYER_MASKS:
            raise ValueError(f"Unknown OffBit layer: {layer_name}")
        
        if not (0 <= layer_value <= 63): # 6-bit value range
            raise ValueError(f"Layer value must be between 0 and 63 for layer {layer_name}")
        
        mask = OFFBIT_LAYER_MASKS[layer_name]
        # Calculate right shift dynamically
        shift = (mask & -mask).bit_length() - 1
        
        # Clear existing bits for the layer and then set new ones
        self.value = (self.value & ~mask) | (np.uint32(layer_value) << shift)
        # Update meta instantly if relevant
        # Note: The meta keys for layers use '_state', '_payload', '_phase', '_context'
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
        Factory method to create an OffBit by setting its ontological layers.
        Each layer expects a 6-bit integer (0-63).
        """
        if not all(0 <= v <= 63 for v in [reality_state, information_payload, coherence_phase, observer_context]):
            raise ValueError("All OffBit layer values must be between 0 and 63.")

        value = np.uint32(0)
        value |= (np.uint32(reality_state) << 0)
        value |= (np.uint32(information_payload) << 6)
        value |= (np.uint32(coherence_phase) << 12)
        value |= (np.uint32(observer_context) << 18)
        
        offbit = cls(value=value)
        # Ensure meta reflects the initial layer values and custom kwargs
        offbit.meta.update({
            "reality_state": reality_state,
            "information_payload": information_payload,
            "coherence_phase": coherence_phase,
            "observer_context": observer_context,
            **kwargs # Allow passing additional initial meta data
        })
        return offbit

    def to_binary_string(self) -> str:
        """Returns the 32-bit binary representation of the OffBit value."""
        return bin(self.value)[2:].zfill(32)

    def __str__(self):
        return f"OffBit(value={self.value}, binary='{self.to_binary_string()}', meta={self.meta})"
    
    def __eq__(self, other):
        if not isinstance(other, OffBit):
            return NotImplemented
        return self.value == other.value

    def __hash__(self):
        return hash(self.value)


class Bitfield:
    """
    The 6D computational lattice for OffBits.
    Dimensions are loaded from the UBP configuration.
    This is a resonant manifold, not just storage.
    """
    def __init__(self, dimensions: Optional[Tuple[int, ...]] = None):
        """
        Initialize the Bitfield.
        
        Args:
            dimensions: A tuple defining the 6D shape of the Bitfield.
                        If None, it expects dimensions to be set by the UBPConfig
                        when the UBPFramework initializes it.
        """
        if dimensions is None:
            # Placeholder, actual dimensions will be passed from ubp_config via UBPFramework
            self.dimensions = (1, 1, 1, 1, 1, 1) 
            print("⚠️ Bitfield initialized with placeholder dimensions. Expected to be updated by UBPConfig.")
        else:
            self.dimensions = dimensions

        # Initialize the grid with zeros as np.uint32
        # Use np.array for grid to allow direct indexing
        self.grid = np.zeros(self.dimensions, dtype=np.uint32)
        self.total_size = np.prod(self.dimensions)
        print(f"Bitfield created with dimensions {self.dimensions} (Total Cells: {self.total_size}).")

    def set_offbit(self, coords: Tuple[int, ...], offbit: OffBit):
        """
        Places an OffBit at specified 6D coordinates.
        Args:
            coords: A tuple of 6 integers (x, y, z, a, b, c)
            offbit: The OffBit object to place
        Raises:
            IndexError: If coordinates are out of bounds.
        """
        if len(coords) != len(self.dimensions) or not all(0 <= coords[i] < self.dimensions[i] for i in range(len(coords))):
            raise IndexError(f"Coordinates {coords} out of bounds for Bitfield with dimensions {self.dimensions}")
        
        self.grid[coords] = offbit.value
        # Optionally, update OffBit's meta with its source_coords if it's not already set
        if offbit.meta.get("source_coords") is None:
            offbit.meta["source_coords"] = coords

    def get_offbit(self, coords: Tuple[int, ...]) -> OffBit:
        """
        Retrieves an OffBit from specified 6D coordinates.
        Args:
            coords: A tuple of 6 integers (x, y, z, a, b, c)
        Returns:
            The OffBit object at the coordinates.
        Raises:
            IndexError: If coordinates are out of bounds.
        """
        if len(coords) != len(self.dimensions) or not all(0 <= coords[i] < self.dimensions[i] for i in range(len(coords))):
            raise IndexError(f"Coordinates {coords} out of bounds for Bitfield with dimensions {self.dimensions}")
        
        value = self.grid[coords]
        # Reconstruct OffBit from its raw value
        offbit = OffBit(value=value)
        # Populate meta fields based on raw value for accurate representation
        offbit.meta.update({
            "reality_state": offbit.get_layer_value("reality"),
            "information_payload": offbit.get_layer_value("information"),
            "coherence_phase": offbit.get_layer_value("coherence"),
            "observer_context": offbit.get_layer_value("observer"),
            "timestamp": None,
            "realm": "unassigned", # This might need to be retrieved from context, or set by processing logic
            "source_coords": coords # Add source coordinates to meta upon retrieval
        })
        return offbit

    def get_all_offbits(self) -> List[OffBit]:
        """Returns a flat list of all OffBit objects in the Bitfield."""
        all_offbits = []
        # Use np.ndenumerate to get both coordinates and value for each element
        for coords, value in np.ndenumerate(self.grid):
            offbit = OffBit(value=value)
            offbit.meta.update({ # Populate meta for consistency
                "reality_state": offbit.get_layer_value("reality"),
                "information_payload": offbit.get_layer_value("information"),
                "coherence_phase": offbit.get_layer_value("coherence"),
                "observer_context": offbit.get_layer_value("observer"),
                "timestamp": None,
                "realm": "unassigned",
                "source_coords": coords # Add source coordinates
            })
            all_offbits.append(offbit)
        return all_offbits

    def __len__(self):
        return self.total_size

    def __str__(self):
        return f"Bitfield(dimensions={self.dimensions}, total_cells={self.total_size})"

if __name__ == "__main__":
    print("--- Testing OffBit ---")
    # Create an OffBit using the factory method
    ob1 = OffBit.create_offbit(reality_state=10, information_payload=20, coherence_phase=30, observer_context=40, custom_field="test")
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

    # Test creating a default OffBit
    ob2 = OffBit()
    print(f"\nDefault OffBit: {ob2}")
    print(f"Default Meta: {ob2.meta}")

    # Test OffBit with partial meta (should still populate defaults)
    ob3 = OffBit(meta={"custom_init": True})
    print(f"\nOffBit with partial initial meta: {ob3}")
    assert ob3.meta.get('reality_state') is not None
    assert ob3.meta.get('custom_init') is True
    
    print("\n--- Testing Bitfield ---")
    # Using the dimensions from the UBPConfig document for testing
    test_dimensions = (1, 1, 1, 1, 1, 2) # Smaller dimensions for easier testing
    bf = Bitfield(dimensions=test_dimensions)
    print(bf)

    # Place an OffBit
    test_coords_1 = (0, 0, 0, 0, 0, 0)
    test_offbit_1 = OffBit.create_offbit(reality_state=5, information_payload=10)
    bf.set_offbit(test_coords_1, test_offbit_1)
    print(f"Placed OffBit at {test_coords_1}")

    test_coords_2 = (0, 0, 0, 0, 0, 1)
    test_offbit_2 = OffBit.create_offbit(reality_state=7, information_payload=12, coherence_phase=25)
    bf.set_offbit(test_coords_2, test_offbit_2)
    bf.set_offbit(test_coords_2, test_offbit_2) # Place again to test meta update if source_coords is None
    print(f"Placed OffBit at {test_coords_2}")

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

    print(f"Total size of Bitfield: {len(bf)}")

    # Test get_all_offbits
    all_offbits_list = bf.get_all_offbits()
    print(f"\nAll OffBits in Bitfield ({len(all_offbits_list)}):")
    for ob in all_offbits_list:
        print(ob)
    assert len(all_offbits_list) == 2
    assert all_offbits_list[0].get_layer_value('information') == 10
    assert all_offbits_list[0].meta['source_coords'] == test_coords_1
    assert all_offbits_list[1].get_layer_value('information') == 12
    assert all_offbits_list[1].meta['source_coords'] == test_coords_2
    
    print("\n✅ bits.py module test completed successfully!")
