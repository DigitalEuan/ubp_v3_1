"""
Universal Binary Principle (UBP) Framework v3.1.1 - OffBit
Author: Euan Craig, New Zealand
Date: 18 August 2025
"""
class OffBit:
    """
    Utility class for manipulating 24-bit OffBit representations.
    An OffBit integer conceptually contains various flags and an activation layer.
    The activation layer is assumed to be 6 bits (0-63) for UBP purposes.
    """
    _ACTIVATION_LAYER_MASK = 0x3F  # Binary 111111 (6 bits)
    _ACTIVATION_LAYER_SHIFT = 0    # Placed at the lowest 6 bits for simplicity

    @staticmethod
    def get_activation_layer(offbit_value: int) -> int:
        """
        Extracts the activation layer (6 bits) from a 24-bit OffBit integer.
        """
        return (offbit_value >> OffBit._ACTIVATION_LAYER_SHIFT) & OffBit._ACTIVATION_LAYER_MASK

    @staticmethod
    def set_activation_layer(offbit_value: int, new_activation: int) -> int:
        """
        Sets the activation layer (6 bits) within a 24-bit OffBit integer.
        The new_activation value will be clamped to the 0-63 range.
        """
        # Clamp the new_activation to 6 bits (0-63)
        clamped_activation = max(0, min(new_activation, OffBit._ACTIVATION_LAYER_MASK))
        
        # Clear the old activation layer bits within the 24-bit integer
        # Mask for a 24-bit integer: 0xFFFFFF (binary 111111111111111111111111)
        clear_mask = ~(OffBit._ACTIVATION_LAYER_MASK << OffBit._ACTIVATION_LAYER_SHIFT) & 0xFFFFFF
        cleared_value = offbit_value & clear_mask
        
        # Set the new activation layer bits
        new_value = cleared_value | ((clamped_activation & OffBit._ACTIVATION_LAYER_MASK) << OffBit._ACTIVATION_LAYER_SHIFT)
        
        return new_value
