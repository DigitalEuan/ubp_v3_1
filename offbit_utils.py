"""
Universal Binary Principle (UBP) Framework v3.2+ - OffBit Utility Module
Author: Euan Craig, New Zealand
Date: 20 August 2025

This module provides utility functions for manipulating 32-bit OffBit representations
as raw integers, focusing on extracting and setting the 6-bit 'reality' (activation) layer.
"""

import numpy as np

class OffBitUtils:
    """
    Utility class for manipulating 32-bit OffBit representations as raw integers.
    Assumes the 'reality' layer (0-5 bits) serves as the primary activation layer.
    """
    _REALITY_LAYER_MASK = 0x3F  # Binary 111111 (6 bits)
    _REALITY_LAYER_SHIFT = 0    # Bits 0-5

    @staticmethod
    def get_activation_layer(offbit_value: int) -> int:
        """
        Extracts the activation layer (6 bits, corresponding to 'reality' layer)
        from a 32-bit OffBit integer value.
        """
        return (offbit_value >> OffBitUtils._REALITY_LAYER_SHIFT) & OffBitUtils._REALITY_LAYER_MASK

    @staticmethod
    def set_activation_layer(offbit_value: int, new_activation: int) -> int:
        """
        Sets the activation layer (6 bits) within a 32-bit OffBit integer value.
        The new_activation value will be clamped to the 0-63 range.
        """
        # Clamp the new_activation to 6 bits (0-63)
        clamped_activation = max(0, min(new_activation, OffBitUtils._REALITY_LAYER_MASK))
        
        # Clear the old activation layer bits within the 32-bit integer
        # Mask for a 32-bit integer (full range): 0xFFFFFFFF
        clear_mask = ~(OffBitUtils._REALITY_LAYER_MASK << OffBitUtils._REALITY_LAYER_SHIFT) & 0xFFFFFFFF
        cleared_value = offbit_value & clear_mask
        
        # Set the new activation layer bits
        new_value = cleared_value | ((clamped_activation & OffBitUtils._REALITY_LAYER_MASK) << OffBitUtils._REALITY_LAYER_SHIFT)
        
        return new_value
    
    @staticmethod
    def calculate_coherence(offbit_value: int) -> float:
        """
        Calculates a simple coherence score for a single OffBit value based on its activation layer.
        Higher activation (closer to 63 or 0) indicates more 'definite' state, thus higher coherence.
        """
        activation = OffBitUtils.get_activation_layer(offbit_value)
        # Normalize activation to be 0-1, then calculate deviation from mid-point (31.5)
        # A value of 0 or 63 is highly coherent, 31-32 is least coherent.
        deviation_from_mid = abs(activation - 31.5) / 31.5
        coherence = deviation_from_mid # Scales from 0 (mid) to 1 (extremes)
        return float(max(0.0, min(1.0, coherence)))
