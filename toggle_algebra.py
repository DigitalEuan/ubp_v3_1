"""
Universal Binary Principle (UBP) Framework v3.1.1 - Enhanced Toggle Algebra Module
Author: Euan Craig, New Zealand
Date: 18 August 2025

This module implements the comprehensive Toggle Algebra operations engine,
providing both basic Boolean operations and advanced physics-inspired
operations for OffBit manipulation within the UBP framework.

Enhanced for v3.1.1 with improved integration with v3.0 components and
better performance optimization.
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
import math
import time
from scipy.special import factorial
from scipy.optimize import minimize_scalar
import json # Added missing import

try:
    from .system_constants import UBPConstants # Corrected import from 'core' to 'system_constants'
    from .bits import Bitfield, OffBit # Corrected import from 'bitfield' to 'bits'
    from .hex_dictionary import HexDictionary
except ImportError:
    from system_constants import UBPConstants # Corrected import from 'core' to 'system_constants'
    from bits import Bitfield, OffBit # Corrected import from 'bitfield' to 'bits'
    from hex_dictionary import HexDictionary


@dataclass
class ToggleOperationResult:
    """Result of a toggle algebra operation."""
    result_value: int
    operation_type: str
    input_values: List[int]
    coherence_change: float
    energy_delta: float
    execution_time: float
    nrci_score: float = 0.0
    stability_score: float = 0.0


@dataclass
class ToggleAlgebraMetrics:
    """Performance metrics for toggle algebra operations."""
    total_operations: int
    successful_operations: int
    average_coherence: float
    total_energy_change: float
    operation_distribution: Dict[str, int]
    average_execution_time: float
    resonance_stability: float
    average_nrci: float = 0.0


class ToggleAlgebra:
    """
    Enhanced Toggle Algebra engine for UBP OffBit operations.
    
    This class provides the fundamental bit-level operations that drive
    all dynamic processes in the UBP framework, from basic Boolean logic
    to advanced physics-inspired operations.
    
    Enhanced for v3.1 with better integration and performance.
    """
    
    def __init__(self, bitfield_instance: Optional[Bitfield] = None, 
                 glr_framework=None,
                 hex_dictionary_instance: Optional[HexDictionary] = None):
        """
        Initialize the Toggle Algebra engine.
        
        Args:
            bitfield_instance: Optional Bitfield instance for operations
            glr_framework: Optional GLR framework for error correction
            hex_dictionary_instance: Optional HexDictionary for data storage
        """
        self.bitfield = bitfield_instance
        self.glr_framework = glr_framework
        self.hex_dictionary = hex_dictionary_instance or HexDictionary()
        self.operation_history = []
        self.metrics = ToggleAlgebraMetrics(
            total_operations=0,
            successful_operations=0,
            average_coherence=0.0,
            total_energy_change=0.0,
            operation_distribution={},
            average_execution_time=0.0,
            resonance_stability=1.0
        )
        
        # Enhanced operation registry for v3.1
        self.operations = {
            # Basic Boolean Operations
            'AND': self.and_operation,
            'OR': self.or_operation,
            'XOR': self.xor_operation,
            'NOT': self.not_operation,
            'NAND': self.nand_operation,
            'NOR': self.nor_operation,
            
            # Advanced Physics-Inspired Operations
            'RESONANCE': self.resonance_operation,
            'ENTANGLEMENT': self.entanglement_operation,
            'SUPERPOSITION': self.superposition_operation,
            'SPIN_TRANSITION': self.spin_transition_operation,
            'HYBRID_PROM': self.hybrid_prom_operation,
            
            # Electromagnetic Operations (WGE)
            'NONLINEAR_MAXWELL': self.nonlinear_maxwell_operation,
            'LORENTZ_FORCE': self.lorentz_force_operation,
            'WEYL_METRIC': self.weyl_metric_operation,
            
            # Rune Protocol Operations
            'GLYPH_QUANTIFY': self.glyph_quantify_operation,
            'GLYPH_CORRELATE': self.glyph_correlate_operation,
            'GLYPH_SELF_REFERENCE': self.glyph_self_reference_operation,
            
            # New v3.1 Operations
            'HTR_RESONANCE': self.htr_resonance_operation,
            'CRV_MODULATION': self.crv_modulation_operation,
            'NRCI_OPTIMIZATION': self.nrci_optimization_operation,
            'QUANTUM_COHERENCE': self.quantum_coherence_operation,
            'TEMPORAL_SYNC': self.temporal_sync_operation
        }
        
        # Operation cache for performance
        self.operation_cache: Dict[str, ToggleOperationResult] = {}
        
        print("✅ UBP Toggle Algebra Engine v3.1 Initialized")
        print(f"   Available Operations: {len(self.operations)}")
        print(f"   Bitfield Connected: {'Yes' if bitfield_instance else 'No'}")
        print(f"   HexDictionary Integration: {'Enabled' if self.hex_dictionary else 'Disabled'}")
    
    def _record_operation(self, result: ToggleOperationResult) -> None:
        """Record an operation in the history and update metrics."""
        self.operation_history.append(result)
        
        # Update metrics
        self.metrics.total_operations += 1
        if result.result_value is not None:
            self.metrics.successful_operations += 1
        
        # Update operation distribution
        op_type = result.operation_type
        if op_type not in self.metrics.operation_distribution:
            self.metrics.operation_distribution[op_type] = 0
        self.metrics.operation_distribution[op_type] += 1
        
        # Update running averages
        self.metrics.total_energy_change += result.energy_delta
        
        if self.metrics.successful_operations > 0:
            total_coherence = sum(op.coherence_change for op in self.operation_history)
            self.metrics.average_coherence = total_coherence / self.metrics.successful_operations
            
            total_time = sum(op.execution_time for op in self.operation_history)
            self.metrics.average_execution_time = total_time / self.metrics.successful_operations
            
            total_nrci = sum(op.nrci_score for op in self.operation_history)
            self.metrics.average_nrci = total_nrci / self.metrics.successful_operations
        
        # Store in HexDictionary if available
        if self.hex_dictionary:
            operation_data = {
                'operation_type': result.operation_type,
                'input_values': result.input_values,
                'result_value': result.result_value,
                'coherence_change': result.coherence_change,
                'energy_delta': result.energy_delta,
                'execution_time': result.execution_time,
                'nrci_score': result.nrci_score,
                'timestamp': time.time()
            }
            self.hex_dictionary.store(operation_data, 'json', 
                                    {'operation_type': result.operation_type})
    
    # ========================================================================
    # BASIC BOOLEAN OPERATIONS
    # ========================================================================
    
    def and_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform AND operation: min(b_i, b_j)
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            
        Returns:
            ToggleOperationResult with AND result
        """
        start_time = time.time()
        
        # Extract activation layers for the operation
        activation_i = OffBit.get_activation_layer(b_i)
        activation_j = OffBit.get_activation_layer(b_j)
        
        # Perform AND on activation layers
        result_activation = min(activation_i, activation_j)
        
        # Create result OffBit preserving other layers from b_i
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        # Calculate coherence change
        coherence_before = (OffBit.calculate_coherence(b_i) + OffBit.calculate_coherence(b_j)) / 2
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        # Calculate NRCI score
        nrci_score = self._calculate_nrci([b_i, b_j], result)
        
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="AND",
            input_values=[b_i, b_j],
            coherence_change=coherence_change,
            energy_delta=coherence_change * 0.1,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def or_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform OR operation: max(b_i, b_j)
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            
        Returns:
            ToggleOperationResult with OR result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        activation_j = OffBit.get_activation_layer(b_j)
        
        result_activation = max(activation_i, activation_j)
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        coherence_before = (OffBit.calculate_coherence(b_i) + OffBit.calculate_coherence(b_j)) / 2
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i, b_j], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="OR",
            input_values=[b_i, b_j],
            coherence_change=coherence_change,
            energy_delta=coherence_change * 0.1,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def xor_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform XOR operation: |b_i - b_j|
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            
        Returns:
            ToggleOperationResult with XOR result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        activation_j = OffBit.get_activation_layer(b_j)
        
        result_activation = abs(activation_i - activation_j)
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        coherence_before = (OffBit.calculate_coherence(b_i) + OffBit.calculate_coherence(b_j)) / 2
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i, b_j], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="XOR",
            input_values=[b_i, b_j],
            coherence_change=coherence_change,
            energy_delta=coherence_change * 0.1,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def not_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform NOT operation: invert activation layer
        
        Args:
            b_i: OffBit value to invert
            
        Returns:
            ToggleOperationResult with NOT result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        result_activation = 63 - activation_i  # Invert 6-bit value
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="NOT",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * 0.1,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def nand_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """Perform NAND operation: NOT(AND(b_i, b_j))"""
        and_result = self.and_operation(b_i, b_j)
        not_result = self.not_operation(and_result.result_value)
        
        result_obj = ToggleOperationResult(
            result_value=not_result.result_value,
            operation_type="NAND",
            input_values=[b_i, b_j],
            coherence_change=not_result.coherence_change,
            energy_delta=not_result.energy_delta,
            execution_time=and_result.execution_time + not_result.execution_time,
            nrci_score=not_result.nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def nor_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """Perform NOR operation: NOT(OR(b_i, b_j))"""
        or_result = self.or_operation(b_i, b_j)
        not_result = self.not_operation(or_result.result_value)
        
        result_obj = ToggleOperationResult(
            result_value=not_result.result_value,
            operation_type="NOR",
            input_values=[b_i, b_j],
            coherence_change=not_result.coherence_change,
            energy_delta=not_result.energy_delta,
            execution_time=or_result.execution_time + not_result.execution_time,
            nrci_score=not_result.nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    # ========================================================================
    # ADVANCED PHYSICS-INSPIRED OPERATIONS
    # ========================================================================
    
    def resonance_operation(self, b_i: int, frequency: float = None, **kwargs) -> ToggleOperationResult:
        """
        Perform resonance operation: b_i * exp(-0.0002 * d^2)
        
        Args:
            b_i: OffBit value
            frequency: Resonance frequency (default: quantum CRV)
            
        Returns:
            ToggleOperationResult with resonance result
        """
        start_time = time.time()
        
        if frequency is None:
            # Removed direct UBPConstants.CRV_QUANTUM access. This is now part of UBPConfig.
            # For this test file to run standalone, a mock or a direct value is needed.
            # In a full framework, this would pull from the initialized UBPConfig.
            # Assuming a default for standalone test.
            frequency = 4.58e14 # Default to a quantum realm frequency example

        # Calculate resonance effect
        t = kwargs.get('time', 1.0)
        d = t * frequency
        resonance_factor = math.exp(-0.0002 * d * d)
        
        # Apply resonance to activation layer
        activation_i = OffBit.get_activation_layer(b_i)
        result_activation = int(activation_i * resonance_factor)
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="RESONANCE",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * resonance_factor,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def entanglement_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform entanglement operation: b_i * b_j * C_ij
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            
        Returns:
            ToggleOperationResult with entanglement result
        """
        start_time = time.time()
        
        # Calculate coherence coefficient
        coherence_ij = kwargs.get('coherence_coefficient', 0.95)
        
        activation_i = OffBit.get_activation_layer(b_i)
        activation_j = OffBit.get_activation_layer(b_j)
        
        # Entanglement operation
        entangled_activation = int((activation_i * activation_j * coherence_ij) / 64)
        result = OffBit.set_activation_layer(b_i, entangled_activation)
        
        coherence_before = (OffBit.calculate_coherence(b_i) + OffBit.calculate_coherence(b_j)) / 2
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i, b_j], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="ENTANGLEMENT",
            input_values=[b_i, b_j],
            coherence_change=coherence_change,
            energy_delta=coherence_change * coherence_ij,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def superposition_operation(self, states: List[int], weights: List[float] = None, **kwargs) -> ToggleOperationResult:
        """
        Perform superposition operation: sum(states * weights)
        
        Args:
            states: List of OffBit states
            weights: List of weights (must sum to 1)
            
        Returns:
            ToggleOperationResult with superposition result
        """
        start_time = time.time()
        
        if weights is None:
            weights = [1.0 / len(states)] * len(states)
        
        if len(states) != len(weights):
            raise ValueError("States and weights must have same length")
        
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError("Weights must sum to 1.0")
        
        # Calculate superposition
        superposition_activation = 0
        for state, weight in zip(states, weights):
            activation = OffBit.get_activation_layer(state)
            superposition_activation += activation * weight
        
        result_activation = int(superposition_activation)
        result = OffBit.set_activation_layer(states[0], result_activation)
        
        coherence_before = np.mean([OffBit.calculate_coherence(state) for state in states])
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci(states, result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="SUPERPOSITION",
            input_values=states,
            coherence_change=coherence_change,
            energy_delta=coherence_change * sum(weights),
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def spin_transition_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform spin transition operation: b_i * ln(1 / p_s)
        
        Args:
            b_i: OffBit value
            
        Returns:
            ToggleOperationResult with spin transition result
        """
        start_time = time.time()
        
        # Removed direct UBPConstants.CRV_QUANTUM access. Assuming a default for standalone test.
        p_s = kwargs.get('spin_probability', 0.2265234857) # Example value formerly UBPConstants.CRV_QUANTUM
        
        activation_i = OffBit.get_activation_layer(b_i)
        
        # Spin transition calculation
        if p_s > 0:
            spin_factor = math.log(1.0 / p_s)
            result_activation = int(activation_i * spin_factor) % 64
        else:
            result_activation = activation_i
        
        result = OffBit.set_activation_layer(b_i, result_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="SPIN_TRANSITION",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * spin_factor if p_s > 0 else 0,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def hybrid_prom_operation(self, b_i: int, b_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform hybrid PROM operation: |b_i - b_j| * exp(-0.0002 * d^2)
        
        Args:
            b_i: First OffBit value
            b_j: Second OffBit value
            
        Returns:
            ToggleOperationResult with hybrid PROM result
        """
        start_time = time.time()
        
        # First perform XOR
        xor_result = self.xor_operation(b_i, b_j)
        
        # Then apply resonance
        # Removed direct UBPConstants.CRV_ELECTROMAGNETIC access.
        frequency = kwargs.get('frequency', 3.141593) # Example value for EM CRV
        resonance_result = self.resonance_operation(xor_result.result_value, frequency)
        
        nrci_score = self._calculate_nrci([b_i, b_j], resonance_result.result_value)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=resonance_result.result_value,
            operation_type="HYBRID_PROM",
            input_values=[b_i, b_j],
            coherence_change=resonance_result.coherence_change,
            energy_delta=resonance_result.energy_delta,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    # ========================================================================
    # ELECTROMAGNETIC OPERATIONS (WGE)
    # ========================================================================
    
    def nonlinear_maxwell_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform nonlinear Maxwell operation for electromagnetic field dynamics.
        
        Args:
            b_i: OffBit representing electromagnetic field state
            
        Returns:
            ToggleOperationResult with Maxwell field result
        """
        start_time = time.time()
        
        # Simulate nonlinear Maxwell equation effects
        activation_i = OffBit.get_activation_layer(b_i)
        
        # Apply Weyl metric influence
        weyl_factor = kwargs.get('weyl_factor', 1.1)
        maxwell_activation = int(activation_i * weyl_factor) % 64
        
        result = OffBit.set_activation_layer(b_i, maxwell_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="NONLINEAR_MAXWELL",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * weyl_factor,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def lorentz_force_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform Lorentz force operation for charged particle dynamics.
        
        Args:
            b_i: OffBit representing charged particle state
            
        Returns:
            ToggleOperationResult with Lorentz force result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        
        # Simulate Lorentz force effects
        charge = kwargs.get('charge', 1.0)
        # Removed direct UBPConstants.CRV_ELECTROMAGNETIC access.
        field_strength = kwargs.get('field_strength', 3.141593) # Example value for EM CRV
        
        lorentz_activation = int(activation_i * charge * field_strength) % 64
        result = OffBit.set_activation_layer(b_i, lorentz_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="LORENTZ_FORCE",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * field_strength,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def weyl_metric_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform Weyl metric operation for spacetime geometry effects.
        
        Args:
            b_i: OffBit representing spacetime state
            
        Returns:
            ToggleOperationResult with Weyl metric result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        
        # Apply Weyl metric transformation
        metric_factor = kwargs.get('metric_factor', 1.2)
        weyl_activation = int(activation_i * metric_factor) % 64
        
        result = OffBit.set_activation_layer(b_i, weyl_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="WEYL_METRIC",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * metric_factor,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    # ========================================================================
    # RUNE PROTOCOL OPERATIONS
    # ========================================================================
    
    def glyph_quantify_operation(self, glyph_state: int, **kwargs) -> ToggleOperationResult:
        """
        Perform glyph quantification operation: Q(G, state) = sum(G_i(state))
        
        Args:
            glyph_state: OffBit representing glyph state
            
        Returns:
            ToggleOperationResult with quantification result
        """
        start_time = time.time()
        
        activation = OffBit.get_activation_layer(glyph_state)
        
        # Quantify glyph components
        glyph_components = [(activation >> i) & 1 for i in range(6)]
        quantification = sum(glyph_components)
        
        result = OffBit.set_activation_layer(glyph_state, quantification)
        
        coherence_before = OffBit.calculate_coherence(glyph_state)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([glyph_state], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="GLYPH_QUANTIFY",
            input_values=[glyph_state],
            coherence_change=coherence_change,
            energy_delta=coherence_change * quantification,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def glyph_correlate_operation(self, glyph_i: int, glyph_j: int, **kwargs) -> ToggleOperationResult:
        """
        Perform glyph correlation operation: C(G, R_i, R_j) = P(R_i) * P(R_j) / P(R_i ∩ R_j)
        
        Args:
            glyph_i: First glyph OffBit
            glyph_j: Second glyph OffBit
            
        Returns:
            ToggleOperationResult with correlation result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(glyph_i)
        activation_j = OffBit.get_activation_layer(glyph_j)
        
        # Calculate probabilities
        p_i = activation_i / 64.0
        p_j = activation_j / 64.0
        
        # Calculate intersection (AND operation)
        intersection = (activation_i & activation_j) / 64.0
        
        # Correlation calculation
        if intersection > 0:
            correlation = (p_i * p_j) / intersection
        else:
            correlation = 0.0
        
        result_activation = int(correlation * 64) % 64
        result = OffBit.set_activation_layer(glyph_i, result_activation)
        
        coherence_before = (OffBit.calculate_coherence(glyph_i) + OffBit.calculate_coherence(glyph_j)) / 2
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([glyph_i, glyph_j], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="GLYPH_CORRELATE",
            input_values=[glyph_i, glyph_j],
            coherence_change=coherence_change,
            energy_delta=coherence_change * correlation,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def glyph_self_reference_operation(self, glyph_state: int, **kwargs) -> ToggleOperationResult:
        """
        Perform glyph self-reference operation with recursive feedback.
        
        Args:
            glyph_state: OffBit representing glyph state
            
        Returns:
            ToggleOperationResult with self-reference result
        """
        start_time = time.time()
        
        activation = OffBit.get_activation_layer(glyph_state)
        
        # Self-reference with feedback
        feedback_factor = kwargs.get('feedback_factor', 0.8)
        iterations = kwargs.get('iterations', 3)
        
        current_activation = activation
        for _ in range(iterations):
            # Apply self-reference transformation
            current_activation = int(current_activation * feedback_factor + 
                                   (current_activation >> 1)) % 64
        
        result = OffBit.set_activation_layer(glyph_state, current_activation)
        
        coherence_before = OffBit.calculate_coherence(glyph_state)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([glyph_state], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="GLYPH_SELF_REFERENCE",
            input_values=[glyph_state],
            coherence_change=coherence_change,
            energy_delta=coherence_change * feedback_factor,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    # ========================================================================
    # NEW v3.1 OPERATIONS
    # ========================================================================
    
    def htr_resonance_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform HTR (Harmonic Toggle Resonance) operation.
        
        Args:
            b_i: OffBit value
            
        Returns:
            ToggleOperationResult with HTR result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        
        # HTR parameters
        harmonic_order = kwargs.get('harmonic_order', 3)
        # Removed direct UBPConstants.CRV_BIOLOGICAL access.
        base_frequency = kwargs.get('base_frequency', 10.0) # Example value for Biological CRV
        
        # Apply harmonic resonance
        htr_activation = activation_i
        for h in range(1, harmonic_order + 1):
            harmonic_freq = h * base_frequency
            harmonic_effect = math.sin(2 * math.pi * harmonic_freq) * (1.0 / h)
            htr_activation += int(activation_i * harmonic_effect * 0.1)
        
        htr_activation = htr_activation % 64
        result = OffBit.set_activation_layer(b_i, htr_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="HTR_RESONANCE",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * harmonic_order,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def crv_modulation_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform CRV (Core Resonance Value) modulation operation.
        
        Args:
            b_i: OffBit value
            
        Returns:
            ToggleOperationResult with CRV modulation result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        
        # CRV modulation parameters
        crv_type = kwargs.get('crv_type', 'quantum')
        modulation_depth = kwargs.get('modulation_depth', 0.2)
        
        # Get CRV value (mock values if UBPConstants is not fully loaded)
        crv_values = {
            'quantum': 0.2265234857, # Example value for Quantum CRV
            'electromagnetic': 3.141593, # Example value for EM CRV
            'gravitational': 100.0, # Example value for Gravitational CRV
            'biological': 10.0, # Example value for Biological CRV
            'cosmological': 0.83203682, # Example value for Cosmological CRV
            'nuclear': 1.2356e20, # Example value for Nuclear CRV
            'optical': 5.0e14 # Example value for Optical CRV
        }
        
        crv_value = crv_values.get(crv_type, crv_values['quantum'])
        
        # Apply CRV modulation
        modulation_factor = 1 + modulation_depth * math.sin(2 * math.pi * crv_value)
        crv_activation = int(activation_i * modulation_factor) % 64
        
        result = OffBit.set_activation_layer(b_i, crv_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="CRV_MODULATION",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * crv_value,
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def nrci_optimization_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform NRCI optimization operation to maximize coherence.
        
        Args:
            b_i: OffBit value
            
        Returns:
            ToggleOperationResult with NRCI optimized result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        # Removed direct UBPConstants.NRCI_TARGET access.
        target_nrci = kwargs.get('target_nrci', 0.999999) # Example value for NRCI_TARGET
        
        # Optimize activation for maximum NRCI
        best_activation = activation_i
        best_nrci = 0.0
        
        # Try different activation values
        for test_activation in range(max(0, activation_i - 5), min(64, activation_i + 6)):
            test_offbit = OffBit.set_activation_layer(b_i, test_activation)
            test_nrci = self._calculate_nrci([b_i], test_offbit)
            
            if test_nrci > best_nrci:
                best_nrci = test_nrci
                best_activation = test_activation
        
        result = OffBit.set_activation_layer(b_i, best_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="NRCI_OPTIMIZATION",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * best_nrci,
            execution_time=execution_time,
            nrci_score=best_nrci
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def quantum_coherence_operation(self, states: List[int], **kwargs) -> ToggleOperationResult:
        """
        Perform quantum coherence operation across multiple states.
        
        Args:
            states: List of OffBit quantum states
            
        Returns:
            ToggleOperationResult with quantum coherence result
        """
        start_time = time.time()
        
        if not states:
            raise ValueError("At least one quantum state required")
        
        # Calculate quantum coherence
        activations = [OffBit.get_activation_layer(state) for state in states]
        
        # Apply quantum coherence formula
        coherence_sum = sum(activations)
        coherence_product = 1
        for activation in activations:
            if activation > 0:
                coherence_product *= activation
        
        # Quantum coherence result
        if len(activations) > 1:
            quantum_activation = int(math.sqrt(coherence_product)) % 64
        else:
            quantum_activation = activations[0]
        
        result = OffBit.set_activation_layer(states[0], quantum_activation)
        
        coherence_before = np.mean([OffBit.calculate_coherence(state) for state in states])
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci(states, result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="QUANTUM_COHERENCE",
            input_values=states,
            coherence_change=coherence_change,
            energy_delta=coherence_change * len(states),
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    def temporal_sync_operation(self, b_i: int, **kwargs) -> ToggleOperationResult:
        """
        Perform temporal synchronization operation using CSC.
        
        Args:
            b_i: OffBit value
            
        Returns:
            ToggleOperationResult with temporal sync result
        """
        start_time = time.time()
        
        activation_i = OffBit.get_activation_layer(b_i)
        
        # Temporal synchronization parameters
        # Removed direct UBPConstants.CSC_PERIOD access.
        csc_period = kwargs.get('csc_period', 1 / np.pi) # Example value for CSC_PERIOD
        sync_phase = kwargs.get('sync_phase', 0.0)
        
        # Apply temporal synchronization
        sync_factor = math.cos(2 * math.pi * sync_phase / csc_period)
        sync_activation = int(activation_i * (1 + 0.1 * sync_factor)) % 64
        
        result = OffBit.set_activation_layer(b_i, sync_activation)
        
        coherence_before = OffBit.calculate_coherence(b_i)
        coherence_after = OffBit.calculate_coherence(result)
        coherence_change = coherence_after - coherence_before
        
        nrci_score = self._calculate_nrci([b_i], result)
        execution_time = time.time() - start_time
        
        result_obj = ToggleOperationResult(
            result_value=result,
            operation_type="TEMPORAL_SYNC",
            input_values=[b_i],
            coherence_change=coherence_change,
            energy_delta=coherence_change * abs(sync_factor),
            execution_time=execution_time,
            nrci_score=nrci_score
        )
        
        self._record_operation(result_obj)
        return result_obj
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def _calculate_nrci(self, inputs: List[int], result: int) -> float:
        """
        Calculate Non-Random Coherence Index for an operation.
        
        Args:
            inputs: Input OffBit values
            result: Result OffBit value
            
        Returns:
            NRCI score (0.0 to 1.0)
        """
        if not inputs:
            return 0.0
        
        # Calculate expected vs actual patterns
        input_activations = [OffBit.get_activation_layer(inp) for inp in inputs]
        result_activation = OffBit.get_activation_layer(result)
        
        # Simple NRCI calculation based on coherence
        input_mean = np.mean(input_activations)
        input_std = np.std(input_activations) if len(input_activations) > 1 else 1.0
        
        if input_std > 0:
            deviation = abs(result_activation - input_mean) / input_std
            nrci = max(0.0, 1.0 - deviation / 10.0)  # Normalize to 0-1 range
        else:
            nrci = 1.0
        
        return min(1.0, nrci)
    
    def execute_operation(self, operation_name: str, *args, **kwargs) -> ToggleOperationResult:
        """
        Execute a named toggle operation.
        
        Args:
            operation_name: Name of the operation to execute
            *args: Positional arguments for the operation
            **kwargs: Keyword arguments for the operation
            
        Returns:
            ToggleOperationResult
        """
        if operation_name not in self.operations:
            raise ValueError(f"Unknown operation: {operation_name}")
        
        operation_func = self.operations[operation_name]
        return operation_func(*args, **kwargs)
    
    def get_metrics(self) -> ToggleAlgebraMetrics:
        """Get current toggle algebra metrics."""
        return self.metrics
    
    def clear_history(self) -> None:
        """Clear operation history and reset metrics."""
        self.operation_history.clear()
        self.operation_cache.clear()
        self.metrics = ToggleAlgebraMetrics(
            total_operations=0,
            successful_operations=0,
            average_coherence=0.0,
            total_energy_change=0.0,
            operation_distribution={},
            average_execution_time=0.0,
            resonance_stability=1.0
        )
        print("✅ Toggle Algebra history and metrics cleared")
    
    def export_operations(self, file_path: str) -> bool:
        """
        Export operation history to a file.
        
        Args:
            file_path: Path to export file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'operations': [],
                'metrics': self.metrics.__dict__
            }
            
            for op in self.operation_history:
                op_data = {
                    'operation_type': op.operation_type,
                    'input_values': op.input_values,
                    'result_value': op.result_value,
                    'coherence_change': op.coherence_change,
                    'energy_delta': op.energy_delta,
                    'execution_time': op.execution_time,
                    'nrci_score': op.nrci_score
                }
                export_data['operations'].append(op_data)
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported {len(self.operation_history)} operations to {file_path}")
            return True
            
        except Exception as e:
            print(f"❌ Export failed: {e}")
            return False


# ========================================================================
# UTILITY FUNCTIONS
# ========================================================================

def create_toggle_algebra(bitfield: Optional[Bitfield] = None,
                         glr_framework=None,
                         hex_dictionary: Optional[HexDictionary] = None) -> ToggleAlgebra:
    """
    Create and return a new ToggleAlgebra instance.
    
    Args:
        bitfield: Optional Bitfield instance
        glr_framework: Optional GLR framework
        hex_dictionary: Optional HexDictionary instance
        
    Returns:
        Initialized ToggleAlgebra instance
    """
    return ToggleAlgebra(bitfield, glr_framework, hex_dictionary)


def benchmark_toggle_algebra(algebra: ToggleAlgebra, num_operations: int = 1000) -> Dict[str, float]:
    """
    Benchmark ToggleAlgebra performance.
    
    Args:
        algebra: ToggleAlgebra instance to benchmark
        num_operations: Number of operations to perform
        
    Returns:
        Dictionary with benchmark results
    """
    import random
    
    start_time = time.time()
    
    # Test various operations
    operations = ['AND', 'OR', 'XOR', 'RESONANCE', 'ENTANGLEMENT', 'HTR_RESONANCE']
    
    for i in range(num_operations):
        op_name = random.choice(operations)
        
        if op_name in ['AND', 'OR', 'XOR', 'ENTANGLEMENT']:
            b_i = random.randint(0, 0xFFFFFF)
            b_j = random.randint(0, 0xFFFFFF)
            algebra.execute_operation(op_name, b_i, b_j)
        else:
            b_i = random.randint(0, 0xFFFFFF)
            algebra.execute_operation(op_name, b_i)
    
    total_time = time.time() - start_time
    metrics = algebra.get_metrics()
    
    return {
        'total_time': total_time,
        'operations_per_second': num_operations / total_time,
        'average_execution_time': metrics.average_execution_time,
        'average_coherence': metrics.average_coherence,
        'average_nrci': metrics.average_nrci,
        'successful_operations': metrics.successful_operations,
        'total_operations': metrics.total_operations
    }


if __name__ == "__main__":
    # Test the Toggle Algebra
    print("🧪 Testing Toggle Algebra v3.1...")
    
    algebra = create_toggle_algebra()
    
    # Test basic operations
    b1 = 0x123456
    b2 = 0x654321
    
    and_result = algebra.and_operation(b1, b2)
    or_result = algebra.or_operation(b1, b2)
    xor_result = algebra.xor_operation(b1, b2)
    
    print(f"AND result: {hex(and_result.result_value)}, NRCI: {and_result.nrci_score:.3f}")
    print(f"OR result: {hex(or_result.result_value)}, NRCI: {or_result.nrci_score:.3f}")
    print(f"XOR result: {hex(xor_result.result_value)}, NRCI: {xor_result.nrci_score:.3f}")
    
    # Test advanced operations
    resonance_result = algebra.resonance_operation(b1)
    htr_result = algebra.htr_resonance_operation(b1)
    crv_result = algebra.crv_modulation_operation(b1, crv_type='quantum')
    
    print(f"Resonance result: {hex(resonance_result.result_value)}, NRCI: {resonance_result.nrci_score:.3f}")
    print(f"HTR result: {hex(htr_result.result_value)}, NRCI: {htr_result.nrci_score:.3f}")
    print(f"CRV result: {hex(crv_result.result_value)}, NRCI: {crv_result.nrci_score:.3f}")
    
    # Test metrics
    metrics = algebra.get_metrics()
    print(f"Total operations: {metrics.total_operations}")
    print(f"Average NRCI: {metrics.average_nrci:.3f}")
    print(f"Average coherence: {metrics.average_coherence:.3f}")
    
    print("✅ Toggle Algebra v3.1 test completed successfully!")
