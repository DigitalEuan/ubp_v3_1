"""
Universal Binary Principle (UBP) Framework v2.0 - Automatic Realm Selection System

This module implements intelligent automatic realm selection based on problem
characteristics, data analysis, and computational requirements. The system
analyzes input data and selects the optimal realm(s) for computation.

Author: Euan Craig
Version: 2.0
Date: August 2025
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
import math
from scipy.stats import entropy, skew, kurtosis
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, welch
import json

from core import UBPConstants


@dataclass
class DataCharacteristics:
    """Characteristics extracted from input data for realm selection."""
    size: int
    complexity: float
    entropy_value: float
    dominant_frequency: float
    frequency_spread: float
    coherence_estimate: float
    noise_level: float
    dimensionality: int
    data_type: str
    statistical_moments: Dict[str, float]
    spectral_features: Dict[str, float]


@dataclass
class RealmScore:
    """Score and reasoning for a specific realm selection."""
    realm_name: str
    score: float
    confidence: float
    reasoning: List[str]
    expected_nrci: float
    computational_cost: float
    frequency_match: float
    scale_compatibility: float


@dataclass
class RealmSelectionResult:
    """Result of automatic realm selection process."""
    primary_realm: str
    secondary_realms: List[str]
    realm_scores: List[RealmScore]
    selection_confidence: float
    multi_realm_recommended: bool
    reasoning: List[str]
    data_characteristics: DataCharacteristics


class AutomaticRealmSelector:
    """
    Intelligent automatic realm selection system for the UBP Framework.
    
    This class analyzes input data characteristics and automatically selects
    the optimal computational realm(s) based on frequency, scale, complexity,
    and other factors.
    """
    
    def __init__(self):
        """Initialize the automatic realm selector."""
        
        # Realm frequency ranges and characteristics
        self.realm_characteristics = {
            'electromagnetic': {
                'frequency_range': (1e6, 1e12),  # MHz to THz
                'wavelength_range': (3e-4, 300),  # 0.3mm to 300m
                'optimal_complexity': 0.5,
                'coordination_number': 6,
                'crv_frequency': 3.141593,
                'typical_nrci': 1.0,
                'computational_cost': 1.0,
                'best_for': ['electromagnetic_fields', 'radio_waves', 'microwaves', 'classical_physics']
            },
            'quantum': {
                'frequency_range': (1e13, 1e16),  # 10-1000 THz
                'wavelength_range': (3e-8, 3e-5),  # 30nm to 30μm
                'optimal_complexity': 0.8,
                'coordination_number': 4,
                'crv_frequency': 4.58e14,
                'typical_nrci': 0.875,
                'computational_cost': 2.0,
                'best_for': ['quantum_mechanics', 'atomic_physics', 'molecular_dynamics', 'coherent_states']
            },
            'gravitational': {
                'frequency_range': (1e-4, 1e4),  # mHz to 10kHz
                'wavelength_range': (3e4, 3e12),  # 30km to 3000Gm
                'optimal_complexity': 0.3,
                'coordination_number': 12,
                'crv_frequency': 100,
                'typical_nrci': 0.915,
                'computational_cost': 1.5,
                'best_for': ['gravitational_waves', 'large_scale_dynamics', 'cosmology', 'general_relativity']
            },
            'biological': {
                'frequency_range': (1e-2, 1e3),  # 10mHz to 1kHz
                'wavelength_range': (3e5, 3e10),  # 300km to 30Gm
                'optimal_complexity': 0.7,
                'coordination_number': 20,
                'crv_frequency': 10,
                'typical_nrci': 0.911,
                'computational_cost': 2.5,
                'best_for': ['biological_systems', 'neural_networks', 'eeg_signals', 'life_processes']
            },
            'cosmological': {
                'frequency_range': (1e-18, 1e-10),  # aHz to 0.1nHz
                'wavelength_range': (3e18, 3e26),  # 3Em to 300Ym
                'optimal_complexity': 0.9,
                'coordination_number': 12,
                'crv_frequency': 1e-11,
                'typical_nrci': 0.797,
                'computational_cost': 3.0,
                'best_for': ['cosmic_microwave_background', 'dark_matter', 'universe_evolution', 'cosmology']
            },
            'nuclear': {
                'frequency_range': (1e16, 1e20),  # 10PHz to 100EHz
                'wavelength_range': (3e-12, 3e-8),  # 3pm to 30nm
                'optimal_complexity': 0.95,
                'coordination_number': 240,
                'crv_frequency': 1.2356e20,
                'typical_nrci': 0.999,
                'computational_cost': 4.0,
                'best_for': ['nuclear_physics', 'particle_interactions', 'high_energy', 'zitterbewegung']
            },
            'optical': {
                'frequency_range': (1e14, 1e15),  # 100THz to 1PHz
                'wavelength_range': (3e-7, 3e-6),  # 300nm to 3μm
                'optimal_complexity': 0.85,
                'coordination_number': 6,
                'crv_frequency': 5e14,
                'typical_nrci': 0.999999,
                'computational_cost': 2.0,
                'best_for': ['photonics', 'optical_systems', 'laser_physics', 'light_matter_interaction']
            }
        }
        
        # Selection weights for different criteria
        self.selection_weights = {
            'frequency_match': 0.3,
            'complexity_match': 0.2,
            'scale_compatibility': 0.2,
            'expected_nrci': 0.15,
            'computational_efficiency': 0.1,
            'domain_expertise': 0.05
        }
        
        print("🎯 Automatic Realm Selector Initialized")
        print(f"   Available Realms: {len(self.realm_characteristics)}")
        print(f"   Selection Criteria: {len(self.selection_weights)}")
    
    def analyze_data_characteristics(self, data: np.ndarray, 
                                   data_type: str = 'unknown',
                                   sampling_rate: Optional[float] = None) -> DataCharacteristics:
        """
        Analyze input data to extract characteristics for realm selection.
        
        Args:
            data: Input data array
            data_type: Type of data ('time_series', 'frequency_domain', 'spatial', etc.)
            sampling_rate: Sampling rate for time series data (Hz)
            
        Returns:
            DataCharacteristics object with extracted features
        """
        # Ensure data is 1D for analysis
        if data.ndim > 1:
            data_flat = data.flatten()
        else:
            data_flat = data.copy()
        
        # Remove any NaN or infinite values
        data_clean = data_flat[np.isfinite(data_flat)]
        if len(data_clean) == 0:
            data_clean = np.array([0.0])
        
        # Basic characteristics
        size = len(data_clean)
        dimensionality = data.ndim
        
        # Statistical moments
        mean_val = np.mean(data_clean)
        std_val = np.std(data_clean)
        skew_val = skew(data_clean) if len(data_clean) > 2 else 0.0
        kurt_val = kurtosis(data_clean) if len(data_clean) > 3 else 0.0
        
        statistical_moments = {
            'mean': mean_val,
            'std': std_val,
            'skewness': skew_val,
            'kurtosis': kurt_val
        }
        
        # Complexity estimation (normalized standard deviation)
        complexity = min(std_val / (abs(mean_val) + 1e-10), 10.0)
        
        # Entropy calculation
        # Discretize data for entropy calculation
        if len(data_clean) > 1:
            hist, _ = np.histogram(data_clean, bins=min(50, len(data_clean)//2))
            hist = hist + 1e-10  # Avoid log(0)
            entropy_value = entropy(hist)
        else:
            entropy_value = 0.0
        
        # Spectral analysis
        spectral_features = {}
        dominant_frequency = 0.0
        frequency_spread = 0.0
        
        if len(data_clean) > 4 and data_type in ['time_series', 'unknown']:
            try:
                # FFT analysis
                fft_data = fft(data_clean)
                fft_magnitude = np.abs(fft_data)
                
                if sampling_rate is not None:
                    freqs = fftfreq(len(data_clean), 1/sampling_rate)
                else:
                    # Assume unit sampling rate
                    freqs = fftfreq(len(data_clean), 1.0)
                
                # Find dominant frequency
                positive_freqs = freqs[:len(freqs)//2]
                positive_magnitude = fft_magnitude[:len(fft_magnitude)//2]
                
                if len(positive_magnitude) > 0:
                    dominant_idx = np.argmax(positive_magnitude)
                    dominant_frequency = abs(positive_freqs[dominant_idx])
                    
                    # Frequency spread (spectral width)
                    power_spectrum = positive_magnitude**2
                    total_power = np.sum(power_spectrum)
                    if total_power > 0:
                        freq_mean = np.sum(positive_freqs * power_spectrum) / total_power
                        freq_var = np.sum(((positive_freqs - freq_mean)**2) * power_spectrum) / total_power
                        frequency_spread = np.sqrt(freq_var)
                
                spectral_features = {
                    'dominant_frequency': dominant_frequency,
                    'spectral_centroid': freq_mean if 'freq_mean' in locals() else 0.0,
                    'spectral_spread': frequency_spread,
                    'spectral_rolloff': np.percentile(positive_freqs, 85) if len(positive_freqs) > 0 else 0.0
                }
                
            except Exception as e:
                # Fallback if FFT fails
                spectral_features = {
                    'dominant_frequency': 0.0,
                    'spectral_centroid': 0.0,
                    'spectral_spread': 0.0,
                    'spectral_rolloff': 0.0
                }
        
        # Coherence estimation (autocorrelation-based)
        coherence_estimate = 0.0
        if len(data_clean) > 10:
            try:
                # Normalized autocorrelation at lag 1
                autocorr = np.corrcoef(data_clean[:-1], data_clean[1:])[0, 1]
                coherence_estimate = abs(autocorr) if not np.isnan(autocorr) else 0.0
            except:
                coherence_estimate = 0.0
        
        # Noise level estimation (high-frequency content)
        noise_level = 0.0
        if len(data_clean) > 3:
            # Estimate noise as the standard deviation of differences
            diff_data = np.diff(data_clean)
            noise_level = np.std(diff_data) / (np.std(data_clean) + 1e-10)
            noise_level = min(noise_level, 1.0)
        
        return DataCharacteristics(
            size=size,
            complexity=complexity,
            entropy_value=entropy_value,
            dominant_frequency=dominant_frequency,
            frequency_spread=frequency_spread,
            coherence_estimate=coherence_estimate,
            noise_level=noise_level,
            dimensionality=dimensionality,
            data_type=data_type,
            statistical_moments=statistical_moments,
            spectral_features=spectral_features
        )
    
    def calculate_realm_score(self, characteristics: DataCharacteristics, 
                            realm_name: str) -> RealmScore:
        """
        Calculate compatibility score for a specific realm.
        
        Args:
            characteristics: Data characteristics
            realm_name: Name of the realm to score
            
        Returns:
            RealmScore object with detailed scoring
        """
        realm_info = self.realm_characteristics[realm_name]
        reasoning = []
        
        # Frequency match score
        freq_range = realm_info['frequency_range']
        dominant_freq = characteristics.dominant_frequency
        
        if dominant_freq == 0:
            frequency_match = 0.5  # Neutral if no dominant frequency
            reasoning.append("No dominant frequency detected")
        elif freq_range[0] <= dominant_freq <= freq_range[1]:
            # Perfect match
            frequency_match = 1.0
            reasoning.append(f"Frequency {dominant_freq:.2e} Hz matches realm range")
        else:
            # Calculate distance from range
            if dominant_freq < freq_range[0]:
                distance = freq_range[0] / dominant_freq
            else:
                distance = dominant_freq / freq_range[1]
            
            frequency_match = 1.0 / (1.0 + np.log10(distance))
            reasoning.append(f"Frequency {dominant_freq:.2e} Hz outside optimal range")
        
        # Complexity match score
        optimal_complexity = realm_info['optimal_complexity']
        complexity_diff = abs(characteristics.complexity - optimal_complexity)
        complexity_match = np.exp(-complexity_diff * 2)  # Exponential decay
        
        if complexity_match > 0.8:
            reasoning.append(f"Complexity {characteristics.complexity:.3f} well-matched")
        else:
            reasoning.append(f"Complexity {characteristics.complexity:.3f} suboptimal")
        
        # Scale compatibility (based on data size and realm coordination)
        coord_number = realm_info['coordination_number']
        size_ratio = characteristics.size / (coord_number * 100)  # Arbitrary scaling
        scale_compatibility = 1.0 / (1.0 + abs(np.log10(max(size_ratio, 1e-10))))
        
        # Expected NRCI (higher is better)
        expected_nrci = realm_info['typical_nrci']
        
        # Computational efficiency (lower cost is better)
        computational_cost = realm_info['computational_cost']
        efficiency_score = 1.0 / computational_cost
        
        # Domain expertise bonus
        domain_bonus = 0.0
        best_for = realm_info['best_for']
        data_type = characteristics.data_type.lower()
        
        for domain in best_for:
            if domain in data_type or data_type in domain:
                domain_bonus = 1.0
                reasoning.append(f"Data type '{data_type}' matches domain expertise")
                break
        
        # Calculate weighted total score
        weights = self.selection_weights
        total_score = (
            weights['frequency_match'] * frequency_match +
            weights['complexity_match'] * complexity_match +
            weights['scale_compatibility'] * scale_compatibility +
            weights['expected_nrci'] * expected_nrci +
            weights['computational_efficiency'] * efficiency_score +
            weights['domain_expertise'] * domain_bonus
        )
        
        # Confidence based on how well-defined the characteristics are
        confidence = min(
            characteristics.coherence_estimate + 0.5,
            1.0 - characteristics.noise_level * 0.5,
            1.0
        )
        
        return RealmScore(
            realm_name=realm_name,
            score=total_score,
            confidence=confidence,
            reasoning=reasoning,
            expected_nrci=expected_nrci,
            computational_cost=computational_cost,
            frequency_match=frequency_match,
            scale_compatibility=scale_compatibility
        )
    
    def select_optimal_realm(self, data: np.ndarray,
                           data_type: str = 'unknown',
                           sampling_rate: Optional[float] = None,
                           multi_realm_threshold: float = 0.8) -> RealmSelectionResult:
        """
        Select the optimal realm(s) for given input data.
        
        Args:
            data: Input data array
            data_type: Type of data
            sampling_rate: Sampling rate for time series data
            multi_realm_threshold: Threshold for recommending multiple realms
            
        Returns:
            RealmSelectionResult with selection details
        """
        # Analyze data characteristics
        characteristics = self.analyze_data_characteristics(data, data_type, sampling_rate)
        
        # Calculate scores for all realms
        realm_scores = []
        for realm_name in self.realm_characteristics.keys():
            score = self.calculate_realm_score(characteristics, realm_name)
            realm_scores.append(score)
        
        # Sort by score (highest first)
        realm_scores.sort(key=lambda x: x.score, reverse=True)
        
        # Primary realm selection
        primary_realm = realm_scores[0].realm_name
        primary_score = realm_scores[0].score
        
        # Secondary realm selection
        secondary_realms = []
        multi_realm_recommended = False
        
        for score in realm_scores[1:]:
            if score.score >= multi_realm_threshold * primary_score:
                secondary_realms.append(score.realm_name)
                multi_realm_recommended = True
        
        # Overall selection confidence
        selection_confidence = realm_scores[0].confidence
        if len(realm_scores) > 1:
            score_gap = realm_scores[0].score - realm_scores[1].score
            selection_confidence *= (1.0 + score_gap)  # Higher gap = higher confidence
        
        selection_confidence = min(selection_confidence, 1.0)
        
        # Generate reasoning
        reasoning = [
            f"Primary realm '{primary_realm}' selected with score {primary_score:.3f}",
            f"Selection confidence: {selection_confidence:.3f}"
        ]
        
        if multi_realm_recommended:
            reasoning.append(f"Multi-realm computation recommended: {secondary_realms}")
        
        # Add top realm's reasoning
        reasoning.extend(realm_scores[0].reasoning[:3])  # Top 3 reasons
        
        return RealmSelectionResult(
            primary_realm=primary_realm,
            secondary_realms=secondary_realms,
            realm_scores=realm_scores,
            selection_confidence=selection_confidence,
            multi_realm_recommended=multi_realm_recommended,
            reasoning=reasoning,
            data_characteristics=characteristics
        )
    
    def get_realm_recommendation_summary(self, result: RealmSelectionResult) -> str:
        """
        Generate a human-readable summary of realm selection.
        
        Args:
            result: RealmSelectionResult object
            
        Returns:
            Formatted summary string
        """
        summary = []
        summary.append("🎯 AUTOMATIC REALM SELECTION SUMMARY")
        summary.append("=" * 50)
        
        # Primary recommendation
        summary.append(f"Primary Realm: {result.primary_realm.upper()}")
        summary.append(f"Confidence: {result.selection_confidence:.1%}")
        
        # Data characteristics
        chars = result.data_characteristics
        summary.append(f"\nData Characteristics:")
        summary.append(f"  Size: {chars.size:,} points")
        summary.append(f"  Complexity: {chars.complexity:.3f}")
        summary.append(f"  Dominant Frequency: {chars.dominant_frequency:.2e} Hz")
        summary.append(f"  Coherence: {chars.coherence_estimate:.3f}")
        
        # Top 3 realm scores
        summary.append(f"\nTop Realm Scores:")
        for i, score in enumerate(result.realm_scores[:3]):
            summary.append(f"  {i+1}. {score.realm_name}: {score.score:.3f}")
        
        # Multi-realm recommendation
        if result.multi_realm_recommended:
            summary.append(f"\nMulti-Realm Recommended:")
            for realm in result.secondary_realms:
                summary.append(f"  - {realm}")
        
        # Key reasoning
        summary.append(f"\nKey Reasoning:")
        for reason in result.reasoning[:3]:
            summary.append(f"  • {reason}")
        
        return "\n".join(summary)
    
    def validate_realm_selection(self, data: np.ndarray, 
                                selected_realm: str) -> Dict[str, Any]:
        """
        Validate a realm selection against data characteristics.
        
        Args:
            data: Input data array
            selected_realm: Name of selected realm
            
        Returns:
            Dictionary containing validation results
        """
        characteristics = self.analyze_data_characteristics(data)
        score = self.calculate_realm_score(characteristics, selected_realm)
        
        # Validation criteria
        validation_results = {
            'realm': selected_realm,
            'score': score.score,
            'frequency_match': score.frequency_match,
            'scale_compatibility': score.scale_compatibility,
            'expected_nrci': score.expected_nrci,
            'validation_passed': score.score > 0.5,
            'confidence': score.confidence,
            'reasoning': score.reasoning
        }
        
        # Performance prediction
        realm_info = self.realm_characteristics[selected_realm]
        predicted_performance = {
            'expected_nrci': score.expected_nrci,
            'computational_cost': realm_info['computational_cost'],
            'optimal_frequency': realm_info['crv_frequency'],
            'coordination_number': realm_info['coordination_number']
        }
        
        validation_results['predicted_performance'] = predicted_performance
        
        return validation_results


# Convenience function for quick realm selection
def select_realm_for_data(data: np.ndarray, 
                         data_type: str = 'unknown',
                         sampling_rate: Optional[float] = None) -> str:
    """
    Quick function to select optimal realm for given data.
    
    Args:
        data: Input data array
        data_type: Type of data
        sampling_rate: Sampling rate for time series data
        
    Returns:
        Name of selected realm
    """
    selector = AutomaticRealmSelector()
    result = selector.select_optimal_realm(data, data_type, sampling_rate)
    return result.primary_realm

