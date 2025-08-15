"""
UBP Framework v3.0 - Enhanced CRV System with Sub-CRVs
Author: Euan Craig, New Zealand
Date: 13 August 2025

Enhanced CRV system with adaptive selection, Sub-CRV fallbacks, and harmonic pattern recognition
based on frequency scanning research showing clear optimization pathways.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
import time
from scipy.signal import find_peaks
from scipy.optimize import minimize_scalar

# Import configuration system
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from crv_database import EnhancedCRVDatabase, CRVProfile, SubCRV
from ubp_config import get_config

@dataclass
class CRVSelectionResult:
    """Result from CRV selection process."""
    selected_crv: float
    selection_reason: str
    confidence: float
    fallback_crvs: List[float]
    performance_prediction: Dict[str, float]
    harmonic_analysis: Optional[Dict] = None

@dataclass
class CRVPerformanceMetrics:
    """Performance metrics for CRV evaluation."""
    nrci_score: float
    computation_time: float
    energy_efficiency: float
    coherence_stability: float
    error_rate: float
    throughput: float

class HarmonicPatternAnalyzer:
    """
    Analyzes harmonic patterns in data to optimize CRV selection.
    
    Based on research showing clear harmonic relationships (0.5x, 2x harmonics)
    that provide optimization pathways for different computational requirements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def analyze_frequency_spectrum(self, data: np.ndarray, sample_rate: float = 1.0) -> Dict:
        """
        Analyze frequency spectrum to identify dominant frequencies and harmonics.
        
        Args:
            data: Input data array
            sample_rate: Sampling rate for frequency analysis
            
        Returns:
            Dictionary with frequency analysis results
        """
        # Compute FFT
        fft = np.fft.fft(data)
        freqs = np.fft.fftfreq(len(data), 1/sample_rate)
        
        # Get magnitude spectrum (positive frequencies only)
        magnitude = np.abs(fft[:len(fft)//2])
        pos_freqs = freqs[:len(freqs)//2]
        
        # Find peaks in spectrum
        peaks, properties = find_peaks(magnitude, height=np.max(magnitude)*0.1)
        
        if len(peaks) == 0:
            return {
                'dominant_frequency': 0.0,
                'harmonics': [],
                'harmonic_ratios': [],
                'spectral_centroid': 0.0,
                'bandwidth': 0.0
            }
        
        # Dominant frequency
        dominant_idx = peaks[np.argmax(magnitude[peaks])]
        dominant_freq = pos_freqs[dominant_idx]
        
        # Find harmonics (integer multiples of dominant frequency)
        harmonics = []
        harmonic_ratios = []
        
        for peak_idx in peaks:
            peak_freq = pos_freqs[peak_idx]
            if peak_freq > 0 and dominant_freq > 0:
                ratio = peak_freq / dominant_freq
                if abs(ratio - round(ratio)) < 0.1:  # Close to integer ratio
                    harmonics.append(peak_freq)
                    harmonic_ratios.append(ratio)
        
        # Spectral centroid (center of mass of spectrum)
        spectral_centroid = np.sum(pos_freqs * magnitude) / np.sum(magnitude) if np.sum(magnitude) > 0 else 0.0
        
        # Bandwidth (spread of spectrum)
        bandwidth = np.sqrt(np.sum(((pos_freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude)) if np.sum(magnitude) > 0 else 0.0
        
        return {
            'dominant_frequency': dominant_freq,
            'harmonics': harmonics,
            'harmonic_ratios': harmonic_ratios,
            'spectral_centroid': spectral_centroid,
            'bandwidth': bandwidth,
            'peak_frequencies': pos_freqs[peaks].tolist(),
            'peak_magnitudes': magnitude[peaks].tolist()
        }
    
    def find_optimal_harmonic_crv(self, base_frequency: float, available_crvs: List[float]) -> Tuple[float, str]:
        """
        Find the CRV that best matches harmonic relationships with the base frequency.
        
        Args:
            base_frequency: Base frequency from data analysis
            available_crvs: List of available CRV frequencies
            
        Returns:
            Tuple of (optimal_crv, harmonic_relationship)
        """
        if base_frequency <= 0 or not available_crvs:
            return available_crvs[0] if available_crvs else 1.0, "default"
        
        best_crv = available_crvs[0]
        best_score = 0.0
        best_relationship = "default"
        
        for crv in available_crvs:
            # Check various harmonic relationships
            relationships = [
                (crv / base_frequency, "fundamental"),
                (base_frequency / crv, "inverse_fundamental"),
                (crv / (2 * base_frequency), "half_harmonic"),
                ((2 * crv) / base_frequency, "double_harmonic"),
                (crv / (3 * base_frequency), "third_harmonic"),
                ((3 * crv) / base_frequency, "triple_harmonic")
            ]
            
            for ratio, relationship in relationships:
                # Score based on how close to integer or simple fraction
                if ratio > 0:
                    # Check for integer ratios
                    int_score = 1.0 / (1.0 + abs(ratio - round(ratio)))
                    
                    # Check for simple fraction ratios (1/2, 1/3, 2/3, etc.)
                    frac_scores = []
                    for num in range(1, 5):
                        for den in range(1, 5):
                            frac_ratio = num / den
                            frac_scores.append(1.0 / (1.0 + abs(ratio - frac_ratio)))
                    
                    score = max(int_score, max(frac_scores))
                    
                    if score > best_score:
                        best_score = score
                        best_crv = crv
                        best_relationship = f"{relationship}_ratio_{ratio:.3f}"
        
        return best_crv, best_relationship

class AdaptiveCRVSelector:
    """
    Adaptive CRV selection system that chooses optimal CRVs based on data characteristics,
    performance requirements, and harmonic analysis.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_config()
        self.crv_database = EnhancedCRVDatabase()
        self.harmonic_analyzer = HarmonicPatternAnalyzer()
        
        # Performance history for learning
        self.performance_history = {}
        self.selection_history = []
        
        # CRV performance monitoring
        self.crv_metrics = {}
        
    def analyze_data_characteristics(self, data: np.ndarray, metadata: Optional[Dict] = None) -> Dict:
        """
        Analyze input data to extract characteristics for CRV selection.
        
        Args:
            data: Input data array
            metadata: Optional metadata about the data
            
        Returns:
            Dictionary of data characteristics
        """
        if len(data) == 0:
            return {'frequency': 0, 'complexity': 0.5, 'noise_level': 0.1}
        
        # Basic statistics
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Frequency analysis
        harmonic_analysis = self.harmonic_analyzer.analyze_frequency_spectrum(data)
        dominant_freq = harmonic_analysis['dominant_frequency']
        
        # Complexity measures
        # Entropy-based complexity
        hist, _ = np.histogram(data, bins=50)
        hist = hist / np.sum(hist)  # Normalize
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        complexity = entropy / np.log(50)  # Normalize to [0,1]
        
        # Noise level estimation (based on high-frequency content)
        if len(data) > 10:
            diff_data = np.diff(data)
            noise_level = np.std(diff_data) / (np.std(data) + 1e-10)
            noise_level = min(1.0, noise_level)  # Clamp to [0,1]
        else:
            noise_level = 0.1
        
        # Data scale
        data_range = np.max(data) - np.min(data)
        data_scale = np.log10(data_range + 1e-10)
        
        # Temporal characteristics (if applicable)
        autocorr = np.corrcoef(data[:-1], data[1:])[0, 1] if len(data) > 1 else 0.0
        
        characteristics = {
            'frequency': dominant_freq,
            'complexity': complexity,
            'noise_level': noise_level,
            'data_scale': data_scale,
            'autocorrelation': autocorr,
            'mean': mean_val,
            'std': std_val,
            'range': data_range,
            'harmonic_analysis': harmonic_analysis,
            'sample_size': len(data)
        }
        
        # Add metadata if provided
        if metadata:
            characteristics.update(metadata)
        
        return characteristics
    
    def select_optimal_crv(self, realm: str, data: np.ndarray, 
                          requirements: Optional[Dict] = None) -> CRVSelectionResult:
        """
        Select optimal CRV for given realm and data characteristics.
        
        Args:
            realm: Target realm name
            data: Input data for analysis
            requirements: Optional requirements dict (speed_priority, accuracy_priority, etc.)
            
        Returns:
            CRVSelectionResult with selected CRV and analysis
        """
        # Analyze data characteristics
        data_chars = self.analyze_data_characteristics(data, requirements)
        
        # Get realm profile
        realm_profile = self.crv_database.get_crv_profile(realm)
        if not realm_profile:
            self.logger.error(f"Unknown realm: {realm}")
            return CRVSelectionResult(
                selected_crv=self.config.crv.electromagnetic,
                selection_reason="unknown_realm_fallback",
                confidence=0.1,
                fallback_crvs=[],
                performance_prediction={}
            )
        
        # Get optimal CRV from database
        optimal_crv, reason = self.crv_database.get_optimal_crv(realm, data_chars)
        
        # Harmonic analysis for additional optimization
        all_crvs = [realm_profile.main_crv] + [sub.frequency for sub in realm_profile.sub_crvs]
        harmonic_crv, harmonic_reason = self.harmonic_analyzer.find_optimal_harmonic_crv(
            data_chars['frequency'], all_crvs
        )
        
        # Combine database selection with harmonic analysis
        if data_chars['frequency'] > 0:
            # Weight harmonic analysis more heavily if we have frequency information
            if harmonic_reason != "default":
                optimal_crv = harmonic_crv
                reason = f"harmonic_{harmonic_reason}"
        
        # Generate fallback CRVs
        fallback_crvs = self._generate_fallback_crvs(realm_profile, data_chars)
        
        # Predict performance
        performance_prediction = self._predict_performance(optimal_crv, data_chars, realm_profile)
        
        # Calculate confidence
        confidence = self._calculate_selection_confidence(optimal_crv, data_chars, realm_profile)
        
        result = CRVSelectionResult(
            selected_crv=optimal_crv,
            selection_reason=reason,
            confidence=confidence,
            fallback_crvs=fallback_crvs,
            performance_prediction=performance_prediction,
            harmonic_analysis=data_chars.get('harmonic_analysis')
        )
        
        # Log selection
        self.logger.info(f"CRV selected for {realm}: {optimal_crv:.6e} Hz "
                        f"(reason: {reason}, confidence: {confidence:.3f})")
        
        # Store selection history
        self.selection_history.append({
            'realm': realm,
            'crv': optimal_crv,
            'reason': reason,
            'confidence': confidence,
            'timestamp': time.time(),
            'data_characteristics': data_chars
        })
        
        return result
    
    def _generate_fallback_crvs(self, realm_profile: CRVProfile, data_chars: Dict) -> List[float]:
        """Generate ordered list of fallback CRVs."""
        fallbacks = []
        
        # Add Sub-CRVs sorted by performance prediction
        sub_crv_scores = []
        for sub_crv in realm_profile.sub_crvs:
            score = self._score_crv_fitness(sub_crv.frequency, data_chars, realm_profile, sub_crv)
            sub_crv_scores.append((sub_crv.frequency, score))
        
        # Sort by score (descending)
        sub_crv_scores.sort(key=lambda x: x[1], reverse=True)
        fallbacks.extend([crv for crv, _ in sub_crv_scores])
        
        # Add main CRV if not already included
        if realm_profile.main_crv not in fallbacks:
            fallbacks.insert(0, realm_profile.main_crv)
        
        return fallbacks[:5]  # Limit to top 5 fallbacks
    
    def _predict_performance(self, crv: float, data_chars: Dict, realm_profile: CRVProfile) -> Dict[str, float]:
        """Predict performance metrics for a given CRV."""
        # Base predictions on realm profile and data characteristics
        base_nrci = realm_profile.nrci_baseline
        
        # Adjust based on data characteristics
        complexity_factor = 1.0 - (data_chars['complexity'] * 0.1)
        noise_factor = 1.0 - (data_chars['noise_level'] * 0.2)
        
        predicted_nrci = base_nrci * complexity_factor * noise_factor
        predicted_nrci = max(0.0, min(1.0, predicted_nrci))
        
        # Predict computation time (inverse relationship with CRV magnitude)
        base_time = 0.00002  # 20 microseconds base
        crv_factor = np.log10(crv + 1) / 10.0
        predicted_time = base_time * (1.0 + crv_factor)
        
        # Predict energy efficiency
        predicted_energy = 0.8 + (predicted_nrci * 0.2)
        
        return {
            'predicted_nrci': predicted_nrci,
            'predicted_computation_time': predicted_time,
            'predicted_energy_efficiency': predicted_energy,
            'predicted_throughput': 1.0 / predicted_time
        }
    
    def _calculate_selection_confidence(self, crv: float, data_chars: Dict, realm_profile: CRVProfile) -> float:
        """Calculate confidence in CRV selection."""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have frequency information
        if data_chars['frequency'] > 0:
            confidence += 0.2
        
        # Increase confidence for low noise data
        confidence += (1.0 - data_chars['noise_level']) * 0.2
        
        # Increase confidence if CRV matches a Sub-CRV (validated)
        for sub_crv in realm_profile.sub_crvs:
            if abs(crv - sub_crv.frequency) / max(crv, sub_crv.frequency) < 0.01:
                confidence += sub_crv.confidence * 0.3
                break
        
        # Increase confidence based on historical performance
        if crv in self.crv_metrics:
            historical_performance = self.crv_metrics[crv].get('avg_nrci', 0.5)
            confidence += historical_performance * 0.2
        
        return min(1.0, confidence)
    
    def _score_crv_fitness(self, crv_freq: float, data_chars: Dict, 
                          profile: CRVProfile, sub_crv: Optional[SubCRV] = None) -> float:
        """Score how well a CRV fits the data characteristics."""
        score = 0.0
        
        # Frequency matching (30% weight)
        data_freq = data_chars.get('frequency', 0)
        if data_freq > 0:
            freq_ratio = min(crv_freq, data_freq) / max(crv_freq, data_freq)
            score += 0.3 * freq_ratio
        else:
            score += 0.15  # Neutral score
        
        # Complexity matching (20% weight)
        complexity = data_chars.get('complexity', 0.5)
        if sub_crv:
            complexity_match = min(1.0, sub_crv.nrci_score + complexity * 0.1)
            score += 0.2 * complexity_match
        else:
            score += 0.2 * profile.nrci_baseline
        
        # Noise tolerance (15% weight)
        noise_level = data_chars.get('noise_level', 0.1)
        if sub_crv:
            noise_tolerance = sub_crv.confidence * (1.0 - noise_level)
            score += 0.15 * noise_tolerance
        else:
            score += 0.15 * (1.0 - noise_level)
        
        # Performance considerations (35% weight)
        if sub_crv:
            perf_score = (sub_crv.nrci_score * 0.7) + ((1.0 - min(1.0, sub_crv.compute_time * 50000)) * 0.3)
            score += 0.35 * perf_score
        else:
            score += 0.35 * profile.nrci_baseline
        
        return score
    
    def update_performance_metrics(self, crv: float, metrics: CRVPerformanceMetrics):
        """Update performance metrics for a CRV based on actual usage."""
        if crv not in self.crv_metrics:
            self.crv_metrics[crv] = {
                'usage_count': 0,
                'total_nrci': 0.0,
                'total_time': 0.0,
                'total_energy': 0.0,
                'error_count': 0
            }
        
        # Update running averages
        self.crv_metrics[crv]['usage_count'] += 1
        self.crv_metrics[crv]['total_nrci'] += metrics.nrci_score
        self.crv_metrics[crv]['total_time'] += metrics.computation_time
        self.crv_metrics[crv]['total_energy'] += metrics.energy_efficiency
        
        if metrics.error_rate > 0.1:  # Threshold for significant errors
            self.crv_metrics[crv]['error_count'] += 1
        
        # Calculate averages
        count = self.crv_metrics[crv]['usage_count']
        self.crv_metrics[crv]['avg_nrci'] = self.crv_metrics[crv]['total_nrci'] / count
        self.crv_metrics[crv]['avg_time'] = self.crv_metrics[crv]['total_time'] / count
        self.crv_metrics[crv]['avg_energy'] = self.crv_metrics[crv]['total_energy'] / count
        self.crv_metrics[crv]['error_rate'] = self.crv_metrics[crv]['error_count'] / count
        
        self.logger.debug(f"Updated metrics for CRV {crv:.6e}: "
                         f"NRCI={self.crv_metrics[crv]['avg_nrci']:.6f}, "
                         f"Time={self.crv_metrics[crv]['avg_time']:.6f}s")
    
    def get_performance_summary(self) -> Dict:
        """Get summary of CRV performance across all realms."""
        summary = {
            'total_selections': len(self.selection_history),
            'unique_crvs_used': len(self.crv_metrics),
            'average_confidence': 0.0,
            'top_performing_crvs': [],
            'realm_usage': {}
        }
        
        if self.selection_history:
            # Average confidence
            summary['average_confidence'] = np.mean([s['confidence'] for s in self.selection_history])
            
            # Realm usage statistics
            for selection in self.selection_history:
                realm = selection['realm']
                if realm not in summary['realm_usage']:
                    summary['realm_usage'][realm] = 0
                summary['realm_usage'][realm] += 1
        
        # Top performing CRVs
        if self.crv_metrics:
            crv_performance = [(crv, metrics['avg_nrci']) for crv, metrics in self.crv_metrics.items()]
            crv_performance.sort(key=lambda x: x[1], reverse=True)
            summary['top_performing_crvs'] = crv_performance[:5]
        
        return summary
    
    def optimize_crv_for_target(self, realm: str, data: np.ndarray, 
                               target_nrci: float = 0.999999) -> float:
        """
        Optimize CRV to achieve target NRCI for specific data.
        
        Uses optimization techniques to fine-tune CRV beyond the database values.
        """
        realm_profile = self.crv_database.get_crv_profile(realm)
        if not realm_profile:
            return self.config.crv.electromagnetic
        
        # Get initial CRV selection
        selection_result = self.select_optimal_crv(realm, data)
        initial_crv = selection_result.selected_crv
        
        # Define optimization bounds around selected CRV
        bounds = (initial_crv * 0.1, initial_crv * 10.0)
        
        def objective(crv):
            """Objective function for CRV optimization."""
            # Simulate NRCI calculation (simplified)
            data_chars = self.analyze_data_characteristics(data)
            predicted_perf = self._predict_performance(crv, data_chars, realm_profile)
            predicted_nrci = predicted_perf['predicted_nrci']
            
            # Return squared error from target
            return (predicted_nrci - target_nrci) ** 2
        
        # Optimize
        try:
            result = minimize_scalar(objective, bounds=bounds, method='bounded')
            if result.success:
                optimized_crv = result.x
                self.logger.info(f"CRV optimized for {realm}: {optimized_crv:.6e} Hz "
                               f"(target NRCI: {target_nrci})")
                return optimized_crv
            else:
                self.logger.warning(f"CRV optimization failed for {realm}: {result.message}")
                return initial_crv
        except Exception as e:
            self.logger.error(f"CRV optimization error: {e}")
            return initial_crv
    
    def get_realm_crvs(self, realm: str) -> Dict[str, Any]:
        """
        Get CRV information for a specific realm.
        
        Args:
            realm: Name of the realm
            
        Returns:
            Dictionary containing CRV information for the realm
        """
        try:
            realm_profile = self.crv_database.get_realm_profile(realm)
            if not realm_profile:
                return {}
            
            return {
                'main_crv': realm_profile.main_crv,
                'sub_crvs': [sub_crv.frequency for sub_crv in realm_profile.sub_crvs],
                'cross_realm_frequencies': realm_profile.cross_realm_frequencies,
                'optimization_target': realm_profile.optimization_target,
                'harmonic_ratios': realm_profile.harmonic_ratios
            }
        except Exception as e:
            self.logger.error(f"Failed to get realm CRVs for {realm}: {e}")
            return {}

class CRVPerformanceMonitor:
    """
    Real-time CRV performance monitoring system.
    
    Tracks CRV performance across different realms and data types,
    providing feedback for continuous optimization.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.performance_data = {}
        self.monitoring_enabled = True
        
    def start_monitoring(self, crv: float, realm: str, data_size: int):
        """Start monitoring a CRV computation."""
        if not self.monitoring_enabled:
            return None
        
        monitor_id = f"{realm}_{crv:.6e}_{int(time.time())}"
        self.performance_data[monitor_id] = {
            'crv': crv,
            'realm': realm,
            'data_size': data_size,
            'start_time': time.time(),
            'end_time': None,
            'nrci_samples': [],
            'energy_samples': [],
            'error_count': 0
        }
        
        return monitor_id
    
    def update_monitoring(self, monitor_id: str, nrci: float, energy: float, error_occurred: bool = False):
        """Update monitoring data during computation."""
        if monitor_id not in self.performance_data:
            return
        
        data = self.performance_data[monitor_id]
        data['nrci_samples'].append(nrci)
        data['energy_samples'].append(energy)
        
        if error_occurred:
            data['error_count'] += 1
    
    def end_monitoring(self, monitor_id: str) -> Optional[CRVPerformanceMetrics]:
        """End monitoring and return performance metrics."""
        if monitor_id not in self.performance_data:
            return None
        
        data = self.performance_data[monitor_id]
        data['end_time'] = time.time()
        
        # Calculate metrics
        computation_time = data['end_time'] - data['start_time']
        avg_nrci = np.mean(data['nrci_samples']) if data['nrci_samples'] else 0.0
        avg_energy = np.mean(data['energy_samples']) if data['energy_samples'] else 0.0
        error_rate = data['error_count'] / max(1, len(data['nrci_samples']))
        
        # Coherence stability (standard deviation of NRCI)
        coherence_stability = 1.0 - np.std(data['nrci_samples']) if len(data['nrci_samples']) > 1 else 1.0
        
        # Throughput (operations per second)
        throughput = len(data['nrci_samples']) / computation_time if computation_time > 0 else 0.0
        
        metrics = CRVPerformanceMetrics(
            nrci_score=avg_nrci,
            computation_time=computation_time,
            energy_efficiency=avg_energy,
            coherence_stability=coherence_stability,
            error_rate=error_rate,
            throughput=throughput
        )
        
        # Clean up
        del self.performance_data[monitor_id]
        
        return metrics

