"""
UBP Framework v3.0 - Enhanced CRV Database with Sub-CRVs
Author: Euan Craig, New Zealand
Date: 13 August 2025

This module contains the refined Core Resonance Values (CRVs) with Sub-CRV fallback systems
based on frequency scanning research and harmonic pattern analysis.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import logging

@dataclass
class SubCRV:
    """Sub-CRV with performance metrics and harmonic relationship."""
    frequency: float
    nrci_score: float
    compute_time: float
    toggle_count: int
    harmonic_type: str  # e.g., "2x_harmonic", "0.5x_subharmonic", "fundamental"
    confidence: float
    
@dataclass
class CRVProfile:
    """Complete CRV profile with main CRV and Sub-CRV fallbacks."""
    realm: str
    main_crv: float
    wavelength: float  # nm
    geometry: str
    coordination_number: int
    sub_crvs: List[SubCRV]
    nrci_baseline: float
    optimization_notes: str

class EnhancedCRVDatabase:
    """
    Enhanced CRV Database with Sub-CRV fallback system and adaptive selection.
    
    Based on frequency scanning research showing harmonic patterns in each realm
    with specific Sub-CRVs that provide optimization pathways for different
    data characteristics and computational requirements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.crv_profiles = self._initialize_crv_profiles()
        self.performance_history = {}
        
    def _initialize_crv_profiles(self) -> Dict[str, CRVProfile]:
        """Initialize CRV profiles with research-based Sub-CRVs."""
        
        profiles = {}
        
        # ELECTROMAGNETIC REALM - Cube geometry, π-resonance
        profiles['electromagnetic'] = CRVProfile(
            realm='electromagnetic',
            main_crv=3.141593,  # π-resonance
            wavelength=635.0,
            geometry='cube',
            coordination_number=6,
            sub_crvs=[
                SubCRV(2.28e7, 0.998, 0.000018, 1175, "legacy_crv", 0.95),
                SubCRV(1.570796, 0.997, 0.000017, 1180, "0.5x_harmonic", 0.92),
                SubCRV(6.283185, 0.996, 0.000019, 1170, "2x_harmonic", 0.90),
                SubCRV(9.424778, 0.995, 0.000020, 1165, "3x_harmonic", 0.88)
            ],
            nrci_baseline=1.0,
            optimization_notes="π-resonance provides maximum electromagnetic coherence"
        )
        
        # QUANTUM REALM - Tetrahedron geometry, e/12-resonance
        profiles['quantum'] = CRVProfile(
            realm='quantum',
            main_crv=0.2265234857,  # e/12
            wavelength=655.0,
            geometry='tetrahedron',
            coordination_number=4,
            sub_crvs=[
                SubCRV(6.4444e13, 0.997, 0.000019, 1160, "legacy_crv", 0.94),
                SubCRV(0.1132617, 0.996, 0.000018, 1175, "0.5x_harmonic", 0.91),
                SubCRV(0.4530470, 0.995, 0.000020, 1155, "2x_harmonic", 0.89),
                SubCRV(0.6795704, 0.994, 0.000021, 1150, "3x_harmonic", 0.87)
            ],
            nrci_baseline=0.875,
            optimization_notes="e/12 resonance optimizes quantum coherence and entanglement"
        )
        
        # GRAVITATIONAL REALM - Octahedron geometry, research-validated Sub-CRVs
        profiles['gravitational'] = CRVProfile(
            realm='gravitational',
            main_crv=160.19,  # Research-validated main CRV
            wavelength=1000.0,
            geometry='octahedron',
            coordination_number=8,
            sub_crvs=[
                # Research results from frequency scanning
                SubCRV(11.266, 0.998999, 0.000017, 1179.73, "0.07x_subharmonic", 0.98),
                SubCRV(40.812, 0.998999, 0.000018, 1178.47, "0.25x_subharmonic", 0.97),
                SubCRV(176.09, 0.998999, 0.000018, 1172.83, "1.1x_harmonic", 0.96),
                SubCRV(0.43693, 0.998999, 0.000017, 1180.19, "fundamental_low", 0.95),
                SubCRV(0.11748, 0.998998, 0.000022, 1180.20, "fundamental_ultra_low", 0.94)
            ],
            nrci_baseline=0.915,
            optimization_notes="Multiple harmonic peaks provide gravitational wave optimization"
        )
        
        # BIOLOGICAL REALM - Dodecahedron geometry, 10 Hz base
        profiles['biological'] = CRVProfile(
            realm='biological',
            main_crv=10.0,
            wavelength=700.0,
            geometry='dodecahedron',
            coordination_number=20,
            sub_crvs=[
                SubCRV(49.931, 0.996, 0.000019, 1165, "legacy_crv", 0.93),
                SubCRV(5.0, 0.995, 0.000018, 1170, "0.5x_harmonic", 0.91),
                SubCRV(20.0, 0.994, 0.000020, 1160, "2x_harmonic", 0.89),
                SubCRV(40.0, 0.993, 0.000021, 1155, "4x_harmonic", 0.87),
                SubCRV(8.0, 0.992, 0.000019, 1168, "alpha_wave", 0.90)  # EEG alpha
            ],
            nrci_baseline=0.911,
            optimization_notes="Biological rhythms and EEG frequency optimization"
        )
        
        # COSMOLOGICAL REALM - Icosahedron geometry, π^φ-resonance
        profiles['cosmological'] = CRVProfile(
            realm='cosmological',
            main_crv=0.832037,  # π^φ
            wavelength=800.0,
            geometry='icosahedron',
            coordination_number=12,
            sub_crvs=[
                SubCRV(1.1128e-18, 0.994, 0.000022, 1145, "legacy_crv", 0.92),
                SubCRV(0.416018, 0.993, 0.000021, 1150, "0.5x_harmonic", 0.89),
                SubCRV(1.664074, 0.992, 0.000023, 1140, "2x_harmonic", 0.87),
                SubCRV(2.496111, 0.991, 0.000024, 1135, "3x_harmonic", 0.85)
            ],
            nrci_baseline=0.797,
            optimization_notes="π^φ resonance for cosmological scale phenomena"
        )
        
        # NUCLEAR REALM - E8-to-G2 lattice, Zitterbewegung frequency
        profiles['nuclear'] = CRVProfile(
            realm='nuclear',
            main_crv=1.2356e20,  # Zitterbewegung frequency
            wavelength=2.4e-12,  # Compton wavelength
            geometry='e8_g2',
            coordination_number=248,  # E8 dimension
            sub_crvs=[
                SubCRV(1.6249e16, 0.996, 0.000020, 1160, "legacy_crv", 0.94),
                SubCRV(6.178e19, 0.995, 0.000021, 1155, "0.5x_harmonic", 0.92),
                SubCRV(2.4712e20, 0.994, 0.000022, 1150, "2x_harmonic", 0.90),
                SubCRV(3.7068e20, 0.993, 0.000023, 1145, "3x_harmonic", 0.88)
            ],
            nrci_baseline=0.950,
            optimization_notes="Zitterbewegung frequency for nuclear physics precision"
        )
        
        # OPTICAL REALM - Hexagonal photonic crystal, 600 nm
        profiles['optical'] = CRVProfile(
            realm='optical',
            main_crv=5.0e14,  # 600 nm frequency
            wavelength=600.0,
            geometry='hexagonal',
            coordination_number=6,
            sub_crvs=[
                SubCRV(1.4398e14, 0.999999, 0.000015, 1190, "legacy_crv", 0.98),
                SubCRV(2.5e14, 0.999998, 0.000016, 1185, "0.5x_harmonic", 0.96),
                SubCRV(1.0e15, 0.999997, 0.000017, 1180, "2x_harmonic", 0.94),
                SubCRV(1.5e15, 0.999996, 0.000018, 1175, "3x_harmonic", 0.92)
            ],
            nrci_baseline=0.999999,
            optimization_notes="Photonic crystal optimization for maximum optical coherence"
        )
        
        return profiles
    
    def get_crv_profile(self, realm: str) -> Optional[CRVProfile]:
        """Get complete CRV profile for a realm."""
        return self.crv_profiles.get(realm.lower())
    
    def get_optimal_crv(self, realm: str, data_characteristics: Dict) -> Tuple[float, str]:
        """
        Select optimal CRV based on data characteristics.
        
        Args:
            realm: Target realm name
            data_characteristics: Dict with keys like 'frequency', 'complexity', 'noise_level'
            
        Returns:
            Tuple of (optimal_crv_frequency, selection_reason)
        """
        profile = self.get_crv_profile(realm)
        if not profile:
            return None, f"Unknown realm: {realm}"
        
        # Extract data characteristics
        data_freq = data_characteristics.get('frequency', 0)
        complexity = data_characteristics.get('complexity', 0.5)
        noise_level = data_characteristics.get('noise_level', 0.1)
        target_nrci = data_characteristics.get('target_nrci', 0.95)
        
        # Start with main CRV
        best_crv = profile.main_crv
        best_score = 0.0
        best_reason = "main_crv_default"
        
        # Evaluate main CRV
        main_score = self._evaluate_crv_fitness(profile.main_crv, data_characteristics, profile)
        if main_score > best_score:
            best_crv = profile.main_crv
            best_score = main_score
            best_reason = "main_crv_optimal"
        
        # Evaluate Sub-CRVs
        for sub_crv in profile.sub_crvs:
            score = self._evaluate_crv_fitness(sub_crv.frequency, data_characteristics, profile, sub_crv)
            
            # Bonus for high NRCI Sub-CRVs
            if sub_crv.nrci_score >= target_nrci:
                score += 0.1
            
            # Bonus for low compute time if speed is priority
            if data_characteristics.get('speed_priority', False) and sub_crv.compute_time < 0.00002:
                score += 0.05
            
            if score > best_score:
                best_crv = sub_crv.frequency
                best_score = score
                best_reason = f"sub_crv_{sub_crv.harmonic_type}"
        
        # Log selection
        self.logger.info(f"Selected CRV {best_crv:.6e} for {realm} (reason: {best_reason}, score: {best_score:.3f})")
        
        return best_crv, best_reason
    
    def _evaluate_crv_fitness(self, crv_freq: float, data_chars: Dict, profile: CRVProfile, sub_crv: Optional[SubCRV] = None) -> float:
        """Evaluate how well a CRV matches the data characteristics."""
        score = 0.0
        
        # Frequency matching (30% weight)
        data_freq = data_chars.get('frequency', 0)
        if data_freq > 0:
            freq_ratio = min(crv_freq, data_freq) / max(crv_freq, data_freq)
            score += 0.3 * freq_ratio
        else:
            score += 0.15  # Neutral score if no frequency info
        
        # Complexity matching (20% weight)
        complexity = data_chars.get('complexity', 0.5)
        if sub_crv:
            # Higher NRCI Sub-CRVs better for complex data
            complexity_match = min(1.0, sub_crv.nrci_score + complexity * 0.1)
            score += 0.2 * complexity_match
        else:
            score += 0.2 * profile.nrci_baseline
        
        # Noise tolerance (15% weight)
        noise_level = data_chars.get('noise_level', 0.1)
        if sub_crv:
            # Sub-CRVs with higher confidence better for noisy data
            noise_tolerance = sub_crv.confidence * (1.0 - noise_level)
            score += 0.15 * noise_tolerance
        else:
            score += 0.15 * (1.0 - noise_level)
        
        # Performance considerations (35% weight)
        if sub_crv:
            # Balance NRCI and compute time
            perf_score = (sub_crv.nrci_score * 0.7) + ((1.0 - min(1.0, sub_crv.compute_time * 50000)) * 0.3)
            score += 0.35 * perf_score
        else:
            score += 0.35 * profile.nrci_baseline
        
        return score
    
    def get_harmonic_crvs(self, realm: str, base_frequency: float, max_harmonics: int = 5) -> List[float]:
        """Generate harmonic CRVs based on a base frequency."""
        harmonics = []
        
        # Fundamental and subharmonics
        for i in range(1, max_harmonics + 1):
            harmonics.append(base_frequency / i)  # Subharmonics
            if i > 1:
                harmonics.append(base_frequency * i)  # Harmonics
        
        return sorted(harmonics)
    


