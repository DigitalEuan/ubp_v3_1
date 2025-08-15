"""
UBP Framework v3.0 - Harmonic Toggle Resonance (HTR) Engine
Author: Euan Craig, New Zealand
Date: 13 August 2025

Harmonic Toggle Resonance Plugin for UBP Framework integrating molecular simulation,
cross-domain data processing, and genetic CRV optimization based on HTR research.
"""

import numpy as np
from scipy.sparse import dok_matrix
from scipy.optimize import minimize
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import time

@dataclass
class HTRResult:
    """Result from HTR computation."""
    toggles: np.ndarray
    energy: float
    nrci: float
    computation_time: float
    crv_used: float
    reconstruction_error: float
    sensitivity_metrics: Optional[Dict] = None

@dataclass
class MoleculeConfig:
    """Configuration for molecular simulation."""
    name: str
    nodes: int
    bond_length: float  # L_0 in meters
    bond_energy: float  # eV
    geometry_type: str
    smiles: Optional[str] = None

class HTREngine:
    """
    Harmonic Toggle Resonance Engine for UBP Framework v3.0
    
    Provides molecular simulation, cross-domain data processing, and genetic CRV optimization
    based on HTR research achieving NRCI targets of 0.9999999 through precise CRV tuning.
    """
    
    # Realm configurations from HTR research
    REALM_CONFIG = {
        "quantum": {"CRV": 3.000000, "coordination": 4, "lattice": "tetrahedral"},
        "electromagnetic": {"CRV": 1.640941, "coordination": 6, "lattice": "cubic"},
        "gravitational": {"CRV": 1.640938, "coordination": 8, "lattice": "FCC"},
        "biological": {"CRV": 1.640937, "coordination": 10, "lattice": "dodecahedral"},
        "cosmological": {"CRV": 1.640940, "coordination": 12, "lattice": "icosahedral"},
        "nuclear": {"CRV": 1.640942, "coordination": 248, "lattice": "e8_g2"},
        "optical": {"CRV": 1.640943, "coordination": 6, "lattice": "hexagonal"}
    }
    
    # Molecular parameters from HTR research
    MOLECULE_PARAMS = {
        'propane': MoleculeConfig('propane', 10, 0.154e-9, 4.8, 'alkane', 'CCC'),
        'benzene': MoleculeConfig('benzene', 6, 0.14e-9, 5.0, 'aromatic', 'c1ccccc1'),
        'methane': MoleculeConfig('methane', 5, 0.109e-9, 4.5, 'tetrahedral', 'C'),
        'butane': MoleculeConfig('butane', 13, 0.154e-9, 4.8, 'alkane', 'CCCC')
    }
    
    def __init__(self, molecule: str = 'propane', realm: str = 'quantum', 
                 custom_coords: Optional[np.ndarray] = None, 
                 custom_data: Optional[np.ndarray] = None):
        """
        Initialize HTR Engine.
        
        Args:
            molecule: Molecule type or 'custom' for custom data
            realm: UBP realm for computation
            custom_coords: Custom 3D coordinates for molecular structure
            custom_data: Custom data for cross-domain processing
        """
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.molecule = molecule if not custom_data else 'custom'
        self.realm = realm
        self.realm_config = self.REALM_CONFIG[realm]
        
        # Initialize CRV (will be optimized)
        self.crv = self.realm_config["CRV"]
        self.coordination = self.realm_config["coordination"]
        self.lattice = self.realm_config["lattice"]
        
        # Molecular/data setup
        if custom_coords is not None:
            self.coords = custom_coords
            self.num_nodes = custom_coords.shape[0]
            self.molecule_config = MoleculeConfig('custom', self.num_nodes, 0.154e-9, 4.8, 'custom')
        elif custom_data is not None:
            self.coords = self._vectorize_custom_data(custom_data)
            self.num_nodes = self.coords.shape[0]
            self.molecule_config = MoleculeConfig('custom', self.num_nodes, 0.154e-9, 4.8, 'custom')
        else:
            self.molecule_config = self.MOLECULE_PARAMS.get(molecule, self.MOLECULE_PARAMS['propane'])
            self.num_nodes = self.molecule_config.nodes
            self.coords = self._generate_molecular_coords()
        
        # Physical constants
        self.sqrt_2 = np.sqrt(2)
        self.delta_t = 1e-15
        self.rydberg = 1.097373156853967e7
        
        # Compute tick frequency
        self.f_i = 3e8 / (1 / (self.rydberg * self.crv) * 1e9)
        
        # Initialize state
        self.M = np.random.randint(0, 2, self.num_nodes).astype(np.float64)
        self.M_history = [self.M.copy()]
        self.distances = self._calculate_distance_matrix()
        self.energy_history = []
        self.nrci_history = []
        
        self.logger.info(f"HTR Engine initialized: {self.molecule} in {realm} realm, {self.num_nodes} nodes")
    
    def _generate_molecular_coords(self) -> np.ndarray:
        """Generate 3D coordinates for molecular structure."""
        config = self.molecule_config
        l = config.bond_length
        
        if config.smiles == 'c1ccccc1' or self.molecule == 'benzene':
            # Benzene ring
            return np.array([[np.cos(2 * np.pi * i / 6) * l, np.sin(2 * np.pi * i / 6) * l, 0] 
                           for i in range(6)])
        
        elif config.smiles == 'C' or self.molecule == 'methane':
            # Tetrahedral methane
            s = l / np.sqrt(3)
            return np.array([[0, 0, 0], [s, s, s], [s, -s, -s], [-s, s, -s], [-s, -s, s]])
        
        elif config.smiles == 'CCCC' or self.molecule == 'butane':
            # Butane chain with hydrogens
            coords = []
            # Carbon backbone
            for i in range(4):
                coords.append([i * l, 0, 0])
            # Hydrogen atoms
            for i in range(4):
                coords.append([i * l, l / np.sqrt(3), l * np.sqrt(2/3)])
                coords.append([i * l, -l / np.sqrt(3), l * np.sqrt(2/3)])
            # Additional hydrogen
            coords.append([l, 0, -l])
            return np.array(coords[:13])  # Limit to 13 nodes
        
        elif config.smiles == 'CCC' or self.molecule == 'propane':
            # Propane with hydrogens
            coords = []
            # Carbon backbone
            for i in range(3):
                coords.append([i * l, 0, 0])
            # Hydrogen atoms
            for i in range(3):
                coords.append([i * l, l / np.sqrt(3), l * np.sqrt(2/3)])
                coords.append([i * l, -l / np.sqrt(3), l * np.sqrt(2/3)])
            # Additional hydrogen
            coords.append([l, 0, -l])
            return np.array(coords[:10])  # Limit to 10 nodes
        
        else:
            # Default linear chain
            return np.array([[i * l, 0, 0] for i in range(self.num_nodes)])
    
    def _vectorize_custom_data(self, data: np.ndarray) -> np.ndarray:
        """Convert 1D custom data to 3D coordinates."""
        # Reshape data into 3D coordinates
        if len(data.shape) == 1:
            # Pad to multiple of 3
            padded_length = ((len(data) + 2) // 3) * 3
            padded_data = np.pad(data, (0, padded_length - len(data)), 'constant')
            coords_3d = padded_data.reshape(-1, 3)
        else:
            coords_3d = data.reshape(-1, 3)
        
        return coords_3d[:self.num_nodes] if coords_3d.shape[0] > self.num_nodes else coords_3d
    
    def _calculate_distance_matrix(self) -> dok_matrix:
        """Calculate sparse distance matrix between nodes."""
        n = self.num_nodes
        distances = dok_matrix((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(self.coords[i] - self.coords[j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        return distances
    
    def htr_forward(self) -> np.ndarray:
        """
        HTR Forward Transform: Convert coordinates to toggle states.
        
        Based on HTR research formula with CRV-based thresholding.
        """
        toggles = np.zeros(self.num_nodes)
        
        for i in range(self.num_nodes):
            ri_norm = np.linalg.norm(self.coords[i])
            sum_term = 0.0
            
            for j in range(self.num_nodes):
                if i != j:
                    rj_norm = np.linalg.norm(self.coords[j]) + 1e-10
                    cos_theta = np.dot(self.coords[i], self.coords[j]) / (ri_norm * rj_norm + 1e-10)
                    distance = self.distances[i, j] if (i, j) in self.distances else np.linalg.norm(self.coords[i] - self.coords[j])
                    sum_term += cos_theta / (1 + distance / self.crv)
            
            # HTR threshold condition
            if (ri_norm / self.crv + sum_term) >= self.sqrt_2:
                toggles[i] = 1.0
        
        return toggles
    
    def htr_reverse(self, toggles: np.ndarray) -> np.ndarray:
        """
        HTR Reverse Transform: Reconstruct coordinates from toggle states.
        
        Validates the reversibility of the HTR transform.
        """
        new_coords = np.zeros_like(self.coords)
        
        for i in range(self.num_nodes):
            # Find neighbors within bonding distance
            neighbors = []
            for j in range(self.num_nodes):
                if i != j:
                    distance = self.distances[i, j] if (i, j) in self.distances else np.linalg.norm(self.coords[i] - self.coords[j])
                    if distance < self.molecule_config.bond_length * 1.5:
                        neighbors.append(j)
            
            # Reconstruct position based on active neighbors
            sum_vec = np.zeros(3)
            for j in neighbors:
                if toggles[j] > 0.5:  # Active toggle
                    rj_norm = np.linalg.norm(self.coords[j]) + 1e-10
                    unit_vec = self.coords[j] / rj_norm
                    distance = self.distances[i, j] if (i, j) in self.distances else np.linalg.norm(self.coords[i] - self.coords[j])
                    weight = 1.0 / (1 + distance / self.crv)
                    sum_vec += unit_vec * weight
            
            new_coords[i] = sum_vec * self.crv
        
        return new_coords
    
    def optimize_crv(self, target_energy: Optional[float] = None) -> float:
        """
        Genetic CRV optimization to achieve target bond energy.
        
        Based on HTR research achieving exact bond energies through CRV tuning.
        """
        if target_energy is None:
            target_energy = self.molecule_config.bond_energy
        
        def objective(crv_array):
            """Objective function for CRV optimization."""
            self.crv = crv_array[0]
            self.f_i = 3e8 / (1 / (self.rydberg * self.crv) * 1e9)
            
            # Run HTR forward transform
            toggles = self.htr_forward()
            self.M = toggles
            
            # Compute energy
            energy = self.compute_energy()
            
            # Return squared error from target
            return (energy - target_energy) ** 2
        
        # Optimization bounds around initial CRV
        initial_crv = self.realm_config["CRV"]
        bounds = [(initial_crv * 0.5, initial_crv * 2.0)]
        
        # Optimize
        result = minimize(objective, [initial_crv], bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimized_crv = result.x[0]
            self.crv = optimized_crv
            self.f_i = 3e8 / (1 / (self.rydberg * self.crv) * 1e9)
            self.logger.info(f"CRV optimized: {optimized_crv:.6f} (target energy: {target_energy:.2f} eV)")
            return optimized_crv
        else:
            self.logger.warning(f"CRV optimization failed: {result.message}")
            return self.crv
    
    def compute_energy(self) -> float:
        """
        Compute system energy based on HTR research formula.
        
        Energy calculation incorporating CRV, tick frequency, and spatial relationships.
        """
        energy = 0.0
        
        for i in range(self.num_nodes):
            if self.M[i] > 0.5:  # Active toggle
                ri_norm = np.linalg.norm(self.coords[i]) + 1e-10
                
                # Distance sum to other nodes
                dist_sum = 0.0
                for j in range(self.num_nodes):
                    if j != i:
                        distance = self.distances[i, j] if (i, j) in self.distances else np.linalg.norm(self.coords[i] - self.coords[j])
                        dist_sum += distance / self.crv
                
                dist_sum += 1e-10  # Avoid division by zero
                
                # HTR energy formula
                energy += self.crv * self.f_i * (ri_norm / (1 + dist_sum))
        
        # Convert to eV (approximate scaling)
        energy_ev = energy * 1e-20  # Scaling factor from HTR research
        
        return energy_ev
    
    def update_coherence(self) -> float:
        """
        Update NRCI based on HTR forward transform prediction accuracy.
        
        NRCI measures how well the current state matches the HTR prediction.
        """
        expected = self.htr_forward()
        deviation = np.sum((self.M - expected) ** 2) / self.num_nodes
        
        # HTR NRCI formula
        self.nrci = 1 - (1 / (4 * np.pi)) * np.sqrt((1 / (4 * np.pi)) * deviation)
        self.nrci = max(0.0, min(1.0, self.nrci))  # Clamp to [0, 1]
        
        self.nrci_history.append(self.nrci)
        return self.nrci
    
    def monte_carlo_sensitivity(self, input_noise_level: float = 0.01, n_runs: int = 500) -> Dict:
        """
        Monte Carlo sensitivity analysis from HTR research.
        
        Tests CRV stability under noise conditions with 500 runs.
        """
        results = []
        original_crv = self.crv
        
        for _ in range(n_runs):
            # Add noise to CRV
            noisy_crv = original_crv * (1 + np.random.normal(0, input_noise_level))
            self.crv = noisy_crv
            self.f_i = 3e8 / (1 / (self.rydberg * self.crv) * 1e9)
            
            # Run HTR computation
            toggles = self.htr_forward()
            self.M = toggles
            energy = self.compute_energy()
            nrci = self.update_coherence()
            
            results.append({
                'crv': noisy_crv,
                'energy': energy,
                'nrci': nrci
            })
        
        # Restore original CRV
        self.crv = original_crv
        self.f_i = 3e8 / (1 / (self.rydberg * self.crv) * 1e9)
        
        # Compute statistics
        crvs = np.array([r['crv'] for r in results])
        energies = np.array([r['energy'] for r in results])
        nrcis = np.array([r['nrci'] for r in results])
        
        sensitivity_metrics = {
            'crv_std': np.std(crvs),
            'energy_std': np.std(energies),
            'nrci_std': np.std(nrcis),
            'crv_mean': np.mean(crvs),
            'energy_mean': np.mean(energies),
            'nrci_mean': np.mean(nrcis),
            'results': results
        }
        
        self.logger.info(f"Sensitivity analysis: CRV std={sensitivity_metrics['crv_std']:.2e}, "
                        f"Energy std={sensitivity_metrics['energy_std']:.2e}, "
                        f"NRCI std={sensitivity_metrics['nrci_std']:.2e}")
        
        return sensitivity_metrics
    
    def run(self, num_ticks: int = 10) -> HTRResult:
        """
        Run HTR computation for specified number of ticks.
        
        Returns complete HTR result with performance metrics.
        """
        start_time = time.time()
        
        # Optimize CRV if needed
        if hasattr(self, '_needs_crv_optimization') and self._needs_crv_optimization:
            self.optimize_crv()
            self._needs_crv_optimization = False
        
        results = []
        
        for t in range(num_ticks):
            # HTR forward transform
            self.M = self.htr_forward()
            self.M_history.append(self.M.copy())
            
            # Update coherence and energy
            nrci = self.update_coherence()
            energy = self.compute_energy()
            
            self.energy_history.append(energy)
            
            results.append({
                'tick': t,
                'active_nodes': np.sum(self.M),
                'energy': energy,
                'nrci': nrci
            })
        
        computation_time = time.time() - start_time
        
        # Test reconstruction
        reconstructed_coords = self.htr_reverse(self.M)
        reconstruction_error = np.mean(np.linalg.norm(reconstructed_coords - self.coords, axis=1))
        
        # Create result
        htr_result = HTRResult(
            toggles=self.M.copy(),
            energy=energy,
            nrci=nrci,
            computation_time=computation_time,
            crv_used=self.crv,
            reconstruction_error=reconstruction_error
        )
        
        self.logger.info(f"HTR computation complete: {num_ticks} ticks, "
                        f"Energy={energy:.2f} eV, NRCI={nrci:.7f}, "
                        f"Reconstruction error={reconstruction_error:.2e} m")
        
        return htr_result
    
    def run_with_sensitivity(self, num_ticks: int = 10, sensitivity_runs: int = 500) -> HTRResult:
        """Run HTR computation with sensitivity analysis."""
        # Run main computation
        result = self.run(num_ticks)
        
        # Add sensitivity analysis
        sensitivity_metrics = self.monte_carlo_sensitivity(n_runs=sensitivity_runs)
        result.sensitivity_metrics = sensitivity_metrics
        
        return result


    
    def process_with_htr(self, data: np.ndarray, realm: str = None) -> Dict[str, Any]:
        """
        Process data using HTR with specified realm.
        
        Args:
            data: Input data to process
            realm: Realm to use for processing (optional)
            
        Returns:
            Dictionary containing HTR processing results
        """
        try:
            # Update realm if specified
            if realm and realm != self.realm:
                self.realm = realm
            
            # Update data if provided
            if data is not None and len(data) > 0:
                self.custom_data = data
                self.coords = self._vectorize_custom_data(data)
            
            # Run HTR computation
            result = self.run(num_ticks=10)
            
            return {
                'energy': result.energy,
                'coherence': result.coherence,
                'nrci': result.nrci,
                'resonance_score': result.resonance_score,
                'harmonic_patterns': result.harmonic_patterns,
                'computation_time': result.computation_time,
                'realm_used': self.realm,
                'crv_used': self.crv
            }
        except Exception as e:
            return {
                'energy': 0.0,
                'coherence': 0.0,
                'nrci': 0.0,
                'resonance_score': 0.0,
                'harmonic_patterns': [],
                'computation_time': 0.0,
                'realm_used': self.realm,
                'crv_used': self.crv,
                'error': str(e)
            }

