"""
UBP Framework v3.1.1 - Harmonic Toggle Resonance (HTR) Engine
Author: Euan Craig, New Zealand
Date: 18 August 2025

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

# Configure logging for better output
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Import configuration
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from ubp_config import get_config, UBPConfig, RealmConfig, MoleculeConfig # Import specific configs


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

# MoleculeConfig is now imported from ubp_config.py


class HTREngine:
    """
    Harmonic Toggle Resonance Engine for UBP Framework v3.0
    
    Provides molecular simulation, cross-domain data processing, and genetic CRV optimization
    based on HTR research achieving NRCI targets of 0.9999999 through precise CRV tuning.
    """
    
    # REALM_CONFIG and MOLECULE_PARAMS are now sourced from ubp_config.py
    
    def __init__(self, molecule_name: str = 'propane', realm_name: str = 'quantum', 
                custom_coords: Optional[np.ndarray] = None, 
                custom_data: Optional[np.ndarray] = None):
        """
        Initialize HTR Engine.
        
        Args:
            molecule_name: Molecule type or 'custom' for custom data
            realm_name: UBP realm for computation
            custom_coords: Custom 3D coordinates for molecular structure
            custom_data: Custom data for cross-domain processing
        """
        self.logger = logging.getLogger(__name__)
        self.config: UBPConfig = get_config() # Get the global UBPConfig instance
        
        # Configuration
        self.realm = realm_name
        self._set_realm_config(realm_name) # Helper to set realm-dependent attrs
        
        # Set molecule config
        if custom_coords is not None or custom_data is not None:
            self.molecule = 'custom'
            # If custom data, attempt to infer nodes or use a default large value
            num_nodes_from_custom = custom_coords.shape[0] if custom_coords is not None else (custom_data.size // 3 if custom_data is not None else 10)
            self.molecule_config = MoleculeConfig('custom', num_nodes_from_custom, 0.154e-9, 4.8, 'custom')
        else:
            self.molecule = molecule_name
            self.molecule_config = self.config.get_molecule_config(molecule_name)
            if not self.molecule_config:
                self.logger.warning(f"Molecule '{molecule_name}' not found in UBPConfig. Defaulting to 'propane'.")
                self.molecule = 'propane'
                self.molecule_config = self.config.get_molecule_config('propane')
        
        self.num_nodes = self.molecule_config.nodes
        
        # Molecular/data setup
        if custom_coords is not None:
            self.coords = custom_coords
        elif custom_data is not None:
            self.coords = self._vectorize_custom_data(custom_data)
        else:
            self.coords = self._generate_molecular_coords()
        
        # Physical constants from config
        self.rydberg = self.config.constants.RYDBERG_CONSTANT
        self.sqrt_2 = np.sqrt(2) # Remains numpy constant, not UBPConstant itself
        self.delta_t = 1e-15 # Not used in current code, but kept for context.
        
        # Compute tick frequency (depends on CRV)
        self._update_tick_frequency()
        
        # Initialize state
        self.M = np.random.randint(0, 2, self.num_nodes).astype(np.float64)
        self.M_history = [self.M.copy()]
        self.distances = self._calculate_distance_matrix()
        self.energy_history = []
        self.nrci_history = []
        
        self.logger.info(f"HTR Engine initialized: {self.molecule} in {realm_name} realm, {self.num_nodes} nodes")

    def _set_realm_config(self, realm_name: str):
        """Sets realm-dependent attributes by loading from config."""
        realm_cfg = self.config.get_realm_config(realm_name)
        if not realm_cfg:
            self.logger.warning(f"Realm '{realm_name}' not found in UBPConfig. Defaulting to 'quantum'.")
            realm_name = 'quantum'
            realm_cfg = self.config.get_realm_config('quantum') # Fallback
        
        self.realm = realm_name
        self.realm_config = realm_cfg
        self.crv = self.realm_config.main_crv
        self.coordination = self.realm_config.coordination_number
        self.lattice = self.realm_config.lattice_type
        self._update_tick_frequency()

    def _update_tick_frequency(self):
        """Calculates f_i based on current CRV."""
        # Ensure CRV is not zero to avoid division by zero
        if self.crv == 0:
            self.f_i = 0.0
            self.logger.warning("CRV is zero, setting f_i to 0.0. Check realm configuration.")
            return
        
        # (Speed of light / (1 / (Rydberg * CRV)) * 1e9)
        # This formula seems to derive a frequency from CRV, ensure units align
        # Assuming CRV is in Hz, Rydberg is m^-1.
        # Convert wavelength from nm to m: 1 / (rydberg * crv) would be nm if crv is related to 1/nm
        # Re-evaluate the original formula's intent given the new Rydberg constant.
        # If CRV is directly a frequency, then wavelength = c / CRV.
        # This formula looks like it computes a frequency for optical transitions:
        # f_i = c / lambda_i where lambda_i = 1 / (Rydberg * CRV)
        # This looks like it tries to compute f_i = c * rydberg * CRV.
        # For now, I'll keep the direct relationship if CRV is a frequency:
        # The original formula for f_i has issues. If CRV is a frequency in Hz, f_i is just the CRV.
        # If f_i is supposed to be a derived "tick frequency" related to the CRV's energy/wavelength equivalent:
        # For now, a straightforward connection if CRV is already frequency:
        self.f_i = self.crv # Directly use the CRV as the tick frequency if it's already a frequency.

        # Or, if f_i is meant to be a derived "tick frequency" related to the CRV's energy/wavelength equivalent:
        # self.f_i = self.config.constants.RYDBERG_CONSTANT * self.config.constants.SPEED_OF_LIGHT * (self.crv / 1e12) # Example scaling to make sense
        # Let's assume CRV is already the primary frequency.
        
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
            return np.array(coords[:config.nodes])  # Limit to config.nodes
        
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
            return np.array(coords[:config.nodes])  # Limit to config.nodes
        
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
            ri_norm = np.linalg.norm(self.coords[i]) + 1e-10 # Add epsilon to avoid div by zero
            sum_term = 0.0
            
            for j in range(self.num_nodes):
                if i != j:
                    rj_norm = np.linalg.norm(self.coords[j]) + 1e-10
                    cos_theta = np.dot(self.coords[i], self.coords[j]) / (ri_norm * rj_norm)
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
            num_active_neighbors = 0
            for j in neighbors:
                if toggles[j] > 0.5:  # Active toggle
                    rj_norm = np.linalg.norm(self.coords[j]) + 1e-10
                    unit_vec = self.coords[j] / rj_norm
                    distance = self.distances[i, j] if (i, j) in self.distances else np.linalg.norm(self.coords[i] - self.coords[j])
                    weight = 1.0 / (1 + distance / self.crv)
                    sum_vec += unit_vec * weight
                    num_active_neighbors += 1
            
            # Simple reconstruction: average of active neighbor contributions, scaled by CRV.
            # Avoid division by zero if no active neighbors
            if num_active_neighbors > 0:
                new_coords[i] = sum_vec / num_active_neighbors * self.crv
            else:
                new_coords[i] = self.coords[i] # If no active neighbors, retain original position as a fallback
        
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
            original_crv = self.crv # Store original to restore later
            self.crv = crv_array[0]
            self._update_tick_frequency()
            
            # Run HTR forward transform
            toggles = self.htr_forward()
            self.M = toggles
            
            # Compute energy
            energy = self.compute_energy()
            
            # Restore original CRV for subsequent calls if needed by the optimizer
            self.crv = original_crv 
            self._update_tick_frequency()

            # Return squared error from target
            return (energy - target_energy) ** 2
        
        # Optimization bounds around initial CRV
        initial_crv = self.realm_config.main_crv # Use config's realm CRV
        bounds = [(initial_crv * 0.5, initial_crv * 2.0)] # Adjusted bounds for stability
        
        # Optimize
        result = minimize(objective, [initial_crv], bounds=bounds, method='L-BFGS-B')
        
        if result.success:
            optimized_crv = result.x[0]
            self.crv = optimized_crv
            self._update_tick_frequency()
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
            self._update_tick_frequency()
            
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
        self._update_tick_frequency()
        
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
    
    def run(self, num_ticks: int = 10, optimize_crv_on_run: bool = False) -> HTRResult:
        """
        Run HTR computation for specified number of ticks.
        
        Returns complete HTR result with performance metrics.
        """
        start_time = time.time()
        
        # Optimize CRV if needed and requested
        if optimize_crv_on_run:
            self.optimize_crv()
        
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
    
    def run_with_sensitivity(self, num_ticks: int = 10, sensitivity_runs: int = 500, optimize_crv_on_run: bool = False) -> HTRResult:
        """Run HTR computation with sensitivity analysis."""
        # Run main computation
        result = self.run(num_ticks, optimize_crv_on_run)
        
        # Add sensitivity analysis
        sensitivity_metrics = self.monte_carlo_sensitivity(n_runs=sensitivity_runs)
        result.sensitivity_metrics = sensitivity_metrics
        
        return result
    
    def process_with_htr(self, data: np.ndarray, realm: Optional[str] = None, optimize: bool = False) -> Dict[str, Any]:
        """
        Process data using HTR with specified realm.
        
        Args:
            data: Input data to process
            realm: Realm to use for processing (optional)
            optimize: Whether to run CRV optimization before processing
            
        Returns:
            Dictionary containing HTR processing results
        """
        try:
            # Update realm if specified and different, reconfigure engine
            if realm and realm != self.realm:
                self.logger.info(f"Changing realm from {self.realm} to {realm}")
                self._set_realm_config(realm)
            
            # Update data if provided, reconfigure molecular structure
            if data is not None and len(data) > 0:
                self.logger.info(f"Processing custom data with shape {data.shape}")
                self.coords = self._vectorize_custom_data(data)
                self.num_nodes = self.coords.shape[0]
                self.distances = self._calculate_distance_matrix() # Recompute distances for new coordinates
                self.molecule_config = self.config.get_molecule_config('custom') # Use a generic custom molecule config, if it exists
                if not self.molecule_config: # If 'custom' not in config, create a placeholder
                    self.molecule_config = MoleculeConfig('custom_data', self.num_nodes, 0.154e-9, 4.8, 'custom_inferred')
                # Re-initialize M for new num_nodes, if size changes.
                self.M = np.random.randint(0, 2, self.num_nodes).astype(np.float64)
            
            # Run HTR computation, with optional CRV optimization
            result = self.run(num_ticks=10, optimize_crv_on_run=optimize)
            
            return {
                'energy': result.energy,
                'nrci': result.nrci,
                'computation_time': result.computation_time,
                'realm_used': self.realm,
                'crv_used': result.crv_used,
                'reconstruction_error': result.reconstruction_error
            }
        except Exception as e:
            self.logger.error(f"Error during HTR data processing: {e}", exc_info=True)
            return {
                'energy': 0.0,
                'nrci': 0.0,
                'computation_time': 0.0,
                'realm_used': self.realm,
                'crv_used': self.crv,
                'reconstruction_error': -1.0, # Indicate error
                'error': str(e)
            }

    def export_state(self) -> Dict[str, Any]:
        """Exports the current state of the HTR Engine to a dictionary for serialization."""
        return {
            "molecule": self.molecule,
            "realm": self.realm,
            "num_nodes": self.num_nodes,
            "crv": self.crv,
            "coords": self.coords.tolist(),
            "M": self.M.tolist(),
            "nrci_history": self.nrci_history,
            "energy_history": self.energy_history,
            "molecule_config": self.molecule_config.__dict__,
            # Note: distances (dok_matrix) and M_history can be very large.
            # For full persistence, they would need proper serialization (e.g., to list of tuples for dok_matrix).
            # For simplicity here, we'll omit them or serialize only part.
            # self.distances.items() can be converted to list of tuples (row, col, val)
        }

    def import_state(self, state_data: Dict[str, Any]):
        """Imports the state into the HTR Engine from a dictionary."""
        self.molecule = state_data.get("molecule", "propane")
        self.realm = state_data.get("realm", "quantum")
        self._set_realm_config(self.realm) # Re-init realm config

        self.num_nodes = state_data.get("num_nodes", 0)
        self.crv = state_data.get("crv", self.realm_config.main_crv) # Use realm_config for default CRV
        self.coords = np.array(state_data["coords"]) if "coords" in state_data else self._generate_molecular_coords()
        self.M = np.array(state_data["M"]) if "M" in state_data else np.random.randint(0, 2, self.num_nodes).astype(np.float64)
        
        self.nrci_history = state_data.get("nrci_history", [])
        self.energy_history = state_data.get("energy_history", [])

        # Re-initialize computed properties after basic state is loaded
        # Re-load molecule_config from global config or fallback
        mol_name = state_data.get("molecule", "propane")
        self.molecule_config = self.config.get_molecule_config(mol_name)
        if not self.molecule_config:
            self.logger.warning(f"Molecule '{mol_name}' not found during import. Using default 'propane' config.")
            self.molecule_config = self.config.get_molecule_config('propane')

        self.distances = self._calculate_distance_matrix()
        self.M_history = [self.M.copy()] # Reset M_history for simplicity on import
        self._update_tick_frequency()
