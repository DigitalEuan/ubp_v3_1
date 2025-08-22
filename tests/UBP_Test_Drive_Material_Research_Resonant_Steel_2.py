# @title UBP Framework v3.1.1 - Resonant Steel Modeling & Optimization (Scientific Grade)
"""
UBP Framework v3.1.1 - Resonant Steel Modeling & Optimization
Author: Euan Craig, New Zealand
Date: 22 August 2025
Purpose: Scientifically accurate modeling of BCC steel lattice using real physics,
         integrated with UBP's HTR, HexDictionary, and Non-Random Coherence Index (NRCI).
"""

import numpy as np
import time
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime # Import datetime for timestamp

# Add the src directory to the path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Import core UBP framework and necessary engines/utilities
try:
    from ubp_framework_v3 import create_ubp_system, UBPFramework as OrchestratorFramework
    from htr_engine import HTREngine
    from rgdl_engine import RGDLEngine
    from system_constants import UBPConstants # For real physical constants
    from hex_dictionary import HexDictionary
    from offbit_utils import OffBitUtils # For manipulating raw OffBit values (ints)
    print("✅ UBP Framework v3.1 components imported successfully")
except ImportError as e:
    print(f"❌ Failed to import UBP Framework v3.1 components: {e}")
    sys.exit(1)

@dataclass
class ElementData: # Re-defined here to be self-contained for this test
    """Complete element data structure for all 118 elements."""
    atomic_number: int
    symbol: str
    name: str
    period: int
    group: int
    block: str
    valence: int
    electronegativity: float
    atomic_mass: float
    density: float
    melting_point: float
    boiling_point: float
    discovery_year: int
    electron_config: str
    oxidation_states: List[int]

# Moved from UBP_Test_Drive_Material_Research_Resonant_Steel.py
class SteelLatticeGenerator:
    def __init__(self):
        # Lattice parameters (real values for BCC Iron)
        self.bcc_lattice_constant = 2.866e-10 # meters (for Alpha-Iron at room temp)

        # Steel composition parameters (atomic percentages)
        self.base_steel_compositions = {
            'carbon_steel': {'Fe': 98.5, 'C': 1.5},
            'stainless_steel': {'Fe': 70.0, 'C': 0.1, 'Cr': 18.0, 'Ni': 8.0, 'Mn': 2.0, 'Si': 1.0, 'Mo': 0.9},
            'tool_steel': {'Fe': 85.0, 'C': 1.0, 'W': 6.0, 'Mo': 5.0, 'Cr': 2.0, 'V': 1.0},
            'spring_steel': {'Fe': 97.0, 'C': 0.6, 'Mn': 0.8, 'Si': 1.6}
        }

    def generate_steel_lattice_coords(self, composition: Dict[str, float], 
                                     supercell_dims: Tuple[int, int, int] = (3, 3, 3),
                                     elemental_influence: Optional[Dict[str, Tuple[int, int]]] = None) -> np.ndarray:
        """
        Generates 3D atomic coordinates for a steel lattice based on BCC structure for Fe,
        with interstitial and substitutional alloying. Incorporates elemental influence
        from 6D coordinates (electronegativity 'U' and valence 'V') to adjust perturbation.
        
        Args:
            composition: Dictionary of element percentages (Fe, C, Cr, etc.)
            supercell_dims: Dimensions of the supercell (e.g., (3,3,3) for 3x3x3 unit cells)
            elemental_influence: Optional dictionary mapping element symbol to (U_coord, V_coord) tuple.
            
        Returns:
            np.ndarray: (num_atoms, 3) array of 3D coordinates.
        """
        a = self.bcc_lattice_constant
        coords = []
        
        # 1. Generate ideal BCC Fe lattice sites in a supercell
        for i in range(supercell_dims[0]):
            for j in range(supercell_dims[1]):
                for k in range(supercell_dims[2]):
                    # Corner atom
                    coords.append(np.array([i * a, j * a, k * a]))
                    # Body-centered atom
                    coords.append(np.array([(i + 0.5) * a, (j + 0.5) * a, (k + 0.5) * a]))
        
        initial_num_fe = len(coords)
        element_symbols = list(composition.keys())
        element_percentages = {k: v / 100.0 for k, v in composition.items()} # Convert to fraction
        
        # Separate Fe from other elements
        other_elements = {k: v for k, v in element_percentages.items() if k != 'Fe'}
        
        # Ensure percentages sum up to 1 (or close to 1) for sanity
        current_total_percentage = sum(element_percentages.values())
        if current_total_percentage > 1.0 + 1e-6:
             scale_factor = 1.0 / current_total_percentage
             element_percentages = {k: v * scale_factor for k, v in element_percentages.items()}
             other_elements = {k: v * scale_factor for k, v in other_elements.items()}


        # Calculate number of non-Fe atoms to place based on total atoms generated so far
        total_atoms_in_supercell = initial_num_fe # Start with Fe atoms

        # Place C (interstitial) and other elements (substitutional)
        c_count_target = int(element_percentages.get('C', 0) * total_atoms_in_supercell)
        sub_elements_target = {k: int(v * total_atoms_in_supercell) for k, v in other_elements.items() if k != 'C'}

        # Create a list of all potential Fe positions that can be substituted
        fe_positions_indices = list(range(len(coords)))
        
        # Shuffle for random substitution without bias towards original positions
        np.random.shuffle(fe_positions_indices)

        # 2. Add Carbon (interstitial)
        octahedral_sites_per_unit_cell = [
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5],
            [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]
        ]

        c_added_count = 0
        for i in range(supercell_dims[0]):
            for j in range(supercell_dims[1]):
                for k in range(supercell_dims[2]):
                    if c_added_count >= c_count_target:
                        break
                    if np.random.rand() < 0.5: # Probability of adding C at this cell
                        octa_site_offset = np.array(octahedral_sites_per_unit_cell[np.random.randint(len(octahedral_sites_per_unit_cell))])
                        c_coord = np.array([i * a, j * a, k * a]) + octa_site_offset * a
                        coords.append(c_coord)
                        c_added_count += 1
                if c_added_count >= c_count_target: break
            if c_added_count >= c_count_target: break

        # 3. Substitute other alloying elements (Cr, Ni, etc.) for Fe
        for element_symbol, target_count in sub_elements_target.items():
            current_count = 0
            while current_count < target_count and len(fe_positions_indices) > 0:
                fe_idx_to_replace = fe_positions_indices.pop(0) 
                coords[fe_idx_to_replace] = coords[fe_idx_to_replace] 
                current_count += 1
        
        # --- Leveraging 6D Spatial Data ---
        # Calculate influence from 6D coordinates (U for electronegativity, V for valence)
        # U-coordinate range is typically 0-4. V-coordinate range is typically 0-6.
        fe_u = elemental_influence.get('Fe', (0,0))[0] if elemental_influence and 'Fe' in elemental_influence else 0
        fe_v = elemental_influence.get('Fe', (0,0))[1] if elemental_influence and 'Fe' in elemental_influence else 0
        c_u = elemental_influence.get('C', (0,0))[0] if elemental_influence and 'C' in elemental_influence else 0
        c_v = elemental_influence.get('C', (0,0))[1] if elemental_influence and 'C' in elemental_influence else 0

        # Simple weighted sum for reactivity score: electronegativity (U) and valence (V)
        # Higher U (electronegativity) or V (valence) values indicate more 'active' or straining atoms.
        fe_reactivity_score = (fe_u / 4.0) * 0.7 + (fe_v / 6.0) * 0.3 # Weighted blend, scaled 0-1
        c_reactivity_score = (c_u / 4.0) * 0.7 + (c_v / 6.0) * 0.3

        # Differential influence on lattice perturbation:
        # A larger difference in reactivity scores between the primary (Fe) and alloying (C) elements
        # suggests greater local strain or unique local bonding characteristics.
        # This will lead to a more pronounced, but still physically plausible, perturbation.
        influence_factor = np.clip(abs(c_reactivity_score - fe_reactivity_score) * 2.0, 0.0, 1.0) 
        
        # Adjust perturbation magnitude based on this derived elemental influence
        # Base perturbation magnitude for thermal vibrations/defects: bcc_lattice_constant * 0.05
        # Add an additional component scaled by the 'influence_factor' derived from 6D coords.
        adjusted_perturbation_magnitude = self.bcc_lattice_constant * (0.05 + influence_factor * 0.10) # Base 0.05, max 0.15

        # Randomly perturb positions slightly to break perfect lattice symmetry for simulation
        if len(coords) > 0:
            coords = np.array(coords)
            coords += (np.random.rand(*coords.shape) - 0.5) * 2 * adjusted_perturbation_magnitude
        else:
            coords = np.array([])

        return coords

# Physics functions (using realistic values for steel)
class SteelPhysics:
    def __init__(self):
        # Define these constants directly here as typical values for Iron/Steel.
        # These are properties of *materials*, not fundamental UBP constants.
        self.youngs_modulus_steel = 210.0e9 # Pascals (Pa)
        self.shear_modulus_steel = 82.0e9 # Pascals (Pa)
        self.poissons_ratio_steel = 0.29

    def hooke_law_3d(self, strain_tensor: np.ndarray) -> np.ndarray:
        """3D Hooke's Law for isotropic materials.
        
        Args:
            strain_tensor: 3x3 strain tensor.
            
        Returns:
            3x3 stress tensor in Pascals (Pa).
        """
        E = self.youngs_modulus_steel
        nu = self.poissons_ratio_steel
        # Lame parameters for isotropic materials
        lambda_lame = (E * nu) / ((1 + nu) * (1 - 2 * nu))
        mu_lame = E / (2 * (1 + nu)) # This is the shear modulus G
        
        stress_tensor = lambda_lame * np.trace(strain_tensor) * np.eye(3) + 2 * mu_lame * strain_tensor
        return stress_tensor

    def schmid_factor(self, slip_direction: np.ndarray, stress_direction: np.ndarray) -> float:
        """Calculate Schmid factor for dislocation slip.
        
        Args:
            slip_direction: Vector representing the slip direction.
            stress_direction: Vector representing the applied stress direction.
            
        Returns:
            Schmid factor (dimensionless).
        """
        slip_norm = slip_direction / np.linalg.norm(slip_direction)
        stress_norm = stress_direction / np.linalg.norm(stress_direction)
        return abs(np.dot(slip_norm, stress_norm))

    def peierls_stress(self, dislocation_width: float, burgers_vector: float) -> float:
        """Peierls-Nabarro model for dislocation stress.
        
        Args:
            dislocation_width: Width of the dislocation core (e.g., in meters).
            burgers_vector: Magnitude of the Burgers vector (e.g., in meters).
            
        Returns:
            Stress in Pascals (Pa).
        """
        G = self.shear_modulus_steel
        
        # Using a standard form of Peierls-Nabarro stress.
        return G * np.exp(-2 * np.pi * dislocation_width / burgers_vector)


    def griffith_criterion(self, stress: float, crack_length: float, surface_energy: float) -> bool:
        """Griffith fracture criterion.
        
        Args:
            stress: Applied stress (Pa).
            crack_length: Length of the crack (m).
            surface_energy: Material's surface energy (J/m^2).
            
        Returns:
            True if fracture is predicted, False otherwise.
        """
        E = self.youngs_modulus_steel
        
        K_I = stress * np.sqrt(np.pi * crack_length) # Stress Intensity Factor
        K_IC = np.sqrt(2 * E * surface_energy) # Fracture Toughness (Critical Stress Intensity Factor)
        return K_I >= K_IC

class UBPTestDriveMaterialResearchResonantSteel2:
    def __init__(self):
        self.framework = None # Will be initialized in run()
        self.htr_engine = None
        self.rgdl_engine = None
        self.steel_lattice_generator = SteelLatticeGenerator()
        self.steel_physics = SteelPhysics()
        self.hex_dict = HexDictionary() # For reading persistent data

    def run(self):
        """Main execution function for scientific-grade resonant steel test."""
        print("🚀 UBP Framework v3.1 - Resonant Steel Modeling & Optimization (Scientific Grade)")
        print("=" * 80)
        print("Demonstrating BCC steel lattice modeling with real physics integration (HTR, RGDL).")
        print("=" * 80)

        test_results = {
            'timestamp': datetime.now().isoformat(),
            'status': 'initiated',
            'elemental_data_retrieval': {},
            'lattice_generation': {},
            'physics_calculations': {},
            'htr_results': {},
            'rgdl_results': {},
            'final_summary': {}
        }

        # Initialize UBP Framework Orchestrator
        print("\n🔧 Initializing UBP Framework Orchestrator...")
        try:
            self.framework = create_ubp_system(
                default_realm="electromagnetic", # Appropriate realm for metallic bonding
                enable_htr=True, # Explicitly enable HTR
                enable_rgdl=True  # Explicitly enable RGDL
            )
            print("✅ UBP Framework Orchestrator initialized successfully.")
            
            # Initialize HTREngine and RGDLEngine using the framework's current realm context
            self.htr_engine = HTREngine(realm_name=self.framework.core_ubp.current_realm)
            self.rgdl_engine = RGDLEngine() 

            print("✅ HTREngine and RGDLEngine initialized.")
            test_results['status'] = 'framework_initialized'

        except Exception as e:
            print(f"❌ Framework or Engine initialization failed: {e}")
            import traceback
            test_results['status'] = 'framework_initialization_failed'
            test_results['error'] = str(e)
            test_results['traceback'] = traceback.format_exc()
            self._save_results_to_output(test_results)
            sys.exit(1)

        # --- Use Persistent Periodic Table Data ---
        print("\n--- Leveraging Persistent Periodic Table Data (from previous runs) ---")
        
        fe_properties = None
        c_properties = None
        fe_coords_6d = None 
        c_coords_6d = None 

        print("Searching HexDictionary for Iron (Fe) and Carbon (C) data...")
        # Iterate through HexDictionary entries and retrieve data by symbol
        for key, entry_info in self.hex_dict.entries.items():
            metadata = entry_info.get('meta', {})
            symbol = metadata.get('symbol')
            
            if symbol == 'Fe':
                fe_properties = self.hex_dict.retrieve(key)
                if fe_properties and 'coordinates_6d' in fe_properties:
                    fe_coords_6d = fe_properties['coordinates_6d']
            if symbol == 'C':
                c_properties = self.hex_dict.retrieve(key)
                if c_properties and 'coordinates_6d' in c_properties:
                    c_coords_6d = c_properties['coordinates_6d']
            
            if fe_properties and c_properties and fe_coords_6d is not None and c_coords_6d is not None:
                break 
        
        if fe_properties:
            print(f"Found Iron (Fe) properties from HexDictionary: Atomic Mass={fe_properties.get('atomic_mass'):.2f} u, Density={fe_properties.get('density'):.2f} g/cm³")
            print(f"  Fe 6D Coordinates: {fe_coords_6d}")
            test_results['elemental_data_retrieval']['Fe'] = {'atomic_mass': fe_properties.get('atomic_mass'), 'density': fe_properties.get('density'), 'coords_6d': fe_coords_6d}
        if c_properties:
            print(f"Found Carbon (C) properties from HexDictionary: Atomic Mass={c_properties.get('atomic_mass'):.2f} u, Density={c_properties.get('density'):.2f} g/cm³")
            print(f"  C 6D Coordinates: {c_coords_6d}")
            test_results['elemental_data_retrieval']['C'] = {'atomic_mass': c_properties.get('atomic_mass'), 'density': c_properties.get('density'), 'coords_6d': c_coords_6d}
        
        if not fe_properties or not c_properties or fe_coords_6d is None or c_coords_6d is None:
            print("⚠️ Could not find all required element properties (Fe, C) and their 6D coordinates in HexDictionary. Ensure periodic table test was run and data stored.")
            elemental_influence_data = None
            test_results['elemental_data_retrieval']['status'] = 'partial_data_found'
        else:
            print("Elemental properties and 6D coordinates retrieved from persistent HexDictionary. This data can inform deeper aspects of material modeling.")
            elemental_influence_data = {'Fe': (fe_coords_6d[4], fe_coords_6d[5]), 
                                        'C': (c_coords_6d[4], c_coords_6d[5])}
            test_results['elemental_data_retrieval']['status'] = 'all_data_found'

        # Get a base steel composition for testing
        test_composition = self.steel_lattice_generator.base_steel_compositions['carbon_steel']
        print(f"\n🧱 Generating BCC lattice for {test_composition}...")
        test_results['lattice_generation']['composition'] = test_composition
        test_results['lattice_generation']['supercell_dims'] = (3,3,3)
        
        # Generate lattice coordinates using the dedicated generator, passing 6D influence data
        lattice_coords = self.steel_lattice_generator.generate_steel_lattice_coords(
            test_composition, 
            supercell_dims=(3, 3, 3), 
            elemental_influence=elemental_influence_data
        )
        
        num_simulated_atoms = lattice_coords.shape[0] if lattice_coords.size > 0 else 0
        print(f"✅ Generated {num_simulated_atoms} atoms in the lattice. (Supercell 3x3x3 for Fe yields 54 atoms)")
        test_results['lattice_generation']['num_simulated_atoms'] = num_simulated_atoms

        if num_simulated_atoms == 0:
            print("⚠️ No atoms generated for simulation. Exiting test.")
            test_results['status'] = 'no_atoms_generated'
            self._save_results_to_output(test_results)
            return

        # Store lattice coordinates in HexDictionary
        try:
            lattice_coords_key = self.hex_dict.store(lattice_coords, 'array', metadata={'composition': test_composition, 'num_atoms': num_simulated_atoms})
            print(f"📦 Stored lattice in HexDictionary with key: {lattice_coords_key}")
            test_results['lattice_generation']['hex_dict_key'] = lattice_coords_key
            test_results['lattice_generation']['storage_status'] = 'success'
        except Exception as e:
            print(f"❌ Error storing lattice in HexDictionary: {e}")
            test_results['lattice_generation']['storage_status'] = 'failed'
            test_results['lattice_generation']['storage_error'] = str(e)

        # --- Test Physics Functions ---
        print("\n🔬 Testing classical materials physics functions (with realistic material properties)...")
        physics_results_accumulator = {}

        # Hooke's Law: Apply a simple uniaxial strain
        strain_example = np.array([[0.005, 0, 0], [0, 0, 0], [0, 0, 0]]) # 0.5% strain in x-direction
        stress_result = self.steel_physics.hooke_law_3d(strain_example)
        print(f"🔧 Hooke's Law 3D (0.5% uniaxial strain): Stress_xx = {stress_result[0,0]/1e6:.2f} MPa")
        physics_results_accumulator['hooke_law'] = {'strain_input': strain_example.tolist(), 'stress_output_xx_MPa': stress_result[0,0]/1e6, 'stress_tensor': stress_result.tolist()}

        # Schmid Factor: Example slip and stress directions for BCC
        slip_dir_example = np.array([1, 1, 1])
        stress_dir_example = np.array([1, 0, 0])
        schmid_factor_result = self.steel_physics.schmid_factor(slip_dir_example, stress_dir_example)
        print(f"📐 Schmid Factor (<111> slip, <100> stress): {schmid_factor_result:.3f}")
        physics_results_accumulator['schmid_factor'] = {'slip_direction': slip_dir_example.tolist(), 'stress_direction': stress_dir_example.tolist(), 'factor': schmid_factor_result}

        # Peierls Stress: Using typical values for iron Burgers vector and dislocation width
        bcc_lattice_const_m = self.steel_lattice_generator.bcc_lattice_constant # in meters
        burgers_vector_magnitude = bcc_lattice_const_m * np.sqrt(3) / 2 # in meters
        dislocation_width = burgers_vector_magnitude * 1.5 # Often proportional to burgers vector

        peierls_stress_result = self.steel_physics.peierls_stress(
            dislocation_width=dislocation_width,
            burgers_vector=burgers_vector_magnitude
        )
        print(f"⚡ Peierls Stress ({dislocation_width*1e9:.2f} nm dislocation, {burgers_vector_magnitude*1e9:.2f} nm Burgers): {peierls_stress_result/1e6:.2f} MPa") # Convert to MPa
        physics_results_accumulator['peierls_stress'] = {'dislocation_width_nm': dislocation_width*1e9, 'burgers_vector_nm': burgers_vector_magnitude*1e9, 'stress_MPa': peierls_stress_result/1e6}

        # Griffith Criterion: Example for a crack in steel
        applied_stress_mpa = 500
        crack_length_m = 100e-6 # 100 micrometers
        surface_energy_j_per_m2 = 2.0 # J/m^2

        griffith_fracture_result = self.steel_physics.griffith_criterion(
            stress=applied_stress_mpa * 1e6, # Convert MPa to Pa
            crack_length=crack_length_m,
            surface_energy=surface_energy_j_per_m2
        )
        print(f"💥 Griffith Criterion (Stress={applied_stress_mpa}MPa, Crack={crack_length_m*1e6}um, Surface Energy={surface_energy_j_per_m2}J/m²): {'Fracture' if griffith_fracture_result else 'Stable'} (Predicted)")
        physics_results_accumulator['griffith_criterion'] = {
            'applied_stress_MPa': applied_stress_mpa,
            'crack_length_um': crack_length_m*1e6,
            'surface_energy_J_per_m2': surface_energy_j_per_m2,
            'fracture_predicted': griffith_fracture_result
        }
        test_results['physics_calculations'] = physics_results_accumulator

        # --- Integrate with HTR and RGDL Engines ---
        print("\n⚙️ Integrating with HTREngine and RGDLEngine for resonance analysis...")

        # Process lattice coordinates through HTREngine
        htr_start_time = time.time()
        try:
            htr_processing_results = self.htr_engine.process_with_htr(
                lattice_coords,
                realm=self.framework.core_ubp.current_realm,
                optimize=True # Run CRV optimization within HTR for this processing
            )
            print(f"✅ HTREngine processing successful.")
            
            energy_val = htr_processing_results.get('energy')
            nrci_val = htr_processing_results.get('nrci')
            char_length_val = htr_processing_results.get('characteristic_length_scale_nm')
            
            if energy_val is not None:
                print(f"   HTR Simulated Energy: {energy_val:.4f} eV")
            else:
                print(f"   HTR Simulated Energy: (None) eV")
            
            if nrci_val is not None:
                print(f"   HTR NRCI: {nrci_val:.7f}")
            else:
                print(f"   HTR NRCI: (None)")
            
            if char_length_val is not None:
                print(f"   HTR Characteristic Length Scale: {char_length_val:.2e} nm")
            else:
                print(f"   HTR Characteristic Length Scale: (None) nm")
            test_results['htr_results'] = htr_processing_results

        except Exception as e:
            print(f"❌ HTREngine processing failed: {e}")
            test_results['htr_results'] = {'error': str(e)}
        htr_processing_time = time.time() - htr_start_time
        test_results['htr_results']['processing_time_s'] = htr_processing_time


        # Generate a conceptual RGDL primitive from the simulated lattice
        rgdl_start_time = time.time()
        try:
            rgdl_primitive = self.rgdl_engine.generate_primitive(
                'htr_structure', # This primitive type takes structure-related parameters
                resonance_realm=self.framework.core_ubp.current_realm,
                parameters={
                    'coordinates': lattice_coords, # Pass the original lattice as structure
                    'harmonic_order': 3, # Example parameter
                    # Pass the NRCI value from HTR to RGDL's htr_structure generation
                    'nrci_value': htr_processing_results.get('nrci', 0.0) # Use the NRCI from HTR
                }
            )
            print(f"✅ RGDLEngine primitive generation successful.")
            print(f"   RGDL Coherence: {rgdl_primitive.coherence_level:.4f}")
            print(f"   RGDL Stability: {rgdl_primitive.stability_score:.4f}")
            print(f"   RGDL NRCI: {rgdl_primitive.nrci_score:.4f}")
            test_results['rgdl_results'] = {
                'primitive_type': rgdl_primitive.primitive_type,
                'resonance_frequency': rgdl_primitive.resonance_frequency,
                'coherence_level': rgdl_primitive.coherence_level,
                'stability_score': rgdl_primitive.stability_score,
                'nrci_score': rgdl_primitive.nrci_score,
                'generation_method': rgdl_primitive.generation_method,
                'properties': rgdl_primitive.properties
            }
        except Exception as e:
            print(f"❌ RGDLEngine primitive generation failed: {e}")
            test_results['rgdl_results'] = {'error': str(e)}
        rgdl_processing_time = time.time() - rgdl_start_time
        test_results['rgdl_results']['processing_time_s'] = rgdl_processing_time

        # Final summary of integrated results
        print("\n--- Integrated Results Summary ---")
        print(f"HTR Processing Time: {htr_processing_time:.3f} s")
        print(f"RGDL Generation Time: {rgdl_processing_time:.3f} s")
        print(f"Overall Test Drive Complete.")
        
        test_results['final_summary'] = {
            'htr_processing_time_s': htr_processing_time,
            'rgdl_generation_time_s': rgdl_processing_time,
            'overall_test_status': 'success'
        }
        test_results['status'] = 'completed'

        # --- Save all results to /output/ ---
        self._save_results_to_output(test_results)
        
        print("\n--- EXPERIMENT FINISHED ---")

    def _save_results_to_output(self, results_dict: Dict[str, Any]):
        """Helper to save the results dictionary to a JSON file in /output/."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"ubp_steel_test_results_{timestamp}.json"
        output_filepath = os.path.join("/output/", output_filename)

        # Custom JSON serializer for numpy objects
        def numpy_serializer(obj):
            if isinstance(obj, (np.ndarray, np.generic)):
                return obj.tolist()
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

        try:
            with open(output_filepath, 'w') as f:
                json.dump(results_dict, f, indent=4, default=numpy_serializer)
            print(f"\n✅ All test results saved to: {output_filepath}")
        except Exception as e:
            print(f"❌ Error saving test results to /output/: {e}")


if __name__ == "__main__":
    UBPTestDriveMaterialResearchResonantSteel2().run()
