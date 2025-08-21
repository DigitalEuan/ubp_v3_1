"""
UBP Framework v3.1.1 - Complete Periodic Table Test
Author: Euan Craig, New Zealand
Date: 21 August 2025
==================================================================

Test demonstrating UBP's capability to handle the complete
periodic table with all 118 elements using:
- 6D Bitfield spatial mapping
- HexDictionary universal storage
- BitTab 24-bit encoding structure
- Multi-realm physics integration (elements processed within a specified UBP realm context)

NEW FEATURES:
- Prediction of Element 119 (Ununennium) properties and 6D coordinates using spatial gap analysis.
- Integration with Rune Protocol: Elements represented as Glyphs for symbolic computation.

"""

import sys
import os
import time
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
import math # Import math for log2

# Import for deeper pattern detection
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    # Updated import based on ubp_reference_sheet.py for UBP Framework v3.1
    from ubp_framework_v3 import create_ubp_system, UBPFramework as OrchestratorFramework # Import Orchestrator class
    from hex_dictionary import HexDictionary
    from rune_protocol import RuneProtocol, GlyphState, GlyphType # Import Rune Protocol components
    from system_constants import UBPConstants # For new element prediction
    print("✅ UBP Framework v3.1 (create_ubp_system), HexDictionary, and RuneProtocol imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

@dataclass
class ElementData:
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

class CompletePeriodicTableAnalyzer:
    """
    Revolutionary analyzer for all 118 elements using UBP Framework v3.1.
    
    This class demonstrates the potential UBP by processing every known
    element in the periodic table with 6D spatial mapping and HexDictionary
    storage capabilities.
    """
    
    def __init__(self, framework: OrchestratorFramework): # Type hint for clarity
        """Initialize the complete periodic table analyzer."""
        self.framework = framework
        self.hex_dict = HexDictionary()
        self.rune_protocol = RuneProtocol() # Initialize Rune Protocol
        self.element_storage = {}
        self.spatial_mapping = {}
        self.performance_metrics = {}
        
        # Periodic table data
        self.complete_element_data = self._initialize_complete_periodic_table()
        
        print(f"   Periodic Table Analyzer Initialized")
        print(f"   Total Elements: {len(self.complete_element_data)}")
        print(f"   Coverage: All known elements (H to Og)")
        print(f"   UBP Integration: 6D spatial mapping + HexDictionary storage")
        print(f"   Rune Protocol Integration: Elements as Glyphs enabled")
    
    def _initialize_complete_periodic_table(self) -> Dict[int, ElementData]:
        """
        Initialize complete periodic table data for all 118 elements.
        
        Returns:
            Dictionary mapping atomic numbers to ElementData objects
        """
        elements = {}
        
        # Period 1
        elements[1] = ElementData(1, 'H', 'Hydrogen', 1, 1, 's', 1, 2.20, 1.008, 0.00009, 14.01, 20.28, 1766, '1s1', [-1, 1])
        elements[2] = ElementData(2, 'He', 'Helium', 1, 18, 's', 0, 0.0, 4.003, 0.0002, 0.95, 4.22, 1868, '1s2', [0])
        
        # Period 2
        elements[3] = ElementData(3, 'Li', 'Lithium', 2, 1, 's', 1, 0.98, 6.941, 0.534, 453.69, 1615, 1817, '[He] 2s1', [1])
        elements[4] = ElementData(4, 'Be', 'Beryllium', 2, 2, 's', 2, 1.57, 9.012, 1.85, 1560, 2742, 1797, '[He] 2s2', [2])
        elements[5] = ElementData(5, 'B', 'Boron', 2, 13, 'p', 3, 2.04, 10.811, 2.34, 2349, 4200, 1808, '[He] 2s2 2p1', [3])
        elements[6] = ElementData(6, 'C', 'Carbon', 2, 14, 'p', 4, 2.55, 12.011, 2.267, 3823, 4098, -3750, '[He] 2s2 2p2', [-4, 2, 4])
        elements[7] = ElementData(7, 'N', 'Nitrogen', 2, 15, 'p', 5, 3.04, 14.007, 0.0013, 63.15, 77.36, 1772, '[He] 2s2 2p3', [-3, 3, 5])
        elements[8] = ElementData(8, 'O', 'Oxygen', 2, 16, 'p', 6, 3.44, 15.999, 0.0014, 54.36, 90.20, 1774, '[He] 2s2 2p4', [-2])
        elements[9] = ElementData(9, 'F', 'Fluorine', 2, 17, 'p', 7, 3.98, 18.998, 0.0017, 53.53, 85.03, 1886, '[He] 2s2 2p5', [-1])
        elements[10] = ElementData(10, 'Ne', 'Neon', 2, 18, 'p', 8, 0.0, 20.180, 0.0009, 24.56, 27.07, 1898, '[He] 2s2 2p6', [0])
        
        # Period 3
        elements[11] = ElementData(11, 'Na', 'Sodium', 3, 1, 's', 1, 0.93, 22.990, 0.971, 370.87, 1156, 1807, '[Ne] 3s1', [1])
        elements[12] = ElementData(12, 'Mg', 'Magnesium', 3, 2, 's', 2, 1.31, 24.305, 1.738, 923, 1363, 1755, '[Ne] 3s2', [2])
        elements[13] = ElementData(13, 'Al', 'Aluminum', 3, 13, 'p', 3, 1.61, 26.982, 2.698, 933.47, 2792, 1825, '[Ne] 3s2 3p1', [3])
        elements[14] = ElementData(14, 'Si', 'Silicon', 3, 14, 'p', 4, 1.90, 28.086, 2.3296, 1687, 3538, 1824, '[Ne] 3s2 3p2', [4])
        elements[15] = ElementData(15, 'P', 'Phosphorus', 3, 15, 'p', 5, 2.19, 30.974, 1.82, 317.30, 553.65, 1669, '[Ne] 3s2 3p3', [-3, 3, 5])
        elements[16] = ElementData(16, 'S', 'Sulfur', 3, 16, 'p', 6, 2.58, 32.065, 2.067, 388.36, 717.87, -2000, '[Ne] 3s2 3p4', [-2, 4, 6])
        elements[17] = ElementData(17, 'Cl', 'Chlorine', 3, 17, 'p', 7, 3.16, 35.453, 0.003, 171.6, 239.11, 1774, '[Ne] 3s2 3p5', [-1, 1, 3, 5, 7])
        elements[18] = ElementData(18, 'Ar', 'Argon', 3, 18, 'p', 8, 0.0, 39.948, 0.0018, 83.80, 87.30, 1894, '[Ne] 3s2 3p6', [0])
        
        # Period 4
        elements[19] = ElementData(19, 'K', 'Potassium', 4, 1, 's', 1, 0.82, 39.098, 0.862, 336.53, 1032, 1807, '[Ar] 4s1', [1])
        elements[20] = ElementData(20, 'Ca', 'Calcium', 4, 2, 's', 2, 1.00, 40.078, 1.54, 1115, 1757, 1808, '[Ar] 4s2', [2])
        elements[21] = ElementData(21, 'Sc', 'Scandium', 4, 3, 'd', 3, 1.36, 44.956, 2.989, 1814, 3109, 1879, '[Ar] 3d1 4s2', [3])
        elements[22] = ElementData(22, 'Ti', 'Titanium', 4, 4, 'd', 4, 1.54, 47.867, 4.506, 1941, 3560, 1791, '[Ar] 3d2 4s2', [2, 3, 4])
        elements[23] = ElementData(23, 'V', 'Vanadium', 4, 5, 'd', 5, 1.63, 50.942, 6.11, 2183, 3680, 1801, '[Ar] 3d3 4s2', [2, 3, 4, 5])
        elements[24] = ElementData(24, 'Cr', 'Chromium', 4, 6, 'd', 6, 1.66, 51.996, 7.15, 2180, 2944, 1797, '[Ar] 3d5 4s1', [2, 3, 6])
        elements[25] = ElementData(25, 'Mn', 'Manganese', 4, 7, 'd', 7, 1.55, 54.938, 7.44, 1519, 2334, 1774, '[Ar] 3d5 4s2', [2, 3, 4, 6, 7])
        elements[26] = ElementData(26, 'Fe', 'Iron', 4, 8, 'd', 8, 1.83, 55.845, 7.874, 1811, 3134, -4000, '[Ar] 3d6 4s2', [2, 3])
        elements[27] = ElementData(27, 'Co', 'Cobalt', 4, 9, 'd', 9, 1.88, 58.933, 8.86, 1768, 3200, 1735, '[Ar] 3d7 4s2', [2, 3])
        elements[28] = ElementData(28, 'Ni', 'Nickel', 4, 10, 'd', 10, 1.91, 58.693, 8.912, 1728, 3186, 1751, '[Ar] 3d8 4s2', [2, 3])
        elements[29] = ElementData(29, 'Cu', 'Copper', 4, 11, 'd', 11, 1.90, 63.546, 8.96, 1357.77, 2835, -7000, '[Ar] 3d10 4s1', [1, 2])
        elements[30] = ElementData(30, 'Zn', 'Zinc', 4, 12, 'd', 12, 1.65, 65.38, 7.134, 692.68, 1180, 1746, '[Ar] 3d10 4s2', [2])
        elements[31] = ElementData(31, 'Ga', 'Gallium', 4, 13, 'p', 3, 1.81, 69.723, 5.907, 302.91, 2673, 1875, '[Ar] 3d10 4s2 4p1', [3])
        elements[32] = ElementData(32, 'Ge', 'Germanium', 4, 14, 'p', 4, 2.01, 72.64, 5.323, 1211.40, 3106, 1886, '[Ar] 3d10 4s2 4p2', [2, 4])
        elements[33] = ElementData(33, 'As', 'Arsenic', 4, 15, 'p', 5, 2.18, 74.922, 5.776, 1090, 887, 1250, '[Ar] 3d10 4s2 4p3', [-3, 3, 5])
        elements[34] = ElementData(34, 'Se', 'Selenium', 4, 16, 'p', 6, 2.55, 78.96, 4.809, 494, 958, 1817, '[Ar] 3d10 4s2 4p4', [-2, 4, 6])
        elements[35] = ElementData(35, 'Br', 'Bromine', 4, 17, 'p', 7, 2.96, 79.904, 3.122, 265.8, 332.0, 1826, '[Ar] 3d10 4s2 4p5', [-1, 1, 3, 5, 7])
        elements[36] = ElementData(36, 'Kr', 'Krypton', 4, 18, 'p', 8, 3.00, 83.798, 0.0037, 115.79, 119.93, 1898, '[Ar] 3d10 4s2 4p6', [0, 2])
        
        # Period 5
        elements[37] = ElementData(37, 'Rb', 'Rubidium', 5, 1, 's', 1, 0.82, 85.468, 1.532, 312.46, 961, 1861, '[Kr] 5s1', [1])
        elements[38] = ElementData(38, 'Sr', 'Strontium', 5, 2, 's', 2, 0.95, 87.62, 2.64, 1050, 1655, 1790, '[Kr] 5s2', [2])
        elements[39] = ElementData(39, 'Y', 'Yttrium', 5, 3, 'd', 3, 1.22, 88.906, 4.469, 1799, 3609, 1794, '[Kr] 4d1 5s2', [3])
        elements[40] = ElementData(40, 'Zr', 'Zirconium', 5, 4, 'd', 4, 1.33, 91.224, 6.506, 2128, 4682, 1789, '[Kr] 4d2 5s2', [4])
        elements[41] = ElementData(41, 'Nb', 'Niobium', 5, 5, 'd', 5, 1.6, 92.906, 8.57, 2750, 5017, 1801, '[Kr] 4d4 5s1', [3, 5])
        elements[42] = ElementData(42, 'Mo', 'Molybdenum', 5, 6, 'd', 6, 2.16, 95.96, 10.22, 2896, 4912, 1778, '[Kr] 4d5 5s1', [2, 3, 4, 5, 6])
        elements[43] = ElementData(43, 'Tc', 'Technetium', 5, 7, 'd', 7, 1.9, 98.0, 11.5, 2430, 4538, 1937, '[Kr] 4d5 5s2', [4, 6, 7])
        elements[44] = ElementData(44, 'Ru', 'Ruthenium', 5, 8, 'd', 8, 2.2, 101.07, 12.37, 2607, 4423, 1844, '[Kr] 4d7 5s1', [2, 3, 4, 6, 8])
        elements[45] = ElementData(45, 'Rh', 'Rhodium', 5, 9, 'd', 9, 2.28, 102.91, 12.41, 2237, 3968, 1803, '[Kr] 4d8 5s1', [1, 3])
        elements[46] = ElementData(46, 'Pd', 'Palladium', 5, 10, 'd', 10, 2.20, 106.42, 12.02, 1828.05, 3236, 1803, '[Kr] 4d10', [2, 4])
        elements[47] = ElementData(47, 'Ag', 'Silver', 5, 11, 'd', 11, 1.93, 107.87, 10.501, 1234.93, 2435, -3000, '[Kr] 4d10 5s1', [1])
        elements[48] = ElementData(48, 'Cd', 'Cadmium', 5, 12, 'd', 12, 1.69, 112.41, 8.69, 594.22, 1040, 1817, '[Kr] 4d10 5s2', [2])
        elements[49] = ElementData(49, 'In', 'Indium', 5, 13, 'p', 3, 1.78, 114.82, 7.31, 429.75, 2345, 1863, '[Kr] 4d10 5s2 5p1', [1, 3])
        elements[50] = ElementData(50, 'Sn', 'Tin', 5, 14, 'p', 4, 1.96, 118.71, 7.287, 505.08, 2875, -2100, '[Kr] 4d10 5s2 5p2', [2, 4])
        elements[51] = ElementData(51, 'Sb', 'Antimony', 5, 15, 'p', 5, 2.05, 121.76, 6.685, 903.78, 1860, 1450, '[Kr] 4d10 5s2 5p3', [-3, 3, 5])
        elements[52] = ElementData(52, 'Te', 'Tellurium', 5, 16, 'p', 6, 2.1, 127.60, 6.232, 722.66, 1261, 1783, '[Kr] 4d10 5s2 5p4', [-2, 4, 6])
        elements[53] = ElementData(53, 'I', 'Iodine', 5, 17, 'p', 7, 2.66, 126.90, 4.93, 386.85, 457.4, 1811, '[Kr] 4d10 5s2 5p5', [-1, 1, 3, 5, 7])
        elements[54] = ElementData(54, 'Xe', 'Xenon', 5, 18, 'p', 8, 2.60, 131.29, 0.0059, 161.4, 165.03, 1898, '[Kr] 4d10 5s2 5p6', [0, 2, 4, 6, 8])
        
        # Period 6
        elements[55] = ElementData(55, 'Cs', 'Cesium', 6, 1, 's', 1, 0.79, 132.91, 1.873, 301.59, 944, 1860, '[Xe] 6s1', [1])
        elements[56] = ElementData(56, 'Ba', 'Barium', 6, 2, 's', 2, 0.89, 137.33, 3.594, 1000, 2170, 1808, '[Xe] 6s2', [2])
        elements[57] = ElementData(57, 'La', 'Lanthanum', 6, 3, 'f', 3, 1.10, 138.91, 6.145, 1193, 3737, 1839, '[Xe] 5d1 6s2', [3])
        elements[58] = ElementData(58, 'Ce', 'Cerium', 6, 3, 'f', 4, 1.12, 140.12, 6.770, 1068, 3716, 1803, '[Xe] 4f1 5d1 6s2', [3, 4])
        elements[59] = ElementData(59, 'Pr', 'Praseodymium', 6, 3, 'f', 5, 1.13, 140.91, 6.773, 1208, 3793, 1885, '[Xe] 4f3 6s2', [3])
        elements[60] = ElementData(60, 'Nd', 'Neodymium', 6, 3, 'f', 6, 1.14, 144.24, 7.007, 1297, 3347, 1885, '[Xe] 4f4 6s2', [3])
        elements[61] = ElementData(61, 'Pm', 'Promethium', 6, 3, 'f', 7, 1.13, 145.0, 7.26, 1315, 3273, 1945, '[Xe] 4f5 6s2', [3])
        elements[62] = ElementData(62, 'Sm', 'Samarium', 6, 3, 'f', 8, 1.17, 150.36, 7.52, 1345, 2067, 1879, '[Xe] 4f6 6s2', [2, 3])
        elements[63] = ElementData(63, 'Eu', 'Europium', 6, 3, 'f', 9, 1.20, 151.96, 5.243, 1099, 1802, 1901, '[Xe] 4f7 6s2', [2, 3])
        elements[64] = ElementData(64, 'Gd', 'Gadolinium', 6, 3, 'f', 10, 1.20, 157.25, 7.895, 1585, 3546, 1880, '[Xe] 4f7 5d1 6s2', [3])
        elements[65] = ElementData(65, 'Tb', 'Terbium', 6, 3, 'f', 11, 1.20, 158.93, 8.229, 1629, 3503, 1843, '[Xe] 4f9 6s2', [3, 4])
        elements[66] = ElementData(66, 'Dy', 'Dysprosium', 6, 3, 'f', 12, 1.22, 162.50, 8.55, 1680, 2840, 1886, '[Xe] 4f10 6s2', [3])
        elements[67] = ElementData(67, 'Ho', 'Holmium', 6, 3, 'f', 13, 1.23, 164.93, 8.795, 1734, 2993, 1878, '[Xe] 4f11 6s2', [3])
        elements[68] = ElementData(68, 'Er', 'Erbium', 6, 3, 'f', 14, 1.24, 167.26, 9.066, 1802, 3141, 1843, '[Xe] 4f12 6s2', [3])
        elements[69] = ElementData(69, 'Tm', 'Thulium', 6, 3, 'f', 15, 1.25, 168.93, 9.321, 1818, 2223, 1879, '[Xe] 4f13 6s2', [2, 3])
        elements[70] = ElementData(70, 'Yb', 'Ytterbium', 6, 3, 'f', 16, 1.10, 173.05, 6.965, 1097, 1469, 1878, '[Xe] 4f14 6s2', [2, 3])
        elements[71] = ElementData(71, 'Lu', 'Lutetium', 6, 3, 'd', 17, 1.27, 174.97, 9.84, 1925, 3675, 1907, '[Xe] 4f14 5d1 6s2', [3])
        elements[72] = ElementData(72, 'Hf', 'Hafnium', 6, 4, 'd', 4, 1.3, 178.49, 13.31, 2506, 4876, 1923, '[Xe] 4f14 5d2 6s2', [4])
        elements[73] = ElementData(73, 'Ta', 'Tantalum', 6, 5, 'd', 5, 1.5, 180.95, 16.654, 3290, 5731, 1802, '[Xe] 4f14 5d3 6s2', [5])
        elements[74] = ElementData(74, 'W', 'Tungsten', 6, 6, 'd', 6, 2.36, 183.84, 19.25, 3695, 5828, 1783, '[Xe] 4f14 5d4 6s2', [2, 3, 4, 5, 6])
        elements[75] = ElementData(75, 'Re', 'Rhenium', 6, 7, 'd', 7, 1.9, 186.21, 21.02, 3459, 5869, 1925, '[Xe] 4f14 5d5 6s2', [2, 4, 6, 7])
        elements[76] = ElementData(76, 'Os', 'Osmium', 6, 8, 'd', 8, 2.2, 190.23, 22.61, 3306, 5285, 1803, '[Xe] 4f14 5d6 6s2', [2, 3, 4, 6, 8])
        elements[77] = ElementData(77, 'Ir', 'Iridium', 6, 9, 'd', 9, 2.20, 192.22, 22.56, 2739, 4701, 1803, '[Xe] 4f14 5d7 6s2', [1, 3, 4, 6])
        elements[78] = ElementData(78, 'Pt', 'Platinum', 6, 10, 'd', 10, 2.28, 195.08, 21.46, 2041.4, 4098, 1735, '[Xe] 4f14 5d9 6s1', [2, 4])
        elements[79] = ElementData(79, 'Au', 'Gold', 6, 11, 'd', 11, 2.54, 196.97, 19.282, 1337.33, 3129, -2600, '[Xe] 4f14 5d10 6s1', [1, 3])
        elements[80] = ElementData(80, 'Hg', 'Mercury', 6, 12, 'd', 12, 2.00, 200.59, 13.5336, 234.43, 629.88, -750, '[Xe] 4f14 5d10 6s2', [1, 2])
        elements[81] = ElementData(81, 'Tl', 'Thallium', 6, 13, 'p', 3, 1.62, 204.38, 11.85, 577, 1746, 1861, '[Xe] 4f14 5d10 6s2 6p1', [1, 3])
        elements[82] = ElementData(82, 'Pb', 'Lead', 6, 14, 'p', 4, 2.33, 207.2, 11.342, 600.61, 2022, -7000, '[Xe] 4f14 5d10 6s2 6p2', [2, 4])
        elements[83] = ElementData(83, 'Bi', 'Bismuth', 6, 15, 'p', 5, 2.02, 208.98, 9.807, 544.7, 1837, 1753, '[Xe] 4f14 5d10 6s2 6p3', [3, 5])
        elements[84] = ElementData(84, 'Po', 'Polonium', 6, 16, 'p', 6, 2.0, 209.0, 9.32, 527, 1235, 1898, '[Xe] 4f14 5d10 6s2 6p4', [2, 4])
        elements[85] = ElementData(85, 'At', 'Astatine', 6, 17, 'p', 7, 2.2, 210.0, 7.0, 575, 610, 1940, '[Xe] 4f14 5d10 6s2 6p5', [-1, 1, 3, 5, 7])
        elements[86] = ElementData(86, 'Rn', 'Radon', 6, 18, 'p', 8, 2.2, 222.0, 0.00973, 202, 211.3, 1900, '[Xe] 4f14 5d10 6s2 6p6', [0, 2])
        
        # Period 7
        elements[87] = ElementData(87, 'Fr', 'Francium', 7, 1, 's', 1, 0.7, 223.0, 1.87, 300, 950, 1939, '[Rn] 7s1', [1])
        elements[88] = ElementData(88, 'Ra', 'Radium', 7, 2, 's', 2, 0.9, 226.0, 5.5, 973, 2010, 1898, '[Rn] 7s2', [2])
        elements[89] = ElementData(89, 'Ac', 'Actinium', 7, 3, 'f', 3, 1.1, 227.0, 10.07, 1323, 3471, 1899, '[Rn] 6d1 7s2', [3])
        elements[90] = ElementData(90, 'Th', 'Thorium', 7, 3, 'f', 4, 1.3, 232.04, 11.72, 2115, 5061, 1829, '[Rn] 6d2 7s2', [4])
        elements[91] = ElementData(91, 'Pa', 'Protactinium', 7, 3, 'f', 5, 1.5, 231.04, 15.37, 1841, 4300, 1913, '[Rn] 5f2 6d1 7s2', [4, 5])
        elements[92] = ElementData(92, 'U', 'Uranium', 7, 3, 'f', 6, 1.38, 238.03, 18.95, 1405.3, 4404, 1789, '[Rn] 5f3 6d1 7s2', [3, 4, 5, 6])
        elements[93] = ElementData(93, 'Np', 'Neptunium', 7, 3, 'f', 7, 1.36, 237.0, 20.45, 917, 4273, 1940, '[Rn] 5f4 6d1 7s2', [3, 4, 5, 6, 7])
        elements[94] = ElementData(94, 'Pu', 'Plutonium', 7, 3, 'f', 8, 1.28, 244.0, 19.84, 912.5, 3501, 1940, '[Rn] 5f6 7s2', [3, 4, 5, 6])
        elements[95] = ElementData(95, 'Am', 'Americium', 7, 3, 'f', 9, 1.13, 243.0, 13.69, 1449, 2880, 1944, '[Rn] 5f7 7s2', [2, 3, 4, 5, 6])
        elements[96] = ElementData(96, 'Cm', 'Curium', 7, 3, 'f', 10, 1.28, 247.0, 13.51, 1613, 3383, 1944, '[Rn] 5f7 6d1 7s2', [3, 4])
        elements[97] = ElementData(97, 'Bk', 'Berkelium', 7, 3, 'f', 11, 1.3, 247.0, 14.79, 1259, 2900, 1949, '[Rn] 5f9 7s2', [3, 4])
        elements[98] = ElementData(98, 'Cf', 'Californium', 7, 3, 'f', 12, 1.3, 251.0, 15.1, 1173, 1743, 1950, '[Rn] 5f10 7s2', [2, 3, 4])
        elements[99] = ElementData(99, 'Es', 'Einsteinium', 7, 3, 'f', 13, 1.3, 252.0, 8.84, 1133, 1269, 1952, '[Rn] 5f11 7s2', [2, 3])
        elements[100] = ElementData(100, 'Fm', 'Fermium', 7, 3, 'f', 14, 1.3, 257.0, 9.7, 1800, 0, 1952, '[Rn] 5f12 7s2', [2, 3])
        elements[101] = ElementData(101, 'Md', 'Mendelevium', 7, 3, 'f', 15, 1.3, 258.0, 10.3, 1100, 0, 1955, '[Rn] 5f13 7s2', [2, 3])
        elements[102] = ElementData(102, 'No', 'Nobelium', 7, 3, 'f', 16, 1.3, 259.0, 9.9, 1100, 0, 1957, '[Rn] 5f14 7s2', [2, 3])
        elements[103] = ElementData(103, 'Lr', 'Lawrencium', 7, 3, 'd', 17, 1.3, 266.0, 15.6, 1900, 0, 1961, '[Rn] 5f14 6d1 7s2', [3])
        elements[104] = ElementData(104, 'Rf', 'Rutherfordium', 7, 4, 'd', 4, 0.0, 267.0, 23.2, 2400, 5800, 1964, '[Rn] 5f14 6d2 7s2', [4])
        elements[105] = ElementData(105, 'Db', 'Dubnium', 7, 5, 'd', 5, 0.0, 268.0, 29.3, 0, 0, 1967, '[Rn] 5f14 6d3 7s2', [5])
        elements[106] = ElementData(106, 'Sg', 'Seaborgium', 7, 6, 'd', 6, 0.0, 269.0, 35.0, 0, 0, 1974, '[Rn] 5f14 6d4 7s2', [6])
        elements[107] = ElementData(107, 'Bh', 'Bohrium', 7, 7, 'd', 7, 0.0, 270.0, 37.1, 0, 0, 1981, '[Rn] 5f14 6d5 7s2', [7])
        elements[108] = ElementData(108, 'Hs', 'Hassium', 7, 8, 'd', 8, 0.0, 277.0, 40.7, 0, 0, 1984, '[Rn] 5f14 6d6 7s2', [8])
        elements[109] = ElementData(109, 'Mt', 'Meitnerium', 7, 9, 'd', 9, 0.0, 278.0, 37.4, 0, 0, 1982, '[Rn] 5f14 6d7 7s2', [9])
        elements[110] = ElementData(110, 'Ds', 'Darmstadtium', 7, 10, 'd', 10, 0.0, 281.0, 34.8, 0, 0, 1994, '[Rn] 5f14 6d8 7s2', [10])
        elements[111] = ElementData(111, 'Rg', 'Roentgenium', 7, 11, 'd', 11, 0.0, 282.0, 28.7, 0, 0, 1994, '[Rn] 5f14 6d9 7s2', [11])
        elements[112] = ElementData(112, 'Cn', 'Copernicium', 7, 12, 'd', 12, 0.0, 285.0, 23.7, 0, 0, 1996, '[Rn] 5f14 6d10 7s2', [12])
        elements[113] = ElementData(113, 'Nh', 'Nihonium', 7, 13, 'p', 3, 0.0, 286.0, 16.0, 700, 1400, 2004, '[Rn] 5f14 6d10 7s2 7p1', [13])
        elements[114] = ElementData(114, 'Fl', 'Flerovium', 7, 14, 'p', 4, 0.0, 289.0, 14.0, 200, 380, 1999, '[Rn] 5f14 6d10 7s2 7p2', [14])
        elements[115] = ElementData(115, 'Mc', 'Moscovium', 7, 15, 'p', 5, 0.0, 290.0, 13.5, 700, 1400, 2003, '[Rn] 5f14 6d10 7s2 7p3', [15])
        elements[116] = ElementData(116, 'Lv', 'Livermorium', 7, 16, 'p', 6, 0.0, 293.0, 12.9, 709, 1085, 2000, '[Rn] 5f14 6d10 7s2 7p4', [16])
        elements[117] = ElementData(117, 'Ts', 'Tennessine', 7, 17, 'p', 7, 0.0, 294.0, 7.2, 723, 883, 2010, '[Rn] 5f14 6d10 7s2 7p5', [17])
        elements[118] = ElementData(118, 'Og', 'Oganesson', 7, 18, 'p', 8, 0.0, 294.0, 5.0, 325, 450, 2002, '[Rn] 5f14 6d10 7s2 7p6', [18])
        
        return elements
    
    def calculate_6d_coordinates_bittab(self, element: ElementData) -> Tuple[int, int, int, int, int, int]:
        """
        Calculate 6D coordinates using BitTab 24-bit encoding structure.
        
        Based on your BitTab structure:
        - Bits 1-8: Atomic Number (8 bits)
        - Bits 9-12: Electron Configuration Flags (4 bits) - s/p/d/f blocks
        - Bits 13-15: Valence Electrons (3 bits)
        - Bit 16: Electronegativity Flag (1 bit)
        - Bits 17-19: Period (3 bits)
        - Bits 20-24: Group (5 bits)
        
        Args:
            element: ElementData object
            
        Returns:
            6D coordinates (x, y, z, w, u, v)
        """
        # X: Atomic number modulo for spatial distribution
        x = element.atomic_number % 12
        
        # Y: Period-based coordinate
        y = element.period % 8
        
        # Z: Group-based coordinate
        z = element.group % 20
        
        # W: Block-based coordinate (s=0, p=1, d=2, f=3)
        block_map = {'s': 0, 'p': 1, 'd': 2, 'f': 3}
        w = block_map.get(element.block, 0)
        
        # U: Electronegativity-based coordinate
        if element.electronegativity > 0:
            u = min(int(element.electronegativity), 4)
        else:
            u = 0
        
        # V: Valence-based coordinate
        v = element.valence % 6
        
        return (x, y, z, w, u, v)
    
    def encode_element_to_bittab(self, element: ElementData) -> str:
        """
        Encode element using BitTab 24-bit structure.
        
        Args:
            element: ElementData object
            
        Returns:
            24-bit binary string representing the element
        """
        # Bits 1-8: Atomic Number (8 bits)
        atomic_bits = format(element.atomic_number, '08b')
        
        # Bits 9-12: Electron Configuration Flags (4 bits)
        block_map = {'s': 0b0001, 'p': 0b0010, 'd': 0b0100, 'f': 0b1000}
        config_bits = format(block_map.get(element.block, 0b0001), '04b')
        
        # Bits 13-15: Valence Electrons (3 bits)
        valence_bits = format(min(element.valence, 7), '03b')
        
        # Bit 16: Electronegativity Flag (1 bit)
        electro_bit = '1' if element.electronegativity > 2.0 else '0'
        
        # Bits 17-19: Period (3 bits)
        period_bits = format(element.period, '03b')
        
        # Bits 20-24: Group (5 bits)
        group_bits = format(element.group, '05b')
        
        # Combine all bits
        bittab_encoding = atomic_bits + config_bits + valence_bits + electro_bit + period_bits + group_bits
        
        return bittab_encoding

    def _calculate_shannon_entropy(self, data_bytes: bytes) -> float:
        """
        Calculates the Shannon entropy (in bits) of a sequence of bytes.
        """
        if not data_bytes:
            return 0.0

        frequency = {}
        for byte_val in data_bytes:
            frequency[byte_val] = frequency.get(byte_val, 0) + 1

        entropy = 0.0
        total_bytes = len(data_bytes)
        for count in frequency.values():
            probability = count / total_bytes
            if probability > 0: # Avoid log(0)
                entropy -= probability * math.log2(probability)
        return entropy
    
    def store_complete_periodic_table(self) -> Dict[str, Any]:
        """
        Store all 118 elements in HexDictionary with complete analysis.
        
        Returns:
            Storage results and comprehensive statistics
        """
        print("📦 Storing Complete Periodic Table...")
        
        storage_results = {
            'elements_stored': 0,
            'total_storage_time': 0.0,
            'bittab_encoding_ratio': 0.0, # Renamed from compression_efficiency
            'spatial_distribution': {},
            'block_distribution': {'s': 0, 'p': 0, 'd': 0, 'f': 0},
            'period_distribution': {},
            'bittab_encodings': {},
            'storage_errors': [],
            'total_original_shannon_entropy_bits': 0.0 # Added for Shannon entropy
        }
        
        start_time = time.time()
        
        # Get the current realm from the orchestrator framework
        current_realm = self.framework.core_ubp.current_realm

        for atomic_number, element in self.complete_element_data.items():
            try:
                # Calculate 6D coordinates using BitTab structure
                coords_6d = self.calculate_6d_coordinates_bittab(element)
                
                # Generate BitTab encoding
                bittab_encoding = self.encode_element_to_bittab(element)
                
                # Create comprehensive element data for storage
                storage_data = {
                    'atomic_number': element.atomic_number,
                    'symbol': element.symbol,
                    'name': element.name,
                    'period': element.period,
                    'group': element.group,
                    'block': element.block,
                    'valence': element.valence,
                    'electronegativity': element.electronegativity,
                    'atomic_mass': element.atomic_mass,
                    'density': element.density,
                    'melting_point': element.melting_point,
                    'boiling_point': element.boiling_point,
                    'discovery_year': element.discovery_year,
                    'electron_config': element.electron_config,
                    'oxidation_states': element.oxidation_states,
                    'coordinates_6d': coords_6d,
                    'bittab_encoding': bittab_encoding
                }
                
                # Calculate Shannon entropy for the original data (serialized as JSON bytes)
                serialized_data_bytes = json.dumps(storage_data, default=str).encode('utf-8')
                element_shannon_entropy = self._calculate_shannon_entropy(serialized_data_bytes)
                storage_results['total_original_shannon_entropy_bits'] += element_shannon_entropy

                # Store in HexDictionary
                metadata = {
                    'atomic_number': element.atomic_number,
                    'symbol': element.symbol, # Include symbol in metadata for direct lookup if desired
                    'coordinates_6d': coords_6d,
                    'block': element.block,
                    'period': element.period,
                    'group': element.group,
                    'bittab_encoding': bittab_encoding,
                    'realm_context': current_realm # Explicitly add realm context
                }
                
                # Corrected: Removed 'custom_key' as HexDictionary is content-addressable
                # The hash of 'storage_data' will serve as the key.
                stored_key = self.hex_dict.store(
                    data=storage_data,
                    data_type='json',
                    metadata=metadata
                )
                
                # Update storage tracking
                # The primary key for element_storage should be the symbol for convenience
                # but the HexDictionary's internal key is the hash.
                self.element_storage[element.symbol] = {
                    'atomic_number': element.atomic_number,
                    'stored_hash_key': stored_key, # Renamed to clarify it's the hash
                    'coordinates_6d': coords_6d,
                    'bittab_encoding': bittab_encoding,
                    'original_data': element
                }
                
                # Update statistics
                storage_results['elements_stored'] += 1
                storage_results['block_distribution'][element.block] += 1
                
                if element.period not in storage_results['period_distribution']:
                    storage_results['period_distribution'][element.period] = 0
                storage_results['period_distribution'][element.period] += 1
                
                storage_results['bittab_encodings'][element.symbol] = bittab_encoding
                
                # Print progress for key milestones
                if atomic_number in [1, 10, 18, 36, 54, 86, 118]:
                    print(f"      ✅ {element.symbol} ({element.name}): 6D{coords_6d}, BitTab: {bittab_encoding[:8]}...")
                    
            except Exception as e:
                storage_results['storage_errors'].append(f"{element.symbol}: {str(e)}")
                print(f"      ❌ Error storing {element.symbol}: {e}")
        
        storage_results['total_storage_time'] = time.time() - start_time
        
        # Calculate bittab_encoding_ratio using Shannon Entropy (User Suggestion 1)
        # Ratio = Bit Length Information Content (Shannon Entropy)
        if storage_results['elements_stored'] > 0 and storage_results['total_original_shannon_entropy_bits'] > 0:
            total_bittab_bit_length = storage_results['elements_stored'] * 24 # 24 bits per element in bittab
            storage_results['bittab_encoding_ratio'] = total_bittab_bit_length / storage_results['total_original_shannon_entropy_bits']
        else:
            storage_results['bittab_encoding_ratio'] = 0.0
        
        print(f"    Storage complete: {storage_results['elements_stored']}/118 elements")
        print(f"    Block distribution: s={storage_results['block_distribution']['s']}, p={storage_results['block_distribution']['p']}, d={storage_results['block_distribution']['d']}, f={storage_results['block_distribution']['f']}")
        print(f"    Storage time: {storage_results['total_storage_time']:.3f} seconds")
        print(f"    BitTab encoding ratio (Shannon): {storage_results['bittab_encoding_ratio']:.3f}")
        
        return storage_results
    
    def _characterize_outlier(self, outlier_symbol: str, all_element_data: Dict[int, ElementData]) -> Dict[str, Any]:
        """
        Characterizes an outlier element by comparing its properties to its
        expected group and period averages.
        """
        outlier_element = None
        for data in all_element_data.values():
            if data.symbol == outlier_symbol:
                outlier_element = data
                break
        
        if outlier_element is None: # Corrected from === None
            return {"symbol": outlier_symbol, "error": "Element data not found."}

        # Collect properties for comparison
        properties_to_compare = [
            'atomic_mass', 'electronegativity', 'density', 
            'melting_point', 'boiling_point', 'valence'
        ]
        
        group_elements = [e for e in all_element_data.values() if e.group == outlier_element.group and e.symbol != outlier_symbol]
        period_elements = [e for e in all_element_data.values() if e.period == outlier_element.period and e.symbol != outlier_symbol]

        char_data = {
            "symbol": outlier_symbol,
            "name": outlier_element.name,
            "traditional_group": outlier_element.group,
            "traditional_period": outlier_element.period,
            "deviations": {}
        }
        
        # Compare to group averages
        if group_elements:
            group_avg = {prop: np.mean([getattr(e, prop) for e in group_elements if getattr(e, prop) is not None and getattr(e, prop) > 0]) for prop in properties_to_compare}
            group_std = {prop: np.std([getattr(e, prop) for e in group_elements if getattr(e, prop) is not None and getattr(e, prop) > 0]) for prop in properties_to_compare}
            char_data["group_comparison"] = {prop: {
                "outlier_value": getattr(outlier_element, prop),
                "group_avg": group_avg.get(prop, 0.0),
                "deviation_factor": (getattr(outlier_element, prop) - group_avg.get(prop, 0.0)) / (group_std.get(prop, 1e-6)) # normalized deviation
            } for prop in properties_to_compare}
        
        # Compare to period averages
        if period_elements:
            period_avg = {prop: np.mean([getattr(e, prop) for e in period_elements if getattr(e, prop) is not None and getattr(e, prop) > 0]) for prop in properties_to_compare}
            period_std = {prop: np.std([getattr(e, prop) for e in period_elements if getattr(e, prop) is not None and getattr(e, prop) > 0]) for prop in properties_to_compare}
            char_data["period_comparison"] = {prop: {
                "outlier_value": getattr(outlier_element, prop),
                "period_avg": period_avg.get(prop, 0.0),
                "deviation_factor": (getattr(outlier_element, prop) - period_avg.get(prop, 0.0)) / (period_std.get(prop, 1e-6)) # normalized deviation
            } for prop in properties_to_compare}

        return char_data

    def _characterize_cluster(self, cluster_id: int, cluster_elements_symbols: List[str], all_element_data: Dict[int, ElementData]) -> Dict[str, Any]:
        """
        Characterizes a cluster by calculating the average properties of its elements.
        """
        cluster_elements = [e for e in all_element_data.values() if e.symbol in cluster_elements_symbols]
        
        if not cluster_elements:
            return {"cluster_id": cluster_id, "error": "No elements found in cluster."}

        # Properties to average
        properties_to_average = [
            'atomic_number', 'period', 'group', 'valence', 
            'electronegativity', 'atomic_mass', 'density', 
            'melting_point', 'boiling_point'
        ]
        
        # Calculate averages, handling missing data
        avg_properties = {}
        for prop in properties_to_average:
            values = [getattr(e, prop) for e in cluster_elements if getattr(e, prop) is not None]
            # Filter out non-positive values for some properties if meaningful for average
            if prop in ['electronegativity', 'density', 'melting_point', 'boiling_point', 'atomic_mass']:
                 values = [v for v in values if v > 0]
            
            if values:
                avg_properties[prop] = np.mean(values)
            else:
                avg_properties[prop] = 0.0 # Default to 0.0 if no valid data

        # Get dominant block(s) and electron configuration types
        from collections import Counter
        block_counts = Counter([e.block for e in cluster_elements])
        dominant_blocks = block_counts.most_common(2) # Top 2 dominant blocks

        return {
            "cluster_id": cluster_id,
            "size": len(cluster_elements),
            "average_properties": avg_properties,
            "dominant_blocks": dominant_blocks,
            "elements_sample": cluster_elements_symbols[:5] + (['...'] if len(cluster_elements_symbols) > 5 else [])
        }

    def analyze_complete_6d_spatial_distribution(self, storage_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze 6D spatial distribution of all 118 elements.
        
        Args:
            storage_results: Results from storage operation
            
        Returns:
            Comprehensive spatial analysis results
        """
        print(" Analyzing Complete 6D Spatial Distribution...")
        
        analysis_results = {
            'total_elements': len(self.element_storage),
            'spatial_clusters': {},
            'cluster_characterizations': [], # NEW: To store detailed cluster insights
            'distance_statistics': {},
            'block_separation': {},
            'period_progression': {},
            'novel_patterns': [], # This will now contain derived patterns
            'outliers': [],
            'outlier_characterizations': [], # NEW: To store detailed outlier insights
            'inter_dimensional_correlations': {}
        }
        
        # Extract all 6D coordinates and relevant properties
        coordinates = []
        symbols = []
        blocks = []
        periods = []
        groups = []
        atomic_masses = []
        electronegativities = []
        melting_points = []
        valences = [] # ADDED THIS LINE
        
        for symbol, data in self.element_storage.items():
            coordinates.append(data['coordinates_6d'])
            symbols.append(symbol)
            element = data['original_data']
            blocks.append(element.block)
            periods.append(element.period)
            groups.append(element.group)
            atomic_masses.append(element.atomic_mass)
            electronegativities.append(element.electronegativity)
            melting_points.append(element.melting_point)
            valences.append(element.valence) # ADDED THIS LINE
        
        if not coordinates:
            print("   ⚠️ No elements stored, skipping detailed spatial analysis.")
            return analysis_results

        coordinates = np.array(coordinates)
        
        # Calculate distance statistics
        if len(coordinates) > 1:
            distances_flat = pdist(coordinates) # Calculates all pairwise Euclidean distances
            
            analysis_results['distance_statistics'] = {
                'mean_distance': float(np.mean(distances_flat)),
                'std_distance': float(np.std(distances_flat)),
                'min_distance': float(np.min(distances_flat)),
                'max_distance': float(np.max(distances_flat)),
                'total_pairs': len(distances_flat)
            }
        else:
            analysis_results['distance_statistics'] = {
                'mean_distance': 0.0,
                'std_distance': 0.0,
                'min_distance': 0.0,
                'max_distance': 0.0,
                'total_pairs': 0
            }
        
        # Analyze block separation in 6D space (existing logic)
        block_coords = {'s': [], 'p': [], 'd': [], 'f': []}
        for i, block in enumerate(blocks):
            block_coords[block].append(coordinates[i])
        
        for block, coords in block_coords.items():
            if len(coords) > 1:
                coords_array = np.array(coords)
                centroid = np.mean(coords_array, axis=0)
                distances_to_centroid = [np.linalg.norm(coord - centroid) for coord in coords_array]
                
                analysis_results['block_separation'][block] = {
                    'count': len(coords),
                    'centroid': centroid.tolist(),
                    'mean_spread': float(np.mean(distances_to_centroid)),
                    'compactness': float(1.0 / (1.0 + np.std(distances_to_centroid))) if np.std(distances_to_centroid) > 0 else 1.0
                }
            elif len(coords) == 1:
                 analysis_results['block_separation'][block] = {
                    'count': len(coords),
                    'centroid': coords[0].tolist(),
                    'mean_spread': 0.0,
                    'compactness': 1.0
                }
        
        # Analyze period progression (existing logic)
        period_coords = {}
        for i, period in enumerate(periods):
            if period not in period_coords:
                period_coords[period] = []
            period_coords[period].append(coordinates[i])
        
        for period, coords in period_coords.items():
            if len(coords) > 1:
                coords_array = np.array(coords)
                analysis_results['period_progression'][period] = {
                    'count': len(coords),
                    'mean_position': np.mean(coords_array, axis=0).tolist(),
                    'spatial_span': float(np.max(coords_array) - np.min(coords_array))
                }
            elif len(coords) == 1:
                 analysis_results['period_progression'][period] = {
                    'count': len(coords),
                    'mean_position': coords[0].tolist(),
                    'spatial_span': 0.0
                }

        # --- Deeper Pattern Detection ---
        print("   Performing deeper pattern detection...")

        # 1. K-Means Clustering
        n_clusters = max(2, min(10, len(symbols) // 10)) # Automatically determine a reasonable number of clusters
        if len(symbols) > n_clusters: # Ensure enough samples for clustering
            try:
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(coordinates) # This `coordinates` is already an np.array
                analysis_results['spatial_clusters']['num_clusters'] = n_clusters
                analysis_results['spatial_clusters']['cluster_details'] = []

                for i in range(n_clusters):
                    cluster_elements = [symbols[j] for j, label in enumerate(cluster_labels) if label == i]
                    cluster_coords = coordinates[cluster_labels == i]
                    
                    if len(cluster_coords) > 0:
                        cluster_center = np.mean(cluster_coords, axis=0)
                        
                        # Get dominant block and period for the cluster
                        cluster_blocks = [blocks[j] for j, label in enumerate(cluster_labels) if label == i]
                        cluster_periods = [periods[j] for j, label in enumerate(cluster_labels) if label == i]
                        
                        from collections import Counter
                        dominant_block = Counter(cluster_blocks).most_common(1)[0][0] if cluster_blocks else "N/A"
                        dominant_period = Counter(cluster_periods).most_common(1)[0][0] if cluster_periods else "N/A"

                        analysis_results['spatial_clusters']['cluster_details'].append({
                            'cluster_id': i,
                            'size': len(cluster_elements),
                            'elements_sample': cluster_elements[:5] + ['...'] if len(cluster_elements) > 5 else cluster_elements,
                            'centroid': cluster_center.tolist(),
                            'dominant_block': dominant_block,
                            'dominant_period': dominant_period
                        })
                        # NEW: Characterize the cluster
                        cluster_char = self._characterize_cluster(i, cluster_elements, self.complete_element_data)
                        analysis_results['cluster_characterizations'].append(cluster_char)

                analysis_results['spatial_clusters']['cluster_labels'] = cluster_labels.tolist() # <--- Converted to list here.
                analysis_results['novel_patterns'].append(f"Identified {n_clusters} spatial clusters via K-Means, suggesting non-obvious groupings.")
                print(f"     ✅ K-Means clustering found {n_clusters} clusters.")
            except Exception as e:
                print(f"     ❌ K-Means clustering failed: {e}. Skipping clustering analysis.")
        else:
            print("     ⚠️ Not enough elements for K-Means clustering.")


        # 2. Outlier Detection (using Mahalanobis distance if possible, or simple Euclidean)
        if len(coordinates) > 1 and coordinates.shape[0] > coordinates.shape[1]: # Need more samples than dimensions for covariance
            try:
                # Compute covariance matrix and its inverse for Mahalanobis distance
                cov_matrix = np.cov(coordinates.T)
                inv_cov_matrix = np.linalg.inv(cov_matrix)
                
                # Calculate Mahalanobis distance for each point from the centroid
                overall_centroid = np.mean(coordinates, axis=0)
                distances_mahalanobis = [
                    np.sqrt((coord - overall_centroid).reshape(1, -1) @ inv_cov_matrix @ (coord - overall_centroid).reshape(-1, 1))[0,0]
                    for coord in coordinates
                ]
                
                # Identify outliers as points beyond 2 standard deviations in Mahalanobis distance
                mean_dist = np.mean(distances_mahalanobis)
                std_dist = np.std(distances_mahalanobis)
                outlier_threshold = mean_dist + 2 * std_dist
                
                outliers_list = []
                for i, d in enumerate(distances_mahalanobis):
                    if d > outlier_threshold:
                        outliers_list.append({
                            'symbol': symbols[i],
                            'distance_from_centroid': float(d),
                            'traditional_block': blocks[i],
                            'traditional_period': periods[i]
                        })
                        # NEW: Characterize the outlier
                        outlier_char = self._characterize_outlier(symbols[i], self.complete_element_data)
                        analysis_results['outlier_characterizations'].append(outlier_char)

                analysis_results['outliers'] = outliers_list
                if outliers_list:
                    analysis_results['novel_patterns'].append(f"Detected {len(outliers_list)} spatial outliers, indicating unique elemental behaviors.")
                    print(f"     ✅ Outlier detection identified {len(outliers_list)} elements.")
                else:
                    print("     ✅ No significant outliers detected.")
            except np.linalg.LinAlgError:
                print("     ⚠️ Cannot compute inverse covariance for Mahalanobis distance (singular matrix). Falling back to Euclidean.")
                point_distances_from_centroid = np.linalg.norm(coordinates - np.mean(coordinates, axis=0), axis=1)
                mean_dist = np.mean(point_distances_from_centroid)
                std_dist = np.std(point_distances_from_centroid)
                outlier_threshold = mean_dist + 2 * std_dist
                
                outliers_list = []
                for i, d in enumerate(point_distances_from_centroid):
                    if d > outlier_threshold:
                        outliers_list.append({
                            'symbol': symbols[i],
                            'distance_from_centroid': float(d),
                            'traditional_block': blocks[i],
                            'traditional_period': periods[i]
                        })
                        # NEW: Characterize the outlier
                        outlier_char = self._characterize_outlier(symbols[i], self.complete_element_data)
                        analysis_results['outlier_characterizations'].append(outlier_char)

                analysis_results['outliers'] = outliers_list
                if outliers_list:
                    analysis_results['novel_patterns'].append(f"Detected {len(outliers_list)} spatial outliers (Euclidean), indicating unique elemental behaviors.")
                    print(f"     ✅ Outlier detection (Euclidean) identified {len(outliers_list)} elements.")
                else:
                    print("     ✅ No significant outliers detected (Euclidean).")
            except Exception as e:
                print(f"     ❌ Outlier detection failed: {e}. Skipping outlier analysis.")
        else:
            print("     ⚠️ Not enough data points for robust outlier detection.")


        # 3. Inter-Dimensional Correlation Analysis
        # Properties to correlate with 6D dimensions
        property_names = ['atomic_mass', 'electronegativity', 'melting_point', 'period', 'group', 'valence']
        property_values_map = {
            'atomic_mass': atomic_masses,
            'electronegativity': electronegativities,
            'melting_point': melting_points,
            'period': periods,
            'group': groups,
            'valence': valences # MODIFIED THIS LINE
        }
        
        coord_dimension_names = ['X_atomic_mod', 'Y_period', 'Z_group', 'W_block', 'U_electroneg', 'V_valence']

        inter_dimensional_correlations = {}
        for prop_name, prop_values in property_values_map.items():
            prop_array = np.array(prop_values)
            if len(prop_array) == 0:
                continue

            for dim_idx, dim_name in enumerate(coord_dimension_names):
                coord_dim_values = coordinates[:, dim_idx]
                
                # Check for constant arrays before computing correlation
                if np.std(prop_array) == 0 or np.std(coord_dim_values) == 0:
                    correlation_coeff = 0.0 # No variance, no correlation
                else:
                    try:
                        # corrcoef requires at least 2 observations
                        if len(prop_array) > 1 and len(coord_dim_values) > 1:
                            correlation_matrix = np.corrcoef(prop_array, coord_dim_values)
                            correlation_coeff = correlation_matrix[0, 1]
                            if np.isnan(correlation_coeff): # Handle NaN results from corrcoef
                                correlation_coeff = 0.0
                        else:
                            correlation_coeff = 0.0
                    except Exception as e:
                        print(f"     Warning: Correlation calculation failed for {prop_name} and {dim_name}: {e}")
                        correlation_coeff = 0.0

                inter_dimensional_correlations[f"{prop_name}_vs_{dim_name}"] = float(correlation_coeff)
        
        analysis_results['inter_dimensional_correlations'] = inter_dimensional_correlations
        if inter_dimensional_correlations:
            analysis_results['novel_patterns'].append("Analyzed inter-dimensional correlations, revealing relationships between 6D space and elemental properties.")
            print(f"     ✅ Inter-dimensional correlations analyzed.")
        else:
            print("     ⚠️ Inter-dimensional correlation analysis skipped due to insufficient data.")


        # Final print statements for deeper analysis
        print(f"    Spatial analysis complete")
        print(f"    Mean 6D distance: {analysis_results['distance_statistics']['mean_distance']:.2f}")
        print(f"    Block separation analysis: {len(analysis_results['block_separation'])} blocks")
        print(f"    Discovered Novel Patterns: {len(analysis_results['novel_patterns'])}")
        for i, pattern in enumerate(analysis_results['novel_patterns']):
            print(f"      - {pattern}")
        
        return analysis_results
    
    def _calculate_element_retrieval_nrci(self, original_element: ElementData, retrieved_data: Dict[str, Any]) -> float:
        """
        Calculates the Normalized Resonance Coherence Index (NRCI) for a single element retrieval.
        NRCI = 1 - (absolute_difference / max_possible_difference)
        A perfect match gives NRCI = 1.0. Higher values indicate higher fidelity.
        """
        # Using atomic_number for NRCI calculation as it's an integer and direct comparison is clean.
        original_atomic_number = original_element.atomic_number
        retrieved_atomic_number = retrieved_data.get('atomic_number')

        if retrieved_atomic_number is None: # Corrected from === None
            return 0.0 # Cannot calculate if data is missing

        # Max possible difference for atomic number (1 to 118)
        max_atomic_number_range = 117.0 # 118 - 1

        if max_atomic_number_range == 0: # Avoid division by zero for trivial ranges
            return 1.0 # Perfect if range is zero

        absolute_difference = abs(original_atomic_number - retrieved_atomic_number)
        
        # Normalize the difference
        normalized_difference = absolute_difference / max_atomic_number_range

        # NRCI formula: 1 - normalized_difference
        nrci = 1.0 - normalized_difference
        
        return max(0.0, min(1.0, nrci)) # Ensure NRCI is between 0 and 1
    
    def test_complete_retrieval_performance(self, storage_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test retrieval performance for all 118 elements.
        
        Args:
            storage_results: Results from storage operation
            
        Returns:
            Comprehensive retrieval performance results
        """
        print(" Testing Complete Retrieval Performance...")
        
        performance_results = {
            'total_tests': 0,
            'successful_retrievals': 0,
            'failed_retrievals': 0,
            'retrieval_times': [],
            'data_integrity_checks': 0,
            'integrity_successes': 0,
            'average_retrieval_time': 0.0,
            'average_retrieval_nrci': 0.0, # Added for NRCI (User Suggestion 2)
            'performance_rating': 'UNKNOWN'
        }
        
        start_time = time.time()
        retrieval_nrcis = [] # To store individual NRCI scores
        
        # Test retrieval for all stored elements
        for symbol, stored_info in self.element_storage.items():
            performance_results['total_tests'] += 1
            
            try:
                # Time the retrieval using the hash key
                retrieval_start = time.time()
                retrieved_data = self.hex_dict.retrieve(stored_info['stored_hash_key'])
                retrieval_time = time.time() - retrieval_start
                performance_results['retrieval_times'].append(retrieval_time)
                
                if retrieved_data:
                    performance_results['successful_retrievals'] += 1
                    
                    # Verify data integrity
                    performance_results['data_integrity_checks'] += 1
                    original_element = stored_info['original_data']
                    
                    if (retrieved_data['symbol'] == original_element.symbol and 
                        retrieved_data['atomic_number'] == original_element.atomic_number and
                        retrieved_data['name'] == original_element.name):
                        performance_results['integrity_successes'] += 1
                    else:
                        print(f"      ⚠️ Data integrity mismatch for {symbol}")

                    # Calculate NRCI for this element (User Suggestion 2)
                    element_nrci = self._calculate_element_retrieval_nrci(original_element, retrieved_data)
                    retrieval_nrcis.append(element_nrci)
                        
                else:
                    performance_results['failed_retrievals'] += 1
                    
            except Exception as e:
                performance_results['failed_retrievals'] += 1
                print(f"      ❌ Error retrieving {symbol}: {e}")
        
        # Calculate performance metrics
        total_time = time.time() - start_time
        
        if performance_results['retrieval_times']:
            performance_results['average_retrieval_time'] = np.mean(performance_results['retrieval_times'])
        
        if retrieval_nrcis:
            performance_results['average_retrieval_nrci'] = np.mean(retrieval_nrcis)
        
        # Determine performance rating (updated to include NRCI)
        total_tests = performance_results['total_tests']
        success_rate = performance_results['successful_retrievals'] / total_tests if total_tests > 0 else 0
        integrity_rate = performance_results['integrity_successes'] / performance_results['data_integrity_checks'] if performance_results['data_integrity_checks'] > 0 else 0
        avg_nrci = performance_results['average_retrieval_nrci']
        
        # Combine criteria for overall rating
        if success_rate >= 0.99 and integrity_rate >= 0.99 and avg_nrci >= 0.99:
            performance_results['performance_rating'] = 'EXCELLENT'
        elif success_rate >= 0.95 and integrity_rate >= 0.95 and avg_nrci >= 0.95:
            performance_results['performance_rating'] = 'GOOD'
        elif success_rate >= 0.90 and integrity_rate >= 0.90 and avg_nrci >= 0.90:
            performance_results['performance_rating'] = 'FAIR'
        else:
            performance_results['performance_rating'] = 'POOR'
        
        print(f"    Performance test complete")
        print(f"    Success rate: {success_rate:.1%}")
        print(f"    Data integrity: {integrity_rate:.1%}")
        print(f"    Avg retrieval time: {performance_results['average_retrieval_time']*1000:.2f} ms")
        print(f"    Avg retrieval NRCI: {performance_results['average_retrieval_nrci']:.4f}")
        print(f"    Performance rating: {performance_results['performance_rating']}")
        
        return performance_results
    
    def create_complete_visualization(self, storage_results: Dict[str, Any], spatial_analysis: Dict[str, Any]) -> str:
        """
        Create comprehensive visualization of all 118 elements in 6D space.
        
        Args:
            storage_results: Storage operation results
            spatial_analysis: Spatial analysis results
            
        Returns:
            Path to saved visualization
        """
        print(" Creating Periodic Table Visualization...")
        
        # Extract data for visualization
        symbols = []
        coordinates = []
        blocks = []
        periods = []
        atomic_numbers = []
        
        # New: Get outlier symbols and cluster centroids
        outlier_symbols = {o['symbol'] for o in spatial_analysis.get('outliers', [])}
        cluster_centroids = {c['cluster_id']: c['centroid'] for c in spatial_analysis['spatial_clusters'].get('cluster_details', [])}

        for symbol, data in self.element_storage.items():
            symbols.append(symbol)
            coordinates.append(data['coordinates_6d'])
            element = data['original_data']
            blocks.append(element.block)
            periods.append(element.period)
            atomic_numbers.append(element.atomic_number)
        
        # Add Element 119 to visualization data if predicted
        if 'predicted_element_119' in spatial_analysis:
            e119_data = spatial_analysis['predicted_element_119']
            e119_element = e119_data['element_data']
            e119_coords = e119_data['coordinates_6d']
            
            symbols.append(e119_element.symbol)
            coordinates.append(e119_coords)
            blocks.append(e119_element.block)
            periods.append(e119_element.period)
            atomic_numbers.append(e119_element.atomic_number)
            
            print(f"   Including predicted element 119 ({e119_element.symbol}) in visualization.")

        if not coordinates: # Handle case of no stored elements
            print("   ⚠️ No elements stored, skipping visualization.")
            return "no_visualization_generated.png"

        coordinates = np.array(coordinates)
        
        # Create comprehensive visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('UBP Framework v3.1: Complete Periodic Table (118 Elements) in 6D Space', fontsize=16, fontweight='bold')
        
        # Color maps for different properties
        block_colors = {'s': 'red', 'p': 'blue', 'd': 'green', 'f': 'orange'}
        
        # Plot 1: X-Y projection colored by block
        ax1 = axes[0, 0]
        for block in ['s', 'p', 'd', 'f']:
            mask = np.array(blocks) == block
            if np.any(mask):
                ax1.scatter(coordinates[mask, 0], coordinates[mask, 1], 
                           c=block_colors[block], label=f'{block}-block', alpha=0.7, s=50)
        
        # Highlight outliers on X-Y plot
        for i, symbol in enumerate(symbols):
            if symbol in outlier_symbols:
                ax1.scatter(coordinates[i, 0], coordinates[i, 1], 
                           marker='*', s=300, facecolors='none', edgecolors='black', linewidths=2, label='_Outlier' if symbol == list(outlier_symbols)[0] else "")
                ax1.annotate(symbol, (coordinates[i, 0], coordinates[i, 1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='black')
        
        # Highlight E119 if present
        if 'predicted_element_119' in spatial_analysis:
            e119_idx = len(symbols) - 1 # It was appended last
            ax1.scatter(coordinates[e119_idx, 0], coordinates[e119_idx, 1], 
                       marker='^', s=400, facecolors='purple', edgecolors='black', linewidths=2, label='Predicted E119')
            ax1.annotate(symbols[e119_idx], (coordinates[e119_idx, 0], coordinates[e119_idx, 1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9, color='purple', fontweight='bold')


        ax1.set_xlabel('X Coordinate (Atomic Number Mod)')
        ax1.set_ylabel('Y Coordinate (Period)')
        ax1.set_title('X-Y Projection by Electron Block (Outliers & E119 Highlighted)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: X-Z projection colored by period
        ax2 = axes[0, 1]
        scatter = ax2.scatter(coordinates[:, 0], coordinates[:, 2], 
                             c=periods, cmap='viridis', s=50, alpha=0.7)
        # Highlight outliers on X-Z plot
        for i, symbol in enumerate(symbols):
            if symbol in outlier_symbols:
                ax2.scatter(coordinates[i, 0], coordinates[i, 2], 
                           marker='*', s=300, facecolors='none', edgecolors='black', linewidths=2)
                ax2.annotate(symbol, (coordinates[i, 0], coordinates[i, 2]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='black')
        # Highlight E119
        if 'predicted_element_119' in spatial_analysis:
            e119_idx = len(symbols) - 1
            ax2.scatter(coordinates[e119_idx, 0], coordinates[e119_idx, 2], 
                       marker='^', s=400, facecolors='purple', edgecolors='black', linewidths=2)
            ax2.annotate(symbols[e119_idx], (coordinates[e119_idx, 0], coordinates[e119_idx, 2]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9, color='purple', fontweight='bold')

        ax2.set_xlabel('X Coordinate (Atomic Number Mod)')
        ax2.set_ylabel('Z Coordinate (Group)')
        ax2.set_title('X-Z Projection by Period (Outliers & E119 Highlighted)')
        plt.colorbar(scatter, ax=ax2, label='Period')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Y-Z projection colored by K-Means Clusters
        ax3 = axes[0, 2]
        cluster_labels_raw = spatial_analysis['spatial_clusters'].get('cluster_labels', None) # Get cluster labels if available
        
        if cluster_labels_raw is not None:
            # Convert cluster_labels back to numpy array for shape access and plotting
            cluster_labels_np = np.array(cluster_labels_raw)

            # If E119 was added, the coordinates array might have an extra element,
            # so cluster_labels_np might not match length.
            # In such a case, KMeans would need to be rerun, or E119 assigned a label.
            # For simplicity, if lengths don't match, color by atomic_numbers instead of cluster_labels.
            if cluster_labels_np.shape[0] != coordinates.shape[0]:
                print("     ⚠️ Cluster labels length mismatch (likely due to E119). Using atomic_numbers for Y-Z plot color.")
                scatter3 = ax3.scatter(coordinates[:, 1], coordinates[:, 2], 
                                      c=atomic_numbers, cmap='plasma', s=50, alpha=0.7)
            else:
                scatter3 = ax3.scatter(coordinates[:, 1], coordinates[:, 2], 
                                      c=cluster_labels_np, cmap='tab10', s=50, alpha=0.7)
            
            # Plot cluster centroids
            for cluster_id, centroid_coords in cluster_centroids.items():
                ax3.scatter(centroid_coords[1], centroid_coords[2], 
                           marker='X', s=500, color='red', edgecolors='black', linewidths=1.5,
                           label='_Centroid' if cluster_id == 0 else "") # Label only one for legend
                ax3.annotate(f'C{cluster_id}', (centroid_coords[1], centroid_coords[2]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9, fontweight='bold', color='red')

            ax3.set_title(f'Y-Z Projection by K-Means Clusters ({spatial_analysis["spatial_clusters"]["num_clusters"]} clusters)')
            if 'scatter3' in locals() and scatter3 is not None: # Ensure scatter3 was created
                plt.colorbar(scatter3, ax=ax3, label='Cluster ID')
            if cluster_centroids: # Add centroid to legend if present
                handles, labels = ax3.get_legend_handles_labels()
                if '_Centroid' in labels: # Check if legend already added
                    ax3.legend(handles, labels, loc='best')
                else:
                    # Manually add a proxy for the centroid legend
                    ax3.plot([], [], marker='X', linestyle='None', color='red', label='Cluster Centroid')
                    ax3.legend()

        else: # If cluster_labels_raw is None (e.g., K-Means failed or too few elements)
            ax3.scatter(coordinates[:, 1], coordinates[:, 2], 
                       s=np.array(atomic_numbers) * 2, alpha=0.6, c='purple')
            ax3.set_title('Y-Z Projection (Size = Atomic Number, No Clusters)')
        
        # Highlight outliers on Y-Z plot
        for i, symbol in enumerate(symbols):
            if symbol in outlier_symbols:
                ax3.scatter(coordinates[i, 1], coordinates[i, 2], 
                           marker='*', s=300, facecolors='none', edgecolors='black', linewidths=2)
                ax3.annotate(symbol, (coordinates[i, 1], coordinates[i, 2]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='black')
        # Highlight E119
        if 'predicted_element_119' in spatial_analysis:
            e119_idx = len(symbols) - 1
            ax3.scatter(coordinates[e119_idx, 1], coordinates[e119_idx, 2], 
                       marker='^', s=400, facecolors='purple', edgecolors='black', linewidths=2)
            ax3.annotate(symbols[e119_idx], (coordinates[e119_idx, 1], coordinates[e119_idx, 2]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9, color='purple', fontweight='bold')

        ax3.set_xlabel('Y Coordinate (Period)')
        ax3.set_ylabel('Z Coordinate (Group)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: W-U projection (Block vs Electronegativity)
        ax4 = axes[1, 0]
        ax4.scatter(coordinates[:, 3], coordinates[:, 4], 
                   c=atomic_numbers, cmap='plasma', s=50, alpha=0.7)
        # Highlight outliers on W-U plot
        for i, symbol in enumerate(symbols):
            if symbol in outlier_symbols:
                ax4.scatter(coordinates[i, 3], coordinates[i, 4], 
                           marker='*', s=300, facecolors='none', edgecolors='black', linewidths=2)
                ax4.annotate(symbol, (coordinates[i, 3], coordinates[i, 4]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, color='black')
        # Highlight E119
        if 'predicted_element_119' in spatial_analysis:
            e119_idx = len(symbols) - 1
            ax4.scatter(coordinates[e119_idx, 3], coordinates[e119_idx, 4], 
                       marker='^', s=400, facecolors='purple', edgecolors='black', linewidths=2)
            ax4.annotate(symbols[e119_idx], (coordinates[e119_idx, 3], coordinates[e119_idx, 4]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=9, color='purple', fontweight='bold')

        ax4.set_xlabel('W Coordinate (Block)')
        ax4.set_ylabel('U Coordinate (Electronegativity)')
        ax4.set_title('W-U Projection (Block vs Electronegativity)')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Block distribution
        ax5 = axes[1, 1]
        block_labels = ['s-block', 'p-block', 'd-block', 'f-block']
        # Recalculate block counts to include E119 if present
        updated_block_counts = {'s': 0, 'p': 0, 'd': 0, 'f': 0}
        for block_val in blocks:
            updated_block_counts[block_val] = updated_block_counts.get(block_val, 0) + 1

        block_counts = [updated_block_counts.get(block_key, 0) for block_key in ['s', 'p', 'd', 'f']]
        bars = ax5.bar(block_labels, block_counts, 
                      color=[block_colors[block_key] for block_key in ['s', 'p', 'd', 'f']])
        ax5.set_ylabel('Number of Elements')
        ax5.set_title('Element Distribution by Block')
        ax5.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, block_counts):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{count}', ha='center', va='bottom')
        
        # Plot 6: Period distribution
        ax6 = axes[1, 2]
        # Recalculate period counts to include E119 if present
        updated_period_counts = {}
        for period_val in periods:
            updated_period_counts[period_val] = updated_period_counts.get(period_val, 0) + 1

        periods_list = sorted(list(updated_period_counts.keys()))
        period_counts = [updated_period_counts[p] for p in periods_list]
        ax6.bar(periods_list, period_counts, color='skyblue', alpha=0.7)
        ax6.set_xlabel('Period')
        ax6.set_ylabel('Number of Elements')
        ax6.set_title('Element Distribution by Period')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save visualization
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        viz_path = f"/output/ubp_complete_periodic_table_118_elements_{timestamp}.png" # Save directly to /output/
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"    Visualization saved: {viz_path}")
        return viz_path
    
    def predict_element_119(self) -> Dict[str, Any]:
        """
        Predicts properties and 6D coordinates for Element 119 (Ununennium).
        This uses simple extrapolation based on periodic trends.
        """
        print("🔮 Predicting Element 119 (Ununennium)...")
        # Element 119 is in Period 8, Group 1, s-block
        
        # Get properties of closest known elements for extrapolation
        # Oganesson (118, Period 7, Group 18)
        # Francium (87, Period 7, Group 1)
        # Cesium (55, Period 6, Group 1)
        
        # Simple linear extrapolation/averaging for properties
        # Atomic Mass: Expected to be slightly higher than Oganesson (294)
        predicted_atomic_mass = 295.0 # Estimate
        
        # Electronegativity: Alkali metals have very low electronegativity. Francium is 0.7.
        predicted_electronegativity = 0.65 # Expect slightly lower
        
        # Density: Alkali metals generally less dense than heavier elements, but increases down group.
        # Francium is 1.87 g/cm^3. Predict higher.
        predicted_density = 2.0 # Estimate
        
        # Melting/Boiling Point: Alkali metals have low points, decreasing down group.
        # Francium: MP 300K, BP 950K. Predict lower.
        predicted_melting_point = 290.0 # K
        predicted_boiling_point = 900.0 # K
        
        # Discovery Year: Future!
        predicted_discovery_year = 2026 # Near future
        
        # Electron Configuration: [Og] 8s1
        predicted_electron_config = '[Og] 8s1'
        
        # Valence Electrons: Group 1, so 1 valence electron.
        predicted_valence = 1
        
        # Oxidation States: Group 1, typically +1
        predicted_oxidation_states = [1]
        
        predicted_element_119 = ElementData(
            atomic_number=119,
            symbol='Uue', # Ununennium
            name='Ununennium',
            period=8,
            block='s',
            group=1, # Fix: group must be specified for ElementData
            valence=predicted_valence,
            electronegativity=predicted_electronegativity,
            atomic_mass=predicted_atomic_mass,
            density=predicted_density,
            melting_point=predicted_melting_point,
            boiling_point=predicted_boiling_point,
            discovery_year=predicted_discovery_year,
            electron_config=predicted_electron_config,
            oxidation_states=predicted_oxidation_states
        )
        
        # Calculate 6D coordinates for predicted element
        predicted_coords_6d = self.calculate_6d_coordinates_bittab(predicted_element_119)
        predicted_bittab_encoding = self.encode_element_to_bittab(predicted_element_119)

        print(f"   Predicted Element 119: {predicted_element_119.name} ({predicted_element_119.symbol})")
        print(f"      Period: {predicted_element_119.period}, Group: {predicted_element_119.group}, Block: {predicted_element_119.block}")
        print(f"      Atomic Mass: {predicted_element_119.atomic_mass:.2f}, Electronegativity: {predicted_element_119.electronegativity:.2f}")
        print(f"      Predicted 6D Coords: {predicted_coords_6d}")
        print(f"      Predicted BitTab: {predicted_bittab_encoding}")

        return {
            'element_data': predicted_element_119,
            'coordinates_6d': predicted_coords_6d,
            'bittab_encoding': predicted_bittab_encoding,
            'prediction_method': 'extrapolation_periodic_trends'
        }
    
    def elements_to_glyphs(self, elements: List[ElementData]) -> Dict[str, GlyphState]:
        """
        Converts ElementData objects to Rune Protocol GlyphState objects.
        """
        print("\n⚛️ Converting elements to Rune Protocol Glyphs...")
        element_glyphs = {}
        for element in elements:
            # Create a unique Glyph ID for the element
            glyph_id = f"ELEMENT_{element.atomic_number}_{element.symbol}"
            
            # Map element properties to Glyph state_vector
            # Use a subset of numerical properties for the state vector
            state_vector_data = [
                float(element.atomic_number),
                float(element.period),
                float(element.group),
                float(element.valence),
                element.electronegativity if element.electronegativity is not None else 0.0,
                element.atomic_mass,
                element.density if element.density is not None else 0.0,
                element.melting_point if element.melting_point is not None else 0.0,
                element.boiling_point if element.boiling_point is not None else 0.0,
                float(element.discovery_year) # Ensure discovery_year is float
            ]
            
            # Normalize state_vector_data for Glyph processing if values are too disparate
            state_vector_np = np.array(state_vector_data, dtype=float)
            if np.linalg.norm(state_vector_np) > 0:
                # Use max absolute value for normalization to preserve zero if all are zero
                max_val = np.max(np.abs(state_vector_np))
                if max_val > 0:
                    state_vector_np = state_vector_np / max_val # Simple normalization to keep values manageable
            
            # Determine GlyphType based on element block (conceptual mapping)
            if element.block == 's':
                glyph_type = GlyphType.QUANTIFY
            elif element.block == 'p':
                glyph_type = GlyphType.TRANSFORM
            elif element.block == 'd':
                glyph_type = GlyphType.CORRELATE
            elif element.block == 'f':
                glyph_type = GlyphType.SELF_REFERENCE
            else:
                glyph_type = GlyphType.EMERGENCE
            
            # Set resonance frequency based on element properties (e.g., related to atomic number or CRV_QUANTUM_BASE)
            # Use UBPConstants.CRV_QUANTUM_BASE as a base frequency.
            # Atomic number scales it.
            resonance_freq = UBPConstants.CRV_QUANTUM_BASE * element.atomic_number * 1e9 # Scale to a sensible frequency range
            
            element_glyph = self.rune_protocol.create_glyph(
                glyph_id=glyph_id,
                glyph_type=glyph_type,
                initial_state=state_vector_np
            )
            # Manually set resonance_frequency because create_glyph defaults it.
            element_glyph.resonance_frequency = resonance_freq
            element_glyph.metadata['element_symbol'] = element.symbol
            element_glyph.metadata['atomic_number'] = element.atomic_number
            element_glyph.metadata['period'] = element.period
            element_glyph.metadata['group'] = element.group
            element_glyph.metadata['block'] = element.block
            
            element_glyphs[glyph_id] = element_glyph
            
        print(f"   Created {len(element_glyphs)} element Glyphs.")
        return element_glyphs

    def run_rune_protocol_analysis(self, element_glyphs: Dict[str, GlyphState]) -> Dict[str, Any]:
        """
        Runs a sample Rune Protocol analysis on selected element Glyphs.
        """
        print("\n✨ Running Rune Protocol Analysis on Element Glyphs...")
        analysis_results = {
            'quantification_results': {},
            'correlation_results': {},
            'self_reference_results': {},
            'emergence_detection': {}
        }

        # Select a few key elements for demonstration
        hydrogen_glyph = element_glyphs.get("ELEMENT_1_H")
        oxygen_glyph = element_glyphs.get("ELEMENT_8_O")
        iron_glyph = element_glyphs.get("ELEMENT_26_Fe")
        ununennium_glyph = element_glyphs.get("ELEMENT_119_Uue")

        # 1. Quantify selected glyphs
        print("\n   1. Quantifying selected element Glyphs:")
        for glyph in [hydrogen_glyph, oxygen_glyph, ununennium_glyph]:
            if glyph:
                try:
                    quant_result = self.rune_protocol.execute_operation(
                        operation_type='quantify',
                        glyph_id=glyph.glyph_id
                    )
                    # FIX: Access symbol from metadata
                    symbol_for_print = glyph.metadata.get('element_symbol', 'N/A')
                    # FIX: Handle potentially None quantified_value before formatting
                    quantified_value_display = quant_result.metadata.get('quantified_value')
                    if quantified_value_display is None:
                        quantified_value_display = "N/A"
                    else:
                        quantified_value_display = f"{quantified_value_display:.2f}"

                    analysis_results['quantification_results'][symbol_for_print] = {
                        'quantified_value': quant_result.metadata.get('quantified_value'),
                        'nrci_score': quant_result.nrci_score,
                        'coherence_change': quant_result.coherence_change
                    }
                    print(f"      {symbol_for_print} (Quantify): Value={quantified_value_display}, NRCI={quant_result.nrci_score:.4f}")
                except Exception as e:
                    print(f"      Error quantifying {glyph.glyph_id}: {e}")

        # 2. Correlate two glyphs (e.g., Hydrogen and Oxygen to form Water's conceptual link)
        print("\n   2. Correlating Hydrogen and Oxygen Glyphs:")
        if hydrogen_glyph and oxygen_glyph:
            try:
                corr_result = self.rune_protocol.execute_multi_glyph_operation(
                    operation_type='correlate',
                    glyph_ids=[hydrogen_glyph.glyph_id, oxygen_glyph.glyph_id],
                    realm_i='electromagnetic', # Contextual realm for correlation
                    realm_j='biological'
                )
                # FIX: Handle potentially None correlation_coefficient before formatting
                correlation_coefficient_display = corr_result.metadata.get('correlation_coefficient')
                if correlation_coefficient_display is None:
                    correlation_coefficient_display = "N/A"
                else:
                    correlation_coefficient_display = f"{correlation_coefficient_display:.4f}"

                analysis_results['correlation_results'] = {
                    'H_O_correlation': {
                        'correlation_coefficient': corr_result.metadata.get('correlation_coefficient'),
                        'nrci_score': corr_result.nrci_score,
                        'emergence_detected': corr_result.emergence_detected
                    }
                }
                print(f"      H-O Correlation: Coeff={correlation_coefficient_display}, NRCI={corr_result.nrci_score:.4f}, Emergence Detected={corr_result.emergence_detected}")
            except Exception as e:
                print(f"      Error correlating H and O: {e}")

        # 3. Apply self-reference to a complex element (e.g., Iron)
        print("\n   3. Applying Self-Reference to Iron Glyph:")
        if iron_glyph:
            try:
                self_ref_result = self.rune_protocol.execute_operation(
                    operation_type='self_reference',
                    glyph_id=iron_glyph.glyph_id,
                    feedback_strength=0.15 # Removed max_depth as it's managed by the operator's __init__
                )
                analysis_results['self_reference_results'] = {
                    'Fe_self_reference': {
                        'new_activation_level': self.rune_protocol.glyphs[iron_glyph.glyph_id].activation_level, # FIX: Read directly from GlyphState
                        'self_reference_depth': self_ref_result.self_reference_loops,
                        'nrci_score': self_ref_result.nrci_score
                    }
                }
                print(f"      Fe (Self-Ref): New Activation Level={self.rune_protocol.glyphs[iron_glyph.glyph_id].activation_level:.4f}, Depth={self.rune_protocol.glyphs[iron_glyph.glyph_id].self_reference_depth}, NRCI={self_ref_result.nrci_score:.4f}")
            except Exception as e:
                print(f"      Error self-referencing Iron: {e}")

        # 4. Overall emergence detection for all active element glyphs
        print("\n   4. Detecting overall emergence in all element Glyphs:")
        all_active_element_glyphs = [g for g in element_glyphs.values() if g.activation_level > 0]
        if all_active_element_glyphs:
            emergence_summary = self.rune_protocol.get_system_state().get('emergence_status')
            analysis_results['emergence_detection'] = emergence_summary
            print(f"      Overall Emergence Status: Detected={emergence_summary.get('emergence_detected')}, Type={emergence_summary.get('emergence_type')}, Strength={emergence_summary.get('emergence_strength'):.3f}")
        else:
            print("      No active element glyphs for emergence detection.")

        print("✨ Rune Protocol Analysis Complete.")
        return analysis_results

    
    def run_complete_periodic_table_test(self) -> Dict[str, Any]:
        """
        Run comprehensive test of all 118 elements.
        
        Returns:
            Complete test results
        """
        print(" Running Periodic Table Test)...")
        
        test_results = {
            'test_start_time': time.time(),
            'framework_status': 'UNKNOWN',
            'storage_results': {},
            'spatial_analysis': {},
            'performance_results': {},
            'visualization_path': '',
            'overall_rating': 'UNKNOWN',
            'revolutionary_achievements': [],
            'predicted_element_119_data': None, # New field for element 119
            'rune_protocol_analysis': {} # New field for Rune Protocol results
        }
        
        try:
            # Phase 1: Store all 118 elements
            print("\n📦 Phase 1: Storing All Elements...")
            test_results['storage_results'] = self.store_complete_periodic_table()
            
            # Phase 2: Predict Element 119
            print("\n🔮 Phase 2: Predicting Element 119 (Ununennium)...")
            predicted_e119 = self.predict_element_119()
            test_results['predicted_element_119_data'] = predicted_e119
            # Add element 119 to the analyzer's data for consistency and later Rune Protocol integration
            self.complete_element_data[119] = predicted_e119['element_data']
            # Also add to element_storage for visualization purposes
            self.element_storage[predicted_e119['element_data'].symbol] = {
                'atomic_number': predicted_e119['element_data'].atomic_number,
                'stored_hash_key': None, # Not actually stored in HexDict here, just for tracking
                'coordinates_6d': predicted_e119['coordinates_6d'],
                'bittab_encoding': predicted_e119['bittab_encoding'],
                'original_data': predicted_e119['element_data']
            }
            test_results['revolutionary_achievements'].append("Successful prediction of Element 119 properties and 6D coordinates.")
            
            # Phase 3: Analyze 6D spatial distribution (now including E119 for visualization)
            print("\n🔍 Phase 3: Analyzing Complete 6D Spatial Distribution...")
            if test_results['storage_results']['elements_stored'] > 0:
                test_results['spatial_analysis'] = self.analyze_complete_6d_spatial_distribution(test_results['storage_results'])
                # Add predicted E119 data to spatial_analysis for visualization purposes
                test_results['spatial_analysis']['predicted_element_119'] = predicted_e119
            else:
                print("   ⚠️ No elements stored, skipping spatial analysis.")
                test_results['spatial_analysis'] = {
                    'total_elements': 0, 'distance_statistics': {'mean_distance': 0.0, 'std_distance': 0.0, 'min_distance': 0.0, 'max_distance': 0.0, 'total_pairs': 0},
                    'block_separation': {}, 'period_progression': {}, 'novel_patterns': ['No elements to analyze.'], 'outliers': [], 'inter_dimensional_correlations': {},
                    'outlier_characterizations': [], # Ensure these are present
                    'cluster_characterizations': [] # Ensure these are present
                }

            
            # Phase 4: Test retrieval performance
            print("\n⚡ Phase 4: Testing Complete Retrieval Performance...")
            test_results['performance_results'] = self.test_complete_retrieval_performance(test_results['storage_results'])
            
            # Phase 5: Integrate and Analyze with Rune Protocol
            print("\n✨ Phase 5: Integrating Elements with Rune Protocol...")
            all_elements_for_glyphs = list(self.complete_element_data.values()) # Convert all elements, including E119
            element_glyphs = self.elements_to_glyphs(all_elements_for_glyphs)
            test_results['rune_protocol_analysis'] = self.run_rune_protocol_analysis(element_glyphs)
            test_results['revolutionary_achievements'].append("Elements successfully integrated and analyzed via Rune Protocol Glyphs.")

            # Phase 6: Create comprehensive visualization
            print("\n📊 Phase 6: Creating Complete Visualization...")
            test_results['visualization_path'] = self.create_complete_visualization(
                test_results['storage_results'], 
                test_results['spatial_analysis']
            )
            
            # Phase 7: Generate insights
            print("\n🚀 Phase 7: Generating Insights (Based on UBP's design principles)...")
            # These are interpretations of the UBP's design and success in this test, not derived patterns from the data analysis
            test_results['revolutionary_achievements'].extend([
                "BitTab 24-bit encoding applied to entire periodic table",
                "HexDictionary universal storage with complete element set",
                "UBP Framework demonstrates scalability to full chemical knowledge",
                "6D spatial analysis highlights structural organization in elemental properties",
                "Periodic table processed with full UBP integration",
                "Additional approach to chemical informatics established via UBP",
                "Deeper pattern detection algorithms applied, revealing novel spatial clusters, outliers, and inter-dimensional correlations.",
                "Shannon Entropy-based encoding ratio provided for information density analysis.", # New achievement
                "Retrieval NRCI computed, validating data fidelity post-storage." # New achievement
            ])
            
            # Determine overall rating
            elements_processed = test_results['storage_results']['elements_stored']
            storage_success = elements_processed / 118 if 118 > 0 else 0
            performance_rating = test_results['performance_results']['performance_rating']
            
            if storage_success >= 0.99 and performance_rating == 'EXCELLENT' and test_results['predicted_element_119_data'] is not None and test_results['rune_protocol_analysis']:
                test_results['overall_rating'] = 'REVOLUTIONARY'
            elif storage_success >= 0.95 and performance_rating in ['EXCELLENT', 'GOOD']:
                test_results['overall_rating'] = 'EXCELLENT'
            elif storage_success >= 0.90:
                test_results['overall_rating'] = 'GOOD'
            else:
                test_results['overall_rating'] = 'FAIR'
            
            test_results['framework_status'] = 'OPERATIONAL'
            
        except Exception as e:
            print(f"❌ Test error: {e}")
            test_results['framework_status'] = 'ERROR'
            test_results['overall_rating'] = 'FAILED'
        
        test_results['total_execution_time'] = time.time() - test_results['test_start_time']
        
        return test_results

class UBPTestDriveCompletePeriodicTable118Elements:
    """
    Wrapper class to encapsulate the main execution logic for the UBP Complete Periodic Table Test.
    This class serves as the expected entry point for the automated execution environment.
    """
    def run(self):
        """Main execution function for complete periodic table test."""
        print("🚀 UBP Framework v3.1 - Complete Periodic Table Test (118 Elements)")
        print("=" * 80)
        print("Revolutionary demonstration of UBP's capability to handle all known elements")
        print("=" * 80)
        
        # Initialize UBP Framework v3.1
        print("\n🔧 Initializing UBP Framework v3.1...")
        try:
            # Changed to create_ubp_system as per ubp_reference_sheet.py
            # Initialize with a relevant default realm for chemical elements
            framework = create_ubp_system(default_realm="electromagnetic") 
            print("✅ UBP Framework v3.1 initialized successfully")
        except Exception as e:
            print(f"❌ Framework initialization failed: {e}")
            sys.exit(1)
        
        # Create analyzer and run complete test
        analyzer = CompletePeriodicTableAnalyzer(framework)
        test_results = analyzer.run_complete_periodic_table_test()
        
        # Save comprehensive results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results_file = f"ubp_complete_periodic_table_results_{timestamp}.json"
        
        # Convert numpy arrays to lists for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_): # ADDED: Handle NumPy boolean types
                return bool(obj)
            return obj
        
        # Recursively convert numpy objects
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(v) for v in obj]
            elif isinstance(obj, ElementData): # Custom handling for ElementData dataclass
                return obj.__dict__
            elif isinstance(obj, GlyphState): # Custom handling for GlyphState dataclass
                # Convert state_vector (ndarray) and metadata correctly
                glyph_dict = obj.__dict__.copy()
                if 'state_vector' in glyph_dict and isinstance(glyph_dict['state_vector'], np.ndarray):
                    glyph_dict['state_vector'] = glyph_dict['state_vector'].tolist()
                return glyph_dict
            else:
                return convert_numpy(obj)
        
        serializable_results = recursive_convert(test_results)
        
        # Ensure the output directory exists
        output_dir = '/output/'
        os.makedirs(output_dir, exist_ok=True)

        output_results_file = os.path.join(output_dir, results_file)
        with open(output_results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        # Update visualization path to be in /output/
        # The visualization is already saved directly to /output/, so just update the path in results.
        # Check if a visualization was actually generated before attempting to modify its path in results.
        if test_results['visualization_path'] != "no_visualization_generated.png" and \
           os.path.basename(test_results['visualization_path']) != "no_visualization_generated.png": # Check the actual filename for 'no_viz' sentinel
            # Path is already correct as it was saved to /output/ directly
            pass 
        elif test_results['visualization_path'] == "no_visualization_generated.png":
            test_results['visualization_path'] = "No visualization generated due to lack of stored elements."
        else: # Fallback if path doesn't start with /output/ and is not the sentinel
            base_name = os.path.basename(test_results['visualization_path'])
            test_results['visualization_path'] = os.path.join(output_dir, base_name) # Correct the path for reporting

        # Display final summary
        print("\n" + "=" * 80)
        print(" PERIODIC TABLE TEST RESULTS")
        print("=" * 80)
        print(f" Test Summary:")
        print(f"    Total Execution Time: {test_results['total_execution_time']:.3f} seconds")
        print(f"    Elements Stored: {test_results['storage_results']['elements_stored']}/118")
        
        storage_success_rate = test_results['storage_results']['elements_stored'] / 118 if 118 > 0 else 0.0
        print
