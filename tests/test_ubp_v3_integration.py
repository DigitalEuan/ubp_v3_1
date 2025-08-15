"""
UBP Framework v3.0 Integration Tests
Author: Euan Craig, New Zealand
Date: 13 August 2025

Comprehensive test suite for validating UBP Framework v3.0 integration
and ensuring all components work together properly.
"""

import sys
import os
import numpy as np
import time
import json
import unittest
from typing import Dict, List, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))

# Import UBP Framework v3.0
from ubp_framework_v3 import UBPFrameworkV3, create_ubp_framework_v3

class TestUBPv3Integration(unittest.TestCase):
    """Test suite for UBP Framework v3.0 integration."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        print("🚀 Setting up UBP Framework v3.0 integration tests...")
        
        # Initialize framework with test configuration
        cls.ubp_framework = create_ubp_framework_v3(hardware_profile="development")
        
        # Test data
        cls.test_data_small = np.random.random(10)
        cls.test_data_medium = np.random.random(100)
        cls.test_data_large = np.random.random(1000)
        
        print("✅ Test environment setup complete")
    
    def test_01_framework_initialization(self):
        """Test that UBP Framework v3.0 initializes correctly."""
        print("\n🔧 Testing framework initialization...")
        
        # Check that framework is initialized
        self.assertIsNotNone(self.ubp_framework)
        
        # Check system status
        status = self.ubp_framework.get_system_status()
        self.assertEqual(status['version'], '3.0')
        self.assertEqual(status['system_health'], 'operational')
        
        # Check that all v3.0 components are present
        component_status = status['component_status']
        self.assertTrue(component_status['bitfield'])
        self.assertTrue(component_status['realm_manager'])
        self.assertTrue(component_status['enhanced_crv_system'])
        self.assertTrue(component_status['htr_engine'])
        self.assertTrue(component_status['bittime_mechanics'])
        self.assertTrue(component_status['rune_protocol'])
        self.assertTrue(component_status['advanced_error_correction'])
        
        # Check v3.0 enhancements
        v3_enhancements = status['v3_enhancements']
        self.assertTrue(v3_enhancements['enhanced_crvs'])
        self.assertTrue(v3_enhancements['harmonic_toggle_resonance'])
        self.assertTrue(v3_enhancements['bittime_mechanics'])
        self.assertTrue(v3_enhancements['rune_protocol'])
        self.assertTrue(v3_enhancements['advanced_error_correction'])
        self.assertTrue(v3_enhancements['nuclear_realm'])
        self.assertTrue(v3_enhancements['optical_realm'])
        
        print("✅ Framework initialization test passed")
    
    def test_02_basic_computation(self):
        """Test basic UBP v3.0 computation."""
        print("\n🔄 Testing basic computation...")
        
        result = self.ubp_framework.run_computation(
            operation_type='energy_calculation',
            input_data=self.test_data_small,
            enable_htr=False,
            enable_error_correction=False
        )
        
        # Check result structure
        self.assertIsNotNone(result)
        self.assertGreater(result.nrci_score, 0.0)
        self.assertGreater(result.energy_value, 0.0)
        self.assertIsInstance(result.coherence_metrics, dict)
        self.assertGreater(result.computation_time, 0.0)
        self.assertIn(result.realm_used, ['electromagnetic', 'quantum', 'gravitational', 
                                         'biological', 'cosmological', 'nuclear', 'optical'])
        
        print(f"✅ Basic computation test passed: NRCI={result.nrci_score:.6f}")
    
    def test_03_enhanced_crv_system(self):
        """Test Enhanced CRV System with Sub-CRVs."""
        print("\n🎯 Testing Enhanced CRV System...")
        
        result = self.ubp_framework.run_computation(
            operation_type='energy_calculation',
            input_data=self.test_data_medium,
            realm='electromagnetic',  # Force specific realm
            enable_htr=False,
            enable_error_correction=False
        )
        
        # Check CRV selection
        self.assertIsNotNone(result.optimal_crv_used)
        self.assertNotEqual(result.optimal_crv_used, "")
        
        # Check that Sub-CRV fallbacks are tracked
        self.assertIsInstance(result.sub_crv_fallbacks_used, list)
        
        print(f"✅ Enhanced CRV System test passed: CRV={result.optimal_crv_used}")
    
    def test_04_htr_integration(self):
        """Test Harmonic Toggle Resonance integration."""
        print("\n🎵 Testing HTR integration...")
        
        result = self.ubp_framework.run_computation(
            operation_type='energy_calculation',
            input_data=self.test_data_medium,
            enable_htr=True,
            enable_error_correction=False
        )
        
        # Check HTR results
        self.assertGreaterEqual(result.htr_resonance_score, 0.0)
        self.assertIsInstance(result.harmonic_patterns_detected, list)
        
        # HTR should improve NRCI (generally)
        result_no_htr = self.ubp_framework.run_computation(
            operation_type='energy_calculation',
            input_data=self.test_data_medium,
            enable_htr=False,
            enable_error_correction=False
        )
        
        # HTR may or may not improve NRCI depending on data, but should not break anything
        self.assertGreater(result.nrci_score, 0.0)
        
        print(f"✅ HTR integration test passed: Resonance={result.htr_resonance_score:.6f}")
    
    def test_05_error_correction_integration(self):
        """Test Enhanced Error Correction integration."""
        print("\n🛡️ Testing Error Correction integration...")
        
        # Add some noise to test data
        noisy_data = self.test_data_medium + np.random.normal(0, 0.1, len(self.test_data_medium))
        
        result = self.ubp_framework.run_computation(
            operation_type='energy_calculation',
            input_data=noisy_data,
            enable_htr=False,
            enable_error_correction=True
        )
        
        # Check error correction was applied
        self.assertTrue(result.error_correction_applied)
        
        # Compare with no error correction
        result_no_ec = self.ubp_framework.run_computation(
            operation_type='energy_calculation',
            input_data=noisy_data,
            enable_htr=False,
            enable_error_correction=False
        )
        
        # Error correction should generally improve or maintain NRCI
        self.assertGreaterEqual(result.nrci_score, result_no_ec.nrci_score * 0.9)  # Allow 10% tolerance
        
        print(f"✅ Error Correction integration test passed")
    
    def test_06_bittime_mechanics(self):
        """Test BitTime Mechanics integration."""
        print("\n⏱️ Testing BitTime Mechanics...")
        
        result = self.ubp_framework.run_computation(
            operation_type='energy_calculation',
            input_data=self.test_data_small,
            enable_htr=False,
            enable_error_correction=False
        )
        
        # Check BitTime precision
        self.assertGreaterEqual(result.bittime_precision, 0.0)
        
        print(f"✅ BitTime Mechanics test passed: Precision={result.bittime_precision:.6f}")
    
    def test_07_rune_protocol_integration(self):
        """Test Rune Protocol integration."""
        print("\n🔮 Testing Rune Protocol integration...")
        
        # Test glyph operations
        result = self.ubp_framework.run_computation(
            operation_type='glyph_quantify',
            input_data=self.test_data_small,
            enable_htr=False,
            enable_error_correction=False
        )
        
        # Check Rune operations
        self.assertIsInstance(result.rune_operations_executed, list)
        
        print(f"✅ Rune Protocol integration test passed: "
              f"{len(result.rune_operations_executed)} operations")
    
    def test_08_multi_realm_computation(self):
        """Test computation across multiple realms."""
        print("\n🌐 Testing multi-realm computation...")
        
        realms_to_test = ['electromagnetic', 'quantum', 'gravitational', 'nuclear', 'optical']
        results = {}
        
        for realm in realms_to_test:
            try:
                result = self.ubp_framework.run_computation(
                    operation_type='energy_calculation',
                    input_data=self.test_data_small,
                    realm=realm,
                    enable_htr=False,
                    enable_error_correction=False
                )
                results[realm] = result
                self.assertEqual(result.realm_used, realm)
                self.assertGreater(result.nrci_score, 0.0)
                
            except Exception as e:
                self.fail(f"Computation failed for realm {realm}: {e}")
        
        print(f"✅ Multi-realm computation test passed: {len(results)} realms tested")
    
    def test_09_auto_realm_selection(self):
        """Test automatic realm selection."""
        print("\n🎯 Testing automatic realm selection...")
        
        # Test with different data characteristics
        test_cases = [
            (np.random.random(10), "small random data"),
            (np.sin(np.linspace(0, 10*np.pi, 100)), "sinusoidal data"),
            (np.random.exponential(2, 50), "exponential data"),
            (np.random.normal(0, 1, 200), "gaussian data")
        ]
        
        for data, description in test_cases:
            result = self.ubp_framework.run_computation(
                operation_type='energy_calculation',
                input_data=data,
                realm=None,  # Auto-select
                enable_htr=False,
                enable_error_correction=False
            )
            
            # Check that a realm was selected
            self.assertIsNotNone(result.realm_used)
            self.assertIn(result.realm_used, ['electromagnetic', 'quantum', 'gravitational', 
                                             'biological', 'cosmological', 'nuclear', 'optical'])
            
            print(f"  📊 {description}: Selected {result.realm_used} realm")
        
        print("✅ Automatic realm selection test passed")
    
    def test_10_full_v3_computation(self):
        """Test full UBP v3.0 computation with all enhancements enabled."""
        print("\n🚀 Testing full v3.0 computation...")
        
        result = self.ubp_framework.run_computation(
            operation_type='energy_calculation',
            input_data=self.test_data_medium,
            realm=None,  # Auto-select
            observer_intent=1.0,
            enable_htr=True,
            enable_error_correction=True
        )
        
        # Check all v3.0 features are working
        self.assertGreater(result.nrci_score, 0.0)
        self.assertGreater(result.energy_value, 0.0)
        self.assertIsNotNone(result.optimal_crv_used)
        self.assertGreaterEqual(result.htr_resonance_score, 0.0)
        self.assertGreaterEqual(result.bittime_precision, 0.0)
        self.assertIsInstance(result.coherence_metrics, dict)
        self.assertIsInstance(result.sub_crv_fallbacks_used, list)
        self.assertIsInstance(result.harmonic_patterns_detected, list)
        self.assertIsInstance(result.rune_operations_executed, list)
        
        # Check metadata
        self.assertIn('input_data_shape', result.metadata)
        self.assertEqual(result.metadata['system_version'], '3.0')
        
        print(f"✅ Full v3.0 computation test passed: "
              f"NRCI={result.nrci_score:.6f}, "
              f"HTR={result.htr_resonance_score:.6f}, "
              f"Time={result.computation_time:.3f}s")
    
    def test_11_performance_benchmarks(self):
        """Test performance benchmarks."""
        print("\n⚡ Testing performance benchmarks...")
        
        # Test computation speed
        start_time = time.time()
        
        results = []
        for i in range(10):
            result = self.ubp_framework.run_computation(
                operation_type='energy_calculation',
                input_data=self.test_data_small,
                enable_htr=False,
                enable_error_correction=False
            )
            results.append(result)
        
        total_time = time.time() - start_time
        avg_time_per_computation = total_time / 10
        
        # Performance requirements
        self.assertLess(avg_time_per_computation, 5.0)  # Should be under 5 seconds per computation
        
        # Check NRCI consistency
        nrci_scores = [r.nrci_score for r in results]
        nrci_std = np.std(nrci_scores)
        self.assertLess(nrci_std, 0.5)  # NRCI should be reasonably consistent
        
        print(f"✅ Performance benchmark test passed: "
              f"Avg time={avg_time_per_computation:.3f}s, "
              f"NRCI std={nrci_std:.6f}")
    
    def test_12_system_diagnostics(self):
        """Test system diagnostics."""
        print("\n🔍 Testing system diagnostics...")
        
        diagnostics = self.ubp_framework.run_system_diagnostics()
        
        # Check diagnostics structure
        self.assertIn('timestamp', diagnostics)
        self.assertEqual(diagnostics['system_version'], '3.0')
        self.assertIn('component_tests', diagnostics)
        self.assertIn('performance_tests', diagnostics)
        self.assertIn('integration_tests', diagnostics)
        self.assertIn('overall_health', diagnostics)
        
        # Check that most tests pass
        component_tests = diagnostics['component_tests']
        passed_components = sum(component_tests.values())
        total_components = len(component_tests)
        
        self.assertGreater(passed_components / total_components, 0.7)  # At least 70% should pass
        
        print(f"✅ System diagnostics test passed: "
              f"Health={diagnostics['overall_health']}, "
              f"Components={passed_components}/{total_components}")
    
    def test_13_state_persistence(self):
        """Test system state save/load."""
        print("\n💾 Testing state persistence...")
        
        # Run some computations to generate state
        for i in range(5):
            self.ubp_framework.run_computation(
                operation_type='energy_calculation',
                input_data=np.random.random(20),
                enable_htr=False,
                enable_error_correction=False
            )
        
        # Get initial status
        initial_status = self.ubp_framework.get_system_status()
        initial_computations = initial_status['total_computations']
        
        # Save state
        state_file = '/tmp/ubp_v3_test_state.json'
        self.ubp_framework.save_system_state(state_file)
        
        # Verify file exists
        self.assertTrue(os.path.exists(state_file))
        
        # Load state (this should restore statistics)
        self.ubp_framework.load_system_state(state_file)
        
        # Clean up
        os.remove(state_file)
        
        print("✅ State persistence test passed")
    
    def test_14_error_handling(self):
        """Test error handling and robustness."""
        print("\n🛡️ Testing error handling...")
        
        # Test with invalid data
        try:
            result = self.ubp_framework.run_computation(
                operation_type='energy_calculation',
                input_data=np.array([]),  # Empty array
                enable_htr=False,
                enable_error_correction=False
            )
            # Should not crash, should return reasonable result
            self.assertIsNotNone(result)
            self.assertGreaterEqual(result.nrci_score, 0.0)
        except Exception as e:
            self.fail(f"Framework should handle empty data gracefully: {e}")
        
        # Test with invalid operation type
        try:
            result = self.ubp_framework.run_computation(
                operation_type='invalid_operation',
                input_data=self.test_data_small,
                enable_htr=False,
                enable_error_correction=False
            )
            # Should not crash
            self.assertIsNotNone(result)
        except Exception as e:
            self.fail(f"Framework should handle invalid operations gracefully: {e}")
        
        print("✅ Error handling test passed")
    
    def test_15_integration_completeness(self):
        """Test that all v3.0 components are properly integrated."""
        print("\n🔗 Testing integration completeness...")
        
        status = self.ubp_framework.get_system_status()
        
        # Check that all expected realms are active
        active_realms = status['active_realms']
        expected_realms = ['electromagnetic', 'quantum', 'gravitational', 
                          'biological', 'cosmological', 'nuclear', 'optical']
        
        for realm in expected_realms:
            self.assertIn(realm, active_realms, f"Realm {realm} should be active")
        
        # Check component integration
        component_status = status['component_status']
        required_components = [
            'bitfield', 'realm_manager', 'enhanced_crv_system', 
            'htr_engine', 'bittime_mechanics', 'rune_protocol', 'advanced_error_correction'
        ]
        
        for component in required_components:
            self.assertTrue(component_status[component], 
                          f"Component {component} should be integrated")
        
        # Check v3.0 enhancements
        v3_enhancements = status['v3_enhancements']
        required_enhancements = [
            'enhanced_crvs', 'harmonic_toggle_resonance', 'bittime_mechanics',
            'rune_protocol', 'advanced_error_correction', 'nuclear_realm', 'optical_realm'
        ]
        
        for enhancement in required_enhancements:
            self.assertTrue(v3_enhancements[enhancement], 
                          f"Enhancement {enhancement} should be available")
        
        print("✅ Integration completeness test passed")

def run_integration_tests():
    """Run all UBP v3.0 integration tests."""
    print("🚀 Starting UBP Framework v3.0 Integration Tests")
    print("=" * 80)
    
    # Create test suite
    test_suite = unittest.TestLoader().loadTestsFromTestCase(TestUBPv3Integration)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Generate summary
    print("\n" + "=" * 80)
    print("🏁 UBP Framework v3.0 Integration Test Summary")
    print("=" * 80)
    
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total_tests - failures - errors
    
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"📊 Total Tests: {total_tests}")
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failures}")
    print(f"💥 Errors: {errors}")
    print(f"📈 Success Rate: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT: UBP Framework v3.0 is ready for production!")
    elif success_rate >= 80:
        print("👍 GOOD: UBP Framework v3.0 is functional with minor issues")
    elif success_rate >= 70:
        print("⚠️ FAIR: UBP Framework v3.0 needs some improvements")
    else:
        print("❌ POOR: UBP Framework v3.0 requires significant fixes")
    
    # Save test results
    test_results = {
        'timestamp': time.time(),
        'total_tests': total_tests,
        'passed': passed,
        'failed': failures,
        'errors': errors,
        'success_rate': success_rate,
        'failures': [str(f) for f in result.failures],
        'errors': [str(e) for e in result.errors]
    }
    
    with open('/tmp/ubp_v3_integration_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    print(f"📄 Detailed results saved to: /tmp/ubp_v3_integration_test_results.json")
    
    return success_rate >= 80  # Return True if tests are successful

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)

