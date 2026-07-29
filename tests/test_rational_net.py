import sys
import os
import torch
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.rational_net import RationalNet

class _BaseTestRationalNet:
    """
    Base testing class containing all shared physics, shape, and tensor checks.
    (Does not inherit from unittest.TestCase directly to prevent isolated execution).
    """
    def _init_base_setup(self, num_local, num_global):
        """
        Initialize the RationalNet instance and dummy data before each test.
        """
        self.batch_size = 8
        self.num_poles = 128
        self.num_ports = 4
        self.num_local = num_local
        self.num_global = num_global
        
        self.model = RationalNet(
            num_poles=self.num_poles,
            num_local_features=self.num_local,
            num_global_features=self.num_global,
            num_ports=self.num_ports
        )

        # Put the model in eval mode to disable dropout for deterministic outputs during testing
        self.model.eval()

        # Dummy input geometric features based on dataset specs
        self.x_local = torch.randn(self.batch_size, self.num_local)
        self.x_global = torch.randn(self.batch_size, self.num_global)

        # Match TUHH dataset frequency distribution (250 MHz to 100 GHz, 401 points)
        self.frequencies_hz = torch.linspace(0.25e9, 100e9, 401)

    def test_forward_pass(self):
        """
        Test the forward pass of the RationalNet model to ensure it produces outputs of the correct shape and type.
        """
        with torch.no_grad():
            poles, residues, d_term = self.model(self.x_local, self.x_global)

        # Check output shapes
        self.assertEqual(poles.shape, (self.batch_size, self.num_poles))
        self.assertEqual(residues.shape, (self.batch_size, self.num_poles, self.num_ports, self.num_ports))
        self.assertEqual(d_term.shape, (self.batch_size, self.num_ports, self.num_ports))
        
        # Check output types (using PyTorch's native tensor method)
        self.assertTrue(poles.is_complex(), "Poles should be complex")
        self.assertTrue(residues.is_complex(), "Residues should be complex")
        self.assertTrue(d_term.is_complex(), "D-term should be complex")

    def test_physics_layer_shapes(self):
        """
        Test the physics layer to ensure it produces S-parameters of the correct shape and perfectly reconstructs the 4x4 matrix.
        """
        with torch.no_grad():
            poles, residues, d_term = self.model(self.x_local, self.x_global)
            s_matrix = self.model.predict_frequency_response(poles, residues, d_term, self.frequencies_hz)

        # Check output shape
        expected_shape = (self.batch_size, len(self.frequencies_hz), self.num_ports, self.num_ports)
        self.assertEqual(s_matrix.shape, expected_shape)

    def test_physics_constraints(self):
        """
        Test the physics constraints to ensure the predicted poles and residues satisfy causality, conjugate symmetry, and passivity.
        """
        with torch.no_grad():
            poles, residues, _ = self.model(self.x_local, self.x_global)
            
        constraints = self.model.verify_physics_constraints(poles, residues)

        # Check that all constraints are satisfied
        self.assertTrue(constraints["causality_preserved"], "Causality constraint violated: poles in right half-plane")
        self.assertTrue(constraints["conjugate_symmetry_preserved"], "Conjugate symmetry constraint violated: poles not symmetric about real axis")
        self.assertTrue(constraints["passivity_preserved"], f"Passivity constraint violated: minimum residue eigenvalue {constraints['min_residue_eigenvalue']} < 0")
    
    def test_conjugate_symmetry_precision(self):
        """
        Mathematically verifies that the residues and poles mirror each other with high precision.
        """
        with torch.no_grad():
            poles, residues, _ = self.model(self.x_local, self.x_global)
        
        half_idx = self.num_poles // 2
        
        # Check Poles
        p_upper = poles[:, :half_idx]
        p_lower = poles[:, half_idx:]
        self.assertTrue(torch.allclose(p_upper, p_lower.conj(), atol=1e-6), "Poles are not exact conjugates.")
        
        # Check Residues
        r_upper = residues[:, :half_idx, :, :]
        r_lower = residues[:, half_idx:, :, :]
        self.assertTrue(torch.allclose(r_upper, r_lower.conj(), atol=1e-6), "Residues are not exact conjugates.")


# ==========================================
# TEST RUNNERS FOR SPECIFIC DATASETS
# ==========================================

class TestRationalNetLink(_BaseTestRationalNet, unittest.TestCase):
    """Executes all tests using the specific feature dimensions of the Universal Link dataset."""
    
    def setUp(self):
        # Link dataset specs: 9 local, 8 global
        self._init_base_setup(num_local=9, num_global=8)

    # Architectural edge cases only need to be tested in one of the dataset classes
    def test_odd_num_poles_assertion(self):
        """Ensures the model violently crashes if given an odd number of poles."""
        with self.assertRaises(AssertionError) as context:
            RationalNet(
                num_poles=11, # Odd number
                num_local_features=self.num_local,
                num_global_features=self.num_global,
                num_ports=self.num_ports
            )
        self.assertTrue("must be even" in str(context.exception))

    def test_minimum_pole_count(self):
        """Ensures the model initializes and passes data successfully with the minimum required poles (2)."""
        min_model = RationalNet(
            num_poles=2, 
            num_local_features=self.num_local,
            num_global_features=self.num_global,
            num_ports=self.num_ports
        )
        min_model.eval()
        with torch.no_grad():
            poles, residues, d_term = min_model(self.x_local, self.x_global)
        
        self.assertEqual(poles.shape, (self.batch_size, 2))
        self.assertEqual(residues.shape, (self.batch_size, 2, self.num_ports, self.num_ports))


class TestRationalNetArray(_BaseTestRationalNet, unittest.TestCase):
    """Executes all tests using the specific feature dimensions of the Array dataset."""
    
    def setUp(self):
        # Array dataset specs: 8 local, 7 global
        self._init_base_setup(num_local=8, num_global=7)


if __name__ == '__main__':
    unittest.main()