"""
Unit tests for FedAvg aggregation logic.

Tests the core aggregation functionality including weighted averaging,
weight validation, and architecture compatibility checks.
"""

import unittest
import torch
import tempfile
import os
from central.aggregator import (
    aggregate_weights,
    load_client_weights,
    save_global_model,
    validate_weight_compatibility,
    aggregate_from_registry
)
from central.model import FederatedModel


class TestAggregator(unittest.TestCase):
    """Test cases for the aggregation module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_aggregate_weights_simple(self):
        """Test FedAvg with two clients and known weights."""
        # Create simple weight dictionaries
        weights1 = {
            'layer1': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            'bias1': torch.tensor([1.0, 2.0])
        }
        weights2 = {
            'layer1': torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            'bias1': torch.tensor([3.0, 4.0])
        }
        
        # Client 1 has 100 samples, Client 2 has 100 samples (equal weight)
        sample_counts = [100, 100]
        
        # Expected: average of the two
        result = aggregate_weights([weights1, weights2], sample_counts)
        
        expected_layer1 = torch.tensor([[3.0, 4.0], [5.0, 6.0]])
        expected_bias1 = torch.tensor([2.0, 3.0])
        
        self.assertTrue(torch.allclose(result['layer1'], expected_layer1))
        self.assertTrue(torch.allclose(result['bias1'], expected_bias1))
    
    def test_aggregate_weights_weighted(self):
        """Test FedAvg with weighted averaging based on sample counts."""
        weights1 = {
            'param': torch.tensor([10.0])
        }
        weights2 = {
            'param': torch.tensor([20.0])
        }
        
        # Client 1: 300 samples (75%), Client 2: 100 samples (25%)
        sample_counts = [300, 100]
        
        result = aggregate_weights([weights1, weights2], sample_counts)
        
        # Expected: 0.75 * 10 + 0.25 * 20 = 7.5 + 5.0 = 12.5
        expected = torch.tensor([12.5])
        
        self.assertTrue(torch.allclose(result['param'], expected))
    
    def test_aggregate_weights_three_clients(self):
        """Test FedAvg with three clients."""
        weights1 = {'w': torch.tensor([1.0])}
        weights2 = {'w': torch.tensor([2.0])}
        weights3 = {'w': torch.tensor([3.0])}
        
        # Equal samples
        sample_counts = [100, 100, 100]
        
        result = aggregate_weights([weights1, weights2, weights3], sample_counts)
        
        # Expected: (1 + 2 + 3) / 3 = 2.0
        expected = torch.tensor([2.0])
        
        self.assertTrue(torch.allclose(result['w'], expected))
    
    def test_aggregate_weights_empty_list(self):
        """Test that empty client list raises ValueError."""
        with self.assertRaises(ValueError) as context:
            aggregate_weights([], [])
        
        self.assertIn("No client weights", str(context.exception))
    
    def test_aggregate_weights_mismatched_counts(self):
        """Test that mismatched weights and sample counts raises ValueError."""
        weights1 = {'w': torch.tensor([1.0])}
        weights2 = {'w': torch.tensor([2.0])}
        
        with self.assertRaises(ValueError) as context:
            aggregate_weights([weights1, weights2], [100])
        
        self.assertIn("Mismatch", str(context.exception))
    
    def test_aggregate_weights_invalid_sample_count(self):
        """Test that zero or negative sample counts raise ValueError."""
        weights1 = {'w': torch.tensor([1.0])}
        
        with self.assertRaises(ValueError) as context:
            aggregate_weights([weights1], [0])
        
        self.assertIn("positive", str(context.exception))
    
    def test_validate_weight_compatibility_success(self):
        """Test weight validation with compatible weights."""
        weights1 = {
            'layer1': torch.tensor([[1.0, 2.0]]),
            'bias1': torch.tensor([1.0])
        }
        weights2 = {
            'layer1': torch.tensor([[3.0, 4.0]]),
            'bias1': torch.tensor([2.0])
        }
        
        is_valid, msg = validate_weight_compatibility([weights1, weights2])
        
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")
    
    def test_validate_weight_compatibility_missing_keys(self):
        """Test weight validation with mismatched keys."""
        weights1 = {
            'layer1': torch.tensor([[1.0, 2.0]]),
            'bias1': torch.tensor([1.0])
        }
        weights2 = {
            'layer1': torch.tensor([[3.0, 4.0]])
            # Missing 'bias1'
        }
        
        is_valid, msg = validate_weight_compatibility([weights1, weights2])
        
        self.assertFalse(is_valid)
        self.assertIn("incompatible keys", msg)
    
    def test_validate_weight_compatibility_mismatched_shapes(self):
        """Test weight validation with mismatched tensor shapes."""
        weights1 = {
            'layer1': torch.tensor([[1.0, 2.0]])  # Shape: (1, 2)
        }
        weights2 = {
            'layer1': torch.tensor([[3.0], [4.0]])  # Shape: (2, 1)
        }
        
        is_valid, msg = validate_weight_compatibility([weights1, weights2])
        
        self.assertFalse(is_valid)
        self.assertIn("incompatible shape", msg)
    
    def test_validate_weight_compatibility_with_model(self):
        """Test weight validation against a reference model."""
        model = FederatedModel(input_dim=16, hidden_dim=64, num_classes=11)
        model_weights = model.state_dict()
        
        # Create compatible weights
        compatible_weights = {k: v.clone() for k, v in model_weights.items()}
        
        is_valid, msg = validate_weight_compatibility([compatible_weights], model)
        
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")
    
    def test_validate_weight_compatibility_incompatible_with_model(self):
        """Test weight validation with incompatible model architecture."""
        model = FederatedModel(input_dim=16, hidden_dim=64, num_classes=11)
        
        # Create weights with wrong shape
        incompatible_weights = {
            'fc1.weight': torch.randn(32, 16),  # Wrong hidden_dim
            'fc1.bias': torch.randn(32),
            'fc2.weight': torch.randn(11, 32),
            'fc2.bias': torch.randn(11)
        }
        
        is_valid, msg = validate_weight_compatibility([incompatible_weights], model)
        
        self.assertFalse(is_valid)
        self.assertIn("incompatible shape", msg)
    
    def test_save_and_load_weights(self):
        """Test saving and loading weights."""
        weights = {
            'layer1': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            'bias1': torch.tensor([1.0, 2.0])
        }
        
        save_path = os.path.join(self.temp_dir, 'test_weights.pth')
        
        # Save weights
        save_global_model(weights, save_path)
        self.assertTrue(os.path.exists(save_path))
        
        # Load weights
        loaded_weights = load_client_weights(save_path)
        
        self.assertEqual(set(weights.keys()), set(loaded_weights.keys()))
        self.assertTrue(torch.allclose(weights['layer1'], loaded_weights['layer1']))
        self.assertTrue(torch.allclose(weights['bias1'], loaded_weights['bias1']))
    
    def test_load_client_weights_nonexistent_file(self):
        """Test loading weights from nonexistent file raises error."""
        with self.assertRaises(FileNotFoundError):
            load_client_weights('/nonexistent/path/weights.pth')
    
    def test_aggregate_from_registry(self):
        """Test aggregation from client registry information."""
        # Create and save client weights
        model1 = FederatedModel()
        model2 = FederatedModel()
        
        weights1_path = os.path.join(self.temp_dir, 'client1.pth')
        weights2_path = os.path.join(self.temp_dir, 'client2.pth')
        
        model1.save_weights(weights1_path)
        model2.save_weights(weights2_path)
        
        # Create client info list
        client_info = [
            {'client_id': 'client1', 'weights_path': weights1_path, 'n_samples': 100},
            {'client_id': 'client2', 'weights_path': weights2_path, 'n_samples': 200}
        ]
        
        global_model_path = os.path.join(self.temp_dir, 'global_model.pth')
        
        # Perform aggregation
        result = aggregate_from_registry(client_info, global_model_path)
        
        self.assertEqual(result['num_clients'], 2)
        self.assertEqual(result['total_samples'], 300)
        self.assertTrue(os.path.exists(global_model_path))
    
    def test_aggregate_from_registry_empty(self):
        """Test aggregation with no clients raises error."""
        with self.assertRaises(ValueError) as context:
            aggregate_from_registry([], '/tmp/model.pth')
        
        self.assertIn("No clients", str(context.exception))


if __name__ == '__main__':
    unittest.main()
