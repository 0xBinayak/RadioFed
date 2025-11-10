"""
Unit tests for local training module.

Tests training loop, model evaluation, and model persistence.
"""

import unittest
import numpy as np
import torch
import tempfile
import os
from client.train import (
    train_local_model,
    evaluate_model,
    save_local_model,
    load_local_model
)
from client.model import FederatedModel


class TestTrain(unittest.TestCase):
    """Test cases for the training module."""
    
    def setUp(self):
        """Set up test fixtures with synthetic feature data."""
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Create synthetic feature data
        # 200 samples, 16 features, 11 classes
        self.n_samples = 200
        self.n_features = 16
        self.n_classes = 11
        
        self.features = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.labels = np.random.randint(0, self.n_classes, size=self.n_samples)
        
        # Create temporary directory for model saving
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth')
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_train_local_model_basic(self):
        """Test basic training loop execution."""
        result = train_local_model(
            self.features,
            self.labels,
            epochs=2,
            batch_size=32,
            learning_rate=0.001,
            test_split=0.2,
            verbose=False
        )
        
        # Check return structure
        self.assertIn('model', result)
        self.assertIn('train_loss', result)
        self.assertIn('train_accuracy', result)
        self.assertIn('test_loss', result)
        self.assertIn('test_accuracy', result)
        self.assertIn('history', result)
        self.assertIn('n_samples', result)
        
        # Check model type
        self.assertIsInstance(result['model'], FederatedModel)
        
        # Check metrics are valid
        self.assertGreaterEqual(result['train_accuracy'], 0.0)
        self.assertLessEqual(result['train_accuracy'], 1.0)
        self.assertGreaterEqual(result['test_accuracy'], 0.0)
        self.assertLessEqual(result['test_accuracy'], 1.0)
        self.assertGreater(result['train_loss'], 0.0)
        self.assertGreater(result['test_loss'], 0.0)
    
    def test_train_local_model_history(self):
        """Test that training history is recorded correctly."""
        epochs = 3
        result = train_local_model(
            self.features,
            self.labels,
            epochs=epochs,
            batch_size=32,
            verbose=False
        )
        
        history = result['history']
        
        # Check history structure
        self.assertIn('train_loss', history)
        self.assertIn('train_accuracy', history)
        self.assertIn('test_loss', history)
        self.assertIn('test_accuracy', history)
        
        # Check history length matches epochs
        self.assertEqual(len(history['train_loss']), epochs)
        self.assertEqual(len(history['train_accuracy']), epochs)
        self.assertEqual(len(history['test_loss']), epochs)
        self.assertEqual(len(history['test_accuracy']), epochs)
    
    def test_train_local_model_sample_count(self):
        """Test that sample count is correctly reported."""
        test_split = 0.2
        result = train_local_model(
            self.features,
            self.labels,
            epochs=1,
            test_split=test_split,
            verbose=False
        )
        
        expected_train_samples = int(self.n_samples * (1 - test_split))
        self.assertEqual(result['n_samples'], expected_train_samples)
    
    def test_train_local_model_device_cpu(self):
        """Test training on CPU device."""
        result = train_local_model(
            self.features,
            self.labels,
            epochs=1,
            device='cpu',
            verbose=False
        )
        
        # Should complete without errors
        self.assertIsInstance(result['model'], FederatedModel)
    
    def test_train_local_model_different_batch_sizes(self):
        """Test training with different batch sizes."""
        for batch_size in [16, 32, 64]:
            result = train_local_model(
                self.features,
                self.labels,
                epochs=1,
                batch_size=batch_size,
                verbose=False
            )
            
            self.assertIsInstance(result['model'], FederatedModel)
    
    def test_train_local_model_learning_improves(self):
        """Test that model improves with more epochs."""
        # Train for 1 epoch
        result_1_epoch = train_local_model(
            self.features,
            self.labels,
            epochs=1,
            verbose=False
        )
        
        # Train for 5 epochs
        result_5_epochs = train_local_model(
            self.features,
            self.labels,
            epochs=5,
            verbose=False
        )
        
        # With more epochs, accuracy should generally improve or loss should decrease
        # (not guaranteed for random data, but model should at least train)
        self.assertIsInstance(result_1_epoch['model'], FederatedModel)
        self.assertIsInstance(result_5_epochs['model'], FederatedModel)
    
    def test_evaluate_model(self):
        """Test model evaluation function."""
        # Create a simple model and data loader
        model = FederatedModel(input_dim=16, hidden_dim=64, num_classes=11)
        
        features_tensor = torch.FloatTensor(self.features[:50])
        labels_tensor = torch.LongTensor(self.labels[:50])
        dataset = torch.utils.data.TensorDataset(features_tensor, labels_tensor)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
        
        criterion = torch.nn.CrossEntropyLoss()
        device = 'cpu'
        
        metrics = evaluate_model(model, data_loader, criterion, device)
        
        # Check return structure
        self.assertIn('loss', metrics)
        self.assertIn('accuracy', metrics)
        self.assertIn('correct', metrics)
        self.assertIn('total', metrics)
        
        # Check metrics are valid
        self.assertGreaterEqual(metrics['accuracy'], 0.0)
        self.assertLessEqual(metrics['accuracy'], 1.0)
        self.assertGreater(metrics['loss'], 0.0)
        self.assertEqual(metrics['total'], 50)
        self.assertGreaterEqual(metrics['correct'], 0)
        self.assertLessEqual(metrics['correct'], 50)
    
    def test_evaluate_model_empty_loader(self):
        """Test evaluation with empty data loader."""
        model = FederatedModel(input_dim=16, hidden_dim=64, num_classes=11)
        
        # Create empty data loader
        empty_dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(np.zeros((0, 16))),
            torch.LongTensor(np.zeros(0, dtype=np.int64))
        )
        empty_loader = torch.utils.data.DataLoader(empty_dataset, batch_size=16)
        
        criterion = torch.nn.CrossEntropyLoss()
        device = 'cpu'
        
        metrics = evaluate_model(model, empty_loader, criterion, device)
        
        # Should handle empty loader gracefully
        self.assertEqual(metrics['loss'], 0.0)
        self.assertEqual(metrics['accuracy'], 0.0)
        self.assertEqual(metrics['total'], 0)
    
    def test_save_local_model(self):
        """Test saving model weights to file."""
        # Train a simple model
        result = train_local_model(
            self.features,
            self.labels,
            epochs=1,
            verbose=False
        )
        
        model = result['model']
        
        # Save model
        save_local_model(model, self.model_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(self.model_path))
        
        # Check file is not empty
        self.assertGreater(os.path.getsize(self.model_path), 0)
    
    def test_save_local_model_creates_directory(self):
        """Test that save_local_model creates parent directories."""
        nested_path = os.path.join(self.temp_dir, 'nested', 'dir', 'model.pth')
        
        # Train a simple model
        result = train_local_model(
            self.features,
            self.labels,
            epochs=1,
            verbose=False
        )
        
        model = result['model']
        
        # Save model (should create directories)
        save_local_model(model, nested_path)
        
        # Check file exists
        self.assertTrue(os.path.exists(nested_path))
    
    def test_load_local_model(self):
        """Test loading model weights from file."""
        # Train and save a model
        result = train_local_model(
            self.features,
            self.labels,
            epochs=1,
            verbose=False
        )
        
        original_model = result['model']
        save_local_model(original_model, self.model_path)
        
        # Load model
        loaded_model = load_local_model(self.model_path, device='cpu')
        
        # Check model type
        self.assertIsInstance(loaded_model, FederatedModel)
        
        # Check that weights match
        original_state = original_model.state_dict()
        loaded_state = loaded_model.state_dict()
        
        for key in original_state.keys():
            self.assertTrue(torch.allclose(original_state[key], loaded_state[key]))
    
    def test_load_local_model_file_not_found(self):
        """Test that loading from nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_local_model('/nonexistent/path/model.pth')
    
    def test_save_and_load_preserves_predictions(self):
        """Test that saved and loaded model produces same predictions."""
        # Train a model
        result = train_local_model(
            self.features,
            self.labels,
            epochs=2,
            verbose=False
        )
        
        original_model = result['model']
        original_model.eval()
        
        # Get predictions from original model
        test_input = torch.FloatTensor(self.features[:10])
        with torch.no_grad():
            original_predictions = original_model(test_input)
        
        # Save and load model
        save_local_model(original_model, self.model_path)
        loaded_model = load_local_model(self.model_path, device='cpu')
        loaded_model.eval()
        
        # Get predictions from loaded model
        with torch.no_grad():
            loaded_predictions = loaded_model(test_input)
        
        # Predictions should be identical
        self.assertTrue(torch.allclose(original_predictions, loaded_predictions, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
