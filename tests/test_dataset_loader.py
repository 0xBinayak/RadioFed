"""
Unit tests for RadioML dataset loading functionality.

Tests dataset loading, metadata extraction, and train/test splitting.
"""

import unittest
import numpy as np
import pickle
import tempfile
import os
from client.dataset_loader import (
    load_radioml_dataset,
    get_dataset_info,
    split_dataset,
    flatten_dataset
)


class TestDatasetLoader(unittest.TestCase):
    """Test cases for the dataset loader module."""
    
    def setUp(self):
        """Set up test fixtures with synthetic RadioML-like dataset."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a synthetic RadioML dataset
        # Format: {(modulation, SNR): samples_array}
        self.synthetic_dataset = {}
        
        modulations = ['BPSK', 'QPSK', '8PSK']
        snrs = [-10, 0, 10]
        samples_per_key = 100
        
        for mod in modulations:
            for snr in snrs:
                # Create random I/Q samples with shape (samples_per_key, 2, 128)
                samples = np.random.randn(samples_per_key, 2, 128).astype(np.float32)
                self.synthetic_dataset[(mod, snr)] = samples
        
        # Save synthetic dataset to pickle file
        self.dataset_path = os.path.join(self.temp_dir, 'test_dataset.pkl')
        with open(self.dataset_path, 'wb') as f:
            pickle.dump(self.synthetic_dataset, f)
    
    def tearDown(self):
        """Clean up temporary files."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_load_radioml_dataset_success(self):
        """Test successful loading of RadioML dataset."""
        dataset = load_radioml_dataset(self.dataset_path)
        
        self.assertIsInstance(dataset, dict)
        self.assertEqual(len(dataset), 9)  # 3 modulations * 3 SNRs
        
        # Check structure
        first_key = list(dataset.keys())[0]
        self.assertIsInstance(first_key, tuple)
        self.assertEqual(len(first_key), 2)  # (modulation, SNR)
        
        # Check sample shape
        samples = dataset[first_key]
        self.assertEqual(samples.shape[1:], (2, 128))
    
    def test_load_radioml_dataset_file_not_found(self):
        """Test loading from nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_radioml_dataset('/nonexistent/path/dataset.pkl')
    
    def test_load_radioml_dataset_invalid_file(self):
        """Test loading invalid pickle file raises error."""
        invalid_path = os.path.join(self.temp_dir, 'invalid.pkl')
        with open(invalid_path, 'w') as f:
            f.write("not a pickle file")
        
        with self.assertRaises(pickle.UnpicklingError):
            load_radioml_dataset(invalid_path)
    
    def test_get_dataset_info(self):
        """Test extraction of dataset metadata."""
        info = get_dataset_info(self.synthetic_dataset)
        
        self.assertEqual(info['modulations'], ['8PSK', 'BPSK', 'QPSK'])
        self.assertEqual(info['snrs'], [-10, 0, 10])
        self.assertEqual(info['sample_count'], 900)  # 3 mods * 3 SNRs * 100 samples
        self.assertEqual(info['shape'], (2, 128))
        
        # Check samples per modulation
        self.assertEqual(info['samples_per_mod']['BPSK'], 300)  # 3 SNRs * 100 samples
        self.assertEqual(info['samples_per_mod']['QPSK'], 300)
        self.assertEqual(info['samples_per_mod']['8PSK'], 300)
    
    def test_get_dataset_info_empty(self):
        """Test dataset info with empty dataset."""
        info = get_dataset_info({})
        
        self.assertEqual(info['modulations'], [])
        self.assertEqual(info['snrs'], [])
        self.assertEqual(info['sample_count'], 0)
        self.assertIsNone(info['shape'])
    
    def test_split_dataset_default_ratio(self):
        """Test dataset splitting with default 80/20 ratio."""
        train_dataset, test_dataset = split_dataset(self.synthetic_dataset)
        
        # Check that all keys are present in both splits
        self.assertEqual(set(train_dataset.keys()), set(self.synthetic_dataset.keys()))
        self.assertEqual(set(test_dataset.keys()), set(self.synthetic_dataset.keys()))
        
        # Check split ratios
        for key in self.synthetic_dataset.keys():
            original_count = self.synthetic_dataset[key].shape[0]
            train_count = train_dataset[key].shape[0]
            test_count = test_dataset[key].shape[0]
            
            self.assertEqual(train_count, 80)  # 80% of 100
            self.assertEqual(test_count, 20)   # 20% of 100
            self.assertEqual(train_count + test_count, original_count)
    
    def test_split_dataset_custom_ratio(self):
        """Test dataset splitting with custom ratio."""
        train_dataset, test_dataset = split_dataset(self.synthetic_dataset, train_ratio=0.7)
        
        for key in self.synthetic_dataset.keys():
            train_count = train_dataset[key].shape[0]
            test_count = test_dataset[key].shape[0]
            
            self.assertEqual(train_count, 70)  # 70% of 100
            self.assertEqual(test_count, 30)   # 30% of 100
    
    def test_split_dataset_invalid_ratio(self):
        """Test that invalid train ratio raises ValueError."""
        with self.assertRaises(ValueError):
            split_dataset(self.synthetic_dataset, train_ratio=1.5)
        
        with self.assertRaises(ValueError):
            split_dataset(self.synthetic_dataset, train_ratio=0.0)
        
        with self.assertRaises(ValueError):
            split_dataset(self.synthetic_dataset, train_ratio=-0.1)
    
    def test_split_dataset_reproducibility(self):
        """Test that splitting with same seed produces same results."""
        train1, test1 = split_dataset(self.synthetic_dataset, random_seed=42)
        train2, test2 = split_dataset(self.synthetic_dataset, random_seed=42)
        
        # Check that splits are identical
        for key in self.synthetic_dataset.keys():
            np.testing.assert_array_equal(train1[key], train2[key])
            np.testing.assert_array_equal(test1[key], test2[key])
    
    def test_flatten_dataset(self):
        """Test flattening dataset into samples and labels."""
        samples, labels = flatten_dataset(self.synthetic_dataset)
        
        # Check shapes
        self.assertEqual(samples.shape, (900, 2, 128))  # 3 mods * 3 SNRs * 100 samples
        self.assertEqual(labels.shape, (900,))
        
        # Check label range (should be 0, 1, 2 for 3 modulations)
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 3)
        self.assertTrue(np.all(unique_labels >= 0))
        self.assertTrue(np.all(unique_labels < 3))
        
        # Check that each label appears correct number of times
        for label in unique_labels:
            count = np.sum(labels == label)
            self.assertEqual(count, 300)  # 3 SNRs * 100 samples
    
    def test_flatten_dataset_label_consistency(self):
        """Test that same modulation always gets same label."""
        samples1, labels1 = flatten_dataset(self.synthetic_dataset)
        samples2, labels2 = flatten_dataset(self.synthetic_dataset)
        
        # Labels should be identical for same dataset
        np.testing.assert_array_equal(labels1, labels2)


if __name__ == '__main__':
    unittest.main()
