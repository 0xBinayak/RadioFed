"""
Unit tests for feature extraction from I/Q samples.

Tests feature extraction functionality including statistical,
frequency domain, and time domain features.
"""

import unittest
import numpy as np
from client.feature_extract import (
    extract_features_from_iq,
    process_dataset,
    normalize_features
)


class TestFeatureExtract(unittest.TestCase):
    """Test cases for the feature extraction module."""
    
    def setUp(self):
        """Set up test fixtures with synthetic I/Q samples."""
        # Create a simple synthetic I/Q sample
        np.random.seed(42)
        self.sample_iq = np.random.randn(2, 128).astype(np.float32)
        
        # Create a batch of samples for dataset processing
        self.batch_samples = np.random.randn(50, 2, 128).astype(np.float32)
        self.batch_labels = np.random.randint(0, 11, size=50)
    
    def test_extract_features_from_iq_shape(self):
        """Test that feature extraction returns correct shape."""
        features = extract_features_from_iq(self.sample_iq)
        
        self.assertEqual(features.shape, (16,))
        self.assertEqual(features.dtype, np.float32)
    
    def test_extract_features_from_iq_invalid_shape(self):
        """Test that invalid input shape raises ValueError."""
        invalid_sample = np.random.randn(3, 128)  # Wrong first dimension
        
        with self.assertRaises(ValueError):
            extract_features_from_iq(invalid_sample)
        
        invalid_sample2 = np.random.randn(2, 64)  # Wrong second dimension
        
        with self.assertRaises(ValueError):
            extract_features_from_iq(invalid_sample2)
    
    def test_extract_features_from_iq_no_nan(self):
        """Test that extracted features contain no NaN values."""
        features = extract_features_from_iq(self.sample_iq)
        
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))
    
    def test_extract_features_from_iq_deterministic(self):
        """Test that same input produces same features."""
        features1 = extract_features_from_iq(self.sample_iq)
        features2 = extract_features_from_iq(self.sample_iq)
        
        np.testing.assert_array_equal(features1, features2)
    
    def test_extract_features_statistical(self):
        """Test that statistical features are computed correctly."""
        # Create a simple known I/Q sample
        i_channel = np.array([1.0, 2.0, 3.0, 4.0, 5.0] + [0.0] * 123)
        q_channel = np.array([2.0, 4.0, 6.0, 8.0, 10.0] + [0.0] * 123)
        sample = np.vstack([i_channel, q_channel]).astype(np.float32)
        
        features = extract_features_from_iq(sample)
        
        # First 5 features are I channel statistics
        # Mean of i_channel should be close to (1+2+3+4+5)/128
        self.assertIsInstance(features[0], (float, np.floating))
        
        # All statistical features should be finite
        for i in range(10):  # First 10 are statistical features
            self.assertTrue(np.isfinite(features[i]))
    
    def test_extract_features_frequency_domain(self):
        """Test that frequency domain features are computed."""
        features = extract_features_from_iq(self.sample_iq)
        
        # Features 10-13 are frequency domain features
        # FFT peak magnitude (feature 10)
        self.assertGreater(features[10], 0)
        
        # FFT peak frequency (feature 11)
        self.assertGreaterEqual(features[11], 0)
        self.assertLessEqual(features[11], 0.5)  # Nyquist limit
        
        # Spectral centroid and bandwidth (features 12-13)
        self.assertTrue(np.isfinite(features[12]))
        self.assertTrue(np.isfinite(features[13]))
    
    def test_extract_features_time_domain(self):
        """Test that time domain features are computed."""
        features = extract_features_from_iq(self.sample_iq)
        
        # Features 14-15 are time domain features
        # Zero-crossing rate (feature 14)
        self.assertGreaterEqual(features[14], 0)
        self.assertLessEqual(features[14], 1)
        
        # Energy (feature 15)
        self.assertGreater(features[15], 0)
    
    def test_extract_features_zero_signal(self):
        """Test feature extraction with zero signal."""
        zero_sample = np.zeros((2, 128), dtype=np.float32)
        features = extract_features_from_iq(zero_sample)
        
        # Should not raise errors and should produce finite values
        self.assertEqual(features.shape, (16,))
        self.assertFalse(np.any(np.isnan(features)))
    
    def test_process_dataset_shape(self):
        """Test that dataset processing returns correct shapes."""
        features, labels = process_dataset(
            self.batch_samples, 
            self.batch_labels, 
            verbose=False
        )
        
        self.assertEqual(features.shape, (50, 16))
        self.assertEqual(labels.shape, (50,))
        np.testing.assert_array_equal(labels, self.batch_labels)
    
    def test_process_dataset_verbose(self):
        """Test that verbose mode doesn't cause errors."""
        # This should print progress but not raise errors
        features, labels = process_dataset(
            self.batch_samples[:10], 
            self.batch_labels[:10], 
            verbose=True
        )
        
        self.assertEqual(features.shape, (10, 16))
    
    def test_process_dataset_handles_errors(self):
        """Test that dataset processing handles individual sample errors."""
        # Create a batch with one invalid sample
        invalid_batch = self.batch_samples.copy()
        invalid_batch[5] = np.nan  # Introduce NaN
        
        features, labels = process_dataset(
            invalid_batch, 
            self.batch_labels, 
            verbose=False
        )
        
        # Should still return correct shape (with zero vector for failed sample)
        self.assertEqual(features.shape, (50, 16))
    
    def test_normalize_features(self):
        """Test feature normalization."""
        features = np.random.randn(100, 16).astype(np.float32)
        
        normalized, mean, std = normalize_features(features)
        
        # Check shapes
        self.assertEqual(normalized.shape, features.shape)
        self.assertEqual(mean.shape, (16,))
        self.assertEqual(std.shape, (16,))
        
        # Check that normalized features have approximately zero mean and unit std
        for i in range(16):
            self.assertAlmostEqual(np.mean(normalized[:, i]), 0.0, places=5)
            self.assertAlmostEqual(np.std(normalized[:, i]), 1.0, places=5)
    
    def test_normalize_features_zero_std(self):
        """Test normalization with zero standard deviation."""
        # Create features where one dimension has zero variance
        features = np.random.randn(100, 16).astype(np.float32)
        features[:, 5] = 1.0  # Constant value
        
        normalized, mean, std = normalize_features(features)
        
        # Should not raise errors and should handle zero std gracefully
        self.assertFalse(np.any(np.isnan(normalized)))
        self.assertFalse(np.any(np.isinf(normalized)))
        
        # Constant dimension should remain constant
        self.assertTrue(np.allclose(normalized[:, 5], 0.0))
    
    def test_feature_extraction_consistency(self):
        """Test that features are consistent across multiple extractions."""
        features1, _ = process_dataset(
            self.batch_samples, 
            self.batch_labels, 
            verbose=False
        )
        features2, _ = process_dataset(
            self.batch_samples, 
            self.batch_labels, 
            verbose=False
        )
        
        np.testing.assert_array_equal(features1, features2)


if __name__ == '__main__':
    unittest.main()
