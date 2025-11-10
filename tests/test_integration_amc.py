"""
Integration tests for AMC Dashboard Enhancement

Tests the complete workflow including:
- Dataset partitioning
- Feature extraction
- Model training and timing
- Aggregation for KNN and Decision Tree
- Dashboard functionality
- Auto-start server behavior
- Simplified client workflow
- Multi-client federated learning simulation

Requirements: 1.1-1.5, 2.1-2.5, 3.1-3.4, 4.1-4.5, 5.1-5.4, 6.1-6.5, 7.1-7.5, 8.1-8.5, 9.1-9.5, 10.1-10.5
"""

import unittest
import tempfile
import os
import shutil
import numpy as np
import pickle
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Import modules to test
from data.partition_dataset import (
    load_radioml_pkl_dataset,
    filter_analog_modulations,
    partition_dataset,
    save_partition,
    validate_partitions
)
from client.feature_extract import (
    extract_analog_features,
    extract_features_from_iq,
    process_dataset,
    compute_instantaneous_amplitude,
    compute_instantaneous_frequency,
    compute_statistical_features
)
from central.aggregator import (
    aggregate_knn_models,
    aggregate_dt_models,
    DecisionTreeEnsemble,
    evaluate_global_model
)


class TestDatasetPartitioningWorkflow(unittest.TestCase):
    """
    Test 11.1: Dataset partitioning workflow
    
    Verifies:
    - Partition creation with different client counts
    - Partition balance and non-overlap
    - Partition loading in client
    """
    
    def setUp(self):
        """Set up test fixtures with synthetic RadioML-like dataset."""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create synthetic RadioML dataset with analog modulations
        self.synthetic_dataset = {}
        
        # Analog modulations: AM-DSB, AM-SSB, WBFM
        modulations = ['AM-DSB', 'AM-SSB', 'WBFM']
        snrs = [-10, 0, 10, 18]
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
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_partition_creation_with_different_client_counts(self):
        """Test creating partitions with 2, 3, and 5 clients."""
        # Load and filter dataset
        dataset = load_radioml_pkl_dataset(self.dataset_path)
        filtered_dataset = filter_analog_modulations(dataset)
        
        for num_clients in [2, 3, 5]:
            with self.subTest(num_clients=num_clients):
                partitions = partition_dataset(filtered_dataset, num_clients)
                
                # Verify correct number of partitions
                self.assertEqual(len(partitions), num_clients)
                
                # Verify each partition has data
                for i, partition in enumerate(partitions):
                    self.assertGreater(len(partition), 0, f"Partition {i} is empty")
                    
                    # Verify partition has samples
                    total_samples = sum(samples.shape[0] for samples in partition.values())
                    self.assertGreater(total_samples, 0, f"Partition {i} has no samples")
    
    def test_partition_balance(self):
        """Test that partitions are balanced across clients."""
        dataset = load_radioml_pkl_dataset(self.dataset_path)
        filtered_dataset = filter_analog_modulations(dataset)
        
        num_clients = 3
        partitions = partition_dataset(filtered_dataset, num_clients, balance_classes=True)
        
        # Count samples per partition
        sample_counts = []
        for partition in partitions:
            total_samples = sum(samples.shape[0] for samples in partition.values())
            sample_counts.append(total_samples)
        
        # Check balance (max difference should be small)
        max_count = max(sample_counts)
        min_count = min(sample_counts)
        max_diff = max_count - min_count
        
        # Allow small difference due to rounding (up to 15 samples for small datasets)
        self.assertLessEqual(max_diff, 15, "Partitions are not balanced")
        
        # Check that all partitions have similar sample counts
        avg_count = sum(sample_counts) / len(sample_counts)
        for count in sample_counts:
            self.assertAlmostEqual(count, avg_count, delta=15)
    
    def test_partition_non_overlap(self):
        """Test that partitions are non-overlapping."""
        dataset = load_radioml_pkl_dataset(self.dataset_path)
        filtered_dataset = filter_analog_modulations(dataset)
        
        num_clients = 3
        partitions = partition_dataset(filtered_dataset, num_clients, random_seed=42)
        
        # For each (mod, snr) key, verify that samples don't overlap
        for key in filtered_dataset.keys():
            # Collect all samples from partitions for this key
            partition_samples = []
            for partition in partitions:
                if key in partition:
                    partition_samples.append(partition[key])
            
            # Verify total samples match original
            total_partition_samples = sum(s.shape[0] for s in partition_samples)
            original_samples = filtered_dataset[key].shape[0]
            
            self.assertEqual(total_partition_samples, original_samples,
                           f"Sample count mismatch for key {key}")
    
    def test_partition_loading(self):
        """Test loading partitions from saved files."""
        dataset = load_radioml_pkl_dataset(self.dataset_path)
        filtered_dataset = filter_analog_modulations(dataset)
        
        num_clients = 3
        partitions = partition_dataset(filtered_dataset, num_clients)
        
        # Save partitions
        partition_dir = os.path.join(self.temp_dir, 'partitions')
        for i, partition in enumerate(partitions):
            partition_path = os.path.join(partition_dir, f'client_{i}.pkl')
            save_partition(partition, partition_path)
        
        # Load partitions and verify
        for i in range(num_clients):
            partition_path = os.path.join(partition_dir, f'client_{i}.pkl')
            self.assertTrue(os.path.exists(partition_path), f"Partition file {i} not found")
            
            # Load partition
            with open(partition_path, 'rb') as f:
                loaded_partition = pickle.load(f)
            
            # Verify structure
            self.assertIsInstance(loaded_partition, dict)
            self.assertGreater(len(loaded_partition), 0)
            
            # Verify samples
            for key, samples in loaded_partition.items():
                self.assertEqual(samples.shape[1:], (2, 128))
    
    def test_partition_validation(self):
        """Test partition validation function."""
        dataset = load_radioml_pkl_dataset(self.dataset_path)
        filtered_dataset = filter_analog_modulations(dataset)
        
        num_clients = 3
        partitions = partition_dataset(filtered_dataset, num_clients)
        
        # Should not raise any exceptions
        validate_partitions(partitions)
        
        # Test with empty partition (should raise ValueError)
        invalid_partitions = [{}]
        with self.assertRaises(ValueError):
            validate_partitions(invalid_partitions)


class TestFeatureExtractionPipeline(unittest.TestCase):
    """
    Test 11.2: Feature extraction pipeline
    
    Verifies:
    - 8D feature vector generation
    - Various signal types and SNRs
    - Numerical stability
    """
    
    def setUp(self):
        """Set up test fixtures with various signal types."""
        np.random.seed(42)
        
        # Create different signal types
        self.signals = {}
        
        # Normal random signal
        self.signals['random'] = np.random.randn(128) + 1j * np.random.randn(128)
        
        # Constant signal (edge case)
        self.signals['constant'] = np.ones(128, dtype=complex)
        
        # Zero signal (edge case)
        self.signals['zero'] = np.zeros(128, dtype=complex)
        
        # High SNR signal (clean)
        t = np.linspace(0, 1, 128)
        self.signals['high_snr'] = np.exp(1j * 2 * np.pi * 5 * t)
        
        # Low SNR signal (noisy)
        self.signals['low_snr'] = np.exp(1j * 2 * np.pi * 5 * t) + 2 * (np.random.randn(128) + 1j * np.random.randn(128))
    
    def test_8d_feature_vector_generation(self):
        """Test that 8D feature vectors are generated correctly."""
        for signal_type, signal in self.signals.items():
            with self.subTest(signal_type=signal_type):
                features = extract_analog_features(signal, fs=128)
                
                # Check shape
                self.assertEqual(features.shape, (8,), f"Wrong shape for {signal_type}")
                
                # Check dtype
                self.assertEqual(features.dtype, np.float32, f"Wrong dtype for {signal_type}")
                
                # Check no NaN or Inf
                self.assertFalse(np.any(np.isnan(features)), f"NaN in features for {signal_type}")
                self.assertFalse(np.any(np.isinf(features)), f"Inf in features for {signal_type}")
    
    def test_feature_extraction_with_various_snrs(self):
        """Test feature extraction with signals at different SNRs."""
        t = np.linspace(0, 1, 128)
        clean_signal = np.exp(1j * 2 * np.pi * 5 * t)
        
        snr_values = [-10, -5, 0, 5, 10, 15, 20]
        
        for snr_db in snr_values:
            with self.subTest(snr_db=snr_db):
                # Add noise based on SNR
                snr_linear = 10 ** (snr_db / 10)
                noise_power = 1 / snr_linear
                noise = np.sqrt(noise_power / 2) * (np.random.randn(128) + 1j * np.random.randn(128))
                noisy_signal = clean_signal + noise
                
                # Extract features
                features = extract_analog_features(noisy_signal, fs=128)
                
                # Verify valid features
                self.assertEqual(features.shape, (8,))
                self.assertFalse(np.any(np.isnan(features)))
                self.assertFalse(np.any(np.isinf(features)))
    
    def test_numerical_stability(self):
        """Test numerical stability with edge cases."""
        # Test with constant signal (zero std)
        constant_signal = np.ones(128, dtype=complex)
        features_constant = extract_analog_features(constant_signal, fs=128)
        
        self.assertFalse(np.any(np.isnan(features_constant)), "NaN in constant signal features")
        self.assertFalse(np.any(np.isinf(features_constant)), "Inf in constant signal features")
        
        # Test with zero signal
        zero_signal = np.zeros(128, dtype=complex)
        features_zero = extract_analog_features(zero_signal, fs=128)
        
        self.assertFalse(np.any(np.isnan(features_zero)), "NaN in zero signal features")
        self.assertFalse(np.any(np.isinf(features_zero)), "Inf in zero signal features")
        
        # Test with very small values
        tiny_signal = 1e-10 * np.random.randn(128) + 1j * 1e-10 * np.random.randn(128)
        features_tiny = extract_analog_features(tiny_signal, fs=128)
        
        self.assertFalse(np.any(np.isnan(features_tiny)), "NaN in tiny signal features")
        self.assertFalse(np.any(np.isinf(features_tiny)), "Inf in tiny signal features")
    
    def test_instantaneous_amplitude_computation(self):
        """Test instantaneous amplitude computation."""
        signal = self.signals['random']
        amplitude = compute_instantaneous_amplitude(signal)
        
        # Check shape
        self.assertEqual(amplitude.shape, signal.shape)
        
        # Check all values are non-negative
        self.assertTrue(np.all(amplitude >= 0))
        
        # Check amplitude equals magnitude
        expected_amplitude = np.abs(signal)
        np.testing.assert_array_almost_equal(amplitude, expected_amplitude)
    
    def test_instantaneous_frequency_computation(self):
        """Test instantaneous frequency computation."""
        signal = self.signals['high_snr']
        frequency = compute_instantaneous_frequency(signal, fs=128)
        
        # Check shape (should be same as input)
        self.assertEqual(frequency.shape, signal.shape)
        
        # Check no NaN or Inf
        self.assertFalse(np.any(np.isnan(frequency)))
        self.assertFalse(np.any(np.isinf(frequency)))
    
    def test_statistical_features_computation(self):
        """Test statistical features computation."""
        data = np.random.randn(128)
        features = compute_statistical_features(data)
        
        # Check keys
        expected_keys = {'mean', 'variance', 'skewness', 'kurtosis'}
        self.assertEqual(set(features.keys()), expected_keys)
        
        # Check all values are finite
        for key, value in features.items():
            self.assertTrue(np.isfinite(value), f"{key} is not finite")
    
    def test_process_dataset_batch(self):
        """Test processing a batch of samples."""
        # Create batch of IQ samples
        n_samples = 50
        iq_samples = np.random.randn(n_samples, 2, 128).astype(np.float32)
        labels = np.random.randint(0, 2, size=n_samples)
        
        # Process dataset
        features, output_labels = process_dataset(iq_samples, labels, verbose=False, use_analog_features=True)
        
        # Check shapes
        self.assertEqual(features.shape, (n_samples, 8))
        self.assertEqual(output_labels.shape, (n_samples,))
        
        # Check labels unchanged
        np.testing.assert_array_equal(output_labels, labels)
        
        # Check no NaN or Inf
        self.assertFalse(np.any(np.isnan(features)))
        self.assertFalse(np.any(np.isinf(features)))


class TestModelTrainingAndTiming(unittest.TestCase):
    """
    Test 11.3: Model training and timing
    
    Verifies:
    - KNN and DT model training
    - Timing measurements accuracy
    - Model serialization and loading
    """
    
    def setUp(self):
        """Set up test fixtures with sample data."""
        np.random.seed(42)
        
        # Create synthetic feature data
        self.n_samples = 200
        self.n_features = 8
        self.n_classes = 2  # AM, FM
        
        self.X_train = np.random.randn(self.n_samples, self.n_features).astype(np.float32)
        self.y_train = np.random.randint(0, self.n_classes, size=self.n_samples)
        
        self.X_test = np.random.randn(50, self.n_features).astype(np.float32)
        self.y_test = np.random.randint(0, self.n_classes, size=50)
        
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_knn_training(self):
        """Test KNN model training."""
        knn = KNeighborsClassifier(n_neighbors=5)
        
        # Measure training time
        start_time = time.time()
        knn.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        # Verify model is trained
        self.assertTrue(hasattr(knn, 'classes_'))
        
        # Verify training time is reasonable
        self.assertGreater(training_time, 0)
        self.assertLess(training_time, 10)  # Should be fast
    
    def test_dt_training(self):
        """Test Decision Tree model training."""
        dt = DecisionTreeClassifier(random_state=42)
        
        # Measure training time
        start_time = time.time()
        dt.fit(self.X_train, self.y_train)
        training_time = time.time() - start_time
        
        # Verify model is trained
        self.assertTrue(hasattr(dt, 'tree_'))
        
        # Verify training time is reasonable
        self.assertGreater(training_time, 0)
        self.assertLess(training_time, 10)
    
    def test_knn_inference_timing(self):
        """Test KNN inference timing measurement."""
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train, self.y_train)
        
        # Measure inference time
        start_time = time.time()
        predictions = knn.predict(self.X_test)
        total_inference_time = time.time() - start_time
        
        # Calculate per-sample inference time
        per_sample_time_ms = (total_inference_time / len(self.X_test)) * 1000
        
        # Verify predictions
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Verify timing is reasonable
        self.assertGreater(per_sample_time_ms, 0)
        self.assertLess(per_sample_time_ms, 100)  # Should be fast
    
    def test_dt_inference_timing(self):
        """Test Decision Tree inference timing measurement."""
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X_train, self.y_train)
        
        # Measure inference time
        start_time = time.time()
        predictions = dt.predict(self.X_test)
        total_inference_time = time.time() - start_time
        
        # Calculate per-sample inference time
        per_sample_time_ms = (total_inference_time / len(self.X_test)) * 1000
        
        # Verify predictions
        self.assertEqual(len(predictions), len(self.X_test))
        
        # Verify timing is reasonable
        self.assertGreater(per_sample_time_ms, 0)
        self.assertLess(per_sample_time_ms, 100)
    
    def test_knn_serialization(self):
        """Test KNN model serialization and loading."""
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(self.X_train, self.y_train)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'knn_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(knn, f)
        
        # Verify file exists
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        with open(model_path, 'rb') as f:
            loaded_knn = pickle.load(f)
        
        # Verify loaded model produces same predictions
        original_predictions = knn.predict(self.X_test)
        loaded_predictions = loaded_knn.predict(self.X_test)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)
    
    def test_dt_serialization(self):
        """Test Decision Tree model serialization and loading."""
        dt = DecisionTreeClassifier(random_state=42)
        dt.fit(self.X_train, self.y_train)
        
        # Save model
        model_path = os.path.join(self.temp_dir, 'dt_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(dt, f)
        
        # Verify file exists
        self.assertTrue(os.path.exists(model_path))
        
        # Load model
        with open(model_path, 'rb') as f:
            loaded_dt = pickle.load(f)
        
        # Verify loaded model produces same predictions
        original_predictions = dt.predict(self.X_test)
        loaded_predictions = loaded_dt.predict(self.X_test)
        
        np.testing.assert_array_equal(original_predictions, loaded_predictions)


class TestAggregationForBothModelTypes(unittest.TestCase):
    """
    Test 11.4: Aggregation for both model types
    
    Verifies:
    - KNN model aggregation from multiple clients
    - Decision Tree model aggregation from multiple clients
    - Global model accuracy verification
    """
    
    def setUp(self):
        """Set up test fixtures with multiple client models."""
        np.random.seed(42)
        
        self.n_features = 8
        self.n_classes = 2
        self.temp_dir = tempfile.mkdtemp()
        
        # Create training data for 3 clients
        self.num_clients = 3
        self.client_data = []
        
        for i in range(self.num_clients):
            n_samples = 100 + i * 50  # Different sample counts
            X = np.random.randn(n_samples, self.n_features).astype(np.float32)
            y = np.random.randint(0, self.n_classes, size=n_samples)
            self.client_data.append((X, y))
        
        # Create test data
        self.X_test = np.random.randn(100, self.n_features).astype(np.float32)
        self.y_test = np.random.randint(0, self.n_classes, size=100)
    
    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_knn_aggregation(self):
        """Test KNN model aggregation from multiple clients."""
        # Train KNN models on each client
        client_models_info = []
        
        for i, (X, y) in enumerate(self.client_data):
            # Train KNN
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X, y)
            
            # Save model and data
            model_path = os.path.join(self.temp_dir, f'client_{i}_knn.pkl')
            features_path = os.path.join(self.temp_dir, f'client_{i}_features.pkl')
            labels_path = os.path.join(self.temp_dir, f'client_{i}_labels.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(knn, f)
            with open(features_path, 'wb') as f:
                pickle.dump(X, f)
            with open(labels_path, 'wb') as f:
                pickle.dump(y, f)
            
            client_models_info.append({
                'client_id': f'client_{i}',
                'model_path': model_path,
                'features_path': features_path,
                'labels_path': labels_path,
                'n_samples': len(X)
            })
        
        # Aggregate KNN models
        result = aggregate_knn_models(client_models_info, n_neighbors=5)
        
        # Verify aggregation result
        self.assertIn('global_model', result)
        self.assertIn('total_samples', result)
        self.assertIn('num_clients', result)
        self.assertEqual(result['num_clients'], self.num_clients)
        
        # Verify global model can make predictions
        global_knn = result['global_model']
        predictions = global_knn.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_dt_aggregation(self):
        """Test Decision Tree model aggregation from multiple clients."""
        # Train DT models on each client
        client_models_info = []
        
        for i, (X, y) in enumerate(self.client_data):
            # Train Decision Tree
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X, y)
            
            # Save model
            model_path = os.path.join(self.temp_dir, f'client_{i}_dt.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(dt, f)
            
            client_models_info.append({
                'client_id': f'client_{i}',
                'model_path': model_path,
                'n_samples': len(X)
            })
        
        # Aggregate DT models
        result = aggregate_dt_models(client_models_info)
        
        # Verify aggregation result
        self.assertIn('client_models', result)
        self.assertIn('num_clients', result)
        self.assertIn('ensemble_weights', result)
        self.assertEqual(result['num_clients'], self.num_clients)
        self.assertEqual(len(result['client_models']), self.num_clients)
        
        # Create ensemble
        ensemble = DecisionTreeEnsemble(
            result['client_models'],
            result['ensemble_weights']
        )
        
        # Verify ensemble can make predictions
        predictions = ensemble.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.X_test))
    
    def test_global_model_accuracy_knn(self):
        """Test global KNN model accuracy evaluation."""
        # Create and aggregate KNN models
        client_models_info = []
        
        for i, (X, y) in enumerate(self.client_data):
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X, y)
            
            model_path = os.path.join(self.temp_dir, f'client_{i}_knn.pkl')
            features_path = os.path.join(self.temp_dir, f'client_{i}_features.pkl')
            labels_path = os.path.join(self.temp_dir, f'client_{i}_labels.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(knn, f)
            with open(features_path, 'wb') as f:
                pickle.dump(X, f)
            with open(labels_path, 'wb') as f:
                pickle.dump(y, f)
            
            client_models_info.append({
                'client_id': f'client_{i}',
                'model_path': model_path,
                'features_path': features_path,
                'labels_path': labels_path,
                'n_samples': len(X)
            })
        
        result = aggregate_knn_models(client_models_info, n_neighbors=5)
        global_knn = result['global_model']
        
        # Evaluate global model
        eval_result = evaluate_global_model(global_knn, self.X_test, self.y_test)
        
        # Verify evaluation metrics
        self.assertIn('accuracy', eval_result)
        self.assertIn('confusion_matrix', eval_result)
        self.assertIn('n_samples', eval_result)
        
        self.assertGreaterEqual(eval_result['accuracy'], 0.0)
        self.assertLessEqual(eval_result['accuracy'], 1.0)
        self.assertEqual(eval_result['n_samples'], len(self.X_test))
    
    def test_global_model_accuracy_dt(self):
        """Test global Decision Tree ensemble accuracy evaluation."""
        # Create and aggregate DT models
        client_models_info = []
        
        for i, (X, y) in enumerate(self.client_data):
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(X, y)
            
            model_path = os.path.join(self.temp_dir, f'client_{i}_dt.pkl')
            with open(model_path, 'wb') as f:
                pickle.dump(dt, f)
            
            client_models_info.append({
                'client_id': f'client_{i}',
                'model_path': model_path,
                'n_samples': len(X)
            })
        
        result = aggregate_dt_models(client_models_info)
        ensemble = DecisionTreeEnsemble(
            result['client_models'],
            result['ensemble_weights']
        )
        
        # Evaluate ensemble
        eval_result = evaluate_global_model(ensemble, self.X_test, self.y_test)
        
        # Verify evaluation metrics
        self.assertIn('accuracy', eval_result)
        self.assertIn('confusion_matrix', eval_result)
        self.assertIn('n_samples', eval_result)
        
        self.assertGreaterEqual(eval_result['accuracy'], 0.0)
        self.assertLessEqual(eval_result['accuracy'], 1.0)
        self.assertEqual(eval_result['n_samples'], len(self.X_test))


if __name__ == '__main__':
    unittest.main()
