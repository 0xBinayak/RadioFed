"""
Test Dashboard Metrics Collection

This script tests that the enhanced aggregation functions properly collect
and store metrics for dashboard visualization.
"""

import pytest
import sys
import os
import numpy as np
import pickle
import tempfile
import shutil
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from central.aggregator import aggregate_knn_models, generate_synthetic_snr_values
from sklearn.neighbors import KNeighborsClassifier


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def test_synthetic_snr_generation():
    """Test synthetic SNR value generation."""
    n_samples = 1000
    snr_values = generate_synthetic_snr_values(n_samples)
    
    print(f"Generated {len(snr_values)} SNR values")
    print(f"Unique SNR levels: {sorted(set(snr_values))}")
    print(f"SNR range: {min(snr_values)} to {max(snr_values)} dB")
    
    
    unique, counts = np.unique(snr_values, return_counts=True)
    print(f"\nDistribution:")
    for snr, count in zip(unique, counts):
        print(f"  SNR {snr:>3} dB: {count:>4} samples")
    
    assert len(snr_values) == n_samples, "Wrong number of SNR values"
    assert min(snr_values) == -20, "Wrong minimum SNR"
    assert max(snr_values) == 18, "Wrong maximum SNR"


def test_knn_aggregation_with_metrics(temp_dir):
    """Test KNN aggregation with metrics collection."""
    
    n_clients = 3
    client_models_info = []
    
    for i in range(n_clients):
        
        n_samples = 500 + i * 100
        features = np.random.randn(n_samples, 8).astype(np.float32)
        labels = np.random.randint(0, 2, size=n_samples)
        
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels)
        
        
        model_path = os.path.join(temp_dir, f'client_{i}_knn_model.pkl')
        features_path = os.path.join(temp_dir, f'client_{i}_features.pkl')
        labels_path = os.path.join(temp_dir, f'client_{i}_labels.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(knn, f)
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f)
        
        client_models_info.append({
            'client_id': f'test_client_{i}',
            'model_path': model_path,
            'features_path': features_path,
            'labels_path': labels_path,
            'n_samples': n_samples
        })
        
        print(f"Created client {i}: {n_samples} samples")
    
    
    print("\nPerforming KNN aggregation with evaluation...")
    result = aggregate_knn_models(client_models_info, n_neighbors=5, evaluate=True)
    
    
    print("\nAggregation Result:")
    print(f"  Number of clients: {result['num_clients']}")
    print(f"  Total samples: {result['total_samples']}")
    print(f"  Feature dimension: {result['feature_dim']}")
    print(f"  Training time: {result['training_time']:.3f}s")
    
    if 'accuracy' in result:
        print(f"  Overall accuracy: {result['accuracy']*100:.2f}%")
        print(f"  Inference time: {result['inference_time_ms_per_sample']:.3f} ms/sample")
        print(f"  Test samples: {result['n_test_samples']}")
    
    
    required_fields = ['global_model', 'num_clients', 'total_samples', 'training_time']
    for field in required_fields:
        assert field in result, f"Missing required field: {field}"
    
    
    if 'accuracy' in result:
        eval_fields = ['accuracy', 'per_snr_accuracy', 'confusion_matrix', 'inference_time_ms_per_sample']
        for field in eval_fields:
            assert field in result, f"Missing evaluation field: {field}"

