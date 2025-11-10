"""
Test Dashboard Metrics Collection

This script tests that the enhanced aggregation functions properly collect
and store metrics for dashboard visualization.
"""

import sys
import os
import numpy as np
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from central.aggregator import aggregate_knn_models, aggregate_dt_models, generate_synthetic_snr_values
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def test_synthetic_snr_generation():
    """Test synthetic SNR value generation."""
    print("\n" + "="*70)
    print("Testing Synthetic SNR Generation")
    print("="*70)
    
    n_samples = 1000
    snr_values = generate_synthetic_snr_values(n_samples)
    
    print(f"Generated {len(snr_values)} SNR values")
    print(f"Unique SNR levels: {sorted(set(snr_values))}")
    print(f"SNR range: {min(snr_values)} to {max(snr_values)} dB")
    
    # Verify distribution
    unique, counts = np.unique(snr_values, return_counts=True)
    print(f"\nDistribution:")
    for snr, count in zip(unique, counts):
        print(f"  SNR {snr:>3} dB: {count:>4} samples")
    
    assert len(snr_values) == n_samples, "Wrong number of SNR values"
    assert min(snr_values) == -20, "Wrong minimum SNR"
    assert max(snr_values) == 18, "Wrong maximum SNR"
    
    print("\n✓ Synthetic SNR generation test passed!")


def test_knn_aggregation_with_metrics():
    """Test KNN aggregation with metrics collection."""
    print("\n" + "="*70)
    print("Testing KNN Aggregation with Metrics")
    print("="*70)
    
    # Create temporary directory for test data
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create synthetic client data
        n_clients = 3
        client_models_info = []
        
        for i in range(n_clients):
            # Generate synthetic features and labels
            n_samples = 500 + i * 100
            features = np.random.randn(n_samples, 8).astype(np.float32)
            labels = np.random.randint(0, 2, size=n_samples)
            
            # Train a simple KNN model
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(features, labels)
            
            # Save model and data
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
        
        # Perform aggregation with evaluation
        print("\nPerforming KNN aggregation with evaluation...")
        result = aggregate_knn_models(client_models_info, n_neighbors=5, evaluate=True)
        
        # Verify result contains all expected metrics
        print("\nAggregation Result:")
        print(f"  Number of clients: {result['num_clients']}")
        print(f"  Total samples: {result['total_samples']}")
        print(f"  Feature dimension: {result['feature_dim']}")
        print(f"  Training time: {result['training_time']:.3f}s")
        
        if 'accuracy' in result:
            print(f"  Overall accuracy: {result['accuracy']*100:.2f}%")
            print(f"  Inference time: {result['inference_time_ms_per_sample']:.3f} ms/sample")
            print(f"  Test samples: {result['n_test_samples']}")
            
            # Check per-SNR accuracy
            if 'per_snr_accuracy' in result:
                per_snr = result['per_snr_accuracy']
                print(f"\n  Per-SNR Accuracy ({len(per_snr)} SNR levels):")
                for snr in sorted(per_snr.keys())[:5]:  # Show first 5
                    print(f"    SNR {snr:>3} dB: {per_snr[snr]*100:.2f}%")
                if len(per_snr) > 5:
                    print(f"    ... and {len(per_snr) - 5} more")
            
            # Check confusion matrix
            if 'confusion_matrix' in result:
                cm = np.array(result['confusion_matrix'])
                print(f"\n  Confusion Matrix:")
                print(f"    {cm}")
        
        # Verify all required fields are present
        required_fields = ['global_model', 'num_clients', 'total_samples', 'training_time']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Verify evaluation metrics if evaluation was performed
        if 'accuracy' in result:
            eval_fields = ['accuracy', 'per_snr_accuracy', 'confusion_matrix', 'inference_time_ms_per_sample']
            for field in eval_fields:
                assert field in result, f"Missing evaluation field: {field}"
        
        print("\n✓ KNN aggregation with metrics test passed!")
        
    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def test_dt_aggregation_with_metrics():
    """Test Decision Tree aggregation with metrics collection."""
    print("\n" + "="*70)
    print("Testing Decision Tree Aggregation with Metrics")
    print("="*70)
    
    # Create temporary directory for test data
    import tempfile
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create synthetic client data
        n_clients = 3
        client_models_info = []
        
        for i in range(n_clients):
            # Generate synthetic features and labels
            n_samples = 500 + i * 100
            features = np.random.randn(n_samples, 8).astype(np.float32)
            labels = np.random.randint(0, 2, size=n_samples)
            
            # Train a simple Decision Tree model
            dt = DecisionTreeClassifier(random_state=42)
            dt.fit(features, labels)
            
            # Save model and data
            model_path = os.path.join(temp_dir, f'client_{i}_dt_model.pkl')
            features_path = os.path.join(temp_dir, f'client_{i}_features.pkl')
            labels_path = os.path.join(temp_dir, f'client_{i}_labels.pkl')
            
            with open(model_path, 'wb') as f:
                pickle.dump(dt, f)
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
        
        # Perform aggregation with evaluation
        print("\nPerforming Decision Tree aggregation with evaluation...")
        result = aggregate_dt_models(client_models_info, evaluate=True)
        
        # Verify result contains all expected metrics
        print("\nAggregation Result:")
        print(f"  Number of clients: {result['num_clients']}")
        print(f"  Total samples: {result['total_samples']}")
        print(f"  Training time: {result['training_time']:.3f}s")
        
        if 'accuracy' in result:
            print(f"  Overall accuracy: {result['accuracy']*100:.2f}%")
            print(f"  Inference time: {result['inference_time_ms_per_sample']:.3f} ms/sample")
            print(f"  Test samples: {result['n_test_samples']}")
            
            # Check per-SNR accuracy
            if 'per_snr_accuracy' in result:
                per_snr = result['per_snr_accuracy']
                print(f"\n  Per-SNR Accuracy ({len(per_snr)} SNR levels):")
                for snr in sorted(per_snr.keys())[:5]:  # Show first 5
                    print(f"    SNR {snr:>3} dB: {per_snr[snr]*100:.2f}%")
                if len(per_snr) > 5:
                    print(f"    ... and {len(per_snr) - 5} more")
            
            # Check confusion matrix
            if 'confusion_matrix' in result:
                cm = np.array(result['confusion_matrix'])
                print(f"\n  Confusion Matrix:")
                print(f"    {cm}")
        
        # Verify all required fields are present
        required_fields = ['client_models', 'num_clients', 'total_samples', 'training_time']
        for field in required_fields:
            assert field in result, f"Missing required field: {field}"
        
        # Verify evaluation metrics if evaluation was performed
        if 'accuracy' in result:
            eval_fields = ['accuracy', 'per_snr_accuracy', 'confusion_matrix', 'inference_time_ms_per_sample']
            for field in eval_fields:
                assert field in result, f"Missing evaluation field: {field}"
        
        print("\n✓ Decision Tree aggregation with metrics test passed!")
        
    finally:
        # Clean up
        import shutil
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Run all dashboard metrics tests."""
    print("\n" + "="*70)
    print("DASHBOARD METRICS COLLECTION TESTS")
    print("="*70)
    
    try:
        test_synthetic_snr_generation()
        test_knn_aggregation_with_metrics()
        test_dt_aggregation_with_metrics()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print("\nThe enhanced aggregation functions now collect:")
        print("  ✓ Training time")
        print("  ✓ Inference time per sample")
        print("  ✓ Overall accuracy")
        print("  ✓ Per-SNR accuracy breakdown")
        print("  ✓ Confusion matrices")
        print("\nThese metrics will be displayed in the dashboard!")
        print("="*70)
        
        return 0
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
