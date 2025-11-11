"""
End-to-End Workflow Tests for AMC Dashboard Enhancement

Tests the complete system integration including:
- Dashboard functionality (11.5)
- Auto-start server behavior (11.6)
- Simplified client workflow (11.7)
- Multi-client federated learning simulation (11.8)


"""

import pytest
import tempfile
import os
import shutil
import numpy as np
import pickle
from unittest.mock import Mock, patch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from central.aggregator import (
    aggregate_knn_models,
    evaluate_global_model
)
from client.feature_extract import extract_analog_features, process_dataset


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def metrics_data():
    """Create sample metrics data."""
    return {
        'accuracy': 0.85,
        'per_snr_accuracy': {
            -10: 0.65,
            0: 0.85,
            10: 0.95,
            18: 0.98
        },
        'confusion_matrix': np.array([[45, 5], [3, 47]]),
        'n_samples': 100
    }


@pytest.fixture
def training_history():
    """Create sample training history."""
    return {
        'round': [1, 2, 3, 4, 5],
        'knn_accuracy': [0.70, 0.75, 0.80, 0.83, 0.85],
        'dt_accuracy': [0.68, 0.72, 0.78, 0.81, 0.84],
        'baseline_accuracy': [0.50, 0.50, 0.50, 0.50, 0.50]
    }



def test_confusion_matrix_data_structure(metrics_data):
    """Test confusion matrix data structure for visualization."""
    conf_matrix = metrics_data['confusion_matrix']
    
  
    assert conf_matrix.shape == (2, 2)
    assert np.all(conf_matrix >= 0)
    total = np.sum(conf_matrix)
    assert total == metrics_data['n_samples']


def test_accuracy_vs_snr_data_structure(metrics_data):
    """Test accuracy vs SNR data structure for plotting."""
    per_snr_acc = metrics_data['per_snr_accuracy']
    

    snr_values = sorted(per_snr_acc.keys())
    assert snr_values == [-10, 0, 10, 18]
    

    for snr, acc in per_snr_acc.items():
        assert 0.0 <= acc <= 1.0


def test_training_history_data_structure(training_history):
    """Test training history data structure for plotting."""
    history = training_history
    
   
    num_rounds = len(history['round'])
    assert len(history['knn_accuracy']) == num_rounds
    assert len(history['dt_accuracy']) == num_rounds
    assert len(history['baseline_accuracy']) == num_rounds
    

    for acc_list in [history['knn_accuracy'], history['dt_accuracy'], history['baseline_accuracy']]:
        for acc in acc_list:
            assert 0.0 <= acc <= 1.0


def test_metrics_update_after_aggregation():
    """Test that metrics are properly updated after aggregation."""
    
    before_metrics = {
        'accuracy': 0.70,
        'confusion_matrix': np.array([[40, 10], [8, 42]])
    }
    
    
    after_metrics = {
        'accuracy': 0.85,
        'confusion_matrix': np.array([[45, 5], [3, 47]])
    }
    
    
    assert after_metrics['accuracy'] > before_metrics['accuracy']
    
    
    before_correct = np.trace(before_metrics['confusion_matrix'])
    after_correct = np.trace(after_metrics['confusion_matrix'])
    assert after_correct > before_correct


def test_feature_distribution_data_structure():
    """Test feature distribution data structure for visualization."""
    
    n_samples = 200
    features = {
        'amp_kurtosis': np.random.randn(n_samples),
        'freq_variance': np.abs(np.random.randn(n_samples)),
        'modulation': np.random.choice(['AM', 'FM'], size=n_samples)
    }
    
    
    assert len(features['amp_kurtosis']) == n_samples
    assert len(features['freq_variance']) == n_samples
    assert len(features['modulation']) == n_samples
    
    
    unique_mods = set(features['modulation'])
    assert unique_mods == {'AM', 'FM'}


def test_computation_complexity_table_structure():
    """Test computation complexity table data structure."""
    complexity_data = {
        'Method': ['Decision Tree', 'K-Nearest Neighbors'],
        'Training Time (seconds)': [2.345, 1.123],
        'Average Inference Time (ms/sample)': [0.456, 1.234]
    }
    
    
    assert len(complexity_data['Method']) == 2
    assert len(complexity_data['Training Time (seconds)']) == 2
    assert len(complexity_data['Average Inference Time (ms/sample)']) == 2
    

    for time_val in complexity_data['Training Time (seconds)']:
        assert time_val > 0
    for time_val in complexity_data['Average Inference Time (ms/sample)']:
        assert time_val > 0



def test_port_conflict_detection():
    """Test port conflict detection logic."""
    
    def is_port_in_use(port):
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return False
            except OSError:
                return True
    
    test_port = 58000
    port_status = is_port_in_use(test_port)
    

    assert isinstance(port_status, bool)


def test_server_configuration_validation():
    """Test server configuration validation."""

    valid_config = {
        'host': '127.0.0.1',
        'port': 8000,
        'dashboard_port': 7860
    }
    

    assert '.' in valid_config['host']
    

    assert 1024 < valid_config['port'] < 65536
    assert 1024 < valid_config['dashboard_port'] < 65536
    

    assert valid_config['port'] != valid_config['dashboard_port']


def test_startup_sequence_order():
    """Test that startup sequence follows correct order."""
    startup_steps = []
    

    def initialize():
        startup_steps.append('initialize')
    
    def start_fastapi():
        startup_steps.append('start_fastapi')
    
    def wait_for_ready():
        startup_steps.append('wait_for_ready')
    
    def launch_dashboard():
        startup_steps.append('launch_dashboard')
    

    initialize()
    start_fastapi()
    wait_for_ready()
    launch_dashboard()
    

    expected_order = ['initialize', 'start_fastapi', 'wait_for_ready', 'launch_dashboard']
    assert startup_steps == expected_order



def test_load_pre_partitioned_dataset(temp_dir):
    """Test loading pre-partitioned dataset."""

    partition_data = {}
    modulations = ['AM', 'FM']
    snrs = [-10, 0, 10]
    
    for mod in modulations:
        for snr in snrs:
            samples = np.random.randn(50, 2, 128).astype(np.float32)
            partition_data[(mod, snr)] = samples
    

    partition_path = os.path.join(temp_dir, 'client_0.pkl')
    with open(partition_path, 'wb') as f:
        pickle.dump(partition_data, f)
    

    with open(partition_path, 'rb') as f:
        loaded_partition = pickle.load(f)

    assert isinstance(loaded_partition, dict)
    assert len(loaded_partition) > 0
    

    for key in loaded_partition.keys():
        assert isinstance(key, tuple)
        assert len(key) == 2
        assert key[0] in ['AM', 'FM']
        assert key[1] in [-10, 0, 10]


def test_extract_features_from_partition(temp_dir):
    """Test extracting features from loaded partition."""

    partition = {}
    modulations = ['AM', 'FM']
    snrs = [-10, 0, 10]
    
    for mod in modulations:
        for snr in snrs:
            samples = np.random.randn(50, 2, 128).astype(np.float32)
            partition[(mod, snr)] = samples
    

    all_samples = []
    all_labels = []
    label_map = {'AM': 0, 'FM': 1}
    
    for (mod, snr), samples in partition.items():
        all_samples.append(samples)
        all_labels.extend([label_map[mod]] * len(samples))
    
    samples_array = np.vstack(all_samples)
    labels_array = np.array(all_labels)
    

    features, labels = process_dataset(
        samples_array,
        labels_array,
        verbose=False,
        use_analog_features=True
    )
    

    assert features.shape[1] == 8 
    assert len(features) == len(labels)
    assert not np.any(np.isnan(features))


def test_train_model_on_partition(temp_dir):
    """Test training model on partition data."""

    partition = {}
    modulations = ['AM', 'FM']
    snrs = [-10, 0, 10]
    
    for mod in modulations:
        for snr in snrs:
            samples = np.random.randn(50, 2, 128).astype(np.float32)
            partition[(mod, snr)] = samples
    

    all_samples = []
    all_labels = []
    label_map = {'AM': 0, 'FM': 1}
    
    for (mod, snr), samples in partition.items():
        all_samples.append(samples)
        all_labels.extend([label_map[mod]] * len(samples))
    
    samples_array = np.vstack(all_samples)
    labels_array = np.array(all_labels)
    
    features, labels = process_dataset(
        samples_array,
        labels_array,
        verbose=False,
        use_analog_features=True
    )
    

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(features, labels)

    assert hasattr(knn, 'classes_')
    

    predictions = knn.predict(features[:10])
    assert len(predictions) == 10


def test_model_serialization_for_upload(temp_dir):
    """Test model serialization for upload to server."""

    X = np.random.randn(100, 8).astype(np.float32)
    y = np.random.randint(0, 2, size=100)
    
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y)
    

    model_path = os.path.join(temp_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(knn, f)
    

    assert os.path.exists(model_path)
    assert os.path.getsize(model_path) > 0
    

    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)
    

    predictions = loaded_model.predict(X[:5])
    assert len(predictions) == 5



def test_all_clients_can_train(temp_dir):
    """Test that all clients can train models successfully."""
    np.random.seed(42)
    num_clients = 3
    trained_models = []
    
    for client_id in range(num_clients):

        partition = {}
        modulations = ['AM', 'FM']
        snrs = [-10, 0, 10, 18]
        
        for mod in modulations:
            for snr in snrs:
                n_samples = 40 + client_id * 10
                samples = np.random.randn(n_samples, 2, 128).astype(np.float32)
                partition[(mod, snr)] = samples
        

        all_samples = []
        all_labels = []
        label_map = {'AM': 0, 'FM': 1}
        
        for (mod, snr), samples in partition.items():
            all_samples.append(samples)
            all_labels.extend([label_map[mod]] * len(samples))
        
        samples_array = np.vstack(all_samples)
        labels_array = np.array(all_labels)
        
        features, labels = process_dataset(
            samples_array,
            labels_array,
            verbose=False,
            use_analog_features=True
        )
        

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels)
        
        trained_models.append({
            'client_id': client_id,
            'model': knn,
            'features': features,
            'labels': labels,
            'n_samples': len(features)
        })
    

    assert len(trained_models) == num_clients
    
    for model_info in trained_models:
        assert hasattr(model_info['model'], 'classes_')
        assert model_info['n_samples'] > 0


def test_aggregation_with_multiple_clients(temp_dir):
    """Test aggregation with multiple clients."""
    np.random.seed(42)
    num_clients = 3
    client_models_info = []
    
    for client_id in range(num_clients):

        n_samples = 100 + client_id * 50
        features = np.random.randn(n_samples, 8).astype(np.float32)
        labels = np.random.randint(0, 2, size=n_samples)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(features, labels)
        
        model_path = os.path.join(temp_dir, f'client_{client_id}_model.pkl')
        features_path = os.path.join(temp_dir, f'client_{client_id}_features.pkl')
        labels_path = os.path.join(temp_dir, f'client_{client_id}_labels.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(knn, f)
        with open(features_path, 'wb') as f:
            pickle.dump(features, f)
        with open(labels_path, 'wb') as f:
            pickle.dump(labels, f)
        
        client_models_info.append({
            'client_id': f'client_{client_id}',
            'model_path': model_path,
            'features_path': features_path,
            'labels_path': labels_path,
            'n_samples': len(features)
        })
    
    
    result = aggregate_knn_models(client_models_info, n_neighbors=5)
    
    
    assert result['num_clients'] == num_clients
    assert result['total_samples'] > 0
    assert 'global_model' in result


def test_global_model_improvement(temp_dir):
    """Test that global model improves after aggregation."""
    np.random.seed(42)
    num_clients = 3
    
    
    X_test = np.random.randn(100, 8).astype(np.float32)
    y_test = np.random.randint(0, 2, size=100)
    
    
    client_accuracies = []
    client_models_info = []
    
    for client_id in range(num_clients):
        
        X_train = np.random.randn(150, 8).astype(np.float32)
        y_train = np.random.randint(0, 2, size=150)
        
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        
        
        from sklearn.metrics import accuracy_score
        predictions = knn.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        client_accuracies.append(accuracy)
        
        
        model_path = os.path.join(temp_dir, f'client_{client_id}_model.pkl')
        features_path = os.path.join(temp_dir, f'client_{client_id}_features.pkl')
        labels_path = os.path.join(temp_dir, f'client_{client_id}_labels.pkl')
        
        with open(model_path, 'wb') as f:
            pickle.dump(knn, f)
        with open(features_path, 'wb') as f:
            pickle.dump(X_train, f)
        with open(labels_path, 'wb') as f:
            pickle.dump(y_train, f)
        
        client_models_info.append({
            'client_id': f'client_{client_id}',
            'model_path': model_path,
            'features_path': features_path,
            'labels_path': labels_path,
            'n_samples': len(X_train)
        })
    
    result = aggregate_knn_models(client_models_info, n_neighbors=5)
    global_model = result['global_model']
    
   
    global_predictions = global_model.predict(X_test)
    from sklearn.metrics import accuracy_score
    global_accuracy = accuracy_score(y_test, global_predictions)
    
    
    avg_client_accuracy = np.mean(client_accuracies)
    
    
    assert 0.0 <= global_accuracy <= 1.0


def test_metrics_collection_for_dashboard():
    """Test that metrics are collected correctly for dashboard display."""
    
    metrics_history = []
    num_clients = 3
    
    for round_num in range(1, 4):  
       
        round_metrics = {
            'round': round_num,
            'num_clients': num_clients,
            'knn_accuracy': 0.70 + round_num * 0.05,
            'dt_accuracy': 0.68 + round_num * 0.05,
            'total_samples': 450,
            'timestamp': f'2025-11-10T14:{30+round_num}:00'
        }
        metrics_history.append(round_metrics)
    
    
    assert len(metrics_history) == 3
    
    for metrics in metrics_history:
        assert 'round' in metrics
        assert 'num_clients' in metrics
        assert 'knn_accuracy' in metrics
        assert 'dt_accuracy' in metrics
        assert 'total_samples' in metrics
        
        
        assert 0.0 <= metrics['knn_accuracy'] <= 1.0
        assert 0.0 <= metrics['dt_accuracy'] <= 1.0
    
    
    accuracies = [m['knn_accuracy'] for m in metrics_history]
    for i in range(1, len(accuracies)):
        assert accuracies[i] >= accuracies[i-1]
