"""
Unit tests for RadioML dataset loading functionality.

Tests dataset loading, metadata extraction, and train/test splitting.
"""

import pytest
import numpy as np
import pickle
import tempfile
import os
import shutil
from client.dataset_loader import (
    load_radioml_dataset,
    get_dataset_info,
    split_dataset,
    flatten_dataset
)


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


@pytest.fixture
def synthetic_dataset(temp_dir):
    """Create a synthetic RadioML-like dataset."""
    
    dataset = {}
    
    modulations = ['BPSK', 'QPSK', '8PSK']
    snrs = [-10, 0, 10]
    samples_per_key = 100
    
    for mod in modulations:
        for snr in snrs:
            
            samples = np.random.randn(samples_per_key, 2, 128).astype(np.float32)
            dataset[(mod, snr)] = samples
    
    
    dataset_path = os.path.join(temp_dir, 'test_dataset.pkl')
    with open(dataset_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    return dataset, dataset_path


def test_load_radioml_dataset_success(synthetic_dataset):
    """Test successful loading of RadioML dataset."""
    dataset, dataset_path = synthetic_dataset
    loaded_dataset = load_radioml_dataset(dataset_path)
    
    assert isinstance(loaded_dataset, dict)
    assert len(loaded_dataset) == 9 
    
    
    first_key = list(loaded_dataset.keys())[0]
    assert isinstance(first_key, tuple)
    assert len(first_key) == 2 
    
    
    samples = loaded_dataset[first_key]
    assert samples.shape[1:] == (2, 128)


def test_load_radioml_dataset_file_not_found():
    """Test loading from nonexistent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_radioml_dataset('/nonexistent/path/dataset.pkl')


def test_load_radioml_dataset_invalid_file(temp_dir):
    """Test loading invalid pickle file raises error."""
    invalid_path = os.path.join(temp_dir, 'invalid.pkl')
    with open(invalid_path, 'w') as f:
        f.write("not a pickle file")
    
    with pytest.raises(pickle.UnpicklingError):
        load_radioml_dataset(invalid_path)


def test_get_dataset_info(synthetic_dataset):
    """Test extraction of dataset metadata."""
    dataset, _ = synthetic_dataset
    info = get_dataset_info(dataset)
    
    assert info['modulations'] == ['8PSK', 'BPSK', 'QPSK']
    assert info['snrs'] == [-10, 0, 10]
    assert info['sample_count'] == 900  
    assert info['shape'] == (2, 128)
    
    
    assert info['samples_per_mod']['BPSK'] == 300 
    assert info['samples_per_mod']['QPSK'] == 300
    assert info['samples_per_mod']['8PSK'] == 300


def test_get_dataset_info_empty():
    """Test dataset info with empty dataset."""
    info = get_dataset_info({})
    
    assert info['modulations'] == []
    assert info['snrs'] == []
    assert info['sample_count'] == 0
    assert info['shape'] is None


def test_split_dataset_default_ratio(synthetic_dataset):
    """Test dataset splitting with default 80/20 ratio."""
    dataset, _ = synthetic_dataset
    train_dataset, test_dataset = split_dataset(dataset)
    assert set(train_dataset.keys()) == set(dataset.keys())
    assert set(test_dataset.keys()) == set(dataset.keys())

    for key in dataset.keys():
        original_count = dataset[key].shape[0]
        train_count = train_dataset[key].shape[0]
        test_count = test_dataset[key].shape[0]
        
        assert train_count == 80  
        assert test_count == 20   
        assert train_count + test_count == original_count


def test_split_dataset_custom_ratio(synthetic_dataset):
    """Test dataset splitting with custom ratio."""
    dataset, _ = synthetic_dataset
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.7)
    
    for key in dataset.keys():
        train_count = train_dataset[key].shape[0]
        test_count = test_dataset[key].shape[0]
        
        assert train_count == 70 
        assert test_count == 30   


def test_split_dataset_invalid_ratio(synthetic_dataset):
    """Test that invalid train ratio raises ValueError."""
    dataset, _ = synthetic_dataset
    
    with pytest.raises(ValueError):
        split_dataset(dataset, train_ratio=1.5)
    
    with pytest.raises(ValueError):
        split_dataset(dataset, train_ratio=0.0)
    
    with pytest.raises(ValueError):
        split_dataset(dataset, train_ratio=-0.1)


def test_split_dataset_reproducibility(synthetic_dataset):
    """Test that splitting with same seed produces same results."""
    dataset, _ = synthetic_dataset
    train1, test1 = split_dataset(dataset, random_seed=42)
    train2, test2 = split_dataset(dataset, random_seed=42)
    
    for key in dataset.keys():
        np.testing.assert_array_equal(train1[key], train2[key])
        np.testing.assert_array_equal(test1[key], test2[key])


def test_flatten_dataset(synthetic_dataset):
    """Test flattening dataset into samples and labels."""
    dataset, _ = synthetic_dataset
    samples, labels = flatten_dataset(dataset)
 
    assert samples.shape == (900, 2, 128)  
    assert labels.shape == (900,)
    

    unique_labels = np.unique(labels)
    assert len(unique_labels) == 3
    assert np.all(unique_labels >= 0)
    assert np.all(unique_labels < 3)
    
    
    for label in unique_labels:
        count = np.sum(labels == label)
        assert count == 300  


def test_flatten_dataset_label_consistency(synthetic_dataset):
    """Test that same modulation always gets same label."""
    dataset, _ = synthetic_dataset
    samples1, labels1 = flatten_dataset(dataset)
    samples2, labels2 = flatten_dataset(dataset)
    

    np.testing.assert_array_equal(labels1, labels2)
