"""
RadioML Dataset Loader

This module handles loading and processing the RML2016.10A dataset.
The dataset is stored as a pickle file with I/Q samples for various
modulation types at different SNR levels.
"""

import pickle
import numpy as np
from typing import Dict, Tuple, List


def load_radioml_dataset(file_path: str) -> Dict:
    """
    Load the RadioML RML2016.10A dataset from a pickle file.
    
    The dataset structure is a dictionary with keys as tuples (modulation, SNR)
    and values as numpy arrays of I/Q samples with shape (num_samples, 2, 128).
    
    Args:
        file_path: Path to the RML2016.10a_dict.pkl file
        
    Returns:
        Dictionary with structure {(modulation, SNR): samples_array}
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        pickle.UnpicklingError: If the file is not a valid pickle file
    """
    try:
        with open(file_path, 'rb') as f:
            dataset = pickle.load(f, encoding='latin1')
        return dataset
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset file not found at {file_path}")
    except Exception as e:
        raise pickle.UnpicklingError(f"Failed to load dataset: {str(e)}")


def get_dataset_info(dataset: Dict) -> Dict:
    """
    Extract metadata and statistics from the RadioML dataset.
    
    Args:
        dataset: Loaded RadioML dataset dictionary
        
    Returns:
        Dictionary containing:
            - modulations: List of modulation types
            - snrs: List of SNR values
            - sample_count: Total number of samples
            - samples_per_mod: Number of samples per modulation type
            - shape: Shape of individual samples
    """
    if not dataset:
        return {
            "modulations": [],
            "snrs": [],
            "sample_count": 0,
            "samples_per_mod": {},
            "shape": None
        }
    
    
    modulations = sorted(list(set([key[0] for key in dataset.keys()])))
    snrs = sorted(list(set([key[1] for key in dataset.keys()])))
    
    
    total_samples = sum([dataset[key].shape[0] for key in dataset.keys()])
    
    
    samples_per_mod = {}
    for mod in modulations:
        mod_samples = sum([dataset[key].shape[0] for key in dataset.keys() if key[0] == mod])
        samples_per_mod[mod] = mod_samples
    
    
    first_key = list(dataset.keys())[0]
    sample_shape = dataset[first_key].shape[1:]  
    
    return {
        "modulations": modulations,
        "snrs": snrs,
        "sample_count": total_samples,
        "samples_per_mod": samples_per_mod,
        "shape": sample_shape
    }


def split_dataset(dataset: Dict, train_ratio: float = 0.8, random_seed: int = 42) -> Tuple[Dict, Dict]:
    """
    Split the RadioML dataset into training and testing sets.
    
    The split is performed per modulation/SNR combination to ensure
    balanced representation across both sets.
    
    Args:
        dataset: Loaded RadioML dataset dictionary
        train_ratio: Proportion of data to use for training (default: 0.8)
        random_seed: Random seed for reproducibility (default: 42)
        
    Returns:
        Tuple of (train_dataset, test_dataset) dictionaries with same structure as input
        
    Raises:
        ValueError: If train_ratio is not between 0 and 1
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be between 0 and 1, got {train_ratio}")
    
    np.random.seed(random_seed)
    
    train_dataset = {}
    test_dataset = {}
    
    for key, samples in dataset.items():
        n_samples = samples.shape[0]
        n_train = int(n_samples * train_ratio)
        
        
        indices = np.random.permutation(n_samples)
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
        
        
        train_dataset[key] = samples[train_indices]
        test_dataset[key] = samples[test_indices]
    
    return train_dataset, test_dataset


def flatten_dataset(dataset: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Flatten the RadioML dataset into arrays of samples and labels.
    
    Args:
        dataset: RadioML dataset dictionary
        
    Returns:
        Tuple of (samples, labels) where:
            - samples: numpy array of shape (total_samples, 2, 128)
            - labels: numpy array of integer labels (total_samples,)
    """
    # Create modulation to label mapping
    modulations = sorted(list(set([key[0] for key in dataset.keys()])))
    mod_to_label = {mod: idx for idx, mod in enumerate(modulations)}
    
    samples_list = []
    labels_list = []
    
    for (modulation, snr), samples in dataset.items():
        label = mod_to_label[modulation]
        samples_list.append(samples)
        labels_list.extend([label] * samples.shape[0])
    
    samples_array = np.vstack(samples_list)
    labels_array = np.array(labels_list, dtype=np.int64)
    
    return samples_array, labels_array


def partition_dataset(dataset: Dict, num_partitions: int, partition_index: int, random_seed: int = 42) -> Dict:
    """
    Partition the dataset into smaller subsets for federated learning.
    Each client gets a different partition.
    
    Args:
        dataset: RadioML dataset dictionary
        num_partitions: Total number of partitions (e.g., 3 for 3 clients)
        partition_index: Which partition to return (0 to num_partitions-1)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with same structure as input but with subset of data
        
    Raises:
        ValueError: If partition_index is out of range
    """
    if partition_index < 0 or partition_index >= num_partitions:
        raise ValueError(f"partition_index must be between 0 and {num_partitions-1}")
    
    np.random.seed(random_seed)
    
    partitioned_dataset = {}
    
    for key, samples in dataset.items():
        n_samples = samples.shape[0]
        
        
        indices = np.random.permutation(n_samples)
        
        
        partition_size = n_samples // num_partitions
        start_idx = partition_index * partition_size
        
        if partition_index == num_partitions - 1:
            end_idx = n_samples
        else:
            end_idx = start_idx + partition_size
        
        
        partition_indices = indices[start_idx:end_idx]
        
        
        partitioned_dataset[key] = samples[partition_indices]
    
    return partitioned_dataset
