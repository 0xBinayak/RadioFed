"""
Local Training Module

This module handles local model training on extracted features from the RadioML dataset.
It provides functions for training, evaluation, and model persistence for KNN models.
Uses the core ML logic from amc-rml2016a-updated.ipynb.
"""

import numpy as np
from typing import Dict
import logging
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def extract_analog_features(signal, fs=128):
    """
    Extracts instantaneous features from a complex signal.
    Based on the feature extraction from amc-rml2016a-updated.ipynb.
    
    Features extracted:
    - Instantaneous Amplitude statistics (mean, variance, skewness, kurtosis)
    - Instantaneous Frequency statistics (mean, variance, skewness, kurtosis)
    
    Args:
        signal: Complex signal (I + jQ)
        fs: Sampling frequency (default: 128 for RML2016.10a)
    
    Returns:
        Dictionary of features
    """
    
    amplitude = np.abs(signal)
    
    
    phase = np.unwrap(np.angle(signal))
    
    
    if fs == 0:
        instantaneous_frequency = np.zeros_like(phase)
    else:
        instantaneous_frequency = np.diff(phase) / (2 * np.pi) * fs
        instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), 'constant')

    
    std_amp = np.std(amplitude)
    std_freq = np.std(instantaneous_frequency)

    features = {
        'amp_mean': np.mean(amplitude),
        'amp_variance': np.var(amplitude),
        'amp_skewness': np.mean((amplitude - np.mean(amplitude))**3) / (std_amp**3 + 1e-9) if std_amp > 1e-9 else 0,
        'amp_kurtosis': np.mean((amplitude - np.mean(amplitude))**4) / (std_amp**4 + 1e-9) - 3 if std_amp > 1e-9 else -3,
        'freq_mean': np.mean(instantaneous_frequency),
        'freq_variance': np.var(instantaneous_frequency),
        'freq_skewness': np.mean((instantaneous_frequency - np.mean(instantaneous_frequency))**3) / (std_freq**3 + 1e-9) if std_freq > 1e-9 else 0,
        'freq_kurtosis': np.mean((instantaneous_frequency - np.mean(instantaneous_frequency))**4) / (std_freq**4 + 1e-9) - 3 if std_freq > 1e-9 else -3,
    }
    return features


def train_knn_model(
    features: np.ndarray,
    labels: np.ndarray,
    test_split: float = 0.3,
    n_neighbors: int = 5,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Train a KNN model on extracted features.
    Based on the training logic from amc-rml2016a-updated.ipynb.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        test_split: Proportion of data to use for testing (default: 0.3, matching notebook)
        n_neighbors: Number of neighbors for KNN (default: 5, matching notebook)
        random_state: Random state for reproducibility (default: 42)
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing:
            - model: Trained KNeighborsClassifier instance
            - train_accuracy: Training accuracy
            - test_accuracy: Test accuracy
            - training_time: Training time in seconds
            - inference_time_ms_per_sample: Inference time in milliseconds per sample
            - n_samples: Number of training samples
            - confusion_matrix: Confusion matrix on test set
    """
    if verbose:
        logger.info(f"Training KNN model (n_neighbors={n_neighbors})")
        logger.info(f"Dataset size: {len(features)} samples")
    
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_split, random_state=random_state, stratify=labels
    )
    
    if verbose:
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    
    start_time_train = time.time()
    model.fit(X_train, y_train)
    end_time_train = time.time()
    training_time = end_time_train - start_time_train
    
    if verbose:
        logger.info(f"Training complete! Time: {training_time:.3f} seconds")
    
    
    train_predictions = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    
    
    start_time_inference = time.time()
    test_predictions = model.predict(X_test)
    end_time_inference = time.time()
    inference_time_ms_per_sample = ((end_time_inference - start_time_inference) / len(X_test)) * 1000
    
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    
    conf_matrix = confusion_matrix(y_test, test_predictions)
    
    if verbose:
        logger.info(f"Train Accuracy: {train_accuracy * 100:.2f}%")
        logger.info(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        logger.info(f"Inference Time: {inference_time_ms_per_sample:.3f} ms/sample")
    
    return {
        'model': model,
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': training_time,
        'inference_time_ms_per_sample': inference_time_ms_per_sample,
        'n_samples': len(X_train),
        'confusion_matrix': conf_matrix
    }


def save_knn_model(model, path: str) -> None:
    """
    Save trained KNN model to a file.
    
    Args:
        model: KNeighborsClassifier instance to save
        path: File path to save the model (.pkl format)
        
    Raises:
        IOError: If the file cannot be written
    """
    import os
    import pickle
    
    
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info(f"Model saved to {path}")


def load_knn_model(path: str):
    """
    Load KNN model from a file.
    
    Args:
        path: File path to load the model from (.pkl format)
        
    Returns:
        KNeighborsClassifier instance with loaded weights
        
    Raises:
        FileNotFoundError: If the model file does not exist
    """
    import os
    import pickle
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    
    with open(path, 'rb') as f:
        model = pickle.load(f)
    
    logger.info(f"Model loaded from {path}")
    
    return model
