"""
KNN Aggregation Logic for Central Server

This module implements aggregation strategies for KNN models in federated learning.
KNN models are aggregated by merging training data from multiple clients.
Based on the ML approach from amc-rml2016a-updated.ipynb.
"""

import os
import logging
import pickle
import numpy as np
from typing import Dict, List, Tuple
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import time


logger = logging.getLogger("federated_central")


def aggregate_knn_models(
    client_models_info: List[Dict],
    n_neighbors: int = 5,
    evaluate: bool = True
) -> Dict:
    """
    Aggregate KNN models from multiple clients by merging training data.
    
    Strategy: Since KNN is instance-based, we merge all training data from
    clients and retrain a global KNN model on the combined dataset.
    
    Args:
        client_models_info: List of dicts with 'model_path', 'features_path', 
                           'labels_path', and 'n_samples' keys
        n_neighbors: Number of neighbors for the global KNN model
        evaluate: Whether to evaluate the model and collect metrics
    
    Returns:
        dict: Aggregation result containing:
            - global_model: Trained KNN model
            - total_samples: Total number of training samples
            - num_clients: Number of clients aggregated
            - feature_dim: Dimension of feature vectors
            - training_time: Time taken to train (if evaluate=True)
            - inference_time_ms_per_sample: Inference time (if evaluate=True)
            - accuracy: Overall accuracy (if evaluate=True)
            - per_snr_accuracy: Per-SNR accuracy dict (if evaluate=True)
            - confusion_matrix: Confusion matrix (if evaluate=True)
    
    Raises:
        ValueError: If no valid client data is available or dimensions mismatch
    """
    import time
    
    if not client_models_info:
        raise ValueError("No client models provided for KNN aggregation")
    
    logger.info(f"Starting KNN aggregation with {len(client_models_info)} clients")
    
    # Collect all training data from clients
    all_features = []
    all_labels = []
    all_snrs = []  # Track SNR values if available
    total_samples = 0
    valid_clients = 0
    
    for client_info in client_models_info:
        try:
            # Load client's training data
            features_path = client_info.get('features_path')
            labels_path = client_info.get('labels_path')
            
            if not features_path or not labels_path:
                logger.warning(f"Client {client_info.get('client_id', 'unknown')} missing data paths, skipping")
                continue
            
            if not os.path.exists(features_path) or not os.path.exists(labels_path):
                logger.warning(f"Client {client_info.get('client_id', 'unknown')} data files not found, skipping")
                continue
            
            # Load features and labels
            with open(features_path, 'rb') as f:
                features = pickle.load(f)
            with open(labels_path, 'rb') as f:
                labels = pickle.load(f)
            
            # Convert to numpy arrays if needed
            features = np.array(features)
            labels = np.array(labels)
            
            # Validate dimensions
            if len(features) != len(labels):
                logger.warning(f"Client {client_info.get('client_id', 'unknown')} has mismatched features/labels, skipping")
                continue
            
            all_features.append(features)
            all_labels.append(labels)
            total_samples += len(features)
            valid_clients += 1
            
            # Try to load SNR values if available
            snrs_path = features_path.replace('_features.pkl', '_snrs.pkl')
            if os.path.exists(snrs_path):
                try:
                    with open(snrs_path, 'rb') as f:
                        snrs = pickle.load(f)
                    all_snrs.append(np.array(snrs))
                except:
                    pass
            
            logger.info(f"Loaded {len(features)} samples from client {client_info.get('client_id', 'unknown')}")
            
        except Exception as e:
            logger.warning(f"Error loading data from client {client_info.get('client_id', 'unknown')}: {e}")
            continue
    
    if not all_features:
        raise ValueError("No valid client data could be loaded for KNN aggregation")
    
    # Merge all features and labels
    merged_features = np.vstack(all_features)
    merged_labels = np.concatenate(all_labels)
    merged_snrs = np.concatenate(all_snrs) if all_snrs else None
    
    logger.info(f"Merged data: {merged_features.shape[0]} samples, {merged_features.shape[1]} features")
    
    # Validate feature dimensions are consistent
    feature_dim = merged_features.shape[1]
    
    # Split into train/test for evaluation (80/20 split)
    if evaluate:
        from sklearn.model_selection import train_test_split
        
        if merged_snrs is not None:
            X_train, X_test, y_train, y_test, snr_train, snr_test = train_test_split(
                merged_features, merged_labels, merged_snrs,
                test_size=0.2, random_state=42, stratify=merged_labels
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                merged_features, merged_labels,
                test_size=0.2, random_state=42, stratify=merged_labels
            )
            snr_test = None
    else:
        X_train, y_train = merged_features, merged_labels
        X_test, y_test, snr_test = None, None, None
    
    # Train global KNN model on merged data (matching notebook approach)
    global_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Measure training time (matching notebook timing approach)
    train_start = time.time()
    global_knn.fit(X_train, y_train)
    training_time = time.time() - train_start
    
    logger.info(f"Global KNN model trained successfully with {n_neighbors} neighbors in {training_time:.3f}s")
    
    result = {
        'global_model': global_knn,
        'total_samples': total_samples,
        'num_clients': valid_clients,
        'feature_dim': feature_dim,
        'n_neighbors': n_neighbors,
        'training_time': training_time
    }
    
    # Evaluate model if requested
    if evaluate and X_test is not None:
        logger.info("Evaluating global KNN model...")
        
        # Generate synthetic SNR values if not available
        if snr_test is None:
            logger.info("Generating synthetic SNR values for evaluation")
            snr_test = generate_synthetic_snr_values(len(X_test))
        
        # Measure inference time
        inference_start = time.time()
        predictions = global_knn.predict(X_test)
        inference_time = time.time() - inference_start
        inference_time_ms_per_sample = (inference_time / len(X_test)) * 1000
        
        # Compute metrics
        eval_metrics = evaluate_global_model(global_knn, X_test, y_test, snr_test)
        
        # Add metrics to result
        result.update({
            'inference_time_ms_per_sample': inference_time_ms_per_sample,
            'accuracy': eval_metrics['accuracy'],
            'per_snr_accuracy': eval_metrics['per_snr_accuracy'],
            'confusion_matrix': eval_metrics['confusion_matrix'].tolist(),  # Convert to list for JSON
            'n_test_samples': eval_metrics['n_samples']
        })
        
        logger.info(f"Evaluation complete: accuracy={eval_metrics['accuracy']:.4f}, inference={inference_time_ms_per_sample:.3f}ms/sample")
    
    return result


def save_knn_model(model, path: str) -> None:
    """
    Save KNN model to file using pickle.
    
    Args:
        model: Trained KNN model
        path: File path to save the model
    
    Raises:
        IOError: If the file cannot be written
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"KNN model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save KNN model to {path}: {e}")
        raise IOError(f"Could not save KNN model: {e}") from e


def load_knn_model(path: str):
    """
    Load KNN model from file.
    
    Args:
        path: File path to load the model from
    
    Returns:
        Loaded KNN model
    
    Raises:
        FileNotFoundError: If the model file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"KNN model file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"KNN model loaded from {path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load KNN model from {path}: {e}")
        raise RuntimeError(f"Invalid KNN model file: {path}") from e


def generate_synthetic_snr_values(n_samples: int, snr_range: Tuple[int, int] = (-20, 18)) -> np.ndarray:
    """
    Generate synthetic SNR values for samples when real SNR data is not available.
    
    Distributes samples evenly across SNR levels from snr_range[0] to snr_range[1]
    in steps of 2 dB (matching RadioML 2016.10a dataset).
    
    Args:
        n_samples: Number of samples to generate SNR values for
        snr_range: Tuple of (min_snr, max_snr) in dB
    
    Returns:
        Array of SNR values
    """
    snr_levels = list(range(snr_range[0], snr_range[1] + 1, 2))
    
    # Distribute samples evenly across SNR levels
    snr_values = []
    samples_per_snr = n_samples // len(snr_levels)
    remainder = n_samples % len(snr_levels)
    
    for i, snr in enumerate(snr_levels):
        count = samples_per_snr + (1 if i < remainder else 0)
        snr_values.extend([snr] * count)
    
    return np.array(snr_values)


def evaluate_global_model(
    model: object,
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray = None
) -> Dict:
    """
    Evaluate global model on validation/test set.
    
    Computes overall accuracy, per-SNR accuracy breakdown, and confusion matrix.
    
    Args:
        model: Trained KNN model
        test_features: Test feature matrix (n_samples, n_features)
        test_labels: True labels for test samples
        test_snrs: Optional SNR values for each test sample (for per-SNR analysis)
    
    Returns:
        dict: Evaluation metrics containing:
            - accuracy: Overall accuracy (0-1)
            - per_snr_accuracy: Dict mapping SNR to accuracy (if test_snrs provided)
            - confusion_matrix: Confusion matrix as numpy array
            - n_samples: Number of test samples
            - predictions: Model predictions
    
    Raises:
        ValueError: If inputs are invalid
    """
    if len(test_features) != len(test_labels):
        raise ValueError("Number of test features must match number of test labels")
    
    if test_snrs is not None and len(test_snrs) != len(test_labels):
        raise ValueError("Number of SNR values must match number of test samples")
    
    logger.info(f"Evaluating global model on {len(test_features)} test samples")
    
    # Get predictions
    predictions = model.predict(test_features)
    
    # Compute overall accuracy (matching notebook approach)
    accuracy = accuracy_score(test_labels, predictions)
    
    logger.info(f"Overall accuracy: {accuracy:.4f}")
    
    # Compute confusion matrix
    conf_matrix = confusion_matrix(test_labels, predictions)
    
    # Compute per-SNR accuracy if SNR values provided
    per_snr_accuracy = {}
    if test_snrs is not None:
        unique_snrs = np.unique(test_snrs)
        for snr in unique_snrs:
            snr_mask = test_snrs == snr
            snr_labels = test_labels[snr_mask]
            snr_predictions = predictions[snr_mask]
            
            if len(snr_labels) > 0:
                snr_accuracy = accuracy_score(snr_labels, snr_predictions)
                per_snr_accuracy[float(snr)] = snr_accuracy
                logger.info(f"SNR {snr} dB: accuracy = {snr_accuracy:.4f} ({len(snr_labels)} samples)")
    
    return {
        'accuracy': float(accuracy),
        'per_snr_accuracy': per_snr_accuracy,
        'confusion_matrix': conf_matrix,
        'n_samples': len(test_features),
        'predictions': predictions
    }
