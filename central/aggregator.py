"""
FedAvg Aggregation Logic for Central Server

This module implements the Federated Averaging (FedAvg) algorithm for
aggregating model weights from multiple clients. The aggregation is weighted
by the number of samples each client used for training.

Also includes aggregation strategies for traditional ML models (KNN, Decision Tree).
"""

import torch
import os
import logging
import pickle
import numpy as np
from typing import Dict, List, Tuple, Union
from central.model import FederatedModel


logger = logging.getLogger("federated_central")


def load_client_weights(weights_path: str) -> Dict[str, torch.Tensor]:
    """
    Load model weights from a client's weight file.
    
    Args:
        weights_path (str): Path to the client's weights file (.pth format)
    
    Returns:
        dict: Dictionary containing model state dict with parameter tensors
    
    Raises:
        FileNotFoundError: If the weights file does not exist
        RuntimeError: If the weights file is corrupted or invalid
    """
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    try:
        # Load weights to CPU to avoid device conflicts
        weights = torch.load(weights_path, map_location='cpu')
        logger.info(f"Loaded weights from {weights_path}")
        return weights
    except Exception as e:
        logger.error(f"Failed to load weights from {weights_path}: {e}")
        raise RuntimeError(f"Invalid weights file: {weights_path}") from e


def validate_weight_compatibility(
    weights_list: List[Dict[str, torch.Tensor]],
    reference_model: FederatedModel = None
) -> Tuple[bool, str]:
    """
    Validate that all client weights are compatible with each other and the model architecture.
    
    Args:
        weights_list (list): List of weight dictionaries from clients
        reference_model (FederatedModel, optional): Reference model to validate against
    
    Returns:
        tuple: (is_valid, error_message) where is_valid is True if all weights are compatible
    """
    if not weights_list:
        return False, "No weights provided for validation"
    
    # Get reference keys and shapes from first client
    reference_weights = weights_list[0]
    reference_keys = set(reference_weights.keys())
    
    # Check all clients have the same keys
    for i, weights in enumerate(weights_list[1:], start=1):
        client_keys = set(weights.keys())
        if client_keys != reference_keys:
            missing = reference_keys - client_keys
            extra = client_keys - reference_keys
            msg = f"Client {i} has incompatible keys. Missing: {missing}, Extra: {extra}"
            return False, msg
    
    # Check all weights have the same shapes
    for key in reference_keys:
        reference_shape = reference_weights[key].shape
        for i, weights in enumerate(weights_list[1:], start=1):
            if weights[key].shape != reference_shape:
                msg = f"Client {i} has incompatible shape for '{key}': {weights[key].shape} vs {reference_shape}"
                return False, msg
    
    # If reference model provided, validate against it
    if reference_model is not None:
        model_keys = set(reference_model.state_dict().keys())
        if reference_keys != model_keys:
            missing = model_keys - reference_keys
            extra = reference_keys - model_keys
            msg = f"Weights incompatible with model architecture. Missing: {missing}, Extra: {extra}"
            return False, msg
        
        # Check shapes match model
        model_state = reference_model.state_dict()
        for key in reference_keys:
            if reference_weights[key].shape != model_state[key].shape:
                msg = f"Weight '{key}' has incompatible shape: {reference_weights[key].shape} vs {model_state[key].shape}"
                return False, msg
    
    return True, ""


def aggregate_weights(
    client_weights: List[Dict[str, torch.Tensor]],
    sample_counts: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Perform Federated Averaging (FedAvg) on client weights.
    
    The aggregation is weighted by the number of samples each client used for training:
    w_global = Σ(n_i / N) * w_i
    where n_i is the number of samples for client i, and N is the total number of samples.
    
    Args:
        client_weights (list): List of weight dictionaries from clients
        sample_counts (list): List of sample counts corresponding to each client
    
    Returns:
        dict: Aggregated global model weights
    
    Raises:
        ValueError: If inputs are invalid or incompatible
    """
    if not client_weights:
        raise ValueError("No client weights provided for aggregation")
    
    if len(client_weights) != len(sample_counts):
        raise ValueError(f"Mismatch: {len(client_weights)} weight sets but {len(sample_counts)} sample counts")
    
    if any(count <= 0 for count in sample_counts):
        raise ValueError("All sample counts must be positive")
    
    # Validate weight compatibility
    is_valid, error_msg = validate_weight_compatibility(client_weights)
    if not is_valid:
        raise ValueError(f"Weight compatibility check failed: {error_msg}")
    
    # Calculate total samples and weights
    total_samples = sum(sample_counts)
    weights_per_client = [count / total_samples for count in sample_counts]
    
    logger.info(f"Aggregating weights from {len(client_weights)} clients")
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Client weights: {weights_per_client}")
    
    # Initialize aggregated weights with zeros
    aggregated_weights = {}
    reference_weights = client_weights[0]
    
    for key in reference_weights.keys():
        aggregated_weights[key] = torch.zeros_like(reference_weights[key])
    
    # Perform weighted averaging
    for client_weight, weight_factor in zip(client_weights, weights_per_client):
        for key in aggregated_weights.keys():
            aggregated_weights[key] += weight_factor * client_weight[key]
    
    logger.info("Weight aggregation completed successfully")
    return aggregated_weights


def save_global_model(
    weights: Dict[str, torch.Tensor],
    path: str
) -> None:
    """
    Save aggregated global model weights to a file.
    
    Args:
        weights (dict): Dictionary containing model weights
        path (str): File path to save the weights (.pth format)
    
    Raises:
        IOError: If the file cannot be written
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save weights
        torch.save(weights, path)
        logger.info(f"Global model saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save global model to {path}: {e}")
        raise IOError(f"Could not save global model: {e}") from e


def aggregate_from_registry(
    client_info_list: List[Dict],
    save_path: str,
    reference_model: FederatedModel = None
) -> Dict:
    """
    Convenience function to aggregate weights from client registry information.
    
    Args:
        client_info_list (list): List of dicts with 'weights_path' and 'n_samples' keys
        save_path (str): Path to save the aggregated global model
        reference_model (FederatedModel, optional): Reference model for validation
    
    Returns:
        dict: Summary of aggregation including number of clients and total samples
    
    Raises:
        ValueError: If aggregation fails
    """
    if not client_info_list:
        raise ValueError("No clients available for aggregation")
    
    # Load all client weights
    client_weights = []
    sample_counts = []
    
    for client_info in client_info_list:
        try:
            weights = load_client_weights(client_info['weights_path'])
            client_weights.append(weights)
            sample_counts.append(client_info['n_samples'])
        except Exception as e:
            logger.warning(f"Skipping client {client_info.get('client_id', 'unknown')}: {e}")
            continue
    
    if not client_weights:
        raise ValueError("No valid client weights could be loaded")
    
    # Validate against reference model if provided
    if reference_model is not None:
        is_valid, error_msg = validate_weight_compatibility(client_weights, reference_model)
        if not is_valid:
            raise ValueError(f"Weight validation failed: {error_msg}")
    
    # Perform aggregation
    aggregated_weights = aggregate_weights(client_weights, sample_counts)
    
    # Save global model
    save_global_model(aggregated_weights, save_path)
    
    return {
        'num_clients': len(client_weights),
        'total_samples': sum(sample_counts),
        'global_model_path': save_path
    }


# ============================================================================
# Traditional ML Model Aggregation Functions
# ============================================================================

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
    
    # Train global KNN model on merged data
    from sklearn.neighbors import KNeighborsClassifier
    global_knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    
    # Measure training time
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



def aggregate_dt_models(
    client_models_info: List[Dict],
    evaluate: bool = True
) -> Dict:
    """
    Aggregate Decision Tree models from multiple clients using ensemble voting.
    
    Strategy: Create an ensemble of client decision trees that vote on predictions.
    This preserves the diversity of local models while combining their knowledge.
    
    Args:
        client_models_info: List of dicts with 'model_path' and 'n_samples' keys
        evaluate: Whether to evaluate the ensemble and collect metrics
    
    Returns:
        dict: Aggregation result containing:
            - client_models: List of loaded Decision Tree models
            - num_clients: Number of clients aggregated
            - total_samples: Total number of training samples
            - ensemble_weights: Weights for each model based on sample counts
            - training_time: Time taken to create ensemble (if evaluate=True)
            - inference_time_ms_per_sample: Inference time (if evaluate=True)
            - accuracy: Overall accuracy (if evaluate=True)
            - per_snr_accuracy: Per-SNR accuracy dict (if evaluate=True)
            - confusion_matrix: Confusion matrix (if evaluate=True)
    
    Raises:
        ValueError: If no valid client models are available
    """
    import time
    
    if not client_models_info:
        raise ValueError("No client models provided for Decision Tree aggregation")
    
    logger.info(f"Starting Decision Tree aggregation with {len(client_models_info)} clients")
    
    ensemble_start = time.time()
    
    # Load all client models and collect test data
    client_models = []
    sample_counts = []
    all_test_features = []
    all_test_labels = []
    all_test_snrs = []
    valid_clients = 0
    
    for client_info in client_models_info:
        try:
            model_path = client_info.get('model_path')
            n_samples = client_info.get('n_samples', 0)
            
            if not model_path:
                logger.warning(f"Client {client_info.get('client_id', 'unknown')} missing model path, skipping")
                continue
            
            if not os.path.exists(model_path):
                logger.warning(f"Client {client_info.get('client_id', 'unknown')} model file not found, skipping")
                continue
            
            # Load the decision tree model
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            client_models.append(model)
            sample_counts.append(n_samples)
            valid_clients += 1
            
            logger.info(f"Loaded Decision Tree model from client {client_info.get('client_id', 'unknown')} ({n_samples} samples)")
            
            # Try to load test data for evaluation
            if evaluate:
                features_path = client_info.get('features_path')
                labels_path = client_info.get('labels_path')
                
                if features_path and labels_path and os.path.exists(features_path) and os.path.exists(labels_path):
                    try:
                        with open(features_path, 'rb') as f:
                            features = pickle.load(f)
                        with open(labels_path, 'rb') as f:
                            labels = pickle.load(f)
                        
                        # Use a portion for testing (20%)
                        n_test = len(features) // 5
                        all_test_features.append(np.array(features[:n_test]))
                        all_test_labels.append(np.array(labels[:n_test]))
                        
                        # Try to load SNR values
                        snrs_path = features_path.replace('_features.pkl', '_snrs.pkl')
                        if os.path.exists(snrs_path):
                            with open(snrs_path, 'rb') as f:
                                snrs = pickle.load(f)
                            all_test_snrs.append(np.array(snrs[:n_test]))
                    except Exception as e:
                        logger.debug(f"Could not load test data from client: {e}")
            
        except Exception as e:
            logger.warning(f"Error loading model from client {client_info.get('client_id', 'unknown')}: {e}")
            continue
    
    if not client_models:
        raise ValueError("No valid client models could be loaded for Decision Tree aggregation")
    
    # Calculate ensemble weights based on sample counts
    total_samples = sum(sample_counts)
    ensemble_weights = [count / total_samples for count in sample_counts] if total_samples > 0 else [1.0 / len(client_models)] * len(client_models)
    
    # Create ensemble
    ensemble = DecisionTreeEnsemble(client_models, ensemble_weights)
    ensemble_time = time.time() - ensemble_start
    
    logger.info(f"Decision Tree ensemble created with {valid_clients} models in {ensemble_time:.3f}s")
    logger.info(f"Total samples: {total_samples}, Weights: {ensemble_weights}")
    
    result = {
        'client_models': client_models,
        'num_clients': valid_clients,
        'total_samples': total_samples,
        'ensemble_weights': ensemble_weights,
        'training_time': ensemble_time
    }
    
    # Evaluate ensemble if requested and test data available
    if evaluate and all_test_features:
        logger.info("Evaluating Decision Tree ensemble...")
        
        # Merge test data from all clients
        X_test = np.vstack(all_test_features)
        y_test = np.concatenate(all_test_labels)
        snr_test = np.concatenate(all_test_snrs) if all_test_snrs else None
        
        # Generate synthetic SNR values if not available
        if snr_test is None:
            logger.info("Generating synthetic SNR values for evaluation")
            snr_test = generate_synthetic_snr_values(len(X_test))
        
        # Measure inference time
        inference_start = time.time()
        predictions = ensemble.predict(X_test)
        inference_time = time.time() - inference_start
        inference_time_ms_per_sample = (inference_time / len(X_test)) * 1000
        
        # Compute metrics
        eval_metrics = evaluate_global_model(ensemble, X_test, y_test, snr_test)
        
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


class DecisionTreeEnsemble:
    """
    Ensemble wrapper for Decision Tree models that performs weighted voting.
    """
    
    def __init__(self, models: List, weights: List[float] = None):
        """
        Initialize the ensemble.
        
        Args:
            models: List of trained Decision Tree models
            weights: Optional weights for each model (defaults to equal weights)
        """
        self.models = models
        self.weights = weights if weights is not None else [1.0 / len(models)] * len(models)
        
        if len(self.models) != len(self.weights):
            raise ValueError("Number of models must match number of weights")
    
    def predict(self, X):
        """
        Predict using weighted majority voting.
        
        Args:
            X: Feature matrix for prediction
        
        Returns:
            Array of predictions
        """
        # Get predictions from all models
        all_predictions = np.array([model.predict(X) for model in self.models])
        
        # Perform weighted voting
        n_samples = X.shape[0]
        final_predictions = []
        
        for i in range(n_samples):
            sample_predictions = all_predictions[:, i]
            
            # Count votes with weights
            unique_classes = np.unique(sample_predictions)
            vote_counts = {}
            
            for cls in unique_classes:
                # Sum weights of models that predicted this class
                vote_counts[cls] = sum(
                    weight for pred, weight in zip(sample_predictions, self.weights)
                    if pred == cls
                )
            
            # Select class with highest weighted vote
            final_predictions.append(max(vote_counts, key=vote_counts.get))
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """
        Predict class probabilities using weighted averaging.
        
        Args:
            X: Feature matrix for prediction
        
        Returns:
            Array of class probabilities
        """
        # Get probability predictions from all models
        all_probas = np.array([model.predict_proba(X) for model in self.models])
        
        # Weighted average of probabilities
        weighted_probas = np.zeros_like(all_probas[0])
        for probas, weight in zip(all_probas, self.weights):
            weighted_probas += weight * probas
        
        return weighted_probas


def save_dt_ensemble(ensemble: DecisionTreeEnsemble, path: str) -> None:
    """
    Save Decision Tree ensemble to file using pickle.
    
    Args:
        ensemble: DecisionTreeEnsemble object
        path: File path to save the ensemble
    
    Raises:
        IOError: If the file cannot be written
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(ensemble, f)
        logger.info(f"Decision Tree ensemble saved to {path}")
    except Exception as e:
        logger.error(f"Failed to save Decision Tree ensemble to {path}: {e}")
        raise IOError(f"Could not save Decision Tree ensemble: {e}") from e


def load_dt_ensemble(path: str) -> DecisionTreeEnsemble:
    """
    Load Decision Tree ensemble from file.
    
    Args:
        path: File path to load the ensemble from
    
    Returns:
        Loaded DecisionTreeEnsemble object
    
    Raises:
        FileNotFoundError: If the ensemble file does not exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Decision Tree ensemble file not found: {path}")
    
    try:
        with open(path, 'rb') as f:
            ensemble = pickle.load(f)
        logger.info(f"Decision Tree ensemble loaded from {path}")
        return ensemble
    except Exception as e:
        logger.error(f"Failed to load Decision Tree ensemble from {path}: {e}")
        raise RuntimeError(f"Invalid Decision Tree ensemble file: {path}") from e



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
    model: Union[object, DecisionTreeEnsemble],
    test_features: np.ndarray,
    test_labels: np.ndarray,
    test_snrs: np.ndarray = None
) -> Dict:
    """
    Evaluate global model on validation/test set.
    
    Computes overall accuracy, per-SNR accuracy breakdown, and confusion matrix.
    
    Args:
        model: Trained model (KNN, Decision Tree, or ensemble)
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
    
    # Compute overall accuracy
    from sklearn.metrics import accuracy_score, confusion_matrix
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
