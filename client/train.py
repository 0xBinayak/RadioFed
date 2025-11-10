"""
Local Training Module

This module handles local model training on extracted features from the RadioML dataset.
It provides functions for training, evaluation, and model persistence.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from typing import Dict, Tuple, Optional
import logging
from client.model import FederatedModel
from client.models.traditional_models import KNNModel, DecisionTreeModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_local_model(
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = 10,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    test_split: float = 0.2,
    device: Optional[str] = None,
    verbose: bool = True
) -> Dict:
    """
    Train a local model using PyTorch training loop.
    
    Args:
        features: Feature array of shape (n_samples, 16)
        labels: Label array of shape (n_samples,)
        epochs: Number of training epochs (default: 10)
        batch_size: Batch size for training (default: 32)
        learning_rate: Learning rate for Adam optimizer (default: 0.001)
        test_split: Proportion of data to use for testing (default: 0.2)
        device: Device to train on ('cpu' or 'cuda'). If None, auto-detect
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing:
            - model: Trained FederatedModel instance
            - train_loss: Final training loss
            - train_accuracy: Final training accuracy
            - test_loss: Final test loss
            - test_accuracy: Final test accuracy
            - history: Training history with loss and accuracy per epoch
    """
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        logger.info(f"Training on device: {device}")
        logger.info(f"Dataset size: {len(features)} samples")
        logger.info(f"Training for {epochs} epochs with batch size {batch_size}")
    
    # Convert to PyTorch tensors
    features_tensor = torch.FloatTensor(features)
    labels_tensor = torch.LongTensor(labels)
    
    # Split into train and test sets
    n_samples = len(features)
    n_test = int(n_samples * test_split)
    n_train = n_samples - n_test
    
    # Random split
    indices = torch.randperm(n_samples)
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    train_features = features_tensor[train_indices]
    train_labels = labels_tensor[train_indices]
    test_features = features_tensor[test_indices]
    test_labels = labels_tensor[test_indices]
    
    # Create data loaders
    train_dataset = TensorDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    test_dataset = TensorDataset(test_features, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    model = FederatedModel(input_dim=16, hidden_dim=64, num_classes=11)
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': []
    }
    
    # Training loop
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Track metrics
            train_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
        
        # Calculate average training metrics
        avg_train_loss = train_loss / train_total
        train_accuracy = train_correct / train_total
        
        # Evaluation phase
        test_metrics = evaluate_model(model, test_loader, criterion, device)
        
        # Store history
        history['train_loss'].append(avg_train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['accuracy'])
        
        # Log progress
        if verbose:
            logger.info(
                f"Epoch [{epoch+1}/{epochs}] - "
                f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Test Loss: {test_metrics['loss']:.4f}, Test Acc: {test_metrics['accuracy']:.4f}"
            )
    
    # Final evaluation
    final_test_metrics = evaluate_model(model, test_loader, criterion, device)
    
    if verbose:
        logger.info("Training complete!")
        logger.info(f"Final Test Accuracy: {final_test_metrics['accuracy']:.4f}")
    
    return {
        'model': model,
        'train_loss': history['train_loss'][-1],
        'train_accuracy': history['train_accuracy'][-1],
        'test_loss': final_test_metrics['loss'],
        'test_accuracy': final_test_metrics['accuracy'],
        'history': history,
        'n_samples': n_train
    }


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict:
    """
    Evaluate model on a dataset and compute accuracy and loss.
    
    Args:
        model: PyTorch model to evaluate
        data_loader: DataLoader containing evaluation data
        criterion: Loss function
        device: Device to evaluate on ('cpu' or 'cuda')
        
    Returns:
        Dictionary containing:
            - loss: Average loss on the dataset
            - accuracy: Classification accuracy
            - correct: Number of correct predictions
            - total: Total number of samples
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_features, batch_labels in data_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            
            # Track metrics
            total_loss += loss.item() * batch_features.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
    
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = correct / total if total > 0 else 0.0
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }


def save_local_model(model: nn.Module, path: str) -> None:
    """
    Save trained model weights to a file.
    
    Args:
        model: PyTorch model to save
        path: File path to save the model weights (.pth format)
        
    Raises:
        IOError: If the file cannot be written
    """
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model state dict
    torch.save(model.state_dict(), path)
    logger.info(f"Model weights saved to {path}")


def load_local_model(path: str, device: Optional[str] = None) -> FederatedModel:
    """
    Load model weights from a file.
    
    Args:
        path: File path to load the model weights from (.pth format)
        device: Device to load the model to ('cpu' or 'cuda'). If None, auto-detect
        
    Returns:
        FederatedModel instance with loaded weights
        
    Raises:
        FileNotFoundError: If the weights file does not exist
    """
    import os
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weights file not found: {path}")
    
    # Determine device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model and load weights
    model = FederatedModel(input_dim=16, hidden_dim=64, num_classes=11)
    model.load_weights(path, device=device)
    model = model.to(device)
    
    logger.info(f"Model weights loaded from {path}")
    
    return model


def train_traditional_model(
    features: np.ndarray,
    labels: np.ndarray,
    model_type: str = 'knn',
    test_split: float = 0.2,
    n_neighbors: int = 5,
    max_depth: Optional[int] = None,
    random_state: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Train a traditional ML model (KNN or Decision Tree) on extracted features.
    
    Args:
        features: Feature array of shape (n_samples, n_features)
        labels: Label array of shape (n_samples,)
        model_type: Type of model to train ('knn' or 'dt')
        test_split: Proportion of data to use for testing (default: 0.2)
        n_neighbors: Number of neighbors for KNN (default: 5)
        max_depth: Maximum depth for Decision Tree (default: None)
        random_state: Random state for reproducibility (default: 42)
        verbose: Whether to print training progress
        
    Returns:
        Dictionary containing:
            - model: Trained model instance (KNNModel or DecisionTreeModel)
            - model_type: Type of model ('knn' or 'dt')
            - train_accuracy: Training accuracy
            - test_accuracy: Test accuracy
            - training_time: Training time in seconds
            - inference_time_ms_per_sample: Inference time in milliseconds per sample
            - n_samples: Number of training samples
    """
    if verbose:
        logger.info(f"Training {model_type.upper()} model")
        logger.info(f"Dataset size: {len(features)} samples")
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_split, random_state=random_state, stratify=labels
    )
    
    if verbose:
        logger.info(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # Initialize model based on type
    if model_type.lower() == 'knn':
        model = KNNModel(n_neighbors=n_neighbors)
    elif model_type.lower() == 'dt':
        model = DecisionTreeModel(max_depth=max_depth, random_state=random_state)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'knn' or 'dt'")
    
    # Train model and get timing metrics
    timing_metrics = model.fit(X_train, y_train)
    
    if verbose:
        logger.info(f"Training complete! Time: {timing_metrics['training_time']:.3f} seconds")
    
    # Evaluate on training set
    train_predictions = model.model.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_predictions)
    
    # Evaluate on test set and measure inference time
    test_predictions, inference_time = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_predictions)
    
    if verbose:
        logger.info(f"Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"Test Accuracy: {test_accuracy:.4f}")
        logger.info(f"Inference Time: {inference_time:.3f} ms/sample")
    
    return {
        'model': model,
        'model_type': model_type.lower(),
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'training_time': timing_metrics['training_time'],
        'inference_time_ms_per_sample': inference_time,
        'n_samples': len(X_train)
    }


def save_traditional_model(model, path: str) -> None:
    """
    Save trained traditional ML model to a file.
    
    Args:
        model: KNNModel or DecisionTreeModel instance to save
        path: File path to save the model (.pkl format)
        
    Raises:
        IOError: If the file cannot be written
    """
    import os
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save model
    model.save(path)
    logger.info(f"Model saved to {path}")


def load_traditional_model(path: str, model_type: str = 'knn') -> object:
    """
    Load traditional ML model from a file.
    
    Args:
        path: File path to load the model from (.pkl format)
        model_type: Type of model to load ('knn' or 'dt')
        
    Returns:
        KNNModel or DecisionTreeModel instance with loaded weights
        
    Raises:
        FileNotFoundError: If the model file does not exist
    """
    import os
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    
    # Create model and load weights
    if model_type.lower() == 'knn':
        model = KNNModel()
    elif model_type.lower() == 'dt':
        model = DecisionTreeModel()
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'knn' or 'dt'")
    
    model.load(path)
    logger.info(f"Model loaded from {path}")
    
    return model
