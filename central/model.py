"""
Federated Model Architecture for Central Server

This module defines the neural network architecture used for the global model
in the federated learning system. The model is designed for RadioML modulation
classification with 16-dimensional feature vectors as input.
"""

import torch
import torch.nn as nn
import os
from typing import Optional


class FederatedModel(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for modulation classification.
    
    Architecture:
    - Input layer: 16 features (extracted from I/Q samples)
    - Hidden layer: 64 neurons with ReLU activation
    - Dropout: 0.3 for regularization
    - Output layer: 11 classes (RadioML modulation types)
    
    Args:
        input_dim (int): Number of input features (default: 16)
        hidden_dim (int): Number of hidden layer neurons (default: 64)
        num_classes (int): Number of output classes (default: 11)
    """
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_classes: int = 11):
        super(FederatedModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # Define layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def save_weights(self, path: str) -> None:
        """
        Save model weights to a file.
        
        Args:
            path (str): File path to save the model weights (.pth format)
        
        Raises:
            IOError: If the file cannot be written
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path: str, device: Optional[str] = None) -> None:
        """
        Load model weights from a file.
        
        Args:
            path (str): File path to load the model weights from (.pth format)
            device (str, optional): Device to load the weights to ('cpu' or 'cuda')
        
        Raises:
            FileNotFoundError: If the weights file does not exist
            RuntimeError: If the weights are incompatible with the model architecture
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Weights file not found: {path}")
        
        # Determine device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load state dict
        state_dict = torch.load(path, map_location=device)
        self.load_state_dict(state_dict)
    
    def get_architecture_info(self) -> dict:
        """
        Get information about the model architecture.
        
        Returns:
            dict: Dictionary containing architecture parameters
        """
        return {
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'num_classes': self.num_classes,
            'total_parameters': sum(p.numel() for p in self.parameters())
        }
