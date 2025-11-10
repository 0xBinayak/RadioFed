"""
MLP Model - Current implementation
Multi-Layer Perceptron for feature-based classification
"""

import torch
import torch.nn as nn


class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron (MLP) for modulation classification.
    Uses extracted features (16-dimensional).
    """
    
    def __init__(self, input_dim: int = 16, hidden_dim: int = 64, num_classes: int = 11):
        super(MLPModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
