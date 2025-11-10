"""
CNN Model - Based on VT-CNN2 architecture from RadioML
Convolutional Neural Network for raw I/Q data classification
"""

import torch
import torch.nn as nn


class CNNModel(nn.Module):
    """
    VT-CNN2 architecture for RadioML dataset.
    Input: (batch_size, 2, 128) - I/Q samples
    Output: (batch_size, num_classes)
    """
    
    def __init__(self, num_classes: int = 11):
        super(CNNModel, self).__init__()
        
        self.num_classes = num_classes
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 256, kernel_size=(1, 3), padding=(0, 1))
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        
        self.conv2 = nn.Conv2d(256, 80, kernel_size=(2, 3), padding=(0, 2))
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        
        # Calculate flattened size: 80 channels * 1 height * 128 width
        self.flatten = nn.Flatten()
        
        # Dense layers
        self.fc1 = nn.Linear(80 * 1 * 128, 256)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, 2, 128)
        # Reshape to (batch, 1, 2, 128) for Conv2D
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        x = self.flatten(x)
        
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout3(x)
        
        x = self.fc2(x)
        
        return x
