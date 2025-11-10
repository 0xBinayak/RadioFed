"""
Traditional ML Models - KNN and Decision Tree
Scikit-learn based classifiers for feature-based classification
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pickle
import time
from typing import Dict, Tuple


class KNNModel:
    """K-Nearest Neighbors classifier wrapper with timing instrumentation"""
    
    def __init__(self, n_neighbors: int = 5):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.n_neighbors = n_neighbors
        self.training_time = None
        self.inference_time_ms_per_sample = None
    
    def fit(self, X, y) -> Dict[str, float]:
        """
        Train the KNN model and measure training time
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary with timing metrics:
            - training_time: Training time in seconds
            - inference_time_ms_per_sample: Not available until predict is called
        """
        start_time = time.time()
        self.model.fit(X, y)
        end_time = time.time()
        
        self.training_time = end_time - start_time
        
        return {
            'training_time': self.training_time,
            'inference_time_ms_per_sample': self.inference_time_ms_per_sample
        }
    
    def predict(self, X) -> Tuple[np.ndarray, float]:
        """
        Predict using KNN and measure inference time
        
        Args:
            X: Test features
            
        Returns:
            Tuple of (predictions, inference_time_ms_per_sample)
        """
        start_time = time.time()
        predictions = self.model.predict(X)
        end_time = time.time()
        
        total_time = end_time - start_time
        self.inference_time_ms_per_sample = (total_time / len(X)) * 1000
        
        return predictions, self.inference_time_ms_per_sample
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        """Save model to file using pickle"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str):
        """Load model from file using pickle"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)


class DecisionTreeModel:
    """Decision Tree classifier wrapper with timing instrumentation"""
    
    def __init__(self, max_depth: int = None, random_state: int = 42):
        self.model = DecisionTreeClassifier(
            max_depth=max_depth,
            random_state=random_state
        )
        self.max_depth = max_depth
        self.training_time = None
        self.inference_time_ms_per_sample = None
    
    def fit(self, X, y) -> Dict[str, float]:
        """
        Train the Decision Tree model and measure training time
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary with timing metrics:
            - training_time: Training time in seconds
            - inference_time_ms_per_sample: Not available until predict is called
        """
        start_time = time.time()
        self.model.fit(X, y)
        end_time = time.time()
        
        self.training_time = end_time - start_time
        
        return {
            'training_time': self.training_time,
            'inference_time_ms_per_sample': self.inference_time_ms_per_sample
        }
    
    def predict(self, X) -> Tuple[np.ndarray, float]:
        """
        Predict using Decision Tree and measure inference time
        
        Args:
            X: Test features
            
        Returns:
            Tuple of (predictions, inference_time_ms_per_sample)
        """
        start_time = time.time()
        predictions = self.model.predict(X)
        end_time = time.time()
        
        total_time = end_time - start_time
        self.inference_time_ms_per_sample = (total_time / len(X)) * 1000
        
        return predictions, self.inference_time_ms_per_sample
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def save(self, path: str):
        """Save model to file using pickle"""
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: str):
        """Load model from file using pickle"""
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
