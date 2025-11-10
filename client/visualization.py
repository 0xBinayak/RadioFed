"""
Visualization utilities for federated learning
Generates plots for confusion matrices, training history, and metrics
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
import io
import base64


def plot_to_base64(fig):
    """Convert matplotlib figure to base64 string for Gradio"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{img_base64}"


def create_confusion_matrix(y_true, y_pred, classes):
    """
    Generate confusion matrix plot
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names
        
    Returns:
        Base64 encoded image string
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_xlabel('Predicted Label', fontsize=12)
    
    return plot_to_base64(fig)


def create_training_history_plot(history):
    """
    Plot training loss and accuracy history
    
    Args:
        history: Dictionary with keys 'train_loss', 'test_loss', 
                'train_accuracy', 'test_accuracy'
                
    Returns:
        Base64 encoded image string
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, history['test_loss'], 'r-', label='Test Loss', linewidth=2)
    ax1.set_title('Loss History', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_accuracy'], 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, history['test_accuracy'], 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_title('Accuracy History', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_accuracy_vs_snr_plot(snr_values, accuracies_dict):
    """
    Plot accuracy vs SNR for different models
    
    Args:
        snr_values: List of SNR values
        accuracies_dict: Dict with model names as keys and accuracy lists as values
        
    Returns:
        Base64 encoded image string
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    markers = ['o', 's', '^', 'D', 'v']
    for idx, (model_name, accuracies) in enumerate(accuracies_dict.items()):
        marker = markers[idx % len(markers)]
        ax.plot(snr_values, accuracies, marker=marker, linewidth=2, 
                markersize=8, label=model_name)
    
    ax.set_title('Classification Accuracy vs. SNR', fontsize=14, fontweight='bold')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    return plot_to_base64(fig)


def create_feature_distribution_plot(features, labels, feature_name, feature_idx):
    """
    Plot feature distribution for different classes
    
    Args:
        features: Feature array (n_samples, n_features)
        labels: Label array (n_samples,)
        feature_name: Name of the feature
        feature_idx: Index of the feature to plot
        
    Returns:
        Base64 encoded image string
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        feature_values = features[mask, feature_idx]
        ax.hist(feature_values, bins=30, alpha=0.5, label=f'Class {label}')
    
    ax.set_title(f'Distribution of {feature_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel(feature_name, fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    return plot_to_base64(fig)
