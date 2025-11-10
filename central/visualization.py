"""
Visualization Utilities for AMC Dashboard

This module provides visualization functions for the AMC federated learning dashboard,
matching the style and format of the reference notebook (knndtamc-amc-rml2016a-updated.ipynb).

All functions return matplotlib figures or pandas DataFrames compatible with Gradio.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
import logging


logger = logging.getLogger(__name__)


# ============================================================================
# Plot Caching for Performance
# ============================================================================

class PlotCache:
    """
    Cache for generated plots to improve dashboard performance.
    
    Stores last 10 rounds of plots and invalidates cache when new aggregation completes.
    """
    
    def __init__(self, max_rounds: int = 10):
        """
        Initialize plot cache.
        
        Args:
            max_rounds: Maximum number of rounds to cache
        """
        self.max_rounds = max_rounds
        self._cache = {}
        self._current_round = 0
    
    def get(self, key: str, round_num: Optional[int] = None) -> Optional[plt.Figure]:
        """
        Get cached plot.
        
        Args:
            key: Cache key (e.g., 'confusion_matrix_knn')
            round_num: Round number (uses current round if None)
        
        Returns:
            Cached figure or None if not found
        """
        if round_num is None:
            round_num = self._current_round
        
        cache_key = f"{key}_round_{round_num}"
        return self._cache.get(cache_key)
    
    def set(self, key: str, figure: plt.Figure, round_num: Optional[int] = None) -> None:
        """
        Store plot in cache.
        
        Args:
            key: Cache key
            figure: Matplotlib figure to cache
            round_num: Round number (uses current round if None)
        """
        if round_num is None:
            round_num = self._current_round
        
        cache_key = f"{key}_round_{round_num}"
        self._cache[cache_key] = figure
        
        # Clean up old rounds
        self._cleanup_old_rounds()
    
    def invalidate(self, round_num: Optional[int] = None) -> None:
        """
        Invalidate cache for a specific round or all rounds.
        
        Args:
            round_num: Round number to invalidate (invalidates all if None)
        """
        if round_num is None:
            self._cache.clear()
            logger.info("Plot cache cleared")
        else:
            keys_to_remove = [k for k in self._cache.keys() if f"_round_{round_num}" in k]
            for key in keys_to_remove:
                del self._cache[key]
            logger.info(f"Plot cache invalidated for round {round_num}")
    
    def set_current_round(self, round_num: int) -> None:
        """
        Set current round number.
        
        Args:
            round_num: Current round number
        """
        self._current_round = round_num
    
    def _cleanup_old_rounds(self) -> None:
        """Remove plots from rounds older than max_rounds."""
        if self._current_round > self.max_rounds:
            min_round = self._current_round - self.max_rounds
            keys_to_remove = []
            
            for key in self._cache.keys():
                # Extract round number from key
                try:
                    round_str = key.split('_round_')[-1]
                    round_num = int(round_str)
                    if round_num < min_round:
                        keys_to_remove.append(key)
                except (ValueError, IndexError):
                    continue
            
            for key in keys_to_remove:
                del self._cache[key]
            
            if keys_to_remove:
                logger.info(f"Removed {len(keys_to_remove)} old plots from cache")


# Global plot cache instance
_plot_cache = PlotCache(max_rounds=10)


# ============================================================================
# Confusion Matrix Visualization
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: List[str],
    title: str = "Confusion Matrix"
) -> plt.Figure:
    """
    Generate confusion matrix heatmap using seaborn.
    
    Creates a heatmap visualization of the confusion matrix with annotations
    showing the count of predictions for each class combination.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class names (e.g., ['AM', 'FM'])
        title: Plot title
    
    Returns:
        Matplotlib figure compatible with Gradio gr.Plot()
    
    Example:
        >>> y_true = np.array([0, 1, 0, 1, 1])
        >>> y_pred = np.array([0, 1, 1, 1, 0])
        >>> fig = plot_confusion_matrix(y_true, y_pred, ['AM', 'FM'], 'KNN Confusion Matrix')
    """
    # Compute confusion matrix
    cm = sklearn_confusion_matrix(y_true, y_pred)
    
    # Create figure with specified size (8, 6)
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Create heatmap with Blues colormap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
        cbar=True,
        square=True
    )
    
    # Format axes labels
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)
    
    # Apply tight layout
    plt.tight_layout()
    
    return fig


# ============================================================================
# Accuracy vs SNR Plot
# ============================================================================

def plot_accuracy_vs_snr(
    snr_values: List[float],
    dt_acc: List[float],
    knn_acc: List[float],
    baseline_acc: List[float],
    title: str = "Model Accuracy vs SNR"
) -> plt.Figure:
    """
    Generate accuracy vs SNR line plot with multiple model curves.
    
    Creates a line plot comparing baseline, Decision Tree, and KNN accuracy
    across different SNR levels. Uses different markers and line styles for
    each model.
    
    Args:
        snr_values: List of SNR values in dB (e.g., [-20, -18, ..., 18])
        dt_acc: Decision Tree accuracy values (%)
        knn_acc: KNN accuracy values (%)
        baseline_acc: Baseline accuracy values (%)
        title: Plot title
    
    Returns:
        Matplotlib figure compatible with Gradio gr.Plot()
    
    Example:
        >>> snr_values = list(range(-20, 20, 2))
        >>> baseline = [50.0] * len(snr_values)
        >>> dt_acc = [45.0 + i*2 for i in range(len(snr_values))]
        >>> knn_acc = [48.0 + i*2.5 for i in range(len(snr_values))]
        >>> fig = plot_accuracy_vs_snr(snr_values, dt_acc, knn_acc, baseline)
    """
    # Create figure with specified size (10, 6)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot three curves with specified markers
    # Baseline: dashed line with 'o' markers
    ax.plot(
        snr_values,
        baseline_acc,
        'k--',
        marker='o',
        label='Baseline',
        alpha=0.7,
        linewidth=1.5,
        markersize=6
    )
    
    # Decision Tree: solid line with 'x' markers
    ax.plot(
        snr_values,
        dt_acc,
        'b-',
        marker='x',
        label='Decision Tree',
        linewidth=2,
        markersize=8
    )
    
    # KNN: solid line with 's' (square) markers
    ax.plot(
        snr_values,
        knn_acc,
        'r-',
        marker='s',
        label='KNN',
        linewidth=2,
        markersize=6
    )
    
    # Configure axes
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.set_title(title)
    
    # Add grid with specified style
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='best')
    
    # Apply tight layout
    plt.tight_layout()
    
    return fig


# ============================================================================
# Feature Distribution Plots
# ============================================================================

def plot_feature_distributions(
    dataframe: pd.DataFrame,
    feature_name: str,
    title: str = "Feature Distribution"
) -> plt.Figure:
    """
    Generate feature distribution histogram with KDE overlay.
    
    Creates a histogram showing the distribution of a feature across different
    modulation classes. Uses seaborn histplot with KDE and viridis palette.
    
    Args:
        dataframe: DataFrame with columns 'Feature' and 'Modulation'
        feature_name: Name of the feature being plotted
        title: Plot title
    
    Returns:
        Matplotlib figure compatible with Gradio gr.Plot()
    
    Example:
        >>> df = pd.DataFrame({
        ...     'Feature': np.concatenate([np.random.normal(0, 1, 500), 
        ...                                np.random.normal(2, 1.5, 500)]),
        ...     'Modulation': ['AM'] * 500 + ['FM'] * 500
        ... })
        >>> fig = plot_feature_distributions(df, 'amp_kurtosis', 'Amplitude Kurtosis')
    """
    # Sample 1000 random points if dataset is large
    if len(dataframe) > 1000:
        sampled_df = dataframe.sample(n=1000, random_state=42)
    else:
        sampled_df = dataframe
    
    # Create figure with specified size (8, 5)
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Create histogram with KDE and viridis palette
    sns.histplot(
        data=sampled_df,
        x='Feature',
        hue='Modulation',
        kde=True,
        palette='viridis',
        ax=ax,
        alpha=0.6,
        stat='count'
    )
    
    # Configure axes and labels
    ax.set_xlabel(feature_name.replace('_', ' ').title())
    ax.set_ylabel('Count')
    ax.set_title(title)
    
    # Add grid on y-axis only
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Apply tight layout
    plt.tight_layout()
    
    return fig


# ============================================================================
# Computation Complexity Table
# ============================================================================

def generate_complexity_table(
    dt_metrics: Dict[str, float],
    knn_metrics: Dict[str, float]
) -> pd.DataFrame:
    """
    Generate computation complexity comparison table.
    
    Creates a pandas DataFrame comparing training time and inference time
    for Decision Tree and KNN models.
    
    Args:
        dt_metrics: Dictionary with 'training_time' and 'inference_time' keys
        knn_metrics: Dictionary with 'training_time' and 'inference_time' keys
    
    Returns:
        DataFrame with columns: Method, Training Time (seconds), 
        Average Inference Time (ms/sample)
    
    Example:
        >>> dt_metrics = {'training_time': 1.234, 'inference_time': 0.456}
        >>> knn_metrics = {'training_time': 2.345, 'inference_time': 1.234}
        >>> df = generate_complexity_table(dt_metrics, knn_metrics)
    """
    # Extract metrics with defaults
    dt_training = dt_metrics.get('training_time', 0.0)
    dt_inference = dt_metrics.get('inference_time', 0.0)
    knn_training = knn_metrics.get('training_time', 0.0)
    knn_inference = knn_metrics.get('inference_time', 0.0)
    
    # Format to 3 decimal places
    data = [
        ["Decision Tree", round(dt_training, 3), round(dt_inference, 3)],
        ["K-Nearest Neighbors", round(knn_training, 3), round(knn_inference, 3)]
    ]
    
    # Create DataFrame
    df = pd.DataFrame(
        data,
        columns=["Method", "Training Time (seconds)", "Average Inference Time (ms/sample)"]
    )
    
    return df


# ============================================================================
# Cache Management Functions
# ============================================================================

def get_plot_cache() -> PlotCache:
    """
    Get the global plot cache instance.
    
    Returns:
        PlotCache instance
    """
    return _plot_cache


def invalidate_plot_cache(round_num: Optional[int] = None) -> None:
    """
    Invalidate plot cache for a specific round or all rounds.
    
    Args:
        round_num: Round number to invalidate (invalidates all if None)
    """
    _plot_cache.invalidate(round_num)


def set_cache_round(round_num: int) -> None:
    """
    Set current round number for plot cache.
    
    Args:
        round_num: Current round number
    """
    _plot_cache.set_current_round(round_num)


# ============================================================================
# Convenience Functions for Dashboard Integration
# ============================================================================

def create_confusion_matrix_from_results(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_type: str,
    snr: Optional[float] = None
) -> plt.Figure:
    """
    Create confusion matrix with automatic title formatting.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_type: Type of model ('knn' or 'dt')
        snr: SNR value for title (optional)
    
    Returns:
        Matplotlib figure
    """
    classes = ['AM', 'FM']
    
    # Format title
    model_name = model_type.upper()
    if snr is not None:
        title = f"{model_name} Confusion Matrix (SNR = {snr} dB)"
    else:
        title = f"{model_name} Confusion Matrix"
    
    return plot_confusion_matrix(y_true, y_pred, classes, title)


def create_feature_distribution_plot(
    features: np.ndarray,
    labels: np.ndarray,
    feature_name: str,
    class_names: List[str] = None
) -> plt.Figure:
    """
    Create feature distribution plot from raw data.
    
    Args:
        features: Feature values (1D array)
        labels: Class labels (1D array)
        feature_name: Name of the feature
        class_names: List of class names (defaults to ['AM', 'FM'])
    
    Returns:
        Matplotlib figure
    """
    if class_names is None:
        class_names = ['AM', 'FM']
    
    # Create DataFrame
    df = pd.DataFrame({
        'Feature': features,
        'Modulation': [class_names[int(label)] for label in labels]
    })
    
    # Format title
    title = f"{feature_name.replace('_', ' ').title()} Distribution"
    
    return plot_feature_distributions(df, feature_name, title)


def create_accuracy_comparison_plot(
    snr_values: List[float],
    dt_per_snr: Dict[float, float],
    knn_per_snr: Dict[float, float],
    baseline: float = 50.0
) -> plt.Figure:
    """
    Create accuracy vs SNR plot from per-SNR accuracy dictionaries.
    
    Args:
        snr_values: List of SNR values
        dt_per_snr: Dictionary mapping SNR to Decision Tree accuracy
        knn_per_snr: Dictionary mapping SNR to KNN accuracy
        baseline: Baseline accuracy (default 50.0 for 2-class)
    
    Returns:
        Matplotlib figure
    """
    # Extract accuracy values in order
    dt_acc = [dt_per_snr.get(snr, 0.0) for snr in snr_values]
    knn_acc = [knn_per_snr.get(snr, 0.0) for snr in snr_values]
    baseline_acc = [baseline] * len(snr_values)
    
    return plot_accuracy_vs_snr(
        snr_values,
        dt_acc,
        knn_acc,
        baseline_acc,
        "Model Accuracy vs SNR"
    )
