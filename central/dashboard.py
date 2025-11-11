"""
Central Dashboard for Federated Learning AMC System

This module provides a comprehensive Gradio-based dashboard for monitoring
federated learning training, visualizing model performance, and tracking
client status for Automatic Modulation Classification (AMC).
"""

import gradio as gr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.metrics import confusion_matrix

from central.state import (
    get_client_status,
    get_registry_stats,
    get_latest_aggregation_result,
    get_all_aggregation_results,
    get_accuracy_trends,
    get_latest_round_metrics,
    get_auto_aggregation_state
)


# Configure logging
logger = logging.getLogger(__name__)


class DashboardState:
    """
    Dashboard state management class for tracking metrics and visualization data.
    """
    
    def __init__(self):
        """Initialize dashboard state."""
        self.current_round = 0
        self.server_running = True
        self.last_update = datetime.now()
        
        # Metrics storage
        self.metrics_history = []
        self.before_aggregation = {}
        self.after_aggregation = {}
        self.complexity_metrics = {}
        
        # SNR levels for RadioML 2016.10a
        self.snr_levels = list(range(-20, 20, 2))  # -20 to 18 dB
        self.modulation_classes = ['AM', 'FM']
        self.num_classes = len(self.modulation_classes)
    
    def update_metrics(self, model_type: str, metrics: Dict):
        """
        Update metrics for a specific model type.
        
        Args:
            model_type: Type of model ('knn' or 'dt')
            metrics: Dictionary containing metrics data
        """
        self.metrics_history.append({
            'timestamp': datetime.now().isoformat(),
            'model_type': model_type,
            'metrics': metrics
        })
        self.last_update = datetime.now()
    
    def get_baseline_accuracy(self) -> float:
        """
        Calculate baseline accuracy (random guess).
        
        Returns:
            Baseline accuracy as percentage
        """
        return 100.0 / self.num_classes
    
    def collect_metrics(self) -> Dict:
        """
        Collect current metrics from state management.
        
        Returns:
            Dictionary containing current metrics
        """
        # Get latest aggregation results (KNN only)
        knn_result = get_latest_aggregation_result('knn')
        
        return {
            'knn': knn_result,
            'timestamp': datetime.now().isoformat()
        }


# Global dashboard state instance
dashboard_state = DashboardState()


def create_dashboard_interface() -> gr.Blocks:
    """
    Create the main Gradio dashboard interface with all sections.
    
    Applies clean UI styling matching notebook aesthetic:
    - Uses matplotlib default style or seaborn style
    - Removes emoji from plot titles (keeps in status indicators only)
    - Organizes plots in 2-column grid layout using gr.Row()
    - Adds clear section headers with gr.Markdown("## Section Name")
    - Sets consistent figure sizes: (10, 6) for main plots, (8, 5) for distributions
    - Uses tight_layout() for all matplotlib figures
    
    Returns:
        Gradio Blocks interface
    """
    # Set matplotlib style for clean aesthetic
    plt.style.use('default')
    sns.set_palette('viridis')
    
    with gr.Blocks(title="AMC Federated Learning Dashboard", theme=gr.themes.Soft()) as dashboard:
        gr.Markdown("# AMC Federated Learning Dashboard")
        gr.Markdown("Real-time monitoring and visualization for analog modulation classification")
        
        # Auto-refresh timer (2 seconds)
        timer = gr.Timer(value=2, active=True)
        
        # System Status Section
        gr.Markdown("## System Status")
        with gr.Row():
            with gr.Column(scale=1):
                server_status = gr.Markdown(value="🟢 Server is running")
                connected_clients = gr.Markdown(value="Connected Clients: 0")
            with gr.Column(scale=1):
                current_round = gr.Markdown(value="Training Round: 0")
                last_aggregation = gr.Markdown(value="Last Aggregation: Never")
                aggregation_progress = gr.Markdown(value="⏳ Waiting for uploads (0/2 clients)")
        
        # Client Monitoring Section
        gr.Markdown("## Client Monitoring")
        with gr.Row():
            with gr.Column():
                client_table = gr.Dataframe(
                    headers=["Client ID", "Status", "Last Upload", "Sample Count"],
                    datatype=["str", "str", "str", "number"],
                    label="Connected Clients"
                )
        
        # Training Progress Section (NEW)
        gr.Markdown("## Training Progress")
        with gr.Row():
            with gr.Column():
                historical_trends_plot = gr.Plot(label="Historical Accuracy Trends")
        
        # Latest Aggregation Results Section (NEW)
        gr.Markdown("## Latest Aggregation Results")
        with gr.Row():
            with gr.Column():
                before_after_table = gr.Dataframe(
                    headers=["Metric", "Before", "After", "Improvement"],
                    datatype=["str", "str", "str", "str"],
                    label="Before/After Comparison"
                )
        
        # Training Metrics Comparison Section
        gr.Markdown("## Training Metrics Comparison")
        with gr.Row():
            with gr.Column():
                metrics_table = gr.Dataframe(
                    headers=["SNR (dB)", "Baseline Accuracy (%)", "KNN Accuracy (%)"],
                    datatype=["number", "number", "number"],
                    label="Accuracy by SNR Level"
                )
        
        # Confusion Matrix Section
        gr.Markdown("## Confusion Matrix")
        with gr.Row():
            with gr.Column():
                knn_confusion_plot = gr.Plot(label="KNN Confusion Matrix")
        
        # Accuracy vs SNR Plot Section
        gr.Markdown("## Accuracy vs SNR")
        with gr.Row():
            with gr.Column():
                accuracy_snr_plot = gr.Plot(label="Model Performance Across SNR Levels")
        
        # Computation Complexity Table Section
        gr.Markdown("## Computation Complexity")
        with gr.Row():
            with gr.Column():
                complexity_table = gr.Dataframe(
                    headers=["Method", "Training Time (seconds)", "Average Inference Time (ms/sample)"],
                    datatype=["str", "number", "number"],
                    label="Model Complexity Comparison"
                )
        
        # Wire up auto-refresh
        timer.tick(
            fn=update_dashboard,
            outputs=[
                server_status,
                connected_clients,
                current_round,
                last_aggregation,
                aggregation_progress,
                client_table,
                historical_trends_plot,
                before_after_table,
                metrics_table,
                knn_confusion_plot,
                accuracy_snr_plot,
                complexity_table
            ]
        )
    
    return dashboard


def update_dashboard() -> Tuple:
    """
    Update all dashboard components with latest data.
    
    Returns:
        Tuple of updated values for all dashboard components
    """
    # Update system status
    server_status_text = get_server_status()
    connected_clients_text = get_connected_clients_text()
    current_round_text = get_current_round_text()
    last_aggregation_text = get_last_aggregation_text()
    aggregation_progress_text = get_aggregation_progress()
    
    # Update client monitoring
    client_table_data = get_client_monitoring_data()
    
    # Update historical trends
    historical_trends_fig = generate_historical_trends_plot()
    
    # Update before/after comparison
    before_after_table_data = get_before_after_comparison()
    
    # Update training metrics
    metrics_table_data = get_training_metrics_table()
    
    # Update confusion matrix (KNN only)
    knn_confusion_fig = generate_confusion_matrix('knn')
    
    # Update accuracy vs SNR plot
    accuracy_snr_fig = generate_accuracy_vs_snr_plot()
    
    # Update complexity table
    complexity_table_data = get_complexity_table()
    
    return (
        server_status_text,
        connected_clients_text,
        current_round_text,
        last_aggregation_text,
        aggregation_progress_text,
        client_table_data,
        historical_trends_fig,
        before_after_table_data,
        metrics_table_data,
        knn_confusion_fig,
        accuracy_snr_fig,
        complexity_table_data
    )


def get_server_status() -> str:
    """
    Get server status indicator with emoji.
    
    Returns:
        Status string with emoji indicator
    """
    if dashboard_state.server_running:
        return "🟢 Server is running"
    else:
        return "🔴 Server is stopped"


def get_connected_clients_text() -> str:
    """
    Get connected clients count and IDs.
    
    Returns:
        Formatted string with client count and IDs
    """
    stats = get_registry_stats()
    client_count = stats['total_clients']
    client_ids = stats.get('client_ids', [])
    
    if client_count == 0:
        return "Connected Clients: 0"
    
    # Format client IDs for display
    if len(client_ids) <= 5:
        ids_display = ", ".join(client_ids)
        return f"Connected Clients: {client_count} ({ids_display})"
    else:
        ids_display = ", ".join(client_ids[:5]) + f", ... (+{len(client_ids) - 5} more)"
        return f"Connected Clients: {client_count} ({ids_display})"


def get_current_round_text() -> str:
    """
    Get current training round number.
    
    Returns:
        Formatted string with round number
    """
    # Get current round from auto-aggregation state
    state = get_auto_aggregation_state()
    current_round = state.get('current_round', 0)
    return f"Training Round: {current_round}"


def get_last_aggregation_text() -> str:
    """
    Get last aggregation timestamp (formatted).
    
    Returns:
        Formatted string with last aggregation time
    """
    # Check for latest aggregation (KNN only)
    knn_result = get_latest_aggregation_result('knn')
    
    if knn_result:
        timestamp = knn_result.get('timestamp')
        try:
            dt = datetime.fromisoformat(timestamp)
            formatted = dt.strftime("%Y-%m-%d %H:%M:%S")
            return f"Last Aggregation: {formatted}"
        except:
            return f"Last Aggregation: {timestamp}"
    else:
        return "Last Aggregation: Never"


def get_client_monitoring_data() -> pd.DataFrame:
    """
    Get client monitoring data for display.
    
    Displays list of connected clients with columns: Client ID, Status,
    Last Upload, Sample Count.
    Shows training progress for active clients (percentage complete).
    Indicates which clients have submitted updates with checkmark.
    Updates client list automatically with gr.Timer().
    
    Returns:
        DataFrame with client information for Gradio gr.Dataframe()
    """
    clients = get_client_status()
    
    if not clients:
        return pd.DataFrame(columns=["Client ID", "Status", "Last Upload", "Sample Count"])
    
    data = []
    for client in clients:
        status = client.get('status', 'unknown')
        
        # Status display with indicators
        if status == 'weights_uploaded':
            status_display = "✓ Uploaded"
        elif status == 'training':
            # Show training progress if available
            progress = client.get('training_progress', 0)
            if progress > 0:
                status_display = f"⏳ Training ({progress}%)"
            else:
                status_display = "⏳ Training"
        elif status == 'idle':
            status_display = "Idle"
        else:
            status_display = "Connected"
        
        # Format last upload timestamp
        last_upload = client.get('last_upload', 'Never')
        if last_upload and last_upload != 'Never':
            try:
                dt = datetime.fromisoformat(last_upload)
                last_upload = dt.strftime("%Y-%m-%d %H:%M:%S")
            except:
                pass
        
        data.append([
            client['client_id'],
            status_display,
            last_upload,
            client.get('n_samples', 0)
        ])
    
    return pd.DataFrame(data, columns=["Client ID", "Status", "Last Upload", "Sample Count"])


def get_training_metrics_table() -> pd.DataFrame:
    """
    Generate training metrics comparison table (Table 1 from notebook).
    
    Creates a DataFrame with columns: SNR (dB), Baseline Accuracy (%),
    Decision Tree Accuracy (%), KNN Accuracy (%)
    
    Displays accuracy for each SNR level (-20 to 18 dB).
    Baseline is calculated as 100 / num_classes (random guess).
    Percentages are formatted to 2 decimal places.
    
    Returns:
        DataFrame with SNR-level accuracy comparison
    """
    # Calculate baseline accuracy (random guess)
    baseline = dashboard_state.get_baseline_accuracy()
    
    # Get latest aggregation results (KNN only)
    knn_result = get_latest_aggregation_result('knn')
    
    # Extract per-SNR accuracy if available
    knn_per_snr = {}
    
    if knn_result and 'result' in knn_result:
        knn_per_snr = knn_result['result'].get('per_snr_accuracy', {})
    
    # Build table data
    data = []
    for snr in dashboard_state.snr_levels:
        # Get accuracy for this SNR level, default to 0.0 if not available
        # Try both int and float keys (JSON may convert float keys to strings)
        knn_acc = knn_per_snr.get(snr, knn_per_snr.get(float(snr), knn_per_snr.get(str(snr), 0.0)))
        
        # Convert to percentage if needed (values might be 0-1 or 0-100)
        if isinstance(knn_acc, (int, float)) and knn_acc <= 1.0 and knn_acc > 0:
            knn_acc = knn_acc * 100
        
        # Format to 2 decimal places
        data.append([
            snr,
            round(baseline, 2),
            round(knn_acc, 2)
        ])
    
    return pd.DataFrame(
        data,
        columns=["SNR (dB)", "Baseline Accuracy (%)", "KNN Accuracy (%)"]
    )


def generate_confusion_matrix(model_type: str) -> plt.Figure:
    """
    Generate confusion matrix visualization (Tables 3 & 4 from notebook).
    
    Uses sklearn.metrics.confusion_matrix to compute the matrix.
    Creates heatmap with sns.heatmap(annot=True, fmt='d', cmap='Blues').
    Displays confusion matrices at SNR = 0 dB for both models.
    Uses modulation labels: ['AM', 'FM'] for analog classification.
    Shows the round number in the title to indicate which aggregation round.
    
    Args:
        model_type: Type of model ('knn' or 'dt')
    
    Returns:
        Matplotlib figure with confusion matrix heatmap for Gradio gr.Plot()
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Get latest aggregation result
    result = get_latest_aggregation_result(model_type)
    
    # Extract confusion matrix and round number if available
    round_num = dashboard_state.current_round
    if result and 'result' in result:
        cm_data = result['result'].get('confusion_matrix')
        # Try to get round number from result metadata
        if 'round' in result:
            round_num = result['round']
        
        if cm_data is not None:
            # Convert to numpy array if needed
            if isinstance(cm_data, list):
                cm = np.array(cm_data)
            else:
                cm = cm_data
        else:
            # Placeholder confusion matrix if no data available
            cm = np.array([[0, 0], [0, 0]])
    else:
        # Placeholder confusion matrix
        cm = np.array([[0, 0], [0, 0]])
    
    # Create heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=dashboard_state.modulation_classes,
        yticklabels=dashboard_state.modulation_classes,
        ax=ax,
        cbar=True
    )
    
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    
    # Update title to include round number
    if round_num > 0:
        ax.set_title(f'{model_type.upper()} Confusion Matrix - Round {round_num} (SNR = 0 dB)')
    else:
        ax.set_title(f'{model_type.upper()} Confusion Matrix (SNR = 0 dB)')
    
    plt.tight_layout()
    return fig


def generate_accuracy_vs_snr_plot() -> plt.Figure:
    """
    Generate accuracy vs SNR line plot (matching notebook figure).
    
    Creates line plot with plt.plot(snr_values, accuracy, marker='o').
    Plots three curves: Baseline (dashed), Decision Tree (solid), KNN (solid).
    Uses different markers: 'o' for baseline, 'x' for DT, 's' for KNN.
    Sets y-axis limits: 0 to 105%.
    Adds grid with plt.grid(True, linestyle='--', alpha=0.7).
    Adds legend with model names and improvement percentages from baseline.
    Uses latest global model metrics after aggregation.
    
    Returns:
        Matplotlib figure for Gradio gr.Plot()
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    snr_values = dashboard_state.snr_levels
    baseline_acc = dashboard_state.get_baseline_accuracy()
    baseline = [baseline_acc] * len(snr_values)
    
    # Get latest aggregation results (KNN only)
    knn_result = get_latest_aggregation_result('knn')
    
    # Extract per-SNR accuracy
    knn_accuracy = []
    
    for snr in snr_values:
        # Get KNN accuracy for this SNR (try multiple key formats)
        if knn_result and 'result' in knn_result:
            knn_per_snr = knn_result['result'].get('per_snr_accuracy', {})
            knn_acc = knn_per_snr.get(snr, knn_per_snr.get(float(snr), knn_per_snr.get(str(snr), 0.0)))
            # Convert to percentage if needed
            if isinstance(knn_acc, (int, float)) and knn_acc <= 1.0 and knn_acc > 0:
                knn_acc = knn_acc * 100
            knn_accuracy.append(knn_acc)
        else:
            knn_accuracy.append(0.0)
    
    # Calculate average improvement from baseline
    knn_avg_improvement = 0.0
    
    if knn_accuracy and any(knn_accuracy):
        valid_knn = [acc for acc in knn_accuracy if acc > 0]
        if valid_knn:
            knn_avg = sum(valid_knn) / len(valid_knn)
            knn_avg_improvement = knn_avg - baseline_acc
    
    # Plot two curves with specified markers and improvement in legend
    ax.plot(snr_values, baseline, 'k--', marker='o', label='Baseline', alpha=0.7, linewidth=1.5)
    
    if knn_avg_improvement > 0:
        ax.plot(snr_values, knn_accuracy, 'r-', marker='s', 
                label=f'KNN (+{knn_avg_improvement:.1f}% avg)', 
                linewidth=2, markersize=6)
    else:
        ax.plot(snr_values, knn_accuracy, 'r-', marker='s', 
                label='KNN', linewidth=2, markersize=6)
    
    # Configure axes and grid
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Accuracy (%)')
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    ax.set_title('Model Accuracy vs SNR')
    
    plt.tight_layout()
    return fig





def get_complexity_table() -> pd.DataFrame:
    """
    Generate computation complexity table for KNN model.
    
    Creates pandas DataFrame with columns: Method, Training Time (seconds),
    Average Inference Time (ms/sample).
    Shows row for "K-Nearest Neighbors" only.
    Formats training time to 3 decimal places (e.g., "2.345").
    Formats inference time to 3 decimal places (e.g., "1.234").
    Shows latest timing metrics from aggregation results.
    
    Returns:
        DataFrame for Gradio gr.Dataframe()
    """
    # Get latest aggregation results (KNN only)
    knn_result = get_latest_aggregation_result('knn')
    
    # Extract timing metrics with consistent defaults
    knn_training_time = 0.000
    knn_inference_time = 0.000
    
    if knn_result and 'result' in knn_result:
        knn_metrics = knn_result['result']
        knn_training_time = float(knn_metrics.get('training_time', 0.0))
        knn_inference_time = float(knn_metrics.get('inference_time_ms_per_sample', 0.0))
    
    # Check dashboard state for complexity metrics (fallback)
    if 'knn' in dashboard_state.complexity_metrics:
        knn_training_time = float(dashboard_state.complexity_metrics['knn'].get('training_time', knn_training_time))
        knn_inference_time = float(dashboard_state.complexity_metrics['knn'].get('inference_time', knn_inference_time))
    
    # Format to 3 decimal places consistently
    data = [
        ["K-Nearest Neighbors", f"{knn_training_time:.3f}", f"{knn_inference_time:.3f}"]
    ]
    
    return pd.DataFrame(
        data,
        columns=["Method", "Training Time (seconds)", "Average Inference Time (ms/sample)"]
    )


def generate_historical_trends_plot() -> plt.Figure:
    """
    Generate historical accuracy trends over training rounds for KNN.
    
    Fetches the last 10 rounds from metrics history and plots before/after
    accuracy for KNN model. Uses line plots with markers to show the 
    progression of model performance over time.
    
    Returns:
        Matplotlib figure for Gradio gr.Plot()
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Fetch accuracy trends from last 10 rounds
    trends = get_accuracy_trends()
    
    # Handle case with no history data
    if not trends['rounds']:
        ax.text(0.5, 0.5, 'No training history available',
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    rounds = trends['rounds']
    
    # Plot KNN trends only
    ax.plot(rounds, trends['knn_before'], 'r--', marker='o', 
            label='KNN (Before Agg)', alpha=0.6, linewidth=1.5, markersize=6)
    ax.plot(rounds, trends['knn_after'], 'r-', marker='o', 
            label='KNN (After Agg)', linewidth=2, markersize=6)
    
    # Configure axes and styling
    ax.set_xlabel('Training Round')
    ax.set_ylabel('Accuracy')
    ax.set_title('KNN Accuracy Trends Over Training Rounds')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc='best')
    
    # Set y-axis limits to show percentage scale
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    return fig


def get_before_after_comparison() -> pd.DataFrame:
    """
    Generate before/after comparison table for the latest aggregation round.
    
    Fetches the latest round metrics and formats them as a DataFrame showing
    the accuracy before aggregation, after aggregation, and the improvement
    percentage for KNN model.
    
    Returns:
        DataFrame for Gradio gr.Dataframe() with columns:
        Metric, Before, After, Improvement
    """
    # Fetch latest round metrics
    latest = get_latest_round_metrics()
    
    # Handle case with no aggregation yet
    if not latest:
        return pd.DataFrame(
            [['No aggregation data available', '-', '-', '-']],
            columns=['Metric', 'Before', 'After', 'Improvement']
        )
    
    # Extract KNN metrics only
    knn_before = latest['before']['knn_accuracy']
    knn_after = latest['after']['knn_accuracy']
    knn_improvement = latest['improvement']['knn']
    
    # Format data with percentage improvement
    data = [
        ['KNN Accuracy', 
         f"{knn_before:.2%}",
         f"{knn_after:.2%}",
         f"+{knn_improvement:.2%}" if knn_improvement >= 0 else f"{knn_improvement:.2%}"]
    ]
    
    return pd.DataFrame(data, columns=['Metric', 'Before', 'After', 'Improvement'])


def get_aggregation_progress() -> str:
    """
    Get current aggregation progress indicator.
    
    Displays the number of clients that have uploaded weights in the current
    round versus the threshold required to trigger auto-aggregation. Shows
    a checkmark when the threshold is reached. Also displays auto-aggregation
    configuration status.
    
    Returns:
        Formatted string with aggregation progress status and configuration
    """
    state = get_auto_aggregation_state()
    pending = state['pending_uploads']
    threshold = state['threshold']
    enabled = state['enabled']
    
    # Build status message
    if enabled:
        status_line = f"**Auto-aggregation:** Enabled (threshold: {threshold})\n\n"
    else:
        status_line = f"**Auto-aggregation:** Disabled\n\n"
    
    # Add progress indicator
    if pending >= threshold:
        progress_line = f"✓ Ready for aggregation ({pending}/{threshold} clients)"
    else:
        progress_line = f"⏳ Waiting for uploads ({pending}/{threshold} clients)"
    
    return status_line + progress_line
