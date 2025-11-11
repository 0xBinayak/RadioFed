"""
Federated Learning Client - Gradio Interface

This module provides a web-based user interface for the federated learning client.
It allows users to:
- Configure client settings (client_id, server_url)
- Load and inspect RadioML datasets
- Extract features from I/Q samples
- Train local models
- Upload weights to central server
- Download global model from server
"""

import gradio as gr
import os
import logging
from datetime import datetime
from typing import Optional, Tuple

from client.dataset_loader import load_radioml_dataset, get_dataset_info, split_dataset, flatten_dataset
from client.feature_extract import process_dataset, normalize_features
from client.train import train_knn_model, save_knn_model
from client.sync import upload_weights, download_global_model, check_server_status
from client.state import load_config, save_config, save_metrics, load_metrics



logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)



class ClientState:
    def __init__(self):
        self.dataset = None
        self.dataset_info = None
        self.features = None
        self.labels = None
        self.trained_model = None
        self.training_results = None
        self.config = None
        
    def reset_dataset(self):
        """Reset dataset-related state"""
        self.dataset = None
        self.dataset_info = None
        self.features = None
        self.labels = None
        
    def reset_training(self):
        """Reset training-related state"""
        self.trained_model = None
        self.training_results = None


state = ClientState()


def generate_random_client_id() -> str:
    """Generate a random client ID"""
    import random
    import string
    random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"client_{random_suffix}"


def load_initial_config(auto_generate_id: bool = False) -> dict:
    """
    Load configuration from file or use defaults
    
    Args:
        auto_generate_id: If True, generate a random client ID
    """
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
        
       
        if auto_generate_id:
            config['client_id'] = generate_random_client_id()
            logger.info(f"Auto-generated client ID: {config['client_id']}")
        
        return config
    except Exception as e:
        logger.warning(f"Failed to load config: {e}. Using defaults.")
        
        
        client_id = generate_random_client_id() if auto_generate_id else "client_001"
        
        return {
            "client_id": client_id,
            "server_url": "http://localhost:8000",
            "dataset_path": "./data/RML2016.10a_dict.pkl",
            "local_model_path": "./client/local/local_model.pth",
            "training": {
                "epochs": 10,
                "batch_size": 32,
                "learning_rate": 0.001
            }
        }


def load_partition_handler(partition_id: int) -> Tuple[str, str]:
    """
    Load pre-partitioned dataset from data/partitions/ directory
    
    Args:
        partition_id: Which partition to load (0, 1, 2, ...)
        
    Returns:
        Tuple of (status_message, dataset_info_text)
    """
    try:
        
        partition_path = f"./data/partitions/client_{partition_id}.pkl"
        
        if not os.path.exists(partition_path):
            return f"[ERROR] Partition file not found: {partition_path}\nPlease run data/partition_dataset.py first", ""
        
        logger.info(f"Loading partition from: {partition_path}")
        
        state.dataset = load_radioml_dataset(partition_path)
        state.dataset_info = get_dataset_info(state.dataset)
        
        
        if state.config:
            state.config['partition_id'] = partition_id
            save_config(state.config)
        
        
        info_text = f"""**Dataset Information (Partition {partition_id}):**
- **Modulations:** {', '.join(state.dataset_info['modulations'])}
- **SNR Range:** {min(state.dataset_info['snrs'])} to {max(state.dataset_info['snrs'])} dB
- **Total Samples:** {state.dataset_info['sample_count']:,}
- **Sample Shape:** {state.dataset_info['shape']}

**Samples per Modulation:**
"""
        for mod, count in state.dataset_info['samples_per_mod'].items():
            info_text += f"- {mod}: {count:,}\n"
        
        info_text += f"\n**Partition Info:**\n"
        info_text += f"- Loaded from: {partition_path}\n"
        info_text += f"- Partition ID: {partition_id}\n"
        
        status_msg = f"[SUCCESS] Partition {partition_id} loaded: {state.dataset_info['sample_count']:,} samples"
        logger.info(status_msg)
        
        return status_msg, info_text
        
    except Exception as e:
        error_msg = f"[ERROR] Error loading partition: {str(e)}"
        logger.error(error_msg)
        state.reset_dataset()
        return error_msg, ""


def extract_features_handler(progress=gr.Progress()) -> str:
    """
    Extract features from loaded dataset
    
    Returns:
        Status message
    """
    try:
        if state.dataset is None:
            return "Error: Please load a dataset first"
        
        logger.info("Starting feature extraction...")
        progress(0, desc="Preparing dataset...")
        
        
        samples, labels = flatten_dataset(state.dataset)
        logger.info(f"Flattened dataset: {samples.shape[0]} samples")
        
        progress(0.1, desc="Extracting features...")
        
        
        import numpy as np
        from client.feature_extract import extract_features_from_iq
        
        n_samples = samples.shape[0]
        features_list = []
        
        for i in range(n_samples):
            try:
                features = extract_features_from_iq(samples[i])
                features_list.append(features)
            except Exception as e:
                logger.warning(f"Failed to extract features for sample {i}: {str(e)}")
                features_list.append(np.zeros(16, dtype=np.float32))
            
            
            if (i + 1) % 1000 == 0:
                progress_pct = 0.1 + (0.7 * (i + 1) / n_samples)
                progress(progress_pct, desc=f"Extracting features... {i+1}/{n_samples}")
        
        features = np.array(features_list, dtype=np.float32)
        
        progress(0.85, desc="Normalizing features...")
        
        
        normalized_features, mean, std = normalize_features(features)
        
        
        state.features = normalized_features
        state.labels = labels
        
        progress(1.0, desc="Complete!")
        
        status_msg = f"Feature extraction complete: {features.shape[0]:,} samples, {features.shape[1]} features"
        logger.info(status_msg)
        
        return status_msg
        
    except Exception as e:
        error_msg = f" Error extracting features: {str(e)}"
        logger.error(error_msg)
        return error_msg


def train_model_handler(
    model_type: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    client_id: str,
    server_url: str,
    auto_upload: bool,
    progress=gr.Progress()
) -> Tuple[str, str]:
    """
    Train local model on extracted features and optionally auto-upload
    
    Args:
        model_type: Type of model to train ("KNN", "Decision Tree", or "Neural Network")
        epochs: Number of training epochs (for Neural Network)
        batch_size: Batch size for training (for Neural Network)
        learning_rate: Learning rate for optimizer (for Neural Network)
        client_id: Client identifier for auto-upload
        server_url: Server URL for auto-upload
        auto_upload: Whether to automatically upload weights after training
        
    Returns:
        Tuple of (metrics_text, sync_status)
    """
    try:
        if state.features is None or state.labels is None:
            return "  Error: Please extract features first", ""
        
        
        update_training_status_on_server(client_id, server_url, training=True)
        
        progress(0, desc="Initializing training...")
        
        
        if model_type == "KNN":
            
            logger.info(f"Starting KNN training")
            progress(0.2, desc=f"Training KNN model...")
            
            model_type_code = 'knn'
            results = train_knn_model(
                features=state.features,
                labels=state.labels,
                test_split=0.3,  # Matching notebook's 30% test split
                n_neighbors=5,
                random_state=42,
                verbose=True
            )
            
            
            state.trained_model = results['model']
            state.training_results = results
            
            progress(0.8, desc="Saving model...")
            
            
            if state.config:
                model_path = state.config.get('local_model_path', './client/local/local_model.pth')
                
                model_path = model_path.replace('.pth', f'_{model_type_code}.pkl')
                save_knn_model(state.trained_model, model_path)
                
                
                features_path = model_path.replace(f'_{model_type_code}.pkl', f'_{model_type_code}_features.pkl')
                labels_path = model_path.replace(f'_{model_type_code}.pkl', f'_{model_type_code}_labels.pkl')
                
                import pickle
                import os
                os.makedirs(os.path.dirname(features_path), exist_ok=True)
                with open(features_path, 'wb') as f:
                    pickle.dump(state.features, f)
                with open(labels_path, 'wb') as f:
                    pickle.dump(state.labels, f)
                
                
                results['model_path'] = model_path
                results['features_path'] = features_path
                results['labels_path'] = labels_path
                results['model_type'] = model_type_code
                
                
                metrics = {
                    'timestamp': datetime.now().isoformat(),
                    'model_type': model_type_code,
                    'train_accuracy': float(results['train_accuracy']),
                    'test_accuracy': float(results['test_accuracy']),
                    'training_time': float(results['training_time']),
                    'inference_time_ms_per_sample': float(results['inference_time_ms_per_sample']),
                    'n_samples': int(results['n_samples'])
                }
                save_metrics(metrics)
            
            progress(0.9, desc="Training complete!")
            
            
            metrics_text = f"""**✅ Training Complete!**

**Model:** {model_type}

**Results:**
- **Test Accuracy:** {results['test_accuracy']*100:.2f}%
- **Train Accuracy:** {results['train_accuracy']*100:.2f}%
- **Training Samples:** {results['n_samples']:,}

**Computation Complexity:**
- **Training Time:** {results['training_time']:.3f} seconds
- **Inference Time:** {results['inference_time_ms_per_sample']:.3f} ms/sample
"""
            
            logger.info(f"Training complete! Test Accuracy: {results['test_accuracy']*100:.2f}%")
        else:
            # Only KNN is supported
            error_msg = f"❌ Model type '{model_type}' is not supported. Only KNN is available."
            logger.error(error_msg)
            return error_msg, ""
        
        
        update_training_status_on_server(client_id, server_url, training=False)
        
        
        sync_status = ""
        if auto_upload:
            progress(0.95, desc="Auto-uploading weights...")
            sync_status = auto_upload_and_aggregate(client_id, server_url)
        
        progress(1.0, desc="Complete!")
        
        return metrics_text, sync_status
        
    except Exception as e:
        error_msg = f"  Error during training: {str(e)}"
        logger.error(error_msg)
        import traceback
        logger.error(traceback.format_exc())
        return error_msg, ""


def auto_upload_and_aggregate(client_id: str, server_url: str) -> str:
    """
    Automatically upload weights to server
    
    Args:
        client_id: Client identifier
        server_url: Central server URL
        
    Returns:
        Status message
    """
    try:
        if state.trained_model is None:
            return " Error: Please train a model first"
        
        if not client_id or not client_id.strip():
            return " Error: Client ID cannot be empty"
        
        if not server_url or not server_url.strip():
            return " Error: Server URL cannot be empty"
        
        logger.info(f"Auto-uploading weights: client_id={client_id}")
        
        
        try:
            status = check_server_status(server_url, timeout=5)
            logger.info(f"Server is online: {status.get('server_status', 'unknown')}")
        except Exception as e:
            return f" Error: Cannot connect to server at {server_url}. Is it running?"
        
        
        n_samples = state.training_results.get('n_samples', len(state.features))
        model_type = state.training_results.get('model_type', 'knn')
        
        # KNN model upload
        model_path = state.training_results.get('model_path')
        features_path = state.training_results.get('features_path')
        labels_path = state.training_results.get('labels_path')
        
        if not model_path or not os.path.exists(model_path):
            return f" Error: Model file not found"
        
        if not features_path or not os.path.exists(features_path):
            return f" Error: Features file not found"
        
        if not labels_path or not os.path.exists(labels_path):
            return f" Error: Labels file not found"
        
        try:
            import requests
            
            # Open all required files for KNN upload
            files = {
                'model_file': open(model_path, 'rb'),
                'features_file': open(features_path, 'rb'),
                'labels_file': open(labels_path, 'rb')
            }
            
            # Use the new /upload_model endpoint
            url = f"{server_url.rstrip('/')}/upload_model/{client_id}"
            params = {
                'n_samples': n_samples
            }
            
            response = requests.post(url, params=params, files=files, timeout=30)
            
            # Close all files
            for f in files.values():
                f.close()
            
            if response.status_code == 200:
                data = response.json()
                upload_status = data.get('upload_status', {})
                pending = upload_status.get('pending_uploads', 0)
                threshold = upload_status.get('threshold', 2)
                
                status_msg = f"✅ KNN model uploaded successfully!\n"
                status_msg += f"📊 Upload progress: {pending}/{threshold} clients\n"
                
                if upload_status.get('ready_for_aggregation', False):
                    status_msg += f"🚀 Auto-aggregation will trigger automatically!"
                else:
                    status_msg += f"⏳ Waiting for more clients..."
                
                return status_msg
            else:
                return f" Error: Upload failed with status {response.status_code}: {response.text}"
                
        except Exception as e:
            logger.error(f"Error uploading KNN model: {e}")
            return f" Error: Failed to upload model: {str(e)}"
        
    except Exception as e:
        error_msg = f" Error in auto-upload: {str(e)}"
        logger.error(error_msg)
        return error_msg


def upload_weights_handler(client_id: str, server_url: str) -> str:
    """
    Upload local model weights to central server
    
    Args:
        client_id: Client identifier
        server_url: Central server URL
        
    Returns:
        Status message
    """
    try:
        if state.trained_model is None:
            return " Error: Please train a model first"
        
        if not client_id or not client_id.strip():
            return " Error: Client ID cannot be empty"
        
        if not server_url or not server_url.strip():
            return " Error: Server URL cannot be empty"
        
        logger.info(f"Checking server connectivity: {server_url}")
        
        
        try:
            status = check_server_status(server_url, timeout=5)
            logger.info(f"Server is online: {status.get('server_status', 'unknown')}")
        except Exception as e:
            return f" Error: Cannot connect to server at {server_url}. Is it running?"
        
        
        model_path = state.config.get('local_model_path', './client/local/local_model.pth')
        n_samples = state.training_results.get('n_samples', len(state.features))
        
        if not os.path.exists(model_path):
            return f" Error: Model file not found at {model_path}"
        
        logger.info(f"Uploading weights: client_id={client_id}, n_samples={n_samples}")
        
        success = upload_weights(
            server_url=server_url,
            client_id=client_id,
            weights_path=model_path,
            n_samples=n_samples,
            max_retries=3,
            timeout=30
        )
        
        if success:
            status_msg = f" Weights uploaded successfully to {server_url}"
            logger.info(status_msg)
            return status_msg
        else:
            return " Error: Failed to upload weights. Check logs for details."
        
    except Exception as e:
        error_msg = f" Error uploading weights: {str(e)}"
        logger.error(error_msg)
        return error_msg


def download_global_model_handler(server_url: str) -> str:
    """
    Download global model from central server
    
    Args:
        server_url: Central server URL
        
    Returns:
        Status message
    """
    try:
        if not server_url or not server_url.strip():
            return " Error: Server URL cannot be empty"
        
        logger.info(f"Checking server connectivity: {server_url}")
        
        
        try:
            status = check_server_status(server_url, timeout=5)
            logger.info(f"Server is online: {status.get('server_status', 'unknown')}")
        except Exception as e:
            return f" Error: Cannot connect to server at {server_url}. Is it running?"
        
        
        save_path = state.config.get('local_model_path', './client/local/local_model.pth')
        
        global_model_path = save_path.replace('local_model.pth', 'global_model.pth')
        
        logger.info(f"Downloading global model to: {global_model_path}")
        
        success = download_global_model(
            server_url=server_url,
            save_path=global_model_path,
            max_retries=3,
            timeout=30
        )
        
        if success:
            status_msg = f" Global model downloaded successfully to {global_model_path}"
            logger.info(status_msg)
            return status_msg
        else:
            return " Error: Failed to download global model. Check logs for details."
        
    except Exception as e:
        error_msg = f" Error downloading global model: {str(e)}"
        logger.error(error_msg)
        return error_msg


def update_config_handler(client_id: str, server_url: str) -> str:
    """
    Update configuration with new values
    
    Args:
        client_id: Client identifier
        server_url: Central server URL
        
    Returns:
        Status message
    """
    try:
        if state.config is None:
            state.config = load_initial_config()
        
        state.config['client_id'] = client_id
        state.config['server_url'] = server_url
        
        save_config(state.config)
        
        status_msg = " Configuration updated successfully"
        logger.info(status_msg)
        return status_msg
        
    except Exception as e:
        error_msg = f" Error updating configuration: {str(e)}"
        logger.error(error_msg)
        return error_msg


def register_with_server(client_id: str, server_url: str) -> bool:
    """Register client with central server"""
    try:
        import requests
        response = requests.post(
            f"{server_url.rstrip('/')}/register/{client_id}",
            timeout=5
        )
        if response.status_code == 200:
            logger.info(f" Registered with server as {client_id}")
            return True
        else:
            logger.warning(f"Failed to register: {response.text}")
            return False
    except Exception as e:
        logger.warning(f"Could not register with server: {e}")
        return False


def update_training_status_on_server(client_id: str, server_url: str, training: bool) -> None:
    """Update training status on server"""
    try:
        import requests
        requests.post(
            f"{server_url.rstrip('/')}/training_status/{client_id}",
            params={"training": training},
            timeout=5
        )
    except Exception as e:
        logger.debug(f"Could not update training status: {e}")


def create_gradio_interface(auto_generate_id: bool = False):
    """
    Create and configure the Gradio interface
    
    Args:
        auto_generate_id: If True, automatically generate a random client ID
    """
    
    
    state.config = load_initial_config(auto_generate_id=auto_generate_id)
    
    
    client_id = state.config.get('client_id', 'unknown')
    server_url = state.config.get('server_url', 'http://localhost:8000')
    register_with_server(client_id, server_url)
    
    
    with gr.Blocks(title="Federated Learning Client", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Federated Learning Client")
        
       
        with gr.Accordion("Configuration", open=True):
            with gr.Row():
                client_id_input = gr.Textbox(
                    label="Client ID",
                    value=state.config.get('client_id', 'client_001')
                )
                server_url_input = gr.Textbox(
                    label="Server URL",
                    value=state.config.get('server_url', 'http://localhost:8000')
                )
            config_status = gr.Textbox(label="Status", interactive=False, visible=False)
        
        # Dataset Section
        with gr.Accordion("Dataset", open=True):
            partition_id_input = gr.Number(
                label="Partition ID",
                value=state.config.get('partition_id', 0),
                precision=0,
                minimum=0
            )
            load_dataset_btn = gr.Button("Load Partition", variant="primary")
            dataset_status = gr.Textbox(label="Status", interactive=False)
            dataset_info = gr.Markdown("")
        
        # Feature Extraction Section
        with gr.Accordion("Feature Extraction", open=True):
            extract_features_btn = gr.Button("Extract Features", variant="primary")
            feature_status = gr.Textbox(label="Status", interactive=False)
        
        # Training Section
        with gr.Accordion("Training", open=True):
            model_type_radio = gr.Radio(
                choices=["KNN"],
                value="KNN",
                label="Model Type (KNN Only)"
            )
            
            with gr.Row():
                epochs_input = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=state.config.get('training', {}).get('epochs', 10),
                    step=1,
                    label="Epochs",
                    visible=False
                )
                batch_size_input = gr.Slider(
                    minimum=8,
                    maximum=256,
                    value=state.config.get('training', {}).get('batch_size', 32),
                    step=8,
                    label="Batch Size",
                    visible=False
                )
                learning_rate_input = gr.Number(
                    value=state.config.get('training', {}).get('learning_rate', 0.001),
                    label="Learning Rate",
                    visible=False
                )
            
            auto_upload_checkbox = gr.Checkbox(
                label="Auto-upload after training",
                value=True
            )
            
            train_btn = gr.Button("Train Model", variant="primary")
            training_metrics = gr.Markdown("")
        
        
        with gr.Accordion("Synchronization", open=True):
            with gr.Row():
                upload_btn = gr.Button("Upload Weights", variant="primary")
                download_btn = gr.Button("Download Global Model", variant="primary")
            sync_status = gr.Textbox(label="Status", interactive=False)
        
        
        client_id_input.change(
            fn=update_config_handler,
            inputs=[client_id_input, server_url_input],
            outputs=[config_status]
        )
        server_url_input.change(
            fn=update_config_handler,
            inputs=[client_id_input, server_url_input],
            outputs=[config_status]
        )
        
        load_dataset_btn.click(
            fn=load_partition_handler,
            inputs=[partition_id_input],
            outputs=[dataset_status, dataset_info]
        )
        
        extract_features_btn.click(
            fn=extract_features_handler,
            outputs=[feature_status]
        )
        
        
        def update_training_params_visibility(model_type):
            """Show/hide training parameters based on model type"""
            is_nn = model_type == "Neural Network"
            return [
                gr.update(visible=is_nn),  
                gr.update(visible=is_nn), 
                gr.update(visible=is_nn)  
            ]
        
        model_type_radio.change(
            fn=update_training_params_visibility,
            inputs=[model_type_radio],
            outputs=[epochs_input, batch_size_input, learning_rate_input]
        )
        
        train_btn.click(
            fn=train_model_handler,
            inputs=[model_type_radio, epochs_input, batch_size_input, learning_rate_input, client_id_input, server_url_input, auto_upload_checkbox],
            outputs=[training_metrics, sync_status]
        )
        
        upload_btn.click(
            fn=upload_weights_handler,
            inputs=[client_id_input, server_url_input],
            outputs=[sync_status]
        )
        
        download_btn.click(
            fn=download_global_model_handler,
            inputs=[server_url_input],
            outputs=[sync_status]
        )
    
    return demo


def launch_gradio(share: bool = False, server_port: int = 7861, client_id: Optional[str] = None, auto_generate_id: bool = False):
    """
    Launch the Gradio interface
    
    Args:
        share: Whether to create a public link
        server_port: Port to run the Gradio server on
        client_id: Optional client ID to pre-populate in the UI
        auto_generate_id: If True, automatically generate a random client ID
    """
    logger.info("Starting Federated Learning Client UI...")
    logger.info(f"Access the interface at: http://localhost:{server_port}")
    
   
    if client_id:
        
        if state.config is None:
            state.config = load_initial_config()
        state.config['client_id'] = client_id
        logger.info(f"Using client ID: {client_id}")
        demo = create_gradio_interface(auto_generate_id=False)
    elif auto_generate_id:
       
        demo = create_gradio_interface(auto_generate_id=True)
        logger.info(f"Auto-generated client ID: {state.config['client_id']}")
    else:
        
        demo = create_gradio_interface(auto_generate_id=False)
    
    demo.launch(
        share=share,
        server_port=server_port,
        server_name="127.0.0.1"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client")
    parser.add_argument(
        "--port",
        type=int,
        default=7861,
        help="Port to run the Gradio interface on (default: 7861)"
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Client ID for this instance (default: from config.json)"
    )
    parser.add_argument(
        "--auto-id",
        action="store_true",
        help="Automatically generate a random client ID (e.g., client_a3f9k2)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio link"
    )
    
    args = parser.parse_args()
    
    launch_gradio(
        share=args.share,
        server_port=args.port,
        client_id=args.client_id,
        auto_generate_id=args.auto_id
    )
