"""
FastAPI Server for Federated Learning Central Server

This module provides REST API endpoints for client communication including
weight upload, aggregation triggering, global model download, and status queries.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import os
import logging
import threading
from typing import Optional

from central.state import (
    load_config,
    register_client_connection,
    update_client_training_status,
    register_client_upload,
    get_client_status,
    get_all_client_weights,
    get_registry_stats,
    store_aggregation_result,
    get_latest_aggregation_result,
    track_client_upload,
    get_pending_uploads_count,
    get_auto_aggregation_threshold,
    should_trigger_aggregation,
    initialize_auto_aggregation_state,
    initialize_metrics_history
)
from central.aggregator import (
    aggregate_from_registry,
    aggregate_knn_models,
    aggregate_dt_models,
    DecisionTreeEnsemble,
    save_knn_model,
    save_dt_ensemble,
    evaluate_global_model
)
from central.model import FederatedModel
from central.utils import setup_logging, ensure_directories


# Initialize FastAPI app
app = FastAPI(
    title="Federated Learning Central Server",
    description="REST API for federated learning weight aggregation and model distribution",
    version="1.0.0"
)

# Add CORS middleware for Gradio UI communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Gradio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
logger = None
config = None
last_aggregation_time = None

# Aggregation status tracking
_aggregation_in_progress = False
_aggregation_lock = threading.Lock()


def _initialize_server():
    """Initialize server configuration and logging."""
    global logger, config
    
    if logger is not None and config is not None:
        return  # Already initialized
    
    # Ensure directories exist
    ensure_directories()
    
    # Load configuration
    try:
        config = load_config()
        log_level = config.get("log_level", "INFO")
    except Exception as e:
        print(f"Warning: Could not load config, using defaults: {e}")
        config = {
            "model_save_path": "./central/model_store/global_model.pth",
            "host": "0.0.0.0",
            "port": 8000,
            "log_level": "INFO",
            "auto_aggregation_enabled": True,
            "auto_aggregation_threshold": 2
        }
        log_level = "INFO"
    
    # Setup logging
    logger = setup_logging(log_level)
    logger.info("Central server starting up")
    logger.info(f"Configuration: {config}")
    
    # Initialize auto-aggregation state from configuration
    try:
        initialize_auto_aggregation_state()
        logger.info("Auto-aggregation state initialized from configuration")
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise
    except Exception as e:
        logger.warning(f"Could not initialize auto-aggregation state: {e}")
    
    # Initialize metrics history
    try:
        initialize_metrics_history()
        logger.info("Metrics history initialized")
    except Exception as e:
        logger.warning(f"Could not initialize metrics history: {e}")


@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    _initialize_server()


def perform_auto_aggregation():
    """
    Execute complete auto-aggregation workflow in background thread.
    
    This function is called asynchronously when auto-aggregation is triggered.
    It performs the following steps:
    1. Capture before-aggregation metrics
    2. Perform aggregation for all model types
    3. Evaluate global model after aggregation
    4. Store round in metrics history
    5. Reset aggregation state
    6. Broadcast global model to clients (models are saved and available for download)
    
    Requirements: 1.2, 1.3, 5.1, 5.2
    """
    global _aggregation_in_progress, last_aggregation_time
    
    try:
        logger.info("Starting auto-aggregation workflow...")
        
        # Step 1: Capture before-aggregation metrics
        logger.info("Capturing before-aggregation metrics...")
        try:
            from central.state import capture_current_metrics
            before_metrics = capture_current_metrics()
            logger.info(f"Before-aggregation metrics captured: KNN={before_metrics.get('knn_accuracy', 0):.4f}, DT={before_metrics.get('dt_accuracy', 0):.4f}")
        except Exception as e:
            logger.warning(f"Could not capture before-aggregation metrics: {e}")
            before_metrics = {
                'knn_accuracy': 0.0,
                'dt_accuracy': 0.0,
                'per_snr_accuracy': {},
                'confusion_matrix': [],
                'num_clients': 0,
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 2: Get all client weights and perform aggregation
        client_weights_info = get_all_client_weights()
        
        if not client_weights_info:
            logger.warning("No client models available for auto-aggregation")
            return
        
        # Group clients by model type
        model_types = {}
        for client in client_weights_info:
            model_type = client.get('model_type', 'neural')
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(client)
        
        # Perform aggregation for each model type
        for model_type, clients in model_types.items():
            try:
                logger.info(f"Aggregating {model_type} models from {len(clients)} clients")
                
                if model_type == 'neural':
                    # Neural network FedAvg aggregation
                    global_model_path = config.get("model_save_path", "./central/model_store/global_model.pth")
                    reference_model = FederatedModel()
                    
                    result = aggregate_from_registry(
                        clients,
                        global_model_path,
                        reference_model
                    )
                    
                    logger.info(f"Neural network aggregation completed: {result['num_clients']} clients, {result['total_samples']} samples")
                
                elif model_type == 'knn':
                    # KNN aggregation
                    result = aggregate_knn_models(clients, n_neighbors=5, evaluate=True)
                    
                    # Save global KNN model
                    global_knn_path = "./central/model_store/global_knn_model.pkl"
                    save_knn_model(result['global_model'], global_knn_path)
                    
                    timestamp = datetime.now().isoformat()
                    store_aggregation_result(model_type, result, timestamp)
                    
                    logger.info(f"KNN aggregation completed: {result['num_clients']} clients, accuracy={result.get('accuracy', 0.0):.4f}")
                
                elif model_type == 'dt':
                    # Decision Tree aggregation
                    result = aggregate_dt_models(clients, evaluate=True)
                    
                    # Create and save ensemble
                    ensemble = DecisionTreeEnsemble(
                        models=result['client_models'],
                        weights=result['ensemble_weights']
                    )
                    global_dt_path = "./central/model_store/global_dt_ensemble.pkl"
                    save_dt_ensemble(ensemble, global_dt_path)
                    
                    result['global_model'] = ensemble
                    timestamp = datetime.now().isoformat()
                    store_aggregation_result(model_type, result, timestamp)
                    
                    logger.info(f"Decision Tree aggregation completed: {result['num_clients']} clients, accuracy={result.get('accuracy', 0.0):.4f}")
            
            except Exception as e:
                logger.error(f"Error aggregating {model_type} models: {e}")
                continue
        
        # Step 3: Evaluate global model after aggregation
        logger.info("Evaluating global model after aggregation...")
        try:
            from central.state import evaluate_global_model as eval_global
            after_metrics = eval_global()
            logger.info(f"After-aggregation metrics captured: KNN={after_metrics.get('knn_accuracy', 0):.4f}, DT={after_metrics.get('dt_accuracy', 0):.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate global model: {e}")
            after_metrics = {
                'knn_accuracy': 0.0,
                'dt_accuracy': 0.0,
                'per_snr_accuracy': {},
                'confusion_matrix': [],
                'timestamp': datetime.now().isoformat()
            }
        
        # Step 4: Store round in metrics history
        logger.info("Storing aggregation round in metrics history...")
        try:
            from central.state import store_aggregation_round
            store_aggregation_round(before_metrics, after_metrics)
            logger.info("Metrics history updated successfully")
        except Exception as e:
            logger.error(f"Failed to store metrics history: {e}")
        
        # Step 5: Reset aggregation state
        logger.info("Resetting aggregation state...")
        try:
            from central.state import reset_aggregation_state, get_auto_aggregation_state
            reset_aggregation_state()
            state = get_auto_aggregation_state()
            logger.info(f"Aggregation state reset. New round: {state['current_round']}")
        except Exception as e:
            logger.error(f"Failed to reset aggregation state: {e}")
        
        # Update last aggregation time
        last_aggregation_time = datetime.now().isoformat()
        
        # Step 6: Global models are already saved and available for client download
        # Clients can download via /global_model endpoint
        logger.info("Auto-aggregation workflow completed successfully. Global models available for download.")
    
    except Exception as e:
        logger.error(f"Auto-aggregation workflow failed: {e}", exc_info=True)
        # Note: State is preserved on failure for retry (as per requirement 1.5, 10.3)
        # The pending_uploads counter and clients_uploaded_this_round list are NOT reset
        # This allows the system to retry aggregation when conditions are met again
        
        # Clients are notified of aggregation status implicitly:
        # - On success: Global models are available via /global_model endpoint
        # - On failure: Previous global models remain available, clients can check /status
        # The /status endpoint shows last_aggregation_time which clients can monitor
    
    finally:
        # Reset aggregation in progress flag to allow future aggregation attempts
        with _aggregation_lock:
            global _aggregation_in_progress
            _aggregation_in_progress = False


def trigger_aggregation_async():
    """
    Trigger aggregation in a background thread.
    
    Prevents concurrent aggregations by checking if one is already in progress.
    Launches aggregation in a separate thread to avoid blocking client uploads.
    """
    global _aggregation_in_progress
    
    with _aggregation_lock:
        if _aggregation_in_progress:
            logger.info("Aggregation already in progress, skipping trigger")
            return
        
        # Set flag to prevent concurrent aggregations
        _aggregation_in_progress = True
    
    # Launch aggregation in background thread
    thread = threading.Thread(target=perform_auto_aggregation, daemon=True)
    thread.start()
    
    logger.info("Auto-aggregation triggered in background thread")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Federated Learning Central Server",
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /register/{client_id}",
            "training_status": "POST /training_status/{client_id}?training={bool}",
            "upload_weights": "POST /upload_weights/{client_id}?n_samples={count} (for neural networks)",
            "upload_traditional_model": "POST /upload_traditional_model/{client_id}?n_samples={count}&model_type={knn|dt}",
            "aggregate": "POST /aggregate?model_type={neural|knn|dt}",
            "global_model": "GET /global_model?model_type={neural|knn|dt}",
            "aggregation_results": "GET /aggregation_results/{model_type}",
            "status": "GET /status"
        }
    }


@app.post("/register/{client_id}")
async def register_client(client_id: str):
    """
    Register a client connection with the server.
    
    Args:
        client_id: Unique identifier for the client
    
    Returns:
        JSON response with registration status
    """
    _initialize_server()
    
    try:
        if not client_id or len(client_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid client_id")
        
        register_client_connection(client_id)
        logger.info(f"Client registered: {client_id}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "client_id": client_id,
                "message": f"Client {client_id} registered successfully",
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error registering client {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/training_status/{client_id}")
async def update_training_status(
    client_id: str,
    training: bool = Query(..., description="Whether client is training")
):
    """
    Update client training status.
    
    Args:
        client_id: Unique identifier for the client
        training: Whether client is currently training
    
    Returns:
        JSON response with status update
    """
    _initialize_server()
    
    try:
        if not client_id or len(client_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid client_id")
        
        update_client_training_status(client_id, training)
        status_text = "training" if training else "idle"
        logger.info(f"Client {client_id} status updated: {status_text}")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "client_id": client_id,
                "training": training,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Error updating training status for {client_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_weights/{client_id}")
async def upload_weights(
    client_id: str,
    n_samples: int = Query(..., description="Number of samples used for training"),
    file: UploadFile = File(..., description="Model weights file (.pth format)")
):
    """
    Upload client model weights to the central server (for neural networks).
    
    Args:
        client_id: Unique identifier for the client
        n_samples: Number of samples used for training (query parameter)
        file: Uploaded weights file in PyTorch .pth format
    
    Returns:
        JSON response with upload status and timestamp
    """
    _initialize_server()
    
    try:
        # Validate client_id
        if not client_id or len(client_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid client_id")
        
        # Validate n_samples
        if n_samples <= 0:
            raise HTTPException(status_code=400, detail="n_samples must be positive")
        
        # Validate file
        if not file.filename.endswith('.pth'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Expected .pth file"
            )
        
        # Create storage path for client weights
        weights_dir = "./central/model_store"
        os.makedirs(weights_dir, exist_ok=True)
        weights_path = os.path.join(weights_dir, f"{client_id}_weights.pth")
        
        # Save uploaded file
        contents = await file.read()
        with open(weights_path, "wb") as f:
            f.write(contents)
        
        logger.info(f"Received weights from {client_id}: {len(contents)} bytes, {n_samples} samples")
        
        # Register client upload in registry
        register_client_upload(client_id, n_samples, weights_path, model_type='neural')
        
        # Track upload for auto-aggregation
        track_client_upload(client_id)
        
        # Get upload tracking info
        pending_uploads = get_pending_uploads_count()
        threshold = get_auto_aggregation_threshold()
        
        logger.info(f"Upload tracked for {client_id}: {pending_uploads}/{threshold} clients uploaded")
        
        # Check if auto-aggregation should be triggered
        if should_trigger_aggregation():
            logger.info(f"Auto-aggregation threshold reached: {pending_uploads}/{threshold} clients")
            trigger_aggregation_async()
        
        # Get stats for response
        stats = get_registry_stats()
        
        timestamp = datetime.now().isoformat()
        
        response_content = {
            "status": "success",
            "client_id": client_id,
            "n_samples": n_samples,
            "timestamp": timestamp,
            "message": f"Weights uploaded successfully from {client_id}",
            "total_clients": stats['total_clients'],
            "total_samples": stats['total_samples'],
            "upload_status": {
                "pending_uploads": pending_uploads,
                "threshold": threshold,
                "ready_for_aggregation": pending_uploads >= threshold
            }
        }
        
        return JSONResponse(
            status_code=200,
            content=response_content
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading weights from {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/upload_traditional_model/{client_id}")
async def upload_traditional_model(
    client_id: str,
    n_samples: int = Query(..., description="Number of samples used for training"),
    model_type: str = Query(..., description="Model type: 'knn' or 'dt'"),
    model_file: UploadFile = File(..., description="Serialized model file (.pkl format)"),
    features_file: UploadFile = File(None, description="Training features file (.pkl format, required for KNN)"),
    labels_file: UploadFile = File(None, description="Training labels file (.pkl format, required for KNN)")
):
    """
    Upload traditional ML model (KNN or Decision Tree) to the central server.
    
    Args:
        client_id: Unique identifier for the client
        n_samples: Number of samples used for training
        model_type: Type of model ('knn' or 'dt')
        model_file: Serialized model file
        features_file: Training features (required for KNN aggregation)
        labels_file: Training labels (required for KNN aggregation)
    
    Returns:
        JSON response with upload status
    """
    _initialize_server()
    
    try:
        # Validate inputs
        if not client_id or len(client_id.strip()) == 0:
            raise HTTPException(status_code=400, detail="Invalid client_id")
        
        if n_samples <= 0:
            raise HTTPException(status_code=400, detail="n_samples must be positive")
        
        if model_type not in ['knn', 'dt']:
            raise HTTPException(status_code=400, detail="model_type must be 'knn' or 'dt'")
        
        # For KNN, features and labels are required
        if model_type == 'knn' and (features_file is None or labels_file is None):
            raise HTTPException(
                status_code=400,
                detail="features_file and labels_file are required for KNN models"
            )
        
        # Create storage directory
        models_dir = "./central/model_store"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model file
        model_path = os.path.join(models_dir, f"{client_id}_{model_type}_model.pkl")
        model_contents = await model_file.read()
        with open(model_path, "wb") as f:
            f.write(model_contents)
        
        logger.info(f"Received {model_type} model from {client_id}: {len(model_contents)} bytes")
        
        # Save features and labels if provided
        features_path = None
        labels_path = None
        
        if features_file is not None:
            features_path = os.path.join(models_dir, f"{client_id}_features.pkl")
            features_contents = await features_file.read()
            with open(features_path, "wb") as f:
                f.write(features_contents)
            logger.info(f"Received features from {client_id}: {len(features_contents)} bytes")
        
        if labels_file is not None:
            labels_path = os.path.join(models_dir, f"{client_id}_labels.pkl")
            labels_contents = await labels_file.read()
            with open(labels_path, "wb") as f:
                f.write(labels_contents)
            logger.info(f"Received labels from {client_id}: {len(labels_contents)} bytes")
        
        # Register client upload
        register_client_upload(
            client_id=client_id,
            n_samples=n_samples,
            weights_path=model_path,  # For compatibility
            model_type=model_type,
            model_path=model_path,
            features_path=features_path,
            labels_path=labels_path
        )
        
        # Track upload for auto-aggregation
        track_client_upload(client_id)
        
        # Get upload tracking info
        pending_uploads = get_pending_uploads_count()
        threshold = get_auto_aggregation_threshold()
        
        logger.info(f"Upload tracked for {client_id}: {pending_uploads}/{threshold} clients uploaded")
        
        # Check if auto-aggregation should be triggered
        if should_trigger_aggregation():
            logger.info(f"Auto-aggregation threshold reached: {pending_uploads}/{threshold} clients")
            trigger_aggregation_async()
        
        # Get stats
        stats = get_registry_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "client_id": client_id,
                "model_type": model_type,
                "n_samples": n_samples,
                "timestamp": datetime.now().isoformat(),
                "message": f"{model_type.upper()} model uploaded successfully from {client_id}",
                "total_clients": stats['total_clients'],
                "total_samples": stats['total_samples'],
                "upload_status": {
                    "pending_uploads": pending_uploads,
                    "threshold": threshold,
                    "ready_for_aggregation": pending_uploads >= threshold
                }
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading traditional model from {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/aggregate")
async def aggregate(
    model_type: str = Query('neural', description="Model type to aggregate: 'neural', 'knn', or 'dt'")
):
    """
    Trigger aggregation of uploaded client models.
    
    Supports different aggregation strategies based on model type:
    - 'neural': FedAvg for neural networks
    - 'knn': Data merging and retraining for KNN
    - 'dt': Ensemble voting for Decision Trees
    
    Args:
        model_type: Type of model to aggregate ('neural', 'knn', or 'dt')
    
    Returns:
        JSON response with aggregation results
    """
    _initialize_server()
    global last_aggregation_time
    
    try:
        # Validate model type
        if model_type not in ['neural', 'knn', 'dt']:
            raise HTTPException(
                status_code=400,
                detail="model_type must be 'neural', 'knn', or 'dt'"
            )
        
        # Get all client info from registry
        client_weights_info = get_all_client_weights()
        
        if not client_weights_info:
            raise HTTPException(
                status_code=400,
                detail="No client models available for aggregation"
            )
        
        # Filter clients by model type
        filtered_clients = [
            client for client in client_weights_info
            if client.get('model_type', 'neural') == model_type
        ]
        
        if not filtered_clients:
            raise HTTPException(
                status_code=400,
                detail=f"No {model_type} models available for aggregation"
            )
        
        logger.info(f"Starting {model_type} aggregation with {len(filtered_clients)} clients")
        
        timestamp = datetime.now().isoformat()
        
        # Route to appropriate aggregation strategy
        if model_type == 'neural':
            # Neural network FedAvg aggregation
            global_model_path = config.get("model_save_path", "./central/model_store/global_model.pth")
            reference_model = FederatedModel()
            
            result = aggregate_from_registry(
                filtered_clients,
                global_model_path,
                reference_model
            )
            
            response_content = {
                "status": "success",
                "model_type": "neural",
                "num_clients": result['num_clients'],
                "total_samples": result['total_samples'],
                "global_model_path": result['global_model_path'],
                "timestamp": timestamp,
                "message": "Neural network aggregation completed successfully"
            }
        
        elif model_type == 'knn':
            # KNN aggregation - merge data and retrain with evaluation
            result = aggregate_knn_models(filtered_clients, n_neighbors=5, evaluate=True)
            
            # Save global KNN model
            global_knn_path = "./central/model_store/global_knn_model.pkl"
            save_knn_model(result['global_model'], global_knn_path)
            
            # Store result with metrics
            store_aggregation_result(model_type, result, timestamp)
            
            response_content = {
                "status": "success",
                "model_type": "knn",
                "num_clients": result['num_clients'],
                "total_samples": result['total_samples'],
                "feature_dim": result['feature_dim'],
                "n_neighbors": result['n_neighbors'],
                "accuracy": result.get('accuracy', 0.0),
                "training_time": result.get('training_time', 0.0),
                "inference_time_ms": result.get('inference_time_ms_per_sample', 0.0),
                "global_model_path": global_knn_path,
                "timestamp": timestamp,
                "message": "KNN aggregation completed successfully"
            }
        
        elif model_type == 'dt':
            # Decision Tree aggregation - create ensemble with evaluation
            result = aggregate_dt_models(filtered_clients, evaluate=True)
            
            # Create and save ensemble
            ensemble = DecisionTreeEnsemble(
                models=result['client_models'],
                weights=result['ensemble_weights']
            )
            global_dt_path = "./central/model_store/global_dt_ensemble.pkl"
            save_dt_ensemble(ensemble, global_dt_path)
            
            # Store result with ensemble
            result['global_model'] = ensemble
            store_aggregation_result(model_type, result, timestamp)
            
            response_content = {
                "status": "success",
                "model_type": "dt",
                "num_clients": result['num_clients'],
                "total_samples": result['total_samples'],
                "ensemble_weights": result['ensemble_weights'],
                "accuracy": result.get('accuracy', 0.0),
                "training_time": result.get('training_time', 0.0),
                "inference_time_ms": result.get('inference_time_ms_per_sample', 0.0),
                "global_model_path": global_dt_path,
                "timestamp": timestamp,
                "message": "Decision Tree aggregation completed successfully"
            }
        
        last_aggregation_time = timestamp
        
        logger.info(f"{model_type.upper()} aggregation completed: {response_content['num_clients']} clients, {response_content['total_samples']} total samples")
        
        return JSONResponse(
            status_code=200,
            content=response_content
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during {model_type} aggregation: {e}")
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}")


@app.get("/global_model")
async def get_global_model(
    model_type: str = Query('neural', description="Model type: 'neural', 'knn', or 'dt'")
):
    """
    Download the current global model.
    
    Args:
        model_type: Type of model to download ('neural', 'knn', or 'dt')
    
    Returns:
        FileResponse with the global model file
    """
    _initialize_server()
    
    try:
        # Validate model type
        if model_type not in ['neural', 'knn', 'dt']:
            raise HTTPException(
                status_code=400,
                detail="model_type must be 'neural', 'knn', or 'dt'"
            )
        
        # Determine model path based on type
        if model_type == 'neural':
            global_model_path = config.get("model_save_path", "./central/model_store/global_model.pth")
            filename = "global_model.pth"
        elif model_type == 'knn':
            global_model_path = "./central/model_store/global_knn_model.pkl"
            filename = "global_knn_model.pkl"
        elif model_type == 'dt':
            global_model_path = "./central/model_store/global_dt_ensemble.pkl"
            filename = "global_dt_ensemble.pkl"
        
        if not os.path.exists(global_model_path):
            raise HTTPException(
                status_code=404,
                detail=f"Global {model_type} model not found. Please run aggregation first."
            )
        
        logger.info(f"Serving global {model_type} model download")
        
        return FileResponse(
            path=global_model_path,
            media_type="application/octet-stream",
            filename=filename
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving global model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to serve global model: {str(e)}")


@app.get("/aggregation_results/{model_type}")
async def get_aggregation_results(model_type: str):
    """
    Get aggregation results for a specific model type.
    
    Args:
        model_type: Type of model ('neural', 'knn', or 'dt')
    
    Returns:
        JSON response with aggregation results
    """
    _initialize_server()
    
    try:
        if model_type not in ['neural', 'knn', 'dt']:
            raise HTTPException(
                status_code=400,
                detail="model_type must be 'neural', 'knn', or 'dt'"
            )
        
        result = get_latest_aggregation_result(model_type)
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"No aggregation results found for {model_type} models"
            )
        
        # Remove non-serializable objects
        response_result = {k: v for k, v in result['result'].items() if k != 'global_model'}
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "model_type": model_type,
                "result": response_result,
                "timestamp": result['timestamp']
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting aggregation results: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get aggregation results: {str(e)}")


@app.get("/status")
async def get_status():
    """
    Get server status and client registry information.
    
    Returns:
        JSON response with server health and connected clients
    """
    _initialize_server()
    
    try:
        # Get client status
        clients = get_client_status()
        
        # Get registry statistics
        stats = get_registry_stats()
        
        # Check if global model exists
        global_model_path = config.get("model_save_path", "./central/model_store/global_model.pth")
        global_model_exists = os.path.exists(global_model_path)
        
        return JSONResponse(
            status_code=200,
            content={
                "server_status": "running",
                "timestamp": datetime.now().isoformat(),
                "last_aggregation": last_aggregation_time,
                "global_model_exists": global_model_exists,
                "global_model_path": global_model_path if global_model_exists else None,
                "total_clients": stats['total_clients'],
                "total_samples": stats['total_samples'],
                "clients": clients
            }
        )
    
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get status: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}
