"""
FastAPI Server for Federated Learning Central Server

This module provides REST API endpoints for client communication including
model upload, aggregation triggering, global model download, and status queries.
Supports KNN models only.
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
    aggregate_knn_models,
    save_knn_model,
    evaluate_global_model
)
from central.utils import setup_logging, ensure_directories


# Initialize FastAPI app
app = FastAPI(
    title="Federated Learning Central Server",
    description="REST API for federated learning KNN model aggregation and distribution",
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
            "model_save_path": "./central/model_store/global_knn_model.pkl",
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
    2. Perform KNN aggregation
    3. Evaluate global model after aggregation
    4. Store round in metrics history
    5. Reset aggregation state
    6. Broadcast global model to clients (models are saved and available for download)
    """
    global _aggregation_in_progress, last_aggregation_time
    
    try:
        logger.info("Starting auto-aggregation workflow...")
        
        # Step 1: Capture before-aggregation metrics
        logger.info("Capturing before-aggregation metrics...")
        try:
            from central.state import capture_current_metrics
            before_metrics = capture_current_metrics()
            logger.info(f"Before-aggregation metrics captured: KNN={before_metrics.get('knn_accuracy', 0):.4f}")
        except Exception as e:
            logger.warning(f"Could not capture before-aggregation metrics: {e}")
            before_metrics = {
                'knn_accuracy': 0.0,
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
        
        # Filter for KNN models only
        knn_clients = [c for c in client_weights_info if c.get('model_type', 'knn') == 'knn']
        
        if not knn_clients:
            logger.warning("No KNN models available for auto-aggregation")
            return
        
        try:
            logger.info(f"Aggregating KNN models from {len(knn_clients)} clients")
            
            # KNN aggregation
            result = aggregate_knn_models(knn_clients, n_neighbors=5, evaluate=True)
            
            # Save global KNN model
            global_knn_path = "./central/model_store/global_knn_model.pkl"
            save_knn_model(result['global_model'], global_knn_path)
            
            timestamp = datetime.now().isoformat()
            store_aggregation_result('knn', result, timestamp)
            
            logger.info(f"KNN aggregation completed: {result['num_clients']} clients, accuracy={result.get('accuracy', 0.0):.4f}")
        
        except Exception as e:
            logger.error(f"Error aggregating KNN models: {e}")
            return
        
        # Step 3: Evaluate global model after aggregation
        logger.info("Evaluating global model after aggregation...")
        try:
            from central.state import evaluate_global_model as eval_global
            after_metrics = eval_global()
            logger.info(f"After-aggregation metrics captured: KNN={after_metrics.get('knn_accuracy', 0):.4f}")
        except Exception as e:
            logger.warning(f"Could not evaluate global model: {e}")
            after_metrics = {
                'knn_accuracy': 0.0,
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
        logger.info("Auto-aggregation workflow completed successfully. Global models available for download.")
    
    except Exception as e:
        logger.error(f"Auto-aggregation workflow failed: {e}", exc_info=True)
    
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
        "message": "Federated Learning Central Server - KNN Only",
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /register/{client_id}",
            "training_status": "POST /training_status/{client_id}?training={bool}",
            "upload_model": "POST /upload_model/{client_id}?n_samples={count}",
            "aggregate": "POST /aggregate",
            "global_model": "GET /global_model",
            "aggregation_results": "GET /aggregation_results",
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


@app.post("/upload_model/{client_id}")
async def upload_model(
    client_id: str,
    n_samples: int = Query(..., description="Number of samples used for training"),
    model_file: UploadFile = File(..., description="Serialized KNN model file (.pkl format)"),
    features_file: UploadFile = File(..., description="Training features file (.pkl format)"),
    labels_file: UploadFile = File(..., description="Training labels file (.pkl format)")
):
    """
    Upload KNN model to the central server.
    
    Args:
        client_id: Unique identifier for the client
        n_samples: Number of samples used for training
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
        
        # Create storage directory
        models_dir = "./central/model_store"
        os.makedirs(models_dir, exist_ok=True)
        
        # Save model file
        model_path = os.path.join(models_dir, f"{client_id}_knn_model.pkl")
        model_contents = await model_file.read()
        with open(model_path, "wb") as f:
            f.write(model_contents)
        
        logger.info(f"Received KNN model from {client_id}: {len(model_contents)} bytes")
        
        # Save features
        features_path = os.path.join(models_dir, f"{client_id}_features.pkl")
        features_contents = await features_file.read()
        with open(features_path, "wb") as f:
            f.write(features_contents)
        logger.info(f"Received features from {client_id}: {len(features_contents)} bytes")
        
        # Save labels
        labels_path = os.path.join(models_dir, f"{client_id}_labels.pkl")
        labels_contents = await labels_file.read()
        with open(labels_path, "wb") as f:
            f.write(labels_contents)
        logger.info(f"Received labels from {client_id}: {len(labels_contents)} bytes")
        
        # Register client upload
        register_client_upload(
            client_id=client_id,
            n_samples=n_samples,
            weights_path=model_path,
            model_type='knn',
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
                "model_type": "knn",
                "n_samples": n_samples,
                "timestamp": datetime.now().isoformat(),
                "message": f"KNN model uploaded successfully from {client_id}",
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
        logger.error(f"Error uploading model from {client_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/aggregate")
async def aggregate():
    """
    Trigger aggregation of uploaded KNN models.
    
    Returns:
        JSON response with aggregation results
    """
    _initialize_server()
    global last_aggregation_time
    
    try:
        # Get all client info from registry
        client_weights_info = get_all_client_weights()
        
        if not client_weights_info:
            raise HTTPException(
                status_code=400,
                detail="No client models available for aggregation"
            )
        
        # Filter for KNN models
        knn_clients = [c for c in client_weights_info if c.get('model_type', 'knn') == 'knn']
        
        if not knn_clients:
            raise HTTPException(
                status_code=400,
                detail="No KNN models available for aggregation"
            )
        
        logger.info(f"Starting KNN aggregation with {len(knn_clients)} clients")
        
        timestamp = datetime.now().isoformat()
        
        # KNN aggregation - merge data and retrain with evaluation
        result = aggregate_knn_models(knn_clients, n_neighbors=5, evaluate=True)
        
        # Save global KNN model
        global_knn_path = "./central/model_store/global_knn_model.pkl"
        save_knn_model(result['global_model'], global_knn_path)
        
        # Store result with metrics
        store_aggregation_result('knn', result, timestamp)
        
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
        
        last_aggregation_time = timestamp
        
        logger.info(f"KNN aggregation completed: {response_content['num_clients']} clients, {response_content['total_samples']} total samples")
        
        return JSONResponse(
            status_code=200,
            content=response_content
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during KNN aggregation: {e}")
        raise HTTPException(status_code=500, detail=f"Aggregation failed: {str(e)}")


@app.get("/global_model")
async def get_global_model():
    """
    Download the current global KNN model.
    
    Returns:
        FileResponse with the global model file
    """
    _initialize_server()
    
    try:
        global_model_path = "./central/model_store/global_knn_model.pkl"
        filename = "global_knn_model.pkl"
        
        if not os.path.exists(global_model_path):
            raise HTTPException(
                status_code=404,
                detail="Global KNN model not found. Please run aggregation first."
            )
        
        logger.info("Serving global KNN model download")
        
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


@app.get("/aggregation_results")
async def get_aggregation_results():
    """
    Get aggregation results for KNN models.
    
    Returns:
        JSON response with aggregation results
    """
    _initialize_server()
    
    try:
        result = get_latest_aggregation_result('knn')
        
        if result is None:
            raise HTTPException(
                status_code=404,
                detail="No aggregation results found for KNN models"
            )
        
        # Remove non-serializable objects
        response_result = {k: v for k, v in result['result'].items() if k != 'global_model'}
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "model_type": "knn",
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
        global_model_path = "./central/model_store/global_knn_model.pkl"
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
