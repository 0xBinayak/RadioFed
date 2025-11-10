"""
Federated Learning Central Server with Auto-Start Dashboard

This module provides an auto-starting central server with integrated dashboard
for monitoring federated learning training, visualizing model performance,
and tracking client status for Automatic Modulation Classification (AMC).
"""

import gradio as gr
import uvicorn
import threading
import time
import requests
import logging
from datetime import datetime
from typing import Optional
import os
import sys
import socket

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from central.state import load_config, save_config
from central.server import app
from central.utils import setup_logging, ensure_directories
from central.dashboard import create_dashboard_interface


# Global variables
server_thread: Optional[threading.Thread] = None
server_running = False
logger = None
config = None


def initialize():
    """Initialize logging and configuration."""
    global logger, config
    
    # Ensure directories exist
    ensure_directories()
    
    # Setup logging
    logger = setup_logging("INFO")
    logger.info("Initializing Federated Learning Central Server")
    
    # Load configuration
    try:
        config = load_config()
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.warning(f"Could not load config, using defaults: {e}")
        config = {
            "model_save_path": "./central/model_store/global_model.pth",
            "host": "127.0.0.1",
            "port": 8000,
            "log_level": "INFO",
            "auto_aggregate_threshold": 2
        }


def is_port_available(host: str, port: int) -> bool:
    """
    Check if a port is available for binding.
    
    Args:
        host: Host address to check
        port: Port number to check
    
    Returns:
        True if port is available, False otherwise
    """
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, port))
            return True
    except OSError:
        return False


def wait_for_server_ready(host: str, port: int, timeout: int = 10) -> bool:
    """
    Wait for the FastAPI server to be ready.
    
    Args:
        host: Server host address
        port: Server port number
        timeout: Maximum time to wait in seconds
    
    Returns:
        True if server is ready, False if timeout
    """
    check_host = "localhost" if host in ["0.0.0.0", "127.0.0.1"] else host
    url = f"http://{check_host}:{port}/health"
    
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    
    return False


def start_server_handler(host: str, port: int) -> str:
    """
    Start the FastAPI server in a background thread.
    
    Args:
        host: Server host address
        port: Server port number
    
    Returns:
        Status message
    """
    global server_thread, server_running, config
    
    if server_running:
        return "[WARNING] Server is already running"
    
    try:
        # Update config
        config["host"] = host
        config["port"] = int(port)
        save_config(config)
        
        # Use 127.0.0.1 for actual binding on Windows, but keep config value
        actual_host = "127.0.0.1" if host == "0.0.0.0" else host
        
        # Start server in background thread
        def run_server():
            global server_running
            server_running = True
            logger.info(f"Starting FastAPI server on {actual_host}:{port}")
            uvicorn.run(app, host=actual_host, port=int(port), log_level="info")
            server_running = False
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(2)
        
        access_url = f"http://localhost:{port}" if actual_host == "127.0.0.1" else f"http://{actual_host}:{port}"
        return f"[SUCCESS] Server started successfully\nAccess at: {access_url}"
    
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        server_running = False
        return f"[ERROR] Failed to start server: {str(e)}"


def start_fastapi_server_background(host: str, port: int) -> bool:
    """
    Start the FastAPI server in a background thread.
    
    Args:
        host: Server host address
        port: Server port number
    
    Returns:
        True if server started successfully, False otherwise
    """
    global server_thread, server_running
    
    if server_running:
        logger.warning("Server is already running")
        return True
    
    try:
        # Check if port is available
        actual_host = "127.0.0.1" if host in ["0.0.0.0", "127.0.0.1"] else host
        
        if not is_port_available(actual_host, port):
            logger.error(f"Port {port} is already in use. Please choose a different port or stop the conflicting service.")
            raise RuntimeError(f"Port {port} is already in use")
        
        # Start server in background thread
        def run_server():
            global server_running
            server_running = True
            logger.info(f"FastAPI server starting on {actual_host}:{port}")
            try:
                uvicorn.run(app, host=actual_host, port=int(port), log_level="info")
            except Exception as e:
                logger.error(f"FastAPI server error: {e}")
            finally:
                server_running = False
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to be ready
        logger.info("Waiting for FastAPI server to be ready...")
        if wait_for_server_ready(actual_host, port):
            logger.info(f"✓ FastAPI server is ready at http://localhost:{port}")
            return True
        else:
            logger.error("FastAPI server failed to start within timeout period")
            return False
    
    except Exception as e:
        logger.error(f"Failed to start FastAPI server: {e}")
        server_running = False
        return False


def main():
    """
    Main entry point for the Federated Learning Central Server.
    
    This function:
    1. Initializes logging and configuration
    2. Auto-starts the FastAPI server on the configured port
    3. Waits for server readiness
    4. Launches the dashboard interface immediately
    
    The server starts automatically without user interaction, and the
    dashboard is displayed as the default view.
    """
    try:
        # Initialize logging and configuration
        initialize()
        
        logger.info("=" * 60)
        logger.info("Federated Learning Central Server - Auto-Start Mode")
        logger.info("=" * 60)
        
        # Get server configuration
        host = config.get("host", "127.0.0.1")
        port = config.get("port", 8000)
        dashboard_port = 7860
        
        # Auto-start FastAPI server
        logger.info(f"Starting FastAPI server on {host}:{port}...")
        
        if not start_fastapi_server_background(host, port):
            logger.error("Failed to start FastAPI server. Exiting.")
            logger.error("Please check if the port is already in use or if there are permission issues.")
            sys.exit(1)
        
        # Log server URLs
        logger.info("=" * 60)
        logger.info(f"✓ FastAPI Server: http://localhost:{port}")
        logger.info(f"  - API Documentation: http://localhost:{port}/docs")
        logger.info(f"  - Health Check: http://localhost:{port}/health")
        logger.info("=" * 60)
        
        # Create and launch dashboard interface
        logger.info(f"Launching dashboard on port {dashboard_port}...")
        
        try:
            # Import and use the dashboard interface
            dashboard = create_dashboard_interface()
            
            logger.info("=" * 60)
            logger.info(f"✓ Dashboard: http://localhost:{dashboard_port}")
            logger.info("=" * 60)
            logger.info("Server is ready! Dashboard will open automatically.")
            logger.info("Press Ctrl+C to stop the server.")
            logger.info("=" * 60)
            
            # Launch dashboard
            dashboard.launch(
                server_name="127.0.0.1",
                server_port=dashboard_port,
                share=False,
                show_error=True,
                quiet=False
            )
        
        except Exception as e:
            logger.error(f"Failed to launch dashboard: {e}")
            logger.error("FastAPI server is still running. You can access the API directly.")
            logger.info(f"API available at: http://localhost:{port}")
            
            # Keep the process alive so FastAPI server continues running
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                logger.info("Shutting down...")
    
    except KeyboardInterrupt:
        logger.info("\nShutdown requested by user")
        logger.info("Stopping server...")
    
    except Exception as e:
        logger.error(f"Fatal error during startup: {e}")
        logger.error("Server failed to start. Please check the logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
