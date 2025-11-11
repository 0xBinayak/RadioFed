"""
Client Synchronization Module

This module handles communication with the central server including:
- Uploading local model weights
- Downloading global model
- Checking server status
- Retry logic with exponential backoff
"""

import requests
import time
import logging
import os
from typing import Dict, Optional, Tuple



logger = logging.getLogger(__name__)


def check_server_status(server_url: str, timeout: int = 5) -> Dict:
    """
    Query server health and status.
    
    Args:
        server_url: Base URL of the central server (e.g., "http://localhost:8000")
        timeout: Request timeout in seconds
    
    Returns:
        Dictionary containing server status information
        
    Raises:
        requests.exceptions.RequestException: If connection fails
    """
    try:
        
        server_url = server_url.rstrip('/')
        response = requests.get(
            f"{server_url}/status",
            timeout=timeout
        )
        response.raise_for_status()
        
        status_data = response.json()
        logger.info(f"Server status retrieved: {status_data.get('server_status', 'unknown')}")
        
        return status_data
    
    except requests.exceptions.Timeout:
        logger.error(f"Server status check timed out after {timeout}s")
        raise
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Failed to connect to server at {server_url}: {e}")
        raise
    except requests.exceptions.RequestException as e:
        logger.error(f"Error checking server status: {e}")
        raise


def upload_weights(
    server_url: str,
    client_id: str,
    weights_path: str,
    n_samples: int,
    max_retries: int = 3,
    timeout: int = 30
) -> bool:
    """
    Upload local model weights to the central server with retry logic.
    
    Args:
        server_url: Base URL of the central server
        client_id: Unique identifier for this client
        weights_path: Path to the local weights file (.pth)
        n_samples: Number of samples used for training
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        True if upload successful, False otherwise
        
    Raises:
        FileNotFoundError: If weights file doesn't exist
        ValueError: If parameters are invalid
    """
    
    if not client_id or len(client_id.strip()) == 0:
        raise ValueError("client_id cannot be empty")
    
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Weights file not found: {weights_path}")
    
    if not weights_path.endswith('.pth'):
        raise ValueError("Weights file must have .pth extension")
    
    server_url = server_url.rstrip('/')
    
    try:
        check_server_status(server_url, timeout=5)
    except requests.exceptions.RequestException as e:
        logger.error(f"Server connectivity check failed before upload: {e}")
        return False
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Uploading weights (attempt {attempt + 1}/{max_retries}): {weights_path}")
            
            
            with open(weights_path, 'rb') as f:
                files = {'file': (os.path.basename(weights_path), f, 'application/octet-stream')}
                
                
                response = requests.post(
                    f"{server_url}/upload_weights/{client_id}",
                    params={'n_samples': n_samples},
                    files=files,
                    timeout=timeout
                )
                
                response.raise_for_status()
                
                result = response.json()
                logger.info(f"Upload successful: {result.get('message', 'OK')}")
                return True
        
        except requests.exceptions.Timeout:
            logger.warning(f"Upload timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Upload failed: Maximum retries exceeded (timeout)")
                return False
        
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error during upload (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Upload failed: Maximum retries exceeded (connection error)")
                return False
        
        except requests.exceptions.HTTPError as e:
            
            if e.response.status_code >= 400 and e.response.status_code < 500:
                logger.error(f"Upload failed with client error: {e.response.status_code} - {e.response.text}")
                return False
            
            logger.warning(f"Server error during upload (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Upload failed: Maximum retries exceeded (server error)")
                return False
        
        except Exception as e:
            logger.error(f"Unexpected error during upload: {e}")
            return False
    
    return False


def download_global_model(
    server_url: str,
    save_path: str,
    max_retries: int = 3,
    timeout: int = 30
) -> bool:
    """
    Download the global model from the central server with retry logic.
    
    Args:
        server_url: Base URL of the central server
        save_path: Path where the global model should be saved
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        True if download successful, False otherwise
    """
    
    server_url = server_url.rstrip('/')
    
    
    try:
        status = check_server_status(server_url, timeout=5)
        if not status.get('global_model_exists', False):
            logger.error("Global model does not exist on server. Run aggregation first.")
            return False
    except requests.exceptions.RequestException as e:
        logger.error(f"Server connectivity check failed before download: {e}")
        return False
    
    
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading global model (attempt {attempt + 1}/{max_retries})")
            
            
            response = requests.get(
                f"{server_url}/global_model",
                timeout=timeout,
                stream=True
            )
            
            response.raise_for_status()
            
            
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            logger.info(f"Global model downloaded successfully to: {save_path}")
            return True
        
        except requests.exceptions.Timeout:
            logger.warning(f"Download timed out (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Download failed: Maximum retries exceeded (timeout)")
                return False
        
        except requests.exceptions.ConnectionError as e:
            logger.warning(f"Connection error during download (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Download failed: Maximum retries exceeded (connection error)")
                return False
        
        except requests.exceptions.HTTPError as e:
            if e.response.status_code >= 400 and e.response.status_code < 500:
                logger.error(f"Download failed with client error: {e.response.status_code} - {e.response.text}")
                return False
            
            logger.warning(f"Server error during download (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Download failed: Maximum retries exceeded (server error)")
                return False
        
        except Exception as e:
            logger.error(f"Unexpected error during download: {e}")
            if os.path.exists(save_path):
                try:
                    os.remove(save_path)
                    logger.info("Cleaned up partial download")
                except:
                    pass
            return False
    
    return False
