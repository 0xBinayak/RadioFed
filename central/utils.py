import logging
import os
from pathlib import Path
from typing import Dict, Any


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """
    Configure logging for the central server.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("federated_central")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler with formatting
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level.upper()))
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(formatter)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        logger.addHandler(handler)
    
    return logger


def ensure_directories() -> None:
    """
    Create necessary directories for the central server if they don't exist.
    Creates: central/model_store/, data/
    """
    directories = [
        "central/model_store",
        "data"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logging.info(f"Ensured directory exists: {directory}")


def validate_weights(weights: Dict[str, Any]) -> bool:
    """
    Validate that weights dictionary has the expected format.
    
    Args:
        weights: Dictionary containing model weights
    
    Returns:
        True if weights are valid, False otherwise
    """
    if not isinstance(weights, dict):
        return False
    
    # Check that all values are tensors or can be converted to tensors
    for key, value in weights.items():
        if not hasattr(value, 'shape'):  # Basic check for tensor-like objects
            return False
    
    return True
