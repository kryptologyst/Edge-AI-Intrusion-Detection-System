"""Utility functions for deterministic seeding and device management."""

import os
import random
from typing import Optional, Union
import numpy as np
import torch
import tensorflow as tf
from omegaconf import DictConfig


def set_deterministic_seed(seed: int = 42) -> None:
    """Set deterministic seeds for all random number generators.
    
    Args:
        seed: Random seed value for reproducibility.
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Environment variables for TensorFlow
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'


def get_device(device: Optional[str] = None) -> str:
    """Get the best available device for computation.
    
    Args:
        device: Preferred device ('cuda', 'cpu', 'mps'). If None, auto-detect.
        
    Returns:
        Available device string.
    """
    if device is not None:
        if device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        elif device == 'mps' and torch.backends.mps.is_available():
            return 'mps'
        else:
            return 'cpu'
    
    # Auto-detect best device
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def configure_tensorflow_device(device: str = 'cpu') -> None:
    """Configure TensorFlow device settings.
    
    Args:
        device: Device to use ('cpu', 'gpu').
    """
    if device == 'cpu':
        tf.config.experimental.set_visible_devices([], 'GPU')
    elif device == 'gpu' and tf.config.list_physical_devices('GPU'):
        # Enable memory growth to avoid allocating all GPU memory
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU memory growth configuration failed: {e}")


def get_model_size_mb(model: Union[torch.nn.Module, tf.keras.Model]) -> float:
    """Calculate model size in megabytes.
    
    Args:
        model: PyTorch or TensorFlow model.
        
    Returns:
        Model size in MB.
    """
    if isinstance(model, torch.nn.Module):
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        total_size = param_size + buffer_size
        return total_size / (1024 * 1024)  # Convert to MB
    elif isinstance(model, tf.keras.Model):
        # For TensorFlow models, estimate based on parameters
        total_params = model.count_params()
        # Assume 4 bytes per parameter (float32)
        return (total_params * 4) / (1024 * 1024)  # Convert to MB
    else:
        raise ValueError("Unsupported model type")


def format_time(seconds: float) -> str:
    """Format time duration in a human-readable format.
    
    Args:
        seconds: Time duration in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    elif seconds < 1e-3:
        return f"{seconds * 1e6:.2f} μs"
    elif seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    else:
        return f"{seconds:.2f} s"


def format_bytes(bytes_count: int) -> str:
    """Format byte count in a human-readable format.
    
    Args:
        bytes_count: Number of bytes.
        
    Returns:
        Formatted byte string.
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_count < 1024.0:
            return f"{bytes_count:.2f} {unit}"
        bytes_count /= 1024.0
    return f"{bytes_count:.2f} PB"
