"""Intrusion Detection System - Edge AI Package."""

__version__ = "1.0.0"
__author__ = "Edge AI Research Team"
__description__ = "Edge AI Intrusion Detection System for IoT Security"

from .models.pytorch_model import IntrusionDetectionModel, CompressedIntrusionDetectionModel
from .models.tensorflow_model import TensorFlowIntrusionDetectionModel
from .utils.device_utils import set_deterministic_seed, get_device
from .utils.data_utils import NetworkTrafficGenerator, DataPreprocessor
from .utils.evaluation_utils import ModelEvaluator

__all__ = [
    "IntrusionDetectionModel",
    "CompressedIntrusionDetectionModel", 
    "TensorFlowIntrusionDetectionModel",
    "set_deterministic_seed",
    "get_device",
    "NetworkTrafficGenerator",
    "DataPreprocessor",
    "ModelEvaluator"
]
