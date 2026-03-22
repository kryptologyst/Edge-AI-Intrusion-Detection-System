#!/usr/bin/env python3
"""Model export and conversion utilities for edge deployment."""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
import torch
import tensorflow as tf
import numpy as np
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.pytorch_model import IntrusionDetectionModel, CompressedIntrusionDetectionModel
from models.tensorflow_model import TensorFlowIntrusionDetectionModel, TensorFlowLiteConverter
from utils.device_utils import set_deterministic_seed


def export_pytorch_to_onnx(model_path: str, output_path: str, config: Dict[str, Any]) -> None:
    """Export PyTorch model to ONNX format.
    
    Args:
        model_path: Path to PyTorch model.
        output_path: Path to save ONNX model.
        config: Model configuration.
    """
    # Load PyTorch model
    model = IntrusionDetectionModel(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(1, config.model.input_features)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    logging.info(f"PyTorch model exported to ONNX: {output_path}")


def export_tensorflow_to_tflite(model_path: str, output_path: str, quantization: str = 'none') -> None:
    """Export TensorFlow model to TensorFlow Lite format.
    
    Args:
        model_path: Path to TensorFlow model.
        output_path: Path to save TFLite model.
        quantization: Quantization method ('none', 'dynamic', 'int8').
    """
    # Load TensorFlow model
    model = tf.keras.models.load_model(model_path)
    
    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantization == 'dynamic':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
    
    tflite_model = converter.convert()
    
    # Save TFLite model
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    logging.info(f"TensorFlow model exported to TFLite: {output_path}")


def benchmark_model(model_path: str, model_format: str, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model performance.
    
    Args:
        model_path: Path to model file.
        model_format: Model format ('pytorch', 'onnx', 'tflite').
        num_runs: Number of benchmark runs.
        
    Returns:
        Dictionary containing benchmark results.
    """
    import time
    
    # Create dummy input
    dummy_input = np.random.randn(1, 5).astype(np.float32)
    
    times = []
    
    if model_format == 'pytorch':
        model = torch.load(model_path, map_location='cpu')
        model.eval()
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(torch.FloatTensor(dummy_input))
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            with torch.no_grad():
                _ = model(torch.FloatTensor(dummy_input))
            end_time = time.time()
            times.append(end_time - start_time)
    
    elif model_format == 'onnx':
        import onnxruntime as ort
        
        session = ort.InferenceSession(model_path)
        
        # Warmup
        for _ in range(10):
            _ = session.run(None, {'input': dummy_input})
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            _ = session.run(None, {'input': dummy_input})
            end_time = time.time()
            times.append(end_time - start_time)
    
    elif model_format == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        # Warmup
        for _ in range(10):
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
        
        # Benchmark
        for _ in range(num_runs):
            start_time = time.time()
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            _ = interpreter.get_tensor(output_details[0]['index'])
            end_time = time.time()
            times.append(end_time - start_time)
    
    # Calculate statistics
    times_ms = np.array(times) * 1000
    
    return {
        'mean_latency_ms': np.mean(times_ms),
        'std_latency_ms': np.std(times_ms),
        'p50_latency_ms': np.percentile(times_ms, 50),
        'p95_latency_ms': np.percentile(times_ms, 95),
        'p99_latency_ms': np.percentile(times_ms, 99),
        'min_latency_ms': np.min(times_ms),
        'max_latency_ms': np.max(times_ms),
        'throughput_fps': 1000 / np.mean(times_ms)
    }


def main():
    """Main export function."""
    parser = argparse.ArgumentParser(description='Export models for edge deployment')
    parser.add_argument('--model-path', type=str, required=True, help='Path to input model')
    parser.add_argument('--output-dir', type=str, default='models/exported', help='Output directory')
    parser.add_argument('--formats', nargs='+', default=['onnx'], 
                       choices=['onnx', 'tflite', 'openvino'], help='Export formats')
    parser.add_argument('--quantization', type=str, default='none',
                       choices=['none', 'dynamic', 'int8'], help='Quantization method')
    parser.add_argument('--benchmark', action='store_true', help='Run benchmark after export')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine model type from path
    model_name = Path(args.model_path).stem
    
    # Export models
    for format_type in args.formats:
        if format_type == 'onnx':
            output_path = os.path.join(args.output_dir, f"{model_name}.onnx")
            export_pytorch_to_onnx(args.model_path, output_path, config)
            
        elif format_type == 'tflite':
            output_path = os.path.join(args.output_dir, f"{model_name}.tflite")
            export_tensorflow_to_tflite(args.model_path, output_path, args.quantization)
        
        # Run benchmark if requested
        if args.benchmark:
            logging.info(f"Benchmarking {format_type} model...")
            results = benchmark_model(output_path, format_type)
            
            logging.info(f"Benchmark Results for {format_type}:")
            for metric, value in results.items():
                logging.info(f"  {metric}: {value:.3f}")
    
    logging.info("Model export completed successfully!")


if __name__ == "__main__":
    main()
