#!/usr/bin/env python3
"""Main training script for Intrusion Detection System."""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from utils.device_utils import set_deterministic_seed, get_device, configure_tensorflow_device
from utils.data_utils import NetworkTrafficGenerator, DataPreprocessor, create_data_splits
from models.pytorch_model import IntrusionDetectionModel, CompressedIntrusionDetectionModel, ModelFactory
from models.tensorflow_model import TensorFlowIntrusionDetectionModel, TensorFlowLiteConverter
from utils.evaluation_utils import ModelEvaluator, calculate_energy_efficiency


def setup_logging(config: Dict[str, Any]) -> None:
    """Setup logging configuration.
    
    Args:
        config: Logging configuration.
    """
    os.makedirs(os.path.dirname(config['file']), exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, config['level']),
        format=config['format'],
        handlers=[
            logging.FileHandler(config['file']),
            logging.StreamHandler(sys.stdout)
        ]
    )


def train_pytorch_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: Dict[str, Any],
    device: str
) -> nn.Module:
    """Train PyTorch model.
    
    Args:
        model: PyTorch model to train.
        X_train: Training features.
        y_train: Training labels.
        X_val: Validation features.
        y_val: Validation labels.
        config: Training configuration.
        device: Device to use for training.
        
    Returns:
        Trained model.
    """
    model.to(device)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            logging.info(f"Epoch {epoch}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= config['early_stopping_patience']:
            logging.info(f"Early stopping at epoch {epoch}")
            break
    
    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train Intrusion Detection System')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file path')
    parser.add_argument('--model-type', type=str, default='base', choices=['base', 'compressed', 'quantized'], help='Model type')
    parser.add_argument('--framework', type=str, default='pytorch', choices=['pytorch', 'tensorflow'], help='Framework to use')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Setup logging
    setup_logging(config.logging)
    logging.info("Starting Intrusion Detection System training")
    
    # Set deterministic seed
    set_deterministic_seed(config.data.random_state)
    
    # Setup device
    if args.device == 'auto':
        device = get_device()
    else:
        device = args.device
    
    logging.info(f"Using device: {device}")
    
    # Configure TensorFlow if using it
    if args.framework == 'tensorflow':
        configure_tensorflow_device('cpu' if device == 'cpu' else 'gpu')
    
    # Generate data
    logging.info("Generating synthetic network traffic data")
    data_generator = NetworkTrafficGenerator(config)
    X, y = data_generator.generate_dataset()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(X)
    
    # Create train/test splits
    X_train, X_test, y_train, y_test = create_data_splits(X_scaled, y, config)
    
    logging.info(f"Dataset split - Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")
    logging.info(f"Intrusion rate: {np.mean(y):.3f}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    if args.framework == 'pytorch':
        # Train PyTorch model
        logging.info("Training PyTorch model")
        model = ModelFactory.create_model(args.model_type, config)
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=config.data.random_state, stratify=y_train
        )
        
        trained_model = train_pytorch_model(
            model, X_train_split, y_train_split, X_val_split, y_val_split, 
            config.training, device
        )
        
        # Evaluate model
        trained_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test).to(device)
            y_pred_prob = torch.sigmoid(trained_model(X_test_tensor)).cpu().numpy().squeeze()
            y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Save model
        model_path = os.path.join(args.output_dir, f"pytorch_{args.model_type}_model.pth")
        torch.save(trained_model.state_dict(), model_path)
        logging.info(f"Model saved to {model_path}")
        
    else:  # TensorFlow
        logging.info("Training TensorFlow model")
        tf_model = TensorFlowIntrusionDetectionModel(config)
        
        # Split training data for validation
        from sklearn.model_selection import train_test_split
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X_train, y_train, test_size=0.2, random_state=config.data.random_state, stratify=y_train
        )
        
        # Train model
        history = tf_model.train(
            tf.constant(X_train_split, dtype=tf.float32),
            tf.constant(y_train_split, dtype=tf.float32),
            tf.constant(X_val_split, dtype=tf.float32),
            tf.constant(y_val_split, dtype=tf.float32)
        )
        
        # Evaluate model
        y_pred_prob = tf_model.predict(tf.constant(X_test, dtype=tf.float32)).numpy().squeeze()
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        # Save model
        model_path = os.path.join(args.output_dir, "tensorflow_model")
        tf_model.save_model(model_path)
        logging.info(f"Model saved to {model_path}")
        
        # Convert to TFLite
        converter = TensorFlowLiteConverter(tf_model.model)
        tflite_path = os.path.join(args.output_dir, "model.tflite")
        converter.save_tflite_model(tflite_path, quantization='dynamic')
        logging.info(f"TFLite model saved to {tflite_path}")
    
    # Comprehensive evaluation
    model_name = f"{args.framework}_{args.model_type}"
    results = evaluator.comprehensive_evaluation(
        trained_model if args.framework == 'pytorch' else tf_model.model,
        X_test, y_test, y_pred, y_pred_prob, model_name
    )
    
    # Print results
    logging.info(f"\n=== {model_name.upper()} RESULTS ===")
    logging.info(f"Accuracy: {results['accuracy_metrics']['accuracy']:.4f}")
    logging.info(f"Precision: {results['accuracy_metrics']['precision']:.4f}")
    logging.info(f"Recall: {results['accuracy_metrics']['recall']:.4f}")
    logging.info(f"F1-Score: {results['accuracy_metrics']['f1_score']:.4f}")
    logging.info(f"Mean Latency: {results['edge_performance']['mean_latency_ms']:.2f} ms")
    logging.info(f"Model Size: {results['model_size']['model_size_mb']:.2f} MB")
    
    # Calculate energy efficiency
    energy_metrics = calculate_energy_efficiency(
        results['accuracy_metrics']['accuracy'],
        results['edge_performance']['mean_latency_ms']
    )
    logging.info(f"Energy per inference: {energy_metrics['energy_per_inference_j']:.6f} J")
    
    # Create performance plots
    evaluator.plot_confusion_matrix(model_name, os.path.join(args.output_dir, f"{model_name}_confusion_matrix.png"))
    
    logging.info("Training completed successfully!")


if __name__ == "__main__":
    main()
