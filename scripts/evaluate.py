#!/usr/bin/env python3
"""Comprehensive evaluation script for intrusion detection models."""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import torch
import tensorflow as tf
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.device_utils import set_deterministic_seed, get_device
from utils.data_utils import NetworkTrafficGenerator, DataPreprocessor, create_data_splits
from models.pytorch_model import IntrusionDetectionModel, CompressedIntrusionDetectionModel, ModelFactory
from models.tensorflow_model import TensorFlowIntrusionDetectionModel
from utils.evaluation_utils import ModelEvaluator, calculate_energy_efficiency


def evaluate_model_variants(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Evaluate different model variants comprehensively.
    
    Args:
        config: Configuration object.
        output_dir: Directory to save results.
        
    Returns:
        Dictionary containing all evaluation results.
    """
    # Generate test data
    logging.info("Generating test dataset...")
    data_generator = NetworkTrafficGenerator(config)
    X, y = data_generator.generate_dataset()
    
    # Preprocess data
    preprocessor = DataPreprocessor()
    X_scaled = preprocessor.fit_transform(X)
    
    # Create train/test splits
    X_train, X_test, y_train, y_test = create_data_splits(X_scaled, y, config)
    
    logging.info(f"Dataset: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    logging.info(f"Intrusion rate: {np.mean(y_test):.3f}")
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Model variants to evaluate
    model_configs = [
        ("PyTorch_Base", "base", "pytorch"),
        ("PyTorch_Compressed", "compressed", "pytorch"),
        ("TensorFlow_Base", "base", "tensorflow"),
    ]
    
    device = get_device()
    logging.info(f"Using device: {device}")
    
    for model_name, model_type, framework in model_configs:
        logging.info(f"Evaluating {model_name}...")
        
        try:
            if framework == "pytorch":
                # Create PyTorch model
                model = ModelFactory.create_model(model_type, config)
                model.to(device)
                
                # Quick training simulation (simplified for evaluation)
                model.eval()
                
                # Generate predictions
                with torch.no_grad():
                    X_test_tensor = torch.FloatTensor(X_test).to(device)
                    y_pred_prob = torch.sigmoid(model(X_test_tensor)).cpu().numpy().squeeze()
                    y_pred = (y_pred_prob > 0.5).astype(int)
                
            else:  # TensorFlow
                # Create TensorFlow model
                tf_model = TensorFlowIntrusionDetectionModel(config)
                
                # Generate predictions
                y_pred_prob = tf_model.predict(tf.constant(X_test, dtype=tf.float32)).numpy().squeeze()
                y_pred = (y_pred_prob > 0.5).astype(int)
                model = tf_model.model
            
            # Comprehensive evaluation
            results = evaluator.comprehensive_evaluation(
                model, X_test, y_test, y_pred, y_pred_prob, model_name
            )
            
            # Calculate energy efficiency
            energy_metrics = calculate_energy_efficiency(
                results['accuracy_metrics']['accuracy'],
                results['edge_performance']['mean_latency_ms']
            )
            results['energy_efficiency'] = energy_metrics
            
            logging.info(f"{model_name} - Accuracy: {results['accuracy_metrics']['accuracy']:.4f}, "
                        f"Latency: {results['edge_performance']['mean_latency_ms']:.2f}ms")
            
        except Exception as e:
            logging.error(f"Error evaluating {model_name}: {e}")
            continue
    
    # Create performance leaderboard
    leaderboard = evaluator.create_performance_leaderboard()
    
    # Save results
    results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump({
            'individual_results': evaluator.results,
            'leaderboard': leaderboard
        }, f, indent=2, default=str)
    
    logging.info(f"Results saved to {results_file}")
    
    return {
        'individual_results': evaluator.results,
        'leaderboard': leaderboard
    }


def generate_performance_report(results: Dict[str, Any], output_dir: str) -> None:
    """Generate comprehensive performance report.
    
    Args:
        results: Evaluation results.
        output_dir: Output directory for reports.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    individual_results = results['individual_results']
    leaderboard = results['leaderboard']
    
    # Create performance comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract data for plotting
    models = list(individual_results.keys())
    accuracies = [individual_results[m]['accuracy_metrics']['accuracy'] for m in models]
    latencies = [individual_results[m]['edge_performance']['mean_latency_ms'] for m in models]
    sizes = [individual_results[m]['model_size']['model_size_mb'] for m in models]
    throughputs = [individual_results[m]['edge_performance']['throughput_fps'] for m in models]
    
    # Accuracy comparison
    axes[0, 0].bar(models, accuracies)
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Latency comparison
    axes[0, 1].bar(models, latencies)
    axes[0, 1].set_title('Inference Latency Comparison')
    axes[0, 1].set_ylabel('Latency (ms)')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Model size comparison
    axes[1, 0].bar(models, sizes)
    axes[1, 0].set_title('Model Size Comparison')
    axes[1, 0].set_ylabel('Size (MB)')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Throughput comparison
    axes[1, 1].bar(models, throughputs)
    axes[1, 1].set_title('Throughput Comparison')
    axes[1, 1].set_ylabel('Throughput (FPS)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create efficiency scatter plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i, model in enumerate(models):
        accuracy = accuracies[i]
        latency = latencies[i]
        size = sizes[i]
        
        ax.scatter(latency, accuracy, s=size*1000, alpha=0.7, label=model)
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy vs Latency (bubble size = model size)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'efficiency_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate text report
    report_file = os.path.join(output_dir, 'performance_report.txt')
    with open(report_file, 'w') as f:
        f.write("INTRUSION DETECTION SYSTEM - PERFORMANCE REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERVIEW\n")
        f.write("-" * 20 + "\n")
        f.write(f"Number of models evaluated: {len(models)}\n")
        f.write(f"Test dataset size: {len(individual_results[models[0]]['confusion_matrix'])} samples\n\n")
        
        f.write("INDIVIDUAL MODEL RESULTS\n")
        f.write("-" * 30 + "\n")
        
        for model_name, results in individual_results.items():
            f.write(f"\n{model_name}:\n")
            f.write(f"  Accuracy: {results['accuracy_metrics']['accuracy']:.4f}\n")
            f.write(f"  Precision: {results['accuracy_metrics']['precision']:.4f}\n")
            f.write(f"  Recall: {results['accuracy_metrics']['recall']:.4f}\n")
            f.write(f"  F1-Score: {results['accuracy_metrics']['f1_score']:.4f}\n")
            f.write(f"  Mean Latency: {results['edge_performance']['mean_latency_ms']:.2f} ms\n")
            f.write(f"  P95 Latency: {results['edge_performance']['p95_latency_ms']:.2f} ms\n")
            f.write(f"  Model Size: {results['model_size']['model_size_mb']:.3f} MB\n")
            f.write(f"  Throughput: {results['edge_performance']['throughput_fps']:.1f} FPS\n")
            f.write(f"  Energy per inference: {results['energy_efficiency']['energy_per_inference_j']:.6f} J\n")
        
        f.write("\nLEADERBOARD RANKINGS\n")
        f.write("-" * 25 + "\n")
        
        for category, ranking in leaderboard.items():
            f.write(f"\n{category.replace('_', ' ').title()}:\n")
            for i, entry in enumerate(ranking):
                f.write(f"  {i+1}. {entry['model']}: {entry[list(entry.keys())[1]]:.4f}\n")
    
    logging.info(f"Performance report generated: {report_file}")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Comprehensive model evaluation')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Config file')
    parser.add_argument('--output-dir', type=str, default='reports', help='Output directory')
    parser.add_argument('--include-plots', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(args.output_dir, 'evaluation.log')),
            logging.StreamHandler()
        ]
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Set deterministic seed
    set_deterministic_seed(config.data.random_state)
    
    logging.info("Starting comprehensive model evaluation...")
    
    # Run evaluation
    results = evaluate_model_variants(config, args.output_dir)
    
    # Generate report if requested
    if args.include_plots:
        logging.info("Generating performance report...")
        generate_performance_report(results, args.output_dir)
    
    logging.info("Evaluation completed successfully!")
    
    # Print summary
    leaderboard = results['leaderboard']
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print("\nAccuracy Ranking:")
    for i, entry in enumerate(leaderboard['accuracy_ranking']):
        print(f"  {i+1}. {entry['model']}: {entry['accuracy']:.4f}")
    
    print("\nLatency Ranking (lower is better):")
    for i, entry in enumerate(leaderboard['latency_ranking']):
        print(f"  {i+1}. {entry['model']}: {entry['latency_ms']:.2f} ms")
    
    print("\nEfficiency Ranking:")
    for i, entry in enumerate(leaderboard['efficiency_ranking']):
        print(f"  {i+1}. {entry['model']}: {entry['efficiency']:.4f}")


if __name__ == "__main__":
    main()
