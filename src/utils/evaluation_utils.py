"""Evaluation metrics and performance benchmarking for intrusion detection models."""

import time
import psutil
import numpy as np
import torch
import tensorflow as tf
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import DictConfig


class ModelEvaluator:
    """Comprehensive model evaluation for intrusion detection."""
    
    def __init__(self, config: DictConfig):
        """Initialize the evaluator.
        
        Args:
            config: Configuration object.
        """
        self.config = config
        self.results = {}
    
    def evaluate_accuracy_metrics(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> Dict[str, float]:
        """Evaluate accuracy-based metrics.
        
        Args:
            y_true: True labels.
            y_pred: Predicted labels.
            y_prob: Prediction probabilities (optional).
            
        Returns:
            Dictionary containing accuracy metrics.
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }
        
        if y_prob is not None:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        
        return metrics
    
    def evaluate_edge_performance(
        self, 
        model: Union[torch.nn.Module, tf.keras.Model],
        X_test: np.ndarray,
        num_runs: int = 100,
        warmup_runs: int = 10
    ) -> Dict[str, float]:
        """Evaluate edge performance metrics.
        
        Args:
            model: Trained model.
            X_test: Test data.
            num_runs: Number of inference runs for benchmarking.
            warmup_runs: Number of warmup runs.
            
        Returns:
            Dictionary containing edge performance metrics.
        """
        # Warmup runs
        for _ in range(warmup_runs):
            if isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    _ = model(torch.FloatTensor(X_test[:1]))
            else:
                _ = model.predict(X_test[:1], verbose=0)
        
        # Benchmark inference time
        times = []
        memory_usage = []
        
        for _ in range(num_runs):
            # Measure memory before inference
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Time inference
            start_time = time.time()
            
            if isinstance(model, torch.nn.Module):
                with torch.no_grad():
                    _ = model(torch.FloatTensor(X_test[:1]))
            else:
                _ = model.predict(X_test[:1], verbose=0)
            
            end_time = time.time()
            
            # Measure memory after inference
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            
            times.append(end_time - start_time)
            memory_usage.append(memory_after - memory_before)
        
        # Calculate statistics
        times_ms = np.array(times) * 1000  # Convert to milliseconds
        
        return {
            'mean_latency_ms': np.mean(times_ms),
            'std_latency_ms': np.std(times_ms),
            'p50_latency_ms': np.percentile(times_ms, 50),
            'p95_latency_ms': np.percentile(times_ms, 95),
            'p99_latency_ms': np.percentile(times_ms, 99),
            'min_latency_ms': np.min(times_ms),
            'max_latency_ms': np.max(times_ms),
            'throughput_fps': 1000 / np.mean(times_ms),
            'avg_memory_mb': np.mean(memory_usage),
            'peak_memory_mb': np.max(memory_usage)
        }
    
    def get_model_size_metrics(self, model: Union[torch.nn.Module, tf.keras.Model]) -> Dict[str, float]:
        """Get model size metrics.
        
        Args:
            model: Model to analyze.
            
        Returns:
            Dictionary containing model size metrics.
        """
        if isinstance(model, torch.nn.Module):
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        else:
            total_params = model.count_params()
            trainable_params = total_params  # TensorFlow doesn't distinguish easily
            model_size_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb
        }
    
    def comprehensive_evaluation(
        self,
        model: Union[torch.nn.Module, tf.keras.Model],
        X_test: np.ndarray,
        y_test: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
        model_name: str = "model"
    ) -> Dict[str, Any]:
        """Perform comprehensive model evaluation.
        
        Args:
            model: Trained model.
            X_test: Test features.
            y_test: True labels.
            y_pred: Predicted labels.
            y_prob: Prediction probabilities.
            model_name: Name of the model for identification.
            
        Returns:
            Dictionary containing all evaluation results.
        """
        results = {
            'model_name': model_name,
            'accuracy_metrics': self.evaluate_accuracy_metrics(y_test, y_pred, y_prob),
            'edge_performance': self.evaluate_edge_performance(model, X_test),
            'model_size': self.get_model_size_metrics(model)
        }
        
        # Add confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        results['confusion_matrix'] = cm.tolist()
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        results['additional_metrics'] = {
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
            'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0
        }
        
        self.results[model_name] = results
        return results
    
    def create_performance_leaderboard(self) -> Dict[str, Any]:
        """Create a performance leaderboard comparing all evaluated models.
        
        Returns:
            Dictionary containing leaderboard data.
        """
        if not self.results:
            return {}
        
        leaderboard = {
            'accuracy_ranking': [],
            'latency_ranking': [],
            'size_ranking': [],
            'efficiency_ranking': []
        }
        
        for model_name, results in self.results.items():
            accuracy = results['accuracy_metrics']['accuracy']
            latency = results['edge_performance']['mean_latency_ms']
            size = results['model_size']['model_size_mb']
            efficiency = accuracy / latency  # Accuracy per ms
            
            leaderboard['accuracy_ranking'].append({
                'model': model_name,
                'accuracy': accuracy
            })
            
            leaderboard['latency_ranking'].append({
                'model': model_name,
                'latency_ms': latency
            })
            
            leaderboard['size_ranking'].append({
                'model': model_name,
                'size_mb': size
            })
            
            leaderboard['efficiency_ranking'].append({
                'model': model_name,
                'efficiency': efficiency
            })
        
        # Sort rankings
        leaderboard['accuracy_ranking'].sort(key=lambda x: x['accuracy'], reverse=True)
        leaderboard['latency_ranking'].sort(key=lambda x: x['latency_ms'])
        leaderboard['size_ranking'].sort(key=lambda x: x['size_mb'])
        leaderboard['efficiency_ranking'].sort(key=lambda x: x['efficiency'], reverse=True)
        
        return leaderboard
    
    def plot_confusion_matrix(self, model_name: str, save_path: Optional[str] = None) -> None:
        """Plot confusion matrix for a specific model.
        
        Args:
            model_name: Name of the model.
            save_path: Path to save the plot (optional).
        """
        if model_name not in self.results:
            raise ValueError(f"Model {model_name} not found in results")
        
        cm = np.array(self.results[model_name]['confusion_matrix'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Intrusion'],
                   yticklabels=['Normal', 'Intrusion'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_performance_comparison(self, save_path: Optional[str] = None) -> None:
        """Plot performance comparison across models.
        
        Args:
            save_path: Path to save the plot (optional).
        """
        if not self.results:
            return
        
        models = list(self.results.keys())
        accuracies = [self.results[m]['accuracy_metrics']['accuracy'] for m in models]
        latencies = [self.results[m]['edge_performance']['mean_latency_ms'] for m in models]
        sizes = [self.results[m]['model_size']['model_size_mb'] for m in models]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Accuracy comparison
        axes[0].bar(models, accuracies)
        axes[0].set_title('Accuracy Comparison')
        axes[0].set_ylabel('Accuracy')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Latency comparison
        axes[1].bar(models, latencies)
        axes[1].set_title('Latency Comparison')
        axes[1].set_ylabel('Latency (ms)')
        axes[1].tick_params(axis='x', rotation=45)
        
        # Size comparison
        axes[2].bar(models, sizes)
        axes[2].set_title('Model Size Comparison')
        axes[2].set_ylabel('Size (MB)')
        axes[2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def calculate_energy_efficiency(
    accuracy: float, 
    latency_ms: float, 
    power_watts: float = 5.0
) -> Dict[str, float]:
    """Calculate energy efficiency metrics.
    
    Args:
        accuracy: Model accuracy.
        latency_ms: Inference latency in milliseconds.
        power_watts: Power consumption in watts.
        
    Returns:
        Dictionary containing energy efficiency metrics.
    """
    energy_per_inference_j = (power_watts * latency_ms) / 1000  # Convert ms to s
    accuracy_per_joule = accuracy / energy_per_inference_j if energy_per_inference_j > 0 else 0
    
    return {
        'energy_per_inference_j': energy_per_inference_j,
        'accuracy_per_joule': accuracy_per_joule,
        'inferences_per_kwh': 3600000 / energy_per_inference_j if energy_per_inference_j > 0 else 0
    }
