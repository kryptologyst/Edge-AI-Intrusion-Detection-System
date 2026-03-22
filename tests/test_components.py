"""Unit tests for intrusion detection system components."""

import pytest
import numpy as np
import torch
import tensorflow as tf
from omegaconf import OmegaConf
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.device_utils import set_deterministic_seed, get_device, get_model_size_mb
from utils.data_utils import NetworkTrafficGenerator, DataPreprocessor, create_data_splits
from models.pytorch_model import IntrusionDetectionModel, CompressedIntrusionDetectionModel, count_parameters
from models.tensorflow_model import TensorFlowIntrusionDetectionModel
from utils.evaluation_utils import ModelEvaluator


@pytest.fixture
def config():
    """Create test configuration."""
    return OmegaConf.create({
        'model': {
            'input_features': 5,
            'hidden_layers': [32, 16],
            'output_classes': 1,
            'activation': 'relu',
            'dropout': 0.2
        },
        'data': {
            'n_samples': 1000,
            'random_state': 42,
            'features': ['duration', 'bytes_sent', 'bytes_received', 'failed_logins', 'suspicious_flags']
        },
        'training': {
            'validation_split': 0.2,
            'epochs': 5,
            'batch_size': 32,
            'learning_rate': 0.001,
            'early_stopping_patience': 3
        },
        'evaluation': {
            'metrics': ['accuracy', 'precision', 'recall', 'f1_score']
        }
    })


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    return X, y


class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_set_deterministic_seed(self):
        """Test deterministic seed setting."""
        set_deterministic_seed(42)
        # Test that numpy random state is set
        assert np.random.get_state()[1][0] == 42
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert device in ['cpu', 'cuda', 'mps']
    
    def test_get_model_size_mb(self, config):
        """Test model size calculation."""
        model = IntrusionDetectionModel(config)
        size_mb = get_model_size_mb(model)
        assert size_mb > 0
        assert isinstance(size_mb, float)


class TestDataUtils:
    """Test data utility functions."""
    
    def test_network_traffic_generator(self, config):
        """Test network traffic data generation."""
        generator = NetworkTrafficGenerator(config)
        X, y = generator.generate_dataset()
        
        assert X.shape == (config.data.n_samples, 5)
        assert y.shape == (config.data.n_samples,)
        assert np.all(y >= 0) and np.all(y <= 1)
    
    def test_data_preprocessor(self, sample_data):
        """Test data preprocessing."""
        X, y = sample_data
        preprocessor = DataPreprocessor()
        
        X_scaled = preprocessor.fit_transform(X)
        assert X_scaled.shape == X.shape
        assert preprocessor.is_fitted
        
        # Test transform on new data
        X_new_scaled = preprocessor.transform(X[:10])
        assert X_new_scaled.shape == (10, 5)
    
    def test_create_data_splits(self, sample_data, config):
        """Test data splitting."""
        X, y = sample_data
        X_train, X_test, y_train, y_test = create_data_splits(X, y, config)
        
        assert len(X_train) + len(X_test) == len(X)
        assert len(y_train) + len(y_test) == len(y)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)


class TestPyTorchModels:
    """Test PyTorch model implementations."""
    
    def test_intrusion_detection_model(self, config):
        """Test base intrusion detection model."""
        model = IntrusionDetectionModel(config)
        
        # Test forward pass
        x = torch.randn(10, config.model.input_features)
        output = model(x)
        
        assert output.shape == (10, config.model.output_classes)
        assert not torch.isnan(output).any()
    
    def test_compressed_model(self, config):
        """Test compressed model."""
        model = CompressedIntrusionDetectionModel(config, compression_ratio=0.5)
        
        # Test forward pass
        x = torch.randn(10, config.model.input_features)
        output = model(x)
        
        assert output.shape == (10, config.model.output_classes)
        
        # Test that compressed model has fewer parameters
        base_model = IntrusionDetectionModel(config)
        compressed_params = count_parameters(model)
        base_params = count_parameters(base_model)
        
        assert compressed_params < base_params
    
    def test_model_parameters(self, config):
        """Test model parameter counting."""
        model = IntrusionDetectionModel(config)
        param_count = count_parameters(model)
        
        assert param_count > 0
        assert isinstance(param_count, int)


class TestTensorFlowModels:
    """Test TensorFlow model implementations."""
    
    def test_tensorflow_model(self, config):
        """Test TensorFlow intrusion detection model."""
        model = TensorFlowIntrusionDetectionModel(config)
        
        # Test model creation
        assert model.model is not None
        assert model.model.input_shape == (None, config.model.input_features)
        assert model.model.output_shape == (None, config.model.output_classes)
    
    def test_model_compilation(self, config):
        """Test model compilation."""
        model = TensorFlowIntrusionDetectionModel(config)
        
        # Test that model is compiled
        assert model.model.optimizer is not None
        assert model.model.loss is not None
        assert len(model.model.metrics) > 0


class TestEvaluationUtils:
    """Test evaluation utility functions."""
    
    def test_model_evaluator(self, config):
        """Test model evaluator initialization."""
        evaluator = ModelEvaluator(config)
        assert evaluator.config == config
        assert evaluator.results == {}
    
    def test_accuracy_metrics(self, config, sample_data):
        """Test accuracy metrics calculation."""
        X, y = sample_data
        evaluator = ModelEvaluator(config)
        
        # Create dummy predictions
        y_pred = np.random.randint(0, 2, len(y))
        y_prob = np.random.rand(len(y))
        
        metrics = evaluator.evaluate_accuracy_metrics(y, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert 'auc' in metrics
        
        assert all(0 <= v <= 1 for v in metrics.values() if isinstance(v, (int, float)))
    
    def test_model_size_metrics(self, config):
        """Test model size metrics."""
        evaluator = ModelEvaluator(config)
        model = IntrusionDetectionModel(config)
        
        size_metrics = evaluator.get_model_size_metrics(model)
        
        assert 'total_parameters' in size_metrics
        assert 'trainable_parameters' in size_metrics
        assert 'model_size_mb' in size_metrics
        
        assert size_metrics['total_parameters'] > 0
        assert size_metrics['model_size_mb'] > 0


class TestIntegration:
    """Integration tests."""
    
    def test_end_to_end_pytorch(self, config):
        """Test end-to-end PyTorch workflow."""
        # Generate data
        generator = NetworkTrafficGenerator(config)
        X, y = generator.generate_dataset()
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X)
        
        # Create splits
        X_train, X_test, y_train, y_test = create_data_splits(X_scaled, y, config)
        
        # Create and test model
        model = IntrusionDetectionModel(config)
        x = torch.FloatTensor(X_test[:5])
        output = model(x)
        
        assert output.shape == (5, 1)
        assert not torch.isnan(output).any()
    
    def test_end_to_end_tensorflow(self, config):
        """Test end-to-end TensorFlow workflow."""
        # Generate data
        generator = NetworkTrafficGenerator(config)
        X, y = generator.generate_dataset()
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.fit_transform(X)
        
        # Create splits
        X_train, X_test, y_train, y_test = create_data_splits(X_scaled, y, config)
        
        # Create and test model
        model = TensorFlowIntrusionDetectionModel(config)
        x = tf.constant(X_test[:5], dtype=tf.float32)
        output = model.predict(x)
        
        assert output.shape == (5, 1)
        assert not np.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__])
