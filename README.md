# Edge AI Intrusion Detection System (IDS)

A research-focused Edge AI project for network intrusion detection using synthetic traffic data. This system demonstrates model compression, quantization, and edge deployment techniques for IoT security applications.

## ⚠️ IMPORTANT DISCLAIMER

**This is a research and educational project only. NOT intended for safety-critical or production deployment.**

This system uses synthetic data and simplified models for educational purposes. Do not use this system for actual security monitoring without proper validation, testing, and security review.

## Project Overview

This Intrusion Detection System (IDS) simulates network traffic monitoring to detect unauthorized access patterns. It demonstrates:

- **Model Efficiency**: Compression, pruning, quantization-aware training
- **Edge Learning**: On-device inference with resource constraints
- **IoT Security**: Network traffic analysis for smart building firewalls
- **Deployment**: Multi-target edge deployment (Raspberry Pi, Jetson, Mobile)

## Architecture

```
src/
├── models/           # Model implementations (PyTorch, TensorFlow)
├── export/           # Model export and conversion utilities
├── runtimes/         # Edge runtime implementations
├── pipelines/        # Data processing pipelines
├── comms/           # Communication protocols (MQTT, etc.)
└── utils/           # Utility functions and helpers

configs/             # Configuration files
├── device/         # Device-specific configs
├── quant/          # Quantization settings
└── comms/         # Communication settings

scripts/            # Training and deployment scripts
demo/              # Streamlit demo application
tests/             # Unit tests
assets/            # Generated plots and artifacts
```

## Quick Start

### Prerequisites

- Python 3.10+
- PyTorch 2.x or TensorFlow 2.x
- CUDA (optional, for GPU acceleration)

### Installation

1. **Clone and setup environment:**
```bash
git clone https://github.com/kryptologyst/Edge-AI-Intrusion-Detection-System.git
cd Edge-AI-Intrusion-Detection-System
pip install -r requirements.txt
```

2. **Run the interactive demo:**
```bash
streamlit run demo/app.py
```

3. **Train a model:**
```bash
python scripts/train.py --config configs/config.yaml --model-type base --framework pytorch
```

### Device-Specific Setup

#### Raspberry Pi 4
```bash
python scripts/train.py --config configs/config.yaml --model-type compressed --framework pytorch --device cpu
```

#### Jetson Nano
```bash
python scripts/train.py --config configs/config.yaml --model-type quantized --framework tensorflow --device cuda
```

## Dataset & Schema

### Synthetic Network Traffic Features

| Feature | Description | Distribution | Range |
|---------|-------------|--------------|-------|
| `duration` | Connection duration (seconds) | Exponential | 0-∞ |
| `bytes_sent` | Data sent (bytes) | Normal | 0-∞ |
| `bytes_received` | Data received (bytes) | Normal | 0-∞ |
| `failed_logins` | Failed login attempts | Poisson | 0-∞ |
| `suspicious_flags` | Suspicious network flags | Poisson | 0-∞ |

### Intrusion Detection Rules

```python
intrusion = (
    (failed_logins > 1) |                    # Multiple failed logins
    (suspicious_flags > 1) |                 # Multiple suspicious flags
    ((duration > 5) & (bytes_sent > 2000)) | # Long duration + large transfer
    ((duration > 3) & (bytes_received > 3000)) # Medium duration + large received
)
```

## Model Variants

### 1. Base Model
- **Architecture**: 5 → 64 → 32 → 1
- **Parameters**: ~2,000
- **Size**: ~8 KB
- **Use Case**: Development and testing

### 2. Compressed Model
- **Compression**: 50% parameter reduction
- **Architecture**: 5 → 32 → 16 → 1
- **Parameters**: ~1,000
- **Size**: ~4 KB
- **Use Case**: Memory-constrained devices

### 3. Quantized Model
- **Quantization**: INT8 post-training
- **Size**: ~2 KB
- **Use Case**: Ultra-low power devices

## Performance Metrics

### Accuracy Metrics
- **Accuracy**: Classification accuracy
- **Precision**: True positive rate
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

### Edge Performance
- **Latency**: P50, P95, P99 inference times
- **Throughput**: Frames per second (FPS)
- **Memory**: Peak RAM usage
- **Model Size**: Compressed model size
- **Energy**: Joules per inference

### Sample Results

| Model | Accuracy | Latency (ms) | Size (MB) | Throughput (FPS) |
|-------|----------|--------------|-----------|------------------|
| Base | 0.923 | 2.1 | 0.008 | 476 |
| Compressed | 0.918 | 1.8 | 0.004 | 556 |
| Quantized | 0.915 | 1.5 | 0.002 | 667 |

## 🔧 Training Commands

### Basic Training
```bash
# PyTorch base model
python scripts/train.py --model-type base --framework pytorch

# TensorFlow compressed model
python scripts/train.py --model-type compressed --framework tensorflow

# Quantized model
python scripts/train.py --model-type quantized --framework pytorch
```

### Advanced Training
```bash
# Custom configuration
python scripts/train.py --config configs/custom_config.yaml --model-type compressed --device cuda

# Export to edge formats
python scripts/train.py --model-type quantized --export-formats tflite,onnx,openvino
```

## Compilation & Deployment

### Export Pipelines

1. **PyTorch → ONNX → Edge Runtime**
```bash
python scripts/export.py --model pytorch_model.pth --format onnx --target edge
```

2. **TensorFlow → TFLite**
```bash
python scripts/export.py --model tensorflow_model --format tflite --quantization int8
```

3. **Multi-target Export**
```bash
python scripts/export.py --model model.pth --formats onnx,tflite,openvino --targets raspberry_pi,jetson_nano
```

### Device Configurations

#### Raspberry Pi 4
- **OS**: Raspberry Pi OS
- **Python**: 3.10+
- **Runtime**: TFLite Runtime
- **Constraints**: 4GB RAM, 5W power

#### Jetson Nano
- **OS**: JetPack 5.0+
- **Python**: 3.8+
- **Runtime**: TensorRT
- **Constraints**: 4GB RAM, 10W power

#### Mobile (Android/iOS)
- **Runtime**: TFLite
- **Constraints**: 256MB RAM, 3W power

## Evaluation & Benchmarking

### Comprehensive Evaluation
```bash
python scripts/evaluate.py --model models/trained_model.pth --test-data data/test.npz
```

### Edge Performance Benchmark
```bash
python scripts/benchmark.py --model model.tflite --device raspberry_pi --runs 1000
```

### Generate Reports
```bash
python scripts/generate_report.py --output-dir reports/ --include-plots
```

## Interactive Demo

### Streamlit Application
```bash
streamlit run demo/app.py
```

### Demo Features
- **Live Detection**: Real-time intrusion detection simulation
- **Performance Metrics**: Comprehensive benchmarking results
- **Model Analysis**: Feature importance and architecture visualization
- **Edge Deployment**: Device configuration and export options

### Demo Screenshots
- Live traffic monitoring dashboard
- Performance comparison charts
- Confusion matrix visualization
- Edge deployment checklist

## Security & Privacy

### Privacy Measures
- No raw PII in logs
- Synthetic data only
- Local processing (no cloud data)
- Encrypted model files

### Security Features
- TLS for MQTT communication
- Device authentication
- Secure model loading
- Input validation

## Testing

### Unit Tests
```bash
pytest tests/ -v
```

### Integration Tests
```bash
pytest tests/integration/ -v
```

### Performance Tests
```bash
python tests/performance_test.py --model model.pth --iterations 1000
```

## API Reference

### Core Classes

#### `IntrusionDetectionModel`
```python
from src.models.pytorch_model import IntrusionDetectionModel

model = IntrusionDetectionModel(config)
model.train(X_train, y_train)
predictions = model.predict(X_test)
```

#### `ModelEvaluator`
```python
from src.utils.evaluation_utils import ModelEvaluator

evaluator = ModelEvaluator(config)
results = evaluator.comprehensive_evaluation(model, X_test, y_test, y_pred)
```

#### `NetworkTrafficGenerator`
```python
from src.utils.data_utils import NetworkTrafficGenerator

generator = NetworkTrafficGenerator(config)
X, y = generator.generate_dataset()
```

## 🛠️ Development

### Code Style
- **Formatting**: Black + Ruff
- **Type Hints**: Required for all functions
- **Docstrings**: NumPy/Google style
- **Testing**: pytest with >90% coverage

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Limitations

### Current Limitations
- Synthetic data only (not real network traffic)
- Simplified intrusion detection rules
- Limited to binary classification
- No real-time network monitoring
- Basic feature extraction

### Future Improvements
- Real network traffic datasets (NSL-KDD, UNSW-NB15)
- Multi-class intrusion types
- Advanced feature engineering
- Real-time packet capture
- Federated learning support

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Edge AI research community
- PyTorch and TensorFlow teams
- Streamlit for the demo framework
- Open source ML tools and libraries

## Support

For questions and support:
- Create an issue on GitHub
- Check the documentation
- Review the demo application

---

**Remember: This is a research and educational project. Not for production use.**
# Edge-AI-Intrusion-Detection-System
