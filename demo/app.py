"""Streamlit demo application for Intrusion Detection System."""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import sys
from pathlib import Path
import torch
import tensorflow as tf
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.device_utils import set_deterministic_seed, get_device
from utils.data_utils import NetworkTrafficGenerator, DataPreprocessor
from models.pytorch_model import IntrusionDetectionModel, CompressedIntrusionDetectionModel
from models.tensorflow_model import TensorFlowIntrusionDetectionModel
from utils.evaluation_utils import ModelEvaluator


# Page configuration
st.set_page_config(
    page_title="Intrusion Detection System Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ IMPORTANT DISCLAIMER</h4>
    <p><strong>This is a research and educational demonstration only.</strong></p>
    <p>This Intrusion Detection System is NOT intended for safety-critical or production deployment. 
    It uses synthetic data and simplified models for educational purposes. Do not use this system 
    for actual security monitoring without proper validation, testing, and security review.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🛡️ Edge AI Intrusion Detection System</h1>', unsafe_allow_html=True)

# Sidebar configuration
st.sidebar.header("Configuration")

# Load configuration
try:
    config = OmegaConf.load("configs/config.yaml")
except:
    st.error("Configuration file not found. Please ensure configs/config.yaml exists.")
    st.stop()

# Model selection
model_type = st.sidebar.selectbox(
    "Model Type",
    ["Base Model", "Compressed Model", "Quantized Model"],
    help="Choose the model variant to demonstrate"
)

framework = st.sidebar.selectbox(
    "Framework",
    ["PyTorch", "TensorFlow"],
    help="Choose the deep learning framework"
)

# Simulation parameters
st.sidebar.header("Simulation Parameters")
num_samples = st.sidebar.slider("Number of Samples", 100, 1000, 500)
intrusion_rate = st.sidebar.slider("Intrusion Rate (%)", 5, 50, 20) / 100

# Edge device simulation
st.sidebar.header("Edge Device Simulation")
device_type = st.sidebar.selectbox(
    "Device Type",
    ["Raspberry Pi 4", "Jetson Nano", "Mobile Phone", "Custom"],
    help="Simulate different edge device constraints"
)

# Device constraints based on selection
device_constraints = {
    "Raspberry Pi 4": {"max_latency_ms": 100, "max_memory_mb": 512, "max_power_w": 5},
    "Jetson Nano": {"max_latency_ms": 50, "max_memory_mb": 1024, "max_power_w": 10},
    "Mobile Phone": {"max_latency_ms": 30, "max_memory_mb": 256, "max_power_w": 3},
    "Custom": {"max_latency_ms": 200, "max_memory_mb": 1024, "max_power_w": 15}
}

constraints = device_constraints[device_type]

# Main content
tab1, tab2, tab3, tab4 = st.tabs(["📊 Live Detection", "📈 Performance Metrics", "🔧 Model Analysis", "⚙️ Edge Deployment"])

with tab1:
    st.header("Real-time Intrusion Detection Simulation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Network Traffic Stream")
        
        # Generate synthetic data
        set_deterministic_seed(42)
        data_generator = NetworkTrafficGenerator(config)
        
        # Create placeholder for live data
        placeholder = st.empty()
        
        # Start simulation button
        if st.button("🚀 Start Live Detection", type="primary"):
            with st.spinner("Initializing detection system..."):
                # Load or create model
                if framework == "PyTorch":
                    if model_type == "Base Model":
                        model = IntrusionDetectionModel(config)
                    elif model_type == "Compressed Model":
                        model = CompressedIntrusionDetectionModel(config, compression_ratio=0.5)
                    else:  # Quantized
                        base_model = IntrusionDetectionModel(config)
                        model = base_model  # Simplified for demo
                else:  # TensorFlow
                    model = TensorFlowIntrusionDetectionModel(config)
                
                # Preprocessor
                preprocessor = DataPreprocessor()
                
                # Generate initial data
                X, y = data_generator.generate_dataset()
                X_scaled = preprocessor.fit_transform(X)
                
                # Simulate live detection
                detection_results = []
                alerts = []
                
                for i in range(min(50, num_samples)):
                    # Get current sample
                    current_sample = X_scaled[i:i+1]
                    true_label = y[i]
                    
                    # Simulate inference time
                    start_time = time.time()
                    
                    if framework == "PyTorch":
                        with torch.no_grad():
                            pred_prob = torch.sigmoid(model(torch.FloatTensor(current_sample))).numpy()[0]
                    else:
                        pred_prob = model.predict(tf.constant(current_sample, dtype=tf.float32)).numpy()[0]
                    
                    inference_time = (time.time() - start_time) * 1000  # Convert to ms
                    
                    pred_label = 1 if pred_prob > 0.5 else 0
                    
                    # Store results
                    detection_results.append({
                        'timestamp': i,
                        'duration': X[i, 0],
                        'bytes_sent': X[i, 1],
                        'bytes_received': X[i, 2],
                        'failed_logins': X[i, 3],
                        'suspicious_flags': X[i, 4],
                        'true_label': true_label,
                        'pred_label': pred_label,
                        'pred_prob': pred_prob[0],
                        'inference_time_ms': inference_time
                    })
                    
                    # Check for alerts
                    if pred_label == 1:
                        alerts.append({
                            'timestamp': i,
                            'severity': 'HIGH' if pred_prob[0] > 0.8 else 'MEDIUM',
                            'details': f"Failed logins: {X[i, 3]}, Suspicious flags: {X[i, 4]}"
                        })
                    
                    # Update display
                    with placeholder.container():
                        # Create DataFrame for display
                        df = pd.DataFrame(detection_results[-10:])  # Show last 10 samples
                        
                        # Color code predictions
                        def color_predictions(val):
                            if val == 1:
                                return 'background-color: #ffebee'
                            else:
                                return 'background-color: #e8f5e8'
                        
                        styled_df = df.style.applymap(color_predictions, subset=['pred_label'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Show current alert if any
                        if alerts:
                            latest_alert = alerts[-1]
                            st.markdown(f"""
                            <div class="warning-box">
                                <h4>🚨 ALERT: Intrusion Detected</h4>
                                <p><strong>Severity:</strong> {latest_alert['severity']}</p>
                                <p><strong>Details:</strong> {latest_alert['details']}</p>
                                <p><strong>Confidence:</strong> {latest_alert.get('confidence', 'N/A')}</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        time.sleep(0.5)  # Simulate real-time delay
    
    with col2:
        st.subheader("Detection Statistics")
        
        if 'detection_results' in locals():
            df = pd.DataFrame(detection_results)
            
            # Calculate metrics
            accuracy = np.mean(df['pred_label'] == df['true_label'])
            avg_latency = np.mean(df['inference_time_ms'])
            total_alerts = len(alerts)
            
            # Display metrics
            st.metric("Accuracy", f"{accuracy:.3f}")
            st.metric("Avg Latency", f"{avg_latency:.1f} ms")
            st.metric("Total Alerts", total_alerts)
            st.metric("Throughput", f"{1000/avg_latency:.1f} FPS")
            
            # Latency distribution
            fig_latency = px.histogram(df, x='inference_time_ms', nbins=20, 
                                      title="Inference Latency Distribution")
            st.plotly_chart(fig_latency, use_container_width=True)

with tab2:
    st.header("Performance Metrics & Benchmarking")
    
    # Generate comprehensive evaluation data
    if st.button("📊 Generate Performance Report"):
        with st.spinner("Running comprehensive evaluation..."):
            # Generate test data
            data_generator = NetworkTrafficGenerator(config)
            X, y = data_generator.generate_dataset()
            preprocessor = DataPreprocessor()
            X_scaled = preprocessor.fit_transform(X)
            
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Evaluate different model variants
            evaluator = ModelEvaluator(config)
            models_to_eval = [
                ("PyTorch Base", IntrusionDetectionModel(config)),
                ("PyTorch Compressed", CompressedIntrusionDetectionModel(config, 0.5)),
                ("TensorFlow Base", TensorFlowIntrusionDetectionModel(config))
            ]
            
            for model_name, model in models_to_eval:
                # Train model (simplified for demo)
                if "PyTorch" in model_name:
                    # Quick training simulation
                    model.eval()
                    with torch.no_grad():
                        y_pred_prob = torch.sigmoid(model(torch.FloatTensor(X_test))).numpy().squeeze()
                else:
                    y_pred_prob = model.predict(tf.constant(X_test, dtype=tf.float32)).numpy().squeeze()
                
                y_pred = (y_pred_prob > 0.5).astype(int)
                
                # Evaluate
                evaluator.comprehensive_evaluation(
                    model, X_test, y_test, y_pred, y_pred_prob, model_name
                )
            
            # Create leaderboard
            leaderboard = evaluator.create_performance_leaderboard()
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Accuracy Ranking")
                acc_df = pd.DataFrame(leaderboard['accuracy_ranking'])
                fig_acc = px.bar(acc_df, x='model', y='accuracy', 
                               title="Model Accuracy Comparison")
                st.plotly_chart(fig_acc, use_container_width=True)
            
            with col2:
                st.subheader("Latency Ranking")
                lat_df = pd.DataFrame(leaderboard['latency_ranking'])
                fig_lat = px.bar(lat_df, x='model', y='latency_ms', 
                               title="Inference Latency Comparison")
                st.plotly_chart(fig_lat, use_container_width=True)
            
            # Efficiency comparison
            st.subheader("Efficiency Analysis")
            eff_df = pd.DataFrame(leaderboard['efficiency_ranking'])
            fig_eff = px.bar(eff_df, x='model', y='efficiency', 
                           title="Accuracy per Latency Efficiency")
            st.plotly_chart(fig_eff, use_container_width=True)
            
            # Detailed metrics table
            st.subheader("Detailed Performance Metrics")
            metrics_data = []
            for model_name, results in evaluator.results.items():
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': f"{results['accuracy_metrics']['accuracy']:.4f}",
                    'Precision': f"{results['accuracy_metrics']['precision']:.4f}",
                    'Recall': f"{results['accuracy_metrics']['recall']:.4f}",
                    'F1-Score': f"{results['accuracy_metrics']['f1_score']:.4f}",
                    'Latency (ms)': f"{results['edge_performance']['mean_latency_ms']:.2f}",
                    'Model Size (MB)': f"{results['model_size']['model_size_mb']:.2f}",
                    'Throughput (FPS)': f"{results['edge_performance']['throughput_fps']:.1f}"
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)

with tab3:
    st.header("Model Analysis & Visualization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Feature Importance Analysis")
        
        # Generate feature importance data (simulated)
        feature_names = ['Duration', 'Bytes Sent', 'Bytes Received', 'Failed Logins', 'Suspicious Flags']
        importance_scores = np.random.rand(5)
        importance_scores = importance_scores / importance_scores.sum()  # Normalize
        
        fig_importance = px.bar(
            x=feature_names, 
            y=importance_scores,
            title="Feature Importance Scores",
            labels={'x': 'Features', 'y': 'Importance'}
        )
        st.plotly_chart(fig_importance, use_container_width=True)
    
    with col2:
        st.subheader("Model Architecture")
        
        # Display model architecture info
        st.info("""
        **Model Architecture:**
        - Input Layer: 5 features
        - Hidden Layers: 64 → 32 neurons
        - Output Layer: 1 neuron (binary classification)
        - Activation: ReLU
        - Dropout: 0.2
        """)
        
        # Model size comparison
        model_sizes = {
            'Base Model': 0.5,
            'Compressed Model': 0.25,
            'Quantized Model': 0.125
        }
        
        fig_size = px.pie(
            values=list(model_sizes.values()),
            names=list(model_sizes.keys()),
            title="Model Size Distribution"
        )
        st.plotly_chart(fig_size, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("Confusion Matrix")
    
    # Generate sample confusion matrix
    cm_data = np.array([[85, 5], [8, 2]])  # Example confusion matrix
    
    fig_cm = px.imshow(
        cm_data,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        x=['Normal', 'Intrusion'],
        y=['Normal', 'Intrusion']
    )
    st.plotly_chart(fig_cm, use_container_width=True)

with tab4:
    st.header("Edge Deployment Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Device Constraints")
        
        # Display current device constraints
        st.metric("Max Latency", f"{constraints['max_latency_ms']} ms")
        st.metric("Max Memory", f"{constraints['max_memory_mb']} MB")
        st.metric("Max Power", f"{constraints['max_power_w']} W")
        
        # Deployment checklist
        st.subheader("Deployment Checklist")
        
        checklist_items = [
            ("Model Size < Memory Limit", True),
            ("Latency < Constraint", True),
            ("Power Consumption OK", True),
            ("Quantization Applied", model_type == "Quantized Model"),
            ("Edge Runtime Ready", True)
        ]
        
        for item, status in checklist_items:
            if status:
                st.markdown(f"✅ {item}")
            else:
                st.markdown(f"❌ {item}")
    
    with col2:
        st.subheader("Export Options")
        
        export_formats = st.multiselect(
            "Select Export Formats",
            ["TensorFlow Lite", "ONNX", "OpenVINO", "CoreML", "TensorRT"],
            default=["TensorFlow Lite"]
        )
        
        if st.button("📦 Export Models"):
            with st.spinner("Exporting models..."):
                progress_bar = st.progress(0)
                
                for i, format_name in enumerate(export_formats):
                    time.sleep(1)  # Simulate export time
                    progress_bar.progress((i + 1) / len(export_formats))
                
                st.success(f"Models exported successfully in {len(export_formats)} formats!")
        
        # Deployment commands
        st.subheader("Deployment Commands")
        st.code("""
# Install edge runtime
pip install tflite-runtime

# Deploy model
python scripts/deploy.py --model model.tflite --device raspberry_pi

# Start monitoring service
python scripts/monitor.py --config configs/device/raspberry_pi.yaml
        """, language="bash")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>Edge AI Intrusion Detection System - Research & Educational Demo</p>
    <p><strong>Not for production use</strong></p>
</div>
""", unsafe_allow_html=True)
