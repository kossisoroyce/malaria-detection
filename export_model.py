#!/usr/bin/env python3
"""
Model Export Utilities
Export trained models to ONNX, TorchScript, or TensorFlow Lite for deployment.
"""

import os
import sys
import argparse
import torch
import torch.onnx
import numpy as np
from pathlib import Path
import onnxruntime as ort
import json

# Add src to path
sys.path.append('src')

from models.efficientnet import create_model


class ModelExporter:
    """Export trained PyTorch models to various formats for deployment."""
    
    def __init__(self, checkpoint_path, config=None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        self.checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.config = config or self.checkpoint.get('config', {})
        
        # Load model
        self.model = self._load_model()
        
    def _load_model(self):
        """Load model from checkpoint."""
        model_config = self.config.get('model', {})
        
        model = create_model(
            model_name=model_config.get('name', 'efficientnet-b0'),
            num_classes=model_config.get('num_classes', 1),
            pretrained=False,  # Don't load pretrained weights
            dropout=model_config.get('dropout', 0.2),
            use_timm=model_config.get('use_timm', False)
        )
        
        # Load trained weights
        model.load_state_dict(self.checkpoint['model_state_dict'])
        model.eval()
        
        return model.to(self.device)
    
    def export_onnx(self, output_path, input_size=(1, 3, 224, 224), opset_version=13):
        """Export model to ONNX format."""
        
        # Create dummy input
        dummy_input = torch.randn(input_size, device=self.device)
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        # Verify ONNX model
        self._verify_onnx_model(output_path, dummy_input)
        
        print(f"Model exported to ONNX: {output_path}")
        return output_path
    
    def _verify_onnx_model(self, onnx_path, test_input):
        """Verify ONNX model produces same output as PyTorch model."""
        
        # PyTorch prediction
        with torch.no_grad():
            pytorch_output = self.model(test_input)
        
        # ONNX prediction
        ort_session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        onnx_input = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
        onnx_output = ort_session.run(None, onnx_input)[0]
        
        # Compare outputs
        pytorch_np = pytorch_output.cpu().numpy()
        diff = np.abs(pytorch_np - onnx_output).max()
        
        print(f"Max difference between PyTorch and ONNX: {diff:.6f}")
        
        if diff < 1e-5:
            print("✓ ONNX model verification passed")
        else:
            print("⚠ ONNX model verification failed - large difference detected")
        
        return diff < 1e-5
    
    def export_torchscript(self, output_path, input_size=(1, 3, 224, 224)):
        """Export model to TorchScript format."""
        
        # Create dummy input
        dummy_input = torch.randn(input_size, device=self.device)
        
        # Trace the model
        traced_model = torch.jit.trace(self.model, dummy_input)
        
        # Save traced model
        traced_model.save(output_path)
        
        # Verify TorchScript model
        self._verify_torchscript_model(output_path, dummy_input)
        
        print(f"Model exported to TorchScript: {output_path}")
        return output_path
    
    def _verify_torchscript_model(self, script_path, test_input):
        """Verify TorchScript model produces same output as PyTorch model."""
        
        # PyTorch prediction
        with torch.no_grad():
            pytorch_output = self.model(test_input)
        
        # TorchScript prediction
        loaded_model = torch.jit.load(script_path, map_location=self.device)
        with torch.no_grad():
            script_output = loaded_model(test_input)
        
        # Compare outputs
        diff = torch.abs(pytorch_output - script_output).max().item()
        
        print(f"Max difference between PyTorch and TorchScript: {diff:.6f}")
        
        if diff < 1e-6:
            print("✓ TorchScript model verification passed")
        else:
            print("⚠ TorchScript model verification failed - large difference detected")
        
        return diff < 1e-6
    
    def optimize_onnx(self, onnx_path, optimized_path):
        """Optimize ONNX model for inference."""
        try:
            import onnxoptimizer
            
            # Load ONNX model
            import onnx
            model = onnx.load(onnx_path)
            
            # Apply optimizations
            optimized_model = onnxoptimizer.optimize(model)
            
            # Save optimized model
            onnx.save(optimized_model, optimized_path)
            
            print(f"Optimized ONNX model saved: {optimized_path}")
            return optimized_path
            
        except ImportError:
            print("onnxoptimizer not available. Skipping optimization.")
            return onnx_path
    
    def quantize_onnx(self, onnx_path, quantized_path, calibration_data=None):
        """Quantize ONNX model to INT8 for faster inference."""
        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            
            # Dynamic quantization (no calibration data needed)
            quantize_dynamic(
                onnx_path,
                quantized_path,
                weight_type=QuantType.QInt8
            )
            
            print(f"Quantized ONNX model saved: {quantized_path}")
            return quantized_path
            
        except ImportError:
            print("onnxruntime quantization not available. Skipping quantization.")
            return onnx_path
    
    def benchmark_model(self, model_path, format_type='onnx', num_runs=100, input_size=(1, 3, 224, 224)):
        """Benchmark model inference speed."""
        
        if format_type == 'onnx':
            return self._benchmark_onnx(model_path, num_runs, input_size)
        elif format_type == 'torchscript':
            return self._benchmark_torchscript(model_path, num_runs, input_size)
        else:
            raise ValueError(f"Unsupported format: {format_type}")
    
    def _benchmark_onnx(self, onnx_path, num_runs, input_size):
        """Benchmark ONNX model."""
        import time
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        input_name = session.get_inputs()[0].name
        
        # Prepare input
        dummy_input = np.random.randn(*input_size).astype(np.float32)
        
        # Warmup
        for _ in range(10):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            session.run(None, {input_name: dummy_input})
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = 1000 / avg_time  # images/second
        
        return {
            'avg_inference_time_ms': avg_time,
            'throughput_fps': throughput,
            'num_runs': num_runs
        }
    
    def _benchmark_torchscript(self, script_path, num_runs, input_size):
        """Benchmark TorchScript model."""
        import time
        
        # Load TorchScript model
        model = torch.jit.load(script_path, map_location=self.device)
        
        # Prepare input
        dummy_input = torch.randn(input_size, device=self.device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                model(dummy_input)
        
        # Benchmark
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                model(dummy_input)
        
        torch.cuda.synchronize() if self.device.type == 'cuda' else None
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        throughput = 1000 / avg_time  # images/second
        
        return {
            'avg_inference_time_ms': avg_time,
            'throughput_fps': throughput,
            'num_runs': num_runs
        }
    
    def create_deployment_config(self, model_info, save_path):
        """Create deployment configuration file."""
        
        config = {
            'model_info': model_info,
            'preprocessing': {
                'image_size': self.config.get('data', {}).get('image_size', 224),
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225]
            },
            'postprocessing': {
                'threshold': self.config.get('clinical', {}).get('confidence_threshold', 0.5),
                'target_sensitivity': self.config.get('clinical', {}).get('target_sensitivity', 0.95)
            },
            'clinical_thresholds': self.config.get('clinical', {}),
            'model_metadata': {
                'training_epochs': self.checkpoint.get('epoch', 'unknown'),
                'best_auc': self.checkpoint.get('best_auc', 'unknown'),
                'best_sensitivity': self.checkpoint.get('best_sensitivity', 'unknown')
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Deployment config saved: {save_path}")
        return config


def main():
    parser = argparse.ArgumentParser(description='Export malaria detection model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--format', type=str, choices=['onnx', 'torchscript', 'both'],
                       default='onnx', help='Export format')
    parser.add_argument('--output-dir', type=str, default='exports',
                       help='Output directory for exported models')
    parser.add_argument('--optimize', action='store_true',
                       help='Apply optimizations to exported model')
    parser.add_argument('--quantize', action='store_true',
                       help='Quantize model for faster inference')
    parser.add_argument('--benchmark', action='store_true',
                       help='Benchmark exported model')
    parser.add_argument('--input-size', type=int, nargs=4, default=[1, 3, 224, 224],
                       help='Input size for export (batch, channels, height, width)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize exporter
    exporter = ModelExporter(args.checkpoint)
    
    # Get model name for file naming
    model_name = exporter.config.get('model', {}).get('name', 'malaria_model')
    
    exported_models = {}
    
    # Export ONNX
    if args.format in ['onnx', 'both']:
        onnx_path = output_dir / f"{model_name}.onnx"
        exporter.export_onnx(str(onnx_path), tuple(args.input_size))
        exported_models['onnx'] = str(onnx_path)
        
        # Optimize ONNX
        if args.optimize:
            optimized_path = output_dir / f"{model_name}_optimized.onnx"
            exporter.optimize_onnx(str(onnx_path), str(optimized_path))
            exported_models['onnx_optimized'] = str(optimized_path)
        
        # Quantize ONNX
        if args.quantize:
            quantized_path = output_dir / f"{model_name}_quantized.onnx"
            exporter.quantize_onnx(str(onnx_path), str(quantized_path))
            exported_models['onnx_quantized'] = str(quantized_path)
    
    # Export TorchScript
    if args.format in ['torchscript', 'both']:
        script_path = output_dir / f"{model_name}.pt"
        exporter.export_torchscript(str(script_path), tuple(args.input_size))
        exported_models['torchscript'] = str(script_path)
    
    # Benchmark models
    if args.benchmark:
        print("\n" + "="*50)
        print("BENCHMARKING RESULTS")
        print("="*50)
        
        for format_name, model_path in exported_models.items():
            if 'onnx' in format_name:
                results = exporter.benchmark_model(model_path, 'onnx')
            else:
                results = exporter.benchmark_model(model_path, 'torchscript')
            
            print(f"\n{format_name.upper()}:")
            print(f"  Average inference time: {results['avg_inference_time_ms']:.2f} ms")
            print(f"  Throughput: {results['throughput_fps']:.1f} images/second")
    
    # Create deployment configuration
    config_path = output_dir / "deployment_config.json"
    model_info = {
        'exported_models': exported_models,
        'input_size': args.input_size,
        'export_timestamp': str(torch.datetime.now() if hasattr(torch, 'datetime') else 'unknown')
    }
    exporter.create_deployment_config(model_info, str(config_path))
    
    print(f"\n✓ Export completed successfully!")
    print(f"Exported models saved in: {output_dir}")
    print(f"Deployment config: {config_path}")


if __name__ == "__main__":
    main()
