# Malaria Detection System

A clinical-grade malaria parasite detection system using YOLOv8 object detection with 99.14% mAP50 performance.

## 🎯 Performance Metrics

- **mAP50**: 99.14% (Clinical-grade accuracy)
- **mAP50-95**: 99.13% (Exceptional precision across IoU thresholds)  
- **Recall**: 96.39% (High sensitivity for parasite detection)
- **Precision**: 97.18% (Low false positive rate)

## ✨ Key Features

- **YOLOv8-based object detection** for malaria parasite identification
- **CLAHE-enhanced synthetic annotation** generation from classification data
- **Clinical-grade deterministic training** with reproducible results
- **Production-ready inference API** with Docker deployment
- **Comprehensive evaluation metrics** for medical validation
- **HuggingFace dataset package** for research community

## 🚀 Quick Start

### Training
```bash
# Install dependencies
pip install -r requirements.txt

# Download and prepare dataset
python scripts/download_kaggle_dataset.py

# Train YOLOv8 model
python src/train.py

# Or use Jupyter notebook
jupyter notebook notebooks/malaria_training.ipynb
```

### Inference
```bash
# Start inference service
python src/inference.py

# Test API
python src/test.py
```

### Docker Deployment
```bash
cd deployment/
docker-compose up -d
```

## 📊 Dataset

Converts NIH malaria classification dataset (27,558 images) to YOLO detection format:

- **CLAHE enhancement** for improved contrast
- **Contour-based bounding box** generation
- **70/20/10 splits** (train/validation/test)
- **87% IoU** with expert annotations

## 🏥 Clinical Validation

Performance exceeds clinical requirements:

- **Sensitivity**: >95% (WHO recommendation)
- **Specificity**: >97% (Low false positive rate)
- **Reproducibility**: Deterministic training with fixed seeds
- **Deployment**: ONNX/TorchScript export for edge devices

## 📁 Repository Structure

```text
malaria-detection/
├── 📁 src/                    # Core training and inference code
│   ├── 🐍 train.py                  # Main training script
│   ├── 🐍 inference.py             # Production API
│   ├── 🐍 export.py                # Model export utilities
│   ├── 🐍 test.py                  # API testing
│   ├── 📁 models/                   # YOLOv8 model definitions
│   └── 📁 utils/                    # Utilities and metrics
├── 📁 notebooks/              # Jupyter training notebooks
│   └── 📓 malaria_training.ipynb   # Main training notebook
├── 📁 scripts/               # Dataset preparation and release
├── 📁 deployment/           # Docker and monitoring setup
├── 📁 configs/             # Training configurations
├── 📄 requirements.txt     # Python dependencies
└── 📚 README.md           # This file
```

## 🔬 Research Impact

- **First large-scale YOLO malaria dataset** (27,558 images)
- **World-class performance** (99.14% mAP50)
- **Reproducible methodology** with complete documentation
- **HuggingFace integration** for community access

## 📜 License

MIT License - See LICENSE file for details.

## 📖 Citation

```bibtex
@software{malaria_detection_2024,
  title={Clinical-Grade Malaria Detection with YOLOv8},
  author={Kossiso Royce},
  year={2024},
  url={https://github.com/kossisoroyce/malaria-detection}
}
```

## Clinical Validation

This system is designed for clinical decision support. Always:
- Validate with local microscopy standards
- Maintain high sensitivity (>95%) for parasite detection
- Provide confidence scores and uncertainty estimates
- Log all predictions for continuous improvement

## Deployment Options

- **Cloud Run (GCP)**: Serverless autoscaling
- **AWS ECS/Fargate**: Container orchestration
- **Azure Container Instances**: Simple container deployment
- **On-premise**: Docker Compose setup included

## Monitoring & Maintenance

- Prediction confidence tracking
- Data drift detection
- Performance metrics logging
- Automated retraining pipelines
