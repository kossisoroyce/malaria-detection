# Malaria Detection Pipeline

A complete end-to-end malaria detection system for clinical microscope integration with parasite counting and localization.

## Features

- **YOLOv8 Detection**: Parasite detection and counting in full microscope fields
- **Clinical Integration**: Designed for hospital and clinic deployment
- **Quantitative Analysis**: Parasite counts, density, and infection level assessment
- **Production Ready**: Docker containerization, monitoring, and autoscaling support
- **Clinical Visualization**: Comprehensive reporting with bounding boxes and heatmaps

## Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
```bash
# Create data directory structure
mkdir -p data/{raw,preprocessed,train/{parasitized,uninfected},val,test}

# Download datasets (you'll need to source these)
# - NIH Malaria Dataset: https://lhncbc.nlm.nih.gov/LHC-publications/pubs/MalariaDatasets.html
# - Additional field data from partner clinics
```

### 3. Train YOLOv8 Model

```bash
python train_yolo.py --config configs/yolov8_malaria.yaml
```

### 4. Export for Inference

```bash
python export_yolo_model.py --model runs/detect/train/weights/best.pt --formats onnx --benchmark
```

### 5. Run Inference Service

```bash
python inference_service_yolo.py --model-path exports/best.onnx
# Or with Docker:
docker build -t malaria-yolo-api .
docker run -p 8080:8080 malaria-yolo-api
```

## Project Structure

```
malaria-detection/
├── data/                   # Data directory (YOLOv8 format)
│   └── yolo_malaria/      # YOLO dataset structure
├── src/
│   ├── data/              # Data loading and preprocessing
│   │   ├── yolo_dataset.py # YOLOv8 dataset handler
│   │   └── dataset.py     # Legacy classification dataset
│   ├── models/            # Model definitions
│   │   ├── yolov8_malaria.py # YOLOv8 malaria detection
│   │   └── efficientnet.py   # Legacy classification model
│   └── utils/             # Utilities
│       ├── yolo_metrics.py    # Detection metrics (mAP, etc.)
│       ├── clinical_visualization.py # Clinical reporting
│       ├── metrics.py     # Legacy classification metrics
│       └── gradcam.py     # Legacy explainability
├── configs/               # Training configurations
│   ├── yolov8_malaria.yaml # YOLOv8 training config
│   └── efficientnet_b0.yaml # Legacy config
├── runs/                  # YOLOv8 training outputs
├── exports/               # Exported models
├── train_yolo.py          # YOLOv8 training script
├── export_yolo_model.py   # YOLOv8 export utilities
├── inference_service_yolo.py # YOLOv8 inference API
└── docker-compose.yml     # Full deployment stack
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
