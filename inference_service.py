#!/usr/bin/env python3
"""
FastAPI Inference Service for Malaria Detection
Production-ready API with monitoring, logging, and clinical safety features.
"""

import os
import sys
import io
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import uuid

import numpy as np
from PIL import Image
import cv2
import torch
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add src to path
sys.path.append('src')

from utils.gradcam import GradCAM, visualize_gradcam
from models.efficientnet import create_model


# Metrics for monitoring
PREDICTION_COUNTER = Counter('malaria_predictions_total', 'Total predictions made', ['result'])
PREDICTION_LATENCY = Histogram('malaria_prediction_duration_seconds', 'Prediction latency')
ERROR_COUNTER = Counter('malaria_errors_total', 'Total errors', ['error_type'])
CONFIDENCE_HISTOGRAM = Histogram('malaria_confidence_scores', 'Distribution of confidence scores')


class PredictionRequest(BaseModel):
    """Request model for prediction."""
    include_gradcam: bool = False
    confidence_threshold: Optional[float] = None
    patient_id: Optional[str] = None
    clinic_id: Optional[str] = None


class PredictionResponse(BaseModel):
    """Response model for prediction."""
    prediction_id: str
    label: str
    probability: float
    confidence_level: str
    clinical_recommendation: str
    processing_time_ms: float
    gradcam_available: bool = False
    timestamp: str


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
    uptime_seconds: float


class MalariaInferenceService:
    """Production malaria inference service with clinical safety features."""
    
    def __init__(self, model_path: str, config_path: Optional[str] = None):
        self.model_path = model_path
        self.config_path = config_path
        self.start_time = time.time()
        
        # Load configuration
        self.config = self._load_config()
        
        # Setup logging
        self._setup_logging()
        
        # Load model
        self.model = None
        self.session = None
        self.model_type = None
        self._load_model()
        
        # Clinical thresholds
        self.confidence_threshold = self.config.get('postprocessing', {}).get('threshold', 0.5)
        self.target_sensitivity = self.config.get('postprocessing', {}).get('target_sensitivity', 0.95)
        
        # Image preprocessing parameters
        self.image_size = self.config.get('preprocessing', {}).get('image_size', 224)
        self.normalize_mean = np.array(self.config.get('preprocessing', {}).get('normalize_mean', [0.485, 0.456, 0.406]))
        self.normalize_std = np.array(self.config.get('preprocessing', {}).get('normalize_std', [0.229, 0.224, 0.225]))
        
        self.logger.info("Malaria inference service initialized successfully")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load deployment configuration."""
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return json.load(f)
        
        # Default configuration
        return {
            'preprocessing': {
                'image_size': 224,
                'normalize_mean': [0.485, 0.456, 0.406],
                'normalize_std': [0.229, 0.224, 0.225]
            },
            'postprocessing': {
                'threshold': 0.5,
                'target_sensitivity': 0.95
            }
        }
    
    def _setup_logging(self):
        """Setup structured logging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('inference_service.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_model(self):
        """Load model based on file extension."""
        model_path = Path(self.model_path)
        
        if model_path.suffix == '.onnx':
            self._load_onnx_model()
        elif model_path.suffix == '.pt':
            self._load_torchscript_model()
        else:
            raise ValueError(f"Unsupported model format: {model_path.suffix}")
    
    def _load_onnx_model(self):
        """Load ONNX model."""
        try:
            providers = ['CPUExecutionProvider']
            if torch.cuda.is_available():
                providers.insert(0, 'CUDAExecutionProvider')
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            self.model_type = 'onnx'
            self.logger.info(f"ONNX model loaded: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _load_torchscript_model(self):
        """Load TorchScript model."""
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = torch.jit.load(self.model_path, map_location=device)
            self.model.eval()
            self.model_type = 'torchscript'
            self.logger.info(f"TorchScript model loaded: {self.model_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load TorchScript model: {e}")
            raise
    
    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for inference."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize
        image = image.resize((self.image_size, self.image_size), Image.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Normalize
        img_array = (img_array - self.normalize_mean) / self.normalize_std
        
        # Add batch dimension and transpose to CHW format
        img_array = np.transpose(img_array, (2, 0, 1))[np.newaxis, :]
        
        return img_array
    
    def _predict_onnx(self, image_array: np.ndarray) -> float:
        """Run inference with ONNX model."""
        input_name = self.session.get_inputs()[0].name
        result = self.session.run(None, {input_name: image_array})
        logit = result[0][0, 0] if result[0].ndim > 1 else result[0][0]
        return float(1 / (1 + np.exp(-logit)))  # Sigmoid
    
    def _predict_torchscript(self, image_array: np.ndarray) -> float:
        """Run inference with TorchScript model."""
        with torch.no_grad():
            tensor = torch.from_numpy(image_array)
            if torch.cuda.is_available():
                tensor = tensor.cuda()
            
            logit = self.model(tensor)
            probability = torch.sigmoid(logit).cpu().item()
            
        return probability
    
    def _get_clinical_recommendation(self, probability: float, confidence_level: str) -> str:
        """Generate clinical recommendation based on prediction."""
        
        if confidence_level == 'low':
            return ("Low confidence prediction. Recommend manual microscopy review "
                   "and consider repeat sample collection.")
        
        if probability > self.confidence_threshold:
            if confidence_level == 'high':
                return ("Parasites detected with high confidence. Recommend immediate "
                       "treatment initiation and confirmatory testing.")
            else:
                return ("Parasites detected with moderate confidence. Recommend "
                       "confirmatory testing before treatment.")
        else:
            if confidence_level == 'high':
                return ("No parasites detected with high confidence. Consider "
                       "clinical symptoms and repeat testing if indicated.")
            else:
                return ("No parasites detected with moderate confidence. Recommend "
                       "manual microscopy review if clinical suspicion remains high.")
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability."""
        distance_from_threshold = abs(probability - 0.5)
        
        if distance_from_threshold > 0.4:
            return 'high'
        elif distance_from_threshold > 0.2:
            return 'moderate'
        else:
            return 'low'
    
    def predict(self, image: Image.Image, request: PredictionRequest) -> PredictionResponse:
        """Main prediction function."""
        start_time = time.time()
        prediction_id = str(uuid.uuid4())
        
        try:
            # Preprocess image
            image_array = self._preprocess_image(image)
            
            # Run inference
            if self.model_type == 'onnx':
                probability = self._predict_onnx(image_array)
            else:
                probability = self._predict_torchscript(image_array)
            
            # Apply custom threshold if provided
            threshold = request.confidence_threshold or self.confidence_threshold
            
            # Generate prediction
            label = 'parasitized' if probability > threshold else 'uninfected'
            confidence_level = self._get_confidence_level(probability)
            clinical_recommendation = self._get_clinical_recommendation(probability, confidence_level)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            PREDICTION_COUNTER.labels(result=label).inc()
            PREDICTION_LATENCY.observe(processing_time / 1000)
            CONFIDENCE_HISTOGRAM.observe(probability)
            
            # Log prediction
            self.logger.info(
                f"Prediction {prediction_id}: {label} (prob={probability:.4f}, "
                f"confidence={confidence_level}, time={processing_time:.1f}ms)"
            )
            
            response = PredictionResponse(
                prediction_id=prediction_id,
                label=label,
                probability=probability,
                confidence_level=confidence_level,
                clinical_recommendation=clinical_recommendation,
                processing_time_ms=processing_time,
                gradcam_available=request.include_gradcam,
                timestamp=datetime.now().isoformat()
            )
            
            return response
            
        except Exception as e:
            ERROR_COUNTER.labels(error_type='prediction_error').inc()
            self.logger.error(f"Prediction error {prediction_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def get_health_status(self) -> HealthResponse:
        """Get service health status."""
        uptime = time.time() - self.start_time
        
        return HealthResponse(
            status='healthy',
            model_loaded=self.model is not None or self.session is not None,
            version='1.0.0',
            uptime_seconds=uptime
        )


# Initialize service
service = None

def get_service():
    """Dependency to get service instance."""
    global service
    if service is None:
        model_path = os.getenv('MODEL_PATH', 'exports/efficientnet-b0.onnx')
        config_path = os.getenv('CONFIG_PATH', 'exports/deployment_config.json')
        service = MalariaInferenceService(model_path, config_path)
    return service


# FastAPI app
app = FastAPI(
    title="Malaria Detection API",
    description="Production-ready malaria detection service with clinical safety features",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict", response_model=PredictionResponse)
async def predict_malaria(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    include_gradcam: bool = False,
    confidence_threshold: Optional[float] = None,
    patient_id: Optional[str] = None,
    clinic_id: Optional[str] = None,
    service: MalariaInferenceService = Depends(get_service)
):
    """Predict malaria from uploaded image."""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Create request
        request = PredictionRequest(
            include_gradcam=include_gradcam,
            confidence_threshold=confidence_threshold,
            patient_id=patient_id,
            clinic_id=clinic_id
        )
        
        # Get prediction
        result = service.predict(image, request)
        
        return result
        
    except Exception as e:
        ERROR_COUNTER.labels(error_type='api_error').inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check(service: MalariaInferenceService = Depends(get_service)):
    """Health check endpoint."""
    return service.get_health_status()


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Malaria Detection API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run malaria detection inference service')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8080, help='Port to bind to')
    parser.add_argument('--model-path', type=str, default='exports/efficientnet-b0.onnx',
                       help='Path to model file')
    parser.add_argument('--config-path', type=str, default='exports/deployment_config.json',
                       help='Path to deployment config')
    parser.add_argument('--workers', type=int, default=1, help='Number of worker processes')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['MODEL_PATH'] = args.model_path
    os.environ['CONFIG_PATH'] = args.config_path
    
    # Run server
    uvicorn.run(
        "inference_service:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info"
    )
