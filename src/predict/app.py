from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator, Field
import pandas as pd
import numpy as np
import mlflow
import yaml
import logging
import time
from typing import List, Dict
from src.monitoring.model_monitor import ModelMonitor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize model monitor
monitor = ModelMonitor(metrics_path="metrics")

# Load config
with open("../../config.yaml") as f:
    config = yaml.safe_load(f)

# Initialize FastAPI app
app = FastAPI(
    title="EEG Eye State Detector API",
    description="API for predicting eye state from EEG signals",
    version="1.0.0"
)

class EEGData(BaseModel):
    AF3: float = Field(..., description="AF3 electrode reading", ge=-6000, le=6000)
    F7: float = Field(..., description="F7 electrode reading", ge=-6000, le=6000)
    F3: float = Field(..., description="F3 electrode reading", ge=-6000, le=6000)
    FC5: float = Field(..., description="FC5 electrode reading", ge=-6000, le=6000)
    T7: float = Field(..., description="T7 electrode reading", ge=-6000, le=6000)
    P7: float = Field(..., description="P7 electrode reading", ge=-6000, le=6000)
    O1: float = Field(..., description="O1 electrode reading", ge=-6000, le=6000)
    O2: float = Field(..., description="O2 electrode reading", ge=-6000, le=6000)
    P8: float = Field(..., description="P8 electrode reading", ge=-6000, le=6000)
    T8: float = Field(..., description="T8 electrode reading", ge=-6000, le=6000)
    FC6: float = Field(..., description="FC6 electrode reading", ge=-6000, le=6000)
    F4: float = Field(..., description="F4 electrode reading", ge=-6000, le=6000)
    F8: float = Field(..., description="F8 electrode reading", ge=-6000, le=6000)
    AF4: float = Field(..., description="AF4 electrode reading", ge=-6000, le=6000)

    @validator('*')
    def check_nan_inf(cls, v):
        if pd.isna(v) or np.isinf(v):
            raise ValueError("Value cannot be NaN or infinite")
        return v

    class Config:
        schema_extra = {
            "example": {
                "AF3": 4329.23,
                "F7": 4009.23,
                "F3": 4289.23,
                "FC5": 4148.21,
                "T7": 4350.26,
                "P7": 4586.15,
                "O1": 4096.00,
                "O2": 4129.23,
                "P8": 4356.41,
                "T8": 4216.41,
                "FC6": 4088.97,
                "F4": 4273.85,
                "F8": 4148.72,
                "AF4": 4163.08
            }
        }

# Load model at startup
@app.on_event("startup")
async def load_model():
    global model
    try:
        logger.info(f"Loading model from MLflow registry (stage: {config['api']['model_stage']})")
        mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
        model_uri = f"models:/{config['mlflow']['model_name']}/{config['api']['model_stage']}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading model")

@app.post("/predict", response_model=dict)
async def predict(data: EEGData):
    start_time = time.time()
    try:
        # Convert input data to DataFrame
        input_dict = data.dict()
        df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = model.predict(df)
        
        # Convert numpy types to Python native types
        prediction = prediction.tolist() if isinstance(prediction, np.ndarray) else prediction
        
        result = {
            "eye_state": "Open" if prediction[0] == 1 else "Closed",
            "prediction": int(prediction[0]),
            "confidence": float(prediction[0])  # Add confidence if available from model
        }
        
        # Log prediction for monitoring
        monitor.log_prediction(
            input_data=input_dict,
            prediction=result,
            prediction_time=time.time() - start_time,
            model_version=config['mlflow']['model_name']
        )
        
        return result
    except ValueError as e:
        logger.warning(f"Validation error in prediction request: {str(e)}")
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_version": config['mlflow']['model_name'],
        "model_stage": config['api']['model_stage']
    }

@app.get("/metrics")
async def metrics():
    """Get model monitoring metrics"""
    try:
        metrics = monitor.calculate_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail="Error calculating metrics")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )
