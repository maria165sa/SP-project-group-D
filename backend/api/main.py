"""
FastAPI Application Entry Point

Main API application with prediction endpoints.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.schemas.prediction import PatientInput, PredictionResponse
from api.services.feature_service import prepare_features
from api.services.model_service import model_service


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=settings.API_DESCRIPTION,
)

CORS_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
     "https://sp-frontend-qtk3.onrender.com", 
    "https://sp-project-group-d-6.onrender.com"
]

# Add CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,  # use the local variable you defined
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """
    Root endpoint - API health check.
    
    Returns:
        dict: API status and information
    """
    return {
        "status": "alive",
        "api": settings.API_TITLE,
        "version": settings.API_VERSION,
        "model_loaded": model_service.is_loaded()
    }


@app.get("/health")
def health_check():
    """
    Detailed health check endpoint.
    
    Returns:
        dict: Detailed health status
    """
    return {
        "status": "healthy" if model_service.is_loaded() else "unhealthy",
        "model_loaded": model_service.is_loaded(),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(patient: PatientInput):
    """
    Predict coronary heart disease risk.
    
    Accepts patient data, calculates derived features,
    and returns risk prediction.
    
    Args:
        patient: Patient medical data
    
    Returns:
        PredictionResponse: Prediction result with probability and risk level
    
    Raises:
        HTTPException: If prediction fails
    """
    if not model_service.is_loaded():
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Service unavailable."
        )
    
    try:
        # Prepare features
        features = prepare_features(patient.model_dump())
        
        # Make prediction
        prediction, probability = model_service.predict(features)
        
        # Determine risk level
        if probability < 0.25:
            risk_level = "Low"
        elif probability < 0.5:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Format response
        response = PredictionResponse(
            prediction=prediction,
            prediction_label="Risk of Coronary Heart Disease" if prediction == 1 else "Low Risk",
            probability=probability,
            risk_level=risk_level
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )