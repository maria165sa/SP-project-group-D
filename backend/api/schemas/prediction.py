"""
Pydantic Schemas for Request/Response Models

Defines the data structures for API input and output.
"""

from pydantic import BaseModel, Field
from typing import Literal


class PatientInput(BaseModel):
    """
    Input data for prediction endpoint.
    
    Accepts raw medical data from which derived features
    (pulse_pressure, age_group_code) are calculated.
    """
    
    age: int = Field(..., ge=18, le=120, description="Patient age in years")
    systolic_bp: float = Field(..., ge=70, le=250, description="Systolic blood pressure (mmHg)")
    diastolic_bp: float = Field(..., ge=40, le=150, description="Diastolic blood pressure (mmHg)")
    bmi: float = Field(..., ge=15, le=60, description="Body Mass Index")
    heart_rate: float = Field(..., ge=40, le=200, description="Heart rate (bpm)")
    total_cholesterol: float = Field(..., ge=100, le=500, description="Total cholesterol (mg/dL)")
    glucose: float = Field(..., ge=50, le=400, description="Fasting glucose (mg/dL)")
    cigarettes_per_day: float = Field(default=0, ge=0, description="Cigarettes per day")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension diagnosis (0=No, 1=Yes)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 55,
                "systolic_bp": 140,
                "diastolic_bp": 90,
                "bmi": 28.5,
                "heart_rate": 75,
                "total_cholesterol": 220,
                "glucose": 95,
                "cigarettes_per_day": 10,
                "hypertension": 1
            }
        }


class PredictionResponse(BaseModel):
    """
    Response from prediction endpoint.
    
    Contains prediction result and probability.
    """
    
    prediction: int = Field(..., description="Prediction: 0 (No risk) or 1 (Risk of CHD)")
    prediction_label: str = Field(..., description="Human-readable prediction")
    probability: float = Field(..., ge=0, le=1, description="Probability of CHD risk")
    risk_level: Literal["Low", "Medium", "High"] = Field(..., description="Risk level category")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prediction": 1,
                "prediction_label": "Risk of Coronary Heart Disease",
                "probability": 0.65,
                "risk_level": "High"
            }
        }