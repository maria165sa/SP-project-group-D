"""
API Configuration Module

Handles all configuration settings for the API.
"""

from pathlib import Path


class Settings:
    """Application settings and constants."""
    
    # API metadata
    API_TITLE = "Coronary Disease Risk Prediction API"
    API_VERSION = "1.0.0"
    API_DESCRIPTION = "API for predicting 10-year coronary heart disease risk"
    
    # Model paths (relative to project root)
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    MODEL_PATH = PROJECT_ROOT / "models" / "best_model__standard_scaled__svm__class_weight.pkl"
    SCALER_PATH = PROJECT_ROOT / "models" / "scaler_standard.pkl"
    
    # CORS settings (for frontend)
    CORS_ORIGINS = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]


settings = Settings()