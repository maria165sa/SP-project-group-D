"""
Model Service

Handles model loading and prediction logic.
"""

import joblib
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional

from api.core.config import settings


class ModelService:
    """
    Service for managing model loading and predictions.
    
    Attributes:
        model: Loaded scikit-learn model
        scaler: Loaded StandardScaler
    """
    
    def __init__(self):
        """Initialize model service."""
        self.model = None
        self.scaler = None
        self._load_model()
    
    def _load_model(self) -> None:
        """
        Load model and scaler from disk.
        
        Raises:
            FileNotFoundError: If model or scaler files are not found
            Exception: If loading fails
        """
        try:
            # Check if files exist
            if not settings.MODEL_PATH.exists():
                raise FileNotFoundError(f"Model file not found: {settings.MODEL_PATH}")
            if not settings.SCALER_PATH.exists():
                raise FileNotFoundError(f"Scaler file not found: {settings.SCALER_PATH}")
            
            # Load model and scaler
            self.model = joblib.load(settings.MODEL_PATH)
            self.scaler = joblib.load(settings.SCALER_PATH)
            
            print(f"Model loaded successfully from {settings.MODEL_PATH}")
            print(f"Scaler loaded successfully from {settings.SCALER_PATH}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, features: pd.DataFrame) -> Tuple[int, float]:
        """
        Make prediction using loaded model.
        
        Args:
            features: DataFrame with features in correct order
        
        Returns:
            Tuple[int, float]: (prediction, probability)
                - prediction: 0 (no risk) or 1 (risk)
                - probability: Probability of positive class
        
        Raises:
            ValueError: If model is not loaded
        """
        if self.model is None or self.scaler is None:
            raise ValueError("Model not loaded")
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_scaled_df = pd.DataFrame(features_scaled, columns=features.columns)
        
        # Predict
        prediction = self.model.predict(features_scaled_df)[0]
        probabilities = self.model.predict_proba(features_scaled_df)[0]
        probability = float(probabilities[1])  # Probability of class 1 (CHD risk)
        
        return int(prediction), probability
    
    def is_loaded(self) -> bool:
        """
        Check if model and scaler are loaded.
        
        Returns:
            bool: True if both are loaded, False otherwise
        """
        return self.model is not None and self.scaler is not None


# Global model service instance
model_service = ModelService()