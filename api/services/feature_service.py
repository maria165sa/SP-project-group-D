"""
Feature Engineering Service

Handles calculation of derived features from raw input data.
"""

import pandas as pd


def calculate_pulse_pressure(systolic_bp: float, diastolic_bp: float) -> float:
    """
    Calculate pulse pressure.
    
    Args:
        systolic_bp: Systolic blood pressure
        diastolic_bp: Diastolic blood pressure
    
    Returns:
        float: Pulse pressure (systolic - diastolic)
    """
    return systolic_bp - diastolic_bp


def calculate_age_group_code(age: int) -> int:
    """
    Calculate age group code based on age ranges.
    
    Age ranges:
        0: <45 years
        1: 45-54 years
        2: 55-64 years
        3: 65+ years
    
    Args:
        age: Patient age in years
    
    Returns:
        int: Age group code (0-3)
    """
    if age < 45:
        return 0
    elif age < 55:
        return 1
    elif age < 65:
        return 2
    else:
        return 3


def prepare_features(patient_data: dict) -> pd.DataFrame:
    """
    Prepare features for model prediction.
    
    Calculates derived features and organizes data in the
    correct order expected by the model.
    
    Args:
        patient_data: Dictionary with patient information
    
    Returns:
        pd.DataFrame: Features ready for model input
    """
    # Calculate derived features
    pulse_pressure = calculate_pulse_pressure(
        patient_data['systolic_bp'],
        patient_data['diastolic_bp']
    )
    age_group_code = calculate_age_group_code(patient_data['age'])
    
    # Prepare features in correct order
    features = {
        'bmi': patient_data['bmi'],
        'hypertension': patient_data['hypertension'],
        'pulse_pressure': pulse_pressure,
        'cigarettes_per_day': patient_data['cigarettes_per_day'],
        'total_cholesterol': patient_data['total_cholesterol'],
        'glucose': patient_data['glucose'],
        'heart_rate': patient_data['heart_rate'],
        'age_group_code': age_group_code
    }
    
    return pd.DataFrame([features])