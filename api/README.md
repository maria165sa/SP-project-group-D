# Coronary Disease Risk Prediction API

RESTful API for predicting 10-year coronary heart disease risk based on patient clinical data.

## Overview

This API provides endpoints to predict the risk of developing coronary heart disease (CHD) within the next 10 years. It uses a trained Support Vector Machine (SVM) model with StandardScaler preprocessing, achieving reliable risk stratification for cardiovascular disease prevention.

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Request/Response Examples](#requestresponse-examples)
- [Project Structure](#project-structure)
- [Model Details](#model-details)
- [Development](#development)
- [Troubleshooting](#troubleshooting)

## Features

- **Risk Prediction**: Predicts 10-year CHD risk from standard medical measurements
- **Automatic Feature Engineering**: Calculates derived features (pulse pressure, age groups)
- **Risk Stratification**: Categorizes patients into Low/Medium/High risk levels
- **Input Validation**: Pydantic-based validation with medical range constraints
- **CORS Support**: Ready for frontend integration
- **Auto-generated Documentation**: Interactive API docs with Swagger UI

## Requirements

- Python 3.11+
- Dependencies listed in `requirements.txt`

### Core Dependencies
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
pydantic==2.5.0
scikit-learn==1.6.1
pandas==2.1.3
numpy==1.26.2
joblib==1.3.2
```

## Installation

### 1. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify Model Files

Ensure the following files exist in the `models/` directory:
```
models/
├── best_model__standard_scaled__svm__class_weight.pkl
└── scaler_standard.pkl
```

If `scaler_standard.pkl` is missing, run:
```bash
python create_scaler.py
```

## Running the API

### Development Mode
```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

The API will be available at:
- API: `http://localhost:8000`
- Interactive Docs: `http://localhost:8000/docs`
- Alternative Docs: `http://localhost:8000/redoc`

## API Endpoints

### Health Check

#### `GET /`

Basic health check endpoint.

**Response:**
```json
{
  "status": "alive",
  "api": "Coronary Disease Risk Prediction API",
  "version": "1.0.0",
  "model_loaded": true
}
```

#### `GET /health`

Detailed health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Prediction

#### `POST /predict`

Predict coronary heart disease risk.

**Request Body:**
```json
{
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
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Risk of Coronary Heart Disease",
  "probability": 0.65,
  "risk_level": "High"
}
```

## Request/Response Examples

### Example 1: Low Risk Patient

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "systolic_bp": 115,
    "diastolic_bp": 75,
    "bmi": 22.5,
    "heart_rate": 68,
    "total_cholesterol": 180,
    "glucose": 85,
    "cigarettes_per_day": 0,
    "hypertension": 0
  }'
```

**Response:**
```json
{
  "prediction": 0,
  "prediction_label": "Low Risk",
  "probability": 0.04,
  "risk_level": "Low"
}
```

### Example 2: High Risk Patient

**Request:**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "age": 65,
    "systolic_bp": 155,
    "diastolic_bp": 95,
    "bmi": 32.0,
    "heart_rate": 88,
    "total_cholesterol": 260,
    "glucose": 140,
    "cigarettes_per_day": 20,
    "hypertension": 1
  }'
```

**Response:**
```json
{
  "prediction": 1,
  "prediction_label": "Risk of Coronary Heart Disease",
  "probability": 0.85,
  "risk_level": "High"
}
```

### Example 3: Using Python
```python
import requests

url = "http://localhost:8000/predict"

patient_data = {
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

response = requests.post(url, json=patient_data)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.2%}")
```

## Project Structure
```
api/
├── __init__.py
├── main.py                     # FastAPI application entry point
├── core/
│   ├── __init__.py
│   └── config.py               # Configuration settings
├── schemas/
│   ├── __init__.py
│   └── prediction.py           # Pydantic models for request/response
└── services/
    ├── __init__.py
    ├── feature_service.py      # Feature engineering functions
    └── model_service.py        # Model loading and prediction logic
```

## Model Details

### Model Type
- **Algorithm**: Support Vector Machine (SVM) with RBF kernel
- **Preprocessing**: StandardScaler normalization
- **Class Balancing**: Class weights applied during training
- **Training Data**: Framingham Heart Study dataset

### Input Features (Raw)

The API accepts 9 raw clinical measurements:

| Feature | Type | Range | Unit | Description |
|---------|------|-------|------|-------------|
| `age` | Integer | 18-120 | years | Patient age |
| `systolic_bp` | Float | 70-250 | mmHg | Systolic blood pressure |
| `diastolic_bp` | Float | 40-150 | mmHg | Diastolic blood pressure |
| `bmi` | Float | 15-60 | kg/m² | Body Mass Index |
| `heart_rate` | Float | 40-200 | bpm | Heart rate |
| `total_cholesterol` | Float | 100-500 | mg/dL | Total cholesterol |
| `glucose` | Float | 50-400 | mg/dL | Fasting glucose |
| `cigarettes_per_day` | Float | ≥0 | count | Cigarettes smoked per day |
| `hypertension` | Integer | 0 or 1 | - | Hypertension diagnosis (0=No, 1=Yes) |

### Derived Features (Calculated Automatically)

The API automatically calculates these features:

1. **Pulse Pressure**: `systolic_bp - diastolic_bp`
2. **Age Group Code**: 
   - 0: <45 years
   - 1: 45-54 years
   - 2: 55-64 years
   - 3: 65+ years

### Model Features (Final)

After preprocessing, these 8 features are passed to the model:

1. `bmi`
2. `hypertension`
3. `pulse_pressure` (derived)
4. `cigarettes_per_day`
5. `total_cholesterol`
6. `glucose`
7. `heart_rate`
8. `age_group_code` (derived)

### Output

- **Prediction**: Binary classification (0 = No risk, 1 = Risk of CHD)
- **Probability**: Float between 0-1 (probability of developing CHD in 10 years)
- **Risk Level**: Categorical (Low/Medium/High)
  - Low: probability < 0.3
  - Medium: 0.3 ≤ probability < 0.6
  - High: probability ≥ 0.6

## Development

### Running Tests
```bash
# Run API validation tests (if available)
python tests/test_api.py
```

### Code Style

The codebase follows these conventions:
- **Docstrings**: All functions have descriptive docstrings
- **Type Hints**: Used throughout for better IDE support
- **Comments**: In English, professional tone
- **Naming**: Clear, descriptive variable and function names

### Adding New Features

1. Update `api/schemas/prediction.py` for new input fields
2. Modify `api/services/feature_service.py` for feature engineering
3. Update this README with new field descriptions

## Troubleshooting

### Model Not Found

**Error**: `FileNotFoundError: Model file not found`

**Solution**: Ensure model files exist in `models/` directory:
```bash
ls models/
# Should show:
# best_model__standard_scaled__svm__class_weight.pkl
# scaler_standard.pkl
```

### Scaler Not Found

**Error**: `FileNotFoundError: Scaler file not found`

**Solution**: Run the scaler creation script:
```bash
python create_scaler.py
```

### Import Errors

**Error**: `ModuleNotFoundError: No module named 'api'`

**Solution**: Ensure you're running from the project root and the virtual environment is activated:
```bash
# Check current directory
pwd  # Should be project root

# Activate venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac
```

### CORS Issues

**Error**: Frontend cannot connect to API

**Solution**: Update CORS origins in `api/core/config.py`:
```python
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://your-frontend-domain.com"
]
```

## License

This project is part of a master's thesis on biomedical engineering.

## Contact

For questions or issues, please contact the development team.

---

**Built with FastAPI** | **Powered by scikit-learn**