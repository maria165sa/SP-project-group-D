# Coronary Heart Disease Risk Predictor - Frontend

Modern, responsive web application for predicting 10-year coronary heart disease risk based on patient clinical data.

## Overview

This is a React-based frontend that provides an intuitive interface for healthcare professionals to assess patient cardiovascular risk. It connects to a FastAPI backend that uses a trained SVM model to generate predictions.

## Features

- **Interactive Form**: Easy-to-use patient data entry with validation
- **Real-time Predictions**: Instant risk assessment upon form submission
- **Visual Results**: Color-coded risk levels and interactive pie chart
- **Responsive Design**: Works seamlessly on desktop, tablet, and mobile devices
- **Professional UI**: Clean, modern interface built with Tailwind CSS
- **Error Handling**: Clear error messages and loading states

## Tech Stack

- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts
- **HTTP Client**: Axios
- **Language**: JavaScript (ES6+)

## Prerequisites

- Node.js 18+ and npm
- Backend API running on `http://localhost:8000`

## Installation

### 1. Install Dependencies
```bash
cd frontend
npm install
```

### 2. Start Development Server
```bash
npm run dev
```

The application will be available at `http://localhost:5173`

## Usage

### 1. Ensure Backend is Running

Make sure the API is running:
```bash
# In the project root
uvicorn api.main:app --reload
```

### 2. Fill Patient Information

Enter the following patient data:

| Field | Description | Range |
|-------|-------------|-------|
| Age | Patient age in years | 18-120 |
| Systolic BP | Systolic blood pressure | 70-250 mmHg |
| Diastolic BP | Diastolic blood pressure | 40-150 mmHg |
| BMI | Body Mass Index | 15-60 kg/m² |
| Heart Rate | Heart rate | 40-200 bpm |
| Total Cholesterol | Total cholesterol | 100-500 mg/dL |
| Glucose | Fasting glucose | 50-400 mg/dL |
| Cigarettes per Day | Daily cigarette consumption | ≥0 |
| Hypertension | Hypertension diagnosis | Yes/No |

### 3. Get Prediction

Click "Predict Risk" to receive:
- **Risk Level**: Low, Medium, or High
- **Probability**: Percentage likelihood of CHD in 10 years
- **Visual Chart**: Risk distribution visualization

## Example Test Cases

### Low Risk Patient
```
Age: 35
Systolic BP: 115
Diastolic BP: 75
BMI: 22.5
Heart Rate: 68
Total Cholesterol: 180
Glucose: 85
Cigarettes per Day: 0
Hypertension: No

Expected: ~4% risk (Low)
```

### Medium Risk Patient
```
Age: 52
Systolic BP: 135
Diastolic BP: 85
BMI: 27.0
Heart Rate: 76
Total Cholesterol: 210
Glucose: 105
Cigarettes per Day: 5
Hypertension: Yes

Expected: ~30-50% risk (Medium)
```

### High Risk Patient
```
Age: 65
Systolic BP: 155
Diastolic BP: 95
BMI: 32.0
Heart Rate: 88
Total Cholesterol: 260
Glucose: 140
Cigarettes per Day: 20
Hypertension: Yes

Expected: ~80-90% risk (High)
```

## Components

### PatientForm
Handles user input with:
- Form validation
- Appropriate input types (number fields with min/max)
- Loading state during prediction
- Clear labels and placeholders

### ResultDisplay
Shows prediction results with:
- Color-coded risk level badge
- Probability percentage
- Visual progress bar
- Risk distribution chart
- Prediction label

### RiskChart
Interactive pie chart displaying:
- Risk vs. No Risk distribution
- Color coding based on risk level
- Percentage labels
- Responsive sizing

## API Integration

The frontend communicates with the backend via `src/services/api.js`:
```javascript
// Predict patient risk
predictRisk(patientData)

// Check API health
checkHealth()
```

### API Configuration

Default API URL: `http://localhost:8000`

To change the API endpoint, edit `src/services/api.js`:
```javascript
const API_BASE_URL = 'http://your-api-url.com';
```