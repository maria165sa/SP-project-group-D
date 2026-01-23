# Coronary Heart Disease Risk Prediction (Group D)

End-to-end project for estimating 10-year coronary heart disease (CHD) risk from the Framingham study data. The repo contains four workstreams: exploratory analysis and preprocessing, traditional ML baselines, boosted/stacked model selection, and an API + React UI for serving predictions.

## Repository Map
- `data/raw/`: Original Framingham CSV (`coronary_disease.csv`, 4,238 rows, 16 columns) plus a data dictionary.
- `data/processed/`: Cleaned dataset (`cleaned_coronary_disease.csv`), engineered feature set (`processed_features.csv`), train/test splits in three scale variants (`*_unscaled/standard_scaled/robust_scaled.csv`), and model comparison tables.
- `notebooks/`: Phase notebooks `01_EDA.ipynb`, `02_ML_Models.ipynb`, `03_Model_Selection.ipynb`, `04_API_Dev_and_Deploy.ipynb`.
- `figures/`: Saved plots for each phase (EDA, ML runs, model selection diagnostics).
- `src/`: Reusable preprocessing and modeling helpers.
- `backend/`: FastAPI service that hosts the production model and feature logic.
- `frontend/`: React + Vite app that calls the API.
- `models/`: Persisted models (`best_model__standard_scaled__svm__class_weight.pkl` for the API; AdaBoost variants and metadata for model-selection experiments) plus `scaler_standard.pkl`.

## Phase 1 — 01_EDA (Exploration & Preprocessing)
- Cleaning (`src/data_cleaning.py`): column renames to snake_case; type fixes; median/mode imputation for <5% missing values; KNN imputation for glucose (9.2% missing); outliers profiled via IQR and z-score and retained due to clinical plausibility.
- Feature engineering (`src/feature_engineering.py`): pulse pressure, BMI category, age groups, risk factor counts, ordinal encodings; multicollinearity checked with VIF (diagnostic only).
- Final modeling feature set (used downstream): `bmi`, `hypertension`, `pulse_pressure`, `cigarettes_per_day`, `total_cholesterol`, `glucose`, `heart_rate`, `age_group_code` plus target.
- Class imbalance after cleaning: ~15% positives (TenYearCHD=1).
- Outputs: `processed_features.csv`, train/test splits for unscaled, StandardScaler, and RobustScaler versions; visual summaries in `figures/01_EDA/`.

## Phase 2 — 02_ML_Models (Traditional Baselines)
- Models: Logistic Regression, KNN, Decision Tree, Random Forest, SVM across unscaled/standard/robust datasets.
- Imbalance handling: class weights and SMOTE (within CV only). Scoring prioritized F1, with precision/recall/accuracy and ROC-AUC reported.
- Best traditional model: Standard-scaled SVM with class weights (`models/best_model__standard_scaled__svm__class_weight.pkl` + `models/scaler_standard.pkl`).
  - Test metrics (standard-scaled set, 848 samples, 15.2% positives): accuracy 0.67, precision 0.25, recall 0.60, F1 0.35, ROC-AUC 0.69, PR-AUC 0.27.
- Figures: per-run confusion/ROC/PR plots and aggregates stored under `figures/02_ML_Models/`.

## Phase 3 — 03_Model_Selection (Boosting & Stacking)
- Goal: push recall while keeping ROC-AUC strong. Weighted score = 0.6 * ROC-AUC + 0.4 * recall; thresholds optimized (min recall target 0.65) via F2-style tuning.
- Models: tuned XGBoost, LightGBM, AdaBoost (Optuna), plus two stackers (XGB+LGBM+RF+ET and XGB+LGBM+RF+SVM) on unscaled, standard, and robust datasets.
- Key test results (see `data/processed/model_comparison_*.csv`):
  - Standard-scaled: Stacking_ET ROC-AUC 0.68, recall 0.71, precision 0.24, F1 0.35; AdaBoost ROC-AUC 0.68, recall 0.80, precision 0.20, F1 0.32.
  - Robust-scaled: Stacking_ET ROC-AUC 0.68, recall 0.72, precision 0.23, F1 0.35; AdaBoost ROC-AUC 0.68, recall 0.80, precision 0.20.
  - Unscaled: AdaBoost ROC-AUC 0.68, recall 0.80 (highest weighted score 0.726).
- Saved artifacts: `models/best_model_unscaled.pkl`, `models/best_model_standard_scaled.pkl`, `models/best_model_robust_scaled.pkl` with matching `model_metadata_*.json` (hyperparameters, thresholds, learning/validation curves, SHAP summaries). Visuals live in `figures/03_Model_Selection/`.

## Phase 4 — 04_API_Dev_and_Deploy (FastAPI + React)
- Backend: FastAPI (`backend/api`) serving the standard-scaled SVM model. Derived features computed automatically (pulse pressure; age_group_code bins: <45, 45-54, 55-64, 65+).
- Endpoints:
  - `GET /` and `GET /health` for liveness/model status.
  - `POST /predict` expects raw clinical inputs (age, systolic_bp, diastolic_bp, bmi, heart_rate, total_cholesterol, glucose, cigarettes_per_day, hypertension) and returns `prediction`, `probability`, and `risk_level` (Low/Medium/High).
- Local run:
  ```bash
  cd backend
  python -m venv .venv && .venv\\Scripts\\activate
  pip install -r requirements.txt
  # regenerate scaler if missing
  python ..\\create_scaler.py
  uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
  ```
- Docker Compose (frontend + backend):
  ```bash
  docker-compose up --build
  # frontend: http://localhost:3000  backend: http://localhost:8000
  ```
- Frontend: `frontend/` React app (Vite + Tailwind). Run with `npm install && npm run dev` (defaults to `http://localhost:5173`, calls API at `http://localhost:8000`). Live demo in the notebook: https://sp-frontend-qtk3.onrender.com (API: https://sp-backend-qkjs.onrender.com).

## How to Reproduce the Analysis
- Open the notebooks in order (01 → 04) to regenerate figures and artifacts. Code utilities live in `src/`.
- Data splits are already materialized; rerun `main.py` or the notebooks if you want to rebuild them from raw.
- Models and metadata are in `models/`; comparison tables in `data/processed/` can be inspected without rerunning training.

## Quick Reference (Files)
- EDA outputs: `figures/01_EDA/*`, `data/processed/cleaned_coronary_disease.csv`, `data/processed/processed_features.csv`.
- Traditional ML: `notebooks/02_ML_Models.ipynb`, `models/best_model__standard_scaled__svm__class_weight.pkl`, `figures/02_ML_Models/*`.
- Model selection: `notebooks/03_Model_Selection.ipynb`, `models/model_metadata_*.json`, `data/processed/model_comparison_*.csv`, `figures/03_Model_Selection/*`.
- API & UI: `backend/api/*`, `frontend/*`, `docker-compose.yml`, `render.yaml`.
