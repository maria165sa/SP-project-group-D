"""
Script to create the StandardScaler using the project's existing code.

This script reuses the standard_scale() function from src/normalization.py
to ensure 100% consistency with the training pipeline.
"""

import sys
from pathlib import Path
import joblib
import pandas as pd

# Add project root to path to import from src
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.normalization import standard_scale

# Load unscaled training data
print("Loading training data...")
train_data = pd.read_csv("data/processed/train_unscaled.csv")

# Features used by the model
features = [
    'bmi',
    'hypertension',
    'pulse_pressure',
    'cigarettes_per_day',
    'total_cholesterol',
    'glucose',
    'heart_rate',
    'age_group_code'
]

# Separate features and target
X_train = train_data[features]
y_train = train_data['TenYearCHD']

print(f"Training data shape: {X_train.shape}")

# Use the SAME function they used in main.py
print("\nCreating scaler using standard_scale() from src/normalization.py...")
_, _, scaler = standard_scale(X_train, X_train)  
# Note: We pass X_train twice because we only need the scaler
# The function returns (X_train_scaled, X_test_scaled, scaler)
# We discard the first two with _

# Save scaler
scaler_path = Path("models/scaler_standard.pkl")
joblib.dump(scaler, scaler_path)

print(f"\n✅ Scaler created and saved to: {scaler_path}")
print(f"\nScaler statistics:")
print(f"  Features: {len(scaler.mean_)}")
print(f"  Mean values: {scaler.mean_}")
print(f"  Scale values: {scaler.scale_}")