"""
Feature Engineering and Feature Selection (Phase 1)

This module implements interpretable feature engineering and exploratory
feature selection for the Coronary Disease Prediction project.

Assumptions:
- Input dataframe is already cleaned
- Target column is named 'ten_year_chd' or standardized upstream
- No scaling, imputation, or resampling is performed here
"""

import os
import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier


# ---------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------

def standardize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize target column name to 'TenYearCHD'."""
    if "ten_year_chd" in df.columns:
        df = df.rename(columns={"ten_year_chd": "TenYearCHD"})
    if "TenYearCHD" not in df.columns:
        raise ValueError("Target column 'TenYearCHD' not found in dataset")
    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create clinically meaningful derived features."""
    df = df.copy()

    # Pulse pressure
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 45, 55, 65, 120],
        labels=["<45", "45–54", "55–64", "65+"],
        right=False
    )

    # BMI categories
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
        right=False
    )

    # Binary cardiovascular risk factors
    binary_risk_factors = [
        "current_smoker",
        "hypertension",
        "diabetes",
        "previous_stroke"
    ]

    for col in binary_risk_factors:
        df[col] = df[col].astype(int)

    # Unweighted risk factor count
    df["risk_factor_count"] = df[binary_risk_factors].sum(axis=1)

    # Ordinal encoding
    df["age_group_code"] = df["age_group"].cat.codes
    df["bmi_category_code"] = df["bmi_category"].cat.codes

    # Cleanup temporary columns
    df = df.drop(columns=["age_group", "bmi_category"])

    return df


# ---------------------------------------------------------------------
# Multicollinearity Diagnostics (VIF)
# ---------------------------------------------------------------------

def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Variance Inflation Factor (diagnostic only)."""
    vif_features = [
        "age",
        "systolic_bp",
        "diastolic_bp",
        "pulse_pressure",
        "bmi",
        "total_cholesterol",
        "heart_rate",
        "glucose",
        "risk_factor_count"
    ]

    X = df[vif_features].dropna()

    vif_df = pd.DataFrame({
        "feature": X.columns,
        "VIF": [
            variance_inflation_factor(X.values, i)
            for i in range(X.shape[1])
        ]
    }).sort_values("VIF", ascending=False)

    return vif_df


# ---------------------------------------------------------------------
# Feature Selection (Exploratory)
# ---------------------------------------------------------------------

def feature_selection(df: pd.DataFrame):
    """Run correlation, RFE, and tree-based feature importance."""
    candidate_features = [
        "age",
        "sex",
        "pulse_pressure",
        "bmi",
        "total_cholesterol",
        "glucose",
        "heart_rate",
        "risk_factor_count",
        "previous_stroke",
        "age_group_code",
        "bmi_category_code"
    ]

    X = df[candidate_features].copy()
    y = df["TenYearCHD"]

    X = X.dropna()
    y = y.loc[X.index]

    # Correlation with target
    corr_with_target = (
        pd.concat([X, y], axis=1)
        .corr(numeric_only=True)["TenYearCHD"]
        .abs()
        .sort_values(ascending=False)
    )

    # Recursive Feature Elimination
    log_reg = LogisticRegression(max_iter=1000, solver="liblinear")
    rfe = RFE(estimator=log_reg, n_features_to_select=8)
    rfe.fit(X, y)

    rfe_results = pd.DataFrame({
        "feature": X.columns,
        "selected": rfe.support_,
        "ranking": rfe.ranking_
    }).sort_values("ranking")

    # Tree-based importance
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X, y)

    rf_importance = (
        pd.DataFrame({
            "feature": X.columns,
            "importance": rf.feature_importances_
        })
        .sort_values("importance", ascending=False)
    )

    return corr_with_target, rfe_results, rf_importance


# ---------------------------------------------------------------------
# Final Dataset Assembly
# ---------------------------------------------------------------------

def build_final_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble final feature set for downstream modeling."""
    final_features = [
        "age",
        "sex",
        "pulse_pressure",
        "bmi",
        "total_cholesterol",
        "glucose",
        "heart_rate",
        "risk_factor_count",
        "previous_stroke",
        "age_group_code",
        "bmi_category_code"
    ]

    return df[final_features + ["TenYearCHD"]]


def save_final_dataset(df: pd.DataFrame, output_dir="data/processed"):
    """Save final dataset to disk."""
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "processed_features.csv")
    df.to_csv(output_path, index=False)
    return output_path
