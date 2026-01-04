import pandas as pd
import numpy as np


def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Pulse pressure
    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]

    # Smoking intensity
    def smoking_intensity(row):
        if row["current_smoker"] == 0:
            return 0
        elif row["cigarettes_per_day"] <= 10:
            return 1
        elif row["cigarettes_per_day"] <= 20:
            return 2
        else:
            return 3

    df["smoking_intensity"] = df.apply(smoking_intensity, axis=1)

    # Age groups
    df["age_group"] = pd.cut(
        df["age"],
        bins=[29, 39, 49, 59, 69, 120],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # BMI categories
    df["bmi_category"] = pd.cut(
        df["bmi"],
        bins=[0, 18.5, 25, 30, 100],
        labels=[0, 1, 2, 3]
    ).astype(int)

    # Cardiovascular risk factor count (UNWEIGHTED)
    risk_factors = [
        "current_smoker",
        "diabetes",
        "hypertension",
        "bp_medication",
        "previous_stroke"
    ]

    df["risk_factor_count"] = df[risk_factors].sum(axis=1)

    return df
