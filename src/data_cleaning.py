# Import necessary libraries 
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.impute import KNNImputer


# Load data
def load_data(path: str) -> pd.DataFrame:
    """Load raw coronary disease data."""
    return pd.read_csv(path)

# Rename columns
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns for consistency."""
    return df.rename(
        columns={
            "education": "education_level",
            "currentSmoker": "current_smoker",
            "cigsPerDay": "cigarettes_per_day",
            "BPMeds": "bp_medication",
            "prevalentStroke": "previous_stroke",
            "prevalentHyp": "hypertension",
            "totChol": "total_cholesterol",
            "sysBP": "systolic_bp",
            "diaBP": "diastolic_bp",
            "BMI": "bmi",
            "heartRate": "heart_rate",
            "TenYearCHD": "ten_year_chd",
        }
    )

# Change variable types
def convert_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Convert columns to appropriate data types."""
    df = df.copy()

    df["education_level"] = df["education_level"].astype("Int64")
    df["sex"] = df["sex"].map({"M": 1, "F": 0}).astype("int")
    df["current_smoker"] = df["current_smoker"].map({"Yes": 1, "No": 0}).astype("int")
    df["bp_medication"] = df["bp_medication"].astype("Int64")
    df["previous_stroke"] = df["previous_stroke"].astype("int")
    df["hypertension"] = df["hypertension"].astype("int")
    df["diabetes"] = df["diabetes"].astype("int")

    return df


# Obtain missing values summary
def missing_value_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing value count and percentage per column."""
    missing_count = df.isnull().sum()
    missing_percent = missing_count / len(df) * 100

    return (
        pd.DataFrame(
            {
                "missing_values": missing_count,
                "percentage": missing_percent,
            }
        )
        .sort_values("missing_values", ascending=False)
    )

# Separate numerical and categorical columns
def get_column_types(df: pd.DataFrame):
    """Separate numerical and categorical numeric columns."""
    numerical_cols = ['age', 'cigarettes_per_day', 'total_cholesterol',
                   'systolic_bp', 'diastolic_bp', 'bmi', 'heart_rate', 'glucose']

    categorical_cols = ['sex', 'education_level','current_smoker', 'bp_medication', 'previous_stroke',
                    'hypertension', 'diabetes', 'ten_year_chd']

    return numerical_cols, categorical_cols

# Replace missing values
def simple_imputation(
    df: pd.DataFrame,
    numerical_cols: list,
    categorical_cols: list,
    threshold: float = 0.05,
) -> pd.DataFrame:
    """
    Impute columns with missing rate <5%.
    Numerical → median
    Categorical → mode
    """
    df = df.copy()

    for col in numerical_cols:
        if df[col].isnull().mean() < threshold:
            df[col] = df[col].fillna(df[col].median())

    for col in categorical_cols:
        if df[col].isnull().mean() < threshold:
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

# Replace missing values
def knn_impute_glucose(df: pd.DataFrame) -> pd.DataFrame:
    """Impute glucose using KNN."""
    df = df.copy()

    predictors = [
        "glucose",
        "age",
        "bmi",
        "systolic_bp",
        "diastolic_bp",
        "total_cholesterol",
        "heart_rate",
        "cigarettes_per_day",
    ]

    imputer = KNNImputer(n_neighbors=5)
    imputed = imputer.fit_transform(df[predictors])

    df["glucose"] = imputed[:, 0]
    return df

# Outliers detection
def outliers_iqr(series: pd.Series) -> pd.Series:
    """Return outliers using the IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    return series[(series < lower) | (series > upper)]

# IQR method summary
def iqr_outlier_summary(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Summarize IQR-based outliers."""
    summary = {}

    for col in numerical_cols:
        outliers = outliers_iqr(df[col])
        summary[col] = {
            "count": len(outliers),
            "percentage": len(outliers) / len(df) * 100,
        }

    return pd.DataFrame(summary).T.sort_values("count", ascending=False)

# Z-score method summary
def zscore_outlier_summary(df: pd.DataFrame, numerical_cols: list) -> pd.DataFrame:
    """Summarize Z-score-based outliers (>3)."""
    summary = {}

    for col in numerical_cols:
        z = np.abs(stats.zscore(df[col]))
        count = (z > 3).sum()

        summary[col] = {
            "count": count,
            "percentage": count / len(df) * 100,
        }

    return pd.DataFrame(summary).T.sort_values("count", ascending=False)