from src import (
    # cleaning
    load_data,
    rename_columns,
    convert_dtypes,
    missing_value_summary,
    get_column_types,
    simple_imputation,
    knn_impute_glucose,
    iqr_outlier_summary,
    zscore_outlier_summary,
    # normalization
    split_data,
    create_unscaled_sets,
    standard_scale,
    robust_scale,
    build_scaled_dataframe,
)

# Load data
df = load_data("data/raw/coronary_disease.csv")

# Basic cleaning
df = rename_columns(df)
df = convert_dtypes(df)

# Missing values
missing_df = missing_value_summary(df)

# Column types
numerical_cols, categorical_cols = get_column_types(df)

# Imputation
df = simple_imputation(df, numerical_cols, categorical_cols)
df = knn_impute_glucose(df)

# Outlier summaries
iqr_summary = iqr_outlier_summary(df, numerical_cols)
zscore_summary = zscore_outlier_summary(df, numerical_cols)

# Save outputs
df.to_csv("data/processed/cleaned_coronary_disease.csv", index=False)

print("Data cleaning completed successfully.")
print(f"Final dataset shape: {df.shape}")

# Feature engineering



# Normalization

df = load_data("data/processed/processed_features.csv")

TARGET = "ten_year_chd"

selected_features = [
    "age",
    "age_group",
    "pulse_pressure",
    "bmi",
    "bmi_category",
    "glucose",
    "total_cholesterol",
    "risk_factor_count",
]

# Train-test split
X_train, X_test, y_train, y_test = split_data(
    df=df,
    selected_features=selected_features,
    target=TARGET,
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# Unscaled datasets (for tree-based models)
train_unscaled, test_unscaled = create_unscaled_sets(
    X_train, X_test, y_train, y_test, TARGET
)

train_unscaled.to_csv("data/processed/train_unscaled.csv", index=False)
test_unscaled.to_csv("data/processed/test_unscaled.csv", index=False)

# StandardScaler (linear models, KNN, SVM)
X_train_std, X_test_std, _ = standard_scale(X_train, X_test)

train_standard = build_scaled_dataframe(
    X_train_std,
    y_train,
    selected_features,
    X_train.index,
    TARGET,
)

test_standard = build_scaled_dataframe(
    X_test_std,
    y_test,
    selected_features,
    X_test.index,
    TARGET,
)

train_standard.to_csv(
    "data/processed/train_standard_scaled.csv", index=False
)
test_standard.to_csv(
    "data/processed/test_standard_scaled.csv", index=False
)

# RobustScaler (robust to outliers)
X_train_rob, X_test_rob, _ = robust_scale(X_train, X_test)

train_robust = build_scaled_dataframe(
    X_train_rob,
    y_train,
    selected_features,
    X_train.index,
    TARGET,
)

test_robust = build_scaled_dataframe(
    X_test_rob,
    y_test,
    selected_features,
    X_test.index,
    TARGET,
)

train_robust.to_csv(
    "data/processed/train_robust_scaled.csv", index=False
)
test_robust.to_csv(
    "data/processed/test_robust_scaled.csv", index=False
)

print("Normalization completed successfully.")