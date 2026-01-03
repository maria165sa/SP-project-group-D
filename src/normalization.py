# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler

# Train-test split
def split_data(
    df: pd.DataFrame,
    selected_features: list,
    target: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split data into train and test sets."""
    X = df[selected_features]
    y = df[target]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

# Unscaled datasets (for tree-based models)
def create_unscaled_sets(X_train, X_test, y_train, y_test, target: str):
    """Create unscaled train and test datasets."""
    train_unscaled = X_train.copy()
    train_unscaled[target] = y_train

    test_unscaled = X_test.copy()
    test_unscaled[target] = y_test

    return train_unscaled, test_unscaled

# StandardScaler (linear models, KNN, SVM)
def standard_scale(X_train, X_test):
    """Apply StandardScaler normalization."""
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler

# RobustScaler (robust to outliers)
def robust_scale(X_train, X_test):
    """Apply RobustScaler normalization."""
    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def build_scaled_dataframe(
    X_scaled,
    y,
    feature_names: list,
    index,
    target: str,
):
    """Combine scaled features and target into a DataFrame."""
    df_scaled = pd.DataFrame(
        X_scaled,
        columns=feature_names,
        index=index,
    )
    df_scaled[target] = y

    return df_scaled