"""
Utility helpers for notebook-driven traditional ML experiments.

"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    HAS_IMBLEARN = True
except ImportError:
    SMOTE = None
    ImbPipeline = None
    HAS_IMBLEARN = False


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(csv_path)


def prepare_features(
    df: pd.DataFrame, feature_cols: Sequence[str], target_col: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a dataframe into X/y using the provided feature and target names."""
    missing = [col for col in list(feature_cols) + [target_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    X = df[list(feature_cols)].copy()
    y = df[target_col].copy()
    return X, y


def make_preprocessor(
    feature_cols: Sequence[str], scaler: Optional[StandardScaler] = None
) -> ColumnTransformer:
    """Create a ColumnTransformer with optional scaling for numeric features."""
    numeric_features = list(feature_cols)
    steps = []
    if scaler is not None:
        steps.append(("scaler", scaler))
    transformer = Pipeline(steps) if steps else "passthrough"
    return ColumnTransformer([("num", transformer, numeric_features)])


# ---------------------------------------------------------------------------
# Model builders and search spaces
# ---------------------------------------------------------------------------

def get_estimators(random_state: int = 42) -> Dict[str, object]:
    """Return base estimators (no imbalance handling applied)."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, solver="liblinear", random_state=random_state
        ),
        "knn": KNeighborsClassifier(),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=random_state, n_jobs=-1
        ),
        "svm": SVC(probability=True, random_state=random_state),
    }


def build_pipeline(
    preprocessor: ColumnTransformer,
    estimator,
    use_smote: bool = False,
    random_state: int = 42,
    allow_smote: bool = HAS_IMBLEARN,
):
    """Wrap a preprocessor and estimator (optionally with SMOTE) into a pipeline."""
    prep = clone(preprocessor)
    model = clone(estimator)

    if use_smote:
        if not (allow_smote and HAS_IMBLEARN):
            raise ValueError("SMOTE requested but imblearn is not available.")
        return ImbPipeline(
            [
                ("preprocessor", prep),
                ("smote", SMOTE(random_state=random_state)),
                ("model", model),
            ]
        )

    return Pipeline([("preprocessor", prep), ("model", model)])


def get_search_spaces(random_state: int = 42) -> Dict[str, Dict]:
    """Hyperparameter grids/distributions keyed by model name."""
    return {
        "logistic_regression": {
            "search_type": "grid",
            "params": {"model__C": [0.1, 1, 10], "model__penalty": ["l1", "l2"]},
        },
        "knn": {
            "search_type": "grid",
            "params": {
                "model__n_neighbors": [3, 5, 7],
                "model__weights": ["uniform", "distance"],
                "model__p": [2],
            },
        },
        "decision_tree": {
            "search_type": "grid",
            "params": {
                "model__max_depth": [None, 10],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1],
                "model__criterion": ["gini", "entropy"],
            },
        },
        "random_forest": {
            "search_type": "random",
            "n_iter": 4,
            "params": {
                "model__n_estimators": [200, 300],
                "model__max_depth": [None, 10],
                "model__min_samples_split": [2, 5],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt"],
            },
        },
        "svm": {
            "search_type": "random",
            "n_iter": 4,
            "params": {
                "model__C": [0.5, 1, 5],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["rbf", "linear"],
            },
        },
    }


# ---------------------------------------------------------------------------
# Training, tuning, and evaluation helpers
# ---------------------------------------------------------------------------

def get_cv(n_splits: int = 5, random_state: int = 42) -> StratifiedKFold:
    """Create a StratifiedKFold splitter."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)


def tune_model(
    estimator,
    param_space: Dict,
    X_train,
    y_train,
    cv,
    scoring: str = "roc_auc",
    search_type: str = "grid",
    n_iter: Optional[int] = None,
    random_state: int = 42,
):
    """Run GridSearchCV or RandomizedSearchCV and return the fitted search object."""
    if search_type == "random":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_space,
            n_iter=n_iter or 10,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            random_state=random_state,
            refit=True,
        )
    else:
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_space,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
    search.fit(X_train, y_train)
    return search


def predict_proba_positive(estimator, X):
    """Return positive-class probabilities or scaled decision scores."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)[:, 1]
    if hasattr(estimator, "decision_function"):
        scores = estimator.decision_function(X)
        return (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    return None


def evaluate_predictions(y_true, y_pred, y_proba=None) -> Dict[str, float]:
    """Compute standard classification metrics."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": np.nan,
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def compute_roc_curve(y_true, y_proba):
    """Return ROC curve components and AUC, or None if probabilities are missing."""
    if y_proba is None:
        return None
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    return {"fpr": fpr, "tpr": tpr, "thresholds": thresholds, "auc": auc}


def compute_pr_curve(y_true, y_proba):
    """Return precision-recall curve components, or None if probabilities are missing."""
    if y_proba is None:
        return None
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    return {"precision": precision, "recall": recall, "thresholds": thresholds}


# ---------------------------------------------------------------------------
# Plot helpers (no saving side effects)
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true, y_pred, title: Optional[str] = None, cmap: str = "Blues"):
    """Create a confusion matrix plot; caller decides whether to save/close."""
    disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap)
    if title:
        disp.ax_.set_title(title)
    return disp.figure_, disp.ax_


def plot_roc_curve(y_true, y_proba, title: Optional[str] = None, label: Optional[str] = None):
    """Create a ROC curve plot; returns None if probabilities are missing."""
    if y_proba is None:
        return None
    disp = RocCurveDisplay.from_predictions(
        y_true, y_proba, name=label if label else None
    )
    if title:
        disp.ax_.set_title(title)
    return disp.figure_, disp.ax_


def plot_precision_recall_curve(
    y_true, y_proba, title: Optional[str] = None, label: Optional[str] = None
):
    """Create a precision-recall plot; returns None if probabilities are missing."""
    if y_proba is None:
        return None
    disp = PrecisionRecallDisplay.from_predictions(
        y_true, y_proba, name=label if label else None
    )
    if title:
        disp.ax_.set_title(title)
    return disp.figure_, disp.ax_


__all__ = [
    "HAS_IMBLEARN",
    "load_dataset",
    "prepare_features",
    "make_preprocessor",
    "get_estimators",
    "build_pipeline",
    "get_search_spaces",
    "get_cv",
    "tune_model",
    "predict_proba_positive",
    "evaluate_predictions",
    "compute_roc_curve",
    "compute_pr_curve",
    "plot_confusion_matrix",
    "plot_roc_curve",
    "plot_precision_recall_curve",
]
