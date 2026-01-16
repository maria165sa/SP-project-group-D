"""
Utility helpers for notebook-driven Model Selection decision from testing traditional ML, boosting and stacked ensemble models.

"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import joblib
import warnings
import time
warnings.filterwarnings('ignore')

# Model libraries
from sklearn.model_selection import (
    StratifiedKFold,
    RepeatedStratifiedKFold,
    cross_validate,
    learning_curve,
    validation_curve
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    precision_recall_curve
)
from sklearn.base import clone

# Ensemble models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    StackingClassifier,
    RandomForestClassifier, 
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Hyperparameter tuning
import optuna
from optuna.samplers import TPESampler

# Interpretability
import shap

RANDOM_STATE = 42

# 1. DATA LOADING AND PREPARATION

def load_presplit_data(train_filepath,
                       test_filepath,
                       target_col_name='TenYearCHD'):
    """
    Load pre-split train and test datasets
    """
    print("LOADING PRESPLIT DATA")
    
    # Load training data
    print(f"\nLoading training data from: {train_filepath}")
    train_df = pd.read_csv(train_filepath)
    print(f"  Shape: {train_df.shape}")
    
    # Load test data
    print(f"\nLoading test data from: {test_filepath}")
    test_df = pd.read_csv(test_filepath)
    print(f"  Shape: {test_df.shape}")
    
    # Find target column
    if target_col_name not in train_df.columns:
        possible_targets = ['TenYearCHD', 'target', 'ten_year_chd']
        target_found = False
        for col in possible_targets:
            if col in train_df.columns:
                target_col_name = col
                target_found = True
                print(f"\nTarget column found: {target_col_name}")
                break
        
        if not target_found:
            raise ValueError(f"Target column '{target_col_name}' not found. Available columns: {train_df.columns.tolist()}")
    else:
        print(f"\nTarget column: {target_col_name}")
    
    # Separate features and target
    X_train = train_df.drop(columns=[target_col_name])
    y_train = train_df[target_col_name]
    
    X_test = test_df.drop(columns=[target_col_name])
    y_test = test_df[target_col_name]
    
    print(f"\nData split information:")
    print(f"  - Training set: {X_train.shape}")
    print(f"  - Test set: {X_test.shape}")
    
    # Verify feature consistency
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError("Training and test sets have different features!")
    
    print(f"\nFeature consistency verified")
    
    # Class distribution
    train_dist = y_train.value_counts().sort_index()
    print(f"\nTraining Set Class Distribution:")
    print(f"  - Class 0 (No CHD): {train_dist[0]} ({train_dist[0]/len(y_train)*100:.2f}%)")
    print(f"  - Class 1 (CHD): {train_dist[1]} ({train_dist[1]/len(y_train)*100:.2f}%)")
    print(f"  - Imbalance Ratio: {train_dist[0]/train_dist[1]:.2f}:1")
    
    test_dist = y_test.value_counts().sort_index()
    print(f"\nTest Set Class Distribution:")
    print(f"  - Class 0 (No CHD): {test_dist[0]} ({test_dist[0]/len(y_test)*100:.2f}%)")
    print(f"  - Class 1 (CHD): {test_dist[1]} ({test_dist[1]/len(y_test)*100:.2f}%)")
    print(f"  - Imbalance Ratio: {test_dist[0]/test_dist[1]:.2f}:1")
    
    # Handle missing values
    train_missing = X_train.isnull().sum()
    test_missing = X_test.isnull().sum()
    
    if train_missing.any() or test_missing.any():
        print(f"\nWarning: Missing values detected")
        for col in X_train.columns:
            if X_train[col].dtype in ['float64', 'int64']:
                median_val = X_train[col].median()
                X_train[col].fillna(median_val, inplace=True)
                X_test[col].fillna(median_val, inplace=True)
            else:
                mode_val = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else X_train[col].iloc[0]
                X_train[col].fillna(mode_val, inplace=True)
                X_test[col].fillna(mode_val, inplace=True)
    
    # Handle categorical variables
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if categorical_cols:
        print(f"\n - Encoding categorical variables: {categorical_cols}")
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            X_test[col] = le.transform(X_test[col].astype(str))
            label_encoders[col] = le
    else:
        label_encoders = None
        print(f"\n - No categorical variables found (all numeric)")
    
    feature_names = X_train.columns.tolist()
    print(f"\n - Features ({len(feature_names)}):")
    for i, feat in enumerate(feature_names, 1):
        print(f"  {i:2d}. {feat}")
    
    return X_train, X_test, y_train, y_test, feature_names, label_encoders


# 2. HYPERPARAMETER TUNING WITH OPTUNA

XGB_PARAM_SPACE = {
    'n_estimators': lambda t: t.suggest_int('n_estimators', 200, 600),
    'max_depth': lambda t: t.suggest_int('max_depth', 3, 6),
    'learning_rate': lambda t: t.suggest_float('learning_rate', 0.01, 0.1),
    'subsample': lambda t: t.suggest_float('subsample', 0.7, 0.9),
    'colsample_bytree': lambda t: t.suggest_float('colsample_bytree', 0.7, 0.9),
    'min_child_weight': lambda t: t.suggest_int('min_child_weight', 1, 10),
    'gamma': lambda t: t.suggest_float('gamma', 0, 5),
    'reg_alpha': lambda t: t.suggest_float('reg_alpha', 0, 5),
    'reg_lambda': lambda t: t.suggest_float('reg_lambda', 0.5, 5)
}

LGBM_PARAM_SPACE = {
    'n_estimators': lambda t: t.suggest_int('n_estimators', 200, 800),
    'num_leaves': lambda t: t.suggest_int('num_leaves', 20, 80),
    'learning_rate': lambda t: t.suggest_float('learning_rate', 0.01, 0.2),
    'subsample': lambda t: t.suggest_float('subsample', 0.6, 1.0),
    'colsample_bytree': lambda t: t.suggest_float('colsample_bytree', 0.3, 1.0),
    'min_data_in_leaf': lambda t: t.suggest_int('min_data_in_leaf', 20, 100),
    'lambda_l1': lambda t: t.suggest_float('lambda_l1', 0, 5),
    'lambda_l2': lambda t: t.suggest_float('lambda_l2', 0, 5)
}

def tune_boosting_model(model_class, param_space, X_train, y_train, n_trials=50, model_name="Model"):
    """
    Optimize XGBoost or LightGBM hyperparameters using Optuna.
    """
    print(f"HYPERPARAMETER TUNING - {model_name}")
    
    # Calculate scale_pos_weight
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\nCalculated scale_pos_weight: {scale_pos_weight:.2f}")
    
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
    
    def objective(trial):
        params = {k: v(trial) for k, v in param_space.items()}
        
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            model = model_class(**params)
            model.fit(X_tr, y_tr)
            
            probs = model.predict_proba(X_val)[:, 1]
            preds = (probs >= 0.5).astype(int)
            
            roc = roc_auc_score(y_val, probs)
            recall = recall_score(y_val, preds)
            pr_auc = average_precision_score(y_val, probs)

            scores.append(0.6*roc + 0.4*recall)
        
        return np.mean(scores)
    
    print(f"\nStarting Optuna optimization ({n_trials} trials)...")
    print("  Objective: 0.6*ROC-AUC + 0.4*Recall")
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials)
    
    print(f"\nOptimization completed.")
    print(f"  - Best ROC-AUC: {study.best_value:.4f}")
    print(f"  - Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    best_params = study.best_params.copy()

    best_params.update({
        'eval_metric': 'auc',
        'random_state': RANDOM_STATE
    })

    if model_class == XGBClassifier:
        best_params.update({'scale_pos_weight': scale_pos_weight, 'use_label_encoder': False})
    elif model_class == LGBMClassifier:
        best_params.update({'class_weight':'balanced', 'verbose': -1})
        
    return best_params


def tune_adaboost(X_train, y_train, n_trials=40):
    """
    Optimize AdaBoost hyperparameters using Optuna
    Uses sample_weight to handle imbalance
    """
    print("HYPERPARAMETER TUNING - AdaBoost")

    # Calculate sample weights for imbalance
    class_counts = y_train.value_counts()
    class_weight = len(y_train) / (2 * class_counts)
    sample_weights = y_train.map(class_weight)
    
    print(f"\nUsing sample_weight to handle imbalance")
    print(f"  - Class 0 weight: {class_weight[0]:.3f}")
    print(f"  - Class 1 weight: {class_weight[1]:.3f}")
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 400),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 1.0),
            'algorithm': 'SAMME',  # Better for probability estimates
            'random_state': RANDOM_STATE,
        }
        
        model = AdaBoostClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        
        scores = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            sw_tr = sample_weights.iloc[train_idx]
            
            # Fit with sample weights
            model.fit(X_tr, y_tr, sample_weight=sw_tr)
            y_pred_proba = model.predict_proba(X_val)[:, 1]
            score = roc_auc_score(y_val, y_pred_proba)
            scores.append(score)
        
        return np.mean(scores)
    
    print(f"\nStarting Optuna optimization ({n_trials} trials)...")
    print("  - Objective: 0.6*ROC-AUC + 0.4*Recall")
    
    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nOptimization completed!")
    print(f"  - Best ROC-AUC: {study.best_value:.4f}")
    print(f"  - Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
    
    best_params = study.best_params.copy()
    best_params['random_state'] = RANDOM_STATE
    
    return best_params


# 3. MODEL TRAINING AND EVALUATION

def train_base_models(X_train, y_train, X_test, y_test, 
                     best_xgb_params, best_lgbm_params, best_ada_params):
    """
    Train base ensemble models with optimized hyperparameters
    """
    print("TRAINING BASE ENSEMBLE MODELS")
    
    models = {
        'XGBoost': XGBClassifier(**best_xgb_params),
        'LightGBM': LGBMClassifier(**best_lgbm_params),
        'AdaBoost': AdaBoostClassifier(**best_ada_params)
    }
    
    results = {}
    trained_models = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"Training: {name}")
        print(f"{'='*80}")
        
        # Cross-validation
        print(f"  Running 5-fold cross-validation...")

        if name == 'AdaBoost':
            # For AdaBoost, we custom CV with sample_weight
            cv_scores = {'test_roc_auc': [], 'test_recall': [], 'test_precision': [], 
                        'test_f1': [], 'test_accuracy': []}
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

                # Calculate sample weights for imbalance
            class_counts = y_train.value_counts()
            class_weight = len(y_train) / (2 * class_counts)
            sample_weights = y_train.map(class_weight)
            for train_idx, val_idx in cv.split(X_train, y_train):
                X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                sw_tr = sample_weights.iloc[train_idx]
                
                model.fit(X_tr, y_tr, sample_weight=sw_tr)
                y_pred = model.predict(X_val)
                y_proba = model.predict_proba(X_val)[:, 1]
                
                cv_scores['test_roc_auc'].append(roc_auc_score(y_val, y_proba))
                cv_scores['test_recall'].append(recall_score(y_val, y_pred))
                cv_scores['test_precision'].append(precision_score(y_val, y_pred, zero_division=0))
                cv_scores['test_f1'].append(f1_score(y_val, y_pred))
                cv_scores['test_accuracy'].append(accuracy_score(y_val, y_pred))
            
            cv_results = {k: np.array(v) for k, v in cv_scores.items()}
            
            # Train final model with sample weights
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            cv_results = cross_validate(
                model, X_train, y_train,
                cv=cv, scoring=scoring, n_jobs=-1,
                return_train_score=True
            )
            model.fit(X_train, y_train)
        
        # Predictions and inference speed
        import time
        start_time = time.time()
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test)[:, 1]
        inference_time = time.time() - start_time
        inference_time_per_sample = inference_time / len(X_test)

        y_train_pred = model.predict(X_train)
        
        # Calculate metrics
        train_metrics = {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'precision': precision_score(y_train, y_train_pred, zero_division=0),
            'recall': recall_score(y_train, y_train_pred),
            'f1': f1_score(y_train, y_train_pred),
            'roc_auc': roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        }
        
        test_metrics = {
            'accuracy': accuracy_score(y_test, y_test_pred),
            'precision': precision_score(y_test, y_test_pred, zero_division=0),
            'recall': recall_score(y_test, y_test_pred),
            'f1': f1_score(y_test, y_test_pred),
            'roc_auc': roc_auc_score(y_test, y_test_proba),
            'inference_time': inference_time,
            'inference_time_per_sample': inference_time_per_sample
        }
        
        cv_metrics = {
            'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
            'cv_accuracy_std': cv_results['test_accuracy'].std(),
            'cv_precision_mean': cv_results['test_precision'].mean(),
            'cv_precision_std': cv_results['test_precision'].std(),
            'cv_recall_mean': cv_results['test_recall'].mean(),
            'cv_recall_std': cv_results['test_recall'].std(),
            'cv_f1_mean': cv_results['test_f1'].mean(),
            'cv_f1_std': cv_results['test_f1'].std(),
            'cv_roc_auc_mean': cv_results['test_roc_auc'].mean(),
            'cv_roc_auc_std': cv_results['test_roc_auc'].std(),
        }
        
        # Store results
        results[name] = {
            'train_metrics': train_metrics,
            'metrics': test_metrics,
            'cv_metrics': cv_metrics,
            'confusion_matrix': confusion_matrix(y_test, y_test_pred),
            'y_pred': y_test_pred,
            'y_proba': y_test_proba
        }
        
        trained_models[name] = model
        
        # Print results
        print(f"\n  Cross-Validation Results (mean ± std):")
        print(f"    Accuracy:  {cv_metrics['cv_accuracy_mean']:.4f} ± {cv_metrics['cv_accuracy_std']:.4f}")
        print(f"    Precision: {cv_metrics['cv_precision_mean']:.4f} ± {cv_metrics['cv_precision_std']:.4f}")
        print(f"    Recall:    {cv_metrics['cv_recall_mean']:.4f} ± {cv_metrics['cv_recall_std']:.4f}")
        print(f"    F1-Score:  {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
        print(f"    ROC-AUC:   {cv_metrics['cv_roc_auc_mean']:.4f} ± {cv_metrics['cv_roc_auc_std']:.4f}")
        
        print(f"\n  Test Set Performance:")
        print(f"    Accuracy:  {test_metrics['accuracy']:.4f}")
        print(f"    Precision: {test_metrics['precision']:.4f}")
        print(f"    Recall:    {test_metrics['recall']:.4f}")
        print(f"    F1-Score:  {test_metrics['f1']:.4f}")
        print(f"    ROC-AUC:   {test_metrics['roc_auc']:.4f}")
    
    return results, trained_models

def create_base_models(pos_weight):
    """
    Create base learners with reasonably good defaults.
    Returns a dict of instantiated models.
    """
    base_models = dict()
    
    base_models['xgb'] = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_lambda=1.0,
        reg_alpha=0.1,
        scale_pos_weight=pos_weight,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    base_models['lgbm'] = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.08,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.1,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    base_models['rf'] = RandomForestClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        class_weight='balanced_subsample',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    base_models['et'] = ExtraTreesClassifier(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=10,
        min_samples_split=20,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    base_models['svm'] = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )
    
    return base_models

def create_stacking_ensemble(base_model_keys, X_train, y_train):
    """
    General function to create a stacking ensemble from a selection of base models.
    """
    base_models = create_base_models(pos_weight=(len(y_train) - sum(y_train)) / sum(y_train))
    
    estimators = [(name, base_models[name]) for name in base_model_keys]

    meta_learner = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        penalty='l2',
        solver='lbfgs',
        random_state=RANDOM_STATE
    )

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_learner,
        stack_method='predict_proba',
        cv=5,
        n_jobs=-1,
        passthrough=False
    )

    print("\nTraining stacking ensemble...")
    print(f"  Base estimators: {[name for name, _ in estimators]}")
    print(f"  Meta-learner: Logistic Regression")
    
    stacking_model.fit(X_train, y_train)

    print("Stacking ensemble trained successfully.")
    
    return stacking_model

# Wrapper functions for the two ensembles

def create_stacking_xgb_lgbm_rf_et(X_train, y_train):
    return create_stacking_ensemble(['xgb', 'lgbm', 'rf', 'et'], X_train, y_train)

def create_stacking_xgb_lgbm_rf_svm(X_train, y_train):
    return create_stacking_ensemble(['xgb', 'lgbm', 'rf', 'svm'], X_train, y_train)

def evaluate_model(model, X_test, y_test, threshold):
    """
    Comprehensive model evaluation
    """
    import time
    
    # Measure inference time
    start_time = time.time()
    probs = model.predict_proba(X_test)[:, 1]
    inference_time = time.time() - start_time
    inference_time_per_sample = inference_time / len(X_test)

    # Apply threshold
    preds = (probs >= threshold).astype(int)

    # Calculate all metrics
    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds),
        'f1': f1_score(y_test, preds),
        'pr_auc': average_precision_score(y_test, probs),
        'roc_auc': roc_auc_score(y_test, probs),
        "threshold": threshold,
        'inference_time': inference_time,
        'inference_time_per_sample': inference_time_per_sample
    }
    
    cm = confusion_matrix(y_test, preds)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'y_pred': preds,
        'y_proba': probs
    }

    
# 4. THRESHOLD OPTIMIZATION

# 4. THRESHOLD OPTIMIZATION

def optimize_threshold_for_recall(model, X, y, min_recall=0.65):
    """
    Optimize classification threshold for medical diagnosis.
    Strategy:
    1. Calculate F2-score (emphasizes recall over precision)
    2. Find thresholds with recall >= min_recall
    3. Among those, pick the one that maximizes F2-score
    4. If no threshold achieves min_recall, maximize F2-score anyway
    
    Args:
        model: Trained classifier with predict_proba method
        X: Feature matrix
        y: True labels
        min_recall: Minimum acceptable recall (default 0.65)
    
    Returns:
        optimal_threshold: Selected threshold
        optimal_recall: Recall achieved at that threshold
    """
    probs = model.predict_proba(X)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y, probs)
    
    # Remove last element (recall=1, precision=class_ratio at threshold=0)
    precision = precision[:-1]
    recall = recall[:-1]
    
    # Calculate F2-score (emphasizes recall: β=2 means recall is 2x more important)
    f2_scores = (5 * precision * recall) / (4 * precision + recall + 1e-8)
    
    # Find thresholds with recall >= min_recall
    high_recall_mask = recall >= min_recall
    
    if high_recall_mask.any():
        # Among high-recall thresholds, maximize F2-score
        high_recall_indices = np.where(high_recall_mask)[0]
        best_idx = high_recall_indices[np.argmax(f2_scores[high_recall_indices])]
        strategy = f"F2-maximization with min_recall={min_recall}"
        print(f"  Found threshold achieving recall >= {min_recall:.2f}")
    else:
        # Fallback: maximize F2-score without constraint
        best_idx = np.argmax(f2_scores)
        strategy = "F2-maximization (min_recall not achievable)"
        print(f" Could not achieve recall >= {min_recall:.2f}")
        print(f" Using threshold that maximizes F2-score")
    
    optimal_threshold = thresholds[best_idx]
    
    # Validate on the dataset
    y_pred = (probs >= optimal_threshold).astype(int)
    actual_recall = recall_score(y, y_pred)
    actual_precision = precision_score(y, y_pred, zero_division=0)
    
    print(f"  Strategy: {strategy}")
    print(f"  Selected threshold: {optimal_threshold:.4f}")
    print(f"  Recall:    {actual_recall:.4f}")
    print(f"  Precision: {actual_precision:.4f}")
    print(f"  F2-score:  {f2_scores[best_idx]:.4f}")
    
    return optimal_threshold, actual_recall



# 5. VISUALIZATION

def plot_roc_curves(results_dict, y_test, scaling_type='unscaled', save_dir='../figures/03_Model_Selection'):
    """
    Plot ROC curves for all models
    """
    filename = f'roc_curve_{scaling_type}.png'
    plt.figure(figsize=(10, 8))

    for name, results in results_dict.items():
        if 'y_proba' not in results:
            raise KeyError(f"Missing y_proba for model '{name}'")
            
        fpr, tpr, _ = roc_curve(y_test, results['y_proba'])
        auc = results['metrics']['roc_auc']
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})", linewidth=2)

    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f'ROC Curves - Model Comparison ({scaling_type})', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ ROC curves saved: {filename}")


def plot_confusion_matrices(results_dict, scaling_type='unscaled', save_dir='../figures/03_Model_Selection'):
    """
    Plot confusion matrices for all models
    """
    filename = f'confusion_matrices_{scaling_type}.png'
    n_models = len(results_dict)
    fig, axes = plt.subplots(1, n_models, figsize=(5*n_models, 4))
    axes = np.atleast_1d(axes)

    for ax, (name, results) in zip(axes, results_dict.items()):
        cm = results['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', ax=ax, cmap='Blues',
                    xticklabels=['No CHD', 'CHD'],
                   yticklabels=['No CHD', 'CHD'])
        ax.set_title(f"{name}", fontweight='bold')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices saved: {filename}")


# 6. FEATURE IMPORTANCE AND INTERPRETABILITY

def analyze_feature_importance(model, feature_names, model_name, scaling_type='unscaled', save_dir='../figures/03_Model_Selection'):
    """
    Analyze and plot feature importance
    """
    print(f"\n{'='*80}")
    print(f"Feature Importance Analysis - {model_name}")
    print(f"{'='*80}")

    filename = f'feature_importance_{model_name.lower().replace(" ", "_")}_{scaling_type}.png'

    # CASE 1: Single tree-based model
    if hasattr(model, 'feature_importances_'):
        # Build and sort DataFrame
        importance_df = (
            pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            })
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
        )

    # CASE 2: Stacked ensemble
    elif hasattr(model, "final_estimator_"):
        print("✓ Detected stacked ensemble model")

        # Attempt to get base model names
        if hasattr(model, "named_estimators_"):
            base_names = list(model.named_estimators_.keys())
        elif hasattr(model, "base_estimators_"):  # custom wrapper
            base_names = [f"model_{i}" for i in range(len(model.base_estimators_))]
        elif hasattr(model, "estimators_"):      # fallback
            base_names = [f"model_{i}" for i in range(len(model.estimators_))]
        else:
            base_names = [f"model_{i}" for i in range(10)]  # fallback

        meta_model = model.final_estimator_

        if not hasattr(meta_model, "coef_"):
            print("Meta-model does not expose coefficients.")
            return pd.DataFrame()

        # Use absolute value of coefficients as importance
        importances = np.abs(meta_model.coef_).ravel()
        meta_feature_names = [f"{name}_pred" for name in base_names]

        importance_df = (
            pd.DataFrame({
                'feature': meta_feature_names,
                'importance': importances
            })
            .sort_values('importance', ascending=False)
            .reset_index(drop=True)
        )
    
    # CASE 3: Unsupported model
    else:
        print(f"{model_name} has no supported feature importance method")
        return pd.DataFrame()
        
    # Print top 10
    print(f"\nTop 10 Most Important Features:")
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")

    # Plot top N
    top_n = min(15, len(importance_df))
    y_pos = np.arange(top_n)
    
    plt.figure(figsize=(10, 8))
    plt.barh(y_pos, importance_df['importance'].head(top_n))
    plt.yticks(y_pos, importance_df['feature'].head(top_n))
    plt.gca().invert_yaxis()
    plt.title(f'Feature Importance - {model_name} ({scaling_type})', fontsize=14, fontweight='bold')
    plt.xlabel('Importance', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Feature importance plot saved: {filename}")

    return importance_df


def shap_analysis(model, X_test, feature_names, model_name, scaling_type='unscaled', max_display=15, save_dir='../figures/03_Model_Selection'):
    """
    SHAP analysis for model interpretability
    
    Handles:
    - Tree-based models: XGBoost, LightGBM, AdaBoost
    - Stacking ensembles with heterogeneous base models
    
    For stacking ensembles:
    - Uses KernelExplainer to explain the final stacked predictions
    - Treats the entire ensemble as a black box
    - Captures the combined effect of all base models + meta-learner
    """
    print(f"SHAP Analysis - {model_name}")
    
    try:
        # Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f'shap_summary_{model_name.lower().replace(" ", "_")}_{scaling_type}.png'
        
        print("\nCreating SHAP explainer...")
        
        # Detect model type and choose appropriate explainer
        is_stacking = 'stack' in model_name.lower() or 'ensemble' in model_name.lower()
        
        if is_stacking:
            # For stacking models: always use KernelExplainer
            print(f"  Detected stacking ensemble - using KernelExplainer")
            print(f"  This explains the final ensemble predictions (all base models + meta-learner)")
            
            # Use a small background dataset for efficiency
            background_size = min(100, len(X_test))
            X_background = shap.sample(X_test, background_size, random_state=RANDOM_STATE)
            
            # Create prediction function for probability outputs
            def model_predict(X):
                if hasattr(model, 'predict_proba'):
                    return model.predict_proba(X)[:, 1]  # Return probability of positive class
                else:
                    return model.predict(X)
            
            explainer = shap.KernelExplainer(model_predict, X_background)
            explainer_type = "Kernel"
            sample_size = min(50, len(X_test))  # Smaller sample for computational efficiency
            
        else:
            # For non-stacking models: try TreeExplainer first
            try:
                explainer = shap.TreeExplainer(model)
                explainer_type = "Tree"
                sample_size = min(500, len(X_test))
                print(f"  Using TreeExplainer")
                
            except Exception as tree_error:
                # Fallback to KernelExplainer for models without tree structure
                print(f"  TreeExplainer not compatible, using KernelExplainer...")
                
                background_size = min(100, len(X_test))
                X_background = shap.sample(X_test, background_size, random_state=RANDOM_STATE)
                
                def model_predict(X):
                    if hasattr(model, 'predict_proba'):
                        return model.predict_proba(X)[:, 1]
                    else:
                        return model.predict(X)
                
                explainer = shap.KernelExplainer(model_predict, X_background)
                explainer_type = "Kernel"
                sample_size = min(50, len(X_test))  # Smaller than trees for KernelExplainer (slow)
                print(f"  Using KernelExplainer")
        
        # Sample data for SHAP calculation
        X_sample = X_test.sample(n=sample_size, random_state=RANDOM_STATE)
        
        print(f"✓ Calculating SHAP values for {sample_size} samples...")
        if explainer_type == "Kernel":
            print(f"  Note: KernelExplainer is slower but model-agnostic")
        
        # Calculate SHAP values
        if explainer_type == "Kernel":
            shap_values = explainer.shap_values(X_sample, nsamples=100)
        else:
            shap_values = explainer.shap_values(X_sample)
        
        # Handle different SHAP value formats
        if isinstance(shap_values, list):
            # For binary classification, take positive class
            shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
        
        # Ensure shap_values is 2D array
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(-1, 1)
        
        print(f"✓ SHAP values shape: {shap_values.shape}")
        
        # Create summary plot
        plt.figure(figsize=(10, 8))
        try:
            shap.summary_plot(
                shap_values, 
                X_sample, 
                feature_names=feature_names,
                max_display=max_display, 
                show=False
            )
        except Exception as plot_error:
            # Fallback: create a bar plot if summary_plot fails
            print(f"  Warning: summary_plot failed ({plot_error}), creating bar plot...")
            shap_importance = np.abs(shap_values).mean(axis=0)
            top_indices = np.argsort(shap_importance)[::-1][:max_display]
            
            plt.figure(figsize=(10, 8))
            plt.barh(
                range(len(top_indices)), 
                shap_importance[top_indices][::-1],
                color='steelblue'
            )
            plt.yticks(range(len(top_indices)), [feature_names[i] for i in top_indices][::-1])
            plt.xlabel('Mean |SHAP value|')
            plt.title(f'SHAP Feature Importance - {model_name}')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ SHAP summary plot saved: {filename}")
        
        # Calculate feature importance
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        # Handle case where feature_names might be shorter than shap_importance
        num_features = min(len(feature_names), len(shap_importance))
        
        shap_df = pd.DataFrame({
            'feature': feature_names[:num_features],
            'shap_importance': shap_importance[:num_features]
        }).sort_values('shap_importance', ascending=False)
        
        print(f"\nTop 10 Features by SHAP Importance:")
        print(f"{'='*60}")
        for idx, (_, row) in enumerate(shap_df.head(10).iterrows(), 1):
            print(f"  {idx:2d}. {row['feature']:30s}: {row['shap_importance']:.4f}")
        print(f"{'='*60}")
        
        if is_stacking:
            print(f"\nInterpretation Note for Stacking Ensemble:")
            print(f"   - SHAP values reflect the COMBINED impact of:")
            print(f"     • All base models' predictions")
            print(f"     • Meta-learner's weighting strategy")
            print(f"   - These are global feature importances for the entire ensemble")
        
        return shap_df
        
    except Exception as e:
        print(f"SHAP analysis failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        print(f"\nTraceback:")
        traceback.print_exc()
        return None


# 7. LEARNING AND VALIDATION CURVES

def plot_learning_curves(model, X_train, y_train, model_name, scaling_type='unscaled', save_dir='../figures/03_Model_Selection'):
    """
    Plot learning curves
    """
    print(f"\n{'='*80}")
    print(f"Learning Curves - {model_name.lower()}_{scaling_type}")
    print(f"{'='*80}")

    filename = f'learning_curves_{model_name.lower()}_{scaling_type}.png'
    
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size', fontsize=12)
    plt.ylabel('ROC-AUC Score', fontsize=12)
    plt.title(f'Learning Curves -  {model_name} ({scaling_type})', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"✓ Learning curves saved: {filename}")
    plt.close()
    
    return {
        'train_sizes': train_sizes.tolist(),
        'train_auc': train_mean.tolist(),
        'val_auc': val_mean.tolist()
    }


def plot_validation_curve(model, X_train, y_train, model_name, scaling_type='unscaled', save_dir='../figures/03_Model_Selection'):
    """
    Plot validation curve for n_estimators
    """
    print(f"Validation Curve - {model_name.lower()}_{scaling_type}")

    filename = f'validation_curve_{model_name.lower()}_{scaling_type}.png'

    if not hasattr(model, 'n_estimators'):
        print(f"  Model does not have n_estimators parameter.")
        return None
    
    param_range = [50, 100, 200, 400]
    
    train_scores, val_scores = validation_curve(
        model, X_train, y_train,
        param_name='n_estimators',
        param_range=param_range,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    plt.plot(param_range, val_mean, 'o-', color='red', label='Cross-validation score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('n_estimators', fontsize=12)
    plt.ylabel('ROC-AUC Score', fontsize=12)
    plt.title(f'Validation Curve - {model_name.lower()}_{scaling_type} - n_estimators', 
              fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f"Validation curve saved: {filename}")
    plt.close()
    
    return {
        'param_name': 'n_estimators',
        'param_range': param_range,
        'train_auc': train_mean.tolist(),
        'val_auc': val_mean.tolist()
    }


# =============================================================================
# 8. MODEL SELECTION
# =============================================================================

def select_best_model(results_dict, models_dict, X_test, y_test):
    """
    Select best model based on weighted score:
    0.6 * ROC-AUC + 0.4 * Recall
    Note: Thresholds are optimized using F2-score (emphasizes recall)
    to ensure high sensitivity in medical diagnosis context.
    
    Thresholds from results_dict are used if available; otherwise 0.5 is default.
    """
    print("\n" + "="*80)
    print("MODEL SELECTION (Weighted: 0.6*ROC-AUC + 0.4*Recall)")
    print("="*80)
    
    weighted_scores = {}
    
    for name, model in models_dict.items():
        # Use threshold from results_dict if present, else default 0.5
        threshold = results_dict[name].get("threshold", 0.5)
        if threshold == 0.5:
            print(f"Warning: threshold not found for {name}, using 0.5 by default.")

        # Evaluate using custom threshold
        metrics = evaluate_model(model, X_test, y_test, threshold)['metrics']
        
        roc_auc = metrics["roc_auc"]
        recall = metrics["recall"]
        pr_auc = metrics["pr_auc"]
        
        weighted_score = 0.6 * roc_auc + 0.4 * recall
        weighted_scores[name] = weighted_score
        
        print(f"\n{name}:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  Recall:  {recall:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        print(f"  Weighted Score: {weighted_score:.4f}")
    
    best_model_name = max(weighted_scores, key=weighted_scores.get)
    best_model = models_dict[best_model_name]
    
    print(f"BEST MODEL: {best_model_name}")
    print(f"   Weighted Score: {weighted_scores[best_model_name]:.4f}")
    
    return best_model_name, best_model, weighted_scores


# =============================================================================
# 9. SAVE MODEL AND METADATA
# =============================================================================

def save_final_model(model, model_name, feature_names, metrics, 
                     hyperparameters, final_scores, optimal_threshold,
                     optimal_recall, learning_curve_data, validation_curve_data,
                     label_encoders=None, scaling_type='unscaled', save_dir='../models'):
    """
    Save the final selected model with complete metadata
    """
    print("SAVING FINAL MODEL")
    
    os.makedirs(save_dir, exist_ok=True)
    
    model_filename = f'best_model_{scaling_type}.pkl'
    metadata_filename = f'model_metadata_{scaling_type}.json'
    encoders_filename = f'label_encoders_{scaling_type}.pkl'
    
    # Save model
    model_path = os.path.join(save_dir, model_filename)
    metadata_path = os.path.join(save_dir, metadata_filename)
    encoders_path = os.path.join(save_dir, encoders_filename)
    
    joblib.dump(model, model_path)
    print(f"\nModel saved: {model_path}")

    if model_name.lower().startswith('stack'):
        learning_curve_data = None
        validation_curve_data = None
    
    # Save metadata
    metadata = {
        'model_name': model_name,
        'model_type': type(model).__name__,
        'scaling_type': scaling_type,
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'feature_names': feature_names,
        'n_features': len(feature_names),
        'test_metrics': metrics,
        'tuned_hyperparameters': hyperparameters,
        'final_weighted_scores': final_scores,
        'optimal_threshold': float(optimal_threshold),
        'optimal_recall': float(optimal_recall),
        'learning_curve': learning_curve_data,
        'validation_curve': validation_curve_data,
        'imbalance_handling': 'scale_pos_weight / class_weight',
        'hyperparameter_tuning': 'Optuna (boosting) + threshold optimization',
        'label_encoders': {k: v.classes_.tolist() for k, v in label_encoders.items()} 
                          if label_encoders else None,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved: {metadata_path}")
    
    if label_encoders:
        joblib.dump(label_encoders, encoders_path)
        print(f"Label encoders saved: {encoders_path}")
    
    print(f"\nAll artifacts saved to: {save_dir}/")
    
    return model_path, metadata_path


def create_model_comparison_csv(results_dict, scaling_type='unscaled', save_dir='../data/processed'):
    """
    Create CSV file comparing all models
    """
    print("CREATING MODEL COMPARISON TABLE")

    filename = f'model_comparison_{scaling_type}.csv'
    
    all_metrics = []
    
    for name, results in results_dict.items():
        metrics = results['metrics']
        metrics['model'] = name

        if 'inference_time' not in metrics:
            metrics['inference_time'] = None
        if 'inference_time_per_sample' not in metrics:
            metrics['inference_time_per_sample'] = None
            
        all_metrics.append(metrics)
    
    df = pd.DataFrame(all_metrics)

    df = df.rename(columns={
        "f1": "f1-score",
        "roc_auc": "ROC-AUC"})

    df['dataset_version'] = scaling_type

    df.loc[df['model'] == 'XGBoost', 'imbalance_method'] = 'scale_pos_weight'
    df.loc[df['model'] == 'LightGBM', 'imbalance_method'] = 'class_weight'
    df.loc[df['model'] == 'AdaBoost', 'imbalance_method'] = 'sample_weight'
    df.loc[df['model'] == 'Stacking_ET', 'imbalance_method'] = 'class_weight (RF, SVM, meta-learner)'
    df.loc[df['model'] == 'Stacking_SVM', 'imbalance_method'] = 'class_weight (RF, SVM, meta-learner)'

    
    # Reorder columns
    column_order = ['dataset_version', 'model', 'imbalance_method', 'accuracy', 'precision',
                    'recall','f1-score', 'roc_auc', 'pr_auc', 'threshold', 'inference_time',
                    'inference_time_per_sample']

    # Keep only columns that exist
    column_order = [col for col in column_order if col in df.columns]
    other_cols = [col for col in df.columns if col not in column_order]
    df = df[column_order + other_cols]
    
    df = df.sort_values('ROC-AUC', ascending=False)
    df.to_csv(os.path.join(save_dir, filename), index=False)
    
    print(f"\nModel comparison saved: {filename}")
    print("\nModel Comparison Table:")
    print(df.to_string(index=False))
    
    return df


# 10. MAIN EXECUTION PIPELINE

def main(train_filepath=None,
         test_filepath=None,
         target_col_name='TenYearCHD', 
         scaling_type='unscaled'):
    """
    Main execution pipeline for for Model Selection with threshold optimization
    
    Pipeline:
    1. Load pre-split data
    2. Hyperparameter tuning (Optuna)
    3. Train base models
    4. Optimize thresholds using F1-score on training set
    5. Evaluate all models with optimized thresholds on test set
    6. Create stacking ensembles (if scaled data)
    7. Select best model
    8. Generate visualizations and interpretability analyses
    9. Save final model and results
    """
    
    print("CORONARY HEART DISEASE PREDICTION PROJECT")
    print("Advanced Models & Model Selection per dataset version.")
    print("Traditional Models not considered. They will be incorporated in the following part of this notebook.")
    print(f"    Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # STEP 1: DATA LOADING
    assert train_filepath and test_filepath, "Please provide train and test file paths"
    X_train, X_test, y_train, y_test, feature_names, label_encoders = load_presplit_data(
        train_filepath=train_filepath,
        test_filepath=test_filepath,
        target_col_name=target_col_name
    )
    
    # STEP 2: HYPERPARAMETER TUNING WITH OPTUNA
    best_xgb_params = tune_boosting_model(XGBClassifier, XGB_PARAM_SPACE, X_train, y_train, model_name="XGBoost")
    best_lgbm_params = tune_boosting_model(LGBMClassifier, LGBM_PARAM_SPACE, X_train, y_train, model_name="LightGBM")
    best_ada_params = tune_adaboost(X_train, y_train, n_trials=40)
    
    all_hyperparameters = {
        'xgboost': best_xgb_params,
        'lightgbm': best_lgbm_params,
        'adaboost': best_ada_params
    }
    
    # STEP 3: TRAINING BASE MODELS (with default threshold 0.5)
    base_results, base_models = train_base_models(
        X_train, y_train, X_test, y_test,
        best_xgb_params, best_lgbm_params, best_ada_params
    )
    
    # STEP 4: THRESHOLD OPTIMIZATION FOR BASE MODELS (F2-Score on Training Set)
    for name, model in base_models.items():
        print(f"Optimizing threshold for: {name}")
        
        # Find optimal threshold on training set using F2-score
        optimal_threshold, optimal_recall = optimize_threshold_for_recall(
            model, X_train, y_train, min_recall=0.65)
        print(f" Optimal Threshold: {optimal_threshold:.4f}")
        print(f" Recall at Threshold (train): {optimal_recall:.4f}")
        
        # Evaluate on test set with optimized threshold   
        print(f"\n  Evaluating on test set with optimized threshold...")
        eval_results = evaluate_model(model, X_test, y_test, optimal_threshold)
        
        # Update results dictionary with threshold-optimized metrics
        base_results[name]['metrics'] = eval_results['metrics']
        base_results[name]['confusion_matrix'] = eval_results['confusion_matrix']
        base_results[name]['y_pred'] = eval_results['y_pred']
        base_results[name]['y_proba'] = eval_results['y_proba']
        base_results[name]['threshold'] = optimal_threshold
        base_results[name]['optimal_recall'] = optimal_recall
        
        # Print threshold-optimized test metrics
        metrics = eval_results['metrics']
        print(f"\n  Threshold-Optimized Test Metrics:")
        print(f"    Accuracy:  {metrics['accuracy']:.4f}")
        print(f"    Precision: {metrics['precision']:.4f}")
        print(f"    Recall:    {metrics['recall']:.4f}")
        print(f"    F1-Score:  {metrics['f1']:.4f}")
        print(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"    PR-AUC:    {metrics['pr_auc']:.4f}")
        
        # Print confusion matrix with interpretation
        cm = eval_results['confusion_matrix']
        tn, fp, fn, tp = cm.ravel()
        total_sick = tp + fn
        total_healthy = tn + fp
        
        print(f"\n  Confusion Matrix:")
        print(f"    TN: {tn:4d}  FP: {fp:4d}")
        print(f"    FN: {fn:4d}  TP: {tp:4d}")
        print(f"\n  Clinical Interpretation:")
        print(f"    • Sick patients correctly identified: {tp}/{total_sick} ({tp/total_sick*100:.1f}%)")
        print(f"    • Sick patients missed: {fn}/{total_sick} ({fn/total_sick*100:.1f}%)")
        print(f"    • Healthy patients correctly identified: {tn}/{total_healthy} ({tn/total_healthy*100:.1f}%)")
        print(f"    • False alarms (healthy flagged as sick): {fp}/{total_healthy} ({fp/total_healthy*100:.1f}%)")
              
    # STEP 5: CREATE STACKING ENSEMBLES (if scaled data)
    stacking_ensembles = {}
    
    if scaling_type != "unscaled":
        print("\nCREATING STACKING ENSEMBLES")

        # Create ensemble 1: XGB + LGBM + RF + ET
        print(f"\n{'-'*80}")
        print("Creating Stacking Ensemble 1: XGB + LGBM + RF + ExtraTrees")
        print(f"{'-'*80}")
        stacking_et = create_stacking_xgb_lgbm_rf_et(X_train, y_train)
        stacking_ensembles['Stacking_ET'] = stacking_et
        
        # Optimize threshold for Stacking_ET using F2-score
        print("\nOptimizing threshold for Stacking_ET...")
        threshold_et, recall_et = optimize_threshold_for_recall(stacking_et, X_train, y_train, min_recall=0.65)
        print(f" Optimal Threshold: {threshold_et:.4f}")
        print(f" Recall at Threshold (train): {recall_et:.4f}")
        
        # Evaluate on test set
        eval_et = evaluate_model(stacking_et, X_test, y_test, threshold_et)
        
        # Add cross-validation metrics
        print("\nCalculating cross-validation metrics...")
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
        cv_roc, cv_recall = [], []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            temp_model = create_stacking_xgb_lgbm_rf_et(X_tr, y_tr)
            y_val_proba = temp_model.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_proba >= threshold_et).astype(int)
            
            cv_roc.append(roc_auc_score(y_val, y_val_proba))
            cv_recall.append(recall_score(y_val, y_val_pred))
        
        # Store results
        base_results['Stacking_ET'] = {
            'metrics': eval_et['metrics'],
            'confusion_matrix': eval_et['confusion_matrix'],
            'y_pred': eval_et['y_pred'],
            'y_proba': eval_et['y_proba'],
            'threshold': threshold_et,
            'optimal_recall': recall_et,
            'cv_roc_auc_mean': np.mean(cv_roc),
            'cv_roc_auc_std': np.std(cv_roc),
            'cv_recall_mean': np.mean(cv_recall),
            'cv_recall_std': np.std(cv_recall)
        }
        base_models['Stacking_ET'] = stacking_et
        
        print(f"\nStacking_ET - Test Metrics:")
        print(f"  ROC-AUC: {eval_et['metrics']['roc_auc']:.4f}")
        print(f"  Recall:  {eval_et['metrics']['recall']:.4f}")
        print(f"  CV ROC-AUC: {np.mean(cv_roc):.4f} ± {np.std(cv_roc):.4f}")
        
        # Create ensemble 2: XGB + LGBM + RF + SVM
        print("\nCreating Stacking Ensemble 2: XGB + LGBM + RF + SVM")
        
        stacking_svm = create_stacking_xgb_lgbm_rf_svm(X_train, y_train)
        stacking_ensembles['Stacking_SVM'] = stacking_svm
        
        # Optimize threshold for Stacking_SVM using F2-score
        print("\nOptimizing threshold for Stacking_SVM...")
        threshold_svm, recall_svm = optimize_threshold_for_recall(stacking_svm, X_train, y_train, min_recall=0.65)
        print(f" - Optimal Threshold: {threshold_svm:.4f}")
        print(f" - Recall at Threshold (train): {recall_svm:.4f}")
        
        # Evaluate on test set
        eval_svm = evaluate_model(stacking_svm, X_test, y_test, threshold_svm)
        
        # Add cross-validation metrics
        print("\nCalculating cross-validation metrics...")
        cv_roc_svm, cv_recall_svm = [], []
        
        for train_idx, val_idx in cv.split(X_train, y_train):
            X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
            
            temp_model = create_stacking_xgb_lgbm_rf_svm(X_tr, y_tr)
            y_val_proba = temp_model.predict_proba(X_val)[:, 1]
            y_val_pred = (y_val_proba >= threshold_svm).astype(int)
            
            cv_roc_svm.append(roc_auc_score(y_val, y_val_proba))
            cv_recall_svm.append(recall_score(y_val, y_val_pred))
        
        # Store results
        base_results['Stacking_SVM'] = {
            'metrics': eval_svm['metrics'],
            'confusion_matrix': eval_svm['confusion_matrix'],
            'y_pred': eval_svm['y_pred'],
            'y_proba': eval_svm['y_proba'],
            'threshold': threshold_svm,
            'optimal_recall': recall_svm,
            'cv_roc_auc_mean': np.mean(cv_roc_svm),
            'cv_roc_auc_std': np.std(cv_roc_svm),
            'cv_recall_mean': np.mean(cv_recall_svm),
            'cv_recall_std': np.std(cv_recall_svm)
        }
        base_models['Stacking_SVM'] = stacking_svm
        
        print(f"\nStacking_SVM - Test Metrics:")
        print(f"  ROC-AUC: {eval_svm['metrics']['roc_auc']:.4f}")
        print(f"  Recall:  {eval_svm['metrics']['recall']:.4f}")
        print(f"  CV ROC-AUC: {np.mean(cv_roc_svm):.4f} ± {np.std(cv_roc_svm):.4f}")
        
    else:
        print("\nSkipping stacking ensembles (unscaled data)")        


    # STEP 6: Visualizations
    print("\nGENERATING VISUALIZATIONS")
    
    plot_roc_curves(base_results, y_test, scaling_type=scaling_type)
    plot_confusion_matrices(base_results, scaling_type=scaling_type)
    
    # STEP 7: Feature importance
    print("\nFEATURE IMPORTANCE & INTERPRETABILITY")
   
    feature_importance_dict = {}
    
    # Analyze all models
    for model_name in base_models:
        importance_df = analyze_feature_importance(
            base_models[model_name], feature_names, model_name, scaling_type=scaling_type
        )
        if importance_df is not None:
            feature_importance_dict[model_name] = importance_df
    
    # STEP 8: Learning and validation curves - TOP 2 BY ROC-AUC
    print("LEARNING AND VALIDATION CURVES - 2 TOP MODELS BY ROC-AUC")
    
    # Sort all models by ROC-AUC (test set performance)
    sorted_models = sorted(
        base_results.items(), 
        key=lambda x: x[1]['metrics']['roc_auc'], 
        reverse=True
    )
    
    # Get top 2 models
    top_2_models = [name for name, _ in sorted_models[:2]]
    
    print(f"\nTop 2 models selected for detailed analysis:")
    for i, model_name in enumerate(top_2_models, 1):
        roc_auc = base_results[model_name]['metrics']['roc_auc']
        print(f"  {i}. {model_name}: ROC-AUC = {roc_auc:.4f}")
    
    # Generate curves for top 2 models
    learning_curves_dict = {}
    validation_curves_dict = {}
    
    for model_name in top_2_models:
        if model_name in base_models:  # Safety check
            print(f"\nGenerating curves for {model_name}...")
            learning_curves_dict[model_name] = plot_learning_curves(
                base_models[model_name], X_train, y_train, model_name
            )
            
            validation_curves_dict[model_name] = plot_validation_curve(
                base_models[model_name], X_train, y_train, model_name
            )

    # STEP 9: Create comparison table
    comparison_df = create_model_comparison_csv(base_results, scaling_type=scaling_type)

    # STEP 10: Model selection (between Boosting Models + 2 Stacked Ensemble Models)
    # Include all base models plus both stacking ensembles
    all_models = {}
    all_models.update(base_models)
    if scaling_type != "unscaled":
        all_models.update(stacking_ensembles)
    # Select the best model based on the evaluation metrics stored in base_results
    best_model_name, best_model, final_scores = select_best_model(base_results, all_models, X_test, y_test)
    
    print(f"\nBest model selected: {best_model_name}")
        
    # STEP 11: SHAP ANALYSIS - Best model only
    print(f"\nSHAP ANALYSIS - {best_model_name} (BEST MODEL)")
    
    best_model_shap_importance = shap_analysis(
        best_model, X_test, feature_names, best_model_name, scaling_type=scaling_type
    )

    # STEP 12: Save final model
    best_metrics = base_results[best_model_name]['metrics']
    optimal_threshold = base_results[best_model_name].get('threshold', 0.5)
    optimal_recall = base_results[best_model_name].get('optimal_recall', None)
    
    model_path, metadata_path = save_final_model(
        best_model, best_model_name, feature_names,
        best_metrics, all_hyperparameters, final_scores,
        optimal_threshold, optimal_recall,
        learning_curves_dict[best_model_name] if best_model_name in learning_curves_dict else None,
        validation_curves_dict[best_model_name] if best_model_name in validation_curves_dict else None,
        label_encoders, scaling_type=scaling_type
    )
    
    print("\nDETAILED RESULTS SUMMARY (All Models)")
    
    for model_name, results in base_results.items():
        print(f"\n{'='*80}")
        print(f"MODEL: {model_name}")
        print(f"{'='*80}")
        
        metrics = results['metrics']
        threshold = results.get('threshold', 0.5)
        
        print(f"\nThreshold: {threshold:.4f}")
        print(f"\nTest Set Performance (Threshold-Optimized):")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")

        if 'inference_time' in metrics:
            print(f"\nInference Performance:")
            print(f"  Total time: {metrics['inference_time']:.4f} seconds")
            print(f"  Per sample: {metrics['inference_time_per_sample']*1000:.2f} ms")
    
        # Cross-validation metrics
        if 'cv_roc_auc_mean' in results:
            print(f"\nCross-Validation:")
            print(f"  ROC-AUC: {results.get('cv_roc_auc_mean', 'N/A'):.4f}")
            print(f"  Recall:  {results.get('cv_recall_mean', 'N/A'):.4f}")
        elif 'cv_metrics' in results:
            cv = results['cv_metrics']
            print(f"\nCross-Validation:")
            print(f"  ROC-AUC: {cv['cv_roc_auc_mean']:.4f} ± {cv['cv_roc_auc_std']:.4f}")
            print(f"  Recall:  {cv['cv_recall_mean']:.4f} ± {cv['cv_recall_std']:.4f}")
        
        cm = results['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"  TN: {cm[0][0]:4d}  FP: {cm[0][1]:4d}")
        print(f"  FN: {cm[1][0]:4d}  TP: {cm[1][1]:4d}")
    
    print("\nFINAL SUMMARY")
    
    print(f"\nBest Model: {best_model_name}")
    print(f"  • Method: F2-score maximization on training set")
    print(f"  • Rationale: Emphasizes recall (catching sick patients)")
    print(f"  • Minimum Recall Target: 0.65 (65% sensitivity)")
    print(f"  • Optimized Threshold: {optimal_threshold:.4f}")
    if optimal_recall:
        print(f"  • Recall at Threshold (train): {optimal_recall:.4f}")
    
    print(f"\nTest Set Performance (Threshold-Optimized):")
    print(f"  • ROC-AUC:   {best_metrics['roc_auc']:.4f}")
    print(f"  • Recall:    {best_metrics['recall']:.4f}")
    print(f"  • Precision: {best_metrics['precision']:.4f}")
    print(f"  • F1-Score:  {best_metrics['f1']:.4f}")
    print(f"  • Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"  • PR-AUC:    {best_metrics.get('pr_auc', 0):.4f}")
    
    weighted_score = final_scores[best_model_name]
    print(f"\nSelection Criterion:")
    print(f"  • Weighted Score: {weighted_score:.4f}")
    print(f"  • Formula: 0.6*ROC-AUC + 0.4*Recall")
    
    print(f"\nModel Development:")
    print(f"  • Hyperparameter Tuning: Optuna")
    print(f"    - XGBoost/LightGBM: 50 trials")
    print(f"    - AdaBoost: 40 trials")
    print(f"  • Threshold Optimization: F1-score maximization")
    print(f"  • Class Imbalance: scale_pos_weight/class_weight/sample_weight")
    
    print(f"End Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return {
        'best_model': best_model,
        'best_model_name': best_model_name,
        'all_models': all_models,
        'all_results': base_results,
        'feature_importance': feature_importance_dict[best_model_name],
        'shap_importance': best_model_shap_importance,
        'comparison_df': comparison_df,
        'hyperparameters': all_hyperparameters,
        'final_scores': final_scores,
        'optimal_threshold': optimal_threshold,
        'optimal_recall': optimal_recall
    }

__all__ = [
    "load_presplit_data",
    "tune_boosting_model",
    "tune_adaboost",
    "train_base_models",
    "create_base_models",
    "create_stacking_ensemble", 
    "create_stacking_xgb_lgbm_rf_et", 
    "create_stacking_xgb_lgbm_rf_svm", 
    "evaluate_model",
    "optimize_threshold_for_recall",
    "plot_roc_curves", 
    "plot_confusion_matrices", 
    "analyze_feature_importance",
    "shap_analysis",
    "plot_learning_curves",
    "plot_validation_curve",
    "select_best_model",
    "save_final_model",
    "create_model_comparison_csv",
    "main"
]