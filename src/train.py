"""
Model training module with hyperparameter optimization.

Uses Optuna for Bayesian hyperparameter optimization and XGBoost
for the classification model.
"""

import os
import json
import random
import numpy as np
import optuna

from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score


def _seed_everything(seed):
    """Set random seeds for reproducibility."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def _make_preprocessor(config):
    """
    Create sklearn preprocessing pipeline.
    
    Args:
        config (dict): Configuration dictionary
        
    Returns:
        tuple: (preprocessor, drop_cols)
    """
    feats = config.get("features", {})
    num_cols = feats.get("numeric", [])
    cat_cols = feats.get("categorical", [])
    drop_cols = feats.get("drop", [])

    # Numeric transformer - scale without centering for sparse compatibility
    num_transformer = Pipeline(steps=[
        ("scale", StandardScaler(with_mean=False))
    ])

    # Categorical transformer - one-hot encode
    cat_transformer = Pipeline(steps=[
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0,
    )
    
    return preprocessor, drop_cols


def train_model(X_train, y_train, config):
    """
    Train XGBoost model with Optuna hyperparameter optimization.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        config (dict): Configuration dictionary 
    Returns:
        tuple: (trained_pipeline, best_params)
    """
    train_cfg = config.get("training", {})
    seed = int(config.get("seed", 42))
    n_trials = int(train_cfg.get("n_trials", 30))
    cv_folds = int(train_cfg.get("cv_folds", 3))

    _seed_everything(seed)
    print(f"Starting hyperparameter optimization with {n_trials} trials.")

    # Create Optuna study with TPE sampler for efficient search
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    # Store CV scores from best trial
    best_trial_cv_scores = []

    def objective(trial):
        """Optuna objective function for hyperparameter search."""
        preprocessor, drop_cols = _make_preprocessor(config)
        
        # Define hyperparameter search space
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 5.0),
            "random_state": seed,
            "tree_method": "hist",
            "n_jobs": -1,
            "verbosity": 0,
        }

        # Create pipeline with preprocessing and model
        pipeline = Pipeline([
            ("pre", preprocessor),
            ("model", XGBClassifier(**params)),
        ])

        # Crossvalidation with stratified folds
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
        aucs = []
        
        for fold_num, (tr_idx, va_idx) in enumerate(cv.split(X_train, y_train), 1):
            X_tr = X_train.iloc[tr_idx].drop(columns=drop_cols, errors="ignore")
            X_va = X_train.iloc[va_idx].drop(columns=drop_cols, errors="ignore")
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            pipeline.fit(X_tr, y_tr)
            preds = pipeline.predict_proba(X_va)[:, 1]
            fold_auc = roc_auc_score(y_va, preds)
            aucs.append(fold_auc)
        
        mean_auc = float(np.mean(aucs))
        
        # Store CV scores if this is the best trial so far
        nonlocal best_trial_cv_scores
        if trial.number == 0 or mean_auc > study.best_value:
            best_trial_cv_scores = aucs.copy()
        
        return mean_auc

    # Run optimization
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # Extract best parameters
    best_params = study.best_params
    best_params["random_state"] = seed
    best_params["tree_method"] = "hist"
    best_params["n_jobs"] = -1
    best_params["verbosity"] = 0

    # Display results
    print(f"\nBest CV ROC-AUC: {study.best_value:.4f}")
    print(f"\nCross-Validation Fold Results (Best Trial):")
    for i, score in enumerate(best_trial_cv_scores, 1):
        print(f"   Fold {i}: {score:.4f}")
    print(f"   Mean:   {np.mean(best_trial_cv_scores):.4f}")
    print(f"   Std:    {np.std(best_trial_cv_scores):.4f}")
    
    print(f"\nBest Hyperparameters:")
    for k, v in best_params.items():
        if k not in ["random_state", "tree_method", "n_jobs", "verbosity"]:
            print(f"   {k}: {v}")

    # Save results to file
    os.makedirs("reports", exist_ok=True)
    results = {
        "best_params": best_params,
        "cv_scores": {
            "folds": [float(score) for score in best_trial_cv_scores],
            "mean": float(np.mean(best_trial_cv_scores)),
            "std": float(np.std(best_trial_cv_scores))
        }
    }
    with open("reports/best_params.json", "w") as f:
        json.dump(results, f, indent=2)

    # Train final model on full training set
    final_preprocessor, drop_cols = _make_preprocessor(config)
    final_pipeline = Pipeline([
        ("pre", final_preprocessor),
        ("model", XGBClassifier(**best_params)),
    ])

    X_tr_full = X_train.drop(columns=drop_cols, errors="ignore")
    final_pipeline.fit(X_tr_full, y_train)
    
    return final_pipeline, best_params