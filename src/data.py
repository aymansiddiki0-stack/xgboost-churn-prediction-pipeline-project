"""
Data loading and preprocessing module.

Handles loading raw data, cleaning, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_data(config):
    """
    Load and preprocess customer churn data.
    Args:
        config (dict): Configuration dictionary containing data parameters     
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    data_cfg = config["data"]
    file_path = Path(data_cfg["file_path"])
    target = data_cfg["target"]

    # Load data based on file type
    if file_path.suffix == ".csv":
        df = pd.read_csv(file_path)
    elif file_path.suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(file_path, sheet_name=data_cfg.get("sheet_name", 0))
    else:
        raise ValueError(f"Unsupported file format: {file_path.suffix}")

    # Drop columns to prevent data leakage
    drop_cols = config["features"]["drop"]
    cols_to_drop = [col for col in drop_cols if col in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Clean numeric columns (handle whitespace, convert types)
    num_cols = config["features"]["numeric"]
    for col in num_cols:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .replace(r"^\s*$", np.nan, regex=True)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Check for missing values
    print(f"\nMissing values before cleaning:")
    missing_counts = df.isnull().sum()
    if missing_counts.sum() > 0:
        print(missing_counts[missing_counts > 0])
    else:
        print("  No missing values found")

    # Drop rows where target is missing
    df = df.dropna(subset=[target])

    # Impute numeric columns with median
    for col in num_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Filled {col} missing values with median: {median_val:.2f}")

    # Impute categorical columns with 'Unknown'
    cat_cols = config["features"]["categorical"]
    for col in cat_cols:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna("Unknown")
            print(f"  Filled {col} missing values with 'Unknown'")

    print(f"\nRemaining missing values: {df.isnull().sum().sum()}")

    # Split into features and target
    y = df[target]
    X = df.drop(columns=[target])

    # Train/test split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=data_cfg["test_size"],
        random_state=data_cfg["random_state"],
        stratify=y
    )

    print(f"\nTraining samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Churn rate - Train: {y_train.mean():.2%}, Test: {y_test.mean():.2%}")
    
    return X_train, X_test, y_train, y_test