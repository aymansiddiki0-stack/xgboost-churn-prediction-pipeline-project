"""
Main training pipeline for telco customer churn prediction.

This script orchestrates the complete ML pipeline including data loading,
model training, evaluation, and saving results.
"""

import yaml
import joblib
import os
import sys
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data import load_data
from src.train import train_model
from src.evaluate import evaluate_model


def main():
    """
    Execute the complete training pipeline.
    """    
    # Load configuration
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Load and split data
    X_train, X_test, y_train, y_test = load_data(config)

    # Train model with hyperparameter optimization
    model, best_params = train_model(X_train, y_train, config)

    # Evaluate model performance
    evaluate_model(model, X_test, y_test)

    # Save trained model
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/churn_model_{timestamp}.pkl"
    joblib.dump(model, model_path)
    
    # Also save as latest for easy access
    latest_path = "models/churn_model_latest.pkl"
    joblib.dump(model, latest_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Latest model:   {latest_path}")
    print(f"Reports saved to: reports/")


if __name__ == "__main__":
    main()