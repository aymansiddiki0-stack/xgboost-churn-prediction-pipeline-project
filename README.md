XGBoost Churn Classifier with Bayesian Optimization
A machine learning project that predicts customer churn using gradient boosting and Bayesian hyperparameter optimization. Achieves 85.9% ROC-AUC by identifying contract type and service usage patterns as primary churn indicators.
Quick Stats

7,043 customers analyzed with 26.5% baseline churn rate
85.9% ROC-AUC on test set
35.4% feature importance attributed to contract type alone
3-fold CV shows consistent 86.6% mean AUC with low variance

Results
Test set performance:
ROC-AUC:    0.859
PR-AUC:     0.673
Precision:  0.67 (churn class)
Recall:     0.53 (churn class)
Brier:      0.142
Log Loss:   0.389
Cross-validation folds:
0.8581 | 0.8641 | 0.8758  →  mean: 0.8660 (±0.0074)
What Drives Churn?
The model reveals that contract structure matters most. Month-to-month customers churn at 42.7% vs just 2.8% for two-year contracts - a 15x difference that dwarfs all other factors.
Feature importance breakdown:

Contract type: 35.4%
Online security: 10.8%
Internet type: 7.0%
Tenure: 6.5%
Monthly charges: 5.2%

Interestingly, price is less important than expected. Service bundling and commitment level are stronger predictors, suggesting churn is more about perceived value and switching costs than absolute price.
How It Works
Data Preprocessing
Started with 33 features, dropped 13 for data leakage or lack of signal:

Removed: Churn Score, CLTV, Churn Reason (post-churn data)
Removed: Geographic fields (Country, State, City, Lat/Long)
Kept: 3 numeric + 16 categorical features

Missing values handled via median imputation (numeric) and "Unknown" category (categorical). StandardScaler applied without centering to maintain sparse matrix compatibility with one-hot encoded features.
Model Training
XGBoost classifier optimized using Optuna's TPE sampler across 30 trials:
pythonn_estimators: 172
max_depth: 4
learning_rate: 0.028
subsample: 0.81
colsample_bytree: 0.77
reg_lambda: 1.46
reg_alpha: 3.06
Heavy regularization (alpha=3.06, lambda=1.46) helps manage the high-dimensional one-hot encoded categorical features. Low learning rate with more estimators trades training time for better generalization.
Synthetic Data Generator
Built a K-Means based generator that preserves realistic feature correlations:

Cluster customers into 4 segments based on tenure, charges, and contract type
Learn distributions within each segment
Generate new samples that maintain segment characteristics

Useful for testing the pipeline without exposing production data. Validates that preprocessing handles edge cases correctly.
bashpython generate_synthetic_data.py --n_samples 1000 --validate
Setup
bashgit clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
Requires Python 3.9+. Main dependencies: pandas, scikit-learn, xgboost, optuna, matplotlib.
Running the Pipeline
Train a new model:
bashpython run_pipeline.py
This runs the full pipeline:

Loads data from data/raw/Telco_customer_churn.xlsx
Preprocesses features per config
Optimizes hyperparameters (30 Optuna trials)
Trains final model on full training set
Generates evaluation reports in reports/
Saves model to models/churn_model_YYYYMMDD_HHMMSS.pkl

Outputs include confusion matrix, ROC curve, and feature importance plots.
Making Predictions
pythonimport joblib
import pandas as pd

model = joblib.load('models/churn_model_latest.pkl')
new_customers = pd.read_csv('customers.csv')

# Get churn probabilities
proba = model.predict_proba(new_customers)[:, 1]

# Flag high-risk customers (>50% churn probability)
high_risk = new_customers[proba > 0.5]
Configuration
Edit config.yaml to change:

Hyperparameter search trials
CV fold count
Train/test split ratio
Feature selection

All random seeds are configurable for reproducibility.
Notes
The model shows strong performance but has some limitations worth noting:

Trained on single company data (generalization unknown)
No temporal validation (should test on future data)
Class imbalance means precision/recall tradeoff
Assumes all features available at prediction time

The regularization parameters suggest the model relies on a sparse subset of the one-hot encoded features, which aligns with the feature importance showing contract type dominates.

Dataset
IBM Telecommunications Customer Churn

License
MIT