"""
Model evaluation module.

Generates comprehensive evaluation metrics and visualizations including
confusion matrix, ROC curve, and feature importance analysis.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from src.metrics import compute_metrics


os.makedirs("reports", exist_ok=True)


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model and generate comprehensive reports.
    Args:
        model: Trained sklearn pipeline
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    """

    # Generate predictions
    try:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
    except Exception as e:
        print(f"Model prediction failed: {e}")
        return

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    # Calculate and display performance metrics
    print("\nPerformance Metrics:")
    metrics = compute_metrics(y_test, proba)
    print(f"  ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC:            {metrics['pr_auc']:.4f}")
    print(f"  Brier Score:       {metrics['brier']:.4f}")
    print(f"  Log Loss:          {metrics['log_loss']:.4f}")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, preds)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()
    
    # Add labels
    classes = ['No Churn', 'Churn']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=16, fontweight='bold')
    
    plt.ylabel('True Label', fontsize=11)
    plt.xlabel('Predicted Label', fontsize=11)
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Generate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='steelblue', lw=2, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate', fontsize=11)
    plt.title('ROC Curve - Churn Prediction', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig("reports/roc_curve.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Compute feature importance
    
    try:
        # Extract trained model from pipeline
        trained_model = model.named_steps["model"]
        
        # Get feature names from preprocessor
        if "pre" in model.named_steps:
            preprocessor = model.named_steps["pre"]
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = np.array(X_test.columns)
        else:
            feature_names = np.array(X_test.columns)
        
        # Clean feature names (remove transformer prefixes)
        clean_names = []
        for name in feature_names:
            new_name = name.replace("num__", "").replace("cat__", "").replace("te__", "")
            clean_names.append(new_name.strip())
        feature_names = np.array(clean_names)
        
        # Get feature importances from model
        if hasattr(trained_model, "feature_importances_"):
            importances = trained_model.feature_importances_
            
            indices = np.argsort(importances)[::-1]
            
            # Print top 15 features
            print("\nTop 15 Most Important Features:")
            for i, idx in enumerate(indices[:15], 1):
                fname = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
                print(f"  {i:2d}. {fname:40s} {importances[idx]:.4f}")
            
            # Generate feature importance visualization
            
            # Plot top 20 features
            top_n = min(20, len(importances))
            top_indices = indices[:top_n]
            top_features = [feature_names[i] if i < len(feature_names) else f"Feature {i}" 
                          for i in top_indices]
            top_importances = importances[top_indices]
            
            # Horizontal bar plot
            plt.figure(figsize=(10, 8))
            y_pos = np.arange(len(top_features))
            plt.barh(y_pos, top_importances, color='steelblue')
            plt.yticks(y_pos, top_features, fontsize=9)
            plt.xlabel('Feature Importance (Gain)', fontsize=11)
            plt.title('Top 20 Most Important Features', fontsize=13, fontweight='bold')
            plt.gca().invert_yaxis()
            plt.grid(alpha=0.3, linestyle='--', axis='x')
            plt.tight_layout()
            plt.savefig("reports/feature_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Cumulative importance plot
            cumsum = np.cumsum(np.sort(importances)[::-1])
            cumsum_pct = cumsum / cumsum[-1] * 100
            
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(cumsum_pct) + 1), cumsum_pct, 
                    marker='o', markersize=3, linewidth=2, color='steelblue')
            plt.axhline(y=80, color='red', linestyle='--', alpha=0.7, 
                       label='80% of importance')
            plt.axhline(y=95, color='orange', linestyle='--', alpha=0.7, 
                       label='95% of importance')
            plt.xlabel('Number of Features', fontsize=11)
            plt.ylabel('Cumulative Importance (%)', fontsize=11)
            plt.title('Cumulative Feature Importance', fontsize=13, fontweight='bold')
            plt.grid(alpha=0.3, linestyle='--')
            plt.legend()
            plt.tight_layout()
            plt.savefig("reports/cumulative_importance.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        else:
            print("Model does not have feature_importances_ attribute")
            
    except Exception as e:
        print(f"Could not compute feature importance: {e}")