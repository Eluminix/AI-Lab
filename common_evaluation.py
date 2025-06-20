# common_evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# === Predict-Wrapper ===

def predict_with_model(model, X):
    """Standard prediction using model.predict"""
    return model.predict(X)

def predict_with_function(predict_func, X):
    """Prediction using a function (e.g. similarity scoring)"""
    return predict_func(X)


# === Evaluation für Klassifikation ===

def evaluate_classification(y_true, y_pred, description="Model"):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\nClassification Evaluation for {description}")
    print(f"Accuracy: {acc:.3f}")
    print(f"Precision (weighted): {prec:.3f}")
    print(f"Recall (weighted): {rec:.3f}")
    print(f"F1 Score (weighted): {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=3, zero_division=0))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'{description} - Confusion Matrix')
    plt.tight_layout()
    plt.show()

    return {
        "Model": description,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }


# === Evaluation für Regression ===

def evaluate_regression(y_true, y_pred, description="Model"):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\nRegression Evaluation for {description}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"R2 Score: {r2:.3f}")

    # Scatter plot
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title(f'{description} - Regression Prediction Plot')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], '--', color='red')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        "Model": description,
        "MSE": mse,
        "MAE": mae,
        "R2": r2
    }
