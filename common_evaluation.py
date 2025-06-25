# common_evaluation.py

import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import pandas as pd
from scipy.stats import pearsonr

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

RESULT_FOLDER_PATH = "eval_results"

# === Predict-Wrapper ===

def predict_with_model(model, X):
    """Standard prediction using model.predict"""
    return model.predict(X)

def predict_with_function(predict_func, X):
    """Prediction using a function (e.g. similarity scoring)"""
    return predict_func(X)


# === Evaluation für Klassifikation ===

def evaluate_classification(y_true, y_pred, description="Model", save_json_file_name=None):
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

    results = {
        "Model": description,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1 Score": f1
    }

    if save_json_file_name:
        os.makedirs(RESULT_FOLDER_PATH, exist_ok=True)
        save_path = os.path.join(RESULT_FOLDER_PATH, save_json_file_name)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {save_path}")

    return results


# === Evaluation für Regression ===

def evaluate_regression(y_true, y_pred, description="Model", save_json_file_name=None):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson_corr, _ = pearsonr(y_true, y_pred)

    print(f"\nRegression Evaluation for {description}")
    print(f"Mean Squared Error (MSE): {mse:.3f}")
    print(f"Mean Absolute Error (MAE): {mae:.3f}")
    print(f"R2 Score: {r2:.3f}")
    print(f"Pearson Correlation (r): {pearson_corr:.3f}")

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

    results = {
        "Model": description,
        "MSE": mse,
        "MAE": mae,
        "R2": r2,
        "Pearson Correlation": pearson_corr
    }

    if save_json_file_name:
        os.makedirs(RESULT_FOLDER_PATH, exist_ok=True)
        save_path = os.path.join(RESULT_FOLDER_PATH, save_json_file_name)
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to: {save_path}")

    return results


# === Optional: Lade & vergleiche gespeicherte Resultate ===

def load_all_results():
    all_results = []

    for file in os.listdir(RESULT_FOLDER_PATH):
        if file.endswith(".json"):
            with open(os.path.join(RESULT_FOLDER_PATH, file), "r") as f:
                result = json.load(f)
                all_results.append(result)

    return pd.DataFrame(all_results)


def plot_model_comparison(df, metric="F1 Score", title="Model Comparison"):
    if metric not in df.columns:
        print(f"Metric '{metric}' not found in results.")
        return

    plt.figure(figsize=(10, 5))
    sorted_df = df.sort_values(by=metric, ascending=False)
    bars = plt.bar(sorted_df["Model"], sorted_df[metric])
    plt.ylabel(metric)
    plt.title(title)
    plt.ylim(0, 1)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.2f}", ha='center')

    plt.xticks(rotation=45, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()
