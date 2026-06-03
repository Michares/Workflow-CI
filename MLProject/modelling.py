"""
modelling.py untuk MLflow Project (CI)
Training Personality Classification dengan param dari CLI.
Tracking di MLflow lokal (di GitHub Actions runner).
"""

import os
import json
import argparse
import warnings
import joblib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

warnings.filterwarnings("ignore")


def load_data(data_dir="personality_preprocessing"):
    X_train = pd.read_csv(f"{data_dir}/X_train.csv")
    X_test = pd.read_csv(f"{data_dir}/X_test.csv")
    y_train = pd.read_csv(f"{data_dir}/y_train.csv").squeeze()
    y_test = pd.read_csv(f"{data_dir}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


def plot_confusion_matrix(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Extrovert', 'Introvert'],
                yticklabels=['Extrovert', 'Introvert'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


def plot_roc(y_true, y_proba, path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(path, dpi=100, bbox_inches='tight')
    plt.close()


def main(args):
    print(f"[INFO] Params: n_estimators={args.n_estimators}, max_depth={args.max_depth}")

    # Autolog ON
    mlflow.sklearn.autolog()

    X_train, X_test, y_train, y_test = load_data()
    print(f"[INFO] Train: {X_train.shape}, Test: {X_test.shape}")

    with mlflow.start_run(run_name="RandomForest_CI") as run:
        model = RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        for k, v in metrics.items():
            mlflow.log_metric(k, v)
            print(f"  {k}: {v:.4f}")

        # Custom artifacts
        os.makedirs("artifacts", exist_ok=True)
        plot_confusion_matrix(y_test, y_pred, "artifacts/confusion_matrix.png")
        plot_roc(y_test, y_proba, "artifacts/roc_curve.png")

        with open("artifacts/metric_info.json", "w") as f:
            json.dump(metrics, f, indent=2)

        with open("artifacts/classification_report.txt", "w") as f:
            f.write(classification_report(y_test, y_pred, target_names=['Extrovert', 'Introvert']))

        joblib.dump(model, "artifacts/model.pkl")

        mlflow.log_artifacts("artifacts")

        print(f"[SUCCESS] Run ID: {run.info.run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=50)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()
    main(args)