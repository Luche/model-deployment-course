"""
Session 05 – Model evaluation (both approaches).
A single evaluate() function works for both NoPipeline and Pipeline runs:
  - NoPipeline: pass encoded x_test (manual preprocessing already applied)
  - Pipeline:   pass raw (renamed) x_test (pipeline handles preprocessing internally)
The MLflow model loaded from run_id handles the difference transparently.
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.config import MLFLOW_TRACKING_URI


def evaluate(x_test, y_test, run_id: str) -> tuple:
    """
    Load model from MLflow run, predict, log metrics back to same run.

    Parameters
    ----------
    x_test   : array-like – features (encoded for manual; raw for pipeline)
    y_test   : array-like – true labels
    run_id   : str        – MLflow run ID returned by train_manual / train_pipeline

    Returns
    -------
    (accuracy, precision, recall) as floats
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(x_test)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, average="macro")
    rec   = recall_score(y_test, preds, average="macro")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)

    print(f"Evaluation | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")
    return acc, prec, rec
