"""
Session 05 – Evaluation (No-Pipeline approach)
"""

import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score


def evaluate(x_test_enc, y_test, run_id):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

    preds = model.predict(x_test_enc)
    acc   = accuracy_score(y_test, preds)
    prec  = precision_score(y_test, preds, average="macro")
    rec   = recall_score(y_test, preds, average="macro")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy",  acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall",    rec)

    print(f"Evaluation | Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f}")
    return acc, prec, rec
