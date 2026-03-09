import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score

from config.config import MLFLOW_TRACKING_URI


def evaluate(x_test, y_test, run_id: str) -> tuple:
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
