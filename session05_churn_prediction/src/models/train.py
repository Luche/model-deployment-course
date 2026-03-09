import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier

from config.config import (
    RF_CRITERION, RF_MAX_DEPTH,
    MLFLOW_TRACKING_URI, MLFLOW_EXP_MANUAL, MLFLOW_EXP_PIPELINE,
    ARTIFACT_MODEL_MANUAL, ARTIFACT_PIPELINE,
)
from src.utils.io import save_artifact


def _setup_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)


def train_manual(x_train_enc, y_train) -> str:
    _setup_mlflow(MLFLOW_EXP_MANUAL)

    with mlflow.start_run() as run:
        model = RandomForestClassifier(criterion=RF_CRITERION, max_depth=RF_MAX_DEPTH)
        model.fit(x_train_enc, y_train)

        mlflow.log_param("criterion", RF_CRITERION)
        mlflow.log_param("max_depth", RF_MAX_DEPTH)
        mlflow.sklearn.log_model(model, name="model")
        save_artifact(model, ARTIFACT_MODEL_MANUAL)

        print(f"Model trained. Run ID: {run.info.run_id}")
        return run.info.run_id


def train_pipeline(pipeline, x_train, y_train) -> str:
    _setup_mlflow(MLFLOW_EXP_PIPELINE)

    with mlflow.start_run() as run:
        mlflow.log_param("criterion", RF_CRITERION)
        mlflow.log_param("max_depth", RF_MAX_DEPTH)

        pipeline.fit(x_train, y_train)

        mlflow.sklearn.log_model(pipeline, name="model")
        save_artifact(pipeline, ARTIFACT_PIPELINE)

        print(f"Pipeline trained. Run ID: {run.info.run_id}")
        return run.info.run_id
