"""
Session 05 – Training (No-Pipeline approach)
"""

import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib

def train(x_train_enc, y_train):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("Churn-NoPipeline")

    with mlflow.start_run() as run:
        model = RandomForestClassifier(criterion='gini', max_depth=4)
        model.fit(x_train_enc, y_train)

        mlflow.log_param("criterion", "gini")
        mlflow.log_param("max_depth", 4)
        mlflow.sklearn.log_model(model, name="model")
        joblib.dump(model, 'artifacts/model_churnNoPipeline.pkl')
        print(f"Model trained. Run ID: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    pass
