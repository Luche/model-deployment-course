"""
Session 04 – Step 3: Training
Trains a Random Forest classifier and logs to MLflow.
"""

import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
import joblib

def train(train_scaled):
    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("Streamlit-Pipeline")

    X_train = train_scaled.drop("species", axis=1)
    y_train = train_scaled["species"]

    with mlflow.start_run() as run:
        model = RandomForestClassifier(criterion='gini', max_depth=4)
        model.fit(X_train, y_train)

        mlflow.log_param("criterion", "gini")
        mlflow.log_param("max_depth", 4)
        mlflow.sklearn.log_model(sk_model=model, name="model")
        joblib.dump(model, "artifacts/model.pkl")
        print(f"Model trained. Run ID: {run.info.run_id}")
        return run.info.run_id


if __name__ == "__main__":
    train(None)
