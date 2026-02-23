"""
Session 05 – Training (sklearn Pipeline approach)
Preprocessing + classifier are bundled into a single sklearn Pipeline.
"""

import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier


def train_model(x_train, y_train):
    os.makedirs("artifacts", exist_ok=True)

    cat_feat = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
    num_feat = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()

    numeric_preprocess = Pipeline([
        ('num_imputer', SimpleImputer(strategy='mean'))
    ])

    categorical_preprocess = Pipeline([
        ('cat_imputer', SimpleImputer(strategy='most_frequent')),
        ('cat_encoder', OrdinalEncoder(
            categories=[['Male', 'Female'],
                        ['Basic', 'Standard', 'Premium'],
                        ['Monthly', 'Quarterly', 'Annual']]
        )),
    ])

    preprocess = ColumnTransformer(transformers=[
        ('numPreprocess', numeric_preprocess, num_feat),
        ('catPreprocess', categorical_preprocess,
         ['Gender', 'SubscriptionType', 'ContractLength']),
    ], remainder='drop')

    churn_pred = Pipeline([
        ('preprocessing', preprocess),
        ('classifier', RandomForestClassifier(criterion='gini', max_depth=4)),
    ])

    mlflow.set_tracking_uri("sqlite:///../mlflow.db")
    mlflow.set_experiment("Customer Churn Prediction")

    with mlflow.start_run() as run:
        mlflow.log_param("criterion", "gini")
        mlflow.log_param("max_depth", 4)

        churn_pred.fit(x_train, y_train)

        joblib.dump(churn_pred, "artifacts/churn_prediction_pipeline.pkl")
        mlflow.sklearn.log_model(churn_pred, name="model")
        print(f"Pipeline trained. Run ID: {run.info.run_id}")

    return run.info.run_id
