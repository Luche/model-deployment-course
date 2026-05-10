"""Build sklearn pipelines for each candidate model.

Each pipeline has the same shape: StandardScaler -> classifier.
This makes them interchangeable in training, evaluation, and inference.

Standardization is essential for Logistic Regression and KNN (both are
distance/scale sensitive). It's harmless for tree-based models like XGBoost.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_pipelines() -> dict[str, Pipeline]:
    """Return a dict of {name: pipeline} for all candidate models."""
    return {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000,
                random_state=42,
            )),
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=15)),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="mlogloss",
                random_state=42,
            )),
        ]),
    }
