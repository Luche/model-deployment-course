from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from config.config import RF_CRITERION, RF_MAX_DEPTH
from src.features.pipeline_preprocessor import build_preprocessor


def build_churn_pipeline(num_features: list, cat_features: list) -> Pipeline:
    preprocessor = build_preprocessor(num_features, cat_features)
    rf = RandomForestClassifier(criterion=RF_CRITERION, max_depth=RF_MAX_DEPTH)
    return Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", rf),
    ])
