"""
Session 05 – Manual preprocessing orchestrator (Approach A).
Thin composition layer: impute → encode.
"""

from src.features.preprocessor import impute_features, encode_features


def run_preprocessing(x_train, x_test, num_features: list, cat_features: list):
    """
    Run full manual preprocessing pipeline on train/test splits.
    Steps: missing-value imputation → ordinal encoding.
    Returns (x_train_encoded, x_test_encoded).
    """
    x_train, x_test = impute_features(x_train, x_test, num_features, cat_features)
    x_train, x_test = encode_features(x_train, x_test, cat_features)
    return x_train, x_test
