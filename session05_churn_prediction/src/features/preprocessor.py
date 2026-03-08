"""
Session 05 – Manual preprocessing (Approach A / NoPipeline).
Uses the exact same sklearn transformers as the Pipeline approach (SimpleImputer,
OrdinalEncoder), but calls them step-by-step instead of wrapping them in a Pipeline.
This makes it clear that Pipeline is just a sequencing mechanism, not magic.
"""

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from config.config import (
    ARTIFACT_NUM_IMPUTER, ARTIFACT_CAT_IMPUTER, ARTIFACT_CAT_ENCODER,
)
from src.utils.io import save_artifact


def impute_features(x_train: pd.DataFrame, x_test: pd.DataFrame,
                    num_features: list, cat_features: list):
    """
    Impute missing values using the same strategy as the Pipeline approach:
      - numeric  → SimpleImputer(strategy="median")
      - categorical → SimpleImputer(strategy="most_frequent")
    Both imputers are fit on x_train only, then applied to both splits.
    Returns (x_train_imputed, x_test_imputed).
    """
    x_train = x_train.copy()
    x_test  = x_test.copy()

    num_imputer = SimpleImputer(strategy="median")
    x_train[num_features] = num_imputer.fit_transform(x_train[num_features])
    x_test[num_features]  = num_imputer.transform(x_test[num_features])

    cat_imputer = SimpleImputer(strategy="most_frequent")
    x_train[cat_features] = cat_imputer.fit_transform(x_train[cat_features])
    x_test[cat_features]  = cat_imputer.transform(x_test[cat_features])

    save_artifact(num_imputer, ARTIFACT_NUM_IMPUTER)
    save_artifact(cat_imputer, ARTIFACT_CAT_IMPUTER)
    return x_train, x_test


def encode_features(x_train: pd.DataFrame, x_test: pd.DataFrame,
                    cat_features: list):
    """
    Ordinal-encode all categorical columns using OrdinalEncoder — same transformer
    the Pipeline approach uses, just called directly without the Pipeline wrapper.
    Encoder is fit on x_train only, then applied to both splits.
    Returns (x_train_encoded, x_test_encoded).
    """
    encoder = OrdinalEncoder()
    x_train[cat_features] = encoder.fit_transform(x_train[cat_features])
    x_test[cat_features]  = encoder.transform(x_test[cat_features])

    save_artifact(encoder, ARTIFACT_CAT_ENCODER)
    return x_train, x_test
