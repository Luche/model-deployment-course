import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

from config.config import (
    ARTIFACT_NUM_IMPUTER, ARTIFACT_CAT_IMPUTER, ARTIFACT_CAT_ENCODER,
)
from src.utils.io import save_artifact


def impute_features(x_train: pd.DataFrame, x_test: pd.DataFrame,
                    num_features: list, cat_features: list):
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
    encoder = OrdinalEncoder()
    x_train[cat_features] = encoder.fit_transform(x_train[cat_features])
    x_test[cat_features]  = encoder.transform(x_test[cat_features])

    save_artifact(encoder, ARTIFACT_CAT_ENCODER)
    return x_train, x_test
