from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

from config.config import PIPELINE_CAT_ORDERS


def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    numeric_preprocess = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median")),
        ("num_scaler", StandardScaler())
    ])

    categorical_preprocess = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="most_frequent")),
        ("cat_encoder", OrdinalEncoder()),
    ])

    return ColumnTransformer(
        transformers=[
            ("numPreprocess", numeric_preprocess, num_features),
            ("catPreprocess", categorical_preprocess, cat_features),
        ],
        remainder="drop",
    )
