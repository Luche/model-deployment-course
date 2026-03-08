"""
Session 05 – sklearn ColumnTransformer definition (Approach B / Pipeline).
Returns an unfitted preprocessor; no training or I/O happens here.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from config.config import PIPELINE_CAT_ORDERS


def build_preprocessor(num_features: list, cat_features: list) -> ColumnTransformer:
    """
    Build and return an unfitted ColumnTransformer that:
      - Imputes numeric columns with the column mean.
      - Imputes categorical columns with the most-frequent value,
        then ordinal-encodes them using PIPELINE_CAT_ORDERS.

    Parameters
    ----------
    num_features : list of str  – numeric column names (post-rename)
    cat_features : list of str  – categorical column names (post-rename)

    Returns
    -------
    ColumnTransformer (unfitted)
    """
    numeric_preprocess = Pipeline([
        ("num_imputer", SimpleImputer(strategy="median")),
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
