"""
train.py
--------
Trains an XGBoost classifier on the UCI Wine Quality (red) dataset and
packages it for SageMaker deployment.

Run this ONCE on your local machine before class:
    pip install xgboost scikit-learn pandas
    python train.py

Produces: model_artifact/model.tar.gz

Then upload to S3 in the Learner Lab account:
    aws s3 cp model_artifact/model.tar.gz s3://<your-bucket>/wine/model.tar.gz

DATASET NOTES (worth mentioning to students):
- Wine Quality has 11 physico-chemical features (acidity, sugar, alcohol, etc.)
  and a quality score 3-8.
- Classes are heavily imbalanced (most wines are 5 or 6).
- We collapse to 3 classes: low (<=5), medium (==6), high (>=7).
  This keeps the demo honest without turning into an imbalance lecture.
"""

import os
import tarfile
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Primary mirror (GitHub) is more reliable than UCI's archive,
# which sometimes returns 403 to non-browser user agents.
DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "winequality-red.csv"
)
DATA_URL_FALLBACK = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

ARTIFACT_DIR = "model_artifact"
MODEL_FILENAME = "xgboost-model.ubj"  # SageMaker XGBoost serves any file in the tarball
TARBALL_PATH = os.path.join(ARTIFACT_DIR, "model.tar.gz")

CLASS_NAMES = ["low", "medium", "high"]
FEATURE_NAMES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
]


def bin_quality(q: int) -> int:
    """Collapse 3-8 quality scores into low(0) / medium(1) / high(2)."""
    if q <= 5:
        return 0
    if q == 6:
        return 1
    return 2


def main() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print(f"Downloading dataset from {DATA_URL}")
    try:
        # jbrownlee mirror: comma-separated, no header
        df = pd.read_csv(DATA_URL, header=None, names=FEATURE_NAMES + ["quality"])
    except Exception as e:
        print(f"Primary mirror failed ({e}); trying UCI fallback.")
        # UCI original: semicolon-separated, with header
        df = pd.read_csv(DATA_URL_FALLBACK, sep=";")
        df.columns = FEATURE_NAMES + ["quality"]

    X = df[FEATURE_NAMES].values
    y = df["quality"].apply(bin_quality).values

    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: "
          f"low={int((y==0).sum())}, "
          f"medium={int((y==1).sum())}, "
          f"high={int((y==2).sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES)

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 5,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "mlogloss",
    }

    booster = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=100,
        evals=[(dtrain, "train"), (dtest, "test")],
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    y_pred = booster.predict(dtest).argmax(axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # Save in the format the SageMaker XGBoost container expects
    model_path = os.path.join(ARTIFACT_DIR, MODEL_FILENAME)
    booster.save_model(model_path)
    print(f"Saved model: {model_path}")

    with tarfile.open(TARBALL_PATH, "w:gz") as tar:
        tar.add(model_path, arcname=MODEL_FILENAME)
    print(f"Packaged tarball: {TARBALL_PATH}")
    print(
        "\nNext:\n"
        f"  aws s3 cp {TARBALL_PATH} s3://<your-bucket>/wine/model.tar.gz"
    )


if __name__ == "__main__":
    main()
