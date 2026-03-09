import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config.config import (
    DATA_RAW_DIR, DATA_ING_DIR, TARGET_COL, DROP_COLS,
    PIPELINE_RENAME_MAP, RANDOM_STATE, TEST_SIZE,
)


def generate_synthetic_churn(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data = {
        "CustomerID":         range(1, n + 1),
        "Age":                rng.integers(18, 70, n),
        "Gender":             rng.choice(["Male", "Female", None],  n, p=[0.55, 0.44, 0.01]),
        "Tenure":             rng.choice(list(range(1, 61)) + [None], n),
        "Usage Frequency":    rng.integers(1, 30, n),
        "Support Calls":      rng.choice(list(range(0, 11)) + [None], n),
        "Payment Delay":      rng.integers(0, 30, n),
        "Subscription Type":  rng.choice(["Basic", "Standard", "Premium"], n),
        "Contract Length":    rng.choice(["Monthly", "Quarterly", "Annual"], n),
        "Total Spend":        rng.choice(list(np.arange(100, 1001, 10).astype(int)) + [None], n),
        "Last Interaction":   rng.integers(1, 30, n),
        "Churn":              rng.choice([0, 1], n, p=[0.45, 0.55]),
    }
    return pd.DataFrame(data)


def ingest_data() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_ING_DIR.mkdir(parents=True, exist_ok=True)

    raw_file = DATA_RAW_DIR / "customer_churn.csv"
    if not raw_file.exists():
        print("customer_churn.csv not found – generating synthetic data...")
        df = generate_synthetic_churn()
        df.to_csv(raw_file, sep=";", index=False)
        print(f"Synthetic data saved to {raw_file}")

    df = pd.read_csv(raw_file, sep=";")
    assert not df.empty, "Dataset is empty"

    out_file = DATA_ING_DIR / "customer_churn.csv"
    df.to_csv(out_file, sep=";", index=False)
    print(f"Data ingested: {raw_file} → {out_file}")


def load_frame(rename_for_pipeline: bool = False) -> pd.DataFrame:
    path = DATA_ING_DIR / "customer_churn.csv"
    df = pd.read_csv(path, sep=";")
    if rename_for_pipeline:
        df = df.rename(columns=PIPELINE_RENAME_MAP)
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(DROP_COLS + [TARGET_COL], axis=1)
    y = df[TARGET_COL]
    return X, y


def split_train_test(X: pd.DataFrame, y: pd.Series):
    return train_test_split(X, y, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE, stratify=y)
