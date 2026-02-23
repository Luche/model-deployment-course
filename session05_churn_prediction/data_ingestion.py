"""
Session 05 – Step 1: Data Ingestion (Churn)
Reads raw customer_churn.csv and saves it to the ingested/ folder.
If the CSV does not exist, synthetic data is generated for lab testing.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR     = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE   = BASE_DIR / "customer_churn.csv"
OUTPUT_FILE  = INGESTED_DIR / "customer_churn.csv"


def generate_synthetic_churn(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic customer churn dataset (semicolon-delimited)."""
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


def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        print("customer_churn.csv not found – generating synthetic data...")
        df = generate_synthetic_churn()
        df.to_csv(INPUT_FILE, sep=";", index=False)
        print(f"Synthetic data saved to {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE, sep=";")
    assert not df.empty, "Dataset is empty"
    df.to_csv(OUTPUT_FILE, sep=";", index=False)
    print(f"Data ingested: {INPUT_FILE} → {OUTPUT_FILE}")


if __name__ == "__main__":
    ingest_data()
