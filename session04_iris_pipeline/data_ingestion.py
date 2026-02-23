"""
Session 04 – Step 1: Data Ingestion
Reads raw IRIS.csv and saves it to the ingested/ folder.
"""

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris

BASE_DIR    = Path(__file__).parent
INGESTED_DIR = BASE_DIR / "ingested"
INPUT_FILE  = BASE_DIR / "IRIS.csv"
OUTPUT_FILE = INGESTED_DIR / "IRIS.csv"


def ingest_data():
    INGESTED_DIR.mkdir(parents=True, exist_ok=True)

    # If IRIS.csv is not present, generate it from sklearn
    if not INPUT_FILE.exists():
        print("IRIS.csv not found – generating from sklearn datasets...")
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=['sepal_length', 'sepal_width',
                                               'petal_length', 'petal_width'])
        df['species'] = [iris.target_names[t] for t in iris.target]
        df.to_csv(INPUT_FILE, index=False)
        print(f"Generated IRIS.csv at {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    assert not df.empty, "Dataset is empty"
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Data ingested: {INPUT_FILE} → {OUTPUT_FILE}")


if __name__ == "__main__":
    ingest_data()
