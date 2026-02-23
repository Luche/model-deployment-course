"""
Session 06-07 – FastAPI: Customer Churn Prediction Endpoint
Run with: uvicorn churn_fastapi:app --reload
Then test at: http://127.0.0.1:8000/docs

NOTE: If churn_prediction_pipeline.pkl does not exist yet, a default model is
      trained and saved automatically on first startup.
"""

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# ── Auto-train model if pkl is missing ────────────────────────────────────────
PKL_PATH = Path("churn_prediction_pipeline.pkl")

if not PKL_PATH.exists():
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    rng = np.random.default_rng(42)
    n   = 2000
    data_syn = {
        "Age":              rng.integers(18, 70, n),
        "Gender":           rng.choice(["Male", "Female"], n),
        "Tenure":           rng.integers(1, 61, n).astype(float),
        "UsageFrequency":   rng.integers(1, 30, n),
        "SupportCalls":     rng.integers(0, 11, n).astype(float),
        "PaymentDelay":     rng.integers(0, 30, n),
        "SubscriptionType": rng.choice(["Basic", "Standard", "Premium"], n),
        "ContractLength":   rng.choice(["Monthly", "Quarterly", "Annual"], n),
        "TotalSpend":       rng.integers(100, 1001, n).astype(float),
        "LastInteraction":  rng.integers(1, 30, n),
    }
    df_syn  = pd.DataFrame(data_syn)
    y_syn   = rng.choice([0, 1], n, p=[0.45, 0.55])

    num_feat = ["Age", "Tenure", "UsageFrequency", "SupportCalls",
                "PaymentDelay", "TotalSpend", "LastInteraction"]
    cat_feat = ["Gender", "SubscriptionType", "ContractLength"]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            categories=[["Male", "Female"],
                        ["Basic", "Standard", "Premium"],
                        ["Monthly", "Quarterly", "Annual"]],
            handle_unknown="use_encoded_value", unknown_value=-1)),
    ])
    preprocessor = ColumnTransformer(transformers=[
        ("num", num_pipe, num_feat),
        ("cat", cat_pipe, cat_feat),
    ], remainder="drop")

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier",    RandomForestClassifier(criterion="gini", max_depth=4, random_state=42)),
    ])
    pipeline.fit(df_syn, y_syn)
    joblib.dump(pipeline, PKL_PATH)
    print(f"[churn_fastapi] No model found – trained default pipeline and saved to {PKL_PATH}")

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load(PKL_PATH)

app = FastAPI(title="Churn Prediction API")


class ChurnFeatures(BaseModel):
    Age:              int
    Gender:           str
    Tenure:           int
    UsageFrequency:   int
    SupportCalls:     int
    PaymentDelay:     int
    SubscriptionType: str
    ContractLength:   str
    TotalSpend:       int
    LastInteraction:  int


@app.get("/")
def root():
    return {"message": "Welcome to the Churn Prediction API"}


@app.post("/predict")
def predict(churn: ChurnFeatures):
    data_in  = churn.dict()
    features = pd.DataFrame([data_in])
    prediction = model.predict(features)
    return {"prediction": int(prediction[0])}
