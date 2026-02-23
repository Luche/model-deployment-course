"""
Session 06-07 – FastAPI: Iris Prediction Endpoint
Run with: uvicorn iris_fastapi:app --reload
Then test at: http://127.0.0.1:8000/docs

NOTE: If RF_class.pkl does not exist yet, a default model is trained and saved
      automatically on first startup so the server always starts cleanly.
"""

from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# ── Auto-train model if pkl is missing ────────────────────────────────────────
PKL_PATH = Path("RF_class.pkl")

if not PKL_PATH.exists():
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    iris = load_iris()
    model_init = RandomForestClassifier(criterion='gini', max_depth=4, random_state=42)
    model_init.fit(iris.data, iris.target)
    joblib.dump(model_init, PKL_PATH)
    print(f"[iris_fastapi] No model found – trained default RF and saved to {PKL_PATH}")

# ── Load model ────────────────────────────────────────────────────────────────
model = joblib.load(PKL_PATH)

app = FastAPI(title="Iris Prediction API")

SPECIES = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}


class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width:  float
    petal_length: float
    petal_width:  float


@app.get("/")
def root():
    return {"message": "Welcome to the Iris ML Model API"}


@app.post("/predict")
def predict(iris: IrisFeatures):
    data = iris.dict()
    features = [[data['sepal_length'], data['sepal_width'],
                 data['petal_length'], data['petal_width']]]
    prediction = model.predict(features)
    return {"prediction": SPECIES[int(prediction[0])]}
