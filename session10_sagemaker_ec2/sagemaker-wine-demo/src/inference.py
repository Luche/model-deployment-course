"""
inference.py
------------
SageMaker XGBoost framework container entry point for the Wine Quality model.

This is THE file to walk through with students. Everything else is plumbing;
this is the contract between your code and SageMaker's model server.

Four functions, called in this order on every request:
    model_fn   -> called ONCE when the endpoint container starts. Loads model.
    input_fn   -> called on EACH request. Deserializes the request body.
    predict_fn -> called on EACH request. Runs the model.
    output_fn  -> called on EACH request. Serializes the response.

Why the separation? Container reuse. The container loads the model once,
then handles thousands of requests. model_fn runs once; the other three
run per request. Conflating them is the most common beginner mistake.

The model expects 11 features in this order:
    fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
    free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol

Output is a 3-class prediction: 0=low, 1=medium, 2=high quality.
"""

import json
import os
import numpy as np
import xgboost as xgb


JSON_CONTENT_TYPE = "application/json"
CSV_CONTENT_TYPE = "text/csv"

CLASS_NAMES = ["low", "medium", "high"]

# Must match the order used in train.py (and the order students send in)
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


def model_fn(model_dir: str):
    """
    Load the model from disk. Called once per container.

    SageMaker extracts model.tar.gz into `model_dir` before calling this.
    Our tarball contains 'xgboost-model.ubj' (see train.py).
    """
    model_path = os.path.join(model_dir, "xgboost-model.ubj")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster


def input_fn(request_body, request_content_type: str):
    """
    Parse the incoming request body into a numpy array predict_fn can use.

    JSON shape (preferred):
        {"instances": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]}
    CSV shape (one row per instance, 11 comma-separated values, no header):
        7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4
    """
    if request_content_type == JSON_CONTENT_TYPE:
        payload = json.loads(request_body)
        instances = payload["instances"]
        return np.array(instances, dtype=np.float32)

    if request_content_type == CSV_CONTENT_TYPE:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")
        rows = [
            [float(x) for x in line.split(",")]
            for line in request_body.strip().splitlines()
            if line.strip()
        ]
        return np.array(rows, dtype=np.float32)

    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data: np.ndarray, model: xgb.Booster):
    """
    Run inference. Returns class probabilities, argmax class, and label name.
    """
    # Feature names must match what the model was trained with, otherwise
    # XGBoost refuses to predict. This is a safety feature, not a bug.
    dmatrix = xgb.DMatrix(input_data, feature_names=FEATURE_NAMES)
    probs = model.predict(dmatrix)            # (n_samples, 3)
    class_ids = probs.argmax(axis=1)
    labels = [CLASS_NAMES[int(i)] for i in class_ids]
    return {
        "probabilities": probs.tolist(),
        "predictions": class_ids.tolist(),
        "labels": labels,
    }


def output_fn(prediction, accept_content_type: str):
    """
    Serialize the prediction dict for the response body.
    """
    if accept_content_type == JSON_CONTENT_TYPE:
        return json.dumps(prediction), JSON_CONTENT_TYPE

    raise ValueError(f"Unsupported accept type: {accept_content_type}")
