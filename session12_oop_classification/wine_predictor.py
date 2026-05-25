"""
Session 12 – Production-Grade OOP: Wine Quality Classification (Inference)

Run AFTER wine_training.py has created wine_model.pkl.

Key production patterns demonstrated:
  - Lazy model loading: model file is read on first predict(), not in __init__
    → a long-running API server creates WinePredictor once at startup, then
      serves hundreds of requests without reloading the model
  - Typed output: PredictionResult dataclass instead of raw tuple/numpy value
  - Vectorised batch inference: one DataFrame → one predict() call, not a loop

Run:
    uv run wine_predictor.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd

# Feature order must match what the model was trained on
FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]

CLASS_NAMES = ["low", "medium", "high"]


# ── Typed output ──────────────────────────────────────────────────────────────

@dataclass
class PredictionResult:
    """Structured prediction output.

    Returning a dataclass instead of a raw value or tuple means:
      - Callers know exactly what they're getting (no positional ambiguity)
      - The shape of the output is part of the public API and can be versioned
      - Easy to serialize to JSON (asdict(result)) for REST endpoints
    """
    label: str                        # "low" / "medium" / "high"
    class_id: int                     # 0 / 1 / 2
    probabilities: dict[str, float]   # {"low": 0.12, "medium": 0.63, "high": 0.25}

    def __str__(self) -> str:
        probs = ", ".join(f"{k}: {v:.2%}" for k, v in self.probabilities.items())
        return f"Quality={self.label!r}  [{probs}]"


# ── Predictor ─────────────────────────────────────────────────────────────────

class WinePredictor:
    """Inference-only wrapper around a saved wine quality pipeline.

    __init__ receives the model path but does NOT load the file.
    The model is loaded on the first call to predict() or predict_batch()
    and cached — subsequent calls reuse the in-memory object.

    Why lazy loading matters in production:
      - A web server instantiates WinePredictor at startup (fast)
      - The file is read from disk exactly once, no matter how many requests come in
      - If the model file is missing, the error surfaces at predict-time with a
        clear message, not silently at import time
    """

    def __init__(self, model_path: str = "wine_model.pkl"):
        self.model_path = Path(model_path)
        self._model = None  # intentionally None — loaded lazily

    # ── Public API ──

    def predict(self, features: dict[str, float]) -> PredictionResult:
        """Predict quality for a single wine sample.

        Args:
            features: dict mapping feature name → value, e.g.
                      {"alcohol": 11.2, "pH": 3.4, ...}
                      Named keys catch column-order mistakes that raw lists hide.

        Returns:
            PredictionResult with label, class_id, and per-class probabilities.
        """
        df = pd.DataFrame([features])[FEATURE_NAMES]
        return self._predict_dataframe(df)[0]

    def predict_batch(self, samples: list[dict[str, float]]) -> list[PredictionResult]:
        """Predict quality for multiple wine samples in one vectorised call.

        Builds a single DataFrame and calls predict/predict_proba once.
        This is NOT a loop over predict() — looping defeats the vectorisation
        that makes sklearn/XGBoost fast and is a common production mistake.

        Args:
            samples: list of feature dicts, same format as predict().

        Returns:
            list of PredictionResult, one per sample, same order as input.
        """
        df = pd.DataFrame(samples)[FEATURE_NAMES]
        return self._predict_dataframe(df)

    # ── Private helpers ──

    @property
    def _loaded_model(self):
        """Load the model from disk on first access; return cached copy after that."""
        if self._model is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Model file not found: {self.model_path}\n"
                    "Run wine_training.py first to generate it."
                )
            self._model = joblib.load(self.model_path)
            print(f"Model loaded from {self.model_path}")
        return self._model

    def _predict_dataframe(self, df: pd.DataFrame) -> list[PredictionResult]:
        model = self._loaded_model
        class_ids = model.predict(df).tolist()
        proba_matrix = model.predict_proba(df).tolist()

        results = []
        for class_id, proba_row in zip(class_ids, proba_matrix):
            results.append(PredictionResult(
                label=CLASS_NAMES[class_id],
                class_id=class_id,
                probabilities={name: round(p, 4) for name, p in zip(CLASS_NAMES, proba_row)},
            ))
        return results

    def __repr__(self) -> str:
        status = "loaded" if self._model is not None else "not loaded"
        return f"WinePredictor(model='{self.model_path}', status={status})"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    predictor = WinePredictor("wine_model.pkl")
    print(f"Created: {predictor}")   # status=not loaded (model not on disk yet)

    # Single prediction — model loads lazily on this call
    sample = {
        "fixed_acidity": 7.4,
        "volatile_acidity": 0.70,
        "citric_acid": 0.00,
        "residual_sugar": 1.9,
        "chlorides": 0.076,
        "free_sulfur_dioxide": 11.0,
        "total_sulfur_dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4,
    }

    result = predictor.predict(sample)
    print(f"\nSingle prediction:  {result}")
    print(f"Predictor state:    {predictor}")   # status=loaded

    # Batch prediction — reuses the already-loaded model, no extra disk I/O
    batch = [
        {**sample, "alcohol": 9.4},    # same wine, low-ish alcohol
        {**sample, "alcohol": 12.5},   # same wine, higher alcohol
        {**sample, "alcohol": 14.0},   # same wine, very high alcohol
    ]

    print("\nBatch prediction:")
    for i, res in enumerate(predictor.predict_batch(batch), 1):
        print(f"  Sample {i}: {res}")
