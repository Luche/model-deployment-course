"""
Session 12 – OOP: Wine Quality Classification (Training)

Refactors the functional Session 10 wine pipeline into OOP.

Run:
    uv run wine_training.py or
    python wine_training.py
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")


DATA_URL = (
    "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"
    "winequality-red.csv"
)
DATA_URL_FALLBACK = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)

FEATURE_NAMES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "pH", "sulphates", "alcohol",
]

CLASS_NAMES = ["low", "medium", "high"]


def _bin_quality(q: int) -> int:
    """Collapse 3-8 quality scores → low(0) / medium(1) / high(2)."""
    if q <= 5:
        return 0
    if q == 6:
        return 1
    return 2


@dataclass
class TrainingConfig:
    """All training hyperparameters in one place.

    Pass one config object around instead of scattering kwargs across methods.
    Swappable, printable, and trivially serializable (it's just a dataclass).
    """
    test_size: float = 0.2
    random_state: int = 42
    metric: str = "test_accuracy"       # which metric to pick the winner by
    model_path: str = "wine_model.pkl"  # where to write the trained artifact


class WineDataHandler:
    """Downloads and prepares the UCI Wine Quality dataset.

    __init__ stores configuration only — zero I/O.
    Call load_and_prepare() explicitly; this makes the object testable
    (you can swap in a different URL or mock the download without patching __init__).
    """

    def __init__(self, config: TrainingConfig, url: str | None = None):
        self.config = config
        self.url = url or DATA_URL
        self.X_train: pd.DataFrame | None = None
        self.X_test: pd.DataFrame | None = None
        self.y_train: pd.Series | None = None
        self.y_test: pd.Series | None = None
        self._n_samples: int | None = None

    def load_and_prepare(self) -> None:
        """Download → bin quality labels → stratified train/test split."""
        df = self._download()
        self._n_samples = len(df)

        X = df[FEATURE_NAMES]
        y = df["quality"].apply(_bin_quality)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

    def _download(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.url, header=None, names=FEATURE_NAMES + ["quality"])
            return df
        except Exception as e:
            print(f"Primary URL failed ({e}); trying fallback...")
            df = pd.read_csv(DATA_URL_FALLBACK, sep=";")
            df.columns = FEATURE_NAMES + ["quality"]
            return df

    def __repr__(self) -> str:
        if self.X_train is None:
            return "WineDataHandler(status=not loaded)"
        return (
            f"WineDataHandler("
            f"n_samples={self._n_samples}, "
            f"train={len(self.X_train)}, "
            f"test={len(self.X_test)})"
        )


class WineModelTrainer:
    """Trains, compares, and saves candidate wine quality classifiers.

    Receives a WineDataHandler via dependency injection — it doesn't know
    or care where the data comes from (URL, local file, database, mock).
    This makes the trainer independently testable and reusable.

    Typical usage (step-by-step, when you need intermediate results):
        trainer = WineModelTrainer(data, config)
        trainer.build()
        trainer.train()
        trainer.compare()
        trainer.select_best()
        trainer.save()

    Or use the factory classmethod for a one-liner:
        trainer = WineModelTrainer.train_and_save(data, config)
    """

    def __init__(self, data: WineDataHandler, config: TrainingConfig):
        # __init__ stores references only — no training, no I/O
        self.data = data
        self.config = config
        self._pipelines: dict[str, Pipeline] = {}
        self._results: dict[str, dict] = {}
        self.best_name: str | None = None
        self.best_pipeline: Pipeline | None = None

    def build(self) -> None:
        """Construct the three candidate sklearn Pipelines."""
        self._pipelines = {
            "logistic_regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(random_state=self.config.random_state)),
            ]),
            "knn": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", KNeighborsClassifier(n_neighbors=15)),
            ]),
            # XGBoost is tree-based — doesn't need scaling
            "xgboost": Pipeline([
                ("clf", XGBClassifier(random_state=self.config.random_state)),
            ]),
        }

    def train(self) -> None:
        """Fit every pipeline and record evaluation metrics."""
        if not self._pipelines:
            raise RuntimeError("Call build() before train().")

        X_train, y_train = self.data.X_train, self.data.y_train
        X_test, y_test = self.data.X_test, self.data.y_test

        for name, pipeline in self._pipelines.items():
            print(f"  Training {name}...")
            pipeline.fit(X_train, y_train)

            self._results[name] = {
                "train_accuracy": accuracy_score(y_train, pipeline.predict(X_train)),
                "test_accuracy":  accuracy_score(y_test,  pipeline.predict(X_test)),
                "test_macro_f1":  f1_score(y_test, pipeline.predict(X_test), average="macro"),
            }

    def compare(self) -> None:
        """Print a side-by-side metric table for all trained models."""
        if not self._results:
            raise RuntimeError("Call train() before compare().")

        print(f"\n{'Model':<22} {'Train Acc':>10} {'Test Acc':>10} {'Test F1':>10}")
        print("-" * 54)
        for name, m in self._results.items():
            print(
                f"{name:<22} "
                f"{m['train_accuracy']:>10.4f} "
                f"{m['test_accuracy']:>10.4f} "
                f"{m['test_macro_f1']:>10.4f}"
            )

    def select_best(self) -> str:
        """Pick the winner by config.metric and store it as self.best_pipeline."""
        if not self._results:
            raise RuntimeError("Call train() before select_best().")

        self.best_name = max(self._results, key=lambda n: self._results[n][self.config.metric])
        self.best_pipeline = self._pipelines[self.best_name]

        print(f"\nWinner ({self.config.metric}): {self.best_name}")
        print(f"\nDetailed report for {self.best_name}:")
        y_pred = self.best_pipeline.predict(self.data.X_test)
        print(classification_report(self.data.y_test, y_pred, target_names=CLASS_NAMES))

        return self.best_name

    def save(self) -> Path:
        """Serialize best_pipeline to disk with joblib. Returns the saved path."""
        if self.best_pipeline is None:
            raise RuntimeError("Call select_best() before save().")

        path = Path(self.config.model_path)
        joblib.dump(self.best_pipeline, path)
        print(f"Saved: {path}")
        return path

    @classmethod
    def train_and_save(cls, data: WineDataHandler, config: TrainingConfig) -> "WineModelTrainer":
        """Run the full build → train → compare → select → save pipeline.

        Use this when you want the complete flow without caring about
        intermediate state (e.g., batch jobs, CI/CD pipelines).
        Returns the trained WineModelTrainer so callers can inspect results.
        """
        trainer = cls(data, config)
        trainer.build()
        trainer.train()
        trainer.compare()
        trainer.select_best()
        trainer.save()
        return trainer

    def __repr__(self) -> str:
        trained = len(self._results)
        winner = self.best_name or "none selected"
        return f"WineModelTrainer(models_trained={trained}, best='{winner}')"


if __name__ == "__main__":
    config = TrainingConfig()

    print("Loading data...")
    data = WineDataHandler(config)
    data.load_and_prepare()
    print(data)

    print("\nTraining models...")
    trainer = WineModelTrainer(data, config)
    trainer.build()
    trainer.train()
    trainer.compare()
    trainer.select_best()
    model_path = trainer.save()

    print(f"\nDone. Run wine_predictor.py to test inference.")
    print(trainer)
