"""
Session 05 – Centralized artifact I/O.
All joblib save/load operations go through these helpers.
"""

from pathlib import Path
import joblib


def save_artifact(obj, path: Path) -> None:
    """Create parent directories if needed, then dump obj to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path):
    """Load a joblib artifact, raising a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Run the corresponding training pipeline first."
        )
    return joblib.load(path)


def load_manual_artifacts(model_path: Path, num_imputer_path: Path,
                           cat_imputer_path: Path, cat_encoder_path: Path) -> tuple:
    """Load all four manual-approach artifacts in one call."""
    return (
        load_artifact(model_path),
        load_artifact(num_imputer_path),
        load_artifact(cat_imputer_path),
        load_artifact(cat_encoder_path),
    )
