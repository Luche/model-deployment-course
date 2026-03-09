from pathlib import Path
import joblib


def save_artifact(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: Path):
    if not path.exists():
        raise FileNotFoundError(
            f"Artifact not found: {path}\n"
            "Run the corresponding training pipeline first."
        )
    return joblib.load(path)


def load_manual_artifacts(model_path: Path, num_imputer_path: Path,
                           cat_imputer_path: Path, cat_encoder_path: Path) -> tuple:
    return (
        load_artifact(model_path),
        load_artifact(num_imputer_path),
        load_artifact(cat_imputer_path),
        load_artifact(cat_encoder_path),
    )
