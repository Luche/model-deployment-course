"""
Session 02 - Part 2: MLflow Experiment Tracking & Model Registry
Topics: mlflow.log_param/metric/artifact, mlflow.sklearn.log_model,
        Model Registry with aliases (champion / challenger)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

import optuna
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

np.random.seed(42)

# ── Setup ─────────────────────────────────────────────────────────────────────
mlflow.set_tracking_uri("sqlite:///../mlflow.db")
experiment_name = "iris-classification-demo"
mlflow.set_experiment(experiment_name)

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Experiment: {experiment_name}")
print("Run `mlflow ui` to view the dashboard at http://localhost:5000")

# ── Data ──────────────────────────────────────────────────────────────────────
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── PART 1: Baseline runs ─────────────────────────────────────────────────────
def log_baseline_model(model, run_name, model_type, params, X_tr, X_te, y_te,
                        fig_name):
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_type", model_type)
        mlflow.set_tag("dataset", "iris")
        mlflow.log_params(params)

        model.fit(X_tr, y_train)
        y_pred = model.predict(X_te)
        acc = accuracy_score(y_te, y_pred)
        mlflow.log_metric("accuracy", acc)

        # Confusion matrix artifact
        cm = confusion_matrix(y_te, y_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=iris.target_names,
                    yticklabels=iris.target_names)
        plt.title(f'{run_name} – Confusion Matrix')
        plt.ylabel('True'); plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(fig_name)
        mlflow.log_artifact(fig_name)
        plt.close()

        sig = infer_signature(X_tr, model.predict(X_tr))
        mlflow.sklearn.log_model(model, name="model", signature=sig)

        print(f"[{run_name}] Run ID: {run.info.run_id} | Accuracy: {acc:.4f}")
        return run.info.run_id, y_pred

lr_model  = LogisticRegression(random_state=42)
svm_model = SVC(random_state=42)
rf_model  = RandomForestClassifier(random_state=42)

_, y_pred_lr  = log_baseline_model(lr_model,  "logistic_regression_baseline",
    "logistic_regression", {"random_state": 42, "solver": "lbfgs", "C": 1.0},
    X_train_scaled, X_test_scaled, y_test, "lr_confusion_matrix.png")

_, y_pred_svm = log_baseline_model(svm_model, "svm_baseline",
    "svm", {"random_state": 42, "kernel": "rbf", "C": 1.0},
    X_train_scaled, X_test_scaled, y_test, "svm_confusion_matrix.png")

_, y_pred_rf  = log_baseline_model(rf_model,  "random_forest_baseline",
    "random_forest", {"random_state": 42, "n_estimators": 100},
    X_train_scaled, X_test_scaled, y_test, "rf_baseline_confusion_matrix.png")

# ── PART 2: Optuna + MLflow nested runs ───────────────────────────────────────
def objective(trial):
    with mlflow.start_run(run_name=f"rf_trial_{trial.number}", nested=True):
        params = {
            "max_depth":          trial.suggest_int('max_depth', 2, 10),
            "min_samples_split":  trial.suggest_int('min_samples_split', 2, 10),
            "min_samples_leaf":   trial.suggest_int('min_samples_leaf', 1, 10),
            "max_features":       trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            "criterion":          trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss']),
        }
        mlflow.log_params(params)
        mlflow.set_tag("model_type", "random_forest")
        mlflow.set_tag("trial_number", trial.number)

        model = RandomForestClassifier(**params, random_state=42)
        scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric("cv_accuracy_mean", scores.mean())
        mlflow.log_metric("cv_accuracy_std", scores.std())
        return scores.mean()

print("\nRunning Optuna optimization with MLflow nested runs...")
with mlflow.start_run(run_name="random_forest_optimization") as parent_run:
    mlflow.set_tag("optimization", "optuna")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    mlflow.log_param("n_trials", 50)
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_cv_accuracy", study.best_value)
    print(f"Parent Run ID: {parent_run.info.run_id}")
    print(f"Best CV accuracy: {study.best_value:.4f}")

# ── PART 3: Final model + Model Registry ─────────────────────────────────────
MODEL_NAME = "iris-random-forest-optimized"

with mlflow.start_run(run_name="random_forest_final_optimized") as final_run:
    mlflow.set_tag("status", "production_candidate")
    best_params = study.best_trial.params
    rf_opt = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
    rf_opt.fit(X_train_scaled, y_train)

    y_pred_train = rf_opt.predict(X_train_scaled)
    y_pred_rf_opt = rf_opt.predict(X_test_scaled)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test, y_pred_rf_opt)

    mlflow.log_params(best_params)
    mlflow.log_metric("train_accuracy", train_acc)
    mlflow.log_metric("test_accuracy",  test_acc)

    # Feature importance artifact
    fi = pd.DataFrame({'Feature': iris.feature_names,
                        'Importance': rf_opt.feature_importances_
                        }).sort_values('Importance', ascending=False)
    plt.figure(figsize=(8, 4))
    plt.barh(fi['Feature'], fi['Importance'], color='#3498db')
    plt.xlabel('Importance'); plt.title('Feature Importance – Optimized RF')
    plt.gca().invert_yaxis(); plt.tight_layout()
    plt.savefig('feature_importance.png')
    mlflow.log_artifact('feature_importance.png')
    plt.close()

    sig = infer_signature(X_train_scaled, rf_opt.predict(X_train_scaled))
    mlflow.sklearn.log_model(rf_opt, name="model", signature=sig,
                              input_example=X_train_scaled[:5],
                              registered_model_name=MODEL_NAME)

    final_run_id = final_run.info.run_id
    print(f"\nFinal Run ID: {final_run_id}")
    print(f"Train accuracy: {train_acc:.4f} | Test accuracy: {test_acc:.4f}")

# ── PART 4: Model Registry – aliases ─────────────────────────────────────────
client = MlflowClient()

# Update model description
client.update_registered_model(
    name=MODEL_NAME,
    description="RF classifier for Iris, optimized with Optuna (50 trials)."
)

versions = client.search_model_versions(f"name='{MODEL_NAME}'")
latest_v = max(versions, key=lambda x: int(x.version))
print(f"\nLatest version: {latest_v.version}")

# Set challenger alias
client.set_registered_model_alias(MODEL_NAME, "challenger", latest_v.version)
print(f"Alias 'challenger' set on version {latest_v.version}")

# Validate loaded model
model_uri = f"models:/{MODEL_NAME}@challenger"
loaded = mlflow.sklearn.load_model(model_uri)
loaded_acc = accuracy_score(y_test, loaded.predict(X_test_scaled))
print(f"Loaded challenger accuracy: {loaded_acc:.4f}")

# Promote to champion
try:
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    print(f"Existing champion: version {champion.version}")
except Exception:
    print("No existing champion – first promotion.")

client.set_registered_model_alias(MODEL_NAME, "champion", latest_v.version)
client.delete_registered_model_alias(MODEL_NAME, "challenger")
print(f"Version {latest_v.version} is now the 'champion'!")

# Add tags
client.set_model_version_tag(MODEL_NAME, latest_v.version, "validation_status", "passed")
client.set_model_version_tag(MODEL_NAME, latest_v.version, "approved_by", "data-science-team")
client.set_registered_model_tag(MODEL_NAME, "task", "classification")

# Load production model
prod_model_uri = f"models:/{MODEL_NAME}@champion"
prod_model = mlflow.sklearn.load_model(prod_model_uri)
preds = prod_model.predict(X_test_scaled[:5])
print(f"\nProduction model sample predictions: {preds}")
print(f"Predicted species: {[iris.target_names[p] for p in preds]}")

# ── PART 5: Model Comparison ──────────────────────────────────────────────────
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'RF (Optimized)'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_rf_opt),
    ]
}).sort_values('Accuracy', ascending=False)

print("\n=== Final Model Comparison ===")
print(results.to_string(index=False))

if __name__ == "__main__":
    print("\nSession 02 – Part 2 completed. Run `mlflow ui` to see all runs.")
