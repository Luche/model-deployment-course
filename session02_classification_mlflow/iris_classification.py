"""
Session 02 - Part 1: Iris Dataset Classification
Topics: Logistic Regression, SVM, Random Forest, Optuna Hyperparameter Optimization
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
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ── 1. Load and Explore Data ────────────────────────────────────────────────
iris = load_iris()
X = iris.data
y = iris.target

df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y
df['species'] = df['target'].map({0: iris.target_names[0],
                                   1: iris.target_names[1],
                                   2: iris.target_names[2]})

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['species'].value_counts())

# ── 2. Data Preparation ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size:     {X_test.shape[0]}")

# ── 3. Logistic Regression ───────────────────────────────────────────────────
lr_model = LogisticRegression(random_state=42)
lr_model.fit(X_train_scaled, y_train)
y_pred_lr = lr_model.predict(X_test_scaled)

print("\n=== Logistic Regression Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(classification_report(y_test, y_pred_lr, target_names=iris.target_names))

# ── 4. Support Vector Machine (SVM) ─────────────────────────────────────────
svm_model = SVC(random_state=42)
svm_model.fit(X_train_scaled, y_train)
y_pred_svm = svm_model.predict(X_test_scaled)

print("\n=== SVM Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(classification_report(y_test, y_pred_svm, target_names=iris.target_names))

# ── 5. Random Forest ─────────────────────────────────────────────────────────
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
y_pred_rf = rf_model.predict(X_test_scaled)

print("\n=== Random Forest Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(classification_report(y_test, y_pred_rf, target_names=iris.target_names))

# ── 6. Random Forest + Optuna Optimization ───────────────────────────────────
def objective(trial):
    max_depth        = trial.suggest_int('max_depth', 2, 10)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)
    min_samples_leaf  = trial.suggest_int('min_samples_leaf', 1, 10)
    max_features     = trial.suggest_categorical('max_features', ['sqrt', 'log2', None])
    criterion        = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])

    model = RandomForestClassifier(
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
    )
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    return scores.mean()

print("\nStarting Optuna optimization (50 trials)...")
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction='maximize', study_name='rf_optimization')
study.optimize(objective, n_trials=50)

print(f"Best CV accuracy: {study.best_trial.value:.4f}")
print(f"Best params:      {study.best_trial.params}")

best_params = study.best_trial.params
rf_optimized = RandomForestClassifier(**best_params, random_state=42, n_jobs=-1)
rf_optimized.fit(X_train_scaled, y_train)
y_pred_rf_opt = rf_optimized.predict(X_test_scaled)

print("\n=== Random Forest (Optimized) Results ===")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_opt):.4f}")
print(classification_report(y_test, y_pred_rf_opt, target_names=iris.target_names))

# ── 7. Model Comparison ──────────────────────────────────────────────────────
results = pd.DataFrame({
    'Model': ['Logistic Regression', 'SVM', 'Random Forest', 'RF (Optuna)'],
    'Accuracy': [
        accuracy_score(y_test, y_pred_lr),
        accuracy_score(y_test, y_pred_svm),
        accuracy_score(y_test, y_pred_rf),
        accuracy_score(y_test, y_pred_rf_opt),
    ]
}).sort_values('Accuracy', ascending=False)

print("\n=== Model Comparison ===")
print(results.to_string(index=False))

# ── 8. Confusion Matrices ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()
models_list = [
    ('Logistic Regression', y_pred_lr),
    ('SVM', y_pred_svm),
    ('Random Forest', y_pred_rf),
    ('RF (Optimized)', y_pred_rf_opt),
]
for idx, (name, y_pred) in enumerate(models_list):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=iris.target_names, yticklabels=iris.target_names)
    axes[idx].set_title(f'{name}\nAccuracy: {accuracy_score(y_test, y_pred):.4f}')
    axes[idx].set_ylabel('True Label')
    axes[idx].set_xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.show()
print("Confusion matrices saved to confusion_matrices.png")

if __name__ == "__main__":
    print("\nAll done! Session 02 - Part 1 completed successfully.")
