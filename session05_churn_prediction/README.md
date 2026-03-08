# Session 05 – Customer Churn Prediction

Two ML workflows side-by-side: **manual preprocessing** (Approach A) vs **sklearn Pipeline** (Approach B).

---

## Project Structure

```
session05_churn_prediction/
├── config/config.py          ← all constants (paths, hyperparams, columns, MLflow)
├── src/
│   ├── data/loader.py        ← data generation, ingestion, train/test split
│   ├── features/
│   │   ├── preprocessor.py          ← manual impute + encode (Approach A)
│   │   └── pipeline_preprocessor.py ← sklearn ColumnTransformer (Approach B)
│   ├── pipelines/
│   │   ├── manual_pipeline.py  ← orchestrates manual preprocessing steps
│   │   └── sklearn_pipeline.py ← assembles ColumnTransformer + RandomForest
│   ├── models/
│   │   ├── train.py          ← train_manual() and train_pipeline()
│   │   └── evaluate.py       ← evaluate() works for both approaches
│   └── utils/io.py           ← centralized joblib save/load helpers
├── apps/
│   ├── app_manual.py         ← Streamlit UI for Approach A
│   └── app_pipeline.py       ← Streamlit UI for Approach B
├── main_manual.py            ← entry point: Approach A
├── main_pipeline.py          ← entry point: Approach B
└── requirements.txt
```

Runtime-generated (not committed):
```
├── data/raw/customer_churn.csv
├── data/ingested/customer_churn.csv
└── artifacts/*.pkl
```

---

## Setup

From the **repo root** (`Code Model Deployment/`):

```bash
uv sync
```

If you don't have `uv`, install it first:
```bash
curl -Lsf https://astral.sh/uv/install.sh | sh
```

---

## Running the Training Pipelines

Run both commands from the **repo root** (`Code Model Deployment/`).

### Approach A — Manual Preprocessing

```bash
uv run python model_deployment_lab/session05_churn_prediction/main_manual.py
```

Steps executed:
1. Ingest data (generates synthetic CSV if none exists)
2. Load CSV and split into features / target
3. Train/test split
4. Manual imputation + ordinal encoding
5. Train RandomForestClassifier → logged to MLflow experiment `Churn-NoPipeline`
6. Evaluate and apply quality gate

Artifacts saved to `artifacts/`:
- `model_churnNoPipeline.pkl`
- `impute_stats.pkl`
- `ordinal_encode_subs.pkl`
- `ordinal_encode_cont.pkl`

### Approach B — sklearn Pipeline

```bash
uv run python model_deployment_lab/session05_churn_prediction/main_pipeline.py
```

Steps executed:
1. Ingest data
2. Load CSV (with column rename) and split
3. Build + train end-to-end sklearn Pipeline → logged to MLflow experiment `Customer Churn Prediction`
4. Evaluate and apply quality gate

Artifact saved to `artifacts/`:
- `churn_prediction_pipeline.pkl`

---

## Running the Streamlit Apps

Run the training pipelines first so the artifact files exist.

### Approach A app

```bash
uv run streamlit run model_deployment_lab/session05_churn_prediction/apps/app_manual.py
```

### Approach B app

```bash
uv run streamlit run model_deployment_lab/session05_churn_prediction/apps/app_pipeline.py
```

Both apps open in your browser at `http://localhost:8501`.

---

## Viewing MLflow Experiments

```bash
uv run mlflow ui --backend-store-uri sqlite:///model_deployment_lab/mlflow.db
```

Then open `http://localhost:5000`. Both experiments (`Churn-NoPipeline` and `Customer Churn Prediction`) will appear with their logged params and metrics.

---

## Bringing Your Own Data

Replace the auto-generated synthetic CSV with a real one before running:

```
model_deployment_lab/session05_churn_prediction/data/raw/customer_churn.csv
```

The file must be **semicolon-delimited** (`;`) with these columns:

| Column | Type | Notes |
|---|---|---|
| CustomerID | int | dropped before training |
| Age | int | |
| Gender | str | `Male` / `Female` / null |
| Tenure | int | nullable |
| Usage Frequency | int | |
| Support Calls | int | nullable |
| Payment Delay | int | |
| Subscription Type | str | `Basic` / `Standard` / `Premium` |
| Contract Length | str | `Monthly` / `Quarterly` / `Annual` |
| Total Spend | int | nullable |
| Last Interaction | int | |
| Churn | int | `0` = retained, `1` = churned |
