# Model Deployment Lab

Course lab files for the **Model Deployment** (MLOps) course.
Each session folder contains runnable Python scripts covering every topic in the course.
A single self-contained notebook (`dependency_check.ipynb`) at the root covers all sessions and is used to verify the lab environment.

---

## Project Structure

```
model_deployment_lab/
├── requirements.txt
├── dependency_check.ipynb
│
├── session02_classification_mlflow/
│   ├── iris_classification.py          # LR, SVM, RF, Optuna tuning
│   └── iris_mlflow_tracking.py         # MLflow experiment tracking + Model Registry
│
├── session04_iris_pipeline/
│   ├── data_ingestion.py               # Step 1 – load and save raw data
│   ├── pre_processing.py               # Step 2 – scale and split
│   ├── train.py                        # Step 3 – train RF with MLflow
│   ├── evaluation.py                   # Step 4 – evaluate and log metrics
│   ├── pipeline.py                     # Orchestrator (runs steps 1-4)
│   └── app_streamlit.py                # Streamlit UI (requires trained model)
│
├── session05_churn_prediction/
│   ├── data_ingestion.py               # Shared ingestion step
│   │
│   │   ── Approach A: Manual preprocessing ──
│   ├── preprocessing_NoPipeline.py     # Imputation + ordinal encoding functions
│   ├── train_churnNoPipeline.py        # Train RF with MLflow
│   ├── evaluation_NoPipeline.py        # Evaluate and log metrics
│   ├── pipeline_churnNoPipeline.py     # Orchestrator for Approach A
│   ├── app_churnNoPipeline.py          # Streamlit UI for Approach A
│   │
│   │   ── Approach B: sklearn Pipeline ──
│   ├── train_churnPipeline.py          # Full sklearn Pipeline + MLflow
│   ├── evaluation_Pipeline.py          # Evaluate and log metrics
│   ├── pipeline_churnPipeline.py       # Orchestrator for Approach B
│   └── app_churnPipeline.py            # Streamlit UI for Approach B
│
├── session06_07_api_serving/
│   ├── iris_fastapi.py                 # FastAPI endpoint for Iris prediction
│   ├── iris_streamlit.py               # Streamlit UI → calls FastAPI
│   ├── churn_fastapi.py                # FastAPI endpoint for Churn prediction
│   ├── churn_streamlit.py              # Streamlit UI → calls FastAPI
│
└── session12_oop_classification/
    ├── mainprogamoop.py                # DataHandler + ModelHandler OOP classes
    └── usepickle.py                    # Load and run inference from pickle
```

---

## Prerequisites

### 1. Install Dependencies

From inside the `model_deployment_lab/` folder:

```bash
pip install -r requirements.txt
```

Or with a virtual environment (recommended):

```bash
conda create -n model_deployment python=3.13
conda activate model_deployment

pip install -r requirements.txt
```

---

## Dependency Check Notebook

Before running any session scripts, verify your environment runs everything cleanly:

Open dependency_check.ipynb. Run all cells top-to-bottom. If no errors appear, the lab is (almost) ready.

---

## Session 02 – Iris Classification with MLflow

**Topics:** Logistic Regression, SVM, Random Forest, Optuna hyperparameter tuning,
MLflow experiment tracking, MLflow Model Registry (champion / challenger aliases)

```bash
# Part 1: classification models + Optuna (no MLflow)
cd session02_classification_mlflow && python iris_classification.py

# Part 2: full MLflow tracking + Model Registry
python iris_mlflow_tracking.py
```

View results in the MLflow dashboard:

```bash
mlflow ui
# Open http://localhost:5000 in your browser
```

---

## Session 04 – Iris ML Pipeline Pattern

**Topics:** Separating code into `data_ingestion → preprocessing → train → evaluation → pipeline`,
MLflow experiment tracking per step, Streamlit deployment

### Run the full pipeline (all steps in order)

```bash
cd session04_iris_pipeline && python pipeline.py
```

This automatically runs all four steps. You should see:

```
Step 1: Data Ingestion
Step 2: Preprocessing
Step 3: Training
Step 4: Evaluation
Model APPROVED for deployment  (acc=0.967)
```

### Run individual steps (Optional)

```bash
python data_ingestion.py    # creates ingested/IRIS.csv
python pre_processing.py    # creates artifacts/preprocessor.pkl
python train.py             # logs model to MLflow, prints run_id
python evaluation.py        # loads model from MLflow, logs metrics
```

> **Note:** `IRIS.csv` is auto-generated from sklearn if the file is not present.

### Streamlit app (requires trained model)

First run the full pipeline, then copy the saved model:

```bash
python pipeline.py
streamlit run app_streamlit.py
```

---

## Session 05 – Customer Churn Prediction

**Topics:** Manual preprocessing vs sklearn Pipeline, MLflow tracking, Streamlit deployment

> **Note:** `customer_churn.csv` is auto-generated as synthetic data if the file is not present.

### Approach A – Manual Preprocessing (No-Pipeline)

```bash
cd session05_churn_prediction
python pipeline_churnNoPipeline.py
```

Steps executed: ingest → impute missing values → ordinal encode → train RF → evaluate

```bash
# Streamlit app (run pipeline first to generate artifact pkl files)
streamlit run app_churnNoPipeline.py
```

### Approach B – sklearn Pipeline

```bash
python pipeline_churnPipeline.py
```

The entire preprocessing + classifier is bundled into one sklearn `Pipeline` object.

```bash
# Streamlit app (run pipeline first to generate churn_prediction_pipeline.pkl)
streamlit run app_churnPipeline.py
```

---

## Session 06-07 – FastAPI & Streamlit Deployment

**Topics:** REST API with FastAPI, Pydantic input validation, Streamlit frontend

### FastAPI backend + Streamlit frontend

Requires two terminals running at the same time.

**Terminal 1 – Start the FastAPI server:**

```bash
cd session06_07_api_serving

# Iris API
uvicorn iris_fastapi:app --reload

# — OR — Churn API
uvicorn churn_fastapi:app --reload
```

The API auto-trains and saves a default model pkl on first startup if none exists.
Interactive API docs available at: `http://127.0.0.1:8000/docs`

**Terminal 2 – Start the Streamlit frontend:**

```bash
cd session06_07_api_serving

# Iris UI (calls iris FastAPI)
streamlit run iris_streamlit.py

# — OR — Churn UI (calls churn FastAPI)
streamlit run churn_streamlit.py
```

### Test the FastAPI endpoint manually

```bash
# Iris prediction
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'

# Churn prediction
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"Age": 30, "Gender": "Male", "Tenure": 12, "UsageFrequency": 15,
          "SupportCalls": 2, "PaymentDelay": 5, "SubscriptionType": "Premium",
          "ContractLength": "Annual", "TotalSpend": 500, "LastInteraction": 7}'
```

---

## Session 12 – OOP Classification

**Topics:** `DataHandler` and `ModelHandler` classes, GridSearchCV hyperparameter tuning,
pickle model serialization and loading

```bash
cd session12_oop_classification

# Step 1: train model using OOP design pattern
python mainprogamoop.py
# Outputs: accuracy before/after tuning, classification report, saves trained_model.pkl

# Step 2: load pkl and run inference
python usepickle.py
```

> **Note:** `dermatology_database_1.csv` is auto-generated as synthetic data if not present.

---

## MLflow UI

All sessions write to a single shared `mlflow.db` at the `model_deployment_lab/` level.
To open the dashboard with **all experiments visible**, run from the `model_deployment_lab/` folder:

```bash
cd model_deployment_lab
mlflow ui --backend-store-uri sqlite:///mlflow.db
# Open http://localhost:5000 in your browser
```

Each session creates its own named experiment visible in the UI:

| Session | Experiment Name |
|---------|----------------|
| Session 02 | `iris-classification-demo` |
| Session 04 | `Iris-Pipeline` / `Streamlit-Pipeline` |
| Session 05 (A) | `Churn-NoPipeline` |
| Session 05 (B) | `Customer Churn Prediction` |

---

## Key Notes

| Situation | What happens |
|-----------|-------------|
| CSV data file missing | Synthetic data is auto-generated and saved on first run |
| Model pkl missing (FastAPI / Streamlit) | A default model is trained and saved on first launch |
| FastAPI not running (Streamlit) | A clear error message is shown with instructions |
| MLflow backend | Single shared `mlflow.db` in `model_deployment_lab/` — run `mlflow ui` from there |
