"""
Session 05 – Central configuration.
All paths, hyperparameters, column names, and MLflow settings live here.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR     = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_ING_DIR = BASE_DIR / "data" / "ingested"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

# Artifact filenames
ARTIFACT_MODEL_MANUAL  = ARTIFACTS_DIR / "model_churnNoPipeline.pkl"
ARTIFACT_NUM_IMPUTER   = ARTIFACTS_DIR / "num_imputer.pkl"
ARTIFACT_CAT_IMPUTER   = ARTIFACTS_DIR / "cat_imputer.pkl"
ARTIFACT_CAT_ENCODER   = ARTIFACTS_DIR / "cat_encoder.pkl"
ARTIFACT_PIPELINE      = ARTIFACTS_DIR / "churn_prediction_pipeline.pkl"

# ---------------------------------------------------------------------------
# Column definitions
# ---------------------------------------------------------------------------
TARGET_COL = "Churn"
DROP_COLS  = ["CustomerID"]

NUM_FEATURES = ["Age", "Tenure", "Usage Frequency", "Support Calls",
                "Payment Delay", "Total Spend", "Last Interaction"]
CAT_FEATURES = ["Gender", "Subscription Type", "Contract Length"]

# Rename map applied when using sklearn Pipeline approach
PIPELINE_RENAME_MAP = {
    "Usage Frequency":   "UsageFrequency",
    "Support Calls":     "SupportCalls",
    "Payment Delay":     "PaymentDelay",
    "Subscription Type": "SubscriptionType",
    "Contract Length":   "ContractLength",
    "Total Spend":       "TotalSpend",
    "Last Interaction":  "LastInteraction",
}

# ---------------------------------------------------------------------------
# Encoding constants
# ---------------------------------------------------------------------------
# OrdinalEncoder category order for sklearn Pipeline (post-rename column names)
PIPELINE_CAT_ORDERS = [
    ["Male", "Female"],
    ["Basic", "Standard", "Premium"],
    ["Monthly", "Quarterly", "Annual"],
]

# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
RF_CRITERION  = "gini"
RF_MAX_DEPTH  = 4
RANDOM_STATE  = 42
TEST_SIZE     = 0.2

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------
MLFLOW_TRACKING_URI  = f"sqlite:///{BASE_DIR.parent / 'mlflow.db'}"
MLFLOW_EXP_MANUAL    = "Churn-NoPipeline"
MLFLOW_EXP_PIPELINE  = "Customer Churn Prediction"

# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------
ACCURACY_THRESHOLD = 0.9
