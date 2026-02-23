"""
Session 05 – Pipeline Runner (sklearn Pipeline approach)
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from data_ingestion import ingest_data
from train_churnPipeline import train_model
from evaluation_Pipeline import evaluate

ACCURACY_THRESHOLD = 0.9


def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    df = pd.read_csv("ingested/customer_churn.csv", sep=";")
    df = df.rename(columns={
        "Usage Frequency":   "UsageFrequency",
        "Support Calls":     "SupportCalls",
        "Payment Delay":     "PaymentDelay",
        "Subscription Type": "SubscriptionType",
        "Contract Length":   "ContractLength",
        "Total Spend":       "TotalSpend",
        "Last Interaction":  "LastInteraction",
    })

    X = df.drop(['Churn', 'CustomerID'], axis=1)
    y = df["Churn"]
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nStep 2: Training (with sklearn Pipeline)")
    run_id = train_model(x_train, y_train)

    print("\nStep 3: Evaluation")
    accuracy, precision, recall = evaluate(x_test, y_test, run_id)

    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < {ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    run_pipeline()
