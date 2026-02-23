"""
Session 05 – Pipeline Runner (No-Pipeline approach)
"""

from data_ingestion import ingest_data
from preprocessing_NoPipeline import load_data, missing_value, encoder
from train_churnNoPipeline import train
from evaluation_NoPipeline import evaluate

ACCURACY_THRESHOLD = 0.9


def run_pipeline():
    print("Step 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Preprocessing")
    x_train, x_test, y_train, y_test = load_data()
    x_train, x_test = missing_value(x_train, x_test)
    x_train_enc, x_test_enc = encoder(x_train, x_test)

    print("\nStep 3: Training")
    run_id = train(x_train_enc, y_train)

    print("\nStep 4: Evaluation")
    accuracy, precision, recall = evaluate(x_test_enc, y_test, run_id)

    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < {ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    run_pipeline()
