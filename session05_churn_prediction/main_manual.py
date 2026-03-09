from config.config import ACCURACY_THRESHOLD
from src.data.loader import ingest_data, load_frame, split_features_target, split_train_test
from src.pipelines.manual_pipeline import run_preprocessing
from src.models.train import train_manual
from src.models.evaluate import evaluate


def main():
    print("=" * 50)
    print("Approach A – Manual Preprocessing (NoPipeline)")
    print("=" * 50)

    print("\nStep 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Load and Split")
    df = load_frame(rename_for_pipeline=False)
    X, y = split_features_target(df)
    x_train, x_test, y_train, y_test = split_train_test(X, y)

    print("\nStep 3: Preprocessing")
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    x_train_enc, x_test_enc = run_preprocessing(x_train, x_test, num_features, cat_features)

    print("\nStep 4: Training")
    run_id = train_manual(x_train_enc, y_train)

    print("\nStep 5: Evaluation")
    accuracy, precision, recall = evaluate(x_test_enc, y_test, run_id)

    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < {ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    main()
