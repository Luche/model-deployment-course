from config.config import ACCURACY_THRESHOLD
from src.data.loader import ingest_data, load_frame, split_features_target, split_train_test
from src.pipelines.sklearn_pipeline import build_churn_pipeline
from src.models.train import train_pipeline
from src.models.evaluate import evaluate


def main():
    print("=" * 50)
    print("Approach B – sklearn Pipeline")
    print("=" * 50)

    print("\nStep 1: Data Ingestion")
    ingest_data()

    print("\nStep 2: Load and Split")
    df = load_frame(rename_for_pipeline=True)
    X, y = split_features_target(df)
    x_train, x_test, y_train, y_test = split_train_test(X, y)

    print("\nStep 3: Build and Train Pipeline")
    num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    pipeline = build_churn_pipeline(num_features, cat_features)
    run_id = train_pipeline(pipeline, x_train, y_train)

    print("\nStep 4: Evaluation")
    accuracy, precision, recall = evaluate(x_test, y_test, run_id)

    print("\n" + "=" * 50)
    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Model APPROVED (accuracy={accuracy:.3f})")
    else:
        print(f"Model REJECTED (accuracy={accuracy:.3f} < {ACCURACY_THRESHOLD})")


if __name__ == "__main__":
    main()
