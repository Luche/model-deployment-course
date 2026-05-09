"""
deploy_endpoint.py
------------------
Deploys the trained XGBoost Wine Quality model to a SageMaker real-time
endpoint. Run this from a SageMaker Notebook Instance in the Learner Lab.

LEARNER LAB CONSTRAINTS (read before running):
- Region: us-east-1. Pick one and stay there.
- Execution role: MUST be 'LabRole'. You cannot create a custom role.
- Endpoint instance type: ml.m5.large (within the medium/large/xlarge cap).
- Endpoint creation takes ~5-8 minutes of provisioning. Be patient.
- DELETE the endpoint when done. Endpoints bill per second of uptime.
"""

import boto3
import sagemaker
from sagemaker.xgboost.model import XGBoostModel


# ---- EDIT THESE THREE LINES ----------------------------------------------
BUCKET = "wine-demo-henry-1234"   # e.g. "sagemaker-studentname-1234"
MODEL_S3_KEY = "wine/model.tar.gz"     # where you uploaded the tarball
ENDPOINT_NAME = "wine-xgb-endpoint"    # must be unique within the account
# --------------------------------------------------------------------------

REGION = "us-east-1"
INSTANCE_TYPE = "ml.m5.large"
FRAMEWORK_VERSION = "1.7-1"


def get_lab_role_arn() -> str:
    """LabRole is pre-created in every Learner Lab account."""
    iam = boto3.client("iam")
    return iam.get_role(RoleName="LabRole")["Role"]["Arn"]


def main() -> None:
    boto3.setup_default_session(region_name=REGION)
    sm_session = sagemaker.Session()
    role_arn = get_lab_role_arn()
    model_s3_uri = f"s3://{BUCKET}/{MODEL_S3_KEY}"

    print(f"Role:       {role_arn}")
    print(f"Model URI:  {model_s3_uri}")
    print(f"Endpoint:   {ENDPOINT_NAME}")

    model = XGBoostModel(
        model_data=model_s3_uri,
        role=role_arn,
        entry_point="inference.py",
        source_dir="src",            # contains inference.py + requirements.txt
        framework_version=FRAMEWORK_VERSION,
        sagemaker_session=sm_session,
    )

    print("\nDeploying endpoint. This takes ~5-8 minutes.")
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    # Smoke test: a real wine from the dataset (first row, quality 5 -> low)
    sample = {
        "instances": [
            [7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]
        ]
    }

    runtime = boto3.client("sagemaker-runtime", region_name=REGION)
    response = runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType="application/json",
        Accept="application/json",
        Body=str(sample).replace("'", '"'),
    )
    print("\nSmoke test response:")
    print(response["Body"].read().decode("utf-8"))

    print(
        f"\nEndpoint '{ENDPOINT_NAME}' is live in {REGION}.\n"
        "Remember to delete it before lab teardown:\n"
        f"  predictor.delete_endpoint()\n"
        f"  or via the SageMaker console -> Endpoints -> Delete\n"
    )


if __name__ == "__main__":
    main()
