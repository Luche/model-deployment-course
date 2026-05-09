# SageMaker Wine Quality Demo (AWS Learner Lab)

End-to-end demo: train an XGBoost classifier locally, deploy it to a
SageMaker real-time endpoint, and call it from a Streamlit UI hosted on
EC2 — with no hardcoded AWS credentials.

## Architecture

```
[Browser]
   |
   v
[EC2 t3.micro: Streamlit]   <-- LabInstanceProfile (LabRole) attached
   |
   |  boto3.invoke_endpoint()  (SigV4-signed, IAM-authenticated)
   v
[SageMaker ml.m5.large endpoint]   <-- LabRole as execution role
   |
   v
[S3: model.tar.gz]
```

The SageMaker endpoint is **never publicly reachable**. Every invocation
requires AWS Signature V4 auth. EC2-with-instance-profile is about
**credential hygiene** (no AWS keys in code), not about hiding the
endpoint from the internet.

## Repo layout

```
sagemaker-wine-demo/
├── train.py              # Run locally. Trains and tarballs the model.
├── deploy_endpoint.py    # Run from SageMaker notebook in Learner Lab.
├── streamlit_app.py      # Runs on EC2.
├── user-data.sh          # EC2 user-data script.
├── src/
│   ├── inference.py      # SageMaker container entry point. STUDY THIS.
│   └── requirements.txt  # Installed inside the SageMaker container.
└── model_artifact/       # Created by train.py. Contains model.tar.gz.
```

## Learner Lab constraints (read first)

- Region: `us-east-1` (or `us-west-2`). Pick one and stay there.
- Execution role: must be `LabRole`. You cannot create custom IAM roles.
- SageMaker endpoint instance: `ml.m5.large` (medium / large / xlarge cap).
- EC2 instance: `t3.micro` is plenty.
- Endpoints bill per second of uptime. **Delete before lab teardown.**

## Step 1 — Train and upload the model (instructor, before class)

```bash
pip install xgboost scikit-learn pandas
python train.py
# produces model_artifact/model.tar.gz
```

In the Learner Lab account, upload the tarball:

```bash
aws s3 cp model_artifact/model.tar.gz s3://<your-bucket>/wine/model.tar.gz
```

Push this whole repo to a public GitHub repo (the EC2 user-data clones it).

## Step 2 — Deploy the SageMaker endpoint (in class, ~30 min including wait)

1. In the Learner Lab AWS Console, open **SageMaker** -> Notebook
   instances -> Create notebook instance.
2. Instance type: `ml.t3.medium`. Execution role: `LabRole`.
3. Open Jupyter, upload `deploy_endpoint.py` and the `src/` folder.
4. Edit the three constants at the top of `deploy_endpoint.py`.
5. Run it. Endpoint creation takes 5-8 minutes — use this time to walk
   through `inference.py` with students.

## Step 3 — Launch EC2 with Streamlit (in class, ~25 min)

1. EC2 -> Launch instances.
2. Amazon Linux 2023, `t3.micro`.
3. **IAM instance profile: `LabInstanceProfile`** (this is the critical step).
4. Security group: inbound TCP 8501 from `0.0.0.0/0`.
5. Advanced details -> User data: paste the contents of `user-data.sh`,
   after editing the `GIT_REPO` line.
6. Launch. Wait ~2 minutes. Open `http://<public-ip>:8501`.

## Step 4 — Demo the credential point (in class, ~5 min)

This is the lesson. With the app running:

1. Show students the `streamlit_app.py` source. No credentials anywhere.
2. EC2 console -> Actions -> Security -> Modify IAM role -> remove role.
3. Reload the Streamlit page, click Predict.
4. App fails with `NoCredentialsError`.
5. Re-attach `LabInstanceProfile`. Wait ~30 seconds. Works again.

## Step 5 — Teardown (non-negotiable, last 10 min of class)

In this order:

1. Delete the SageMaker endpoint:
   - SageMaker console -> Endpoints -> select -> Delete.
   - Or in a notebook cell: `predictor.delete_endpoint()`.
2. Delete the endpoint configuration: SageMaker -> Endpoint configurations.
3. Delete the model: SageMaker -> Models.
4. Stop / delete the notebook instance.
5. Terminate the EC2 instance.
6. End the Lab session.

Have students screenshot the empty Endpoints list and submit it as proof.

## Sample requests

JSON (preferred):
```json
{"instances": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]}
```

CSV:
```
7.4,0.7,0.0,1.9,0.076,11.0,34.0,0.9978,3.51,0.56,9.4
```

Feature order: fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
alcohol.

Output: `{"probabilities": [[...]], "predictions": [int], "labels": ["low"|"medium"|"high"]}`.

## Common failure modes

| Symptom | Likely cause |
|---|---|
| `iam:PassRole` denied during deploy | You're trying to pass a role other than LabRole. |
| `ResourceLimitExceeded` on endpoint create | Wrong instance type. Use `ml.m5.large`. |
| `NoCredentialsError` in Streamlit | LabInstanceProfile not attached to EC2. |
| `Connection refused` on port 8501 | Security group missing inbound rule. |
| Endpoint stuck in `Creating` >15 min | Check CloudWatch logs for the endpoint; usually a bad inference.py. |
| `ModuleNotFoundError: xgboost` in container logs | `requirements.txt` missing from `src/`. |
