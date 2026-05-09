#!/bin/bash
# user-data.sh
# ------------
# Runs once when the EC2 instance first boots. Installs Python, pulls the
# Streamlit app from a GitHub repo, and starts it as a systemd service so
# it survives reboots.
#
# How to use:
#   1. Replace the GIT_REPO line below with your own public repo URL.
#   2. In the EC2 launch wizard, paste this entire file into "User data".
#   3. Make sure the IAM instance profile is set to LabInstanceProfile.
#   4. Make sure the security group allows inbound TCP 8501 from 0.0.0.0/0.
#
# After ~2 minutes the app is reachable at:
#   http://<EC2-PUBLIC-IP>:8501
#
# Logs (for debugging):
#   sudo journalctl -u streamlit -f

set -eux

# -------- EDIT THESE TWO LINES -------------------------------------------
GIT_REPO="https://github.com/<your-handle>/sagemaker-wine-demo.git"
ENDPOINT_NAME="wine-xgb-endpoint"
# -------------------------------------------------------------------------

REGION="us-east-1"
APP_DIR="/opt/wine-app"

# 1. System packages
dnf update -y
dnf install -y python3 python3-pip git

# 2. Pull the app
git clone "$GIT_REPO" "$APP_DIR"
chown -R ec2-user:ec2-user "$APP_DIR"

# 3. Python deps (system-wide is fine on a throwaway EC2 instance)
pip3 install --upgrade pip
pip3 install streamlit boto3

# 4. systemd unit so the app auto-restarts and survives reboots
cat >/etc/systemd/system/streamlit.service <<EOF
[Unit]
Description=Streamlit Wine Quality App
After=network.target

[Service]
Type=simple
User=ec2-user
WorkingDirectory=$APP_DIR
Environment=ENDPOINT_NAME=$ENDPOINT_NAME
Environment=AWS_REGION=$REGION
ExecStart=/usr/bin/streamlit run streamlit_app.py \\
  --server.address 0.0.0.0 \\
  --server.port 8501 \\
  --server.headless true
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now streamlit.service
