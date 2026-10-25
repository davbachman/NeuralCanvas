#!/usr/bin/env bash
set -euo pipefail
apt-get update
apt-get install -y python3 python3-venv
python3 -m venv /opt/mlp-grader
/opt/mlp-grader/bin/pip install --disable-pip-version-check numpy==1.26.4
