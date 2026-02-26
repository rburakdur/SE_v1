#!/usr/bin/env bash
set -euo pipefail

# Polls remote branch, deploys latest commit, reinstalls project, and restarts the service.
# Intended for systemd timer usage on Oracle Cloud / Ubuntu.

REPO_DIR="${REPO_DIR:-/opt/rbdcrypt}"
BRANCH="${BRANCH:-main}"
SERVICE_NAME="${SERVICE_NAME:-rbdcrypt.service}"
VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
USE_SUDO_FOR_RESTART="${USE_SUDO_FOR_RESTART:-true}"

cd "$REPO_DIR"

git fetch --quiet origin "$BRANCH"
LOCAL_SHA="$(git rev-parse HEAD)"
REMOTE_SHA="$(git rev-parse "origin/$BRANCH")"

if [[ "$LOCAL_SHA" == "$REMOTE_SHA" ]]; then
  exit 0
fi

git pull --ff-only origin "$BRANCH"

if [[ -x "$VENV_DIR/bin/python" ]]; then
  "$VENV_DIR/bin/python" -m pip install -e "$REPO_DIR"
else
  "$PYTHON_BIN" -m pip install -e "$REPO_DIR"
fi

if [[ "$USE_SUDO_FOR_RESTART" == "true" ]]; then
  sudo systemctl restart "$SERVICE_NAME"
else
  systemctl restart "$SERVICE_NAME"
fi
