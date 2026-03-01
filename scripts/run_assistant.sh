#!/usr/bin/env bash
set -euo pipefail

# Create and/or activate a Python virtualenv, install requirements, and start the assistant.
# Usage: ./scripts/run_assistant.sh [--show-settings] [--other-flags]

VENV_DIR="${VENV_DIR:-.venv}"
REQ_FILE="requirements.txt"
PY_CMD="${PYTHON:-python3}"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtualenv in $VENV_DIR..."
  $PY_CMD -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
echo "Activated virtualenv: $VENV_DIR"

echo "Upgrading pip and installing requirements (if present)..."
pip install --upgrade pip
if [ -f "$REQ_FILE" ]; then
  pip install -r "$REQ_FILE"
fi

echo "Starting assistant..."
exec python "$PWD/assistant.py" "$@"
