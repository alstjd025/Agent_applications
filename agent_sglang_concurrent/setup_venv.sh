#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "[setup] Working directory: ${SCRIPT_DIR}"
echo "[setup] Python: ${PYTHON_BIN}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[setup] Error: ${PYTHON_BIN} not found" >&2
  exit 1
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "[setup] Creating virtual environment at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[setup] Reusing existing virtual environment at ${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"

echo "[setup] Upgrading pip"
python -m pip install --upgrade pip

echo "[setup] Installing dependencies from requirements.txt"
python -m pip install -r "${SCRIPT_DIR}/requirements.txt"

echo
echo "[setup] Done."
echo "[setup] Activate with:"
echo "source \"${VENV_DIR}/bin/activate\""
