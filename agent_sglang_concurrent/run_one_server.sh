#!/usr/bin/env bash
set -euo pipefail

echo "This workload expects an already-running SGlang server."
echo
echo "Current default endpoint:"
echo "  http://localhost:30000/v1"
echo
echo "Example workload run:"
echo "  python run_swebench.py --server-base-url http://localhost:30000"
