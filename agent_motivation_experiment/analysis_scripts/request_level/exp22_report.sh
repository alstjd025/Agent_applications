#!/bin/bash
# Post-process an EXP-22 pair of runs and produce every artefact in one go.
#
#   ./exp22_report.sh <out-dir> <run-dir> [<run-dir> ...]
#
# Per run: attribute each request to the engine that served it (needed for the
# routing-concentration measure), then report what the controller did from the
# scraped scheduler series. Across runs: attainment against both denominators,
# goodput, the time course, and the hour re-expressed as a rate sweep.
set -uo pipefail
cd "$(dirname "$0")"
OUT=${1:?usage: exp22_report.sh <out-dir> <run-dir>...}
shift
RUNS=("$@")
[ ${#RUNS[@]} -gt 0 ] || { echo "no run dirs given"; exit 1; }

mkdir -p "$OUT"
for r in "${RUNS[@]}"; do
  echo "=== $(basename "$r") ==="
  python3 build_request_engine_map.py "$r" 2>&1 | tail -3
  python3 exp22_controller.py --run "$r" --out-dir "$OUT" 2>&1 | tail -40
done

echo
echo "=== across arms ==="
python3 exp22_fluidserve.py --runs "${RUNS[@]}" --out-dir "$OUT" | tee "$OUT/summary.txt"
echo
echo "artefacts in $OUT:"
ls -la "$OUT"
