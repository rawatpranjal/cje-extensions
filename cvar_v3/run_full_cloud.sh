#!/usr/bin/env bash
# Run the --full Monte Carlo on a cloud CPU box (e.g. EC2 c7i.16xlarge,
# GCP n2-highcpu-64). GPUs add no value — see make_power_report.py notes.
#
# Pre-reqs on the box:
#   - Python 3.11 with: numpy, polars, scikit-learn, scipy
#   - The Arena dataset at $DATA_ROOT (default below; override via env)
#   - This repo cloned/synced
#
# Usage (locally):
#   rsync -av ~/Dropbox/cvar-cje-data/ <user>@<box>:~/Dropbox/cvar-cje-data/
#   rsync -av $(git rev-parse --show-toplevel)/ <user>@<box>:~/cvar-cje/
#   ssh <user>@<box> 'cd ~/cvar-cje && bash cvar_v3/run_full_cloud.sh'
#
# Then sync the result back:
#   rsync -av <user>@<box>:~/cvar-cje/cvar_v3/results_mc_full.csv ./cvar_v3/
#   python3.11 cvar_v3/make_power_report.py
#
# Compute target: ~30 min on 64 vCPU at ~$1.40 of compute (c7i pricing
# circa 2026-04). Local Mac equivalent: ~6 hours.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/Dropbox/cvar-cje-data/cje-arena-experiments/data}"
N_WORKERS="${N_WORKERS:-$(python3.11 -c 'import multiprocessing; print(multiprocessing.cpu_count())')}"
OUT="${OUT:-cvar_v3/results_mc_full.csv}"

echo "=== run_full_cloud.sh ==="
echo "Python:    $(python3.11 --version)"
echo "Workers:   $N_WORKERS"
echo "Data root: $DATA_ROOT"
echo "Output:    $OUT"
echo

# Sanity checks
if [ ! -d "$DATA_ROOT" ]; then
    echo "ERROR: DATA_ROOT not found at $DATA_ROOT" >&2
    echo "Sync the Arena dataset first." >&2
    exit 1
fi
if [ ! -f "$DATA_ROOT/cje_dataset.jsonl" ]; then
    echo "ERROR: cje_dataset.jsonl not at $DATA_ROOT/cje_dataset.jsonl" >&2
    exit 1
fi
python3.11 -c "import polars, numpy, sklearn, scipy" || {
    echo "ERROR: missing python deps. Try: pip install polars numpy scikit-learn scipy" >&2
    exit 1
}

# Smoke probe before the full run (fail fast on env issues)
echo "--- smoke probe (~2 min) ---"
time python3.11 -u cvar_v3/run_monte_carlo.py --n-workers "$N_WORKERS" --out /tmp/_probe.csv
rm -f /tmp/_probe.csv
echo

# The full run
echo "--- FULL MC ---"
time python3.11 -u cvar_v3/run_monte_carlo.py --full --n-workers "$N_WORKERS" --out "$OUT"
echo

echo "=== done ==="
echo "Result CSV: $OUT  ($(wc -l < "$OUT") rows)"
echo
echo "Sync back to your laptop, then run:  python3.11 cvar_v3/make_power_report.py"
