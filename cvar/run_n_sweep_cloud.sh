#!/usr/bin/env bash
# Run the synthetic n-sweep fully parallelised on a cloud CPU box
# (e.g. EC2 c7i.16xlarge with 64 vCPUs, GCP n2-highcpu-64, or larger).
# GPUs add no value — the bottleneck is sklearn isotonic regression.
#
# Pre-reqs on the box:
#   - Python 3.11 with: numpy, polars, scikit-learn, scipy
#   - The Arena dataset at $DATA_ROOT (default below; override via env)
#   - This repo cloned/synced
#
# Usage (from your laptop):
#   rsync -av ~/Dropbox/cvar-cje-data/ <user>@<box>:~/Dropbox/cvar-cje-data/
#   rsync -av $(git rev-parse --show-toplevel)/ <user>@<box>:~/cvar-cje/
#   ssh <user>@<box> 'cd ~/cvar-cje && bash cvar/run_n_sweep_cloud.sh'
#
# Sync result back:
#   rsync -av <user>@<box>:~/cvar-cje/cvar/results_n_sweep.csv ./cvar/
#
# Compute target: ~10 min on 64 vCPU at well under $1 of compute (c7i
# pricing circa 2026-04). Local 8-core Mac equivalent: ~70 min.
# Sweep: 2 policies × 4 n_evals × 3 alphas × SEEDS seeds (default 20)
# = 480 tasks at SEEDS=20, weighted by the n=500,000 cells.

set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/Dropbox/cvar-cje-data/cje-arena-experiments/data}"
N_WORKERS="${N_WORKERS:-$(python3.11 -c 'import multiprocessing; print(multiprocessing.cpu_count())')}"
SEEDS="${SEEDS:-20}"
B="${B:-200}"
OUT="${OUT:-cvar/results_n_sweep.csv}"

echo "=== run_n_sweep_cloud.sh ==="
echo "Python:    $(python3.11 --version)"
echo "Workers:   $N_WORKERS"
echo "Seeds:     $SEEDS"
echo "Bootstrap: $B"
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

# Smoke probe: 2 seeds, fast to verify env before the real run
echo "--- smoke probe (~2 min) ---"
time python3.11 -u cvar/n_sweep_synthetic.py \
    --n-workers "$N_WORKERS" \
    --seeds 2 \
    --b 100 \
    --data "$DATA_ROOT" \
    --out /tmp/_n_sweep_probe.csv
rm -f /tmp/_n_sweep_probe.csv
echo

# The full run
echo "--- FULL n-sweep ---"
time python3.11 -u cvar/n_sweep_synthetic.py \
    --n-workers "$N_WORKERS" \
    --seeds "$SEEDS" \
    --b "$B" \
    --data "$DATA_ROOT" \
    --out "$OUT"
echo

echo "=== done ==="
echo "Result CSV: $OUT  ($(wc -l < "$OUT") rows)"
echo
echo "Sync back to your laptop, then re-run the summary block at the bottom"
echo "of cvar/n_sweep_synthetic.py if you want the table reprinted, or feed"
echo "the CSV into the appendix table-fill step."
