#!/usr/bin/env bash
# Run the FULL Monte Carlo on a cloud CPU box. ~5–6 hr on a 16-core box.
# sklearn isotonic + numpy bootstrap; no matmul, no GPU benefit.
#
# Recommended host: AWS c7i.4xlarge (16 vCPU) or GCP n2-standard-16.
#
# Usage (on the box, after `git clone` and `pip install -r requirements.txt`):
#   bash cvar_v4/eda/deeper/mc_validation/run_full_cloud.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$HERE/../../../.." && pwd)"
cd "$REPO_ROOT"

WORKERS="${WORKERS:-16}"
echo "Running --full with $WORKERS workers from $REPO_ROOT"

python3 -m cvar_v4.eda.deeper.mc_validation.runner --full \
  --n-workers "$WORKERS" \
  --out cvar_v4/eda/deeper/mc_validation/results_mc.csv

python3 -m cvar_v4.eda.deeper.mc_validation.report
