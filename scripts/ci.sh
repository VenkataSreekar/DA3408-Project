#!/usr/bin/env bash
# Local CI for Next Problem (no cloud services).
#   1. unit tests        2. reproduce the DVC pipeline (only changed stages run)
#   3. show metrics      4. acceptance gate: the model must beat the Elo baseline
# Usage: scripts/ci.sh        (exit code 0 = CI passed, non-zero = CI failed)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 1/4 Unit tests"
pytest -q

echo "==> 2/4 Reproduce the DVC pipeline (dvc repro)"
dvc repro

echo "==> 3/4 Pipeline graph and metrics"
dvc dag
dvc metrics show

echo "==> 4/4 Acceptance gate"
python - <<'PY'
import json
import sys

m = json.load(open("metrics/train_metrics.json"))
best, elo = m["best_model"]["test"], m["elo_baseline"]["test"]
print(f"test log loss: model {best['log_loss']:.5f} vs Elo {elo['log_loss']:.5f}")
print(f"test AUC:      model {best['auc']:.5f} vs Elo {elo['auc']:.5f}")
if not m["beats_baseline"]:
    print("ACCEPTANCE FAILED: model does not beat the Elo baseline")
    sys.exit(1)
print("ACCEPTANCE PASSED")
PY

echo "CI passed"
