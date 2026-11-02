#!/usr/bin/env bash
# Training only (no evaluation): fresh seeds 9-14 for a possible confirmatory test of the exploratory uphill finding.
# The confirmatory test was not conducted (docs/research_decisions.md): these runs were never evaluated and are
# not used in any result.
set -euo pipefail
cd "$(dirname "$0")"
for cond in C2 CTRL C0; do
    for s in 9 10 11 12 13 14; do
        uv run python train.py --cond "$cond" --seed "$s" > "logs/train_${cond}_s${s}.out" 2>&1 &
    done
    wait
done
