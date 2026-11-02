#!/usr/bin/env bash
# Full pipeline: C0 -> calibration -> constrained conditions -> evaluation -> analysis.
# Seeds of one condition run in parallel on one GPU (total throughput is about the same as sequential).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results

train_cond() {
    local cond=$1
    for s in 1 2 3; do
        uv run python train.py --cond "$cond" --seed "$s" > "logs/train_${cond}_s${s}.out" 2>&1 &
    done
    wait
}

stage=${1:-all}
if [[ $stage == all || $stage == c0 ]]; then
    train_cond C0
fi
if [[ $stage == all || $stage == calib ]]; then
    uv run python calibrate.py
fi
if [[ $stage == all || $stage == train ]]; then
    for cond in C2 C3 CTRL C1 C4; do  # minimal set first
        train_cond "$cond"
    done
fi
if [[ $stage == all || $stage == eval ]]; then
    for mode in primary secondary; do
        for env in E0 E1_friction E2_payload E3_torque E4_push; do
            uv run python evaluate.py --env "$env" --mode "$mode" > "logs/eval_${env}_${mode}.out" 2>&1 &
        done
        wait
    done
fi
if [[ $stage == all || $stage == analyze ]]; then
    uv run python analyze.py
fi
