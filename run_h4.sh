#!/usr/bin/env bash
# Pre-registered follow-up for H4 (docs/research_decisions.md): new seeds 4-8 for C0/C1/C3,
# new environments E5 (gravity) and E6 (uphill slope), replication of E1-E3 on the new seeds.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results/eval_h4

stage=${1:-all}
if [[ $stage == all || $stage == train ]]; then
    for cond in C0 C1 C3; do
        for s in 4 5 6 7 8; do
            uv run python train.py --cond "$cond" --seed "$s" > "logs/train_${cond}_s${s}.out" 2>&1 &
        done
        wait
    done
fi
if [[ $stage == all || $stage == eval_old ]]; then
    # new environments on the original runs (seeds 1-3)
    for env in E5_gravity E6_slope; do
        uv run python evaluate.py --env "$env" --mode secondary --runs "logs/*_s[123]" --out-dir results/eval_h4 \
            --suffix _s123 > "logs/eval_${env}_secondary_s123.out" 2>&1 &
    done
    wait
fi
if [[ $stage == all || $stage == eval_new ]]; then
    # new seeds: training env (both modes, for margins and exclusion), replication envs, new envs
    # optional: a subset of runs and an output suffix, so evaluation can start while later conditions still train
    new=${2:-"logs/C[013]_s[45678]"}
    sfx=${3:-_s45678}
    for mode in primary secondary; do
        uv run python evaluate.py --env E0 --mode "$mode" --runs "$new" --out-dir results/eval_h4 --suffix "$sfx" \
            > "logs/eval_E0_${mode}${sfx}.out" 2>&1 &
    done
    for env in E1_friction E2_payload E3_torque; do
        uv run python evaluate.py --env "$env" --mode primary --runs "$new" --out-dir results/eval_h4 --suffix "$sfx" \
            > "logs/eval_${env}_primary${sfx}.out" 2>&1 &
    done
    wait
    for env in E5_gravity E6_slope; do
        uv run python evaluate.py --env "$env" --mode secondary --runs "$new" --out-dir results/eval_h4 \
            --suffix "$sfx" > "logs/eval_${env}_secondary${sfx}.out" 2>&1 &
    done
    wait
fi
if [[ $stage == all || $stage == analyze ]]; then
    uv run python analyze_h4.py
fi
