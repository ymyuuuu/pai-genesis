"""Calibrate the energy model from the unconstrained baseline (C0), see docs/research_decisions.md.

alpha  : Joule heat equals mechanical work over the baseline gait (1:1)
p_basal: 20% of the baseline's mean total power
E_base : baseline total energy over one full episode; A_base: same for the action-usage control quantity
"""
import argparse
import glob
import json
import os

import genesis as gs

from rollout import evaluate_episodes, load_cfgs, load_policy, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=1000)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0, performance_mode=True)
    log_dirs = sorted(glob.glob("logs/C0_s*"))
    log_dirs = [d for d in log_dirs if os.path.isdir(d)]
    cfgs = load_cfgs(log_dirs[0])
    env = make_env(cfgs, args.num_envs, no_budget=True)

    mech_rate, heat_rate, action_rate, survival, per_seed = 0.0, 0.0, 0.0, 0.0, {}
    for d in log_dirs:
        policy = load_policy(env, load_cfgs(d), d)
        m = evaluate_episodes(env, policy)
        t = m["survival_s"].sum()
        seed_stats = {
            "mech_W": float(m["mech_J"].sum() / t),
            "heat_raw_per_s": float(m["heat_raw"].sum() / t),
            "action_per_s": float(m["action_usage"].sum() / t),
            "mean_survival_s": float(m["survival_s"].mean()),
            "fall_rate": float(m["fallen"].float().mean()),
        }
        per_seed[d] = seed_stats
        print(d, seed_stats)
        mech_rate += seed_stats["mech_W"] / len(log_dirs)
        heat_rate += seed_stats["heat_raw_per_s"] / len(log_dirs)
        action_rate += seed_stats["action_per_s"] / len(log_dirs)

    episode_s = cfgs["env"]["episode_length_s"]
    alpha = mech_rate / heat_rate
    p_motor = 2.0 * mech_rate  # mechanical + equal heat
    p_basal = 0.25 * p_motor  # basal = 20% of total => basal = 0.25 * (mech + heat)
    calib = {
        "alpha": alpha,
        "p_basal": p_basal,
        "baseline_total_W": p_motor + p_basal,
        "E_base": (p_motor + p_basal) * episode_s,
        "A_base": action_rate * episode_s,
        "baseline_mech_W": mech_rate,
        "per_seed": per_seed,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/calibration.json", "w") as f:
        json.dump(calib, f, indent=2)
    print(json.dumps({k: v for k, v in calib.items() if k != "per_seed"}, indent=2))


if __name__ == "__main__":
    main()
