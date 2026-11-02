"""Evaluate every trained run in one evaluation environment, over its grid of perturbation strengths.

primary  : each policy under its own training budget rule
secondary: budget removed, remaining-energy observation fixed at 1.0
One scene is built per environment; strengths and budget settings are changed per rollout without rebuilding.
"""
import argparse
import glob
import json
import os

import genesis as gs

from rollout import EVAL_NUM_ENVS, EVAL_SWEEPS, evaluate_episodes, load_cfgs, load_policy, make_env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True, choices=list(EVAL_SWEEPS))
    parser.add_argument("--mode", required=True, choices=["primary", "secondary"])
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--runs", default="logs/*_s*")
    parser.add_argument("--override", default=None, help="JSON env overrides for pilot runs, e.g. '{\"added_mass\": 1}'")
    parser.add_argument("--tag", default="", help="suffix for a pilot output file (written to results/pilot)")
    parser.add_argument("--out-dir", default="results/eval")
    parser.add_argument("--suffix", default="", help="suffix for the output file in --out-dir")
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=0, performance_mode=True)
    run_dirs = sorted(d for d in glob.glob(args.runs) if os.path.isdir(d))
    levels = [json.loads(args.override)] if args.override else EVAL_SWEEPS[args.env]
    num_envs = args.num_envs or EVAL_NUM_ENVS[args.env]
    env = make_env(load_cfgs(run_dirs[0]), num_envs)

    # energy is always measured with the calibrated model so that consumption is comparable across runs
    with open("results/calibration.json") as f:
        calib = json.load(f)
    env.alpha, env.p_basal = calib["alpha"], calib["p_basal"]

    policies = {d: (load_cfgs(d), None) for d in run_dirs}
    rows = []
    for level in levels:
        env.set_perturbation(**level)
        for d in run_dirs:
            cfgs, policy = policies[d]
            if policy is None:
                policy = load_policy(env, cfgs, d)
                policies[d] = (cfgs, policy)
            ecfg = cfgs["env"]
            env.budget_mode = "none" if args.mode == "secondary" else ecfg["budget_mode"]
            env.budget = float(ecfg["budget"])
            m = evaluate_episodes(env, policy)
            row = {"run": os.path.basename(d), "cond": cfgs["cond"], "seed": cfgs["seed"], "eval_env": args.env,
                   "mode": args.mode, "level": level}
            for k, v in m.items():
                row[k] = float(v.float().mean())
            row["energy_J"] = row["mech_J"] + env.alpha * row["heat_raw"] + env.p_basal * row["survival_s"]
            row["cot_J_per_m"] = row["energy_J"] / max(row["distance"], 1e-6)  # divide by m*g for dimensionless CoT
            rows.append(row)
            print(row["run"], args.env, args.mode, level, f"tracking={row['tracking']:.2f}",
                  f"fall={row['fallen']:.3f}", f"depleted={row['depleted']:.3f}", flush=True)

    out_dir = "results/pilot" if args.tag else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    with open(f"{out_dir}/{args.env}_{args.mode}{args.tag or args.suffix}.json", "w") as f:
        json.dump(rows, f, indent=2)


if __name__ == "__main__":
    main()
