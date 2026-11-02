# Based on Genesis examples/locomotion/go2_train.py (Apache-2.0, Genesis-Embodied-AI/Genesis).
import argparse
import json
import os
import shutil

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from energy_env import Go2EnergyEnv

# budget fraction of the baseline's per-episode consumption (E_base or A_base), see docs/research_decisions.md
CONDITIONS = {
    "C0": ("none", None),
    "C1": ("energy", 1.0),
    "C2": ("energy", 0.7),
    "C3": ("energy", 0.5),
    "C4": ("energy", 0.3),
    "CTRL": ("action", 0.7),
}
CALIBRATION_PATH = "results/calibration.json"


def get_train_cfg(exp_name, gamma):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": gamma,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs():
    env_cfg = {
        "num_actions": 12,
        "default_joint_angles": {  # [rad]
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 0.8,
            "FR_thigh_joint": 0.8,
            "RL_thigh_joint": 1.0,
            "RR_thigh_joint": 1.0,
            "FL_calf_joint": -1.5,
            "FR_calf_joint": -1.5,
            "RL_calf_joint": -1.5,
            "RR_calf_joint": -1.5,
        },
        "joint_names": [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ],
        # PD
        "kp": 20.0,
        "kd": 0.5,
        # termination
        "termination_if_roll_greater_than": 10,  # degree
        "termination_if_pitch_greater_than": 10,
        # base pose
        "base_init_pos": [0.0, 0.0, 0.42],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 4.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
        # energy budget
        "budget_mode": "none",
        "budget": float("inf"),
        "alpha": 0.0,
        "p_basal": 0.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
        },
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.3,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.2,
            "lin_vel_z": -1.0,
            "base_height": -50.0,
            "action_rate": -0.005,
            "similar_to_default": -0.1,
        },
    }
    command_cfg = {
        "num_commands": 3,
        "lin_vel_x_range": [0.5, 0.5],
        "lin_vel_y_range": [0, 0],
        "ang_vel_range": [0, 0],
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg


def apply_condition(env_cfg, cond):
    """Set budget mode/size from the condition name and the baseline calibration."""
    mode, frac = CONDITIONS[cond]
    env_cfg["budget_mode"] = mode
    if os.path.exists(CALIBRATION_PATH):
        with open(CALIBRATION_PATH) as f:
            calib = json.load(f)
        env_cfg["alpha"] = calib["alpha"]
        env_cfg["p_basal"] = calib["p_basal"]
    elif mode != "none":
        raise FileNotFoundError(f"{CALIBRATION_PATH} is required for {cond}; train C0 and run calibrate.py first")
    if mode == "energy":
        env_cfg["budget"] = frac * calib["E_base"]
    elif mode == "action":
        env_cfg["budget"] = frac * calib["A_base"]
    return env_cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cond", type=str, required=True, choices=list(CONDITIONS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("-b", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--gamma", type=float, default=0.995)
    args = parser.parse_args()

    exp_name = f"{args.cond}_s{args.seed}"
    log_dir = f"logs/{exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    env_cfg = apply_condition(env_cfg, args.cond)
    train_cfg = get_train_cfg(exp_name, args.gamma)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/cfgs.json", "w") as f:
        json.dump(
            {"cond": args.cond, "seed": args.seed, "env": env_cfg, "obs": obs_cfg, "reward": reward_cfg,
             "command": command_cfg, "train": train_cfg},
            f,
            indent=2,
        )

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = Go2EnergyEnv(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
