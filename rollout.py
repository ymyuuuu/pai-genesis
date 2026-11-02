"""Shared helpers: build an env from a training run's config, load its policy, and roll out one episode per env."""
import glob
import json
import os
import re

import torch
from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from energy_env import Go2EnergyEnv, JOINT_TYPES

# feet are fixed to the calves in the URDF and get merged into the calf links by Genesis
FOOT_LINKS = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")

# evaluation environments, see docs/research_decisions.md (E5 terrain is optional and not implemented)
EVAL_ENVS = {
    "E0": {},
    "E1_friction": {"friction": 0.3},
    "E2_payload": {"added_mass": 3.0},
    "E3_torque": {"torque_scale": 0.7},
    "E4_push": {"push_interval_s": 3.0, "push_vel": 0.6},
}
# The simulation is deterministic (fixed command and initial state), so a single strength gives one all-or-nothing
# outcome per seed. Each unseen environment is therefore evaluated over a grid of strengths, chosen from a pilot
# with the baseline C0 only so that the grid spans "no effect" to "fails" for C0.
EVAL_SWEEPS = {
    "E0": [{}],
    "E1_friction": [{"friction": f} for f in (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)],
    "E2_payload": [{"added_mass": m} for m in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)],
    "E3_torque": [{"torque_scale": s} for s in (0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3)],
    "E4_push": [{"push_interval_s": 3.0, "push_vel": v} for v in (0.3, 0.6, 0.9, 1.2, 1.5)],
    # pre-registered follow-up environments for H4 (not used in the main experiment)
    "E5_gravity": [{"gravity_scale": g} for g in (1.1, 1.2, 1.3, 1.4, 1.5)],
    "E6_slope": [{"slope_deg": d} for d in (3.0, 6.0, 9.0, 12.0, 15.0)],
}
# only the push environment is stochastic and needs many parallel episodes
EVAL_NUM_ENVS = {"E0": 1, "E1_friction": 1, "E2_payload": 1, "E3_torque": 1, "E4_push": 200, "E5_gravity": 1,
                 "E6_slope": 1}


def load_cfgs(log_dir):
    with open(os.path.join(log_dir, "cfgs.json")) as f:
        return json.load(f)


def latest_checkpoint(log_dir):
    ckpts = glob.glob(os.path.join(log_dir, "model_*.pt"))
    return max(ckpts, key=lambda p: int(re.search(r"model_(\d+)\.pt", p).group(1)))


def make_env(cfgs, num_envs, overrides=None, no_budget=False, camera=False, n_rendered_envs=1):
    env_cfg = dict(cfgs["env"])
    env_cfg.update(overrides or {})
    env_cfg["n_rendered_envs"] = n_rendered_envs
    if no_budget:
        # secondary evaluation: budget removed, remaining-energy observation fixed at 1.0
        env_cfg["budget_mode"] = "none"
    reward_cfg = dict(cfgs["reward"])
    return Go2EnergyEnv(num_envs, env_cfg, cfgs["obs"], reward_cfg, cfgs["command"], camera=camera)


def load_policy(env, cfgs, log_dir):
    runner = OnPolicyRunner(env, cfgs["train"], None, device=gs.device)
    runner.load(latest_checkpoint(log_dir))
    return runner.get_inference_policy(device=gs.device)


@torch.no_grad()
def evaluate_episodes(env, policy):
    """Run exactly one episode in every env and return per-episode metrics (tensors of shape [num_envs])."""
    n, dt = env.num_envs, env.dt
    alive = torch.ones(n, dtype=torch.bool, device=gs.device)
    m = {
        "tracking": torch.zeros(n, device=gs.device),  # integral of the tracking term [s], max = episode length
        "survival_s": torch.zeros(n, device=gs.device),
        "distance": torch.zeros(n, device=gs.device),
        "consumed": torch.zeros(n, device=gs.device),
        "fallen": torch.zeros(n, dtype=torch.bool, device=gs.device),
        "depleted": torch.zeros(n, dtype=torch.bool, device=gs.device),
        "mech_J": torch.zeros(n, device=gs.device),
        "heat_raw": torch.zeros(n, device=gs.device),
        "action_usage": torch.zeros(n, device=gs.device),
    }
    for k in JOINT_TYPES:
        m[f"tau2_{k}"] = torch.zeros(n, device=gs.device)
        m[f"mech_{k}"] = torch.zeros(n, device=gs.device)
    m["tau_peak_ratio"] = torch.zeros(n, device=gs.device)
    m["contact_steps"] = torch.zeros(n, device=gs.device)  # summed over the 4 feet

    foot_idx = [env.robot.get_link(name).idx_local for name in FOOT_LINKS]
    obs = env.reset()
    for _ in range(env.max_episode_length + 2):
        obs, _, _, _ = env.step(policy(obs))
        s = env.step_snapshot
        a = alive.float()
        m["tracking"] += s["tracking"] * dt * a
        m["survival_s"] += dt * a
        m["distance"] += s["vx_world"] * dt * a
        m["mech_J"] += env.mech_power.sum(1) * dt * a
        m["heat_raw"] += env.heat_raw.sum(1) * dt * a
        m["action_usage"] += env.action_usage * dt * a
        tau2 = torch.square(env.tau)
        for i, k in enumerate(JOINT_TYPES):
            sel = env.joint_type_idx == i
            m[f"tau2_{k}"] += tau2[:, sel].sum(1) * dt * a
            m[f"mech_{k}"] += env.mech_power[:, sel].sum(1) * dt * a
        peak = (torch.abs(env.tau) / env.torque_limit).max(1).values
        m["tau_peak_ratio"] = torch.where(alive, torch.maximum(m["tau_peak_ratio"], peak), m["tau_peak_ratio"])
        contact = env.robot.get_links_net_contact_force()[:, foot_idx, 2] > 1.0
        m["contact_steps"] += contact.float().sum(1) * a

        done = s["fallen"] | s["depleted"] | s["time_out"]
        newly = alive & done
        m["consumed"] = torch.where(newly, s["consumed"], m["consumed"])
        m["fallen"] |= newly & s["fallen"]
        m["depleted"] |= newly & s["depleted"] & ~s["fallen"]
        alive &= ~done
        if not alive.any():
            break
    steps = m["survival_s"] / dt
    m["duty_factor"] = m["contact_steps"] / (4 * steps.clamp(min=1))
    del m["contact_steps"]
    return m
