"""Record one episode of a trained policy with an energy gauge overlay (one robot per clip).

Clips are composed side by side with compose_video.py.
"""
import argparse
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import genesis as gs

from rollout import EVAL_ENVS, load_cfgs, load_policy, make_env

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


def overlay(frame, title, subtitle, t, consumed, budget, e_base, fallen, depleted):
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img, "RGBA")
    big, small = ImageFont.truetype(FONT, 30), ImageFont.truetype(FONT, 20)
    d.rectangle([0, 0, img.width, 92], fill=(0, 0, 0, 150))
    d.text((16, 8), title, font=big, fill="white")
    d.text((16, 52), subtitle, font=small, fill=(220, 220, 220))
    d.text((img.width - 120, 16), f"{t:5.1f} s", font=big, fill="white")

    # gauge: consumed energy relative to the baseline's per-episode consumption (E_base)
    x0, y0, w, h = 16, img.height - 56, img.width - 32, 26
    d.rectangle([x0 - 4, y0 - 34, x0 + w + 4, y0 + h + 6], fill=(0, 0, 0, 150))
    frac = min(consumed / e_base, 1.0)
    if np.isfinite(budget):
        remain = max(1.0 - consumed / budget, 0.0)
        label = f"残りエネルギー {remain * 100:5.1f}%   （予算 = ベースラインの {budget / e_base * 100:.0f}%）"
        d.rectangle([x0, y0, x0 + w * budget / e_base, y0 + h], fill=(60, 60, 60, 255))
        color = (80, 200, 120) if remain > 0.3 else (240, 170, 40) if remain > 0.1 else (230, 60, 60)
    else:
        label = "エネルギー予算なし（消費量のみ表示）"
        d.rectangle([x0, y0, x0 + w, y0 + h], fill=(60, 60, 60, 255))
        color = (90, 150, 230)
    d.rectangle([x0, y0, x0 + w * frac, y0 + h], fill=color)
    d.text((x0, y0 - 30), f"{label}   消費 {consumed:6.0f} J", font=small, fill="white")
    if fallen or depleted:
        msg = "転倒（傾き10度超）" if fallen else "エネルギー切れで停止"
        d.text((img.width // 2 - 230, img.height // 2 - 30), msg, font=ImageFont.truetype(FONT, 48),
               fill=(255, 80, 80))
    return np.asarray(img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, help="e.g. logs/C2_s1")
    parser.add_argument("--env", default="E0", choices=list(EVAL_ENVS))
    parser.add_argument("--override", default=None, help="JSON perturbation, e.g. '{\"added_mass\": 1.0}'")
    parser.add_argument("--no-budget", action="store_true")
    parser.add_argument("--title", required=True)
    parser.add_argument("--subtitle", default="")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed)
    cfgs = load_cfgs(args.run)
    with open("results/calibration.json") as f:
        calib = json.load(f)
    overrides = json.loads(args.override) if args.override else EVAL_ENVS[args.env]
    env = make_env(cfgs, 1, overrides=overrides, no_budget=args.no_budget, camera=True)
    env.alpha, env.p_basal = calib["alpha"], calib["p_basal"]
    policy = load_policy(env, cfgs, args.run)
    budget = env.budget if env.budget_mode == "energy" else float("inf")

    frames, prev_rgb = [], None
    obs = env.reset()
    with torch.no_grad():
        for i in range(int(args.seconds / env.dt)):
            obs, _, _, _ = env.step(policy(obs))
            s = env.step_snapshot
            t, consumed = (i + 1) * env.dt, float(s["consumed"][0])
            fallen, depleted = bool(s["fallen"][0]), bool(s["depleted"][0])
            if fallen or depleted:
                # the env has already reset, so hold the last pre-reset frame with the end message for 1 s
                last = overlay(prev_rgb, args.title, args.subtitle, t, consumed, budget, calib["E_base"], fallen,
                               depleted)
                frames += [last] * 50
                break
            pos = env.robot.get_pos()[0].cpu().numpy()
            env.cam.set_pose(pos=(pos[0] + 1.2, pos[1] - 1.8, 0.75), lookat=(pos[0], pos[1], 0.25))
            prev_rgb = env.cam.render()[0]
            frames.append(overlay(prev_rgb, args.title, args.subtitle, t, consumed, budget, calib["E_base"], False,
                                  False))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    imageio.mimwrite(args.out, frames, fps=int(1 / env.dt), codec="libx264", quality=8)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
