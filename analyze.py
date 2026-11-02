"""Aggregate results/eval/*.json into figures and a markdown summary (results/summary.md)."""
import glob
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

CONDS = ["C0", "C1", "C2", "C3", "C4", "CTRL"]
BUDGET_FRAC = {"C1": 1.0, "C2": 0.7, "C3": 0.5, "C4": 0.3}
UNSEEN = ["E1_friction", "E2_payload", "E3_torque", "E4_push"]
ENV_LABEL = {"E0": "E0 訓練環境", "E1_friction": "E1 低摩擦", "E2_payload": "E2 荷物",
             "E3_torque": "E3 トルク低下", "E4_push": "E4 外乱"}
OUT = "results"


def load_rows():
    rows = []
    for p in glob.glob(f"{OUT}/eval/*.json"):
        with open(p) as f:
            rows += json.load(f)
    return rows


def index(rows):
    """(mode, eval_env, cond) -> {seed: [rows over strength levels]}"""
    idx = defaultdict(lambda: defaultdict(list))
    for r in rows:
        idx[(r["mode"], r["eval_env"], r["cond"])][r["seed"]].append(r)
    return idx


def level_value(level):
    keys = [k for k in level if k != "push_interval_s"]
    return level[keys[0]] if keys else 0.0


# In the secondary evaluation some policies fail even in the training environment (they use the remaining-energy
# observation as a clock, and a constant 1.0 is out of distribution as a sequence). Retention is meaningless for
# them, so seeds whose E0 tracking is below this threshold are excluded there and reported.
MIN_E0_TRACKING_SECONDARY = 10.0


def valid_seed(idx, mode, cond, seed):
    base = idx[(mode, "E0", cond)].get(seed)
    if not base:
        return False
    return mode != "secondary" or base[0]["tracking"] >= MIN_E0_TRACKING_SECONDARY


def excluded_seeds(idx, mode, cond):
    return sorted(s for s in idx[(mode, "E0", cond)] if not valid_seed(idx, mode, cond, s))


def retention_by_seed(idx, mode, cond, env):
    """Mean over strength levels of tracking(level) / tracking(E0), per seed."""
    base = idx[(mode, "E0", cond)]
    test = idx[(mode, env, cond)]
    out = {}
    for s, rs in test.items():
        if valid_seed(idx, mode, cond, s) and base[s][0]["tracking"] > 0:
            out[s] = float(np.mean([r["tracking"] for r in rs]) / base[s][0]["tracking"])
    return out


def generalization_by_seed(idx, mode, cond):
    """Mean retention over the unseen environments, per seed."""
    per_env = [retention_by_seed(idx, mode, cond, e) for e in UNSEEN]
    seeds = set.intersection(*(set(d) for d in per_env)) if per_env else set()
    return {s: float(np.mean([d[s] for d in per_env])) for s in sorted(seeds)}


def metric_by_seed(idx, mode, env, cond, key):
    """Per seed, the metric averaged over the strength levels of the environment."""
    return {s: float(np.mean([r[key] for r in rs])) for s, rs in idx[(mode, env, cond)].items()}


def plot_robustness_curves(idx, conds):
    fig, axes = plt.subplots(2, len(UNSEEN), figsize=(15, 7), sharey=True)
    xlabel = {"E1_friction": "摩擦係数（公称1.0）", "E2_payload": "追加質量 [kg]", "E3_torque": "トルク上限の倍率",
              "E4_push": "外乱の速度 [m/s]"}
    for row, (mode, mtitle) in enumerate([("primary", "主評価"), ("secondary", "副評価")]):
        for col, env in enumerate(UNSEEN):
            ax = axes[row, col]
            for c in conds:
                by_level = defaultdict(list)
                for rs in idx[(mode, env, c)].values():
                    for r in rs:
                        by_level[level_value(r["level"])].append(r["tracking"])
                if not by_level:
                    continue
                xs = sorted(by_level)
                ys = [np.mean(by_level[x]) for x in xs]
                style = {"color": "black", "lw": 2.2, "ls": "--"} if c == "C0" else (
                    {"color": "tab:red", "lw": 1.8, "ls": ":"} if c == "CTRL" else {"lw": 1.5})
                ax.plot(xs, ys, marker="o", ms=3, label=c, **style)
            if env in ("E1_friction", "E3_torque"):
                ax.invert_xaxis()  # left = mild, right = severe
            ax.set_title(f"{mtitle}: {ENV_LABEL[env]}")
            ax.set_xlabel(xlabel[env])
            ax.grid(alpha=0.3)
        axes[row, 0].set_ylabel("追従報酬の累積 [s]（最大20）")
    axes[0, -1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_robustness_curves.png", dpi=160)
    plt.close(fig)


def ms(values):
    v = np.array(list(values), dtype=float)
    if len(v) == 0:
        return float("nan"), float("nan")
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0


def welch(a, b):
    a, b = list(a), list(b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(stats.ttest_ind(a, b, equal_var=False).pvalue)


def plot_retention_curve(idx, conds):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, mode, title in [(axes[0], "primary", "主評価（学習時と同じ予算）"), (axes[1], "secondary", "副評価（予算なし）")]:
        xs = [c for c in ["C1", "C2", "C3", "C4"] if c in conds]
        x = [BUDGET_FRAC[c] for c in xs]
        for env in UNSEEN:
            y = [ms(retention_by_seed(idx, mode, c, env).values())[0] for c in xs]
            ax.plot(x, y, marker="o", alpha=0.35, lw=1, label=ENV_LABEL[env])
        g = [ms(generalization_by_seed(idx, mode, c).values()) for c in xs]
        ax.errorbar(x, [m for m, _ in g], yerr=[s for _, s in g], marker="o", color="black", lw=2.2, capsize=4,
                    label="未知環境の平均")
        if "C0" in conds:
            m0, s0 = ms(generalization_by_seed(idx, mode, "C0").values())
            ax.axhspan(m0 - s0, m0 + s0, color="tab:gray", alpha=0.2)
            ax.axhline(m0, color="tab:gray", ls="--", label="C0 制約なし")
        if "CTRL" in conds:
            mc, sc = ms(generalization_by_seed(idx, mode, "CTRL").values())
            ax.errorbar([0.7], [mc], yerr=[sc], marker="s", color="tab:red", ms=8, capsize=4, ls="none",
                        label="CTRL 非物理量70%")
        ax.set_xlabel("エネルギー予算（ベースライン消費に対する割合）")
        ax.set_ylabel("性能維持率（強度平均の未知環境 / 訓練環境）")
        ax.set_title(title)
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_retention.png", dpi=160)
    plt.close(fig)


def plot_absolute(idx, conds, mode, fname, title):
    envs = ["E0"] + UNSEEN
    fig, ax = plt.subplots(figsize=(11, 4))
    w = 0.8 / len(conds)
    for i, c in enumerate(conds):
        stats_ = [ms(metric_by_seed(idx, mode, e, c, "tracking").values()) for e in envs]
        ax.bar(np.arange(len(envs)) + i * w, [m for m, _ in stats_], w, yerr=[s for _, s in stats_], capsize=2,
               label=c)
    ax.set_xticks(np.arange(len(envs)) + w * (len(conds) - 1) / 2, [ENV_LABEL[e] for e in envs])
    ax.set_ylabel("追従報酬の累積 [s]（最大20）")
    ax.set_title(title)
    ax.legend(ncol=len(conds), fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{fname}", dpi=160)
    plt.close(fig)


def mechanism_rows(idx, conds):
    rows = []
    for c in conds:
        rs = {seed: v[0] for seed, v in idx[("primary", "E0", c)].items()}
        calf_share = [r["tau2_calf"] / (r["tau2_hip"] + r["tau2_thigh"] + r["tau2_calf"]) for r in rs.values()]
        rows.append({
            "cond": c,
            "calf_share": ms(calf_share),
            "duty": ms(r["duty_factor"] for r in rs.values()),
            "peak": ms(r["tau_peak_ratio"] for r in rs.values()),
            "power": ms(r["energy_J"] / max(r["survival_s"], 1e-6) for r in rs.values()),
            "cot": ms(r["cot_J_per_m"] for r in rs.values()),
            "speed": ms(r["distance"] / max(r["survival_s"], 1e-6) for r in rs.values()),
        })
    return rows


def plot_mechanism(mrows):
    keys = [("calf_share", "膝のトルク²分担"), ("duty", "接地時間の割合"), ("peak", "トルクのピーク/上限"),
            ("power", "平均消費電力 [W]")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (k, label) in zip(axes, keys):
        ax.bar([r["cond"] for r in mrows], [r[k][0] for r in mrows], yerr=[r[k][1] for r in mrows], capsize=3,
               color="tab:blue")
        ax.set_title(label)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("メカニズム（訓練環境・学習時と同じ予算で計測）")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_mechanism.png", dpi=160)
    plt.close(fig)


def fmt(m_s, nd=3):
    return f"{m_s[0]:.{nd}f} ± {m_s[1]:.{nd}f}"


def main():
    rows = load_rows()
    idx = index(rows)
    conds = [c for c in CONDS if any(k[2] == c for k in idx)]
    plot_retention_curve(idx, conds)
    plot_robustness_curves(idx, conds)
    plot_absolute(idx, conds, "primary", "fig_tracking_primary.png", "主評価: 追従報酬の累積（学習時と同じ予算）")
    plot_absolute(idx, conds, "secondary", "fig_tracking_secondary.png", "副評価: 追従報酬の累積（予算なし）")
    mrows = mechanism_rows(idx, conds)
    plot_mechanism(mrows)

    lines = ["# 結果の要約", "", "値はシード間の平均 ± 標準偏差。維持率は各未知環境の強度グリッドで平均した追従報酬 / 訓練環境の追従報酬。p値はWelchのt検定（n=シード数、参考値）。", ""]
    for mode, title in [("primary", "主評価（学習時と同じ予算）"), ("secondary", "副評価（予算なし・観測1.0固定）")]:
        lines += [f"## {title}", "", "| 条件 | 未知環境の平均維持率 | " + " | ".join(ENV_LABEL[e] for e in UNSEEN) +
                  " | E0の追従報酬 | 転倒率(E0) | エネルギー切れ率(E0) |", "|" + "---|" * (len(UNSEEN) + 5)]
        for c in conds:
            g = generalization_by_seed(idx, mode, c)
            cells = [fmt(ms(retention_by_seed(idx, mode, c, e).values())) for e in UNSEEN]
            lines.append(f"| {c} | {fmt(ms(g.values()))} | " + " | ".join(cells) +
                         f" | {fmt(ms(metric_by_seed(idx, mode, 'E0', c, 'tracking').values()), 2)}"
                         f" | {fmt(ms(metric_by_seed(idx, mode, 'E0', c, 'fallen').values()))}"
                         f" | {fmt(ms(metric_by_seed(idx, mode, 'E0', c, 'depleted').values()))} |")
        if mode == "secondary":
            excl = {c: excluded_seeds(idx, mode, c) for c in conds if excluded_seeds(idx, mode, c)}
            lines.append(f"- 訓練環境の追従報酬が{MIN_E0_TRACKING_SECONDARY:.0f}未満で維持率から除外したシード: "
                         + (", ".join(f"{c}: {v}" for c, v in excl.items()) if excl else "なし"))
        lines.append("")
        g0 = generalization_by_seed(idx, mode, "C0").values()
        for c in ["C1", "C2", "C3", "C4", "CTRL"]:
            if c in conds:
                lines.append(f"- {c} vs C0（維持率）: p = {welch(generalization_by_seed(idx, mode, c).values(), g0):.3f}")
        if "C2" in conds and "CTRL" in conds:
            lines.append(f"- C2 vs CTRL（維持率）: p = "
                         f"{welch(generalization_by_seed(idx, mode, 'C2').values(), generalization_by_seed(idx, mode, 'CTRL').values()):.3f}")
        lines.append("")
        lines += ["### 未知環境での追従報酬の累積（強度グリッドの平均、絶対値）", "", "| 条件 | " + " | ".join(ENV_LABEL[e] for e in UNSEEN) + " |",
                  "|" + "---|" * (len(UNSEEN) + 1)]
        for c in conds:
            lines.append(f"| {c} | " + " | ".join(fmt(ms(metric_by_seed(idx, mode, e, c, 'tracking').values()), 2)
                                              for e in UNSEEN) + " |")
        lines.append("")
    lines += ["## メカニズム（訓練環境・学習時と同じ予算）", "",
              "| 条件 | 膝のトルク²分担 | 接地時間の割合 | トルクのピーク/上限 | 平均消費電力 [W] | 移動コスト [J/m] | 平均速度 [m/s] |",
              "|---|---|---|---|---|---|---|"]
    for r in mrows:
        lines.append(f"| {r['cond']} | {fmt(r['calf_share'])} | {fmt(r['duty'])} | {fmt(r['peak'])} | "
                     f"{fmt(r['power'], 1)} | {fmt(r['cot'], 1)} | {fmt(r['speed'])} |")
    with open(f"{OUT}/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
