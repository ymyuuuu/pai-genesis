"""Pre-registered analysis of the H4 follow-up (see docs/research_decisions.md, 追加検証の事前登録).

P1/P2: Spearman correlation between torque margin and retention in E5 (gravity) / E6 (uphill slope), secondary mode,
       over all valid runs (seeds 1-8), plus the condition-level comparison C3,C4 vs C1,C2.
P3   : replication on the new seeds 4-8 only (primary mode), one-sided Welch tests with Holm correction.
"""
import glob
import json
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]

OUT = "results"
MIN_E0_TRACKING_SECONDARY = 10.0  # same exclusion rule as the main experiment
COLORS = {"C0": "black", "C1": "tab:blue", "C2": "tab:orange", "C3": "tab:green", "C4": "tab:red", "CTRL": "tab:purple"}
NEW_ENVS = {"E5_gravity": ("E5 重力増加", "重力加速度の倍率", "gravity_scale"),
            "E6_slope": ("E6 登り斜面", "斜面の角度 [度]", "slope_deg")}


def load(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            rows += json.load(f)
    return rows


def by_run(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["run"]].append(r)
    return out


def run_info():
    """run -> (cond, seed, margin, E0 secondary tracking, valid) from the secondary training-environment evaluation."""
    info = {}
    for r in load(f"{OUT}/eval/E0_secondary.json") + load(f"{OUT}/eval_h4/E0_secondary_s45678*.json"):
        info[r["run"]] = {"cond": r["cond"], "seed": r["seed"], "margin": 1.0 - r["tau_peak_ratio"],
                          "e0": r["tracking"], "valid": r["tracking"] >= MIN_E0_TRACKING_SECONDARY}
    return info


def new_env_retention(env, info):
    rows = load(f"{OUT}/eval_h4/{env}_secondary_*.json")
    ret, curves = {}, defaultdict(lambda: defaultdict(list))
    for run, rs in by_run(rows).items():
        if not info[run]["valid"]:
            continue
        ret[run] = float(np.mean([r["tracking"] for r in rs]) / info[run]["e0"])
        for r in rs:
            curves[info[run]["cond"]][r["level"][NEW_ENVS[env][2]]].append(r["tracking"])
    return ret, curves


def cond_mean(ret, info, cond):
    v = [x for run, x in ret.items() if info[run]["cond"] == cond]
    return float(np.mean(v)) if v else float("nan"), len(v)


def holm(pvals):
    """Holm-adjusted p-values, in the input order."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj, running = [0.0] * m, 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adj[i] = min(running, 1.0)
    return adj


def replication():
    """Primary-mode retention per new seed for E1-E3 (seeds 4-8 only)."""
    base = {r["run"]: r["tracking"] for r in load(f"{OUT}/eval_h4/E0_primary_s45678*.json")}
    out = defaultdict(lambda: defaultdict(dict))  # env -> cond -> seed -> retention
    for env in ("E1_friction", "E2_payload", "E3_torque"):
        for run, rs in by_run(load(f"{OUT}/eval_h4/{env}_primary_s45678*.json")).items():
            out[env][rs[0]["cond"]][rs[0]["seed"]] = float(np.mean([r["tracking"] for r in rs]) / base[run])
    return out


def old_seed_retention(env, cond):
    """Seeds 1-3 primary retention from the main experiment, for reference in the figure."""
    base = {r["run"]: r["tracking"] for r in load(f"{OUT}/eval/E0_primary.json")}
    vals = []
    for run, rs in by_run(load(f"{OUT}/eval/{env}_primary.json")).items():
        if rs[0]["cond"] == cond:
            vals.append(float(np.mean([r["tracking"] for r in rs]) / base[run]))
    return vals


def main():
    info = run_info()
    lines = ["# 追加検証（H4）の結果", "",
             "事前登録（docs/research_decisions.md）どおりの判定。H4 は本実験の結果を見てから立てた仮説で、"
             "本実験で使っていない環境（E5・E6）と新しいシード（4〜8）で検証した。", ""]

    # P1 / P2
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    verdicts = {}
    for col, (env, (label, xlabel, key)) in enumerate(NEW_ENVS.items()):
        ret, curves = new_env_retention(env, info)
        runs = sorted(ret)
        x = [info[r]["margin"] for r in runs]
        y = [ret[r] for r in runs]
        rho, p = stats.spearmanr(x, y, alternative="greater")
        means = {c: cond_mean(ret, info, c) for c in COLORS}
        hi = np.mean([means["C3"][0], means["C4"][0]])
        lo = np.mean([means["C1"][0], means["C2"][0]])
        tag = "P1" if env == "E5_gravity" else "P2"
        verdicts[f"{tag}a"] = (rho > 0 and p < 0.05, f"ρ = {rho:.3f}, 片側 p = {p:.4f}, n = {len(runs)}")
        verdicts[f"{tag}b"] = (hi > lo, f"C3・C4 平均 {hi:.3f} vs C1・C2 平均 {lo:.3f}")

        ax = axes[0, col]
        for c, color in COLORS.items():
            xs = [info[r]["margin"] for r in runs if info[r]["cond"] == c]
            ys = [ret[r] for r in runs if info[r]["cond"] == c]
            ax.scatter(xs, ys, color=color, label=c, s=40, alpha=0.85)
        ax.set_xlabel("トルクの余力（1 − ピーク/上限、訓練環境）")
        ax.set_ylabel(f"{label} の性能維持率")
        ax.set_title(f"{label}: Spearman ρ = {rho:.2f}（片側 p = {p:.3f}）")
        ax.grid(alpha=0.3)
        ax = axes[1, col]
        for c, color in COLORS.items():
            if c not in curves:
                continue
            xs = sorted(curves[c])
            ax.plot(xs, [np.mean(curves[c][v]) for v in xs], marker="o", color=color, label=c,
                    ls="--" if c == "C0" else "-")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("追従報酬の累積 [s]（最大20）")
        ax.set_title(f"{label}: 強度と性能（有効なrunの平均）")
        ax.grid(alpha=0.3)

        lines += [f"## {label}（副評価）", "",
                  f"- 順位相関（余力 vs 維持率）: {verdicts[tag + 'a'][1]}",
                  f"- {verdicts[tag + 'b'][1]}", "", "| 条件 | 維持率（有効なrunの平均） | n |", "|---|---|---|"]
        lines += [f"| {c} | {m:.3f} | {n} |" for c, (m, n) in means.items()]
        lines.append("")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_h4_margin.png", dpi=160)
    plt.close(fig)
    excluded = sorted(r for r, v in info.items() if not v["valid"])
    lines += [f"- 除外したrun（副評価の訓練環境で追従報酬 < {MIN_E0_TRACKING_SECONDARY:.0f}）: "
              + (", ".join(excluded) if excluded else "なし"), ""]

    # P3
    rep = replication()
    tests = [("P3a", "E2_payload", "C1", "C0", "E2 荷物: C1 > C0"),
             ("P3b", "E3_torque", "C3", "C0", "E3 トルク低下: C3 > C0"),
             ("P3c", "E1_friction", "C0", "C3", "E1 低摩擦: C0 > C3")]
    raw = []
    for _, env, a, b, _ in tests:
        va, vb = list(rep[env][a].values()), list(rep[env][b].values())
        raw.append(float(stats.ttest_ind(va, vb, equal_var=False, alternative="greater").pvalue))
    adj = holm(raw)
    lines += ["## 再現（新シード4〜8のみ、主評価）", "",
              "| 予測 | 新シードの平均維持率 | 片側 p | Holm補正後 p | 判定 |", "|---|---|---|---|---|"]
    for (tag, env, a, b, text), p, pa in zip(tests, raw, adj):
        ma, mb = np.mean(list(rep[env][a].values())), np.mean(list(rep[env][b].values()))
        verdicts[tag] = (pa < 0.05, f"{a} {ma:.3f} vs {b} {mb:.3f}, p = {p:.4f}, Holm p = {pa:.4f}")
        lines.append(f"| {tag} {text} | {a} {ma:.3f} / {b} {mb:.3f} | {p:.4f} | {pa:.4f} | "
                     f"{'再現' if pa < 0.05 else '再現せず'} |")
    lines.append("")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (tag, env, a, b, text) in zip(axes, tests):
        for i, c in enumerate(("C0", "C1", "C3")):
            vals = list(rep[env][c].values())
            ax.bar(i, np.mean(vals), color=COLORS[c], alpha=0.35)
            ax.scatter([i] * len(vals), vals, color=COLORS[c], s=25, zorder=3)
            old = old_seed_retention(env, c)
            ax.scatter([i + 0.25] * len(old), old, marker="x", color="gray", s=25, zorder=3)
        ax.set_xticks(range(3), ["C0", "C1", "C3"])
        ax.set_title(f"{tag} {text}")
        ax.set_ylabel("性能維持率（主評価）")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("再現: 棒と点 = 新シード4〜8（判定に使用）、灰色の× = 本実験のシード1〜3（参考）")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig_h4_replication.png", dpi=160)
    plt.close(fig)

    lines += ["## 判定のまとめ", "", "| 予測 | 結果 | 詳細 |", "|---|---|---|"]
    for k in ("P1a", "P1b", "P2a", "P2b", "P3a", "P3b", "P3c"):
        ok, detail = verdicts[k]
        lines.append(f"| {k} | {'成立' if ok else '不成立'} | {detail} |")
    lines += ["", f"**H4（主要な検定 P1a による）: {'支持' if verdicts['P1a'][0] else '不支持'}**"]
    with open(f"{OUT}/summary_h4.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
