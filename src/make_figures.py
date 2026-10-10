"""Render publication figures (vector PDF) from experiment result JSON files.

Produces the rollout-drift, re-encode-budget Pareto, speed/memory, and
teacher-law figures with multi-seed error bands, proper legends, axis labels,
and readable fonts. The teacher-law panel reads the per-teacher distill JSONs
directly from experiments/results/ and is skipped if fewer than four are
present.

Usage:
    python make_figures.py --distill experiments/results/distill_fixed.json \
                           --speed experiments/results/speed.json \
                           --outdir ../paper/assets
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from experiments.utils import load_results  # noqa: E402

plt.rcParams.update({
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "figure.figsize": (3.4, 2.4),
    "savefig.bbox": "tight",
    "legend.frameon": False,
})

BLUE, ORANGE, GREEN, RED = "#3b6ea5", "#e07b39", "#4c9a5a", "#b5483d"


def fig_rollout_drift(distill: dict, out: Path,
                      linear: dict | None = None) -> None:
    seeds = distill["main"]["per_seed"]
    steps = sorted(int(k) for k in seeds[0]["rollout_drift"])
    arr = np.array([[s["rollout_drift"][str(k)] for k in steps] for s in seeds])
    mean, std = arr.mean(0), arr.std(0)

    plt.figure(figsize=(4, 1.8))
    plt.plot(steps, mean, color=BLUE, lw=1.6, label="ISE (mean of 3 seeds)")
    plt.fill_between(steps, mean - std, mean + std, color=BLUE, alpha=0.25)
    # Copy-forward null (hold the seed embedding): the training-free curve
    # the cell must beat, review round 6.
    if linear and "hold_curve" in linear:
        hold = [linear["hold_curve"][str(k)] for k in steps
                if str(k) in linear["hold_curve"]]
        plt.plot(steps[:len(hold)], hold, color="gray", lw=1.3, ls="-.",
                 label="Copy-forward (hold seed)")
    plt.axhline(0.9, ls="--", c="gray", lw=1)
    # Mark where the curve actually crosses 0.90.
    cross = next((k for k, m in zip(steps, mean) if m < 0.90), None)
    if cross:
        plt.axvline(cross, ls=":", c=RED, lw=1)
        plt.annotate(f"0.90 at t={cross}", (cross, 0.905), color=RED, fontsize=9,
                     textcoords="offset points", xytext=(2, 2))
    plt.xlabel("Incremental steps $t$", fontsize=9.5)
    plt.ylabel(r"$\cos(\hat{E}_t,\ E^{\ast}_t)$", fontsize=9.5)
    plt.ylim(0.75, 1.0)
    plt.legend(loc="upper right", fontsize=9)
    plt.savefig(out / "rollout_drift.pdf", bbox_inches="tight",pad_inches=0)
    plt.close()


def fig_teacher_law(out: Path) -> bool:
    """Decision map: teacher copy-forward residual -> drift horizon, five
    teachers. The x-value is measurable with zero training; the y-value is
    what decides streaming viability (review round 6 centerpiece)."""
    import json

    teachers = [
        ("MiniLM-L6", "distill.json"),
        ("e5-base-v2", "distill_e5.json"),
        ("bge-base", "distill_bge.json"),
        ("mpnet-base", "distill_mpnet.json"),
        ("gte-large", "distill_gte.json"),
    ]
    base = Path("experiments/results")
    pts = []
    for label, fname in teachers:
        p = base / fname
        if not p.exists():
            continue
        seeds = json.loads(p.read_text(encoding="utf-8"))["main"]["per_seed"]
        shift = (1 - float(np.mean([s["single_step"]["zero_delta_cos"]
                                    for s in seeds]))) * 100
        steps = sorted(int(k) for k in seeds[0]["rollout_drift"])
        curve = {k: float(np.mean([s["rollout_drift"][str(k)] for s in seeds]))
                 for k in steps}
        horizon = next((k for k in steps if curve[k] < 0.90), 65)
        pts.append((label, shift, horizon))
    if len(pts) < 4:
        print("SKIP teacher_law: fewer than 4 teachers available")
        return False

    plt.figure(figsize=(3.4, 2.5))
    xs = [p[1] for p in pts]
    ys = [p[2] for p in pts]
    order = np.argsort(xs)
    plt.plot(np.array(xs)[order], np.array(ys)[order], c="lightgray", lw=1,
             zorder=1)
    offsets = {"MiniLM-L6": (-56, -3), "e5-base-v2": (5, 2),
               "bge-base": (5, -9), "mpnet-base": (-38, 7),
               "gte-large": (6, -3)}
    for (label, x, y), color in zip(pts, [BLUE, GREEN, ORANGE, RED, "#7d5ba6"]):
        censored = y >= 65
        plt.scatter([x], [y], c=color, s=34,
                    marker="^" if censored else "o", zorder=3)
        txt = label + (" (>64)" if censored else "")
        plt.annotate(txt, (x, y), textcoords="offset points",
                     xytext=offsets.get(label, (5, 3)), fontsize=8)
    plt.yscale("log")
    plt.yticks([4, 8, 16, 32, 64], ["4", "8", "16", "32", "64"])
    plt.xlabel(r"Teacher copy-forward residual ($\times 10^{-2}$)",
               fontsize=9.5)
    plt.ylabel("Drift horizon (steps)", fontsize=9.5)
    plt.savefig(out / "teacher_law.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close()
    return True


def fig_pareto(distill: dict, out: Path) -> None:
    offsets = [
        (5, -2),
        (-20, 5),
        (-2.5, -11.5),
        (3, -9),
        (3, -9),
        (3, -9),
    ]
    seeds = distill["main"]["per_seed"]
    budgets = [r["reencode_every"] for r in seeds[0]["pareto"]]
    frac = np.array([[r["teacher_call_fraction"] for r in s["pareto"]] for s in seeds]).mean(0)
    mean_cos = np.array([[r["mean_cos"] for r in s["pareto"]] for s in seeds]).mean(0)
    pre = np.array([[r["pre_reset_cos"] for r in s["pareto"]] for s in seeds]).mean(0)

    order = np.argsort(frac)
    plt.figure()
    plt.plot(frac[order] * 100, mean_cos[order], marker="o", ms=4, color=BLUE,
             label="fixed $K$")
    plt.plot(frac[order] * 100, pre[order], marker="s", ms=4, color=ORANGE,
             ls="--", label="fixed $K$, pre-reset")
    for j, i in enumerate(order):
        lbl = "no re-encode" if budgets[i] == 0 else f"$K$={budgets[i]}"
        dx, dy = offsets[j]
        plt.annotate(
            lbl,
            (frac[i] * 100, mean_cos[i]),
            textcoords="offset points",
            xytext=(dx, dy),
            fontsize=8,
        )

    # Adaptive (path-length) policy, if present.
    ad = distill["main"].get("adaptive")
    if ad:
        af = np.array([r["teacher_call_fraction"] for r in ad]) * 100
        ac = np.array([r["mean_cos"] for r in ad])
        o = np.argsort(af)
        plt.plot(af[o], ac[o], marker="^", ms=5, color=GREEN,
                 label=r"adaptive ($\tau$)")

    plt.xlabel("Teacher calls (% of full re-encoding)", fontsize=9)
    plt.ylabel(r"$\cos(\hat{E},\ E^{\ast})$", fontsize=9)
    plt.legend(loc="lower right", fontsize=8)
    plt.savefig(out / "pareto_reencode.pdf")
    plt.close()


def fig_speed(speed: dict, out: Path) -> None:
    """Two panels: throughput and peak memory vs generation length."""
    # Plot the largest batch whose sweep is complete for every model. A batch
    # with a failed cell (the no-cache baseline exhausts the allocator at
    # batch 4, 2048 tokens) would truncate exactly the series that carries the
    # O(N) memory story, and would also disagree with the batch the paper
    # quotes in Appendix F.
    keys = ["ISE", "GRU", "TF_cached", "TF_nocache"]
    batches = sorted({r["batch"] for r in speed["rows"]})
    complete = [bb for bb in batches
                if all(r.get(f"{k}_tok_per_s") is not None
                       for r in speed["rows"] if r["batch"] == bb
                       for k in keys)]
    b = (complete or batches)[-1]
    rows = sorted([r for r in speed["rows"] if r["batch"] == b], key=lambda r: r["gen_tokens"])
    x = [r["gen_tokens"] for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.9, 2.4))
    series = [("ISE", BLUE, "-"), ("GRU", GREEN, "-"),
              ("TF_cached", ORANGE, "-"), ("TF_nocache", RED, "--")]
    labels = {"ISE": "ISE", "GRU": "GRU",
              "TF_cached": "Transformer (KV cache)", "TF_nocache": "Transformer (no cache)"}
    for key, color, ls in series:
        # A cell can be absent (the no-cache baseline at batch 4, 2048 tokens
        # exhausts the allocator); plot only the points that were measured.
        pts = [(g, r.get(f"{key}_tok_per_s"), r.get(f"{key}_peak_mb"))
               for g, r in zip(x, rows) if r.get(f"{key}_tok_per_s") is not None]
        axes[0].plot([p[0] for p in pts], [p[1] for p in pts], marker="o", ms=3,
                     color=color, ls=ls, label=labels[key])
        axes[1].plot([p[0] for p in pts], [p[2] for p in pts], marker="o", ms=3,
                     color=color, ls=ls, label=labels[key])
    axes[0].set_xlabel("Generated tokens", fontsize=10)
    axes[0].set_ylabel("Throughput (tok/s)", fontsize=10)
    axes[0].set_yscale("log")
    axes[1].set_xlabel("Generated tokens", fontsize=10)
    axes[1].set_ylabel("Peak memory (MB)", fontsize=10)
    axes[1].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Autoregressive generation (batch {b})", fontsize=11)
    fig.savefig(out / "speed_scaling.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--distill", type=str, default=None)
    ap.add_argument("--speed", type=str, default=None)
    ap.add_argument("--linear", type=str,
                    default="experiments/results/linear.json",
                    help="linear.json providing the copy-forward null curve")
    ap.add_argument("--outdir", type=str, default="../paper/assets")
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    linear = (load_results(args.linear)
              if args.linear and Path(args.linear).exists() else None)
    if args.distill and Path(args.distill).exists():
        d = load_results(args.distill)
        fig_rollout_drift(d, out, linear)
        print("wrote rollout_drift.pdf")
        if "mean_cos" in d["main"]["per_seed"][0]["pareto"][0]:
            fig_pareto(d, out)
            print("wrote pareto_reencode.pdf")
        else:
            print("SKIP pareto: results use the old degenerate schema; re-run distill.py")
    if args.speed and Path(args.speed).exists():
        fig_speed(load_results(args.speed), out)
        print("wrote speed_scaling.pdf")
    if fig_teacher_law(out):
        print("wrote teacher_law.pdf")


if __name__ == "__main__":
    main()
