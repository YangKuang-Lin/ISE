"""Bootstrap intervals for the decomposition shares and the residual contrasts.

Three reviews asked for quantities that need no new model evaluation, because
``beir`` already stores per-query nDCG for every policy:

  1. Confidence intervals on the SciFact and FiQA schedule/update shares.
     Table~\\ref{tab:decomp} reported point estimates there while the ArguAna
     ladder carried intervals, so the "within noise of SciFact's 60%" claim
     could not be checked from the paper.
  2. The budget-matched margin (hybrid - hold) with an interval on every
     dataset, which is the only contrast measured at equal teacher cost.
  3. The hold-above-teacher residual on ArguAna (stale-K minus teacher),
     dismissed in the text as "noise" without a reported number.

Everything here resamples queries from stored vectors, so it is a re-analysis
rather than an experiment: same 10,000-resample paired protocol as the nDCG
margins, and the same generator seed.

Usage:
    python share_ci.py --iters 10000
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

from experiments.utils import load_results, save_results

logger = logging.getLogger(__name__)

# (label, results file, seed-only key, unreset key, hold key, hybrid key)
DATASETS: List[Tuple[str, str, str, str, str, str]] = [
    ("scifact", "beir.json", "seed_n10", "ise_n10", "stale8_n10", "hybrid8_n10"),
    ("fiqa", "beir_fiqa.json", "seed_n10", "ise_n10", "stale8_n10",
     "hybrid8_n10"),
    ("arguana", "beir_arguana_ksweep.json", "seed_n10", "ise_n10",
     "stale8_n10", "hybrid8_n10"),
]


def _ci(boots: np.ndarray, alpha: float = 5.0) -> List[float]:
    return [float(np.percentile(boots, alpha / 2)),
            float(np.percentile(boots, 100 - alpha / 2))]


def _shapley(seed: float, unre: float, hold: float,
             hybr: float) -> Tuple[float, float, float]:
    """Average-marginal split; the two parts sum exactly to hybrid - seed."""
    sched = ((hold - seed) + (hybr - unre)) / 2.0
    upd = ((unre - seed) + (hybr - hold)) / 2.0
    return sched, upd, hybr - seed


def bootstrap_dataset(pq: Dict[str, List[float]], keys: Tuple[str, ...],
                      iters: int, seed: int) -> Optional[Dict]:
    """Resample queries; recompute the split and the matched margin each time."""
    if any(k not in pq for k in keys):
        return None
    cols = np.array([pq[k] for k in keys], dtype=np.float64)
    n = cols.shape[1]
    rng = np.random.default_rng(seed)

    point = _shapley(*cols.mean(axis=1))
    shares = np.empty(iters)
    margins = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        m = cols[:, idx].mean(axis=1)
        s, _, tot = _shapley(*m)
        shares[i] = s / tot * 100.0 if tot != 0 else np.nan
        margins[i] = m[3] - m[2]          # hybrid - hold, the matched edge
    shares = shares[np.isfinite(shares)]
    return {
        "n_queries": int(n),
        "sched": point[0], "upd": point[1], "total": point[2],
        "sched_pct": point[0] / point[2] * 100.0,
        "sched_pct_ci95": _ci(shares),
        "matched_margin": float(cols[3].mean() - cols[2].mean()),
        "matched_margin_ci95": _ci(margins),
    }


def hold_vs_teacher(pq: Dict[str, List[float]], iters: int,
                    seed: int) -> Dict[str, Dict]:
    """The residual by which hold policies score above the teacher."""
    out: Dict[str, Dict] = {}
    if "teacher" not in pq:
        return out
    tea = np.asarray(pq["teacher"], dtype=np.float64)
    rng = np.random.default_rng(seed)
    for k in sorted(pq):
        if not k.startswith("stale"):
            continue
        diff = np.asarray(pq[k], dtype=np.float64) - tea
        n = len(diff)
        boots = np.array([diff[rng.integers(0, n, n)].mean()
                          for _ in range(iters)])
        p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        out[k] = {"mean_diff": float(diff.mean()), "p": min(p, 1.0),
                  "n": int(n), "ci95": _ci(boots)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="experiments/results/share_ci.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    out: Dict[str, Dict] = {}
    for label, path, *keys in DATASETS:
        try:
            src = load_results(f"experiments/results/{path}")
        except FileNotFoundError:
            logger.warning("%s: %s missing, skipping", label, path)
            continue
        pq = src.get("per_query")
        if not pq:
            logger.warning("%s: no per-query vectors, skipping", label)
            continue
        r = bootstrap_dataset(pq, tuple(keys), args.iters, args.seed)
        if r is None:
            logger.warning("%s: missing a policy column, skipping", label)
            continue
        if label == "arguana":
            r["hold_vs_teacher"] = hold_vs_teacher(pq, args.iters, args.seed)
        out[label] = r
        logger.info("  %-8s sched=%5.1f%% CI95=[%5.1f, %5.1f]  "
                    "matched margin=%+.2f CI95=[%+.2f, %+.2f]  n=%d",
                    label, r["sched_pct"], r["sched_pct_ci95"][0],
                    r["sched_pct_ci95"][1], r["matched_margin"],
                    r["matched_margin_ci95"][0], r["matched_margin_ci95"][1],
                    r["n_queries"])
        for k, v in r.get("hold_vs_teacher", {}).items():
            logger.info("      %s - teacher = %+.2f CI95=[%+.2f, %+.2f]",
                        k, v["mean_diff"], v["ci95"][0], v["ci95"][1])

    save_results(args.out, {"iters": args.iters, "seed": args.seed,
                            "by_dataset": out})


if __name__ == "__main__":
    main()
