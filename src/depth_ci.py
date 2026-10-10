"""Bootstrap confidence intervals for the depth-ladder decomposition shares.

Review-6 W3 objects that the schedule/update shares of Table~\\ref{tab:depth}
are six point estimates with no uncertainty attached, and that the ladder is
read as monotone when its first two rungs are not ordered. The shares are
ratios of differences between mean nDCG values, so their sampling variability
follows from resampling *queries* -- which the depth run already stores, at
every truncation depth, as ``per_query`` vectors of length ``n_queries``.

No new model evaluation is therefore needed. This script re-derives, per
resample, all four cells of the 2x2, both average-marginal (Shapley)
contributions and the share, and reports percentile intervals.

Two statistics are produced per depth:

  sched_pct   the schedule's share of the attributed gain (the headline)
  cond        the update's conditional marginal, hybrid - stale, in nDCG

The share is a ratio whose denominator (hybrid - seed) is itself estimated, so
its bootstrap distribution is heavier-tailed than the margins'. The
conditional marginal has no such problem and is the cleaner statistic; we
report both so a reader can prefer it.

Usage:
    python depth_ci.py --iters 10000
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Tuple

import numpy as np

from experiments.utils import load_results, save_results

logger = logging.getLogger(__name__)

# The four cells of the 2x2, in the order (seed-only, unreset, stale, hybrid).
CELLS: Tuple[str, str, str, str] = ("seed_n10", "ise_n10", "stale8_n10",
                                    "hybrid8_n10")


def _marginals(seed: float, unre: float, stal: float,
               hybr: float) -> Tuple[float, float, float]:
    """Average-marginal (Shapley) split over the two orderings of the 2x2.

    Returns (schedule, update, total); the first two sum exactly to the third.
    """
    sched = ((stal - seed) + (hybr - unre)) / 2.0
    upd = ((unre - seed) + (hybr - stal)) / 2.0
    return sched, upd, hybr - seed


def _percentile_ci(boots: np.ndarray, alpha: float = 5.0) -> List[float]:
    return [float(np.percentile(boots, alpha / 2)),
            float(np.percentile(boots, 100 - alpha / 2))]


def bootstrap_depth(per_query: Dict[str, List[float]], iters: int,
                    seed: int) -> Dict:
    """Resample queries and recompute the split on each resample.

    The same resampled index set is used for all four cells, so the four
    policies stay paired exactly as they are in the point estimate.
    """
    cols = np.array([per_query[c] for c in CELLS], dtype=np.float64)
    n = cols.shape[1]
    rng = np.random.default_rng(seed)

    point = _marginals(*cols.mean(axis=1))
    shares = np.empty(iters, dtype=np.float64)
    conds = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        idx = rng.integers(0, n, n)
        s, u, t = _marginals(*cols[:, idx].mean(axis=1))
        shares[i] = s / t * 100.0 if t != 0 else np.nan
        conds[i] = cols[3, idx].mean() - cols[2, idx].mean()

    shares = shares[np.isfinite(shares)]
    return {
        "n_queries": int(n),
        "sched": point[0], "upd": point[1], "total": point[2],
        "sched_pct": point[0] / point[2] * 100.0,
        "upd_pct": point[1] / point[2] * 100.0,
        "sched_pct_ci95": _percentile_ci(shares),
        "sched_pct_sd": float(shares.std(ddof=1)),
        "cond": float(cols[3].mean() - cols[2].mean()),
        "cond_ci95": _percentile_ci(conds),
        "n_valid": int(shares.size),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth-json", type=str,
                    default="experiments/results/beir_arguana_depth.json")
    ap.add_argument("--full-json", type=str,
                    default="experiments/results/beir_arguana_ksweep.json",
                    help="untruncated queries: the ladder's 'full' rung, which "
                         "lives in the cadence sweep rather than the depth run")
    ap.add_argument("--iters", type=int, default=10000,
                    help="bootstrap resamples; matches the nDCG protocol")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="experiments/results/depth_ci.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    src = load_results(args.depth_json)
    by_length = {k: v for k, v in src["by_length"].items()}
    # The untruncated rung is scored by the cadence sweep on the same split.
    full = load_results(args.full_json)
    if "per_query" in full:
        by_length["full"] = {"per_query": full["per_query"]}

    out: Dict[str, Dict] = {}
    for depth in sorted(by_length, key=lambda d: (d == "full", d != "full"
                                                  and int(d))):
        pq = by_length[depth].get("per_query")
        if not pq or any(c not in pq for c in CELLS):
            logger.warning("depth %s: no per-query vectors, skipping", depth)
            continue
        out[depth] = bootstrap_depth(pq, args.iters, args.seed)
        r = out[depth]
        logger.info("  depth %-5s sched=%5.1f%%  CI95=[%5.1f, %5.1f]  "
                    "cond=%+.2f CI95=[%+.2f, %+.2f]",
                    depth, r["sched_pct"], r["sched_pct_ci95"][0],
                    r["sched_pct_ci95"][1], r["cond"],
                    r["cond_ci95"][0], r["cond_ci95"][1])

    # Do the endpoints separate? This is the claim the ladder actually rests on.
    keys = [k for k in ("32", "full") if k in out]
    endpoints = None
    if len(keys) == 2:
        lo, hi = out[keys[0]], out[keys[1]]
        endpoints = {
            "shallow": keys[0], "deep": keys[1],
            "shallow_ci95": lo["sched_pct_ci95"],
            "deep_ci95": hi["sched_pct_ci95"],
            "disjoint": bool(lo["sched_pct_ci95"][1] < hi["sched_pct_ci95"][0]),
        }
        logger.info("endpoint intervals disjoint: %s", endpoints["disjoint"])

    save_results(args.out, {
        "source": args.depth_json, "iters": args.iters, "seed": args.seed,
        "cells": list(CELLS), "by_length": out, "endpoints": endpoints,
    })


if __name__ == "__main__":
    main()
