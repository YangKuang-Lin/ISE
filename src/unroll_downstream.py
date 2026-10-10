"""U=4 unrolled training with the full downstream suite (review round 5).

Appendix C shows U=4 beating U=1 on every fidelity axis (three seeds), while
the bottleneck baseline's U=4 reimplementation shows slightly worse retrieval
than U=1 -- and the two U=4 implementations disagree with each other. This
run settles the question with the canonical implementation: distill's
multistep trainer, three seeds, full evaluation (rollout drift, partial-query
retrieval, STS-B, Pareto sweep), with checkpoints saved so SciFact retrieval
can be run on the same models.

Usage:
    python unroll_downstream.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from experiments.cells import DEFAULT_NUM_LAYERS, default_hidden
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
)
from distill import DistillConfig, run_once

UNROLL = 4


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--ckpt-dir", type=str, default="experiments/results/ckpt")
    ap.add_argument("--out", type=str,
                    default="experiments/results/unroll_downstream.json")
    args = ap.parse_args()

    device = get_device()
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    cfg = DistillConfig(cell="gated_film", teacher=args.teacher,
                        hidden_dim=default_hidden(teacher.emb_dim),
                        num_layers=DEFAULT_NUM_LAYERS)
    docs = load_wikitext_docs(cfg.dataset_config, cfg.num_docs + 2000,
                              cfg.min_words)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    per_seed = []
    for seed in args.seeds:
        r, model = run_once(cfg, seed, device, teacher, docs,
                            unroll=UNROLL, full_eval=True)
        # In-memory rollout_drift uses int keys; normalize to the string keys
        # every consumer (and the on-disk JSON format) expects.
        r["rollout_drift"] = {str(k): float(v)
                              for k, v in r["rollout_drift"].items()}
        torch.save(model.state_dict(),
                   ckpt_dir / f"gated_film_u4full_seed{seed}.pt")
        per_seed.append(r)
        print(f"  [seed={seed}] ss={r['single_step']['ise_cos']:.5f} "
              f"cos64={float(r['rollout_drift']['64']):.3f} "
              f"retr={r['retrieval']['ise'] * 100:.1f} "
              f"sts={r['sts']['ise_spearman'] * 100:.1f}")

    steps = sorted(int(k) for k in per_seed[0]["rollout_drift"])
    mean_curve = {k: float(np.mean([float(p["rollout_drift"][str(k)])
                                    for p in per_seed])) for k in steps}
    horizon = next((k for k in steps if mean_curve[k] < 0.90), steps[-1] + 1)

    results: Dict = {
        "unroll": UNROLL, "seeds": args.seeds,
        "per_seed": per_seed,
        "single_step_cos": aggregate_seeds(
            [p["single_step"]["ise_cos"] for p in per_seed]),
        "retrieval_ise": aggregate_seeds(
            [p["retrieval"]["ise"] for p in per_seed]),
        "sts_ise": aggregate_seeds(
            [p["sts"]["ise_spearman"] for p in per_seed]),
        "mean_curve": {str(k): mean_curve[k] for k in steps},
        "horizon_mean_curve": horizon,
        "cos64": aggregate_seeds(
            [float(p["rollout_drift"]["64"]) for p in per_seed]),
    }
    save_results(args.out, results)
    print(f"  U={UNROLL}: horizon={horizon} "
          f"cos64={results['cos64']['mean']:.3f} "
          f"retr={results['retrieval_ise']['mean'] * 100:.1f} "
          f"sts={results['sts_ise']['mean'] * 100:.1f}")


if __name__ == "__main__":
    main()
