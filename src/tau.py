"""Sensitivity of the adaptive reset threshold tau (review round 2, Q2).

Loads an already-trained cell and sweeps the path-length threshold at several
rollout lengths, so the question "does a tau tuned at one horizon still behave
sensibly at another?" can be answered without retraining. Also re-runs the
fixed-K sweep at each length, so the adaptive-vs-fixed comparison is like-for-like.

Usage:
    python tau.py --ckpt experiments/results/ckpt/gated_film_unroll1.pt \
                      --lengths 32 64 128
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import torch

from experiments.cells import DEFAULT_NUM_LAYERS, build_cell, default_hidden
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import aggregate_over_ckpts, get_device, log_environment, save_results, set_seed
from distill import DistillConfig, adaptive_sweep, pareto_sweep


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"],
                    help="one checkpoint per seed; results are averaged")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--lengths", type=int, nargs="+", default=[32, 64, 128])
    ap.add_argument("--taus", type=float, nargs="+",
                    default=[0.15, 0.25, 0.4, 0.6, 0.9, 1.4])
    ap.add_argument("--min-k", type=int, nargs="+", default=[0],
                    help="minimum steps between adaptive resets (0 = no floor)")
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--normalize-state", action="store_true")
    ap.add_argument("--out", type=str, default="experiments/results/tau.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()

    teacher = Teacher(args.teacher, device=device)
    base = DistillConfig(teacher=args.teacher,
                         hidden_dim=default_hidden(teacher.emb_dim),
                         num_layers=DEFAULT_NUM_LAYERS,
                         normalize_state=args.normalize_state)
    # Evaluation documents must be long enough for the longest rollout tested.
    longest = max(args.lengths)
    docs = load_wikitext_docs(base.dataset_config, base.num_docs + 3000, base.min_words)
    eval_docs = [d for d in docs[base.num_docs:]
                 if len(teacher.token_ids(d, base.max_tokens + longest + 1))
                 >= base.min_tokens + longest + 1][:args.docs]
    print(f"  {len(eval_docs)} eval docs long enough for rollout_k={longest}")

    def sweep(model) -> Dict[str, Dict]:
        """Full length x min-K sweep for one cell checkpoint."""
        by_length: Dict[str, Dict] = {}
        for L in args.lengths:
            cfg_L = replace(base, rollout_k=L)
            fx = pareto_sweep(model, teacher, eval_docs, cfg_L, device)
            by_mink = {}
            for mk in args.min_k:
                ad = adaptive_sweep(model, teacher, eval_docs, cfg_L, device,
                                    args.taus, mk)
                by_mink[str(mk)] = ad
                print(f"  L={L} min_k={mk}:")
                for r in ad:
                    print(f"    tau={r['tau']:<5} "
                          f"calls={r['teacher_call_fraction']*100:5.1f}%"
                          f"  cos={r['mean_cos']:.4f}"
                          f"  rho={r['path_vs_drift_spearman']:+.3f}")
            # Keep the no-floor sweep under "adaptive" for backward compatibility.
            by_length[str(L)] = {"adaptive": by_mink[str(args.min_k[0])],
                                 "by_min_k": by_mink, "fixed": fx}
        return by_length

    # Averaged over seed checkpoints, matching the distillation protocol.
    per_ckpt = []
    for ck in args.ckpt:
        model = build_cell(base.cell, teacher.emb_dim, base.hidden_dim,
                           base.num_layers).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        print(f"  [{Path(ck).stem}]")
        per_ckpt.append({"by_length": sweep(model)})

    mean, std = aggregate_over_ckpts(per_ckpt)
    save_results(args.out, {"teacher": args.teacher, "ckpt": args.ckpt,
                            "n_ckpt": len(args.ckpt), "taus": args.taus,
                            "by_length": mean["by_length"],
                            "by_length_std": std["by_length"],
                            "per_ckpt": per_ckpt})


if __name__ == "__main__":
    main()
