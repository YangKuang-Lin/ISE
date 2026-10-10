"""Stale-cadence baseline: re-encode every K tokens, hold constant in between.

Both round-5 reviews ask the same pointed question: why run the distilled cell
at all, rather than simply re-encoding at a fixed cadence and reusing the last
teacher embedding until the next reset? This measures exactly that policy on
the same rollouts as the Pareto sweep, so cell-vs-stale can be compared at an
identical teacher-call budget. The cell earns its keep only if it adds fidelity
between resets.

Usage:
    python stale.py --ckpt experiments/results/ckpt/gated_film_unroll1.pt
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from pathlib import Path

import numpy as np
import torch

from experiments.cells import build_cell, build_default_cell
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import aggregate_over_ckpts, get_device, log_environment, save_results, set_seed
from distill import DistillConfig, _cos, rollout_embed


@torch.no_grad()
def stale_curve(teacher: Teacher, ids: List[int], n0: int, steps: int,
                budget: int) -> List[float]:
    """Per-step cosine of the held (stale) embedding against the truth."""
    ks = list(range(n0, n0 + steps + 1))
    true = teacher.encode_prefixes(ids, ks)
    held = true[0]
    out = []
    for t in range(1, steps + 1):
        if budget and t % budget == 0:
            held = true[t]          # teacher reset; exact at this step
            continue                # reset steps are excluded, as for the cell
        out.append(_cos(held, true[t]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"],
                    help="one checkpoint per seed; results are averaged")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--out", type=str, default="experiments/results/stale.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()

    cfg = DistillConfig()
    teacher = Teacher(args.teacher, device=device)

    docs = load_wikitext_docs(cfg.dataset_config, cfg.num_docs + 2000, cfg.min_words)
    eval_docs = [d for d in docs[cfg.num_docs:]
                 if len(teacher.token_ids(d, cfg.max_tokens + cfg.rollout_k + 1))
                 >= cfg.min_tokens + cfg.rollout_k + 1][:args.docs]
    print(f"  {len(eval_docs)} eval docs, rollout_k={cfg.rollout_k}")

    def build_rows(model) -> List[Dict]:
        rows: List[Dict] = []
        for budget in cfg.reencode_budgets:
            stale_cos, cell_cos, frac = [], [], []
            for doc in eval_docs:
                ids = teacher.token_ids(doc, cfg.max_tokens + cfg.rollout_k + 1)
                sc = stale_curve(teacher, ids, cfg.min_tokens, cfg.rollout_k, budget)
                stale_cos.append(float(np.mean(sc)))
                preds, true, calls = rollout_embed(model, teacher, ids, cfg.min_tokens,
                                                   cfg.rollout_k, budget, device)
                reset = {k for k in range(1, cfg.rollout_k + 1)
                         if budget and k % budget == 0}
                cc = [_cos(preds[k], true[k]) for k in range(1, cfg.rollout_k + 1)
                      if k not in reset]
                cell_cos.append(float(np.mean(cc)))
                frac.append(calls / (cfg.rollout_k + 1))
            rows.append({"reencode_every": budget,
                         "stale_mean_cos": float(np.mean(stale_cos)),
                         "cell_mean_cos": float(np.mean(cell_cos)),
                         "teacher_call_fraction": float(np.mean(frac))})
            r = rows[-1]
            print(f"  K={budget if budget else 'none':<5} calls={r['teacher_call_fraction']*100:5.1f}%"
                  f"  stale={r['stale_mean_cos']:.4f}  cell={r['cell_mean_cos']:.4f}"
                  f"  gain={r['cell_mean_cos']-r['stale_mean_cos']:+.4f}")

        return rows

    per_ckpt = []
    for ck in args.ckpt:
        model = build_default_cell(cfg.cell, teacher.emb_dim).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        print(f"  [{Path(ck).stem}]")
        per_ckpt.append({"rows": build_rows(model)})

    mean, std = aggregate_over_ckpts(per_ckpt)
    save_results(args.out, {"teacher": args.teacher, "ckpt": args.ckpt,
                            "n_ckpt": len(args.ckpt),
                            "rollout_k": cfg.rollout_k,
                            "rows": mean["rows"], "rows_std": std["rows"],
                            "per_ckpt": per_ckpt})


if __name__ == "__main__":
    main()
