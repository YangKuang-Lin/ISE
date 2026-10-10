"""Does reset *placement* matter at a fixed teacher-call budget?

The K=24-vs-K=32 observation in the paper shows two schedules with the same
number of teacher calls reaching different fidelity, but it is confounded: the
two schedules differ in gap *sizes* as well as gap *positions*, so it cannot
separate "where resets land" from "how far apart they are."

This isolates placement. For a fixed rollout length T and a fixed multiset of
inter-reset gaps (hence a fixed number of resets = teacher calls, and a fixed
set of segment lengths), it evaluates every distinct *ordering* of those gaps.
Every schedule spends identical teacher calls and contains identical segment
lengths; only the order -- i.e. where each reset lands -- changes.

  * If mean fidelity is ~constant across orderings, placement does NOT matter
    beyond spacing: the drift rate is stationary and even spacing is optimal.
  * If it varies well beyond seed noise, placement genuinely matters, because
    the per-step drift rate depends on absolute position in the sequence.

It also reports even / front-loaded / back-loaded schedules at the same reset
count for the practical takeaway.

Usage:
    python placement.py --ckpt ckpt/gated_film_unroll1_seed41.pt ... [3 seeds]
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from experiments.cells import build_cell, build_default_cell
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import (
    aggregate_over_ckpts,
    get_device,
    log_environment,
    save_results,
    set_seed,
)
from distill import DistillConfig, _cos


@torch.no_grad()
def rollout_with_resets(model, teacher: Teacher, ids: List[int], n0: int,
                        steps: int, reset_set: frozenset, device: str
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Roll forward ``steps`` tokens, re-encoding exactly at steps in ``reset_set``."""
    ks = list(range(n0, n0 + steps + 1))
    true = teacher.encode_prefixes(ids, ks)
    e = torch.tensor(true[0], dtype=torch.float32, device=device).unsqueeze(0)
    preds = [true[0]]
    tok_table = teacher.token_embed(ids)
    for k in range(1, steps + 1):
        tok = torch.tensor(tok_table[n0 + k - 1], dtype=torch.float32,
                           device=device).unsqueeze(0)
        e = e + model(tok, e)
        if k in reset_set:
            e = torch.tensor(true[k], dtype=torch.float32, device=device).unsqueeze(0)
        preds.append(e.squeeze(0).cpu().numpy())
    return np.array(preds), true


def mean_fidelity(preds, true, reset_set: frozenset, steps: int) -> float:
    """Mean cosine over inter-reset steps (reset steps excluded, as in SS4.3)."""
    vals = [_cos(preds[k], true[k]) for k in range(1, steps + 1) if k not in reset_set]
    return float(np.mean(vals)) if vals else float("nan")


def resets_from_gaps(gaps: Sequence[int]) -> frozenset:
    """Reset positions are the cumulative sums of all but the final (tail) gap."""
    pos, acc = [], 0
    for g in gaps[:-1]:
        acc += g
        pos.append(acc)
    return frozenset(pos)


def evaluate_schedule(model, teacher, docs, n0: int, steps: int,
                      reset_set: frozenset, device: str) -> float:
    fids = []
    for ids in docs:
        preds, true = rollout_with_resets(model, teacher, ids, n0, steps,
                                          reset_set, device)
        fids.append(mean_fidelity(preds, true, reset_set, steps))
    return float(np.mean(fids))


def build_schedules(steps: int, gaps: Sequence[int]) -> Dict[str, Dict]:
    """Named schedules, all with the same reset count over ``steps`` tokens."""
    n_resets = len(gaps) - 1
    seg = steps // (n_resets + 1)
    sched: Dict[str, Dict] = {}
    # Practical comparison: even vs clustered, fixed reset count.
    sched["even"] = {"resets": frozenset(seg * (i + 1) for i in range(n_resets)),
                     "kind": "spacing"}
    sched["front"] = {"resets": frozenset(seg // 2 * (i + 1) for i in range(n_resets)),
                      "kind": "spacing"}
    back0 = steps - seg // 2 * n_resets
    sched["back"] = {"resets": frozenset(back0 + seg // 2 * i for i in range(n_resets)),
                     "kind": "spacing"}
    # Isolation: every distinct ordering of one fixed gap multiset.
    for perm in sorted(set(itertools.permutations(gaps))):
        rs = resets_from_gaps(perm)
        if len(rs) == n_resets:
            sched[f"perm:{'-'.join(map(str, perm))}"] = {"resets": rs, "kind": "perm"}
    return sched


def run_for_ckpt(ckpt: str, teacher, docs, n0: int, steps: int,
                 gaps: Sequence[int], device: str) -> Dict[str, float]:
    model = build_default_cell("gated_film", teacher.emb_dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    out = {}
    for name, spec in build_schedules(steps, gaps).items():
        out[name] = evaluate_schedule(model, teacher, docs, n0, steps,
                                      spec["resets"], device)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"],
                    help="one checkpoint per seed; results are averaged")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--n0", type=int, default=10)
    # Fixed gap multiset (4 gaps -> 3 resets over 64 steps). Distinct values so
    # orderings give distinct reset positions; sums to ``steps``.
    ap.add_argument("--gaps", type=int, nargs="+", default=[8, 16, 16, 24])
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--out", type=str, default="experiments/results/placement.json")
    args = ap.parse_args()

    assert sum(args.gaps) == args.steps, "gaps must sum to steps"
    device = get_device()
    set_seed(42)
    log_environment()
    cfg = DistillConfig()
    teacher = Teacher(args.teacher, device=device)

    raw = load_wikitext_docs(cfg.dataset_config, cfg.num_docs + 2000, cfg.min_words)
    eval_docs = []
    for doc in raw[cfg.num_docs:]:
        ids = teacher.token_ids(doc, cfg.max_tokens + args.steps + 1)
        if len(ids) >= args.n0 + args.steps + 1:
            eval_docs.append(ids)
        if len(eval_docs) >= args.docs:
            break
    print(f"  {len(eval_docs)} eval docs | steps={args.steps} n0={args.n0} "
          f"gaps={args.gaps} ({len(args.gaps) - 1} resets)")

    per_ckpt = []
    for ck in args.ckpt:
        r = run_for_ckpt(ck, teacher, eval_docs, args.n0, args.steps, args.gaps, device)
        print(f"  [{Path(ck).stem}]  even={r['even']:.4f} front={r['front']:.4f} "
              f"back={r['back']:.4f}")
        per_ckpt.append({"schedules": r})

    mean, std = aggregate_over_ckpts(per_ckpt)
    sched = mean["schedules"]
    sstd = std["schedules"]

    perms = {k: v for k, v in sched.items() if k.startswith("perm:")}
    pvals = list(perms.values())
    seed_std = float(np.mean([sstd[k] for k in perms]))  # avg across-seed noise
    summary = {
        "steps": args.steps, "n0": args.n0, "gaps": args.gaps,
        "n_resets": len(args.gaps) - 1, "n_ckpt": len(args.ckpt),
        "schedules": sched, "schedules_std": sstd,
        "perm_min": min(pvals), "perm_max": max(pvals),
        "perm_range": max(pvals) - min(pvals),
        "perm_mean_seed_std": seed_std,
        "spacing": {k: sched[k] for k in ("even", "front", "back")},
    }
    save_results(args.out, {**summary, "per_ckpt": per_ckpt})

    print("\n  === placement isolation (same gaps, permuted order) ===")
    for k in sorted(perms, key=lambda x: -perms[x]):
        resets = sorted(build_schedules(args.steps, args.gaps)[k]["resets"])
        print(f"    {k:<24} fid={perms[k]:.4f}  resets@{resets}")
    print(f"\n  perm range = {summary['perm_range']:.4f} "
          f"(vs mean across-seed std {seed_std:.4f})")
    verdict = ("PLACEMENT MATTERS: range exceeds seed noise"
               if summary["perm_range"] > 3 * seed_std
               else "placement within seed noise: spacing alone explains it")
    print(f"  VERDICT: {verdict}")
    print(f"\n  even={sched['even']:.4f} front={sched['front']:.4f} back={sched['back']:.4f}")


if __name__ == "__main__":
    main()
