"""Drift as a function of absolute prefix position (review round 4).

The geometric model over-predicts late drift, which suggests per-step drift
decelerates as the prefix grows: each new token contributes proportionally
less to a longer prefix, so the teacher's own per-token shift shrinks. If so,
the "10-step horizon" is partly a property of *short* prefixes, a fixed K
could safely grow with position, and drift-proportional spacing has a
principled account of why it wins on long documents.

This probe measures, on documents long enough to support it:
  * the teacher's own shift norm ||E_{n+1} - E_n|| by absolute position n,
  * the unreset rollout drift curve started from seeds at n0 in {10, 50, 100},
    i.e. the same 32-step rollout launched at different absolute depths.

Usage:
    python drift_position.py --ckpt <seed ckpts...>
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch

from experiments.cells import build_default_cell
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import (
    aggregate_over_ckpts,
    get_device,
    log_environment,
    save_results,
    set_seed,
)

SEEDS_AT = [10, 50, 100]
STEPS = 32
POS_BUCKETS = [(10, 30), (30, 60), (60, 100), (100, 140)]
N_DOCS = 100


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@torch.no_grad()
def teacher_shift_by_position(teacher: Teacher, docs: List[str]) -> Dict[str, float]:
    """Mean ||E_{n+1} - E_n|| for n in each absolute-position bucket."""
    sums: Dict[str, List[float]] = {f"{a}-{b}": [] for a, b in POS_BUCKETS}
    used = 0
    for doc in docs:
        ids = teacher.token_ids(doc, POS_BUCKETS[-1][1] + 2)
        if len(ids) < POS_BUCKETS[-1][1] + 1:
            continue
        ns = list(range(POS_BUCKETS[0][0], POS_BUCKETS[-1][1]))
        embs = teacher.encode_prefixes(ids, ns + [ns[-1] + 1])
        for i, n in enumerate(ns):
            d = float(np.linalg.norm(embs[i + 1] - embs[i]))
            for a, b in POS_BUCKETS:
                if a <= n < b:
                    sums[f"{a}-{b}"].append(d)
        used += 1
        if used >= N_DOCS:
            break
    return {k: float(np.mean(v)) for k, v in sums.items() if v}


@torch.no_grad()
def rollout_from(model, teacher: Teacher, docs: List[str], n0: int,
                 device: str) -> Dict[str, float]:
    """Mean unreset drift curve for a STEPS-step rollout seeded at depth n0."""
    per_step: Dict[int, List[float]] = {k: [] for k in range(1, STEPS + 1)}
    used = 0
    for doc in docs:
        ids = teacher.token_ids(doc, n0 + STEPS + 2)
        if len(ids) < n0 + STEPS + 1:
            continue
        ks = list(range(n0, n0 + STEPS + 1))
        true = teacher.encode_prefixes(ids, ks)
        rows = teacher.token_embed(ids)
        e = torch.tensor(true[0], dtype=torch.float32, device=device).unsqueeze(0)
        for k in range(1, STEPS + 1):
            tok = torch.tensor(rows[n0 + k - 1], dtype=torch.float32,
                               device=device).unsqueeze(0)
            e = e + model(tok, e)
            per_step[k].append(_cos(e.squeeze(0).cpu().numpy(), true[k]))
        used += 1
        if used >= N_DOCS:
            break
    return {str(k): float(np.mean(v)) for k, v in per_step.items() if v}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--out", type=str,
                    default="experiments/results/drift_position.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    docs = load_wikitext_docs("wikitext-2-raw-v1", 8000, 80)[4000:]

    shift = teacher_shift_by_position(teacher, docs)
    print("  teacher shift by position:", {k: round(v, 4) for k, v in shift.items()})

    per_ckpt = []
    for ck in args.ckpt:
        model = build_default_cell("gated_film", teacher.emb_dim).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        entry = {"curves": {str(n0): rollout_from(model, teacher, docs, n0, device)
                            for n0 in SEEDS_AT}}
        for n0, curve in entry["curves"].items():
            lastk = str(STEPS)
            print(f"  [{ck}] n0={n0}: cos@{STEPS}={curve[lastk]:.3f}")
        per_ckpt.append(entry)

    mean, std = aggregate_over_ckpts(per_ckpt)
    save_results(args.out, {"teacher_shift_by_position": shift,
                            "seeds_at": SEEDS_AT, "steps": STEPS,
                            "curves": mean["curves"],
                            "curves_std": std["curves"],
                            "per_ckpt": per_ckpt})


if __name__ == "__main__":
    main()
