"""Broader semantic evaluation across an MTEB-style STS suite.

Addresses the main reservation in review round 2: semantic quality rested on
STS-B alone. This evaluates seven STS tasks and adds the two reference points
the review asked for at either end of the contextualization spectrum:

  teacher   -- full re-encode with the frozen teacher (upper bound)
  ise       -- pure incremental rollout, no resets
  hybrid-K  -- incremental with a teacher re-encode every K tokens
  static    -- mean of the teacher's *input* embedding rows, i.e. a static
               token-lookup embedder with no contextualization at all. This is
               a proxy for the SwiftEmbed-style extreme, not that system itself.

It also sweeps the seed length n0: STS sentences are short, so with a 10-token
seed only a few steps are actually incremental. Reporting n0=4 as well makes the
incremental regime (and hence drift) visible rather than hidden by the seed.

Usage:
    python broader.py --ckpt experiments/results/ckpt/gated_film_unroll1.pt
"""

from __future__ import annotations

import argparse
from typing import Dict, List

from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

from experiments.cells import DEFAULT_NUM_LAYERS, build_cell, default_hidden
from experiments.distill_data import Teacher
from experiments.utils import aggregate_over_ckpts, get_device, log_environment, save_results, set_seed

# (label, HF dataset id, split). All expose sentence1 / sentence2 / score.
TASKS = [
    ("STS12", "mteb/sts12-sts", "test"),
    ("STS13", "mteb/sts13-sts", "test"),
    ("STS14", "mteb/sts14-sts", "test"),
    ("STS15", "mteb/sts15-sts", "test"),
    ("STS16", "mteb/sts16-sts", "test"),
    ("STS-B", "mteb/stsbenchmark-sts", "test"),
    ("SICK-R", "mteb/sickr-sts", "test"),
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@torch.no_grad()
def embed_ise(model, teacher: Teacher, text: str, n0: int, device: str,
              reencode_every: int = 0, normalize: bool = False,
              tau: float = 0.0) -> np.ndarray:
    """Incremental embedding of a sentence.

    ``reencode_every > 0`` re-encodes on a fixed cadence (hybrid-K); ``tau > 0``
    instead re-encodes adaptively, whenever the accumulated update path length
    $\\sum\\lVert\\delta\\rVert$ since the last reset exceeds ``tau`` (the
    content-density-aware rule of \\S4.3). Only one of the two should be set.
    """
    emb, _ = embed_ise_calls(model, teacher, text, n0, device,
                             reencode_every, normalize, tau)
    return emb


@torch.no_grad()
def embed_ise_calls(model, teacher: Teacher, text: str, n0: int, device: str,
                    reencode_every: int = 0, normalize: bool = False,
                    tau: float = 0.0):
    """As ``embed_ise`` but also returns (embedding, teacher_calls / n_tokens),
    the fraction of tokens that triggered a teacher re-encode. Lets the adaptive
    and fixed-K policies be compared at their true teacher-call budgets."""
    ids = teacher.token_ids(text, 256)
    if len(ids) <= n0 + 1:
        return teacher.encode_texts([text])[0], 1.0 / max(len(ids), 1)
    tok_table = teacher.token_embed(ids)
    e = torch.tensor(teacher.encode_prefixes(ids, [n0])[0],
                     dtype=torch.float32, device=device).unsqueeze(0)
    calls, s = 1, 0.0                       # seed counts as the first call
    for k in range(n0, len(ids)):
        tok = torch.tensor(tok_table[k], dtype=torch.float32, device=device).unsqueeze(0)
        delta = model(tok, e)
        e = e + delta
        if normalize:
            e = torch.nn.functional.normalize(e, dim=-1)
        s += float(delta.norm().item())
        step = k - n0 + 1
        reset = (reencode_every and step % reencode_every == 0) or (tau and s > tau)
        if reset:
            e = torch.tensor(teacher.encode_prefixes(ids, [k + 1])[0],
                             dtype=torch.float32, device=device).unsqueeze(0)
            calls, s = calls + 1, 0.0
    return e.squeeze(0).cpu().numpy(), calls / len(ids)


def embed_static(teacher: Teacher, text: str) -> np.ndarray:
    """Static token-lookup baseline: mean of input-embedding rows, no context."""
    ids = teacher.token_ids(text, 256)
    if not ids:
        return np.zeros(teacher.emb_dim, dtype=np.float32)
    return teacher.token_embed(ids).mean(axis=0)


def _ise_embed(model, teacher, text, method, device, normalize, tau):
    """Dispatch one incremental-embedding method: ise / hybridK / adaptive."""
    parts = method.split("_")
    n0 = int(parts[-1][1:])
    if parts[0] == "adaptive":
        emb, frac = embed_ise_calls(model, teacher, text, n0, device,
                                    0, normalize, tau)
        return emb, frac
    k = int(parts[0][6:]) if parts[0].startswith("hybrid") else 0
    emb, frac = embed_ise_calls(model, teacher, text, n0, device, k, normalize, 0.0)
    return emb, frac


def run_task(model, teacher, name: str, ds_id: str, split: str, device: str,
             methods: List[str], limit: int, normalize: bool = False,
             tau: float = 0.6) -> Dict[str, float]:
    from datasets import load_dataset
    data = load_dataset(ds_id, split=split)
    if limit and len(data) > limit:
        data = data.select(range(limit))
    s1 = [r["sentence1"] for r in data]
    s2 = [r["sentence2"] for r in data]
    gold = [float(r["score"]) for r in data]

    out: Dict[str, float] = {"n_pairs": len(gold)}
    for method in methods:
        if method == "teacher":
            e1 = teacher.encode_texts(s1)
            e2 = teacher.encode_texts(s2)
            pred = [_cos(a, b) for a, b in zip(e1, e2)]
        elif method == "static":
            pred = [_cos(embed_static(teacher, a), embed_static(teacher, b))
                    for a, b in zip(s1, s2)]
        else:
            # ise_n10, hybrid8_n4, adaptive_n4; the adaptive path also records
            # its teacher-call fraction so it can be compared to fixed-K at cost.
            fracs = []
            pred = []
            for a, b in zip(s1, s2):
                ea, fa = _ise_embed(model, teacher, a, method, device, normalize, tau)
                eb, fb = _ise_embed(model, teacher, b, method, device, normalize, tau)
                pred.append(_cos(ea, eb))
                fracs.append(0.5 * (fa + fb))
            out[method] = float(spearmanr(pred, gold).correlation) * 100
            out[f"{method}_callfrac"] = float(np.mean(fracs)) * 100
            continue
        out[method] = float(spearmanr(pred, gold).correlation) * 100
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"],
                    help="one checkpoint per seed; results are averaged")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--hidden", type=int, default=None,
                    help="cell hidden dim (default: half the teacher dim)")
    ap.add_argument("--layers", type=int, default=None,
                    help="cell depth (default: the deployed 3 blocks)")
    ap.add_argument("--normalize", action="store_true",
                    help="L2-normalize the state after every update")
    ap.add_argument("--limit", type=int, default=1000,
                    help="max pairs per task (0 = all); keeps runtime bounded")
    ap.add_argument("--tau", type=float, default=0.6,
                    help="path-length threshold for the adaptive-reset method")
    ap.add_argument("--out", type=str, default="experiments/results/broader.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()

    teacher = Teacher(args.teacher, device=device)
    # adaptive_n4 is the deployed content-adaptive policy; hybrid8_n4 is the
    # fixed-cadence reference at the same operating point.
    methods = ["teacher", "ise_n10", "ise_n4", "hybrid8_n4", "adaptive_n4", "static"]

    # One pass per seed checkpoint, then averaged, so the STS retention
    # headline shares the three-seed protocol of the distillation results.
    per_ckpt = []
    for ck in args.ckpt:
        model = build_cell("gated_film", teacher.emb_dim,
                           args.hidden or default_hidden(teacher.emb_dim),
                           args.layers or DEFAULT_NUM_LAYERS).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        rows: Dict[str, Dict[str, float]] = {}
        for name, ds_id, split in TASKS:
            try:
                rows[name] = run_task(model, teacher, name, ds_id, split, device,
                                      methods, args.limit, args.normalize, args.tau)
                print(f"  [{Path(ck).stem}] {name:<8} " + "  ".join(
                    f"{m}={rows[name][m]:.2f}" for m in methods))
            except Exception as exc:  # dataset unavailable / schema mismatch
                print(f"  {name:<8} SKIPPED ({type(exc).__name__}: {exc})")
        avg = ({m: float(np.mean([r[m] for r in rows.values()])) for m in methods}
               if rows else {})
        per_ckpt.append({"tasks": rows, "average": avg})

    mean, std = aggregate_over_ckpts(per_ckpt)
    rows, avg = mean["tasks"], mean["average"]
    if avg:
        print("  " + "-" * 60)
        print("  AVG      " + "  ".join(
            f"{m}={avg[m]:.2f}+-{std['average'][m]:.2f}" for m in methods))

    save_results(args.out, {"teacher_model": args.teacher, "ckpt": args.ckpt,
                            "n_ckpt": len(args.ckpt),
                            "normalize_state": args.normalize,
                            "limit": args.limit, "methods": methods,
                            "tasks": rows, "average": avg,
                            "tasks_std": std.get("tasks", {}),
                            "average_std": std.get("average", {}),
                            "per_ckpt": per_ckpt})


if __name__ == "__main__":
    main()
