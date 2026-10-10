"""The decisive content-vs-position test for the adaptive reset rule (#5).

The probe (delta_probe.py) showed the cell's update magnitude
$\\lVert\\delta_t\\rVert$ is large on content tokens and small on function
tokens, and tracks the teacher's true per-token shift (Spearman ~0.79). This
asks the operational question: at a *fixed* teacher budget, is it better to
spend re-groundings on the high-$\\delta$ (content) tokens?

For a fixed rollout length $T$ and a fixed number of resets $R$ (hence identical
teacher calls), it compares reset policies that differ only in *which* tokens
they reset at:

  * content    -- the $R$ tokens with the largest cell $\\lVert\\delta\\rVert$
                  (teacher-free: the cell computes it online),
  * filler     -- the $R$ tokens with the smallest $\\lVert\\delta\\rVert$,
  * teacher    -- the $R$ tokens with the largest true teacher shift
                  $\\lVert E_t - E_{t-1}\\rVert$ (an oracle upper bound),
  * even       -- $R$ evenly spaced resets,
  * random     -- $R$ random resets, averaged over samples.

If content $\\approx$ teacher $>$ even $\\approx$ random $>$ filler, then
re-grounding at content tokens is what matters and the cell's own $\\delta$
identifies them without the teacher. The script also dumps a few
token-level examples (word, $\\delta$, whether a reset fires) for a qualitative
figure.

Usage:
    python content_reset.py --ckpt ckpt/gated_film_unroll1_seed41.pt ... [3 seeds]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

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
from placement import mean_fidelity, rollout_with_resets


@torch.no_grad()
def per_token_signals(model, teacher: Teacher, ids: List[int], n0: int,
                      steps: int, device: str):
    """Teacher-forced cell |delta| and true teacher shift for each rollout step."""
    ks = list(range(n0, n0 + steps + 1))
    E = teacher.encode_prefixes(ids, ks)                    # (steps+1, D)
    emb = teacher.token_embed(ids)
    delta = np.zeros(steps + 1)
    shift = np.zeros(steps + 1)
    for k in range(1, steps + 1):
        e_prev = torch.tensor(E[k - 1], dtype=torch.float32, device=device).unsqueeze(0)
        tok = torch.tensor(emb[n0 + k - 1], dtype=torch.float32, device=device).unsqueeze(0)
        delta[k] = float(model(tok, e_prev).norm().item())
        shift[k] = float(np.linalg.norm(E[k] - E[k - 1]))
    return delta, shift


def topk_positions(scores: np.ndarray, r: int, steps: int, largest: bool) -> frozenset:
    """R distinct reset positions in 1..steps-1 by score (last step excluded)."""
    cand = np.arange(1, steps)              # 1..steps-1
    order = cand[np.argsort(scores[cand])]
    picks = order[-r:] if largest else order[:r]
    return frozenset(int(p) for p in picks)


def even_positions(r: int, steps: int) -> frozenset:
    seg = steps // (r + 1)
    return frozenset(seg * (i + 1) for i in range(r))


def adaptive_positions(delta: np.ndarray, r: int, steps: int) -> frozenset:
    """Equal-drift spacing: reset when accumulated |delta| crosses each of the
    R equal thresholds. This is what the cumulative path-length rule does --
    content-aware (denser where delta is high) yet self-spacing (no clustering,
    no long gaps), unlike resetting *at* the top-delta tokens.
    """
    cum = np.cumsum(delta[1:steps])         # positions 1..steps-1
    total = float(cum[-1]) if len(cum) else 0.0
    if total <= 0:
        return even_positions(r, steps)
    thresholds = [(k + 1) * total / (r + 1) for k in range(r)]
    pos, ti = [], 0
    for i, c in enumerate(cum, start=1):
        while ti < r and c >= thresholds[ti]:
            if i not in pos:
                pos.append(i)
            ti += 1
    return frozenset(pos[:r])


def evaluate_policies(model, teacher, docs, n0, steps, r, device,
                      rng: np.random.Generator, n_random: int = 8) -> Dict[str, float]:
    acc = {k: [] for k in ("adaptive", "content", "filler", "teacher", "even", "random")}
    for ids in docs:
        delta, shift = per_token_signals(model, teacher, ids, n0, steps, device)
        sets = {
            "adaptive": adaptive_positions(delta, r, steps),
            "content": topk_positions(delta, r, steps, largest=True),
            "filler": topk_positions(delta, r, steps, largest=False),
            "teacher": topk_positions(shift, r, steps, largest=True),
            "even": even_positions(r, steps),
        }
        for name, rs in sets.items():
            preds, true = rollout_with_resets(model, teacher, ids, n0, steps, rs, device)
            acc[name].append(mean_fidelity(preds, true, rs, steps))
        # random: average over several draws
        rvals = []
        for _ in range(n_random):
            rs = frozenset(int(p) for p in rng.choice(np.arange(1, steps), r, replace=False))
            preds, true = rollout_with_resets(model, teacher, ids, n0, steps, rs, device)
            rvals.append(mean_fidelity(preds, true, rs, steps))
        acc["random"].append(float(np.mean(rvals)))
    return {k: float(np.mean(v)) for k, v in acc.items()}


@torch.no_grad()
def qualitative(model, teacher, ids: List[int], n0: int, steps: int, r: int,
                device: str) -> Dict:
    """One example: tokens with |delta| and the content-policy reset marks."""
    delta, _ = per_token_signals(model, teacher, ids, n0, steps, device)
    resets = topk_positions(delta, r, steps, largest=True)
    toks = teacher.tokenizer.convert_ids_to_tokens(ids)
    seq = []
    for k in range(1, steps + 1):
        seq.append({"token": toks[n0 + k - 1], "step": k,
                    "delta": round(float(delta[k]), 4), "reset": k in resets})
    return {"seed_tokens": toks[:n0], "sequence": seq,
            "reset_steps": sorted(resets)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"],
                    help="one checkpoint per seed; results are averaged")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--resets", type=int, default=3)
    ap.add_argument("--docs", type=int, default=100)
    ap.add_argument("--out", type=str, default="experiments/results/content_reset.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()
    cfg = DistillConfig()
    teacher = Teacher(args.teacher, device=device)

    raw = load_wikitext_docs(cfg.dataset_config, cfg.num_docs + 2000, cfg.min_words)
    docs = []
    for doc in raw[cfg.num_docs:]:
        ids = teacher.token_ids(doc, cfg.max_tokens + args.steps + 1)
        if len(ids) >= args.n0 + args.steps + 1:
            docs.append(ids)
        if len(docs) >= args.docs:
            break
    print(f"  {len(docs)} docs | steps={args.steps} n0={args.n0} resets={args.resets}")

    per_ckpt = []
    examples = None
    for ci, ck in enumerate(args.ckpt):
        model = build_default_cell("gated_film", teacher.emb_dim).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        rng = np.random.default_rng(12345)
        pol = evaluate_policies(model, teacher, docs, args.n0, args.steps,
                                args.resets, device, rng)
        print(f"  [{Path(ck).stem}] " + "  ".join(f"{k}={v:.4f}" for k, v in pol.items()))
        per_ckpt.append({"policies": pol})
        if ci == 0:   # qualitative examples from the first checkpoint
            examples = [qualitative(model, teacher, docs[j], args.n0, args.steps,
                                    args.resets, device) for j in range(min(3, len(docs)))]

    mean, std = aggregate_over_ckpts(per_ckpt)
    p = mean["policies"]
    ps = std["policies"]
    save_results(args.out, {
        "steps": args.steps, "n0": args.n0, "resets": args.resets,
        "n_ckpt": len(args.ckpt), "policies": p, "policies_std": ps,
        "adaptive_minus_even": p["adaptive"] - p["even"],
        "content_minus_filler": p["content"] - p["filler"],
        "adaptive_minus_content": p["adaptive"] - p["content"],
        "examples": examples, "per_ckpt": per_ckpt})

    print("\n  === reset placement at a fixed budget (3-seed mean) ===")
    for k in ("adaptive", "teacher", "even", "content", "random", "filler"):
        print(f"    {k:<9} fidelity={p[k]:.4f} +- {ps[k]:.4f}")
    noise = float(np.mean(list(ps.values())))
    print(f"\n  content - filler   = {p['content'] - p['filler']:+.4f}  (content matters)")
    print(f"  even - content     = {p['even'] - p['content']:+.4f}  (naive content clusters)")
    print(f"  adaptive - even    = {p['adaptive'] - p['even']:+.4f}  (equal-drift spacing wins)")
    print(f"  (seed noise ~{noise:.4f})")


if __name__ == "__main__":
    main()
