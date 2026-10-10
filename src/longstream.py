"""Long-stream retrieval with a task metric (review round 5).

The downstream evaluations use short queries where the incremental path
barely runs, and the long-stream benchmark reports only fidelity. This
experiment closes the gap: documents long enough that per-token re-encoding
is genuinely expensive, scored with a retrieval decision at every depth.

Setup: 500 synthetic long documents built by concatenating consecutive
WikiText-2 paragraphs (a simulated document stream). The corpus is the
teacher encoding of each document's first 300 tokens. A query is a growing
prefix of one document, embedded under each policy; at rollout checkpoints
{32, 64, 128, 256} past a 10-token seed we record retrieval accuracy@1/@5
of the source document and the teacher calls spent so far.

Policies: teacher (re-encode at the checkpoint; per-token upper bound),
hybrid-8, stale-4, stale-8 (the matched-wall-clock GPU competitor),
adaptive tau=0.6, unreset ISE, and window-64 (teacher encodes only the last
64 tokens: O(w) per step, training-free, approximately in-space).

Checkpoints are deliberately PRE-reset depths {31, 63, 127, 255}: one step
before the K=8 cadence fires (and mid-interval for K=4), so cadence policies
are measured at their between-reset worst case rather than at a step where
they have just been handed the teacher's own embedding. Depths aligned with
the cadence would make hybrid/stale trivially identical to the teacher.

Usage:
    python longstream.py --ckpt <seed ckpts...>
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

N0 = 10
CHECKPOINTS = [31, 63, 127, 255]
CORPUS_TOKENS = 300
N_CORPUS = 500          # the paper's setting; accuracy@1 saturates at depth
N_QUERIES = 200         # 255 for every re-grounding policy, and enlarging the
                        # corpus does not change that (round 6 checked 3000) --
                        # the query prefix is 265 of the document's 300 tokens
TAU = 0.6


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def build_long_docs(teacher: Teacher, paragraphs: List[str], need: int,
                    min_tokens: int) -> List[List[int]]:
    """Concatenate consecutive paragraphs into documents of >= min_tokens."""
    docs, cur = [], ""
    for p in paragraphs:
        cur = (cur + " " + p).strip()
        ids = teacher.token_ids(cur, min_tokens + 8)
        if len(ids) >= min_tokens:
            docs.append(ids[:min_tokens])
            cur = ""
            if len(docs) >= need:
                break
    return docs


@torch.no_grad()
def states_at_checkpoints(model, teacher: Teacher, ids: List[int],
                          policy: str, device: str) -> Dict[int, Dict]:
    """Embedding + cumulative calls at each rollout checkpoint."""
    rows = teacher.token_embed(ids)
    out: Dict[int, Dict] = {}
    if policy == "teacher":
        for c in CHECKPOINTS:
            out[c] = {"emb": teacher.encode_prefixes(ids, [N0 + c])[0],
                      "calls": 1 + c}     # per-token re-encoding up to depth c
        return out
    if policy == "window64":
        for c in CHECKPOINTS:
            t = N0 + c
            lo = max(0, t - 64)           # clamp: no negative-index wraparound
            out[c] = {"emb": teacher.encode_prefixes(ids[lo:t], [t - lo])[0],
                      "calls": 1 + c}     # also a teacher pass per token (O(w))
        return out
    e = torch.tensor(teacher.encode_prefixes(ids, [N0])[0], dtype=torch.float32,
                     device=device).unsqueeze(0)
    held = e.squeeze(0).cpu().numpy()
    calls, s = 1, 0.0
    k = {"hybrid8": 8, "stale4": 4, "stale8": 8}.get(policy, 0)
    for step in range(1, CHECKPOINTS[-1] + 1):
        t = N0 + step - 1
        if policy.startswith("stale"):
            if step % k == 0:
                held = teacher.encode_prefixes(ids, [t + 1])[0]
                calls += 1
            if step in CHECKPOINTS:
                out[step] = {"emb": held.copy(), "calls": calls}
            continue
        tok = torch.tensor(rows[t], dtype=torch.float32, device=device).unsqueeze(0)
        delta = model(tok, e)
        e = e + delta
        s += float(delta.norm().item())
        reset = (policy == "hybrid8" and step % 8 == 0) or \
                (policy == "adaptive" and s > TAU)
        if reset:
            e = torch.tensor(teacher.encode_prefixes(ids, [t + 1])[0],
                             dtype=torch.float32, device=device).unsqueeze(0)
            calls, s = calls + 1, 0.0
        if step in CHECKPOINTS:
            out[step] = {"emb": e.squeeze(0).cpu().numpy(), "calls": calls}
    return out


def evaluate(model, teacher, queries, corpus, device: str):
    """Return (aggregate metrics, per-query top-1 hit vectors) per policy."""
    policies = ["teacher", "hybrid8", "stale4", "stale8", "adaptive",
                "ise", "window64"]
    res: Dict[str, Dict] = {}
    raw: Dict[str, Dict[int, List[int]]] = {}
    for policy in policies:
        hits1 = {c: [] for c in CHECKPOINTS}   # per-query, for significance
        hits5 = {c: [] for c in CHECKPOINTS}
        calls = {c: [] for c in CHECKPOINTS}
        for gold, ids in enumerate(queries):
            states = states_at_checkpoints(model, teacher, ids, policy, device)
            for c, st in states.items():
                sims = corpus @ _norm(st["emb"])
                order = np.argsort(-sims)[:5]
                hits1[c].append(int(order[0] == gold))
                hits5[c].append(int(gold in order))
                calls[c].append(st["calls"])
        res[policy] = {str(c): {"acc1": float(np.mean(hits1[c])),
                                "acc5": float(np.mean(hits5[c])),
                                "calls": float(np.mean(calls[c]))}
                       for c in CHECKPOINTS}
        raw[policy] = hits1
        line = "  ".join(f"@{c}: {res[policy][str(c)]['acc1'] * 100:4.1f}/"
                         f"{res[policy][str(c)]['acc5'] * 100:4.1f} "
                         f"({res[policy][str(c)]['calls']:.1f}c)"
                         for c in CHECKPOINTS)
        print(f"  {policy:>9}: {line}")
    return res, raw


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 10000):
    """Paired bootstrap of mean(a) - mean(b) on per-query scores.

    Reports the 95% percentile interval alongside the p-value: reviewers
    asked whether the pre-reset edge over holding survives at depth, and a
    point estimate with a p cannot answer that on its own.
    """
    rng = np.random.default_rng(0)
    diffs = a - b
    n = len(diffs)
    boot = rng.choice(diffs, size=(iters, n), replace=True).mean(axis=1)
    p = 2 * min(float(np.mean(boot <= 0)), float(np.mean(boot >= 0)))
    return {"delta": float(diffs.mean()), "p": min(p, 1.0), "n": int(n),
            "ci95": [float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5))]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--n-corpus", type=int, default=N_CORPUS)
    ap.add_argument("--n-queries", type=int, default=N_QUERIES)
    ap.add_argument("--out", type=str,
                    default="experiments/results/longstream.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()
    teacher = Teacher(args.teacher, device=device)

    paragraphs = load_wikitext_docs("wikitext-2-raw-v1", 120000, 30)
    docs = build_long_docs(teacher, paragraphs, args.n_corpus, CORPUS_TOKENS)
    print(f"  built {len(docs)} synthetic long documents "
          f"({CORPUS_TOKENS} tokens each)")
    corpus = np.stack([teacher.encode_prefixes(ids, [len(ids)], role="passage")[0]
                       for ids in docs])
    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
    queries = docs[:args.n_queries]

    per_ckpt, per_ckpt_raw = [], []
    for ck in args.ckpt:
        model = build_default_cell("gated_film", teacher.emb_dim).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        print(f"  [{ck}]")
        res, raw = evaluate(model, teacher, queries, corpus, device)
        per_ckpt.append({"results": res})
        per_ckpt_raw.append(raw)

    # Paired bootstrap on per-query hit rates averaged over seed checkpoints:
    # the review asked whether the cell's pre-reset edge over holding is
    # statistically real, not just a point estimate.
    def pooled(policy: str, c: int) -> np.ndarray:
        return np.mean([r[policy][c] for r in per_ckpt_raw], axis=0)

    significance = {}
    for other in ("stale8", "stale4", "adaptive"):
        for c in CHECKPOINTS:
            key = f"hybrid8_vs_{other}_at_{c}"
            significance[key] = paired_bootstrap(pooled("hybrid8", c),
                                                 pooled(other, c))
    for key, v in significance.items():
        print(f"  {key}: delta={v['delta'] * 100:+.1f} "
              f"CI95=[{v['ci95'][0] * 100:+.1f}, {v['ci95'][1] * 100:+.1f}] "
              f"p={v['p']:.4f}")

    mean, std = aggregate_over_ckpts(per_ckpt)
    save_results(args.out, {"n_corpus": len(docs), "n_queries": len(queries),
                            "corpus_tokens": CORPUS_TOKENS, "n0": N0,
                            "checkpoints": CHECKPOINTS, "tau": TAU,
                            "results": mean["results"],
                            "results_std": std["results"],
                            "significance": significance,
                            "per_ckpt": per_ckpt})


if __name__ == "__main__":
    main()
