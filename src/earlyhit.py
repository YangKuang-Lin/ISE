"""Retrieval-while-streaming: when does the gold document first surface?

Review round 4 asked for an experiment where a per-token embedding changes a
downstream *decision*, not just a fidelity number. This measures exactly that:
as a query streams in token by token, at which token does the gold document
first enter the top-1 (and top-5) of the ranking -- and how many teacher calls
has each policy spent by then?

Setup mirrors the partial-query protocol: 500 teacher-encoded WikiText
documents as the corpus, the query is a prefix of one of them, streamed for 64
incremental steps past a 10-token seed. Policies: unreset ISE, hybrid-8,
adaptive tau=0.6, stale-8, and the teacher re-encoding every token (upper
bound). A policy that tracks the teacher closely surfaces the right document
earlier and at fewer teacher calls; a stale policy can only surface it at
reset boundaries.

Usage:
    python earlyhit.py --ckpt <seed ckpts...>
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

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

MIN_TOKENS, ROLLOUT_K, MAX_TOKENS = 10, 64, 80
N_QUERIES = 300


def _norm(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


@torch.no_grad()
def stream_states(model, teacher: Teacher, ids: List[int], policy: str,
                  device: str, tau: float = 0.6, k: int = 8):
    """Yield (state, cumulative_teacher_calls) after each incremental step."""
    n0 = MIN_TOKENS
    rows = teacher.token_embed(ids)
    if policy == "teacher":
        ks = list(range(n0 + 1, n0 + ROLLOUT_K + 1))
        true = teacher.encode_prefixes(ids, ks)
        for i, _ in enumerate(ks):
            yield true[i], 1 + (i + 1)
        return
    e = torch.tensor(teacher.encode_prefixes(ids, [n0])[0], dtype=torch.float32,
                     device=device).unsqueeze(0)
    held = e.squeeze(0).cpu().numpy()
    calls, s = 1, 0.0
    for step in range(1, ROLLOUT_K + 1):
        t = n0 + step - 1
        if policy == "stale":
            if step % k == 0:
                held = teacher.encode_prefixes(ids, [t + 1])[0]
                calls += 1
            yield held, calls
            continue
        tok = torch.tensor(rows[t], dtype=torch.float32, device=device).unsqueeze(0)
        delta = model(tok, e)
        e = e + delta
        s += float(delta.norm().item())
        reset = (policy == "hybrid" and step % k == 0) or \
                (policy == "adaptive" and s > tau)
        if reset:
            e = torch.tensor(teacher.encode_prefixes(ids, [t + 1])[0],
                             dtype=torch.float32, device=device).unsqueeze(0)
            calls, s = calls + 1, 0.0
        yield e.squeeze(0).cpu().numpy(), calls


def evaluate(model, teacher, usable, corpus, device: str) -> Dict[str, Dict]:
    """First-hit latency plus the incremental-processing consumer metrics
    (review round 6): a downstream system reads the top-1 document after every
    token, so what it experiences is (a) how early the answer surfaces,
    (b) how often the answer *flips* (edit overhead: each flip is a revision
    the consumer must undo), and (c) from which step the answer is stable.

    Review round 8 (reviewer Q2) added the per-query arrays this returns
    alongside the means. The conditional means (``mean_first_top1``) average
    only over queries a policy ever gets right, so they are computed on
    different query subsets for different policies and cannot be compared
    pairwise; the ``*_censored`` arrays assign ``ROLLOUT_K + 1`` to a query the
    policy never surfaces, which keeps every policy on the same 300 queries and
    refuses to reward a policy for failing quietly. Those are the arrays the
    significance test in :func:`main` uses.
    """
    policies = ["teacher", "ise", "hybrid", "stale", "adaptive"]
    censor = ROLLOUT_K + 1
    out: Dict[str, Dict] = {}
    for policy in policies:
        first1, first5, calls1, hit1, hit5 = [], [], [], 0, 0
        flips, frac_correct, stables, stable_hits = [], [], [], 0
        pq_first1, pq_stable = [], []
        for gold, ids in enumerate(usable[:N_QUERIES]):
            top1: List[int] = []
            f5: Optional[int] = None
            calls_by_step = []
            for step, (state, calls) in enumerate(
                    stream_states(model, teacher, ids, policy, device), start=1):
                sims = corpus @ _norm(state)
                order = np.argsort(-sims)[:5]
                top1.append(int(order[0]))
                calls_by_step.append(calls)
                if f5 is None and gold in order:
                    f5 = step
            correct = [t == gold for t in top1]
            if any(correct):
                f1 = correct.index(True) + 1
                hit1 += 1
                first1.append(f1)
                calls1.append(calls_by_step[f1 - 1])
                pq_first1.append(float(f1))
            else:
                pq_first1.append(float(censor))
            if f5 is not None:
                hit5 += 1
                first5.append(f5)
            flips.append(sum(a != b for a, b in zip(top1, top1[1:]))
                         / max(len(top1) - 1, 1))
            frac_correct.append(float(np.mean(correct)))
            if correct and correct[-1]:
                stable = len(correct)
                while stable > 1 and correct[stable - 2]:
                    stable -= 1
                stables.append(stable)
                stable_hits += 1
                pq_stable.append(float(stable))
            else:
                pq_stable.append(float(censor))
        n = len(usable[:N_QUERIES])
        nan = float("nan")
        out[policy] = {
            "hit1_rate": hit1 / n, "hit5_rate": hit5 / n,
            "median_first_top1": float(np.median(first1)) if first1 else nan,
            "mean_first_top1": float(np.mean(first1)) if first1 else nan,
            "median_first_top5": float(np.median(first5)) if first5 else nan,
            "mean_calls_at_top1": float(np.mean(calls1)) if calls1 else nan,
            "flip_rate": float(np.mean(flips)),
            "frac_correct": float(np.mean(frac_correct)),
            "stable_rate": stable_hits / n,
            "median_stable_step": float(np.median(stables)) if stables else nan,
            # Paired, uncensored-subset-free versions: every policy is scored on
            # all n queries, with a never-surfaced query charged ROLLOUT_K + 1.
            "mean_first_top1_censored": float(np.mean(pq_first1)),
            "mean_stable_step_censored": float(np.mean(pq_stable)),
            "per_query_first_top1": pq_first1,
            "per_query_stable_step": pq_stable,
        }
        print(f"  {policy:>9}: top1 {out[policy]['hit1_rate'] * 100:5.1f}% "
              f"@ median tok {out[policy]['median_first_top1']} "
              f"(mean {out[policy]['mean_first_top1']:.2f}, "
              f"censored {out[policy]['mean_first_top1_censored']:.2f}) "
              f"(calls {out[policy]['mean_calls_at_top1']}) | "
              f"top5 {out[policy]['hit5_rate'] * 100:5.1f}% | "
              f"flips {out[policy]['flip_rate'] * 100:4.1f}% "
              f"correct {out[policy]['frac_correct'] * 100:4.1f}% "
              f"stable@{out[policy]['median_stable_step']}")
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 10000):
    """Paired bootstrap over queries on the difference a - b.

    Same protocol as beir.py / longstream.py, so every significance
    number in the paper is produced one way. Lower is better for latency
    metrics, so callers should read a negative ``mean_diff`` as ``a`` deciding
    earlier than ``b``.
    """
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(iters, len(d)))
    boot = d[idx].mean(axis=1)
    obs = float(d.mean())
    return {"mean_diff": obs,
            "p": float(2 * min((boot <= 0).mean(), (boot >= 0).mean())),
            "ci95": [float(np.percentile(boot, 2.5)),
                     float(np.percentile(boot, 97.5))],
            "n": int(len(d))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--out", type=str, default="experiments/results/earlyhit.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()
    teacher = Teacher(args.teacher, device=device)

    docs = load_wikitext_docs("wikitext-2-raw-v1", 8000, 40)[4000:]
    query_len = MIN_TOKENS + ROLLOUT_K
    usable = []
    for doc in docs:
        ids = teacher.token_ids(doc, 160)
        if len(ids) >= query_len + 1:
            usable.append(ids)
        if len(usable) >= 500:
            break
    corpus = np.stack([teacher.encode_prefixes(ids, [len(ids)], role="passage")[0]
                       for ids in usable])
    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)

    per_ckpt, per_query_ckpts = [], []
    for ck in args.ckpt:
        model = build_default_cell("gated_film", teacher.emb_dim).to(device)
        model.load_state_dict(torch.load(ck, map_location=device))
        model.eval()
        print(f"  [{ck}]")
        res = evaluate(model, teacher, usable, corpus, device)
        # Per-query arrays are averaged across checkpoints separately, as in
        # beir.py: aggregate_over_ckpts collapses scalars, not vectors.
        per_query_ckpts.append(
            {p: {"first_top1": v.pop("per_query_first_top1"),
                 "stable_step": v.pop("per_query_stable_step")}
             for p, v in res.items()})
        per_ckpt.append({"results": res})

    mean, std = aggregate_over_ckpts(per_ckpt)
    policies = list(per_query_ckpts[0])
    per_query_mean = {
        p: {m: np.mean([pq[p][m] for pq in per_query_ckpts], axis=0)
            for m in ("first_top1", "stable_step")}
        for p in policies}

    # Reviewer Q2: the decision-latency gaps carried the between-reset claim
    # with no dispersion and no test. hybrid-vs-stale at an identical cadence
    # is the comparison the paper argues; the rest are reported for context.
    pairs = [("hybrid", "stale"), ("hybrid", "ise"), ("hybrid", "teacher"),
             ("hybrid", "adaptive"), ("adaptive", "stale")]
    significance = {}
    for metric in ("first_top1", "stable_step"):
        for a, b in pairs:
            if a in per_query_mean and b in per_query_mean:
                significance[f"{metric}:{a}_vs_{b}"] = paired_bootstrap(
                    per_query_mean[a][metric], per_query_mean[b][metric])
    for key, v in significance.items():
        print(f"  {key}: diff={v['mean_diff']:+.3f} steps  p={v['p']:.4f}  "
              f"ci95=[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]  n={v['n']}")

    save_results(args.out, {"n_queries": min(N_QUERIES, len(usable)),
                            "n_corpus": len(usable), "tau": args.tau,
                            "censor_step": ROLLOUT_K + 1,
                            "bootstrap_iters": 10000,
                            "results": mean["results"],
                            "results_std": std["results"],
                            "significance": significance,
                            "per_query": {p: {m: list(v)
                                              for m, v in d.items()}
                                          for p, d in per_query_mean.items()},
                            "per_ckpt": per_ckpt})


if __name__ == "__main__":
    main()
