"""BEIR-style retrieval with incrementally embedded queries (review round 3, Q4).

The earlier retrieval study used 500 WikiText documents with self-retrieval gold
labels, which reviewers reasonably called a toy setting. This runs the same
comparison on SciFact from BEIR: a real corpus (5.2k abstracts), real queries,
and human relevance judgements, scored with nDCG@10.

The corpus is always encoded once by the frozen teacher -- that is the premise of
the method, since the whole point is to keep an existing index intact. What
varies is how the *query* is embedded as it streams in:

  teacher   -- full re-encode of the query (upper bound)
  ise       -- incremental rollout from an n0-token seed, never resetting
  hybrid-K  -- incremental with a teacher re-encode every K tokens
  static    -- mean of the teacher's input embedding rows (no contextualization)

Usage:
    python beir.py --ckpt experiments/results/ckpt/gated_film_unroll1.pt
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Dict, List

from pathlib import Path

import numpy as np
import torch

from experiments.cells import build_cell, build_default_cell
from experiments.distill_data import Teacher
from experiments.utils import aggregate_over_ckpts, get_device, log_environment, save_results, set_seed
from broader import embed_ise, embed_ise_calls, embed_static


def ndcg_at_k(ranked_ids: List[str], relevance: Dict[str, int], k: int = 10) -> float:
    """Standard nDCG@k with graded relevance from the qrels."""
    gains = [relevance.get(doc, 0) for doc in ranked_ids[:k]]
    dcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / np.log2(i + 2) for i, g in enumerate(ideal))
    return float(dcg / idcg) if idcg > 0 else 0.0


DATASETS = {
    "scifact": ("mteb/scifact", "BeIR/scifact-qrels", "test"),
    "fiqa": ("mteb/fiqa", "BeIR/fiqa-qrels", "test"),
    # Counter-argument retrieval: queries are full argument passages
    # (~200 subword tokens), so the mean rollout finally exceeds the drift
    # horizon on a task with real relevance judgments (review round 7).
    "arguana": ("mteb/arguana", "BeIR/arguana-qrels", "test"),
    # Used as a *distillation corpus* only (scientific abstracts disjoint from
    # the SciFact index), never as an evaluation set here.
    "nfcorpus": ("mteb/nfcorpus", "BeIR/nfcorpus-qrels", "test"),
}


def load_beir(dataset: str, max_corpus: int):
    """Return (corpus_ids, corpus_texts, queries, qrels)."""
    from datasets import load_dataset

    ds_id, qrels_id, split = DATASETS[dataset]
    corpus = load_dataset(ds_id, "corpus", split="corpus")
    queries = load_dataset(ds_id, "queries", split="queries")
    qrels = load_dataset(qrels_id, split=split)

    rel: Dict[str, Dict[str, int]] = defaultdict(dict)
    for r in qrels:
        rel[str(r["query-id"])][str(r["corpus-id"])] = int(r["score"])

    # Keep every judged document, then pad up to max_corpus with distractors so
    # the ranking task stays hard without exceeding the memory budget.
    judged = {d for q in rel.values() for d in q}
    ids, texts = [], []
    for row in corpus:
        cid = str(row["_id"])
        if cid in judged:
            ids.append(cid)
            texts.append((row.get("title", "") + " " + row["text"]).strip())
    for row in corpus:
        if len(ids) >= max_corpus:
            break
        cid = str(row["_id"])
        if cid not in judged:
            ids.append(cid)
            texts.append((row.get("title", "") + " " + row["text"]).strip())

    qtexts = {str(r["_id"]): r["text"] for r in queries}
    keep = [q for q in rel if q in qtexts]
    return ids, texts, {q: qtexts[q] for q in keep}, rel


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"],
                    help="one checkpoint per seed; results are averaged over them")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--n0", type=int, nargs="+", default=[4, 10],
                    help="seed lengths to evaluate for the incremental policies")
    ap.add_argument("--budgets", type=int, nargs="+", default=[8])
    ap.add_argument("--dataset", type=str, default="scifact",
                    choices=["scifact", "fiqa", "arguana"])
    ap.add_argument("--max-corpus", type=int, default=60000)
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--truncate-queries", type=int, nargs="+", default=[],
                    help="evaluate once per query-truncation length "
                         "(teacher subword tokens); separates rollout "
                         "depth from dataset identity")
    ap.add_argument("--normalize-state", action="store_true")
    ap.add_argument("--tau", type=float, default=0.6,
                    help="path-length threshold for the adaptive-reset policy")
    ap.add_argument("--out", type=str, default="experiments/results/beir.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()

    teacher = Teacher(args.teacher, device=device)

    ids, texts, queries, rel = load_beir(args.dataset, args.max_corpus)
    # --max-queries 0 evaluates the full split. Round 11: the ArguAna
    # equivalence argument leans on the width of the bootstrap interval, so the
    # query count has to be stated and, where affordable, complete.
    qids = sorted(queries)
    if args.max_queries > 0:
        qids = qids[:args.max_queries]
    print(f"  corpus={len(ids)} docs | queries={len(qids)} (of {len(queries)} judged)")

    # Corpus is encoded once by the teacher and then frozen, as in deployment.
    # It is also identical across checkpoints, so it is encoded once here
    # rather than once per seed.
    corpus_emb = teacher.encode_texts(texts, role="passage")
    corpus_emb = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-9)

    methods = ["teacher", "static"]
    # Seed-only control (review round 5): the teacher encoding of the first
    # n0 tokens, with no cell and no further calls -- isolates how much of the
    # incremental policies' retention is the truncated teacher embedding.
    methods += [f"seed_n{n}" for n in args.n0]
    methods += [f"ise_n{n}" for n in args.n0]
    methods += [f"hybrid{b}_n{n}" for b in args.budgets for n in args.n0]
    # Deployed content-adaptive policy, one per seed length, to compare against
    # the fixed hybrid-K at its true teacher-call budget.
    methods += [f"adaptive_n{n}" for n in args.n0]
    methods += [f"stale{b}_n{n}" for b in args.budgets for n in args.n0]

    def evaluate(model):
        """nDCG@10 (and teacher-call fraction for incremental policies).

        Also returns the per-query score vector per method so paired
        significance tests can be run over queries (review round 4)."""
        out: Dict[str, float] = {}
        per_query: Dict[str, List[float]] = {}
        for method in methods:
            scores, fracs = [], []
            for qid in qids:
                text = queries[qid]
                if method == "teacher":
                    q = teacher.encode_texts([text])[0]
                elif method == "static":
                    q = embed_static(teacher, text)
                elif method.startswith("seed_n"):
                    n0 = int(method.split("_n")[-1])
                    ids_q = teacher.token_ids(text, 256)
                    q = teacher.encode_prefixes(ids_q, [min(n0, len(ids_q))])[0] \
                        if ids_q else embed_static(teacher, text)
                elif method.startswith("stale"):
                    # Re-encode every K, hold in between, no cell: the final
                    # query embedding is the teacher encoding of the longest
                    # prefix coinciding with a reset (or the seed if none).
                    parts = method.split("_")
                    n0 = int(parts[-1][1:])
                    k = int(parts[0][5:])
                    ids_q = teacher.token_ids(text, 256)
                    steps = max(0, len(ids_q) - n0)
                    last = n0 + k * (steps // k) if steps >= k else n0
                    q = teacher.encode_prefixes(ids_q, [min(last, len(ids_q))])[0] \
                        if ids_q else embed_static(teacher, text)
                elif method.startswith("adaptive"):
                    n0 = int(method.split("_")[-1][1:])
                    q, fr = embed_ise_calls(model, teacher, text, n0, device,
                                            0, args.normalize_state, args.tau)
                    fracs.append(fr)
                else:
                    parts = method.split("_")
                    n0 = int(parts[-1][1:])
                    k = int(parts[0][6:]) if parts[0].startswith("hybrid") else 0
                    q, fr = embed_ise_calls(model, teacher, text, n0, device, k,
                                            args.normalize_state, 0.0)
                    fracs.append(fr)
                q = q / (np.linalg.norm(q) + 1e-9)
                order = np.argsort(-(corpus_emb @ q))[:10]
                scores.append(ndcg_at_k([ids[i] for i in order], rel[qid], 10))
            out[method] = float(np.mean(scores)) * 100
            per_query[method] = [float(s) * 100 for s in scores]
            if fracs:
                out[f"{method}_callfrac"] = float(np.mean(fracs)) * 100
        return out, per_query

    def one_pass():
        """Full per-seed evaluation + aggregation for the current `queries`."""
        per_ckpt, per_query_ckpts = [], []
        for ck in args.ckpt:
            model = build_default_cell("gated_film", teacher.emb_dim).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            model.eval()
            r, pq = evaluate(model)
            print("  [" + Path(ck).name + "] "
                  + "  ".join(f"{m}={v:.2f}" for m, v in r.items()))
            per_ckpt.append({"results": r,
                             "retention": {m: v / r["teacher"] * 100
                                           for m, v in r.items()
                                           if not m.endswith("_callfrac")}})
            per_query_ckpts.append(pq)

        per_query_mean = {m: np.mean([pq[m] for pq in per_query_ckpts], axis=0)
                          for m in per_query_ckpts[0]}

        def paired_bootstrap(a, b, iters: int = 10000):
            rng = np.random.default_rng(0)
            diff = a - b
            n = len(diff)
            boots = np.array([diff[rng.integers(0, n, n)].mean()
                              for _ in range(iters)])
            p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
            return {"mean_diff": float(diff.mean()), "p": min(p, 1.0),
                    "ci95": [float(np.percentile(boots, 2.5)),
                             float(np.percentile(boots, 97.5))]}

        candidates = [(f"hybrid{b}_n{n}", f"stale{b}_n{n}")
                      for b in args.budgets for n in args.n0]
        candidates += [("teacher", f"hybrid{b}_n{n}")
                       for b in args.budgets for n in args.n0]
        candidates += [(f"hybrid{b}_n{n}", f"ise_n{n}")
                       for b in args.budgets for n in args.n0]
        candidates += [("hybrid8_n10", "adaptive_n10"),
                       ("adaptive_n4", "hybrid8_n4")]
        seen, pairs = set(), []
        for a, b in candidates:
            if (a, b) in seen or a not in per_query_mean or b not in per_query_mean:
                continue
            seen.add((a, b))
            pairs.append((a, b))
        significance = {f"{a}_vs_{b}": paired_bootstrap(per_query_mean[a],
                                                        per_query_mean[b])
                        for a, b in pairs}
        for k, v in significance.items():
            print(f"  {k}: diff={v['mean_diff']:+.2f} nDCG  p={v['p']:.4f}  "
                  f"CI95=[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]")

        mean, std = aggregate_over_ckpts(per_ckpt)
        return {"results": mean["results"], "retention": mean["retention"],
                "results_std": std["results"], "retention_std": std["retention"],
                "significance": significance,
                "per_query": {m: list(v) for m, v in per_query_mean.items()},
                "per_ckpt": per_ckpt}

    payload = {"dataset": args.dataset, "teacher_model": args.teacher,
               "ckpt": args.ckpt, "n_ckpt": len(args.ckpt),
               "n_corpus": len(ids), "n_queries": len(qids),
               "metric": "ndcg@10"}

    if not args.truncate_queries:
        payload.update(one_pass())
    else:
        # Truncate the query *text* so the teacher baseline sees the same input
        # as the incremental policies at every depth.
        full = dict(queries)
        payload["truncate_tokens"] = list(args.truncate_queries)
        payload["by_length"] = {}
        for t in args.truncate_queries:
            queries = {q: teacher.tokenizer.decode(
                teacher.tokenizer.encode(full[q], add_special_tokens=False)[:t])
                for q in qids}
            kept = [len(teacher.tokenizer.encode(full[q], add_special_tokens=False))
                    for q in qids]
            print(f"\n== truncation {t} tokens "
                  f"({sum(1 for k in kept if k > t)} of {len(qids)} queries cut) ==")
            payload["by_length"][str(t)] = one_pass()
        queries = full

    save_results(args.out, payload)


if __name__ == "__main__":
    main()
