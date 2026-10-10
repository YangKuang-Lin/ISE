"""Causal (KV-cached) teacher-aligned students -- the decisive missing baseline.

Review round 8, reviewer Q1. Among every alternative to the update cell, the
causal student with a KV cache is the sharpest: it reuses its prefix
*exactly*, so it has no drift and needs no resets, and a step costs one
attention pass over the cache -- O(N) compute at a very small constant. It is
therefore the baseline that most directly threatens the cell's positioning in
the long-stream regime, which is precisely the regime where the cell's
final-embedding margin over a training-free hold cadence disappears
(ArguAna p=0.53, WoW p=0.054).

Round 6 removed the conditioning asymmetry for the *bidirectional* students
(SeededBiStudent) and round 7 added the O(1) recurrent student
(SeededGRUStudent), but the causal student was never given either treatment.
Worse, the round-5 numbers it is judged by are not trustworthy:

  A TRAINING-BUDGET CONFOUND, FOUND WHILE PREPARING THIS ROUND.
  student.py batches causal students by *document* (batch 64) and
  bidirectional students by *position* (batch 256). From logs/students.log the
  two 2.72M students are geometrically identical, yet over 60 epochs

      bi:     110,978 examples / 256  = 434 steps/epoch -> ~26,000 updates
      causal:  ~3,175 docs     /  64  =  50 steps/epoch -> ~3,000 updates

  an 8.7x optimizer-step deficit. The reported causal collapse (2.4% / 13.6%
  SciFact retention against a size-matched bidirectional student's 51.1% /
  62.1%) is therefore evidence about our harness, not about causal
  architectures, and it cannot support the claim that token-only students
  trail badly.

This script settles the question with four variants, all trained under a
step-matched budget and evaluated on the identical held-out yardsticks:

  causal_matched          2.72M  step-matched, unseeded, last-token pooling
                                 -> isolates the optimizer-budget confound
  causal_matched_meanpool 2.72M  step-matched, unseeded, causal running-mean
                                 pooling -> isolates the second candidate
                                 cause, the mismatch against a mean-pooled
                                 teacher (still exactly incremental: the
                                 running sum is O(1) state)
  causal_seeded           2.84M  anchor at position 0, matched to Seeded bi 2.84M
  causal_seeded_scaled   13.2M   anchor at position 0, matched to Seeded bi 13.2M

Seeded variants are evaluated seed-only *and* re-grounded on the cell's exact
K=8 schedule, so they meet the cell at an identical 9.2% teacher-call budget,
on partial-query retrieval, SciFact, and -- the regime that decides the
paper's claim -- ArguAna's long queries.

Usage:
    python student_causal.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim

from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.students import CausalStudent, SeededCausalStudent
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)
from student import (
    LR,
    MAX_TOKENS,
    MIN_TOKENS,
    NUM_DOCS_BASE,
    NUM_DOCS_SCALED,
    REGROUND_K,
    ROLLOUT_K,
    TRAIN_SPLIT,
    _cos,
    _loss,
    build_student_data,
)

# Document batch size. The round-5 harness used 64, which starved the causal
# students of optimizer updates relative to the position-batched bidirectional
# students; 8 brings the update count to ~24k against the bi students' ~26k at
# the same epoch count and the same total examples seen. The realized counts
# are recorded per variant so the paper can state the match rather than assert
# it.
DOC_BATCH = 8
BI_REFERENCE_UPDATES = 26_000     # logs/students.log, 2.72M bi student, 60 ep

# (label, seeded, pool, d_model, layers, ff_dim, num_docs, epochs)
VARIANTS = [
    ("causal_matched", False, "last", 320, 3, None, NUM_DOCS_BASE, 60),
    ("causal_matched_meanpool", False, "mean", 320, 3, None, NUM_DOCS_BASE, 60),
    ("causal_seeded", True, "last", 320, 3, None, NUM_DOCS_BASE, 60),
    ("causal_seeded_scaled", True, "last", 512, 4, 2048, NUM_DOCS_SCALED, 120),
]

LATENCY_LENGTHS = [128, 512]


def _pad_docs(items: List[Dict], device: str):
    """Pad a batch of documents into (rows, targets, supervision mask)."""
    T = max(len(d["ids"]) for d in items)
    D = items[0]["rows"].shape[1]
    rows = np.zeros((len(items), T, D), dtype=np.float32)
    tgt = np.zeros((len(items), T, D), dtype=np.float32)
    sup = np.zeros((len(items), T), dtype=bool)
    for i, d in enumerate(items):
        n = len(d["ids"])
        rows[i, :n] = d["rows"]
        for j, k in enumerate(range(MIN_TOKENS, n + 1)):
            tgt[i, k - 1] = d["targets"][j]
            sup[i, k - 1] = True
    return (torch.tensor(rows, device=device), torch.tensor(tgt, device=device),
            torch.tensor(sup, device=device))


def _sample_anchors(items: List[Dict], rng: np.random.Generator, device: str):
    """Per-document anchor embedding plus the mask of positions it may serve.

    The anchor is the teacher embedding at a uniformly sampled supervision
    position ``a``; only positions ``j >= a`` are supervised, because in
    deployment the anchor is always the most recent teacher call *at or
    before* the current token. Sampling ``a`` uniformly covers both the
    far-anchor (seed-only) and near-anchor (re-grounded) deployments, exactly
    as student_seeded.py does for the bidirectional students.
    """
    T = max(len(d["ids"]) for d in items)
    D = items[0]["rows"].shape[1]
    anchors = np.zeros((len(items), D), dtype=np.float32)
    allow = np.zeros((len(items), T), dtype=bool)
    for i, d in enumerate(items):
        n = len(d["ids"])
        n_sup = n - MIN_TOKENS + 1
        a = int(rng.integers(0, n_sup))
        anchors[i] = d["targets"][a]
        allow[i, MIN_TOKENS - 1 + a:n] = True
    return torch.tensor(anchors, device=device), torch.tensor(allow, device=device)


def train_causal_matched(data: List[Dict], teacher_dim: int, seeded: bool,
                         pool: str, d_model: int, layers: int, ff_dim,
                         epochs: int, device: str, rng: np.random.Generator
                         ) -> Tuple[nn.Module, int]:
    """Train a causal student; return (model, optimizer updates performed)."""
    cls = SeededCausalStudent if seeded else CausalStudent
    model = cls(teacher_dim, d_model, layers, ff_dim=ff_dim, pool=pool).to(device)
    n_train = int(len(data) * TRAIN_SPLIT)
    train, val = data[:n_train], data[n_train:]
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    steps_per_epoch = (len(train) + DOC_BATCH - 1) // DOC_BATCH
    print(f"  causal seeded={seeded} pool={pool} d={d_model} L={layers} "
          f"params={count_params(model):,} docs={len(train):,} "
          f"updates={steps_per_epoch * epochs:,} "
          f"(bi reference {BI_REFERENCE_UPDATES:,})")

    def batch_loss(items: List[Dict]) -> torch.Tensor:
        rows, tgt, sup = _pad_docs(items, device)
        if seeded:
            anchor, allow = _sample_anchors(items, rng, device)
            mask = sup & allow
            if not bool(mask.any()):
                mask = sup
            pred = model(rows, seed=anchor)
        else:
            mask = sup
            pred = model(rows)
        return _loss(pred[mask], tgt[mask])

    updates = 0
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(train))
        for i in range(0, len(order), DOC_BATCH):
            loss = batch_loss([train[j] for j in order[i:i + DOC_BATCH]])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss in causal student.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            updates += 1
        sched.step()
        model.eval()
        cs = []
        with torch.no_grad():
            for i in range(0, len(val), 64):
                items = val[i:i + 64]
                rows, tgt, sup = _pad_docs(items, device)
                if seeded:
                    anchor, allow = _sample_anchors(items, rng, device)
                    mask = sup & allow
                    if not bool(mask.any()):
                        mask = sup
                    pred = model(rows, seed=anchor)
                else:
                    mask, pred = sup, model(rows)
                cs.append(nn.functional.cosine_similarity(
                    pred[mask], tgt[mask], dim=-1).mean().item())
        v = float(np.mean(cs))
        if v > best:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, updates


# --- incremental inference ---------------------------------------------------
@torch.no_grad()
def causal_rollout(model, teacher: Teacher, ids: List[int], n0: int,
                   device: str, seeded: bool, k: int = 0
                   ) -> Tuple[np.ndarray, int]:
    """Stream ``ids`` past an ``n0`` seed with a KV cache; return (emb, calls).

    Mirrors student_seeded.rollout_embed's control flow exactly so the two
    student families are compared like for like: ``k == 0`` keeps the anchor at
    the seed (one teacher call), ``k > 0`` refreshes it every ``k`` tokens and
    the returned state at a reset step *is* the teacher embedding, as in the
    cell's hybrid policy. Re-grounding re-prefills the cache with the fresh
    anchor; between resets each token costs one cached attention pass.
    """
    if len(ids) <= n0 + 1:
        return teacher.encode_prefixes(ids, [min(n0, len(ids))])[0], 1
    rows = torch.tensor(teacher.token_embed(ids), dtype=torch.float32,
                        device=device).unsqueeze(0)
    anchor = teacher.encode_prefixes(ids, [n0])[0]
    calls = 1

    def prefill(upto: int, anc: np.ndarray):
        seed = torch.tensor(anc, dtype=torch.float32, device=device).unsqueeze(0)
        if seeded:
            return model.prefill(rows[:, :upto], seed=seed)
        return model.prefill(rows[:, :upto])

    emb_t, state = prefill(n0, anchor)
    emb = emb_t.squeeze(0).cpu().numpy()
    for t in range(n0, len(ids)):
        step = t - n0 + 1
        if k and step % k == 0:
            anchor = teacher.encode_prefixes(ids, [t + 1])[0]
            calls += 1
            emb = anchor
            emb_t, state = prefill(t + 1, anchor)
            continue
        emb_t, state = model.step(rows[:, t], state, t)
        emb = emb_t.squeeze(0).cpu().numpy()
    return emb, calls


@torch.no_grad()
def fidelity_by_length(model, teacher, docs, device: str, seeded: bool
                       ) -> Dict[str, float]:
    checkpoints = [MIN_TOKENS + s for s in (1, 8, 16, 32, 64)]
    per_len: Dict[int, List[float]] = {c: [] for c in checkpoints}
    used = 0
    for doc in docs:
        ids = teacher.token_ids(doc, MAX_TOKENS + ROLLOUT_K + 1)
        if len(ids) < MIN_TOKENS + ROLLOUT_K + 1:
            continue
        true = teacher.encode_prefixes(ids, checkpoints)
        for i, c in enumerate(checkpoints):
            emb, _ = causal_rollout(model, teacher, ids[:c], MIN_TOKENS,
                                    device, seeded, k=0)
            per_len[c].append(_cos(emb, true[i]))
        used += 1
        if used >= 100:
            break
    return {str(c): float(np.mean(v)) for c, v in per_len.items()}


@torch.no_grad()
def partial_query_retrieval(model, teacher, docs, device: str, seeded: bool,
                            k: int = 0) -> float:
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
    hits = 0
    for gold, ids in enumerate(usable):
        q, _ = causal_rollout(model, teacher, ids[:query_len], MIN_TOKENS,
                              device, seeded, k=k)
        q = q / (np.linalg.norm(q) + 1e-9)
        if int(np.argmax(corpus @ q)) == gold:
            hits += 1
    return hits / len(usable)


@torch.no_grad()
def beir_eval(model, teacher, device: str, seeded: bool, dataset: str,
              max_queries: int, k: int = 0) -> Dict[str, float]:
    """nDCG@10 on a BEIR dataset, query embedded incrementally.

    Per-query scores are returned so the paper can bootstrap the cell-vs-student
    comparison instead of quoting two point estimates.
    """
    from beir import load_beir, ndcg_at_k

    ids, texts, queries, rel = load_beir(dataset, 60000)
    qids = sorted(queries)[:max_queries]
    corpus_emb = teacher.encode_texts(texts, role="passage")
    corpus_emb = corpus_emb / (np.linalg.norm(corpus_emb, axis=1, keepdims=True) + 1e-9)
    t_scores, s_scores, callfracs = [], [], []
    for qid in qids:
        text = queries[qid]
        tq = teacher.encode_texts([text])[0]
        tq = tq / (np.linalg.norm(tq) + 1e-9)
        order = np.argsort(-(corpus_emb @ tq))[:10]
        t_scores.append(ndcg_at_k([ids[i] for i in order], rel[qid], 10))
        q_ids = teacher.token_ids(text, 256)
        if not q_ids:
            sq = tq
        else:
            sq, calls = causal_rollout(model, teacher, q_ids, MIN_TOKENS,
                                       device, seeded, k=k)
            callfracs.append(calls / len(q_ids))
        sq = sq / (np.linalg.norm(sq) + 1e-9)
        order = np.argsort(-(corpus_emb @ sq))[:10]
        s_scores.append(ndcg_at_k([ids[i] for i in order], rel[qid], 10))
    t, s = float(np.mean(t_scores)) * 100, float(np.mean(s_scores)) * 100
    return {"student": s, "teacher": t, "retention": s / t * 100,
            "callfrac": float(np.mean(callfracs)) * 100 if callfracs else 0.0,
            "per_query": [float(x) * 100 for x in s_scores],
            "per_query_teacher": [float(x) * 100 for x in t_scores]}


@torch.no_grad()
def incremental_latency(model, teacher, docs, device: str, seeded: bool
                        ) -> Dict[str, float]:
    """Student-only wall clock for embedding an N-token stream token by token.

    Excludes teacher calls, so it isolates how the causal student's O(N) cached
    step scales against the cell's O(1) update. The policy-level comparison at
    a matched teacher-call budget lives in crossover.py.
    """
    out: Dict[str, float] = {}
    long_ids: List[int] = []
    for doc in docs:
        long_ids.extend(teacher.token_ids(doc, 512))
        if len(long_ids) >= max(LATENCY_LENGTHS) + MIN_TOKENS + 1:
            break
    for n in LATENCY_LENGTHS:
        ids = long_ids[:n]
        if len(ids) < n:
            continue
        rows = torch.tensor(teacher.token_embed(ids), dtype=torch.float32,
                            device=device).unsqueeze(0)
        anchor = torch.tensor(teacher.encode_prefixes(ids, [MIN_TOKENS])[0],
                              dtype=torch.float32, device=device).unsqueeze(0)

        def once() -> float:
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            if seeded:
                _, state = model.prefill(rows[:, :MIN_TOKENS], seed=anchor)
            else:
                _, state = model.prefill(rows[:, :MIN_TOKENS])
            for t in range(MIN_TOKENS, n):
                _, state = model.step(rows[:, t], state, t)
            if device == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        once()                                   # warm-up
        out[str(n)] = float(min(once() for _ in range(3)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--ckpt-dir", type=str, default="experiments/results/ckpt")
    ap.add_argument("--arguana-queries", type=int, default=300)
    ap.add_argument("--out", type=str,
                    default="experiments/results/student_causal.json")
    args = ap.parse_args()

    device = get_device()
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    docs = load_wikitext_docs("wikitext-2-raw-v1", NUM_DOCS_SCALED + 2000, 40)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("  building student data (one pass over all documents)...")
    data_all = build_student_data(teacher, docs[:NUM_DOCS_SCALED])
    eval_docs = docs[NUM_DOCS_SCALED:]
    print(f"  data: {len(data_all)} docs total, eval pool {len(eval_docs)}")

    per_variant: Dict[str, List[Dict]] = {}
    for seed in args.seeds:
        for label, seeded, pool, d_model, layers, ff_dim, ndocs, epochs in VARIANTS:
            set_seed(seed)
            base = data_all if ndocs >= NUM_DOCS_SCALED else \
                data_all[:int(len(data_all) * ndocs / NUM_DOCS_SCALED)]
            rng = np.random.default_rng(seed)
            data = [base[i] for i in rng.permutation(len(base))]
            model, updates = train_causal_matched(
                data, teacher.emb_dim, seeded, pool, d_model, layers, ff_dim,
                epochs, device, rng)
            model.eval()
            torch.save(model.state_dict(),
                       ckpt_dir / f"student_{label}_seed{seed}.pt")

            entry: Dict = {
                "seed": seed,
                "params": count_params(model),
                "optimizer_updates": updates,
                "pool": pool,
                "seeded": seeded,
                "fidelity_by_length": fidelity_by_length(model, teacher,
                                                         eval_docs, device, seeded),
                "retrieval": partial_query_retrieval(model, teacher, eval_docs,
                                                     device, seeded),
                "scifact": beir_eval(model, teacher, device, seeded,
                                     "scifact", 300),
                "latency_ms": incremental_latency(model, teacher, eval_docs,
                                                  device, seeded),
            }
            if seeded:
                # The regime that decides the paper's claim: long real queries
                # with human judgments, where the cell's margin over holding
                # the last re-encoding is statistically indistinguishable.
                entry["reground"] = {
                    "retrieval": partial_query_retrieval(
                        model, teacher, eval_docs, device, seeded, k=REGROUND_K),
                    "scifact": beir_eval(model, teacher, device, seeded,
                                         "scifact", 300, k=REGROUND_K),
                    "arguana": beir_eval(model, teacher, device, seeded,
                                         "arguana", args.arguana_queries,
                                         k=REGROUND_K),
                }
            per_variant.setdefault(label, []).append(entry)

            sf = entry["scifact"]
            msg = (f"  [{label} seed={seed}] params={entry['params']:,} "
                   f"updates={updates:,} retr={entry['retrieval'] * 100:.1f} "
                   f"scifact={sf['student']:.1f} ({sf['retention']:.1f}%)")
            if seeded:
                rg = entry["reground"]["scifact"]
                ag = entry["reground"]["arguana"]
                msg += (f" | +reground scifact={rg['student']:.1f} "
                        f"({rg['retention']:.1f}% @ {rg['callfrac']:.1f}% calls)"
                        f" arguana={ag['student']:.1f} ({ag['retention']:.1f}%)")
            print(msg)
            save_results(args.out + ".partial",
                         {"teacher_model": args.teacher,
                          "reground_k": REGROUND_K,
                          "partial": per_variant})

    results: Dict = {"teacher_model": args.teacher, "seeds": args.seeds,
                     "reground_k": REGROUND_K, "doc_batch": DOC_BATCH,
                     "bi_reference_updates": BI_REFERENCE_UPDATES,
                     "sizes": {}}
    for label, entries in per_variant.items():
        agg = {
            "per_seed": entries,
            "params": entries[0]["params"],
            "optimizer_updates": entries[0]["optimizer_updates"],
            "pool": entries[0]["pool"],
            "seeded": entries[0]["seeded"],
            "retrieval": aggregate_seeds([e["retrieval"] for e in entries]),
            "scifact": aggregate_seeds([e["scifact"]["student"] for e in entries]),
            "scifact_retention": aggregate_seeds(
                [e["scifact"]["retention"] for e in entries]),
            "latency_ms": {n: aggregate_seeds([e["latency_ms"][n] for e in entries])
                           for n in entries[0]["latency_ms"]},
        }
        if entries[0]["seeded"]:
            agg.update({
                "reground_retrieval": aggregate_seeds(
                    [e["reground"]["retrieval"] for e in entries]),
                "reground_scifact": aggregate_seeds(
                    [e["reground"]["scifact"]["student"] for e in entries]),
                "reground_scifact_retention": aggregate_seeds(
                    [e["reground"]["scifact"]["retention"] for e in entries]),
                "reground_arguana": aggregate_seeds(
                    [e["reground"]["arguana"]["student"] for e in entries]),
                "reground_arguana_retention": aggregate_seeds(
                    [e["reground"]["arguana"]["retention"] for e in entries]),
                # Seed-mean per-query scores, so downstream comparisons against
                # the cell can be bootstrapped rather than eyeballed.
                "per_query_scifact": np.mean(
                    [e["reground"]["scifact"]["per_query"] for e in entries],
                    axis=0).tolist(),
                "per_query_arguana": np.mean(
                    [e["reground"]["arguana"]["per_query"] for e in entries],
                    axis=0).tolist(),
            })
        results["sizes"][label] = agg

    save_results(args.out, results)
    print("done.")


if __name__ == "__main__":
    main()
