"""Seeded recurrent (GRU) student baseline (review round 7).

The seeded bidirectional students re-encode the prefix at every token, so
their cost grows with length; the cell updates in O(1). Reviewers asked for
the missing O(1) competitor occupying exactly the cell's niche: a
teacher-aligned recurrent student whose hidden state is initialized from the
most recent teacher embedding (seed, or last re-grounding) and advanced one
token row at a time, reading out into the teacher's space after every step.

Two variants: ``gru_seeded`` is parameter-matched to the 0.96M cell, and
``gru_scaled_seeded`` is ~4x larger. Training tuples mirror the seeded-bi
protocol (anchor sampled uniformly over supervision positions), and the
evaluation yardsticks are identical to student_seeded.py.

Usage:
    python student_rnn.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim

from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.students import SeededGRUStudent
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

# (label, d_model, layers, num_docs, epochs)
VARIANTS = [
    ("gru_seeded", 320, 1, NUM_DOCS_BASE, 60),
    ("gru_scaled_seeded", 512, 2, NUM_DOCS_SCALED, 60),
]


def train_gru(data: List[Dict], teacher_dim: int, d_model: int, layers: int,
              epochs: int, device: str, rng: np.random.Generator
              ) -> SeededGRUStudent:
    model = SeededGRUStudent(teacher_dim, d_model, layers).to(device)
    n_train = int(len(data) * TRAIN_SPLIT)
    examples, val_examples = [], []
    for di, d in enumerate(data):
        bucket = examples if di < n_train else val_examples
        for j, k in enumerate(range(MIN_TOKENS, len(d["ids"]) + 1)):
            if j % 2 == 0:
                a = int(rng.integers(0, j + 1))
                # Post-anchor token rows: the anchor prefix is ids[:MIN+a],
                # the target prefix ids[:MIN+j]; the GRU consumes the rows in
                # between with its hidden state initialized from the anchor.
                bucket.append((d["rows"][MIN_TOKENS + a: k],
                               d["targets"][a], d["targets"][j]))
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"  seeded-gru d={d_model} L={layers} params={count_params(model):,} "
          f"examples={len(examples):,}")

    def batchify(items):
        T = max(len(r) for r, _, _ in items)
        D = items[0][1].shape[0]
        rows = np.zeros((len(items), max(T, 1), D), dtype=np.float32)
        lengths = np.zeros(len(items), dtype=np.int64)
        seed = np.stack([s for _, s, _ in items]).astype(np.float32)
        tgt = np.stack([t for _, _, t in items]).astype(np.float32)
        for i, (r, _, _) in enumerate(items):
            rows[i, :len(r)] = r
            lengths[i] = len(r)
        return (torch.tensor(rows, device=device),
                torch.tensor(lengths, device=device),
                torch.tensor(seed, device=device),
                torch.tensor(tgt, device=device))

    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(examples))
        for i in range(0, len(order), 256):
            rows, lengths, seed, tgt = batchify([examples[j] for j in order[i:i + 256]])
            loss = _loss(model(rows, lengths, seed=seed), tgt)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss in seeded GRU student.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        cs = []
        with torch.no_grad():
            for i in range(0, len(val_examples), 256):
                rows, lengths, seed, tgt = batchify(val_examples[i:i + 256])
                cs.append(nn.functional.cosine_similarity(
                    model(rows, lengths, seed=seed), tgt, dim=-1).mean().item())
        v = float(np.mean(cs))
        if v > best:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def gru_rollout(model: SeededGRUStudent, teacher: Teacher, ids: List[int],
                n0: int, device: str, k: int = 0,
                checkpoints: List[int] | None = None
                ) -> Tuple[np.ndarray, int, Dict[int, np.ndarray]]:
    """Stream ids past an n0 seed with O(1) updates; optional re-ground every k.

    Returns (final embedding, teacher calls, {prefix_len: embedding} at the
    requested checkpoints). At a re-ground step the state re-initializes from
    the fresh teacher embedding, which is also the emitted embedding (exact by
    construction), matching the cell's protocol.
    """
    want = set(checkpoints or [])
    grabbed: Dict[int, np.ndarray] = {}
    if len(ids) <= n0:
        e = teacher.encode_prefixes(ids, [len(ids)])[0]
        return e, 1, {len(ids): e}
    anchor = teacher.encode_prefixes(ids, [n0])[0]
    calls = 1
    emb = anchor
    h = model.init_state(torch.tensor(anchor, dtype=torch.float32,
                                      device=device).unsqueeze(0))
    if n0 in want:
        grabbed[n0] = emb
    rows = teacher.token_embed(ids)
    for t in range(n0, len(ids)):
        step = t - n0 + 1
        if k and step % k == 0:
            anchor = teacher.encode_prefixes(ids, [t + 1])[0]
            calls += 1
            emb = anchor
            h = model.init_state(torch.tensor(anchor, dtype=torch.float32,
                                              device=device).unsqueeze(0))
        else:
            row = torch.tensor(rows[t], dtype=torch.float32,
                               device=device).unsqueeze(0)
            out, h = model.step(row, h)
            emb = out.squeeze(0).cpu().numpy()
        if t + 1 in want:
            grabbed[t + 1] = emb
    return emb, calls, grabbed


@torch.no_grad()
def fidelity_by_length(model, teacher, docs, device: str) -> Dict[str, float]:
    checkpoints = [MIN_TOKENS + s for s in (1, 8, 16, 32, 64)]
    per_len: Dict[int, List[float]] = {c: [] for c in checkpoints}
    used = 0
    for doc in docs:
        ids = teacher.token_ids(doc, MAX_TOKENS + ROLLOUT_K + 1)
        if len(ids) < MIN_TOKENS + ROLLOUT_K + 1:
            continue
        true = teacher.encode_prefixes(ids, checkpoints)
        _, _, grabbed = gru_rollout(model, teacher, ids[:checkpoints[-1]],
                                    MIN_TOKENS, device, checkpoints=checkpoints)
        for i, c in enumerate(checkpoints):
            per_len[c].append(_cos(grabbed[c], true[i]))
        used += 1
        if used >= 100:
            break
    return {str(c): float(np.mean(v)) for c, v in per_len.items()}


@torch.no_grad()
def partial_query_retrieval(model, teacher, docs, device: str,
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
        q, _, _ = gru_rollout(model, teacher, ids[:query_len], MIN_TOKENS,
                              device, k=k)
        q = q / (np.linalg.norm(q) + 1e-9)
        if int(np.argmax(corpus @ q)) == gold:
            hits += 1
    return hits / len(usable)


@torch.no_grad()
def scifact_eval(model, teacher, device: str, max_queries: int = 300,
                 k: int = 0) -> Dict[str, float]:
    from beir import load_beir, ndcg_at_k

    ids, texts, queries, rel = load_beir("scifact", 60000)
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
            sq, calls, _ = gru_rollout(model, teacher, q_ids, MIN_TOKENS,
                                       device, k=k)
            callfracs.append(calls / len(q_ids))
        sq = sq / (np.linalg.norm(sq) + 1e-9)
        order = np.argsort(-(corpus_emb @ sq))[:10]
        s_scores.append(ndcg_at_k([ids[i] for i in order], rel[qid], 10))
    t, s = float(np.mean(t_scores)) * 100, float(np.mean(s_scores)) * 100
    return {"student": s, "teacher": t, "retention": s / t * 100,
            "callfrac": float(np.mean(callfracs)) * 100 if callfracs else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--ckpt-dir", type=str, default="experiments/results/ckpt")
    ap.add_argument("--out", type=str,
                    default="experiments/results/student_rnn.json")
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
        for label, d_model, layers, ndocs, epochs in VARIANTS:
            set_seed(seed)
            base = data_all if ndocs >= NUM_DOCS_SCALED else \
                data_all[:int(len(data_all) * ndocs / NUM_DOCS_SCALED)]
            rng = np.random.default_rng(seed)
            data = [base[i] for i in rng.permutation(len(base))]
            model = train_gru(data, teacher.emb_dim, d_model, layers,
                              epochs, device, rng)
            model.eval()
            torch.save(model.state_dict(),
                       ckpt_dir / f"student_{label}_seed{seed}.pt")
            entry = {
                "seed": seed,
                "params": count_params(model),
                "fidelity_by_length": fidelity_by_length(model, teacher,
                                                         eval_docs, device),
                "retrieval": partial_query_retrieval(model, teacher,
                                                     eval_docs, device),
                "scifact": scifact_eval(model, teacher, device),
                "reground": {
                    "retrieval": partial_query_retrieval(
                        model, teacher, eval_docs, device, k=REGROUND_K),
                    "scifact": scifact_eval(model, teacher, device,
                                            k=REGROUND_K),
                },
            }
            per_variant.setdefault(label, []).append(entry)
            rg = entry["reground"]["scifact"]
            print(f"  [{label} seed={seed}] params={entry['params']:,} "
                  f"retr={entry['retrieval'] * 100:.1f} "
                  f"scifact={entry['scifact']['student']:.1f} "
                  f"({entry['scifact']['retention']:.1f}%) | "
                  f"+reground scifact={rg['student']:.1f} "
                  f"({rg['retention']:.1f}% @ {rg['callfrac']:.1f}% calls)")
            save_results(args.out + ".partial",
                         {"teacher_model": args.teacher,
                          "reground_k": REGROUND_K,
                          "partial": {k: v for k, v in per_variant.items()}})

    results: Dict = {"teacher_model": args.teacher, "seeds": args.seeds,
                     "reground_k": REGROUND_K, "sizes": {}}
    for label, entries in per_variant.items():
        results["sizes"][label] = {
            "per_seed": entries,
            "params": entries[0]["params"],
            "retrieval": aggregate_seeds([e["retrieval"] for e in entries]),
            "scifact": aggregate_seeds([e["scifact"]["student"] for e in entries]),
            "scifact_retention": aggregate_seeds(
                [e["scifact"]["retention"] for e in entries]),
            "reground_scifact": aggregate_seeds(
                [e["reground"]["scifact"]["student"] for e in entries]),
            "reground_scifact_retention": aggregate_seeds(
                [e["reground"]["scifact"]["retention"] for e in entries]),
            "reground_retrieval": aggregate_seeds(
                [e["reground"]["retrieval"] for e in entries]),
        }
    save_results(args.out, results)
    print("done.")


if __name__ == "__main__":
    main()
