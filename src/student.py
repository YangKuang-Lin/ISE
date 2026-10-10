"""Teacher-aligned student baselines, round 2 (review round 5).

Round-4 trained causal/bidirectional students at the cell's budget with zero
teacher access at inference; reviewers correctly noted that comparison is
unmatched at test time and that the 2.7M ceiling is a budget artifact. This
round adds the two controls that decide the question:

  * student + re-grounding -- the bidirectional student embeds between teacher
    re-encodes on the cell's exact cadence (K=8), so student and cell meet at
    identical teacher-call budgets;
  * a scaled student (~13M, 2x epochs, 3x data) probing whether the
    student curve saturates or keeps rising.

Everything is evaluated on a held-out document slice unseen by every variant,
on the same yardsticks as before (partial-query retrieval, SciFact nDCG@10,
STS-B, fidelity by length, streaming latency GPU+CPU). Checkpoints are saved
so downstream scripts can reuse the students.

Usage:
    python student.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim

from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.students import BiStudent, CausalStudent
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)

MIN_TOKENS = 10          # matches DistillConfig.min_tokens
MAX_TOKENS = 80          # matches DistillConfig.max_tokens (training prefixes)
ROLLOUT_K = 64           # evaluation window, matches the cell protocol
NUM_DOCS_BASE = 4000     # the cell's training-doc budget
NUM_DOCS_SCALED = 12000  # 3x data for the scaled student
TRAIN_SPLIT = 0.8
LR = 3e-4

# (label, kind, d_model, layers, ff_dim, num_docs, epochs)
VARIANTS = [
    ("causal_small", "causal", 192, 2, None, NUM_DOCS_BASE, 60),
    ("causal_large", "causal", 320, 3, None, NUM_DOCS_BASE, 60),
    ("bi_small", "bi", 192, 2, None, NUM_DOCS_BASE, 60),
    ("bi_large", "bi", 320, 3, None, NUM_DOCS_BASE, 60),
    ("bi_scaled", "bi", 512, 4, 2048, NUM_DOCS_SCALED, 120),
]
REGROUND_VARIANTS = ["bi_large", "bi_scaled"]   # evaluated with teacher resets
REGROUND_K = 8


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def build_student_data(teacher: Teacher, docs: List[str]) -> List[Dict]:
    """Per-document ids, frozen token rows, and teacher targets per prefix."""
    out = []
    for doc in docs:
        ids = teacher.token_ids(doc, MAX_TOKENS)
        if len(ids) <= MIN_TOKENS + 1:
            continue
        ks = list(range(MIN_TOKENS, len(ids) + 1))
        targets = teacher.encode_prefixes(ids, ks)          # (P, D_t)
        rows = teacher.token_embed(ids)                     # (T, D_t)
        out.append({"ids": ids, "rows": rows, "targets": targets})
    return out


def _pad_batch(items: List[Dict], device: str):
    T = max(len(d["ids"]) for d in items)
    D = items[0]["rows"].shape[1]
    rows = np.zeros((len(items), T, D), dtype=np.float32)
    tgt = np.zeros((len(items), T, D), dtype=np.float32)
    sup = np.zeros((len(items), T), dtype=bool)
    pad = np.ones((len(items), T), dtype=bool)
    for i, d in enumerate(items):
        n = len(d["ids"])
        rows[i, :n] = d["rows"]
        pad[i, :n] = False
        for j, k in enumerate(range(MIN_TOKENS, n + 1)):
            tgt[i, k - 1] = d["targets"][j]
            sup[i, k - 1] = True
    return (torch.tensor(rows, device=device), torch.tensor(tgt, device=device),
            torch.tensor(sup, device=device), torch.tensor(pad, device=device))


def _loss(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    mse = nn.functional.mse_loss(pred, tgt)
    cos = (1 - nn.functional.cosine_similarity(pred, tgt, dim=-1)).mean()
    return mse + 0.1 * cos


def train_causal(data, teacher_dim, d_model, layers, ff_dim, epochs, device):
    model = CausalStudent(teacher_dim, d_model, layers, ff_dim=ff_dim).to(device)
    n_train = int(len(data) * TRAIN_SPLIT)
    train, val = data[:n_train], data[n_train:]
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"  causal d={d_model} L={layers} params={count_params(model):,}")
    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(train))
        for i in range(0, len(order), 64):
            batch = [train[j] for j in order[i:i + 64]]
            rows, tgt, sup, _ = _pad_batch(batch, device)
            loss = _loss(model(rows)[sup], tgt[sup])
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss in causal student.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        cs = []
        with torch.no_grad():
            for i in range(0, len(val), 64):
                rows, tgt, sup, _ = _pad_batch(val[i:i + 64], device)
                cs.append(nn.functional.cosine_similarity(
                    model(rows)[sup], tgt[sup], dim=-1).mean().item())
        v = float(np.mean(cs))
        if v > best:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def train_bi(data, teacher_dim, d_model, layers, ff_dim, epochs, device):
    model = BiStudent(teacher_dim, d_model, layers, ff_dim=ff_dim).to(device)
    n_train = int(len(data) * TRAIN_SPLIT)
    examples, val_examples = [], []
    for di, d in enumerate(data):
        bucket = examples if di < n_train else val_examples
        for j, k in enumerate(range(MIN_TOKENS, len(d["ids"]) + 1)):
            if j % 2 == 0:
                bucket.append((d["rows"][:k], d["targets"][j]))
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    print(f"  bi d={d_model} L={layers} params={count_params(model):,} "
          f"examples={len(examples):,}")

    def batchify(items):
        T = max(len(r) for r, _ in items)
        D = items[0][0].shape[1]
        rows = np.zeros((len(items), T, D), dtype=np.float32)
        pad = np.ones((len(items), T), dtype=bool)
        tgt = np.stack([t for _, t in items])
        for i, (r, _) in enumerate(items):
            rows[i, :len(r)] = r
            pad[i, :len(r)] = False
        return (torch.tensor(rows, device=device), torch.tensor(pad, device=device),
                torch.tensor(tgt, dtype=torch.float32, device=device))

    best, best_state = -1.0, None
    for _ in range(epochs):
        model.train()
        order = np.random.permutation(len(examples))
        for i in range(0, len(order), 256):
            rows, pad, tgt = batchify([examples[j] for j in order[i:i + 256]])
            loss = _loss(model(rows, pad), tgt)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite loss in bi student.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        cs = []
        with torch.no_grad():
            for i in range(0, len(val_examples), 256):
                rows, pad, tgt = batchify(val_examples[i:i + 256])
                cs.append(nn.functional.cosine_similarity(
                    model(rows, pad), tgt, dim=-1).mean().item())
        v = float(np.mean(cs))
        if v > best:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def student_embed(model, teacher: Teacher, ids: List[int], device: str) -> np.ndarray:
    rows = torch.tensor(teacher.token_embed(ids), dtype=torch.float32,
                        device=device).unsqueeze(0)
    if isinstance(model, CausalStudent):
        emb, _ = model.prefill(rows)
        return emb.squeeze(0).cpu().numpy()
    return model(rows).squeeze(0).cpu().numpy()


@torch.no_grad()
def reground_embed(model, teacher: Teacher, ids: List[int], n0: int,
                   device: str, k: int = REGROUND_K) -> Tuple[np.ndarray, int]:
    """Student embedding with periodic teacher re-grounding on the cell's
    cadence: teacher encodes the seed and every k-th position; the student
    re-encodes the prefix at every position in between. Returns the final
    embedding and the teacher-call count (identical to hybrid-k's budget)."""
    if len(ids) <= n0 + 1:
        return teacher.encode_texts([teacher.tokenizer.decode(ids)])[0], 1
    calls, emb = 1, teacher.encode_prefixes(ids, [n0])[0]
    for t in range(n0, len(ids)):
        step = t - n0 + 1
        if step % k == 0:
            emb = teacher.encode_prefixes(ids, [t + 1])[0]
            calls += 1
        else:
            emb = student_embed(model, teacher, ids[:t + 1], device)
    return emb, calls


@torch.no_grad()
def fidelity_by_length(model, teacher, docs, device: str) -> Dict[str, float]:
    checkpoints = [MIN_TOKENS + s for s in (1, 8, 16, 32, 64)]
    per_len: Dict[int, List[float]] = {k: [] for k in checkpoints}
    used = 0
    for doc in docs:
        ids = teacher.token_ids(doc, MAX_TOKENS + ROLLOUT_K + 1)
        if len(ids) < MIN_TOKENS + ROLLOUT_K + 1:
            continue
        true = teacher.encode_prefixes(ids, checkpoints)
        for i, k in enumerate(checkpoints):
            per_len[k].append(_cos(student_embed(model, teacher, ids[:k], device),
                                   true[i]))
        used += 1
        if used >= 100:
            break
    return {str(k): float(np.mean(v)) for k, v in per_len.items()}


@torch.no_grad()
def partial_query_retrieval(model, teacher, docs, device: str,
                            reground: bool = False) -> float:
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
        if reground:
            q, _ = reground_embed(model, teacher, ids[:query_len], MIN_TOKENS, device)
        else:
            q = student_embed(model, teacher, ids[:query_len], device)
        q = q / (np.linalg.norm(q) + 1e-9)
        if int(np.argmax(corpus @ q)) == gold:
            hits += 1
    return hits / len(usable)


@torch.no_grad()
def scifact_eval(model, teacher, device: str, max_queries: int = 300,
                 reground: bool = False) -> Dict[str, float]:
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
        elif reground:
            sq, calls = reground_embed(model, teacher, q_ids, MIN_TOKENS, device)
            callfracs.append(calls / len(q_ids))
        else:
            sq = student_embed(model, teacher, q_ids, device)
        sq = sq / (np.linalg.norm(sq) + 1e-9)
        order = np.argsort(-(corpus_emb @ sq))[:10]
        s_scores.append(ndcg_at_k([ids[i] for i in order], rel[qid], 10))
    t, s = float(np.mean(t_scores)) * 100, float(np.mean(s_scores)) * 100
    out = {"student": s, "teacher": t, "retention": s / t * 100}
    if callfracs:
        out["callfrac"] = float(np.mean(callfracs)) * 100
    return out


@torch.no_grad()
def stsb_eval(model, teacher, device: str) -> Dict[str, float]:
    from datasets import load_dataset
    from scipy.stats import spearmanr

    data = load_dataset("glue", "stsb", split="validation")
    gold, s_sc, t_sc = [], [], []
    for row in data:
        gold.append(row["label"])
        pair = []
        for text in (row["sentence1"], row["sentence2"]):
            q_ids = teacher.token_ids(text, 256)
            pair.append(student_embed(model, teacher, q_ids, device)
                        if q_ids else teacher.encode_texts([text])[0])
        s_sc.append(_cos(pair[0], pair[1]))
        t = teacher.encode_texts([row["sentence1"], row["sentence2"]])
        t_sc.append(_cos(t[0], t[1]))
    return {"student_spearman": float(spearmanr(s_sc, gold).correlation),
            "teacher_spearman": float(spearmanr(t_sc, gold).correlation)}


@torch.no_grad()
def streaming_fidelity(model, teacher, docs, device: str, N: int,
                       n0: int = 10, n_docs: int = 20, repeats: int = 3,
                       reground: bool = False) -> Dict:
    """Wall-clock + streaming fidelity (mean and worst step) at length N."""
    def _sync():
        if device == "cuda":
            torch.cuda.synchronize()

    usable = [teacher.token_ids(d, N) for d in docs]
    usable = [i for i in usable if len(i) >= N][:n_docs]
    times, fids, mins = [], [], []
    for ids in usable:
        ref = teacher.encode_prefixes(ids, list(range(n0, len(ids) + 1)))
        rows_np = teacher.token_embed(ids)
        best = float("inf")
        for _ in range(repeats + 1):
            _sync()
            t0 = time.perf_counter()
            embs = []
            if reground:
                emb = teacher.encode_prefixes(ids, [n0])[0]
                embs.append(emb)
                for t in range(n0, len(ids)):
                    step = t - n0 + 1
                    if step % REGROUND_K == 0:
                        emb = teacher.encode_prefixes(ids, [t + 1])[0]
                    else:
                        emb = student_embed(model, teacher, ids[:t + 1], device)
                    embs.append(emb)
            elif isinstance(model, CausalStudent):
                rows = torch.tensor(rows_np[:n0], dtype=torch.float32,
                                    device=device).unsqueeze(0)
                emb, caches = model.prefill(rows)
                embs.append(emb.squeeze(0).cpu().numpy())
                for t in range(n0, len(ids)):
                    row = torch.tensor(rows_np[t], dtype=torch.float32,
                                       device=device).unsqueeze(0)
                    emb, caches = model.step(row, caches, t)
                    embs.append(emb.squeeze(0).cpu().numpy())
            else:
                for t in range(n0, len(ids) + 1):
                    rows = torch.tensor(rows_np[:t], dtype=torch.float32,
                                        device=device).unsqueeze(0)
                    embs.append(model(rows).squeeze(0).cpu().numpy())
            _sync()
            dt = time.perf_counter() - t0
            if dt < best:
                best, best_embs = dt, embs
        times.append(best)
        m = min(len(ref), len(best_embs))
        step_fids = [_cos(best_embs[i], ref[i]) for i in range(m)]
        fids.append(float(np.mean(step_fids)))
        mins.append(float(np.min(step_fids)))
    return {"length": N, "time_ms": float(np.mean(times)) * 1000,
            "fidelity": float(np.mean(fids)),
            "min_fidelity": float(np.mean(mins)), "n_docs": len(usable)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--ckpt-dir", type=str, default="experiments/results/ckpt")
    ap.add_argument("--out", type=str, default="experiments/results/student2.json")
    args = ap.parse_args()

    device = get_device()
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    docs = load_wikitext_docs("wikitext-2-raw-v1", NUM_DOCS_SCALED + 2000, 40)
    stream_docs = load_wikitext_docs("wikitext-2-raw-v1", 200, 60)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Teacher targets are seed-independent: build the full dataset once and
    # slice per variant. Held-out evaluation uses documents unseen by every
    # variant, including the 12k-doc scaled student.
    print("  building student data (one pass over all documents)...")
    data_all = build_student_data(teacher, docs[:NUM_DOCS_SCALED])
    eval_docs = docs[NUM_DOCS_SCALED:]
    print(f"  data: {len(data_all)} docs total, eval pool {len(eval_docs)}")

    per_variant: Dict[str, List[Dict]] = {}
    for seed in args.seeds:
        for label, kind, d_model, layers, ff_dim, ndocs, epochs in VARIANTS:
            set_seed(seed)
            # Slice the doc budget, then a seed-specific shuffle.
            base = data_all if ndocs >= NUM_DOCS_SCALED else \
                data_all[:int(len(data_all) * ndocs / NUM_DOCS_SCALED)]
            rng = np.random.default_rng(seed)
            data = [base[i] for i in rng.permutation(len(base))]
            train_fn = train_causal if kind == "causal" else train_bi
            model = train_fn(data, teacher.emb_dim, d_model, layers, ff_dim,
                             epochs, device)
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
                "stsb": stsb_eval(model, teacher, device),
                "streaming": [streaming_fidelity(model, teacher, stream_docs,
                                                 device, N) for N in (128, 256)],
            }
            if label in REGROUND_VARIANTS:
                entry["reground"] = {
                    "retrieval": partial_query_retrieval(
                        model, teacher, eval_docs, device, reground=True),
                    "scifact": scifact_eval(model, teacher, device,
                                            reground=True),
                    "streaming": [streaming_fidelity(
                        model, teacher, stream_docs, device, N, reground=True)
                        for N in (128, 256)],
                }
            per_variant.setdefault(label, []).append(entry)
            msg = (f"  [{label} seed={seed}] params={entry['params']:,} "
                   f"retr={entry['retrieval'] * 100:.1f} "
                   f"scifact={entry['scifact']['student']:.1f} "
                   f"({entry['scifact']['retention']:.1f}%)")
            if "reground" in entry:
                rg = entry["reground"]["scifact"]
                msg += (f" | +reground scifact={rg['student']:.1f} "
                        f"({rg['retention']:.1f}% @ {rg.get('callfrac', 0):.1f}% calls)")
            print(msg)
            # Crash insurance: persist partial per-variant results after every
            # model, so an interrupted run can be salvaged (the scaled student
            # alone trains for hours per seed).
            save_results(args.out + ".partial",
                         {"teacher_model": args.teacher,
                          "reground_k": REGROUND_K,
                          "partial": {k: v for k, v in per_variant.items()}})

    results: Dict = {"teacher_model": args.teacher, "seeds": args.seeds,
                     "reground_k": REGROUND_K, "sizes": {}}
    for label, entries in per_variant.items():
        agg = {
            "per_seed": entries,
            "params": entries[0]["params"],
            "retrieval": aggregate_seeds([e["retrieval"] for e in entries]),
            "scifact": aggregate_seeds([e["scifact"]["student"] for e in entries]),
            "scifact_retention": aggregate_seeds(
                [e["scifact"]["retention"] for e in entries]),
            "stsb": aggregate_seeds(
                [e["stsb"]["student_spearman"] for e in entries]),
        }
        if all("reground" in e for e in entries):
            agg["reground_scifact"] = aggregate_seeds(
                [e["reground"]["scifact"]["student"] for e in entries])
            agg["reground_scifact_retention"] = aggregate_seeds(
                [e["reground"]["scifact"]["retention"] for e in entries])
            agg["reground_retrieval"] = aggregate_seeds(
                [e["reground"]["retrieval"] for e in entries])
        results["sizes"][label] = agg
    save_results(args.out, results)
    print("done.")


if __name__ == "__main__":
    main()
