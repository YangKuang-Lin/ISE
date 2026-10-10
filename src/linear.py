"""Linear (ridge) shift predictor baseline (review round 5).

Copy-forward is already at cosine 0.98943, so the entire learned gain is a
small residual. This asks how much of it a closed-form least-squares map
recovers: ridge regression from [E_n; e_{n+1}] (768-d for MiniLM) to the
shift, fit once on the training triples, evaluated under the identical
single-step and full-rollout protocol as the cells. No iterative training,
no nonlinearity, deterministic given the data.

Usage:
    python linear.py
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np
import torch
from torch import nn

from experiments.distill_data import Teacher, build_trajectories, load_wikitext_docs
from experiments.utils import get_device, log_environment, save_results, set_seed

MIN_TOKENS, MAX_TOKENS, ROLLOUT_K = 10, 80, 64
NUM_DOCS, RIDGE_LAMBDA = 4000, 1.0


class LinearCell(nn.Module):
    """Wraps the ridge solution as a (token, state) -> shift module."""

    def __init__(self, W: np.ndarray, b: np.ndarray) -> None:
        super().__init__()
        self.W = torch.tensor(W, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)

    def forward(self, tok: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        x = torch.cat([e, tok], dim=-1)
        return x @ self.W.to(x.device) + self.b.to(x.device)


def _cos(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--out", type=str, default="experiments/results/linear.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    docs = load_wikitext_docs("wikitext-2-raw-v1", NUM_DOCS + 2000, 40)
    trajectories = build_trajectories(teacher, docs[:NUM_DOCS],
                                      MIN_TOKENS, MAX_TOKENS)
    X, Y = [], []
    for embs, toks in trajectories:
        for t in range(len(toks)):
            X.append(np.concatenate([embs[t], toks[t]]))
            Y.append(embs[t + 1] - embs[t])
    X, Y = np.array(X, dtype=np.float64), np.array(Y, dtype=np.float64)
    n_train = int(len(X) * 0.8)
    Xt, Yt = X[:n_train], Y[:n_train]
    mu = Xt.mean(axis=0)
    Xc = Xt - mu
    A = Xc.T @ Xc + RIDGE_LAMBDA * np.eye(Xc.shape[1])
    W = np.linalg.solve(A, Xc.T @ Yt)
    b = Yt.mean(axis=0) - mu @ W
    model = LinearCell(W.astype(np.float32), b.astype(np.float32)).to(device)
    model.eval()
    print(f"  ridge fit on {n_train:,} triples "
          f"({X.shape[1]}->{Y.shape[1]}, lambda={RIDGE_LAMBDA})")

    # Single-step accuracy on held-out triples.
    with torch.no_grad():
        Xv = torch.tensor(X[n_train:], dtype=torch.float32, device=device)
        Yv = torch.tensor(Y[n_train:], dtype=torch.float32, device=device)
        Ev, Tv = Xv[:, :teacher.emb_dim], Xv[:, teacher.emb_dim:]
        pred = model(Tv, Ev)
        cos = nn.functional.cosine_similarity(Ev + pred, Ev + Yv).mean().item()
        zero = nn.functional.cosine_similarity(Ev, Ev + Yv).mean().item()
    err_red = ((1 - zero) - (1 - cos)) / (1 - zero) * 100
    print(f"  single-step cos={cos:.5f}  zero={zero:.5f}  err.red={err_red:.1f}%")

    # Rollout drift + partial-query retrieval, identical protocol.
    eval_docs = docs[NUM_DOCS:]
    per_step: Dict[int, List[float]] = {k: [] for k in range(1, ROLLOUT_K + 1)}
    used = 0
    with torch.no_grad():
        for doc in eval_docs:
            ids = teacher.token_ids(doc, MAX_TOKENS + ROLLOUT_K + 1)
            if len(ids) < MIN_TOKENS + ROLLOUT_K + 1:
                continue
            ks = list(range(MIN_TOKENS, MIN_TOKENS + ROLLOUT_K + 1))
            true = teacher.encode_prefixes(ids, ks)
            rows = teacher.token_embed(ids)
            e = torch.tensor(true[0], dtype=torch.float32, device=device).unsqueeze(0)
            for k in range(1, ROLLOUT_K + 1):
                tok = torch.tensor(rows[MIN_TOKENS + k - 1], dtype=torch.float32,
                                   device=device).unsqueeze(0)
                e = e + model(tok, e)
                per_step[k].append(_cos(e.squeeze(0).cpu().numpy(), true[k]))
            used += 1
            if used >= 100:
                break
    curve = {str(k): float(np.mean(v)) for k, v in per_step.items() if v}
    horizon = next((int(k) for k in sorted(curve, key=int)
                    if curve[k] < 0.90), ROLLOUT_K + 1)

    query_len = MIN_TOKENS + ROLLOUT_K
    usable = []
    for doc in eval_docs:
        ids = teacher.token_ids(doc, 160)
        if len(ids) >= query_len + 1:
            usable.append(ids)
        if len(usable) >= 500:
            break
    corpus = np.stack([teacher.encode_prefixes(ids, [len(ids)], role="passage")[0]
                       for ids in usable])
    corpus = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-9)
    hits = 0
    with torch.no_grad():
        for gold, ids in enumerate(usable):
            e = torch.tensor(teacher.encode_prefixes(ids, [MIN_TOKENS])[0],
                             dtype=torch.float32, device=device).unsqueeze(0)
            rows = teacher.token_embed(ids)
            for k in range(ROLLOUT_K):
                tok = torch.tensor(rows[MIN_TOKENS + k], dtype=torch.float32,
                                   device=device).unsqueeze(0)
                e = e + model(tok, e)
            q = e.squeeze(0).cpu().numpy()
            q = q / (np.linalg.norm(q) + 1e-9)
            if int(np.argmax(corpus @ q)) == gold:
                hits += 1
    retrieval = hits / len(usable)
    print(f"  horizon={horizon}  cos@64={curve[str(ROLLOUT_K)]:.3f}  "
          f"retr={retrieval * 100:.1f}")

    # Zero-parameter trajectory baselines (review round 6): hold the seed
    # (copy-forward null, also plotted in Figure 1) and momentum
    # extrapolation e_{t+1} = e_t + beta (e_t - e_{t-1}), the cheapest
    # trajectory-aware update. If the cell only extrapolated inertia, momentum
    # would match it.
    def null_curve(beta: float) -> Dict[str, float]:
        # Two-point initialization (teacher at n0-1 and n0, i.e. one extra
        # seed call): with e_prev == e the velocity term is identically zero
        # and "momentum" degenerates to "hold".
        per: Dict[int, List[float]] = {k: [] for k in range(1, ROLLOUT_K + 1)}
        used = 0
        for doc in eval_docs:
            ids = teacher.token_ids(doc, MAX_TOKENS + ROLLOUT_K + 1)
            if len(ids) < MIN_TOKENS + ROLLOUT_K + 1:
                continue
            ks = list(range(MIN_TOKENS, MIN_TOKENS + ROLLOUT_K + 1))
            true = teacher.encode_prefixes(ids, ks)
            e_prev = teacher.encode_prefixes(ids, [MIN_TOKENS - 1])[0]
            e = true[0]
            for k in range(1, ROLLOUT_K + 1):
                e, e_prev = e + beta * (e - e_prev), e
                per[k].append(_cos(e, true[k]))
            used += 1
            if used >= 100:
                break
        return {str(k): float(np.mean(v)) for k, v in per.items() if v}

    def null_retrieval(beta: float) -> float:
        hits = 0
        for gold, ids in enumerate(usable):
            e_prev = teacher.encode_prefixes(ids, [MIN_TOKENS - 1])[0]
            e = teacher.encode_prefixes(ids, [MIN_TOKENS])[0]
            for _ in range(ROLLOUT_K):
                e, e_prev = e + beta * (e - e_prev), e
            q = e / (np.linalg.norm(e) + 1e-9)
            if int(np.argmax(corpus @ q)) == gold:
                hits += 1
        return hits / len(usable)

    hold_curve = null_curve(0.0)
    momentum = {}
    for beta in (0.25, 0.5, 0.9):
        c = null_curve(beta)
        momentum[str(beta)] = {
            "cos64": c[str(ROLLOUT_K)],
            "horizon": next((int(k) for k in sorted(c, key=int)
                             if c[k] < 0.90), ROLLOUT_K + 1),
            "rollout_drift": c,
        }
        print(f"  momentum beta={beta}: cos@64={c[str(ROLLOUT_K)]:.3f} "
              f"horizon={momentum[str(beta)]['horizon']}")
    best_beta = max(momentum, key=lambda b: momentum[b]["cos64"])
    momentum[best_beta]["retrieval"] = null_retrieval(float(best_beta))
    print(f"  best beta={best_beta}: hold cos@64={hold_curve[str(ROLLOUT_K)]:.3f}"
          f"  momentum retr={momentum[best_beta]['retrieval'] * 100:.1f}")

    save_results(args.out, {
        "params": int(W.size + b.size), "ridge_lambda": RIDGE_LAMBDA,
        "single_step_cos": cos, "zero_delta_cos": zero, "err_red": err_red,
        "rollout_drift": curve, "horizon": horizon,
        "cos64": curve[str(ROLLOUT_K)], "retrieval": retrieval,
        "hold_curve": hold_curve, "momentum": momentum,
        "momentum_best_beta": best_beta, "momentum_init": "two_point"})


if __name__ == "__main__":
    main()
