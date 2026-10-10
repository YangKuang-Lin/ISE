"""Distillation track: learn a frozen sentence encoder as an incremental updater.

This is the paper's core contribution -- "embedding-shift distillation". A cell
learns ΔE = f(x_{n+1}, E_n) so that E_{n+1} ~= E_n + ΔE, turning a frozen
teacher (all-MiniLM-L6-v2) into an O(1)-per-token streaming encoder.

Produces every number the distillation results section needs:
  * single-step accuracy vs the zero-delta baseline,
  * multi-step rollout drift,
  * document retrieval accuracy vs the teacher,
  * STS-B correlation (standard sentence-embedding yardstick),
  * the re-encode-every-K compute/quality Pareto sweep,
  * ablations over cell type, loss, and state initialization.

Usage:
    python distill.py --seeds 41 42 43 --cell gated_film
    python distill.py --ablate-cells --ablate-loss
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from experiments.cells import DEFAULT_NUM_LAYERS, build_cell, default_hidden
from experiments.distill_data import (
    Teacher,
    build_trajectories,
    build_triples,
    load_wikitext_docs,
)
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)


@dataclass(frozen=True)
class DistillConfig:
    teacher: str = "all-MiniLM-L6-v2"
    dataset_config: str = "wikitext-2-raw-v1"
    num_docs: int = 4000
    min_tokens: int = 10
    max_tokens: int = 80
    min_words: int = 40
    train_split: float = 0.8
    batch_size: int = 1024
    lr: float = 3e-4
    epochs: int = 60
    # Deployed cell geometry for the default 384-d teacher (0.96M parameters);
    # see experiments/cells.default_hidden. The superseded default was
    # (384, 4) = 4.29M -- leaving it here silently gave any caller that built a
    # bare DistillConfig() a 4.5x larger cell than the paper's.
    hidden_dim: int = 192
    num_layers: int = 3
    cell: str = "gated_film"
    mse_weight: float = 1.0
    cos_weight: float = 0.1
    # Scale-invariant MSE: computed between L2-normalized updated embeddings, so
    # it penalizes direction without being dominated by embedding magnitude.
    nmse_weight: float = 0.0
    # L2-normalize the state after every update. The teacher's embeddings are
    # normalized, so an un-normalized state can drift off the unit sphere; this
    # tests whether that is a contributing cause of rollout drift.
    normalize_state: bool = False
    rollout_k: int = 64
    # 0 == never re-encode. Deliberately not all divisors of rollout_k.
    reencode_budgets: tuple = (0, 8, 12, 16, 24, 32)
    retrieval_corpus: int = 500
    retrieval_queries: int = 500
    corpus_tokens: int = 160       # corpus entries are longer than the query prefix


class TripleDataset(Dataset):
    def __init__(self, triples):
        self.E = torch.tensor(np.array([t[0] for t in triples]), dtype=torch.float32)
        self.tok = torch.tensor(np.array([t[1] for t in triples]), dtype=torch.float32)
        self.delta = torch.tensor(np.array([t[2] for t in triples]), dtype=torch.float32)

    def __len__(self):
        return len(self.E)

    def __getitem__(self, i):
        return self.E[i], self.tok[i], self.delta[i]


def maybe_norm(e: torch.Tensor, cfg: "DistillConfig") -> torch.Tensor:
    """L2-normalize the state when ``cfg.normalize_state`` is set."""
    return nn.functional.normalize(e, dim=-1) if cfg.normalize_state else e


def combined_loss(pred, target, e_n, w_mse, w_cos, w_nmse: float = 0.0):
    """MSE on the raw shift + cosine on the updated embedding + optional
    scale-invariant MSE on the L2-normalized updated embedding."""
    loss = w_mse * nn.functional.mse_loss(pred, target)
    if w_cos:
        loss = loss + w_cos * (
            1 - nn.functional.cosine_similarity(e_n + pred, e_n + target)).mean()
    if w_nmse:
        pn = nn.functional.normalize(e_n + pred, dim=-1)
        tn = nn.functional.normalize(e_n + target, dim=-1)
        loss = loss + w_nmse * nn.functional.mse_loss(pn, tn)
    return loss


class TrajectoryDataset(Dataset):
    """Fixed-length windows sampled from per-document trajectories."""

    def __init__(self, trajectories, unroll: int):
        self.E, self.T = [], []
        for embs, toks in trajectories:
            n = len(toks)
            for s in range(0, max(1, n - unroll + 1)):
                if s + unroll > n:
                    break
                self.E.append(embs[s:s + unroll + 1])   # (U+1, D)
                self.T.append(toks[s:s + unroll])       # (U, D)
        self.E = torch.tensor(np.array(self.E), dtype=torch.float32)
        self.T = torch.tensor(np.array(self.T), dtype=torch.float32)

    def __len__(self):
        return len(self.E)

    def __getitem__(self, i):
        return self.E[i], self.T[i]


def train_cell_multistep(cfg: DistillConfig, trajectories, device, unroll: int) -> nn.Module:
    """Train with an unrolled multi-step loss to directly penalize drift.

    Single-step training optimizes each shift in isolation, so compounding error
    is never observed at training time. Here the state is rolled forward
    ``unroll`` steps from its own predictions (no teacher forcing) and every
    intermediate state is supervised against the teacher trajectory.
    """
    ds = TrajectoryDataset(trajectories, unroll)
    n_train = int(len(ds) * cfg.train_split)
    idx = torch.randperm(len(ds))
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    emb_dim = ds.E.shape[-1]
    model = build_cell(cfg.cell, emb_dim, cfg.hidden_dim, cfg.num_layers).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    print(f"  multistep unroll={unroll} windows={len(ds):,} params={count_params(model):,}")

    def run_batch(E, T):
        # E: (B, U+1, D), T: (B, U, D). Roll forward from the model's own state.
        e = E[:, 0]
        loss = 0.0
        for t in range(T.size(1)):
            e = maybe_norm(e + model(T[:, t], e), cfg)
            tgt = E[:, t + 1]
            loss = loss + cfg.mse_weight * nn.functional.mse_loss(e, tgt) \
                        + cfg.cos_weight * (1 - nn.functional.cosine_similarity(e, tgt)).mean()
        return loss / T.size(1)

    best, best_state = float("inf"), None
    for _ in range(cfg.epochs):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            E, T = ds.E[b].to(device), ds.T[b].to(device)
            loss = run_batch(E, T)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite loss in multi-step training (unroll={unroll}); "
                    f"lower --lr or reduce --epochs.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = []
            for i in range(0, len(val_idx), cfg.batch_size):
                b = val_idx[i:i + cfg.batch_size]
                vl.append(run_batch(ds.E[b].to(device), ds.T[b].to(device)).item())
        v = float(np.mean(vl))
        if not np.isfinite(v):
            raise FloatingPointError("Non-finite validation loss in multi-step training.")
        if v < best:
            best, best_state = v, {k: t.detach().clone() for k, t in model.state_dict().items()}
    if best_state is None:
        raise FloatingPointError(
            "Multi-step training produced no finite validation score.")
    model.load_state_dict(best_state)
    return model


def train_cell(cfg: DistillConfig, triples, device) -> nn.Module:
    n_train = int(len(triples) * cfg.train_split)
    train_ds, val_ds = TripleDataset(triples[:n_train]), TripleDataset(triples[n_train:])
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)

    emb_dim = triples[0][0].shape[0]
    model = build_cell(cfg.cell, emb_dim, cfg.hidden_dim, cfg.num_layers).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)
    print(f"  cell={cfg.cell} params={count_params(model):,}")

    best_cos, best_state = -1.0, None
    for epoch in range(cfg.epochs):
        model.train()
        for e_n, tok, delta in train_loader:
            e_n, tok, delta = e_n.to(device), tok.to(device), delta.to(device)
            loss = combined_loss(model(tok, e_n), delta, e_n,
                                 cfg.mse_weight, cfg.cos_weight, cfg.nmse_weight)
            # Fail loudly on divergence. Without this the run continues, every
            # validation score is NaN, ``best_state`` is never set (NaN > -1.0
            # is False), and the corrupted model surfaces much later inside an
            # unrelated sklearn call.
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite training loss at epoch {epoch + 1}/{cfg.epochs} "
                    f"(cell={cfg.cell}, teacher={cfg.teacher}, lr={cfg.lr}, "
                    f"hidden={cfg.hidden_dim}). Training diverged; lower --lr or "
                    f"reduce --epochs.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        cs = []
        with torch.no_grad():
            for e_n, tok, delta in val_loader:
                e_n, tok, delta = e_n.to(device), tok.to(device), delta.to(device)
                pred_e = e_n + model(tok, e_n)
                cs.append(nn.functional.cosine_similarity(pred_e, e_n + delta).mean().item())
        vc = float(np.mean(cs))
        if not np.isfinite(vc):
            raise FloatingPointError(
                f"Non-finite validation cosine at epoch {epoch + 1}/{cfg.epochs}.")
        if vc > best_cos:
            best_cos, best_state = vc, {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        raise FloatingPointError(
            "No epoch produced a finite validation score; refusing to return an "
            "untrained model.")
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def single_step(model, triples, device) -> Dict[str, float]:
    ds = TripleDataset(triples)
    loader = DataLoader(ds, batch_size=1024)
    cos, base = [], []
    for e_n, tok, delta in loader:
        e_n, tok, delta = e_n.to(device), tok.to(device), delta.to(device)
        pred_e, true_e = e_n + model(tok, e_n), e_n + delta
        cos.append(nn.functional.cosine_similarity(pred_e, true_e).mean().item())
        base.append(nn.functional.cosine_similarity(e_n, true_e).mean().item())
    return {"ise_cos": float(np.mean(cos)), "zero_delta_cos": float(np.mean(base))}


@torch.no_grad()
def rollout_embed(model, teacher: Teacher, ids: List[int], n0: int, steps: int,
                  reencode_every: int, device: str, normalize: bool = False
                  ) -> Tuple[np.ndarray, np.ndarray, int]:
    """Roll the state forward ``steps`` tokens, re-encoding every ``reencode_every``
    steps (0 == never). Returns (pred_states, true_states, teacher_calls)."""
    ks = list(range(n0, n0 + steps + 1))
    true = teacher.encode_prefixes(ids, ks)                 # (steps+1, D)
    e = torch.tensor(true[0], dtype=torch.float32, device=device).unsqueeze(0)
    preds, calls = [true[0]], 1
    for k in range(1, steps + 1):
        tok = torch.tensor(teacher.token_embed([ids[n0 + k - 1]])[0],
                           dtype=torch.float32, device=device).unsqueeze(0)
        e = e + model(tok, e)
        if not torch.isfinite(e).all():
            raise FloatingPointError(
                f"Non-finite state at rollout step {k}/{steps} "
                f"(norm before failure diverged); the cell is unstable.")
        if normalize:
            e = nn.functional.normalize(e, dim=-1)
        if reencode_every and k % reencode_every == 0:
            e = torch.tensor(true[k], dtype=torch.float32, device=device).unsqueeze(0)
            calls += 1
        preds.append(e.squeeze(0).cpu().numpy())
    return np.array(preds), true, calls


def rollout_drift(model, teacher, docs, cfg, device) -> Dict[int, float]:
    """Mean cosine(pred, true) per rollout step (no re-encoding)."""
    per_step: Dict[int, List[float]] = {k: [] for k in range(1, cfg.rollout_k + 1)}
    for doc in docs:
        ids = teacher.token_ids(doc, cfg.max_tokens + cfg.rollout_k + 1)
        if len(ids) < cfg.min_tokens + cfg.rollout_k + 1:
            continue
        preds, true, _ = rollout_embed(model, teacher, ids, cfg.min_tokens,
                                       cfg.rollout_k, 0, device, cfg.normalize_state)
        for k in range(1, cfg.rollout_k + 1):
            d = np.dot(preds[k], true[k]) / (np.linalg.norm(preds[k]) * np.linalg.norm(true[k]) + 1e-9)
            per_step[k].append(float(d))
    return {k: float(np.mean(v)) for k, v in per_step.items() if v}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def pareto_sweep(model, teacher, docs, cfg, device) -> List[Dict]:
    """Quality vs teacher-call fraction for each re-encode budget.

    Reports the *mean* cosine over the whole trajectory (the quality a streaming
    consumer actually sees) and the mean *pre-reset* cosine (the worst case just
    before each re-encode). Measuring only the final step would be degenerate:
    when ``rollout_k % budget == 0`` the last step is always a teacher reset, so
    the final cosine is 1.0 by construction.
    """
    rows = []
    for budget in cfg.reencode_budgets:
        traj, worst, call_frac = [], [], []
        for doc in docs:
            ids = teacher.token_ids(doc, cfg.max_tokens + cfg.rollout_k + 1)
            if len(ids) < cfg.min_tokens + cfg.rollout_k + 1:
                continue
            preds, true, calls = rollout_embed(model, teacher, ids, cfg.min_tokens,
                                               cfg.rollout_k, budget, device,
                                               cfg.normalize_state)
            # Steps that are teacher resets are excluded: they are exact by
            # construction and would inflate the mean.
            reset = {k for k in range(1, cfg.rollout_k + 1) if budget and k % budget == 0}
            step_cos = [_cos(preds[k], true[k]) for k in range(1, cfg.rollout_k + 1)
                        if k not in reset]
            pre_reset = [_cos(preds[k], true[k]) for k in range(1, cfg.rollout_k + 1)
                         if budget and (k + 1) % budget == 0] or step_cos
            if not step_cos:
                continue
            traj.append(float(np.mean(step_cos)))
            worst.append(float(np.mean(pre_reset)))
            call_frac.append(calls / (cfg.rollout_k + 1))
        rows.append({"reencode_every": budget,
                     "mean_cos": float(np.mean(traj)),
                     "pre_reset_cos": float(np.mean(worst)),
                     "teacher_call_fraction": float(np.mean(call_frac))})
    return rows


@torch.no_grad()
def retrieval(model, teacher, docs, cfg, device) -> Dict[str, float]:
    """Partial-query document retrieval (self-retrieval).

    Each corpus entry is a document's full text (``corpus_tokens`` tokens). Each
    query is a *prefix* of the same document (``min_tokens + rollout_k`` tokens)
    embedded incrementally. Gold is the document's own index, so the task is
    well-posed and the teacher's own prefix embedding is a non-trivial upper
    bound. (An earlier version defined gold as the argmax of teacher similarity,
    which made the teacher trivially 100% correct.)
    """
    from sklearn.metrics.pairwise import cosine_similarity

    query_len = cfg.min_tokens + cfg.rollout_k
    usable = []
    for doc in docs:
        ids = teacher.token_ids(doc, cfg.corpus_tokens)
        if len(ids) >= query_len + 1:
            usable.append(ids)
        if len(usable) >= cfg.retrieval_corpus:
            break
    if not usable:
        return {"n": 0}

    corpus_emb = np.stack([teacher.encode_prefixes(ids, [len(ids)], role="passage")[0]
                           for ids in usable])

    hits_ise = hits_prefix = hits_seed = n = 0
    for gold, ids in enumerate(usable[:cfg.retrieval_queries]):
        # Only the final state is scored, so skip encoding the ground-truth
        # trajectory: one teacher call for the seed instead of rollout_k+1.
        e_final = rollout_predict(model, teacher, ids, cfg.min_tokens,
                                  cfg.rollout_k, device, cfg.normalize_state)
        e_prefix = teacher.encode_prefixes(ids, [query_len])[0]   # upper bound
        e_seed = teacher.encode_prefixes(ids, [cfg.min_tokens])[0]  # lower bound
        if np.argmax(cosine_similarity([e_final], corpus_emb)[0]) == gold:
            hits_ise += 1
        if np.argmax(cosine_similarity([e_prefix], corpus_emb)[0]) == gold:
            hits_prefix += 1
        if np.argmax(cosine_similarity([e_seed], corpus_emb)[0]) == gold:
            hits_seed += 1
        n += 1
    return {"ise": hits_ise / max(n, 1),
            "teacher_prefix": hits_prefix / max(n, 1),
            "seed_only": hits_seed / max(n, 1), "n": n}


@torch.no_grad()
def rollout_predict(model, teacher: Teacher, ids: List[int], n0: int, steps: int,
                    device: str, normalize: bool = False) -> np.ndarray:
    """Roll the state forward and return only the final prediction.

    Unlike :func:`rollout_embed` this encodes just the seed prefix instead of the
    whole ground-truth trajectory, which is all that is needed when no fidelity
    metric is being computed (e.g. STS). That is one teacher call instead of
    ``steps+1``.
    """
    e = torch.tensor(teacher.encode_prefixes(ids, [n0])[0],
                     dtype=torch.float32, device=device).unsqueeze(0)
    tok_table = teacher.token_embed(ids)
    for k in range(steps):
        tok = torch.tensor(tok_table[n0 + k], dtype=torch.float32, device=device).unsqueeze(0)
        e = e + model(tok, e)
        if not torch.isfinite(e).all():
            raise FloatingPointError(
                f"Non-finite state at rollout step {k}/{steps}; the cell is unstable.")
        if normalize:
            e = nn.functional.normalize(e, dim=-1)
    return e.squeeze(0).cpu().numpy()


@torch.no_grad()
def sts_eval(model, teacher, cfg, device) -> Dict[str, float]:
    """Spearman on STS-B: incremental ISE embedding vs full teacher re-encode."""
    from datasets import load_dataset
    from scipy.stats import spearmanr

    data = load_dataset("glue", "stsb", split="validation")

    def ise_embed(text: str) -> np.ndarray:
        ids = teacher.token_ids(text, cfg.max_tokens)
        if len(ids) <= cfg.min_tokens + 1:
            return teacher.encode_texts([text])[0]
        steps = len(ids) - cfg.min_tokens
        return rollout_predict(model, teacher, ids, cfg.min_tokens, steps, device,
                               cfg.normalize_state)

    gold, ise_scores, teacher_scores = [], [], []
    for row in data:
        s1, s2 = row["sentence1"], row["sentence2"]
        gold.append(row["label"])
        e1, e2 = ise_embed(s1), ise_embed(s2)
        ise_scores.append(float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-9)))
        t = teacher.encode_texts([s1, s2])
        teacher_scores.append(float(np.dot(t[0], t[1]) / (np.linalg.norm(t[0]) * np.linalg.norm(t[1]) + 1e-9)))
    return {"ise_spearman": float(spearmanr(ise_scores, gold).correlation),
            "teacher_spearman": float(spearmanr(teacher_scores, gold).correlation)}


@torch.no_grad()
def adaptive_sweep(model, teacher, docs, cfg, device, taus: List[float],
                   min_k: int = 0) -> List[Dict]:
    """Content-adaptive re-encoding: reset when accumulated drift risk exceeds tau.

    Fixed-K resets ignore the input: some prefixes drift fast, others barely at
    all. We use the accumulated update path length since the last reset,
    ``s = sum_t ||delta_t||``, as an online, teacher-free drift proxy and reset
    when ``s > tau``. Comparing this curve against the fixed-K curve tests
    whether adaptivity buys accuracy at equal teacher cost.
    """
    rows = []
    for tau in taus:
        cos_all, calls_all, corr = [], [], []
        for doc in docs:
            ids = teacher.token_ids(doc, cfg.max_tokens + cfg.rollout_k + 1)
            if len(ids) < cfg.min_tokens + cfg.rollout_k + 1:
                continue
            ks = list(range(cfg.min_tokens, cfg.min_tokens + cfg.rollout_k + 1))
            true = teacher.encode_prefixes(ids, ks)
            e = torch.tensor(true[0], dtype=torch.float32, device=device).unsqueeze(0)
            path, calls, step_cos, since = 0.0, 1, [], 0
            for t in range(1, cfg.rollout_k + 1):
                tok = torch.tensor(teacher.token_embed([ids[cfg.min_tokens + t - 1]])[0],
                                   dtype=torch.float32, device=device).unsqueeze(0)
                delta = model(tok, e)
                e = maybe_norm(e + delta, cfg)
                path += float(delta.norm().item())
                since += 1
                cur = e.squeeze(0).cpu().numpy()
                c = _cos(cur, true[t])
                step_cos.append(c)
                corr.append((path, c))
                # A minimum spacing prevents bursts of resets on volatile spans.
                if path > tau and since >= min_k:
                    e = torch.tensor(true[t], dtype=torch.float32, device=device).unsqueeze(0)
                    path, calls, since = 0.0, calls + 1, 0
            cos_all.append(float(np.mean(step_cos)))
            calls_all.append(calls / (cfg.rollout_k + 1))
        # How well does the proxy track true drift? (Spearman, lower = better proxy)
        from scipy.stats import spearmanr
        rho = float(spearmanr([c[0] for c in corr], [c[1] for c in corr]).correlation)
        rows.append({"tau": tau, "min_k": min_k, "mean_cos": float(np.mean(cos_all)),
                     "teacher_call_fraction": float(np.mean(calls_all)),
                     "path_vs_drift_spearman": rho})
    return rows


def _triples_from_trajectories(trajectories) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Derive single-step triples from ordered trajectories (avoids re-encoding)."""
    out = []
    for embs, toks in trajectories:
        for t in range(len(toks)):
            out.append((embs[t], toks[t], embs[t + 1] - embs[t]))
    return out


def run_once(cfg: DistillConfig, seed: int, device: str, teacher: Teacher,
             docs: List[str], unroll: int = 1, full_eval: bool = True) -> Dict:
    set_seed(seed)
    trajectories = build_trajectories(teacher, docs[:cfg.num_docs],
                                      cfg.min_tokens, cfg.max_tokens)
    triples = _triples_from_trajectories(trajectories)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(triples))
    triples = [triples[i] for i in order]
    n_train = int(len(triples) * cfg.train_split)

    if unroll > 1:
        model = train_cell_multistep(cfg, trajectories, device, unroll)
    else:
        model = train_cell(cfg, triples, device)

    out: Dict = {"single_step": single_step(model, triples[n_train:], device),
                 "unroll": unroll}
    if not full_eval:
        return out, model
    eval_docs = docs[cfg.num_docs:]
    out.update({
        "rollout_drift": rollout_drift(model, teacher, eval_docs[:100], cfg, device),
        "retrieval": retrieval(model, teacher, eval_docs, cfg, device),
        "sts": sts_eval(model, teacher, cfg, device),
        "pareto": pareto_sweep(model, teacher, eval_docs[:100], cfg, device),
    })
    return out, model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--cell", type=str, default="gated_film")
    ap.add_argument("--ablate-cells", action="store_true")
    ap.add_argument("--ablate-loss", action="store_true")
    ap.add_argument("--unroll", type=int, nargs="+", default=[1],
                    help="multi-step unroll lengths to train (1 == single-step)")
    ap.add_argument("--adaptive", action="store_true",
                    help="sweep the adaptive (path-length) reset policy")
    ap.add_argument("--adaptive-lengths", type=int, nargs="+", default=None,
                    help="also sweep the adaptive policy at these rollout lengths")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2",
                    help="frozen teacher encoder to distill")
    ap.add_argument("--docs-source", type=str, default="wikitext",
                    choices=["wikitext", "scifact", "nfcorpus", "fiqa"],
                    help="distillation corpus: WikiText-2 paragraphs, or a "
                         "BEIR corpus (scifact = in-domain ablation; nfcorpus "
                         "= in-domain-but-disjoint control; fiqa = financial)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="training epochs (default 60); raise to give a larger "
                         "cell a matched compute budget")
    ap.add_argument("--lr", type=float, default=None,
                    help="learning rate (default 3e-4); lower it if a longer "
                         "or larger run diverges")
    ap.add_argument("--layers", type=int, default=None,
                    help="cell depth (default 4); use with --hidden to "
                         "parameter-match against another cell")
    ap.add_argument("--hidden", type=int, default=None,
                    help="cell hidden dim (default: the teacher's embedding dim)")
    ap.add_argument("--normalize-state", action="store_true",
                    help="L2-normalize the state after every update")
    ap.add_argument("--ckpt-tag", type=str, default="",
                    help="suffix for checkpoint filenames, e.g. '_mpnet'")
    ap.add_argument("--ckpt-dir", type=str, default="experiments/results/ckpt")
    ap.add_argument("--out", type=str, default="experiments/results/distill.json")
    args = ap.parse_args()

    from dataclasses import replace

    device = get_device()
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    # Default cell geometry: width-half the teacher dim, three blocks (0.96M for
    # 384-d MiniLM, 3.84M for 768-d mpnet). This tracks the teacher while staying
    # small; see experiments/cells.default_hidden. --hidden/--layers override it
    # (e.g. the 4.29M large variant uses --hidden 384 --layers 4).
    base = DistillConfig(cell=args.cell, teacher=args.teacher,
                         hidden_dim=args.hidden or default_hidden(teacher.emb_dim),
                         epochs=args.epochs or DistillConfig.epochs,
                         num_layers=args.layers or DEFAULT_NUM_LAYERS,
                         lr=args.lr or DistillConfig.lr,
                         normalize_state=args.normalize_state)
    print(f"  teacher={args.teacher} dim={teacher.emb_dim} hidden={base.hidden_dim} "
          f"layers={base.num_layers} epochs={base.epochs} lr={base.lr} "
          f"normalize_state={base.normalize_state}")
    if args.docs_source != "wikitext":
        # BEIR-corpus distillation: scifact = in-domain ablation; nfcorpus =
        # scientific abstracts disjoint from the SciFact index (the leakage
        # control -- domain match without corpus overlap); fiqa = financial.
        from beir import load_beir
        _, texts, _, _ = load_beir(args.docs_source, 60000)
        docs = [t for t in texts if len(t.split()) >= base.min_words]
        # Corpora smaller than num_docs + eval pool (NFCorpus: 3623 docs)
        # would leave docs[num_docs:] EMPTY and crash the rollout/retrieval
        # evaluations. Cap the training slice so at least ~15% (>= 200 docs
        # when available) is held out. SciFact (4405 usable) keeps the
        # original num_docs=4000 split, preserving round-5 reproducibility.
        held_out = max(min(500, len(docs) - base.num_docs), len(docs) // 7)
        if len(docs) - base.num_docs < 200:
            from dataclasses import replace as _replace
            base = _replace(base, num_docs=len(docs) - held_out)
            print(f"  small corpus: training on {base.num_docs}, "
                  f"holding out {held_out} for evaluation")
        print(f"  docs-source={args.docs_source}: {len(docs)} documents")
    else:
        docs = load_wikitext_docs(base.dataset_config, base.num_docs + 2000,
                                  base.min_words)
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results: Dict = {"config": base, "seeds": args.seeds, "main": {},
                     "unroll": {}, "ablations": {}}

    # Main run (single-step training): full metric suite across seeds.
    per_seed, models = [], []
    for s in args.seeds:
        r, m = run_once(base, s, device, teacher, docs, unroll=1)
        per_seed.append(r)
        models.append(m)
    # Save every seed's checkpoint, not just the first: the downstream
    # evaluations (retrieval, STS, stale-cadence, tau) are run once per
    # checkpoint and averaged, so they share the main protocol instead of
    # reporting one seed. The unsuffixed name is kept as an alias for the
    # first seed so existing commands and paths continue to resolve.
    for seed, m in zip(args.seeds, models):
        torch.save(m.state_dict(),
                   ckpt_dir / f"{base.cell}{args.ckpt_tag}_unroll1_seed{seed}.pt")
    torch.save(models[0].state_dict(),
               ckpt_dir / f"{base.cell}{args.ckpt_tag}_unroll1.pt")
    results["main"]["per_seed"] = per_seed
    results["main"]["single_step_cos"] = aggregate_seeds([p["single_step"]["ise_cos"] for p in per_seed])
    results["main"]["retrieval_ise"] = aggregate_seeds([p["retrieval"]["ise"] for p in per_seed])
    results["main"]["sts_ise"] = aggregate_seeds([p["sts"]["ise_spearman"] for p in per_seed])

    if args.adaptive:
        taus = [0.15, 0.25, 0.4, 0.6, 0.9, 1.4]
        eval_docs = docs[base.num_docs:][:100]
        results["main"]["adaptive"] = adaptive_sweep(
            models[0], teacher, eval_docs, base, device, taus)
        # Sensitivity of the tau frontier to rollout length: does a threshold
        # tuned at one horizon still behave sensibly at another?
        if args.adaptive_lengths:
            by_len: Dict = {}
            for L in args.adaptive_lengths:
                cfg_L = replace(base, rollout_k=L)
                by_len[str(L)] = adaptive_sweep(
                    models[0], teacher, eval_docs, cfg_L, device, taus)
                print(f"  adaptive @rollout_k={L}: "
                      + ", ".join(f"tau={r['tau']}:{r['mean_cos']:.3f}" for r in by_len[str(L)]))
            results["main"]["adaptive_by_length"] = by_len

    # Multi-step (unrolled) training: does it extend the drift horizon?
    for u in args.unroll:
        if u <= 1:
            continue
        r, m = run_once(base, args.seeds[0], device, teacher, docs, unroll=u)
        torch.save(m.state_dict(), ckpt_dir / f"{base.cell}{args.ckpt_tag}_unroll{u}.pt")
        results["unroll"][f"unroll={u}"] = r

    def ablate(cfg_variant, key: str, full_eval: bool) -> None:
        """Run one ablation across every seed and store mean/std plus per-seed.

        Ablations used to run on a single seed, which left the paper reporting
        three-seed means in the body and single-seed values in the ablation
        tables. They now share the main protocol.
        """
        per, models_ = [], []
        for seed in args.seeds:
            r, m = run_once(cfg_variant, seed, device, teacher, docs,
                            full_eval=full_eval)
            per.append(r)
            models_.append(m)
        entry = {
            "per_seed": per,
            "ise_cos": float(np.mean([p["single_step"]["ise_cos"] for p in per])),
            "ise_cos_agg": aggregate_seeds([p["single_step"]["ise_cos"] for p in per]),
            "zero_delta_cos": float(np.mean(
                [p["single_step"]["zero_delta_cos"] for p in per])),
        }
        if full_eval:
            entry["retrieval_ise"] = aggregate_seeds(
                [p["retrieval"]["ise"] for p in per])
            entry["sts_ise"] = aggregate_seeds(
                [p["sts"]["ise_spearman"] for p in per])
            entry["rollout_drift"] = {
                str(k): float(np.mean([float(p["rollout_drift"][k]) for p in per]))
                for k in per[0]["rollout_drift"]}
        results["ablations"][key] = entry
        print(f"  ablation {key}: cos={entry['ise_cos']:.5f}"
              + (f" retr={entry['retrieval_ise']['mean'] * 100:.1f}"
                 f" sts={entry['sts_ise']['mean'] * 100:.2f}" if full_eval else ""))

    if args.ablate_cells:
        for cell in ["concat_mlp", "gru", "gated_residual", "gated_film"]:
            ablate(replace(base, cell=cell), f"cell={cell}", full_eval=False)
    if args.ablate_loss:
        # Evaluated in full, not just single-step: the loss ablation is what
        # decides whether the MSE term earns its place, and a term that shapes
        # the state's magnitude cannot show its effect in a one-step cosine.
        for w_mse, w_cos in [(1.0, 0.0), (0.0, 1.0), (1.0, 0.1)]:
            ablate(replace(base, mse_weight=w_mse, cos_weight=w_cos),
                   f"mse={w_mse},cos={w_cos}", full_eval=True)
        # Scale-invariant MSE (on L2-normalized updated embeddings), as an
        # alternative to the raw-magnitude MSE term.
        ablate(replace(base, mse_weight=0.0, cos_weight=0.0, nmse_weight=1.0),
               "nmse=1.0", full_eval=True)

    save_results(args.out, results)
    print("Main:", results["main"]["single_step_cos"], results["main"]["retrieval_ise"], results["main"]["sts_ise"])


if __name__ == "__main__":
    main()
