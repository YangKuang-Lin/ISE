"""State-bottleneck probes (review round 4: is the horizon architectural?).

The deployed cell is Markovian in the teacher embedding itself: its entire
state is E(x_{1..n}). The review hypothesis is that the drift horizon is a
*state bottleneck*, not intrinsic difficulty: a sentence embedding optimized
for semantics may be a poor sufficient statistic for its own dynamics. Two
O(1)-preserving variants test this:

  aux     -- the cell carries a small auxiliary hidden state h alongside E
             (zeroed at every reset, so re-grounding still fully restores the
             tracked state);
  window  -- the cell conditions on the last 3 token-embedding rows instead of
             only the newest one (re-contextualization is largely local).

All three variants (baseline FiLM included) are trained with the identical
unrolled U=4 trajectory objective across three seeds, so the comparison
isolates the state design from the training regime. The baseline row doubles
as the three-seed downstream evaluation of U=4 training that review round 4
also requested.

Usage:
    python bottleneck.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn, optim

from experiments.cells import DEFAULT_NUM_LAYERS, default_hidden
from experiments.cells import _FiLMBlock  # shared FiLM block
from experiments.distill_data import Teacher, build_trajectories, load_wikitext_docs
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)

MIN_TOKENS, MAX_TOKENS, ROLLOUT_K = 10, 80, 64
NUM_DOCS, TRAIN_SPLIT, EPOCHS, LR, BATCH = 4000, 0.8, 60, 3e-4, 1024
UNROLL = 4
AUX_DIM = 192          # overridden to emb_dim // 2 at build time
WINDOW = 3


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class BaseFiLM(nn.Module):
    """The deployed FiLM cell, restated locally so all three variants share
    exactly one implementation of the block stack."""

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int,
                 ctx_in: int = 0, tok_in: int = 0) -> None:
        super().__init__()
        ctx_in = ctx_in or emb_dim
        tok_in = tok_in or emb_dim
        self.ctx_net = nn.Sequential(nn.Linear(ctx_in, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.tok_net = nn.Sequential(nn.Linear(tok_in, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([_FiLMBlock(hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, emb_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def features(self, tok_in: torch.Tensor, ctx_in: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx_net(ctx_in)
        x = self.tok_net(tok_in)
        for block in self.blocks:
            x = block(x, ctx)
        return x

    def forward(self, tok: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.out(self.features(tok, e))


class AuxFiLM(nn.Module):
    """FiLM cell with an auxiliary carried state h (zeroed at resets).

    forward(tok, e, h) -> (delta, h_new). Still O(1) per step; the tracked
    state remains E, so a teacher reset fully re-grounds the system with h=0.
    """

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int,
                 aux_dim: int = AUX_DIM) -> None:
        super().__init__()
        self.aux_dim = aux_dim
        self.core = BaseFiLM(emb_dim, hidden_dim, num_layers,
                             ctx_in=emb_dim + aux_dim)
        self.aux_update = nn.GRUCell(hidden_dim, aux_dim)

    def forward(self, tok: torch.Tensor, e: torch.Tensor, h: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.core.features(tok, torch.cat([e, h], dim=-1))
        return self.core.out(x), self.aux_update(x, h)

    def init_aux(self, batch: int, device: str) -> torch.Tensor:
        return torch.zeros(batch, self.aux_dim, device=device)


class WindowFiLM(nn.Module):
    """FiLM cell conditioning on the last WINDOW token rows (concatenated)."""

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int,
                 window: int = WINDOW) -> None:
        super().__init__()
        self.window = window
        self.core = BaseFiLM(emb_dim, hidden_dim, num_layers,
                             tok_in=emb_dim * window)

    def forward(self, tok_window: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return self.core(tok_window, e)


class PosFiLM(nn.Module):
    """FiLM cell additionally conditioned on the absolute prefix position.

    Review round 8 (reviewer Q4). The deployed cell reads only
    ``(token_row, state)`` and has no position signal, yet the teacher's own
    per-token shift falls monotonically with absolute position (0.168 over
    tokens 10-30 down to 0.050 over 100-140, Appendix "Drift and Absolute
    Prefix Position"), and rollouts seeded deep in a document never cross the
    0.90 horizon. So the cell must infer prefix depth implicitly from a state
    geometry that was optimized for sentence semantics, not for encoding
    depth -- a plausible and cheap explanation for the horizon that the
    existing probes do not rule out.

    Position enters as a Fourier feature bank over ``log(1 + n)`` appended to
    the context pathway, which keeps resolution at small ``n`` (where drift is
    fastest) without saturating at large ``n``. The variant remains O(1) per
    step and re-grounding still fully restores the tracked state, since ``n``
    is an observed counter rather than a carried hidden state.
    """

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int,
                 n_freq: int = 8) -> None:
        super().__init__()
        self.n_freq = n_freq
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq).float())
        self.core = BaseFiLM(emb_dim, hidden_dim, num_layers,
                             ctx_in=emb_dim + 2 * n_freq)

    def pos_feat(self, pos: torch.Tensor) -> torch.Tensor:
        """(B,) absolute prefix lengths -> (B, 2 * n_freq) Fourier features."""
        x = torch.log1p(pos.float()).unsqueeze(-1) * self.freqs
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

    def forward(self, tok: torch.Tensor, e: torch.Tensor,
                pos: torch.Tensor) -> torch.Tensor:
        return self.core(tok, torch.cat([e, self.pos_feat(pos)], dim=-1))


class WindowedTraj(torch.utils.data.Dataset):
    """U-step windows carrying WINDOW-1 rows of pre-window token context.

    Pre-window rows that precede the trajectory start are zero-padded; at
    evaluation time real seed-token rows exist, a negligible mismatch confined
    to the first two positions of each document.
    """

    def __init__(self, trajectories, unroll: int, window: int) -> None:
        E, T, C, P = [], [], [], []
        for embs, toks in trajectories:
            n = len(toks)
            for s in range(0, n - unroll + 1):
                E.append(embs[s:s + unroll + 1])
                T.append(toks[s:s + unroll])
                pre = np.zeros((window - 1, toks.shape[1]), dtype=toks.dtype)
                lo = max(0, s - (window - 1))
                got = toks[lo:s]
                if len(got):
                    pre[-len(got):] = got
                C.append(pre)
                # Absolute prefix length of the window's first state: state t
                # of this window is the embedding of prefix MIN_TOKENS + s + t.
                P.append(MIN_TOKENS + s)
        self.E = torch.tensor(np.array(E), dtype=torch.float32)
        self.T = torch.tensor(np.array(T), dtype=torch.float32)
        self.C = torch.tensor(np.array(C), dtype=torch.float32)
        self.P = torch.tensor(np.array(P), dtype=torch.float32)

    def __len__(self):
        return len(self.E)

    def __getitem__(self, i):
        return self.E[i], self.T[i], self.C[i], self.P[i]


def _tok_window(T: torch.Tensor, C: torch.Tensor, t: int, window: int) -> torch.Tensor:
    """Concatenated rows [t-window+1 .. t] using pre-window context C."""
    parts = []
    for off in range(window - 1, -1, -1):
        j = t - off
        parts.append(T[:, j] if j >= 0 else C[:, (window - 1) + j])
    return torch.cat(parts, dim=-1)


def train_variant(variant: str, trajectories, emb_dim: int, device: str
                  ) -> nn.Module:
    hidden = default_hidden(emb_dim)
    if variant == "baseline":
        model = BaseFiLM(emb_dim, hidden, DEFAULT_NUM_LAYERS)
    elif variant == "aux":
        # Aux width follows the cell width (emb_dim // 2), so the probe stays
        # proportional under larger teachers.
        model = AuxFiLM(emb_dim, hidden, DEFAULT_NUM_LAYERS, aux_dim=hidden)
    elif variant == "pos":
        model = PosFiLM(emb_dim, hidden, DEFAULT_NUM_LAYERS)
    else:
        model = WindowFiLM(emb_dim, hidden, DEFAULT_NUM_LAYERS)
    model = model.to(device)
    ds = WindowedTraj(trajectories, UNROLL, WINDOW)
    idx = torch.randperm(len(ds))
    n_train = int(len(ds) * TRAIN_SPLIT)
    train_idx, val_idx = idx[:n_train], idx[n_train:]
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    print(f"  variant={variant} params={count_params(model):,} windows={len(ds):,}")

    def run_batch(E, T, C, P):
        e = E[:, 0]
        h = model.init_aux(E.size(0), device) if variant == "aux" else None
        loss = 0.0
        for t in range(T.size(1)):
            if variant == "aux":
                delta, h = model(T[:, t], e, h)
            elif variant == "window":
                delta = model(_tok_window(T, C, t, WINDOW), e)
            elif variant == "pos":
                delta = model(T[:, t], e, P + t)
            else:
                delta = model(T[:, t], e)
            e = e + delta
            tgt = E[:, t + 1]
            loss = loss + nn.functional.mse_loss(e, tgt) \
                        + 0.1 * (1 - nn.functional.cosine_similarity(e, tgt)).mean()
        return loss / T.size(1)

    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        perm = train_idx[torch.randperm(len(train_idx))]
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            loss = run_batch(ds.E[b].to(device), ds.T[b].to(device),
                             ds.C[b].to(device), ds.P[b].to(device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss ({variant}).")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            vl = [run_batch(ds.E[b].to(device), ds.T[b].to(device),
                            ds.C[b].to(device), ds.P[b].to(device)).item()
                  for b in [val_idx[i:i + BATCH]
                            for i in range(0, len(val_idx), BATCH)]]
        v = float(np.mean(vl))
        if v < best:
            best = v
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def roll_final(model, variant: str, teacher: Teacher, ids: List[int], n0: int,
               steps: int, device: str) -> np.ndarray:
    """Roll unreset from the seed and return the final state (one teacher call)."""
    e = torch.tensor(teacher.encode_prefixes(ids, [n0])[0], dtype=torch.float32,
                     device=device).unsqueeze(0)
    rows = teacher.token_embed(ids)
    h = model.init_aux(1, device) if variant == "aux" else None
    for k in range(steps):
        t = n0 + k
        if variant == "aux":
            tok = torch.tensor(rows[t], dtype=torch.float32, device=device).unsqueeze(0)
            delta, h = model(tok, e, h)
        elif variant == "window":
            w = np.concatenate([rows[t - WINDOW + 1 + j] for j in range(WINDOW)])
            tok = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)
            delta = model(tok, e)
        elif variant == "pos":
            tok = torch.tensor(rows[t], dtype=torch.float32, device=device).unsqueeze(0)
            # State e is the embedding of the prefix of length t; the cell is
            # asked for the shift that appends token t.
            delta = model(tok, e, torch.tensor([float(t)], device=device))
        else:
            tok = torch.tensor(rows[t], dtype=torch.float32, device=device).unsqueeze(0)
            delta = model(tok, e)
        e = e + delta
        if not torch.isfinite(e).all():
            raise FloatingPointError(f"Non-finite state ({variant}) at step {k}.")
    return e.squeeze(0).cpu().numpy()


@torch.no_grad()
def drift_curve(model, variant: str, teacher, docs, device: str) -> Dict[str, float]:
    per_step: Dict[int, List[float]] = {k: [] for k in range(1, ROLLOUT_K + 1)}
    used = 0
    for doc in docs:
        ids = teacher.token_ids(doc, MAX_TOKENS + ROLLOUT_K + 1)
        if len(ids) < MIN_TOKENS + ROLLOUT_K + 1:
            continue
        ks = list(range(MIN_TOKENS, MIN_TOKENS + ROLLOUT_K + 1))
        true = teacher.encode_prefixes(ids, ks)
        rows = teacher.token_embed(ids)
        e = torch.tensor(true[0], dtype=torch.float32, device=device).unsqueeze(0)
        h = model.init_aux(1, device) if variant == "aux" else None
        for k in range(1, ROLLOUT_K + 1):
            t = MIN_TOKENS + k - 1
            if variant == "aux":
                tok = torch.tensor(rows[t], dtype=torch.float32, device=device).unsqueeze(0)
                delta, h = model(tok, e, h)
            elif variant == "window":
                w = np.concatenate([rows[t - WINDOW + 1 + j] for j in range(WINDOW)])
                tok = torch.tensor(w, dtype=torch.float32, device=device).unsqueeze(0)
                delta = model(tok, e)
            elif variant == "pos":
                tok = torch.tensor(rows[t], dtype=torch.float32, device=device).unsqueeze(0)
                delta = model(tok, e, torch.tensor([float(t)], device=device))
            else:
                tok = torch.tensor(rows[t], dtype=torch.float32, device=device).unsqueeze(0)
                delta = model(tok, e)
            e = e + delta
            per_step[k].append(_cos(e.squeeze(0).cpu().numpy(), true[k]))
        used += 1
        if used >= 100:
            break
    return {str(k): float(np.mean(v)) for k, v in per_step.items() if v}


@torch.no_grad()
def retrieval(model, variant: str, teacher, docs, device: str) -> float:
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
        q = roll_final(model, variant, teacher, ids, MIN_TOKENS, ROLLOUT_K, device)
        q = q / (np.linalg.norm(q) + 1e-9)
        if int(np.argmax(corpus @ q)) == gold:
            hits += 1
    return hits / len(usable)


def horizon(curve: Dict[str, float]) -> int:
    steps = sorted(int(k) for k in curve)
    for k in steps:
        if curve[str(k)] < 0.90:
            return k
    return steps[-1] + 1        # never crossed within the window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--out", type=str, default="experiments/results/bottleneck.json")
    args = ap.parse_args()

    device = get_device()
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    docs = load_wikitext_docs("wikitext-2-raw-v1", NUM_DOCS + 2000, 40)

    variants = ["baseline", "aux", "window", "pos"]
    per: Dict[str, List[Dict]] = {v: [] for v in variants}
    for seed in args.seeds:
        set_seed(seed)
        trajectories = build_trajectories(teacher, docs[:NUM_DOCS],
                                          MIN_TOKENS, MAX_TOKENS)
        eval_docs = docs[NUM_DOCS:]
        for v in variants:
            set_seed(seed)
            model = train_variant(v, trajectories, teacher.emb_dim, device)
            model.eval()
            curve = drift_curve(model, v, teacher, eval_docs, device)
            entry = {"seed": seed, "params": count_params(model),
                     "rollout_drift": curve,
                     "cos64": curve[str(ROLLOUT_K)],
                     "horizon": horizon(curve),
                     "retrieval": retrieval(model, v, teacher, eval_docs, device)}
            per[v].append(entry)
            print(f"  [{v} seed={seed}] horizon={entry['horizon']} "
                  f"cos64={entry['cos64']:.3f} retr={entry['retrieval'] * 100:.1f}")

    out: Dict = {"unroll": UNROLL, "aux_dim": AUX_DIM, "window": WINDOW,
                 "seeds": args.seeds, "variants": {}}
    for v in variants:
        # Horizon of the seed-mean curve (the canonical statistic).
        steps = sorted(int(k) for k in per[v][0]["rollout_drift"])
        mean_curve = {k: float(np.mean([p["rollout_drift"][str(k)] for p in per[v]]))
                      for k in steps}
        out["variants"][v] = {
            "per_seed": per[v],
            "params": per[v][0]["params"],
            "mean_curve": {str(k): mean_curve[k] for k in steps},
            "horizon_mean_curve": horizon({str(k): mean_curve[k] for k in steps}),
            "cos64": aggregate_seeds([p["cos64"] for p in per[v]]),
            "retrieval": aggregate_seeds([p["retrieval"] for p in per[v]]),
        }
    save_results(args.out, out)
    print("done.")


if __name__ == "__main__":
    main()
