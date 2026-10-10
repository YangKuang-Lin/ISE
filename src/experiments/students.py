"""Teacher-aligned student encoders (review round 4: the missing baseline).

Two small students trained to map a token prefix directly to the frozen
teacher's sentence embedding of that prefix (LEAF-style alignment):

  CausalStudent -- causal self-attention with last-token pooling. With a KV
                   cache it yields every prefix embedding incrementally and
                   *exactly* (no drift, no resets): appending a token costs one
                   attention pass over the cache. Its error is pure regression
                   error, constant in rollout length.
  BiStudent     -- bidirectional encoder with mean pooling that re-encodes the
                   whole prefix at every token (O(N) per step, small constant).

Both read the teacher's frozen input-embedding table (as the cell does), so no
student carries its own vocabulary; trained parameters are the projections and
transformer blocks only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from .encoders import SinusoidalPE, _CachedBlock


@dataclass
class CausalState:
    """Opaque incremental-decode state for the causal students.

    ``caches`` holds one (K, V) pair per layer; ``run_sum`` and ``n`` carry the
    running hidden-state sum and position count used by the mean readout. Both
    grow only in the KV dimension, so a step costs one attention pass over the
    cache -- O(N) compute, O(N) memory, and exact (no drift, no resets).
    """

    caches: List
    run_sum: Optional[torch.Tensor]
    n: int


class CausalStudent(nn.Module):
    """Causal prefix embedder aligned to the teacher's space.

    Training: a single dense forward with a causal mask supervises every
    position's projected hidden state against the teacher embedding of the
    corresponding prefix. Inference: ``prefill`` + ``step`` maintain a KV cache
    so each new token costs one attention pass over the cache.

    ``pool`` selects the readout. ``"last"`` projects the final position's
    hidden state; ``"mean"`` projects the causal running mean over positions
    ``0..t``, which matches the mean-pooled teacher's aggregation and is still
    exactly incremental (the running sum is O(1) state). Review round 8: the
    last-token/mean-pool mismatch is one of two candidate explanations for the
    causal students' collapse, so it is ablated rather than assumed.
    """

    def __init__(self, teacher_dim: int, d_model: int, num_layers: int,
                 nhead: int = 4, ff_dim: Optional[int] = None,
                 max_len: int = 4096, pool: str = "last") -> None:
        super().__init__()
        if pool not in ("last", "mean"):
            raise ValueError(f"pool must be 'last' or 'mean', got {pool!r}")
        ff_dim = ff_dim or 2 * d_model
        self.pool = pool
        self.in_proj = nn.Linear(teacher_dim, d_model)
        self.pos = SinusoidalPE(d_model, max_len)
        self.blocks = nn.ModuleList(
            [_CachedBlock(d_model, nhead, ff_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, teacher_dim)

    def forward(self, tok_rows: torch.Tensor) -> torch.Tensor:
        """Dense causal forward: (B, T, D_t) -> per-position prefix embeddings
        (B, T, D_t); position ``t`` sees tokens ``0..t`` only."""
        B, T, _ = tok_rows.shape
        pos = torch.arange(T, device=tok_rows.device)
        x = self.in_proj(tok_rows) + self.pos(pos).unsqueeze(0)
        for block in self.blocks:
            x, _ = block(x, None)          # cache=None + T>1 => causal mask
        h = self.norm(x)
        if self.pool == "mean":
            # Causal running mean: position t averages hidden states 0..t, so
            # every row still depends only on its own prefix.
            denom = torch.arange(1, T + 1, device=h.device,
                                 dtype=h.dtype).view(1, T, 1)
            h = h.cumsum(dim=1) / denom
        return self.out_proj(h)

    @torch.no_grad()
    def prefill(self, tok_rows: torch.Tensor) -> Tuple[torch.Tensor, "CausalState"]:
        """Encode a prompt; return (prefix embedding (B, D_t), decode state).

        The returned state is opaque to callers and is threaded back into
        :meth:`step`; it holds the per-layer KV caches and, under mean pooling,
        the running hidden-state sum that makes the mean readout O(1).
        """
        B, T, _ = tok_rows.shape
        pos = torch.arange(T, device=tok_rows.device)
        x = self.in_proj(tok_rows) + self.pos(pos).unsqueeze(0)
        caches: List = []
        for block in self.blocks:
            x, c = block(x, None)
            caches.append(c)
        h = self.norm(x)
        if self.pool == "mean":
            run_sum = h.sum(dim=1)
            state = CausalState(caches, run_sum, T)
            return self.out_proj(run_sum / T), state
        return self.out_proj(h[:, -1, :]), CausalState(caches, None, T)

    @torch.no_grad()
    def step(self, tok_row: torch.Tensor, state: "CausalState", position: int
             ) -> Tuple[torch.Tensor, "CausalState"]:
        """Advance by one token: (B, D_t) row -> (embedding, new state)."""
        pos = torch.tensor([position], device=tok_row.device)
        x = self.in_proj(tok_row).unsqueeze(1) + self.pos(pos).unsqueeze(0)
        new_caches: List = []
        for block, c in zip(self.blocks, state.caches):
            x, nc = block(x, c)
            new_caches.append(nc)
        h = self.norm(x[:, -1, :])
        n = state.n + 1
        if self.pool == "mean":
            run_sum = state.run_sum + h
            return self.out_proj(run_sum / n), CausalState(new_caches, run_sum, n)
        return self.out_proj(h), CausalState(new_caches, None, n)


class BiStudent(nn.Module):
    """Bidirectional mean-pooled prefix embedder aligned to the teacher's space.

    No incremental path exists: every prefix is re-encoded in full, which is the
    LEAF-style deployment (cost grows with prefix length, but the encoder is
    tiny).
    """

    def __init__(self, teacher_dim: int, d_model: int, num_layers: int,
                 nhead: int = 4, ff_dim: Optional[int] = None,
                 max_len: int = 4096) -> None:
        super().__init__()
        ff_dim = ff_dim or 2 * d_model
        self.in_proj = nn.Linear(teacher_dim, d_model)
        self.pos = SinusoidalPE(d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, teacher_dim)

    def forward(self, tok_rows: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, T, D_t) [+ pad mask True=pad] -> (B, D_t) mean-pooled embedding."""
        B, T, _ = tok_rows.shape
        pos = torch.arange(T, device=tok_rows.device)
        x = self.in_proj(tok_rows) + self.pos(pos).unsqueeze(0)
        h = self.encoder(x, src_key_padding_mask=pad_mask)
        h = self.norm(h)
        if pad_mask is not None:
            keep = (~pad_mask).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        else:
            pooled = h.mean(1)
        return self.out_proj(pooled)


class SeededBiStudent(BiStudent):
    """BiStudent additionally conditioned on a teacher seed embedding.

    Review round 6: the cell is handed a teacher embedding as its state while
    the plain student regresses from token rows alone, so the student/cell gap
    confounds parameter count with conditioning. This control removes the
    asymmetry: the most recent teacher embedding (seed, or last re-grounding)
    is projected and prepended as an extra input position, giving the student
    the same information the cell starts from at the same teacher-call budget.
    """

    def __init__(self, teacher_dim: int, d_model: int, num_layers: int,
                 nhead: int = 4, ff_dim: Optional[int] = None,
                 max_len: int = 4096) -> None:
        super().__init__(teacher_dim, d_model, num_layers, nhead, ff_dim, max_len)
        self.seed_proj = nn.Linear(teacher_dim, d_model)
        # Learned type vector marking the seed position as non-token input.
        self.seed_type = nn.Parameter(torch.zeros(d_model))

    def forward(self, tok_rows: torch.Tensor,
                pad_mask: Optional[torch.Tensor] = None,
                seed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, T, D_t) tokens + (B, D_t) seed -> (B, D_t) embedding."""
        if seed is None:
            raise ValueError("SeededBiStudent requires a seed embedding.")
        B, T, _ = tok_rows.shape
        pos = torch.arange(T, device=tok_rows.device)
        x = self.in_proj(tok_rows) + self.pos(pos).unsqueeze(0)
        s = (self.seed_proj(seed) + self.seed_type).unsqueeze(1)   # (B, 1, d)
        x = torch.cat([s, x], dim=1)
        if pad_mask is not None:
            keep_seed = torch.zeros(B, 1, dtype=torch.bool,
                                    device=pad_mask.device)
            pad_mask = torch.cat([keep_seed, pad_mask], dim=1)
        h = self.norm(self.encoder(x, src_key_padding_mask=pad_mask))
        if pad_mask is not None:
            keep = (~pad_mask).unsqueeze(-1).float()
            pooled = (h * keep).sum(1) / keep.sum(1).clamp(min=1.0)
        else:
            pooled = h.mean(1)
        return self.out_proj(pooled)


class SeededGRUStudent(nn.Module):
    """Teacher-aligned recurrent student: O(1) per token, seeded like the cell.

    Review round 7: the remaining student gap is architectural class -- the
    seeded students re-encode the prefix (cost grows with length) while the
    cell updates in constant time. This is the missing O(1) competitor: a GRU
    whose hidden state is initialized from the most recent teacher embedding
    (seed, or last re-grounding) and advanced one token row at a time, read
    out into the teacher's space after every step. Unlike the cell it carries
    a free-form hidden state and regresses the embedding directly instead of
    predicting a shift added to the teacher state.
    """

    def __init__(self, teacher_dim: int, d_model: int, num_layers: int = 1) -> None:
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model
        self.in_proj = nn.Linear(teacher_dim, d_model)
        self.seed_proj = nn.Linear(teacher_dim, d_model * num_layers)
        self.gru = nn.GRU(d_model, d_model, num_layers=num_layers,
                          batch_first=True)
        self.out_proj = nn.Linear(d_model, teacher_dim)

    def init_state(self, anchor: torch.Tensor) -> torch.Tensor:
        """(B, D_t) teacher embedding -> (L, B, d) initial hidden state."""
        B = anchor.shape[0]
        h = self.seed_proj(anchor).view(B, self.num_layers, self.d_model)
        return torch.tanh(h).transpose(0, 1).contiguous()

    def forward(self, tok_rows: torch.Tensor,
                lengths: Optional[torch.Tensor] = None,
                seed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, T, D_t) rows after the anchor + (B, D_t) anchor -> (B, D_t).

        ``lengths`` gives the true (unpadded) length per batch item; the
        readout is taken at that step. T == 0 returns the anchor readout.
        """
        if seed is None:
            raise ValueError("SeededGRUStudent requires an anchor embedding.")
        h0 = self.init_state(seed)
        if tok_rows.shape[1] == 0:
            return self.out_proj(h0[-1])
        out, _ = self.gru(self.in_proj(tok_rows), h0)
        if lengths is None:
            return self.out_proj(out[:, -1])
        idx = (lengths.clamp(min=1) - 1).view(-1, 1, 1).expand(-1, 1, out.shape[-1])
        gathered = out.gather(1, idx).squeeze(1)
        # Items with zero post-anchor tokens read out the anchor state instead.
        zero = (lengths == 0).unsqueeze(-1)
        return self.out_proj(torch.where(zero, h0[-1], gathered))

    def step(self, tok_row: torch.Tensor, h: torch.Tensor):
        """One O(1) update: (B, D_t) row + (L, B, d) hidden -> (emb, hidden)."""
        out, h = self.gru(self.in_proj(tok_row).unsqueeze(1), h)
        return self.out_proj(out[:, -1]), h


class SeededCausalStudent(CausalStudent):
    """Causal student additionally conditioned on a teacher anchor embedding.

    Review round 8 (reviewer Q1): the causal student with a KV cache is the one
    baseline with *exact* prefix reuse -- no drift, no resets, O(N) per step at
    a tiny constant -- and it is the sharpest competitor to the cell in the
    long-stream regime. Round 6 removed the conditioning asymmetry for the
    bidirectional students but never for this one, so the comparison was
    incomplete.

    The anchor (the seed teacher embedding, or the most recent re-grounding) is
    projected and placed at position 0, ahead of every token, so causal
    attention lets every later position read it. Re-grounding is a re-prefill
    with the fresh anchor: the cache is rebuilt, which costs one dense pass at
    the reset step and leaves the incremental path exact in between.
    """

    def __init__(self, teacher_dim: int, d_model: int, num_layers: int,
                 nhead: int = 4, ff_dim: Optional[int] = None,
                 max_len: int = 4096, pool: str = "last") -> None:
        super().__init__(teacher_dim, d_model, num_layers, nhead, ff_dim,
                         max_len, pool)
        self.seed_proj = nn.Linear(teacher_dim, d_model)
        # Learned type vector marking position 0 as an anchor, not a token.
        self.seed_type = nn.Parameter(torch.zeros(d_model))

    def _embed_inputs(self, tok_rows: torch.Tensor, seed: torch.Tensor,
                      start: int = 0) -> torch.Tensor:
        """Project tokens (+ the anchor at position 0 when ``start == 0``)."""
        B, T, _ = tok_rows.shape
        pos = torch.arange(start, start + T, device=tok_rows.device)
        x = self.in_proj(tok_rows) + self.pos(pos + 1).unsqueeze(0)
        if start == 0:
            s = (self.seed_proj(seed) + self.seed_type).unsqueeze(1)
            zero = torch.zeros(1, dtype=torch.long, device=tok_rows.device)
            x = torch.cat([s + self.pos(zero).unsqueeze(0), x], dim=1)
        return x

    def forward(self, tok_rows: torch.Tensor,
                seed: Optional[torch.Tensor] = None) -> torch.Tensor:
        """(B, T, D_t) tokens + (B, D_t) anchor -> (B, T, D_t) prefix embeddings.

        The anchor occupies position 0 and is dropped from the output, so row
        ``t`` remains the embedding of the prefix ending at token ``t``.
        """
        if seed is None:
            raise ValueError("SeededCausalStudent requires an anchor embedding.")
        x = self._embed_inputs(tok_rows, seed)
        for block in self.blocks:
            x, _ = block(x, None)
        h = self.norm(x)
        if self.pool == "mean":
            denom = torch.arange(1, h.shape[1] + 1, device=h.device,
                                 dtype=h.dtype).view(1, -1, 1)
            h = h.cumsum(dim=1) / denom
        return self.out_proj(h[:, 1:, :])      # drop the anchor position

    @torch.no_grad()
    def prefill(self, tok_rows: torch.Tensor,
                seed: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, CausalState]:
        if seed is None:
            raise ValueError("SeededCausalStudent requires an anchor embedding.")
        x = self._embed_inputs(tok_rows, seed)
        caches: List = []
        for block in self.blocks:
            x, c = block(x, None)
            caches.append(c)
        h = self.norm(x)
        T = h.shape[1]
        if self.pool == "mean":
            run_sum = h.sum(dim=1)
            return self.out_proj(run_sum / T), CausalState(caches, run_sum, T)
        return self.out_proj(h[:, -1, :]), CausalState(caches, None, T)

    @torch.no_grad()
    def step(self, tok_row: torch.Tensor, state: CausalState, position: int
             ) -> Tuple[torch.Tensor, CausalState]:
        """Advance one token. ``position`` is the token index; the anchor holds
        position 0, so the token's own encoding is shifted by one."""
        x = self._embed_inputs(tok_row.unsqueeze(1), None, start=position)
        new_caches: List = []
        for block, c in zip(self.blocks, state.caches):
            x, nc = block(x, c)
            new_caches.append(nc)
        h = self.norm(x[:, -1, :])
        n = state.n + 1
        if self.pool == "mean":
            run_sum = state.run_sum + h
            return self.out_proj(run_sum / n), CausalState(new_caches, run_sum, n)
        return self.out_proj(h), CausalState(new_caches, None, n)


__all__ = ["CausalState", "CausalStudent", "BiStudent", "SeededBiStudent",
           "SeededCausalStudent", "SeededGRUStudent"]
