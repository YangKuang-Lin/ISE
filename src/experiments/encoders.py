"""Sequence encoders and causal language models.

Contains the from-scratch ISE encoder (a cell unrolled over time), a
bidirectional Transformer encoder, and three causal language models used by the
speed benchmark: ISE (O(1) state update), a GRU baseline, and a Transformer
with an explicit **KV cache**. The cached Transformer is the fair inference
baseline that the original benchmark was missing.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
from torch import nn

from .cells import build_cell

PAD_ID = 0


class SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        return self.pe[positions]


# ─────────────────────────────────────────────────────────────────────────────
# From-scratch encoders (MLM pre-training + classification)
# ─────────────────────────────────────────────────────────────────────────────
class ISEEncoder(nn.Module):
    """Unrolls an incremental cell over the sequence, exposing every state.

    The state is advanced by ``e = LayerNorm(e + cell(token, e))`` which keeps
    the residual/skip-connection structure while bounding the state norm.
    """

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 num_layers: int, cell: str = "gated_residual") -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.tok_embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.pos_enc = SinusoidalPE(emb_dim)
        self.cell = build_cell(cell, emb_dim, hidden_dim, num_layers)
        self.state_norm = nn.LayerNorm(emb_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        tok_emb = self.tok_embed(input_ids) + self.pos_enc(pos)
        e = torch.zeros(B, self.emb_dim, device=input_ids.device)
        states: List[torch.Tensor] = []
        for t in range(T):
            e = self.state_norm(e + self.cell(tok_emb[:, t, :], e))
            states.append(e)
        return torch.stack(states, dim=1)


class TFEncoder(nn.Module):
    """Bidirectional Transformer encoder baseline."""

    def __init__(self, vocab_size: int, emb_dim: int, nhead: int, num_layers: int,
                 ff_dim: int, max_len: int = 512) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_ID)
        self.pos_embed = nn.Embedding(max_len, emb_dim)
        self.drop = nn.Dropout(0.1)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=ff_dim,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(emb_dim)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.tok_embed(input_ids) + self.pos_embed(pos))
        pad_mask = input_ids == PAD_ID
        return self.norm(self.transformer(x, src_key_padding_mask=pad_mask))


# ─────────────────────────────────────────────────────────────────────────────
# Causal LMs for the speed benchmark
# ─────────────────────────────────────────────────────────────────────────────
class ISECausalLM(nn.Module):
    """O(1)-per-token generator: one cell step + output head."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int,
                 num_layers: int, cell: str = "gated_residual") -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.tok_embed = nn.Embedding(vocab_size, emb_dim)
        self.cell = build_cell(cell, emb_dim, hidden_dim, num_layers)
        self.state_norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def step(self, token_id: torch.Tensor, e_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tok_embed(token_id)
        e_new = self.state_norm(e_prev + self.cell(tok, e_prev))
        return self.head(e_new), e_new


class GRUCausalLM(nn.Module):
    """GRU baseline generator (also O(1) per token) for the speed comparison."""

    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def step(self, token_id: torch.Tensor, h: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        tok = self.tok_embed(token_id).unsqueeze(1)  # (B, 1, D)
        out, h = self.gru(tok, h)
        return self.head(out.squeeze(1)), h


class _CachedMHA(nn.Module):
    """Multi-head self-attention with an incremental KV cache."""

    def __init__(self, d_model: int, nhead: int) -> None:
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.dh = d_model // nhead
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor,
                cache: Optional[Tuple[torch.Tensor, torch.Tensor]]
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # x: (B, T_new, D). During cached decoding T_new == 1.
        B, T, D = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.nhead, self.dh).transpose(1, 2)
        k = k.view(B, T, self.nhead, self.dh).transpose(1, 2)
        v = v.view(B, T, self.nhead, self.dh).transpose(1, 2)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        # Causal attention. When decoding a single new token against the cache,
        # it may attend to all cached positions, so no mask is required there.
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.dh)
        if T > 1:
            mask = torch.triu(torch.ones(T, k.size(2), device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))
        attn = attn.softmax(dim=-1)
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, D)
        return self.out(ctx), new_cache


class _CachedBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_dim: int) -> None:
        super().__init__()
        self.n1 = nn.LayerNorm(d_model)
        self.attn = _CachedMHA(d_model, nhead)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_dim), nn.GELU(), nn.Linear(ff_dim, d_model))

    def forward(self, x, cache):
        a, new_cache = self.attn(self.n1(x), cache)
        x = x + a
        x = x + self.ff(self.n2(x))
        return x, new_cache


class TFCausalLMCached(nn.Module):
    """Causal Transformer LM supporting incremental KV-cached decoding.

    ``step`` advances generation by one token in O(N) attention (over the cache)
    rather than re-encoding the whole prefix, which is the standard efficient
    inference path and the fair baseline against ISE.
    """

    def __init__(self, vocab_size: int, emb_dim: int, nhead: int, num_layers: int,
                 ff_dim: int = 512, max_len: int = 8192) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(max_len, emb_dim)
        self.blocks = nn.ModuleList([_CachedBlock(emb_dim, nhead, ff_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def prefill(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, List]:
        """Encode a prompt, returning last-token logits and per-layer caches."""
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        caches: List = []
        for block in self.blocks:
            x, c = block(x, None)
            caches.append(c)
        logits = self.head(self.norm(x[:, -1, :]))
        return logits, caches

    def step(self, token_id: torch.Tensor, caches: List, pos: int) -> Tuple[torch.Tensor, List]:
        x = self.tok_embed(token_id).unsqueeze(1) + self.pos_embed(
            torch.tensor([[pos]], device=token_id.device))
        new_caches: List = []
        for block, c in zip(self.blocks, caches):
            x, nc = block(x, c)
            new_caches.append(nc)
        logits = self.head(self.norm(x[:, -1, :]))
        return logits, new_caches


class TFCausalLMNoCache(nn.Module):
    """Causal Transformer that re-encodes the full prefix each step.

    This reproduces the original (unfair) benchmark path and is kept so the
    paper can report the with/without-cache gap side by side.
    """

    def __init__(self, vocab_size: int, emb_dim: int, nhead: int, num_layers: int,
                 ff_dim: int = 512, max_len: int = 8192) -> None:
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, emb_dim)
        self.pos_embed = nn.Embedding(max_len, emb_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, nhead=nhead, dim_feedforward=ff_dim,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_embed(input_ids) + self.pos_embed(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        hidden = self.norm(self.transformer(x, mask=mask, is_causal=True))
        return self.head(hidden[:, -1, :])


__all__ = [
    "PAD_ID", "SinusoidalPE", "ISEEncoder", "TFEncoder",
    "ISECausalLM", "GRUCausalLM", "TFCausalLMCached", "TFCausalLMNoCache",
]
