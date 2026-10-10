"""Incremental update cells.

Every cell maps ``(token_emb, e_prev) -> update`` where ``update`` is the
predicted embedding shift ``ΔE``. The state is advanced by the caller as
``e_new = e_prev + update`` (optionally re-normalized). Casting each variant --
FiLM gating, a plain concat-MLP, a GRU, and a gated-residual cell -- into the
same "predict the shift" interface lets the ablation compare them fairly,
which addresses the confounded comparison flagged in review.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch
from torch import nn

CELL_REGISTRY: Dict[str, Type[nn.Module]] = {}


def register_cell(name: str) -> Callable[[Type[nn.Module]], Type[nn.Module]]:
    """Class decorator that registers a cell under ``name``."""

    def decorator(cls: Type[nn.Module]) -> Type[nn.Module]:
        CELL_REGISTRY[name] = cls
        return cls

    return decorator


def build_cell(name: str, emb_dim: int, hidden_dim: int, num_layers: int) -> nn.Module:
    """Factory: instantiate a registered cell by name."""
    if name not in CELL_REGISTRY:
        raise ValueError(f"Unknown cell '{name}'. Available: {sorted(CELL_REGISTRY)}")
    return CELL_REGISTRY[name](emb_dim, hidden_dim, num_layers)


# --- default cell geometry (single source of truth) --------------------------
# The deployed FiLM cell is width-half the teacher embedding dimension and three
# blocks: 0.96M parameters for the 384-d MiniLM teacher, 3.84M for 768-d mpnet.
# The earlier default was (emb_dim, 4) = 4.29M; the small cell matches it on
# retrieval and STS while being 4.5x smaller (see size.py / the paper's
# capacity-scaling appendix).
# EVERY training and evaluation script must build the cell through this helper
# so the saved checkpoint and the eval-time model share one geometry -- a
# mismatch is a silent state_dict load failure.
DEFAULT_NUM_LAYERS = 3


def default_hidden(emb_dim: int) -> int:
    """Hidden width of the default cell for a teacher of dimension ``emb_dim``."""
    return emb_dim // 2


def build_default_cell(name: str, emb_dim: int) -> nn.Module:
    """Instantiate the default-geometry cell for a teacher embedding dim."""
    return build_cell(name, emb_dim, default_hidden(emb_dim), DEFAULT_NUM_LAYERS)


@register_cell("concat_mlp")
class ConcatMLPCell(nn.Module):
    """Baseline: an MLP over ``[token_emb, e_prev]`` predicting the shift."""

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(emb_dim * 2, hidden_dim), nn.GELU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, emb_dim)]
        self.net = nn.Sequential(*layers)
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, token_emb: torch.Tensor, e_prev: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([token_emb, e_prev], dim=-1))


@register_cell("gru")
class GRUUpdateCell(nn.Module):
    """A standard ``nn.GRUCell`` re-expressed as a shift predictor.

    The GRU produces the next state directly; we return ``e_new - e_prev`` so it
    plugs into the same residual update rule as the other cells.
    """

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        # Project token into the state space, then run a GRU whose hidden size
        # equals the embedding dim so the state lives in the embedding space.
        self.in_proj = nn.Linear(emb_dim, emb_dim)
        self.gru = nn.GRUCell(emb_dim, emb_dim)

    def forward(self, token_emb: torch.Tensor, e_prev: torch.Tensor) -> torch.Tensor:
        e_new = self.gru(self.in_proj(token_emb), e_prev)
        return e_new - e_prev


@register_cell("gated_residual")
class GatedResidualCell(nn.Module):
    """Gated candidate update ``g * cand + (1 - g) * e_prev`` (from the original
    from-scratch ISE cell), returned as a shift."""

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        in_dim = emb_dim * 2
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        layers += [nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, emb_dim)]
        self.candidate_net = nn.Sequential(*layers)
        self.gate_net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, emb_dim), nn.Sigmoid(),
        )

    def forward(self, token_emb: torch.Tensor, e_prev: torch.Tensor) -> torch.Tensor:
        x = torch.cat([token_emb, e_prev], dim=-1)
        candidate = self.candidate_net(x)
        gate = self.gate_net(x)
        e_new = gate * candidate + (1 - gate) * e_prev
        return e_new - e_prev


class _FiLMBlock(nn.Module):
    """Feature-wise linear modulation block used by :class:`GatedFiLMCell`."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(hidden_dim, 2 * hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(ctx).chunk(2, dim=-1)
        x_mod = self.norm(x) * (1 + scale) + shift
        return x + self.ff(x_mod)


@register_cell("gated_film")
class GatedFiLMCell(nn.Module):
    """FiLM-gated cell (the proposed ISE update): the context ``e_prev``
    modulates the token pathway via per-feature scale/shift.
    """

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.ctx_net = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.tok_net = nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.blocks = nn.ModuleList([_FiLMBlock(hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, emb_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, token_emb: torch.Tensor, e_prev: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx_net(e_prev)
        x = self.tok_net(token_emb)
        for block in self.blocks:
            x = block(x, ctx)
        return self.out(x)


# --- linear-time sequence blocks as one-step update cells --------------------
# Reviewer request (ARR 2026 August, uFGZ): Mamba2-, RWKV-, and xLSTM-style
# blocks "re-trained along with GRU". Each cell below runs the core recurrence
# of its family for exactly one step. The cell is Markovian in the teacher
# embedding (Appendix I), so the family's recurrent state is *lifted* from
# ``e_prev`` rather than carried, and the block's output is projected back to
# a shift. Token and context enter through the same two-layer SiLU encoders as
# the FiLM cell, blocks are stacked ``num_layers`` deep with residuals, and the
# output layer is zero-initialized so every cell starts as an exact
# copy-forward. Widths are chosen per cell to match the 0.96M FiLM budget (see
# cells_rollout.py --hidden).


def _two_layer(emb_dim: int, hidden_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(emb_dim, hidden_dim), nn.SiLU(),
                         nn.Linear(hidden_dim, hidden_dim))


class _LiftedSeqCell(nn.Module):
    """Shared skeleton: encoders, a state lift from ``e_prev``, stacked blocks,
    zero-initialized shift head. Subclasses supply ``_block()`` and ``lift_dim``.
    """

    lift_dim: int = 0

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.ctx_net = _two_layer(emb_dim, hidden_dim)
        self.tok_net = _two_layer(emb_dim, hidden_dim)
        self.state_lift = (nn.Linear(emb_dim, self.lift_dim)
                           if self.lift_dim else nn.Identity())
        self.blocks = nn.ModuleList([self._block(hidden_dim) for _ in range(num_layers)])
        self.out = nn.Linear(hidden_dim, emb_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def _block(self, hidden_dim: int) -> nn.Module:  # pragma: no cover - abstract
        raise NotImplementedError

    def forward(self, token_emb: torch.Tensor, e_prev: torch.Tensor) -> torch.Tensor:
        ctx = self.ctx_net(e_prev)
        lifted = self.state_lift(e_prev)
        x = self.tok_net(token_emb)
        for block in self.blocks:
            x = block(x, ctx, lifted)
        return self.out(x)


class _SSDBlock(nn.Module):
    """One step of a Mamba2-style selective state-space (SSD) block.

    ``s' = exp(dt * A) * s + (dt * x) B^T``, ``y = s' C + D * x``, gated by
    ``silu(z)``; ``dt``, ``B``, ``C``, ``z`` are input-dependent. The
    ``hidden x d_state`` SSM state is lifted as the outer product of the
    context vector and a ``d_state``-dim projection of ``e_prev``.
    """

    def __init__(self, hidden_dim: int, d_state: int) -> None:
        super().__init__()
        self.d_state = d_state
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(2 * hidden_dim, 3 * hidden_dim + 2 * d_state)
        self.dt_bias = nn.Parameter(torch.zeros(hidden_dim))
        self.A_log = nn.Parameter(torch.zeros(hidden_dim))       # A = -exp(A_log)
        self.D = nn.Parameter(torch.ones(hidden_dim))
        # Mamba2's gated RMSNorm before out_proj. Without it the read-out
        # ``s' C`` scales with |ctx| * |u| ~ |e_prev|^2, and the state diverged
        # at rollout step 62 in the first full run (26 Sep).
        self.y_norm = nn.RMSNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        xs, z, dt_raw, b, c = torch.split(
            self.proj(torch.cat([h, ctx], dim=-1)),
            [h.size(-1), h.size(-1), h.size(-1), self.d_state, self.d_state], dim=-1)
        dt = nn.functional.softplus(dt_raw + self.dt_bias)
        decay = torch.exp(dt * -torch.exp(self.A_log))                # (B, H)
        state = ctx.unsqueeze(-1) * u.unsqueeze(-2)                    # (B, H, N)
        state = decay.unsqueeze(-1) * state + (dt * xs).unsqueeze(-1) * b.unsqueeze(-2)
        y = (state * c.unsqueeze(-2)).sum(-1) + self.D * xs
        return x + self.out(self.y_norm(y * nn.functional.silu(z)))


@register_cell("mamba2_ssd")
class Mamba2SSDCell(_LiftedSeqCell):
    """Mamba2-style SSD blocks as the update cell (``d_state`` = 16)."""

    lift_dim = 16

    def _block(self, hidden_dim: int) -> nn.Module:
        return _SSDBlock(hidden_dim, self.lift_dim)


class _RWKV7Block(nn.Module):
    """One step of the RWKV-7 time-mixing recurrence.

    Per head, ``S' = S (diag(w) - k_hat (a * k_hat)^T) + v k^T`` and
    ``o = S' r``, with decay ``w``, in-context learning rate ``a``, receptance
    ``r``, key ``k``, value ``v``, and output gate ``g`` all input-dependent.
    The matrix state ``S`` is lifted per head as ``outer(ctx, u)``.
    """

    def __init__(self, hidden_dim: int, n_heads: int) -> None:
        super().__init__()
        if hidden_dim % n_heads:
            raise ValueError(f"hidden_dim {hidden_dim} not divisible by {n_heads} heads")
        self.n_heads = n_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(2 * hidden_dim, 6 * hidden_dim)       # r, w, k, v, a, g
        self.group_norm = nn.GroupNorm(n_heads, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        bsz, hid = x.shape
        heads, d = self.n_heads, hid // self.n_heads
        r, w_raw, k, v, a_raw, g = self.proj(torch.cat([self.norm(x), ctx], dim=-1)).chunk(6, dim=-1)
        w = torch.exp(-torch.exp(w_raw))                               # decay in (0, 1)
        a = torch.sigmoid(a_raw)
        k_hat = nn.functional.normalize(k.view(bsz, heads, d), dim=-1)
        view = lambda t: t.view(bsz, heads, d, 1)                      # column vectors
        state = view(ctx) * u.view(bsz, heads, 1, d)                   # (B, heads, d, d)
        state = (state * w.view(bsz, heads, 1, d)
                 - (state @ view(k_hat)) @ (a.view(bsz, heads, d) * k_hat).view(bsz, heads, 1, d)
                 + view(v) @ k.view(bsz, heads, 1, d))
        o = (state @ view(r)).view(bsz, hid)
        return x + self.out(self.group_norm(o) * nn.functional.silu(g))


@register_cell("rwkv7")
class RWKV7Cell(_LiftedSeqCell):
    """RWKV-7 time-mixing blocks as the update cell (head size 8)."""

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int) -> None:
        self.lift_dim = hidden_dim
        super().__init__(emb_dim, hidden_dim, num_layers)

    def _block(self, hidden_dim: int) -> nn.Module:
        return _RWKV7Block(hidden_dim, n_heads=max(1, hidden_dim // 8))


class _SLSTMBlock(nn.Module):
    """One step of an xLSTM sLSTM cell with exponential gating.

    ``c' = f c + i z``, ``n' = f n + i``, ``h = o * c' / n'`` with
    ``i = exp(i_raw - m)``, ``f = exp(f_raw - m)``, stabilizer
    ``m = max(f_raw, i_raw)``. The cell state ``c`` is lifted from ``e_prev``
    and the normalizer restarts at 1, as after a re-grounding.
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.proj = nn.Linear(2 * hidden_dim, 4 * hidden_dim)       # z, i, f, o
        # xLSTM's sLSTM block normalizes the cell output head-wise before the
        # projection; it also keeps the lifted cell state from feeding back
        # into the shift unbounded.
        self.group_norm = nn.GroupNorm(max(1, hidden_dim // 8), hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor, c_prev: torch.Tensor) -> torch.Tensor:
        z_raw, i_raw, f_raw, o_raw = self.proj(torch.cat([self.norm(x), ctx], dim=-1)).chunk(4, dim=-1)
        m = torch.maximum(f_raw, i_raw)
        i_gate = torch.exp(i_raw - m)
        f_gate = torch.exp(f_raw - m)
        c_new = f_gate * c_prev + i_gate * torch.tanh(z_raw)
        n_new = f_gate + i_gate
        h = torch.sigmoid(o_raw) * c_new / n_new
        return x + self.out(self.group_norm(h))


@register_cell("xlstm_s")
class SLSTMCell(_LiftedSeqCell):
    """xLSTM sLSTM blocks as the update cell."""

    def __init__(self, emb_dim: int, hidden_dim: int, num_layers: int) -> None:
        self.lift_dim = hidden_dim
        super().__init__(emb_dim, hidden_dim, num_layers)

    def _block(self, hidden_dim: int) -> nn.Module:
        return _SLSTMBlock(hidden_dim)


__all__ = ["CELL_REGISTRY", "register_cell", "build_cell",
           "ConcatMLPCell", "GRUUpdateCell", "GatedResidualCell", "GatedFiLMCell",
           "Mamba2SSDCell", "RWKV7Cell", "SLSTMCell"]
