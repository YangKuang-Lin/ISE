"""Fair inference-speed benchmark: ISE vs GRU vs Transformer (cached / uncached).

Fixes the original benchmark's two problems: (1) it now includes a Transformer
with a proper **KV cache**, the standard efficient-inference path, and (2) it
sweeps sequence length and batch size instead of a single setting. Wall-clock is
reported alongside peak memory. Models are randomly initialized -- weights do
not affect throughput, only architecture does.

Every (model, batch, length) cell is measured in its own subprocess. That is
not fastidiousness; three earlier in-process protocols each produced a wrong
table, and the reasons are worth recording because they are easy to repeat:

  * Warming the no-cache baseline at 64 tokens but timing it at 2048 left the
    first timed repetition paying a cold 1.3 GB allocation: 15.7 tok/s cold
    against ~310 tok/s warm at batch 4.
  * Interleaving repetitions across the grid, meant to dilute transient system
    load, was worse: the intervening cells free and re-grow that pool, so every
    repetition paid the cold cost and best-of-N could not recover it. It
    reported 74 tok/s for the same cell.
  * Keeping all four models resident in one process contaminated both axes.
    Throughput: the no-cache model's allocation churn fragments the caching
    allocator, and a cell that reads 10.3k tok/s in the grid reads 14.3k alone.
    Memory: ``max_memory_allocated`` counts the *other three* models' resident
    weights, so ISE appeared to occupy 112.9 MB when it allocates 26.7 MB.

One process per cell removes all three at the cost of a CUDA init per cell.

Usage:
    python speed.py                       # default sweep
    python speed.py --gen 4096 --batch 1 8 32
    python speed.py --cell ISE --batch 4 --gen 512    # single-cell worker
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch

from experiments.encoders import (
    GRUCausalLM,
    ISECausalLM,
    TFCausalLMCached,
    TFCausalLMNoCache,
)
from experiments.utils import count_params, get_device, log_environment, save_results, set_seed

MODELS = ["ISE", "GRU", "TF_cached", "TF_nocache"]


@dataclass(frozen=True)
class SpeedConfig:
    vocab_size: int = 8192
    emb_dim: int = 256
    hidden_dim: int = 256
    nhead: int = 4
    num_layers: int = 4
    ff_dim: int = 512
    prompt_len: int = 50
    # Warm-up is duration-based, not pass-based: a single pass over the
    # shortest configuration is only ~160 ms, which is inside the GPU's clock
    # ramp, and short cells then under-report. We spin until this many seconds
    # have elapsed (always at least one pass, so slow cells are not repeated).
    # A longer warm-up is not better: at 3 s the GPU leaves boost clocks and
    # throughput drops (batch 4, 512 tokens: 14.0k -> 9.6k), so this is set to
    # reach a repeatable state, not a thermally saturated one.
    warmup_seconds: float = 1.0
    # Interference only ever makes a repetition slower, so the maximum is the
    # right estimator and more repetitions strictly help. Cheap cells get
    # ``max_repeats``; expensive ones stop early once ``repeat_budget_s`` of
    # wall time is spent, keeping the whole grid to ~15 min.
    min_repeats: int = 3
    max_repeats: int = 9
    repeat_budget_s: float = 15.0
    # Fraction by which the best and median repetition may differ before a
    # cell is flagged as possibly contaminated by background load.
    spread_tolerance: float = 0.15


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _peak_mem_mb(device: str) -> float:
    if device != "cuda":
        return float("nan")
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


@torch.no_grad()
def bench_state_model(model, step_fn, prompt, gen_tokens, device) -> float:
    """Benchmark an O(1) state model (ISE or GRU). Returns tokens/sec."""
    model.eval()
    state = None
    for i in range(prompt.size(1)):
        _, state = step_fn(model, prompt[:, i], state)
    cur = prompt[:, -1]
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        logits, state = step_fn(model, cur, state)
        cur = logits.argmax(dim=-1)
    _sync(device)
    return gen_tokens * prompt.size(0) / (time.perf_counter() - t0)


@torch.no_grad()
def bench_tf_cached(model: TFCausalLMCached, prompt, gen_tokens, device) -> float:
    model.eval()
    logits, caches = model.prefill(prompt)
    cur = logits.argmax(dim=-1)
    pos = prompt.size(1)
    _sync(device)
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        logits, caches = model.step(cur, caches, pos)
        cur = logits.argmax(dim=-1)
        pos += 1
    _sync(device)
    return gen_tokens * prompt.size(0) / (time.perf_counter() - t0)


@torch.no_grad()
def bench_tf_nocache(model: TFCausalLMNoCache, prompt, gen_tokens, device) -> float:
    """Re-encode the whole prefix every step (no KV cache).

    The buffer is preallocated and sliced rather than grown with ``torch.cat``.
    Growing it allocated a new tensor on each of the 2048 steps, which
    fragmented the caching allocator badly enough to fail a driver allocation
    with only ~0.6 GB of a 12 GB card in use (Windows has no
    ``expandable_segments`` support to defragment). Slicing is also the fairer
    baseline: it charges this policy for its quadratic attention, which is the
    point of the comparison, not for repeated buffer reallocation.
    """
    model.eval()
    n0 = prompt.size(1)
    seq = torch.empty(prompt.size(0), n0 + gen_tokens,
                      dtype=prompt.dtype, device=prompt.device)
    seq[:, :n0] = prompt
    _sync(device)
    t0 = time.perf_counter()
    for i in range(gen_tokens):
        logits = model(seq[:, :n0 + i])
        seq[:, n0 + i] = logits.argmax(dim=-1)
    _sync(device)
    return gen_tokens * prompt.size(0) / (time.perf_counter() - t0)


def _ise_step(model, tok, state):
    return model.step(tok, state if state is not None else _zeros(model, tok))


def _zeros(model, tok):
    return torch.zeros(tok.size(0), model.emb_dim, device=tok.device)


def _gru_step(model, tok, state):
    return model.step(tok, state)


def build_one(cfg: SpeedConfig, name: str, device: str):
    """Build only the model under test, so memory is attributed to it alone."""
    if name == "ISE":
        m = ISECausalLM(cfg.vocab_size, cfg.emb_dim, cfg.hidden_dim, cfg.num_layers).to(device)
        return m, (lambda p, g: bench_state_model(m, _ise_step, p, g, device))
    if name == "GRU":
        m = GRUCausalLM(cfg.vocab_size, cfg.emb_dim, cfg.hidden_dim, cfg.num_layers).to(device)
        return m, (lambda p, g: bench_state_model(m, _gru_step, p, g, device))
    if name == "TF_cached":
        m = TFCausalLMCached(cfg.vocab_size, cfg.emb_dim, cfg.nhead, cfg.num_layers, cfg.ff_dim).to(device)
        return m, (lambda p, g: bench_tf_cached(m, p, g, device))
    m = TFCausalLMNoCache(cfg.vocab_size, cfg.emb_dim, cfg.nhead, cfg.num_layers, cfg.ff_dim).to(device)
    return m, (lambda p, g: bench_tf_nocache(m, p, g, device))


def run_cell(cfg: SpeedConfig, name: str, batch: int, gen: int) -> Dict:
    """Measure a single cell. Intended to be the whole of a fresh process."""
    device = get_device()
    set_seed(42)
    model, fn = build_one(cfg, name, device)
    prompt = torch.randint(0, cfg.vocab_size, (batch, cfg.prompt_len), device=device)

    # Duration-based warm-up: reach steady clocks and a warm allocator.
    t0 = time.perf_counter()
    while True:
        fn(prompt, gen)
        if time.perf_counter() - t0 >= cfg.warmup_seconds:
            break

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    reps: List[float] = []
    t1 = time.perf_counter()
    while len(reps) < cfg.max_repeats:
        reps.append(fn(prompt, gen))
        if len(reps) >= cfg.min_repeats and time.perf_counter() - t1 >= cfg.repeat_budget_s:
            break
    reps.sort()
    return {
        "model": name, "batch": batch, "gen_tokens": gen,
        "tok_per_s": round(reps[-1], 1),
        "median_tok_per_s": round(reps[len(reps) // 2], 1),
        "reps": [round(r, 1) for r in reps],
        "peak_mb": round(_peak_mem_mb(device), 1),
        "reserved_mb": round(torch.cuda.max_memory_reserved() / 2 ** 20, 1)
        if device == "cuda" else float("nan"),
        "params": count_params(model),
    }


def run(cfg: SpeedConfig, gen_list: List[int], batch_list: List[int]) -> Dict:
    """Parent: spawn one subprocess per cell and assemble the grid.

    The parent deliberately does not touch CUDA, so no context of its own
    competes with the workers. A failing cell is recorded and skipped rather
    than aborting: one bad cell used to discard a whole 15-minute grid.
    """
    cells: Dict[tuple, Dict] = {}
    failed: List[str] = []
    for batch in batch_list:
        for gen in gen_list:
            for name in MODELS:
                cmd = [sys.executable, str(Path(__file__).resolve()),
                       "--cell", name, "--batch", str(batch), "--gen", str(gen)]
                proc = subprocess.run(cmd, capture_output=True, text=True)
                lines = [ln for ln in proc.stdout.splitlines() if ln.startswith("{")]
                if proc.returncode != 0 or not lines:
                    tail = (proc.stderr or proc.stdout).strip().splitlines()
                    reason = tail[-1][:120] if tail else f"exit {proc.returncode}"
                    failed.append(f"b{batch}/g{gen}/{name}: {reason}")
                    print(f"  {name:<11} b{batch}/g{gen}: FAILED -- {reason}")
                    continue
                cells[(batch, gen, name)] = json.loads(lines[-1])
                print(f"  {name:<11} b{batch}/g{gen}: "
                      f"{cells[(batch, gen, name)]['tok_per_s']:>9.1f} tok/s  "
                      f"{cells[(batch, gen, name)]['peak_mb']:>7.1f} MB")

    rows, suspect = [], []
    for batch in batch_list:
        for gen in gen_list:
            row: Dict = {"batch": batch, "gen_tokens": gen}
            for name in MODELS:
                c = cells.get((batch, gen, name))
                if c is None:
                    continue
                row[f"{name}_tok_per_s"] = c["tok_per_s"]
                row[f"{name}_peak_mb"] = c["peak_mb"]
                row[f"{name}_reserved_mb"] = c["reserved_mb"]
                row[f"{name}_reps"] = c["reps"]
                best, med = c["tok_per_s"], c["median_tok_per_s"]
                if best > 0 and (best - med) / best > cfg.spread_tolerance:
                    suspect.append(f"b{batch}/g{gen}/{name}: {c['reps']}")
            if "ISE_tok_per_s" in row and "TF_cached_tok_per_s" in row:
                row["speedup_vs_cached"] = round(
                    row["ISE_tok_per_s"] / row["TF_cached_tok_per_s"], 2)
            if "ISE_tok_per_s" in row and "TF_nocache_tok_per_s" in row:
                row["speedup_vs_nocache"] = round(
                    row["ISE_tok_per_s"] / row["TF_nocache_tok_per_s"], 2)
            rows.append(row)

    params = {n: c["params"] for (_, _, n), c in cells.items()}
    if failed:
        print(f"\n(!) {len(failed)} cell(s) did not produce a measurement:")
        for f in failed:
            print("   ", f)
    if suspect:
        print(f"\n(!) {len(suspect)} of {len(rows) * len(MODELS)} cells show a wide "
              f"repetition spread, which indicates background load rather than "
              f"model cost; re-run these on an idle machine before using them:")
        for s in suspect:
            print("   ", s)
    else:
        print("\nAll cells within tolerance: no evidence of interference.")

    return {"config": cfg, "warmup_mode": "subprocess", "params": params,
            "device": "cuda", "rows": rows, "suspect_cells": suspect,
            "failed_cells": failed}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8])
    ap.add_argument("--out", type=str, default="experiments/results/speed.json")
    # Worker mode: measure one cell and print it as JSON.
    ap.add_argument("--cell", type=str, default=None, choices=MODELS)
    args = ap.parse_args()

    cfg = SpeedConfig()
    if args.cell:
        print(json.dumps(run_cell(cfg, args.cell, args.batch[0], args.gen[0])))
        return
    save_results(args.out, run(cfg, args.gen, args.batch))


if __name__ == "__main__":
    main()
