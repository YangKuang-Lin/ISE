"""Serving-throughput: when is the teacher-call budget the binding constraint?

Review round 6: the GPU wall-clock comparison favors a denser stale cadence,
so reviewers asked in what deployment the matched-*calls* axis (where the cell
wins) is the operative one. Answer: multi-stream serving, where one teacher
GPU is shared by many concurrent token streams and every teacher re-encode
competes for the same accelerator. There the number of streams one device can
sustain is set by teacher calls per token, not by single-stream latency.

Protocol (all measured, no simulation of the models themselves):

  1. Teacher capacity C_T: batched re-encodes/s of 74-token prefixes
     (the paper's n0 + 64 query protocol), across batch sizes.
  2. Cell capacity C_C: batched (token, state) -> shift updates/s.
  3. Sustainable streams per policy at a per-stream token rate rho:
         S*(policy) = min( C_T / (rho * f_call), C_C / (rho * f_cell) )
     with steady-state call fractions f_call = 1 (per-token), 1/4 (stale-4),
     1/8 (stale-8 and hybrid-8), ~1/64 (unreset ISE, seed only).
  4. Closed-loop validation: one simulated second of hybrid-8 load at
     0.5 * S* is executed for real (interleaved teacher batches + cell
     batches) and its measured utilization compared with the prediction.

Usage:
    python serving.py --ckpt <ckpt>
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np
import torch

from experiments.cells import build_default_cell
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import get_device, log_environment, save_results, set_seed

PREFIX_TOKENS = 74          # n0 + 64, the query protocol's working length
RHO = 5.0                   # tokens per second per stream (typing/ASR-rate)
TEACHER_BATCHES = [1, 8, 32, 128, 256]
CELL_BATCHES = [1, 256, 2048, 8192]
POLICIES = {
    # steady-state (call fraction, cell fraction) per emitted token
    "per_token": (1.0, 0.0),
    "stale4": (0.25, 0.0),
    "stale8": (0.125, 0.0),
    "hybrid8": (0.125, 0.875),
    "ise": (1.0 / 64.0, 1.0 - 1.0 / 64.0),
}


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def measure_teacher(teacher: Teacher, texts: List[str], device: str) -> Dict:
    """Re-encodes/s of 74-token prefixes at each batch size (best of 3)."""
    out = {}
    for bs in TEACHER_BATCHES:
        batch = (texts * ((bs // len(texts)) + 1))[:max(bs, 64)]
        teacher.model.encode(batch[:bs], batch_size=bs)          # warm-up
        best = float("inf")
        for _ in range(3):
            _sync(device)
            t0 = time.perf_counter()
            teacher.model.encode(batch, batch_size=bs)
            _sync(device)
            best = min(best, (time.perf_counter() - t0) / len(batch))
        out[str(bs)] = 1.0 / best
        print(f"  teacher bs={bs:>4}: {out[str(bs)]:,.0f} re-encodes/s")
    return out


def measure_cell(model, dim: int, device: str) -> Dict:
    """Batched cell updates/s at each batch size (best of 3)."""
    out = {}
    with torch.no_grad():
        for bs in CELL_BATCHES:
            tok = torch.randn(bs, dim, device=device)
            e = torch.randn(bs, dim, device=device)
            model(tok, e)                                        # warm-up
            reps = max(1, 4096 // bs)
            best = float("inf")
            for _ in range(3):
                _sync(device)
                t0 = time.perf_counter()
                for _ in range(reps):
                    model(tok, e)
                _sync(device)
                best = min(best, (time.perf_counter() - t0) / (reps * bs))
            out[str(bs)] = 1.0 / best
            print(f"  cell    bs={bs:>4}: {out[str(bs)]:,.0f} updates/s")
    return out


def closed_loop(teacher: Teacher, model, texts: List[str], dim: int,
                streams: int, f_call: float, f_cell: float,
                device: str) -> float:
    """Execute one simulated second of load; return measured utilization."""
    n_tokens = int(streams * RHO)
    n_calls = int(round(n_tokens * f_call))
    n_cell = int(round(n_tokens * f_cell))
    call_texts = (texts * ((n_calls // len(texts)) + 1))[:n_calls]
    _sync(device)
    t0 = time.perf_counter()
    if call_texts:
        teacher.model.encode(call_texts, batch_size=128)
    with torch.no_grad():
        done = 0
        while done < n_cell:
            b = min(4096, n_cell - done)
            model(torch.randn(b, dim, device=device),
                  torch.randn(b, dim, device=device))
            done += b
    _sync(device)
    return time.perf_counter() - t0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="experiments/results/ckpt/gated_film_unroll1.pt")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--out", type=str, default="experiments/results/serving.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    model = build_default_cell("gated_film", teacher.emb_dim).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    docs = load_wikitext_docs("wikitext-2-raw-v1", 400, 60)
    texts = []
    for d in docs:
        ids = teacher.token_ids(d, PREFIX_TOKENS)
        if len(ids) >= PREFIX_TOKENS:
            texts.append(teacher.tokenizer.decode(ids[:PREFIX_TOKENS]))
        if len(texts) >= 64:
            break

    print("teacher capacity:")
    t_cap = measure_teacher(teacher, texts, device)
    print("cell capacity:")
    c_cap = measure_cell(model, teacher.emb_dim, device)
    C_T = max(t_cap.values())
    C_C = max(c_cap.values())

    policies = {}
    for name, (f_call, f_cell) in POLICIES.items():
        s_teacher = C_T / (RHO * f_call) if f_call else float("inf")
        s_cell = C_C / (RHO * f_cell) if f_cell else float("inf")
        s_star = min(s_teacher, s_cell)
        policies[name] = {"f_call": f_call, "f_cell": f_cell,
                          "streams": s_star,
                          "bound": "teacher" if s_teacher <= s_cell else "cell"}
        print(f"  {name:>10}: S* = {s_star:>10,.0f} streams "
              f"({policies[name]['bound']}-bound)")

    # Closed-loop validation at half the predicted hybrid-8 capacity.
    s_half = int(policies["hybrid8"]["streams"] / 2)
    f_call, f_cell = POLICIES["hybrid8"]
    closed_loop(teacher, model, texts, teacher.emb_dim, s_half,
                f_call, f_cell, device)                          # warm-up
    utils = [closed_loop(teacher, model, texts, teacher.emb_dim, s_half,
                         f_call, f_cell, device) for _ in range(3)]
    measured = float(min(utils))
    print(f"  closed-loop hybrid-8 @ {s_half:,} streams: "
          f"utilization {measured:.3f} (predicted 0.500)")

    save_results(args.out, {
        "prefix_tokens": PREFIX_TOKENS, "rho": RHO,
        "teacher_throughput": t_cap, "cell_throughput": c_cap,
        "teacher_capacity": C_T, "cell_capacity": C_C,
        "policies": policies,
        "streams_multiplier_stale8": policies["stale8"]["streams"]
        / policies["per_token"]["streams"],
        "streams_multiplier_hybrid8": policies["hybrid8"]["streams"]
        / policies["per_token"]["streams"],
        "closed_loop": {"streams": s_half, "predicted_utilization": 0.5,
                        "measured_utilization": measured},
    })


if __name__ == "__main__":
    main()
