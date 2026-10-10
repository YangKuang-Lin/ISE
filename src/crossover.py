"""The regime map: which incremental embedder wins at which stream length.

Review round 8 (reviewer W1, "the scissors"). Aggregated across our own
results, the cell's final-embedding margin over a training-free hold cadence
is large on short queries (SciFact +3.8 nDCG) and statistically absent on long
ones (ArguAna p=0.53, WoW p=0.054) -- yet long streams are exactly where the
cell's constant per-token cost is supposed to be the differentiator. The
reviewer's reading is that the paper defends a method in the corner of the
space where its advantage is smallest.

The answer is not a better defence but a map. This script measures, at a
matched teacher-call budget and on one machine, what each candidate actually
costs and delivers as a function of stream length N:

  restart          teacher re-encodes every token          (quality ceiling)
  stale-8          teacher every 8 tokens, hold in between (training-free floor)
  hybrid-8         the cell between those calls            (ours, O(1))
  causal-8         seeded causal student + KV cache        (O(N), exact prefixes)
  bi-8             seeded bidirectional student            (O(N), re-encodes)

All four budgeted policies spend the identical number of teacher calls, so the
comparison is on wall clock and on fidelity -- both the mean and the worst step
between resets, which is what a consumer reading the embedding mid-stream
experiences.

The headline output is the *crossover length*: the N beyond which the cell's
O(1) update is cheaper per stream than the causal student's O(N) cached step.
Below it, the student is the better choice at equal calls; above it, the cell
is. That sentence is the paper's practical contribution.

Student checkpoints are optional: absent ones are skipped with a warning so
the cell/stale/restart rows still land.

Usage:
    python crossover.py --ckpt <cell seed ckpts...>
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from experiments.cells import build_default_cell
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.students import SeededBiStudent, SeededCausalStudent
from experiments.utils import (
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)
from streaming import _cos, _sync, stream_ise, stream_restart, stream_stale

N0 = 10
REGROUND_K = 8
LENGTHS = [32, 64, 128, 256, 512, 1024]
REPEATS = 3


@torch.no_grad()
def stream_causal_student(model, teacher: Teacher, ids: List[int], n0: int,
                          k: int, device: str) -> Dict:
    """Seeded causal student with a KV cache, re-grounded every ``k`` tokens.

    Between resets each token costs one cached attention pass over a growing
    cache (O(N)); at a reset the anchor is refreshed and the cache rebuilt, and
    the emitted embedding is the teacher's, exactly as in hybrid-``k``.
    """
    rows = torch.tensor(teacher.token_embed(ids), dtype=torch.float32,
                        device=device).unsqueeze(0)
    _sync(device)
    t0 = time.perf_counter()
    anchor = teacher.encode_prefixes(ids, [n0])[0]
    calls, resets = 1, [0]
    seed = torch.tensor(anchor, dtype=torch.float32, device=device).unsqueeze(0)
    emb_t, state = model.prefill(rows[:, :n0], seed=seed)
    embs = [emb_t.squeeze(0).cpu().numpy()]
    for t in range(n0, len(ids)):
        step = t - n0 + 1
        if k and step % k == 0:
            anchor = teacher.encode_prefixes(ids, [t + 1])[0]
            calls += 1
            resets.append(step)
            seed = torch.tensor(anchor, dtype=torch.float32,
                                device=device).unsqueeze(0)
            emb_t, state = model.prefill(rows[:, :t + 1], seed=seed)
            embs.append(anchor)
            continue
        emb_t, state = model.step(rows[:, t], state, t)
        embs.append(emb_t.squeeze(0).cpu().numpy())
    _sync(device)
    return {"time_s": time.perf_counter() - t0, "teacher_calls": calls,
            "embs": np.array(embs), "steps": len(embs), "resets": resets}


@torch.no_grad()
def stream_bi_student(model, teacher: Teacher, ids: List[int], n0: int,
                      k: int, device: str) -> Dict:
    """Seeded bidirectional student: re-encodes the whole prefix every token."""
    rows = torch.tensor(teacher.token_embed(ids), dtype=torch.float32,
                        device=device).unsqueeze(0)
    _sync(device)
    t0 = time.perf_counter()
    anchor = teacher.encode_prefixes(ids, [n0])[0]
    calls, resets = 1, [0]
    seed = torch.tensor(anchor, dtype=torch.float32, device=device).unsqueeze(0)
    embs = [model(rows[:, :n0], seed=seed).squeeze(0).cpu().numpy()]
    for t in range(n0, len(ids)):
        step = t - n0 + 1
        if k and step % k == 0:
            anchor = teacher.encode_prefixes(ids, [t + 1])[0]
            calls += 1
            resets.append(step)
            seed = torch.tensor(anchor, dtype=torch.float32,
                                device=device).unsqueeze(0)
            embs.append(anchor)
            continue
        embs.append(model(rows[:, :t + 1], seed=seed).squeeze(0).cpu().numpy())
    _sync(device)
    return {"time_s": time.perf_counter() - t0, "teacher_calls": calls,
            "embs": np.array(embs), "steps": len(embs), "resets": resets}


def fidelity(run: Dict, truth: np.ndarray) -> Dict[str, float]:
    """Mean and worst-step cosine to the teacher over the whole stream.

    Reset steps are included (they are exact by construction), matching the
    *streaming* fidelity definition of the paper's efficiency section rather
    than the reset-excluding rollout definition of the drift section.
    """
    n = min(len(run["embs"]), len(truth))
    step_fid = [_cos(run["embs"][i], truth[i]) for i in range(n)]
    return {"fid_mean": float(np.mean(step_fid)),
            "fid_min": float(np.min(step_fid))}


def load_student(cls, path: Path, teacher_dim: int, d_model: int, layers: int,
                 ff_dim, device: str, label: str):
    if not path.exists():
        print(f"  [skip] {label}: checkpoint not found ({path.name}); "
              f"run the students stage first.")
        return None
    model = cls(teacher_dim, d_model, layers, ff_dim=ff_dim).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    print(f"  loaded {label} ({count_params(model):,} params)")
    return model


def build_long_stream(teacher: Teacher, docs: List[str], need: int) -> List[int]:
    """Concatenate documents until a single stream of ``need`` tokens exists."""
    ids: List[int] = []
    for doc in docs:
        ids.extend(teacher.token_ids(doc, 512))
        if len(ids) >= need:
            break
    return ids[:need]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"])
    ap.add_argument("--ckpt-dir", type=str, default="experiments/results/ckpt")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    ap.add_argument("--docs", type=int, default=10)
    ap.add_argument("--cpu", action="store_true",
                    help="also time every policy on CPU (slow; ~4x the runtime)")
    ap.add_argument("--out", type=str, default="experiments/results/crossover.json")
    args = ap.parse_args()

    log_environment()
    devices = ["cuda"] if get_device() == "cuda" else ["cpu"]
    if args.cpu and "cpu" not in devices:
        devices.append("cpu")

    ckpt_dir = Path(args.ckpt_dir)
    per_device: Dict[str, List[Dict]] = {}

    for device in devices:
        set_seed(42)
        teacher = Teacher(args.teacher, device=device)
        cell = build_default_cell("gated_film", teacher.emb_dim).to(device)
        cell.load_state_dict(torch.load(args.ckpt[0], map_location=device))
        cell.eval()
        causal = load_student(
            SeededCausalStudent, ckpt_dir / "student_causal_seeded_seed41.pt",
            teacher.emb_dim, 320, 3, None, device, "seeded causal student")
        bi = load_student(
            SeededBiStudent, ckpt_dir / "student_bi_large_seeded_seed41.pt",
            teacher.emb_dim, 320, 3, None, device, "seeded bi student")

        docs = load_wikitext_docs("wikitext-2-raw-v1", 8000, 40)[4000:]
        rows: List[Dict] = []
        for n in args.lengths:
            streams = []
            pool = docs
            for d in range(args.docs):
                s = build_long_stream(teacher, pool[d * 40:(d + 1) * 40], n)
                if len(s) == n:
                    streams.append(s)
            if not streams:
                print(f"  [skip] N={n}: no stream of that length available.")
                continue
            print(f"\n  --- {device} N={n} ({len(streams)} streams) ---")

            policies = {
                "restart": lambda ids: stream_restart(teacher, ids, N0, device),
                "stale8": lambda ids: stream_stale(teacher, ids, N0, REGROUND_K, device),
                "hybrid8": lambda ids: stream_ise(cell, teacher, ids, N0, device,
                                                  reencode_every=REGROUND_K),
            }
            if causal is not None:
                policies["causal8"] = lambda ids: stream_causal_student(
                    causal, teacher, ids, N0, REGROUND_K, device)
            if bi is not None:
                policies["bi8"] = lambda ids: stream_bi_student(
                    bi, teacher, ids, N0, REGROUND_K, device)

            truths = [stream_restart(teacher, ids, N0, device)["embs"]
                      for ids in streams]
            for name, fn in policies.items():
                for ids in streams[:1]:      # warm-up: exclude allocator costs
                    fn(ids)
                best = None
                for _ in range(REPEATS):
                    runs = [fn(ids) for ids in streams]
                    total = sum(r["time_s"] for r in runs)
                    if best is None or total < best[0]:
                        best = (total, runs)
                total, runs = best
                fids = [fidelity(r, t) for r, t in zip(runs, truths)]
                row = {
                    "device": device, "length": n, "policy": name,
                    "latency_ms": total / len(runs) * 1000.0,
                    "teacher_calls": float(np.mean([r["teacher_calls"] for r in runs])),
                    "fid_mean": float(np.mean([f["fid_mean"] for f in fids])),
                    "fid_min": float(np.mean([f["fid_min"] for f in fids])),
                    "n_streams": len(runs),
                }
                rows.append(row)
                print(f"    {name:>8}: {row['latency_ms']:8.1f} ms  "
                      f"calls={row['teacher_calls']:6.1f}  "
                      f"fid={row['fid_mean']:.4f}  min={row['fid_min']:.4f}")
        per_device[device] = rows

    # The headline: where does O(1) overtake O(N)?
    crossover: Dict[str, Optional[int]] = {}
    for device, rows in per_device.items():
        by = {(r["policy"], r["length"]): r for r in rows}
        lengths = sorted({r["length"] for r in rows})
        for rival in ("causal8", "bi8"):
            hit = None
            for n in lengths:
                a, b = by.get(("hybrid8", n)), by.get((rival, n))
                if a and b and a["latency_ms"] < b["latency_ms"]:
                    hit = n
                    break
            crossover[f"{device}:hybrid8_beats_{rival}"] = hit
    for key, v in crossover.items():
        print(f"  crossover {key}: N >= {v}" if v else
              f"  crossover {key}: not reached within the swept lengths")

    save_results(args.out, {"teacher_model": args.teacher, "n0": N0,
                            "reground_k": REGROUND_K, "repeats": REPEATS,
                            "lengths": args.lengths, "docs": args.docs,
                            "rows": [r for rs in per_device.values() for r in rs],
                            "crossover": crossover})
    print("done.")


if __name__ == "__main__":
    main()
