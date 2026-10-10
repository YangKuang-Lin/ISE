"""Streaming sentence-embedding benchmark (end-to-end latency, memory, fidelity).

This is the benchmark that matches the actual use case, replacing the earlier
autoregressive-*generation* benchmark, which measured a decoder LM and was
therefore a confusing proxy for a sentence-*encoder* updater.

Task: maintain an up-to-date sentence embedding as a document arrives token by
token. For a prefix of length N we compare three policies:

  restart   -- re-encode the whole prefix with the teacher at every token
               (the standard "restart-incremental" baseline; N teacher calls)
  ise       -- one O(1) cell update per token, never re-encoding (1 teacher call)
  hybrid-K  -- ISE updates, with a teacher re-encode every K tokens
  stale-K   -- teacher re-encode every K tokens, held in between (no cell), so
               the learned update and the training-free cadence meet at
               matched wall clock, not only matched calls

We report wall-clock, peak memory, teacher calls, and fidelity (cosine to the
teacher's embedding of the same prefix) so latency and accuracy can be read off
one Pareto curve. Unlike the generation benchmark, everything here uses the same
encoder, so the comparison is apples-to-apples.

Usage:
    python streaming.py --ckpt experiments/results/ckpt/gated_film_unroll1.pt \
                            --lengths 128 256 512 1024
"""

from __future__ import annotations

import argparse
import time
from typing import Dict, List

import numpy as np
import torch

from experiments.cells import build_cell, build_default_cell
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import get_device, log_environment, save_results, set_seed


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def _peak_mb(device: str) -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2) if device == "cuda" else float("nan")


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def _paired_bootstrap(a: List[float], b: List[float], iters: int,
                      seed: int = 0) -> Dict:
    """Paired bootstrap of mean(a) - mean(b) over documents.

    Same protocol as the nDCG margins: resample the pairing unit with
    replacement, two-sided, 95% percentile interval.
    """
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n = len(diff)
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(iters)])
    p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"mean_diff": float(diff.mean()), "p": min(p, 1.0), "n": int(n),
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))]}


def _floor_significance(per_doc_min: Dict[str, List[float]],
                        budgets: List[int], stale_budgets: List[int],
                        iters: int) -> Dict[str, Dict]:
    """Test every hybrid-K floor against every stale-K' floor.

    The claim in the paper is that no hold cadence reaches the cell's floor,
    including cadences that spend more teacher calls, so the comparison is run
    across budgets rather than only at matched K.
    """
    out: Dict[str, Dict] = {}
    for b in budgets:
        for sb in stale_budgets:
            h, s = f"hybrid{b}", f"stale{sb}"
            if h in per_doc_min and s in per_doc_min:
                out[f"{h}_vs_{s}"] = _paired_bootstrap(per_doc_min[h],
                                                       per_doc_min[s], iters)
        if f"hybrid{b}" in per_doc_min and "adaptive" in per_doc_min:
            out[f"adaptive_vs_hybrid{b}"] = _paired_bootstrap(
                per_doc_min["adaptive"], per_doc_min[f"hybrid{b}"], iters)
    return out


@torch.no_grad()
def stream_restart(teacher: Teacher, ids: List[int], n0: int, device: str) -> Dict:
    """Re-encode the full prefix at every step (restart-incremental baseline)."""
    ks = list(range(n0, len(ids) + 1))
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    t0 = time.perf_counter()
    embs = [teacher.encode_prefixes(ids, [k])[0] for k in ks]
    _sync(device)
    return {"time_s": time.perf_counter() - t0, "peak_mb": _peak_mb(device),
            "teacher_calls": len(ks), "embs": np.array(embs), "steps": len(ks)}


@torch.no_grad()
def stream_stale(teacher: Teacher, ids: List[int], n0: int, k: int,
                 device: str) -> Dict:
    """Re-encode every ``k`` tokens and hold the embedding in between (no cell).

    Strictly cheaper per step than hybrid-``k`` (no cell forward, no lookup),
    so it is the training-free policy the learned update must beat at matched
    wall clock, not only at matched teacher calls."""
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    t0 = time.perf_counter()
    held = teacher.encode_prefixes(ids, [n0])[0]
    calls, embs, resets = 1, [held], [0]
    for t in range(n0, len(ids)):
        step = t - n0 + 1
        if step % k == 0:
            held = teacher.encode_prefixes(ids, [t + 1])[0]
            calls += 1
            resets.append(step)
        embs.append(held)
    _sync(device)
    return {"time_s": time.perf_counter() - t0, "peak_mb": _peak_mb(device),
            "teacher_calls": calls, "embs": np.array(embs), "steps": len(embs),
            "resets": resets}


@torch.no_grad()
def stream_ise(model, teacher: Teacher, ids: List[int], n0: int, device: str,
               reencode_every: int = 0, tau: float = 0.0) -> Dict:
    """Incremental updates. ``reencode_every > 0`` re-encodes on a fixed cadence
    (hybrid-K); ``tau > 0`` re-encodes adaptively when the accumulated update
    path length since the last reset exceeds ``tau``."""
    tok_table = teacher.token_embed(ids)                    # (len, D) precomputed lookup
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    _sync(device)
    t0 = time.perf_counter()
    e = torch.tensor(teacher.encode_prefixes(ids, [n0])[0],
                     dtype=torch.float32, device=device).unsqueeze(0)
    calls, s, embs, resets = 1, 0.0, [e.squeeze(0).cpu().numpy()], [0]
    for t in range(n0, len(ids)):
        tok = torch.tensor(tok_table[t], dtype=torch.float32, device=device).unsqueeze(0)
        delta = model(tok, e)
        e = e + delta
        s += float(delta.norm().item())
        step = t - n0 + 1
        if (reencode_every and step % reencode_every == 0) or (tau and s > tau):
            e = torch.tensor(teacher.encode_prefixes(ids, [t + 1])[0],
                             dtype=torch.float32, device=device).unsqueeze(0)
            calls, s = calls + 1, 0.0
            resets.append(step)
        embs.append(e.squeeze(0).cpu().numpy())
    _sync(device)
    return {"time_s": time.perf_counter() - t0, "peak_mb": _peak_mb(device),
            "teacher_calls": calls, "embs": np.array(embs), "steps": len(embs),
            "resets": resets}


def run(cfg_ckpt: str, lengths: List[int], budgets: List[int], n_docs: int,
        n0: int, device: str, repeats: int = 3,
        teacher_name: str = "all-MiniLM-L6-v2", tau: float = 0.6,
        stale_budgets: List[int] = (), concat: bool = False,
        boot_iters: int = 10000) -> Dict:
    teacher = Teacher(teacher_name, device=device)
    model = build_default_cell("gated_film", teacher.emb_dim).to(device)
    model.load_state_dict(torch.load(cfg_ckpt, map_location=device))
    model.eval()

    if concat:
        # Synthetic long documents from consecutive paragraphs, so lengths
        # beyond natural WikiText paragraphs (512/1024) can be benchmarked.
        paragraphs = load_wikitext_docs("wikitext-2-raw-v1", 30000, 30)
        top = max(lengths)
        long_ids, cur = [], ""
        for p in paragraphs:
            cur = (cur + " " + p).strip()
            ids = teacher.token_ids(cur, top + 8)
            if len(ids) >= top:
                long_ids.append(ids[:top])
                cur = ""
                if len(long_ids) >= n_docs:
                    break
        docs = None
    else:
        docs = load_wikitext_docs("wikitext-2-raw-v1", n_docs * 4, 60)
    rows: List[Dict] = []
    floor_tests: Dict[str, Dict] = {}

    for N in lengths:
        if concat:
            usable = [ids[:N] for ids in long_ids][:n_docs]
        else:
            usable = [teacher.token_ids(d, N) for d in docs]
            usable = [i for i in usable if len(i) >= N][:n_docs]
        if not usable:
            print(f"  (skip N={N}: no documents that long)")
            continue

        # Warm up both code paths at this shape before any timing: the first
        # calls pay CUDA context / autotune costs that would otherwise be
        # charged to whichever policy happens to run first.
        for _ in range(2):
            stream_restart(teacher, usable[0], n0, device)
            stream_ise(model, teacher, usable[0], n0, device, 0)
            stream_ise(model, teacher, usable[0], n0, device, budgets[0])

        agg = {"restart": [], **{f"hybrid{b}": [] for b in budgets},
               "adaptive": [], "ise": [],
               **{f"stale{b}": [] for b in stale_budgets}}
        ref_embs = []
        for ids in usable:
            best = min((stream_restart(teacher, ids, n0, device) for _ in range(repeats)),
                       key=lambda r: r["time_s"])
            ref_embs.append(best["embs"])
            agg["restart"].append((best["time_s"], best["peak_mb"], best["teacher_calls"], 1.0, 1.0, 1.0))

        def _inter(step_fids, resets, m):
            keep = [f for i, f in enumerate(step_fids)
                    if i < m and i not in set(resets)]
            return float(np.mean(keep)) if keep else 1.0

        # (label, reencode_every, tau): the adaptive policy resets on path length.
        policies = ([("ise", 0, 0.0)]
                    + [(f"hybrid{b}", b, 0.0) for b in budgets]
                    + [("adaptive", 0, tau)])
        for label, budget, ptau in policies:
            for ids, ref in zip(usable, ref_embs):
                # Best-of-N: the minimum is the least noise-contaminated estimate
                # of the true cost; means here are inflated by scheduler jitter.
                runs = [stream_ise(model, teacher, ids, n0, device, budget, ptau)
                        for _ in range(repeats)]
                r = min(runs, key=lambda x: x["time_s"])
                m = min(len(ref), len(r["embs"]))
                step_fids = [_cos(r["embs"][i], ref[i]) for i in range(m)]
                agg[label].append((r["time_s"], r["peak_mb"], r["teacher_calls"],
                                   float(np.mean(step_fids)), float(np.min(step_fids)),
                                   _inter(step_fids, r["resets"], m)))
        for b in stale_budgets:
            for ids, ref in zip(usable, ref_embs):
                runs = [stream_stale(teacher, ids, n0, b, device)
                        for _ in range(repeats)]
                r = min(runs, key=lambda x: x["time_s"])
                m = min(len(ref), len(r["embs"]))
                step_fids = [_cos(r["embs"][i], ref[i]) for i in range(m)]
                agg[f"stale{b}"].append((r["time_s"], r["peak_mb"], r["teacher_calls"],
                                         float(np.mean(step_fids)), float(np.min(step_fids)),
                                         _inter(step_fids, r["resets"], m)))

        per_doc_min = {}
        for label, vals in agg.items():
            a = np.array([[v[0], v[1], v[2], v[3], v[4], v[5]] for v in vals], dtype=np.float64)
            # The worst-step floor is a mean over documents of a within-document
            # minimum, so its sampling variability is a property of the document
            # sample. Retaining the per-document values is what makes it
            # bootstrappable; aggregating here would throw that away.
            per_doc_min[label] = [float(x) for x in a[:, 4]]
            rows.append({
                "length": N, "policy": label,
                "time_ms": float(a[:, 0].mean() * 1000),
                "peak_mb": float(np.nanmean(a[:, 1])),
                "teacher_calls": float(a[:, 2].mean()),
                "fidelity": float(a[:, 3].mean()),
                "min_fidelity": float(a[:, 4].mean()),
                "min_fidelity_per_doc": per_doc_min[label],
                "fidelity_per_doc": [float(x) for x in a[:, 3]],
                "inter_fidelity": float(a[:, 5].mean()),
                "speedup_vs_restart": None,
            })
        floor_tests[str(N)] = _floor_significance(per_doc_min, budgets,
                                                  stale_budgets, boot_iters)
        base_t = next(r["time_ms"] for r in rows if r["length"] == N and r["policy"] == "restart")
        for r in rows:
            if r["length"] == N:
                r["speedup_vs_restart"] = round(base_t / r["time_ms"], 2)
        for r in [r for r in rows if r["length"] == N]:
            print(f"  N={N:>5} {r['policy']:>9}: {r['time_ms']:8.1f} ms  "
                  f"{r['peak_mb']:7.1f} MB  calls={r['teacher_calls']:6.1f}  "
                  f"fid={r['fidelity']:.4f}  min={r['min_fidelity']:.4f}  "
                  f"speedup={r['speedup_vs_restart']}x")
        for k, v in floor_tests[str(N)].items():
            print(f"    floor {k}: {v['mean_diff']:+.4f}  "
                  f"CI95=[{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  "
                  f"p={v['p']:.4f}")
    return {"rows": rows, "n0": n0, "n_docs": n_docs, "repeats": repeats,
            "ckpt": cfg_ckpt, "floor_significance": floor_tests,
            "bootstrap_iters": boot_iters}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str,
                    default="experiments/results/ckpt/gated_film_unroll1.pt")
    ap.add_argument("--lengths", type=int, nargs="+", default=[128, 256, 512])
    ap.add_argument("--budgets", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--docs", type=int, default=20)
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--device", type=str, default=None,
                    help="force a device (e.g. cpu) to report on-device latency")
    ap.add_argument("--repeats", type=int, default=3,
                    help="timed repetitions per document; the minimum is reported")
    ap.add_argument("--tau", type=float, default=0.6,
                    help="path-length threshold for the adaptive-reset policy")
    ap.add_argument("--stale-budgets", type=int, nargs="*", default=[4, 8],
                    help="stale-cadence intervals to time (re-encode and hold)")
    ap.add_argument("--concat", action="store_true",
                    help="build synthetic long documents by concatenating "
                         "paragraphs (for lengths beyond natural documents)")
    ap.add_argument("--boot-iters", type=int, default=10000,
                    help="paired-bootstrap resamples for the worst-step floor")
    ap.add_argument("--out", type=str, default="experiments/results/streaming.json")
    args = ap.parse_args()

    device = args.device or get_device()
    set_seed(42)
    log_environment()
    res = run(args.ckpt, args.lengths, args.budgets, args.docs, args.n0, device,
              args.repeats, args.teacher, args.tau, args.stale_budgets,
              args.concat, args.boot_iters)
    res["concat"] = args.concat
    save_results(args.out, res)


if __name__ == "__main__":
    main()
