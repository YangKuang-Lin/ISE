"""Copy-forward null horizons and residual domain-stability (review round 6).

Two cheap measurements the reviews identified as decisive for the
teacher-scaling analysis:

  1. Per-teacher copy-forward (null) horizon: how long does simply HOLDING
     the seed embedding stay above each fidelity threshold? The cell's
     horizon is only a finding to the extent it is not a constant multiple
     of this training-free null. Computed for all five teachers, with the
     cell horizons (from the existing distill JSONs) and their ratio, at
     thresholds {0.95, 0.90, 0.85, 0.80}.
  2. Residual domain-stability: the teacher's copy-forward residual is the
     proposed training-free diagnostic, measured on WikiText; is it stable
     when measured on other domains (SciFact abstracts, FiQA posts)?

No training; teacher encodes only. Runs in ~20-30 minutes.

Usage:
    python nullhorizon.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import get_device, log_environment, save_results, set_seed

MIN_TOKENS, ROLLOUT_K = 10, 64
THRESHOLDS = [0.95, 0.90, 0.85, 0.80]
N_DOCS = 100
N_RESIDUAL_DOCS = 100
RESIDUAL_STEPS = 16

TEACHERS = [
    ("minilm", "all-MiniLM-L6-v2", "distill.json"),
    ("e5", "intfloat/e5-base-v2", "distill_e5.json"),
    ("bge", "BAAI/bge-base-en-v1.5", "distill_bge.json"),
    ("mpnet", "all-mpnet-base-v2", "distill_mpnet.json"),
    ("gte", "thenlper/gte-large", "distill_gte.json"),
]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def null_curve(teacher: Teacher, docs: List[str]) -> Dict[str, float]:
    """Mean cos(E_seed, E_{seed+t}) over documents, t = 1..ROLLOUT_K."""
    per: Dict[int, List[float]] = {t: [] for t in range(1, ROLLOUT_K + 1)}
    used = 0
    for doc in docs:
        ids = teacher.token_ids(doc, 80 + ROLLOUT_K + 1)
        if len(ids) < MIN_TOKENS + ROLLOUT_K + 1:
            continue
        ks = list(range(MIN_TOKENS, MIN_TOKENS + ROLLOUT_K + 1))
        true = teacher.encode_prefixes(ids, ks)
        for t in range(1, ROLLOUT_K + 1):
            per[t].append(_cos(true[0], true[t]))
        used += 1
        if used >= N_DOCS:
            break
    return {str(t): float(np.mean(v)) for t, v in per.items() if v}


def horizon(curve: Dict[str, float], thr: float) -> int:
    """First step below thr; ROLLOUT_K + 1 marks right-censoring."""
    return next((int(t) for t in sorted(curve, key=int)
                 if curve[t] < thr), ROLLOUT_K + 1)


def residual_on(teacher: Teacher, docs: List[str], n_docs: int) -> float:
    """Mean copy-forward residual 1 - cos(E_n, E_{n+1}) on a corpus."""
    vals, used = [], 0
    for doc in docs:
        ids = teacher.token_ids(doc, MIN_TOKENS + RESIDUAL_STEPS + 1)
        if len(ids) < MIN_TOKENS + RESIDUAL_STEPS + 1:
            continue
        ks = list(range(MIN_TOKENS, MIN_TOKENS + RESIDUAL_STEPS + 1))
        true = teacher.encode_prefixes(ids, ks)
        vals.extend(1 - _cos(true[i], true[i + 1]) for i in range(len(ks) - 1))
        used += 1
        if used >= n_docs:
            break
    return float(np.mean(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", type=str, default="experiments/results")
    ap.add_argument("--out", type=str,
                    default="experiments/results/nullhorizon.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()

    # Same document protocol as the distillation evaluations.
    docs = load_wikitext_docs("wikitext-2-raw-v1", 6000, 40)[4000:]

    from beir import load_beir
    _, scifact_texts, _, _ = load_beir("scifact", 60000)
    _, fiqa_texts, _, _ = load_beir("fiqa", 60000)

    results: Dict[str, Dict] = {}
    for tag, name, distill_json in TEACHERS:
        dpath = Path(args.results_dir) / distill_json
        if not dpath.exists():
            print(f"  SKIP {tag}: {distill_json} missing")
            continue
        seeds = json.loads(dpath.read_text(encoding="utf-8"))["main"]["per_seed"]
        steps = sorted(int(k) for k in seeds[0]["rollout_drift"])
        cell_curve = {str(k): float(np.mean([s["rollout_drift"][str(k)]
                                             for s in seeds])) for k in steps}

        teacher = Teacher(name, device=device)
        ncurve = null_curve(teacher, docs)
        entry: Dict = {"model": name, "null_curve": ncurve,
                       "horizons": {}, "residuals": {}}
        for thr in THRESHOLDS:
            hn, hc = horizon(ncurve, thr), horizon(cell_curve, thr)
            entry["horizons"][str(thr)] = {
                "null": hn, "cell": hc,
                "ratio": (hc / hn) if hn <= ROLLOUT_K and hc <= ROLLOUT_K
                else None,
            }
        entry["residuals"] = {
            "wikitext": residual_on(teacher, docs, N_RESIDUAL_DOCS),
            "scifact": residual_on(teacher, scifact_texts, N_RESIDUAL_DOCS),
            "fiqa": residual_on(teacher, fiqa_texts, N_RESIDUAL_DOCS),
        }
        results[tag] = entry
        h90 = entry["horizons"]["0.9"]
        r = entry["residuals"]
        print(f"  {tag:>6}: null h={h90['null']:>3} cell h={h90['cell']:>3} "
              f"ratio={h90['ratio'] if h90['ratio'] else 'cens.'}  "
              f"residual wiki={r['wikitext'] * 100:.2f} "
              f"scifact={r['scifact'] * 100:.2f} fiqa={r['fiqa'] * 100:.2f}")
        del teacher

    save_results(args.out, {"thresholds": THRESHOLDS, "n_docs": N_DOCS,
                            "rollout_k": ROLLOUT_K, "teachers": results})


if __name__ == "__main__":
    main()
