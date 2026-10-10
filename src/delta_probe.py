"""Is the adaptive reset signal content-aware? (#5 step-1 probe.)

The adaptive rule resets when the accumulated update path length
$s_t = \\sum \\lVert\\delta_t\\rVert$ crosses a threshold, where
$\\delta_t = f_\\theta(\\mathbf{e}_t, E_{t-1})$ is how far the cell moves the state
when it reads token $t$. The hypothesis is that this magnitude is a *content*
signal: function words ("the", "a", "to") barely change sentence meaning and
should give small $\\lVert\\delta\\rVert$, while content words and discourse
pivots ("but", "however", negations) change it a lot and should give large
$\\lVert\\delta\\rVert$. If so, the path-length rule re-grounds at semantically
important tokens for free.

This measures, per token and teacher-forced (state fixed to the true
$E_{t-1}$, so drift and position are not confounds):

  * the cell update magnitude  $\\lVert\\delta_t\\rVert$, and
  * the teacher's own shift     $\\lVert E_t - E_{t-1}\\rVert$ (ground-truth
    semantic change at that token),

bucketed by token category (function / pivot / content / punctuation /
subword). A clean separation (pivot, content >> function) supports the
content-aware reframing; a muddy one says the signal is not linguistic and the
adaptive rule should stay a modest drift heuristic.

Usage:
    python delta_probe.py --ckpt ckpt/gated_film_unroll1_seed41.pt ... [3 seeds]
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from experiments.cells import build_cell, build_default_cell
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import (
    aggregate_over_ckpts,
    get_device,
    log_environment,
    save_results,
    set_seed,
)
from distill import DistillConfig

# Function words: high-frequency closed-class tokens that carry little standalone
# meaning. Negation / contrast are deliberately NOT here (they are pivots).
FUNCTION = {
    "the", "a", "an", "of", "to", "in", "for", "on", "at", "by", "with", "and",
    "or", "is", "are", "was", "were", "be", "been", "being", "as", "that",
    "this", "these", "those", "it", "its", "he", "she", "they", "them", "his",
    "her", "their", "we", "you", "i", "from", "up", "out", "then", "so", "than",
    "too", "can", "will", "would", "should", "could", "has", "have", "had",
    "do", "does", "did", "s", "'s", "which", "who", "whom", "there", "here",
    "into", "over", "under", "about", "also", "such", "our", "your", "my",
}

# Discourse pivots: contrast / negation / consequence markers that reorganize
# or reverse sentence meaning.
PIVOT = {
    "but", "however", "although", "though", "yet", "whereas", "nevertheless",
    "nonetheless", "instead", "despite", "unless", "because", "therefore",
    "thus", "hence", "moreover", "furthermore", "conversely", "rather", "not",
    "no", "never", "without", "except", "versus", "while", "nor", "neither",
    "otherwise", "meanwhile", "regardless", "albeit",
}

BUCKETS = ["function", "pivot", "content", "punct", "subword"]


def classify(tok: str) -> str:
    """Bucket a WordPiece token by linguistic role."""
    if tok.startswith("##"):
        return "subword"
    low = tok.lower()
    if not any(c.isalnum() for c in low):
        return "punct"
    if low in PIVOT:
        return "pivot"
    if low in FUNCTION:
        return "function"
    return "content"


@torch.no_grad()
def probe_ckpt(ckpt: str, teacher: Teacher, docs: List[List[int]],
               n0: int, max_tokens: int, device: str) -> Dict:
    """Per-bucket mean cell-|delta| and teacher-|shift|, teacher-forced."""
    model = build_default_cell("gated_film", teacher.emb_dim).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    cell = defaultdict(list)   # bucket -> [||delta||]
    true = defaultdict(list)   # bucket -> [||E_t - E_{t-1}||]
    paired = []                # (||delta||, ||true shift||) for correlation
    for ids in docs:
        ids = ids[:max_tokens]
        ks = list(range(1, len(ids) + 1))
        E = teacher.encode_prefixes(ids, ks)                 # (len, D)
        toks = teacher.tokenizer.convert_ids_to_tokens(ids)
        emb = teacher.token_embed(ids)                       # input-table rows
        for k in range(n0, len(ids)):                        # teacher-forced
            e_prev = torch.tensor(E[k - 1], dtype=torch.float32,
                                  device=device).unsqueeze(0)
            tok_vec = torch.tensor(emb[k], dtype=torch.float32,
                                   device=device).unsqueeze(0)
            delta = float(model(tok_vec, e_prev).norm().item())
            shift = float(np.linalg.norm(E[k] - E[k - 1]))
            b = classify(toks[k])
            cell[b].append(delta)
            true[b].append(shift)
            paired.append((delta, shift))

    def summarize(store):
        return {b: {"mean": float(np.mean(store[b])) if store[b] else float("nan"),
                    "std": float(np.std(store[b])) if store[b] else float("nan"),
                    "n": len(store[b])} for b in BUCKETS}

    from scipy.stats import spearmanr
    dvals = [p[0] for p in paired]
    svals = [p[1] for p in paired]
    rho = float(spearmanr(dvals, svals).correlation)

    cs = summarize(cell)
    ts = summarize(true)
    # Headline ratios of the cell signal.
    def ratio(a, b):
        return (cs[a]["mean"] / cs[b]["mean"]) if cs[b]["mean"] else float("nan")
    return {
        "cell_delta": cs, "teacher_shift": ts,
        "delta_vs_shift_spearman": rho,
        "pivot_over_function": ratio("pivot", "function"),
        "content_over_function": ratio("content", "function"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"],
                    help="one checkpoint per seed; results are averaged")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--n0", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=80)
    ap.add_argument("--docs", type=int, default=300)
    ap.add_argument("--out", type=str, default="experiments/results/delta_probe.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()
    cfg = DistillConfig()
    teacher = Teacher(args.teacher, device=device)

    raw = load_wikitext_docs(cfg.dataset_config, cfg.num_docs + 2000, cfg.min_words)
    docs = []
    for doc in raw[cfg.num_docs:]:
        ids = teacher.token_ids(doc, args.max_tokens)
        if len(ids) >= args.n0 + 5:
            docs.append(ids)
        if len(docs) >= args.docs:
            break
    print(f"  {len(docs)} docs, n0={args.n0}, max_tokens={args.max_tokens}")

    per_ckpt = []
    for ck in args.ckpt:
        r = probe_ckpt(ck, teacher, docs, args.n0, args.max_tokens, device)
        cd = r["cell_delta"]
        print(f"  [{Path(ck).stem}] "
              + "  ".join(f"{b}={cd[b]['mean']:.4f}(n{cd[b]['n']})" for b in BUCKETS)
              + f"  pivot/func={r['pivot_over_function']:.2f}")
        per_ckpt.append(r)

    mean, std = aggregate_over_ckpts(per_ckpt)
    mean["n_ckpt"] = len(args.ckpt)
    mean["std"] = std
    save_results(args.out, {**mean, "per_ckpt": per_ckpt})

    print("\n  === cell update magnitude |delta| by token type (3-seed mean) ===")
    for b in BUCKETS:
        m = mean["cell_delta"][b]
        print(f"    {b:<9} |delta|={m['mean']:.4f}  (teacher shift "
              f"{mean['teacher_shift'][b]['mean']:.4f}, n={m['n']})")
    print(f"\n  pivot/function ratio    = {mean['pivot_over_function']:.2f}")
    print(f"  content/function ratio  = {mean['content_over_function']:.2f}")
    print(f"  |delta| vs teacher-shift Spearman = {mean['delta_vs_shift_spearman']:.3f}")
    strong = (mean["pivot_over_function"] > 1.5 and mean["content_over_function"] > 1.5)
    print(f"  VERDICT: {'content-aware signal (pivot,content >> function)' if strong else 'weak/muddy separation'}")


if __name__ == "__main__":
    main()
