"""Trained static-embedder baseline (review round 6).

The paper's static baseline is a mean of the teacher's input-embedding rows,
which reviewers correctly called an untrained proxy that understates the
static family. This scores an off-the-shelf Model2Vec static model
(potion-base-8M, the SwiftEmbed backbone) on the same yardsticks:

  * SciFact nDCG@10 with full queries and with the partial-query protocol
    (n0 + 64 token prefixes decoded back to text);
  * STS-B Spearman.

The static model has its own embedding space, so it is an *absolute*
comparator (a system you could deploy instead), not a teacher-retention
method: it re-embeds the corpus into its own space, which is exactly the
re-indexing cost the paper's setting disallows. Both readings are reported.

Usage:
    python potion.py
"""

from __future__ import annotations

import argparse
from typing import Dict, List

import numpy as np

from experiments.distill_data import Teacher
from experiments.utils import log_environment, save_results, set_seed

MIN_TOKENS, ROLLOUT_K = 10, 64


def _norm_rows(a: np.ndarray) -> np.ndarray:
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def scifact_eval(static, teacher: Teacher, max_queries: int = 300) -> Dict:
    from beir import load_beir, ndcg_at_k

    ids, texts, queries, rel = load_beir("scifact", 60000)
    qids = sorted(queries)[:max_queries]
    corpus = _norm_rows(np.asarray(static.encode(texts)))
    scores: Dict[str, List[float]] = {"full": [], "partial": []}
    for qid in qids:
        text = queries[qid]
        q_ids = teacher.token_ids(text, 256)
        partial_text = teacher.tokenizer.decode(
            q_ids[:MIN_TOKENS + ROLLOUT_K]) if q_ids else text
        for key, t in (("full", text), ("partial", partial_text)):
            q = np.asarray(static.encode([t]))[0]
            q = q / (np.linalg.norm(q) + 1e-9)
            order = np.argsort(-(corpus @ q))[:10]
            scores[key].append(ndcg_at_k([ids[i] for i in order], rel[qid], 10))
    return {k: float(np.mean(v)) * 100 for k, v in scores.items()}


def stsb_eval(static) -> float:
    from datasets import load_dataset
    from scipy.stats import spearmanr

    data = load_dataset("glue", "stsb", split="validation")
    gold, sims = [], []
    for row in data:
        a, b = np.asarray(static.encode([row["sentence1"], row["sentence2"]]))
        sims.append(float(np.dot(a, b)
                          / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)))
        gold.append(row["label"])
    return float(spearmanr(sims, gold).correlation)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="minishlab/potion-base-8M")
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--out", type=str, default="experiments/results/potion.json")
    args = ap.parse_args()

    from model2vec import StaticModel

    set_seed(42)
    log_environment()
    static = StaticModel.from_pretrained(args.model)
    teacher = Teacher(args.teacher, device="cpu")   # tokenizer only

    scifact = scifact_eval(static, teacher)
    stsb = stsb_eval(static)
    print(f"  {args.model}: SciFact full={scifact['full']:.1f} "
          f"partial={scifact['partial']:.1f}  STS-B={stsb:.4f}")

    save_results(args.out, {"model": args.model, "scifact": scifact,
                            "stsb_spearman": stsb})


if __name__ == "__main__":
    main()
