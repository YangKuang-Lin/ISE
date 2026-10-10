"""Token-length statistics for the STS suite (review round 4, item 10).

A reviewer asked what fraction of STS pairs actually trigger a re-encode under a
given (n0, K) schedule, since many STS sentences are short. This records the
length distribution per task, the fraction of sentences long enough to reach the
first and second reset, and the mean number of genuinely incremental steps --
which is what makes the STS numbers interpretable.

Usage:
    python sts_stats.py
"""

from __future__ import annotations

import argparse
from typing import Dict

import numpy as np

from experiments.distill_data import Teacher
from experiments.utils import save_results, set_seed
from broader import TASKS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--limit", type=int, default=800)
    ap.add_argument("--n0", type=int, nargs="+", default=[4, 10])
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", type=str, default="experiments/results/sts_stats.json")
    args = ap.parse_args()

    from datasets import load_dataset
    set_seed(42)
    teacher = Teacher(args.teacher, device="cpu")

    per_task: Dict[str, Dict] = {}
    all_len: list[int] = []
    for name, ds_id, split in TASKS:
        data = load_dataset(ds_id, split=split)
        if args.limit and len(data) > args.limit:
            data = data.select(range(args.limit))
        lens = [len(teacher.token_ids(s, 256))
                for r in data for s in (r["sentence1"], r["sentence2"])]
        all_len += lens
        arr = np.array(lens)
        per_task[name] = {
            "n_sentences": int(arr.size),
            "median": float(np.median(arr)),
            "mean": float(arr.mean()),
            "p90": float(np.percentile(arr, 90)),
        }
        print(f"  {name:<8} median={np.median(arr):>5.0f}  mean={arr.mean():>5.1f}")

    arr = np.array(all_len)
    schedules: Dict[str, Dict] = {}
    for n0 in args.n0:
        first, second = n0 + args.k, n0 + 2 * args.k
        schedules[f"n0={n0},K={args.k}"] = {
            "first_reset_at_token": first,
            "frac_reaching_first_reset": float((arr >= first).mean()),
            "second_reset_at_token": second,
            "frac_reaching_second_reset": float((arr >= second).mean()),
            "mean_incremental_steps": float(np.maximum(arr - n0, 0).mean()),
        }
        s = schedules[f"n0={n0},K={args.k}"]
        print(f"  n0={n0},K={args.k}: first reset at token {first} fires for "
              f"{s['frac_reaching_first_reset']*100:.1f}% of sentences; "
              f"mean {s['mean_incremental_steps']:.1f} incremental steps")

    save_results(args.out, {
        "teacher": args.teacher, "limit": args.limit,
        "per_task": per_task,
        "overall": {"median": float(np.median(arr)), "mean": float(arr.mean()),
                    "n_sentences": int(arr.size)},
        "schedules": schedules,
    })


if __name__ == "__main__":
    main()
