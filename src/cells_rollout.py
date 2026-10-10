"""Full-rollout evaluation of the remaining update-cell baselines (E1).

The cell ablation of distill.py scores concat_mlp and gated_residual at a
single step only (``full_eval=False``), while the paper's own argument is that
a single step is a weak proxy for streaming. The GRU and FiLM cells already
have full-rollout numbers (distill_gru.json, distill.json / parity.json); this
script closes the gap by training the two remaining baselines under the exact
main protocol (same data, epochs, optimizer, seeds, default geometry) and
evaluating drift horizon, cos@64, partial-query retrieval, and STS-B.

Usage:
    python cells_rollout.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.cells import DEFAULT_NUM_LAYERS, build_cell, default_hidden
from experiments.distill_data import Teacher, load_wikitext_docs
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)
from distill import DistillConfig, run_once
from size import horizon_and_tail


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--cells", type=str, nargs="+",
                    default=["concat_mlp", "gated_residual"])
    ap.add_argument("--hidden", type=int, default=None,
                    help="override the hidden width for every cell in this "
                         "run (e.g. to parameter-match the 0.96M FiLM cell); "
                         "results are stored under '<cell>_h<hidden>'")
    ap.add_argument("--out", type=str,
                    default="experiments/results/cells_rollout.json")
    ap.add_argument("--force", action="store_true",
                    help="retrain every cell, ignoring cached entries in --out")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override the training epochs (smoke tests only)")
    ap.add_argument("--num-docs", type=int, default=None,
                    help="override the number of distillation documents "
                         "(smoke tests only)")
    args = ap.parse_args()

    device = get_device()
    log_environment()

    teacher = Teacher("all-MiniLM-L6-v2", device=device)
    # DistillConfig still defaults to the pre-switch geometry; pin the deployed
    # default explicitly so these baselines match Table 1's parameter counts.
    base = DistillConfig(hidden_dim=default_hidden(teacher.emb_dim),
                         num_layers=DEFAULT_NUM_LAYERS)
    if args.epochs is not None:
        base = replace(base, epochs=args.epochs)
    if args.num_docs is not None:
        base = replace(base, num_docs=args.num_docs)
    docs = load_wikitext_docs(base.dataset_config, base.num_docs + 2000,
                              base.min_words)

    results: Dict = {"teacher": base.teacher, "seeds": args.seeds,
                     "hidden_dim": base.hidden_dim,
                     "num_layers": base.num_layers, "cells": {}}
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if prior.get("seeds") == args.seeds:
                results["cells"] = prior.get("cells", {})
                if results["cells"]:
                    print(f"resuming: {', '.join(results['cells'])} already done")
        except (OSError, ValueError) as e:
            print(f"could not reuse {out_path} ({e}); starting fresh")

    for cell in args.cells:
        hidden = args.hidden or base.hidden_dim
        key = cell if args.hidden is None else f"{cell}_h{hidden}"
        if key in results["cells"]:
            print(f"== {key}: cached, skipping ==")
            continue
        cfg = replace(base, cell=cell, hidden_dim=hidden)
        params = count_params(build_cell(cell, teacher.emb_dim,
                                         cfg.hidden_dim, cfg.num_layers))
        print(f"== {key}: {params:,} params ==")

        per_seed: List[Dict] = []
        for seed in args.seeds:
            r, _ = run_once(cfg, seed, device, teacher, docs, full_eval=True)
            per_seed.append(r)
            print(f"  seed {seed}: 1step={r['single_step']['ise_cos']:.5f} "
                  f"retr={r['retrieval']['ise']:.3f} "
                  f"sts={r['sts']['ise_spearman']:.4f}")

        entry = {"params": params, "hidden_dim": hidden, "per_seed": per_seed,
                 "single_step_cos": aggregate_seeds(
                     [p["single_step"]["ise_cos"] for p in per_seed]),
                 "retrieval_ise": aggregate_seeds(
                     [p["retrieval"]["ise"] for p in per_seed]),
                 "sts_ise": aggregate_seeds(
                     [p["sts"]["ise_spearman"] for p in per_seed])}
        entry.update(horizon_and_tail(per_seed))
        results["cells"][key] = entry
        save_results(args.out, results)   # checkpoint after every cell

    save_results(args.out, results)
    for cell, e in results["cells"].items():
        print(f"{cell:<15} params={e['params']:>8,} "
              f"1step={e['single_step_cos']['mean']:.5f} "
              f"horizon={e['horizon']} cos64={e['cos64']:.3f} "
              f"retr={e['retrieval_ise']['mean']:.3f} "
              f"sts={e['sts_ise']['mean']:.4f}")


if __name__ == "__main__":
    main()
