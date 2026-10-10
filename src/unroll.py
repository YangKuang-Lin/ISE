"""Multi-step (unrolled) training at three seeds, matching the main protocol.

The original unrolled runs trained U=4 and U=8 on a single seed inside the
distill stage, which forced Table 8's U=1 row onto that seed's values (horizon
13, cos@64 0.808) while the rest of the paper reports three-seed means (12,
0.799). This script re-trains U=4 and U=8 across the three seeds so the whole
table lives on one protocol; the U=1 configuration is NOT retrained, since it
is exactly the main run of distill.json. Every U trains the deployed 0.96M
cell, so the table isolates the objective from capacity.

Evaluation covers what Table 8 reports: single-step cosine and rollout drift.

Usage:
    python unroll.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.cells import DEFAULT_NUM_LAYERS, default_hidden
from experiments.distill_data import Teacher, build_trajectories, load_wikitext_docs
from experiments.utils import (
    aggregate_seeds,
    get_device,
    log_environment,
    save_results,
    set_seed,
)
from distill import (
    DistillConfig,
    _triples_from_trajectories,
    rollout_drift,
    single_step,
    train_cell_multistep,
)

UNROLLS = [4, 8]


def drift_at(run: Dict, step: int) -> float:
    """Rollout similarity at ``step``.

    ``rollout_drift`` is keyed by int in memory and by str once it has been
    through JSON, and a resumed sweep holds both forms at once, so neither
    lookup can be assumed.
    """
    curve = {int(k): v for k, v in run["rollout_drift"].items()}
    return float(curve[step])


def mean_curve_horizon(per_seed: List[Dict]) -> int:
    """Horizon of the seed-mean drift curve, the paper's canonical statistic."""
    curves = [{int(k): v for k, v in s["rollout_drift"].items()} for s in per_seed]
    steps = sorted(curves[0])
    mu = np.array([[c[k] for k in steps] for c in curves]).mean(axis=0)
    return next(k for k, v in zip(steps, mu) if v < 0.90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--out", type=str, default="experiments/results/unroll3.json")
    ap.add_argument("--force", action="store_true",
                    help="retrain every (U, seed), ignoring cached runs in --out")
    args = ap.parse_args()

    device = get_device()
    log_environment()
    base = DistillConfig()
    teacher = Teacher(base.teacher, device=device)
    # The cell geometry is pinned to the deployed one explicitly: the U=1 row of
    # this table is taken from distill.json, so U=4/U=8 must train the same
    # 0.96M cell or the objective ablation would also vary capacity.
    base = replace(base, hidden_dim=default_hidden(teacher.emb_dim),
                   num_layers=DEFAULT_NUM_LAYERS)
    docs = load_wikitext_docs(base.dataset_config, base.num_docs + 2000, base.min_words)

    # Teacher encodings are identical across seeds and unroll lengths, so they
    # are computed once (distill rebuilds them per run).
    trajectories = build_trajectories(teacher, docs[:base.num_docs],
                                      base.min_tokens, base.max_tokens)
    triples_ordered = _triples_from_trajectories(trajectories)
    eval_docs = docs[base.num_docs:]

    # Each (U, seed) run costs ~15-30 min, so completed runs are checkpointed
    # and reused; a crash resumes instead of retraining.
    out_path = Path(args.out)
    results: Dict = {"seeds": args.seeds, "runs": {f"unroll={u}": {"per_seed": []}
                                                   for u in UNROLLS}}
    if out_path.exists() and not args.force:
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if prior.get("seeds") == args.seeds:
                for key in results["runs"]:
                    results["runs"][key]["per_seed"] = \
                        prior.get("runs", {}).get(key, {}).get("per_seed", [])
                done = {k: len(v["per_seed"]) for k, v in results["runs"].items()}
                if any(done.values()):
                    print(f"resuming: {done} runs already cached")
        except (OSError, ValueError) as e:
            print(f"could not reuse {out_path} ({e}); starting fresh")

    for u in UNROLLS:
        key = f"unroll={u}"
        per_seed = results["runs"][key]["per_seed"]
        for i, seed in enumerate(args.seeds):
            if i < len(per_seed):
                print(f"== {key} seed {seed}: cached, skipping ==")
                continue
            print(f"== {key} seed {seed} ==")
            set_seed(seed)
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(triples_ordered))
            triples = [triples_ordered[i] for i in order]
            n_train = int(len(triples) * base.train_split)

            model = train_cell_multistep(base, trajectories, device, unroll=u)
            r = {"single_step": single_step(model, triples[n_train:], device),
                 "rollout_drift": rollout_drift(model, teacher, eval_docs[:100],
                                                base, device)}
            per_seed.append(r)
            print(f"  seed {seed}: 1step={r['single_step']['ise_cos']:.5f} "
                  f"H={mean_curve_horizon([r])} cos64={drift_at(r, 64):.4f}")
            save_results(args.out, results)   # checkpoint after every run

        results["runs"][key]["single_step_cos"] = aggregate_seeds(
            [p["single_step"]["ise_cos"] for p in per_seed])
        results["runs"][key]["cos64"] = aggregate_seeds(
            [drift_at(p, 64) for p in per_seed])
        results["runs"][key]["horizon"] = mean_curve_horizon(per_seed)
        save_results(args.out, results)

    results["complete"] = True
    save_results(args.out, results)
    for u in UNROLLS:
        e = results["runs"][f"unroll={u}"]
        print(f"U={u}: 1step={e['single_step_cos']['mean']:.5f} "
              f"H={e['horizon']} cos64={e['cos64']['mean']:.4f}")


if __name__ == "__main__":
    main()
