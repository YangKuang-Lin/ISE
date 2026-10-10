"""Cell-size sensitivity: how small can the update cell get before it is useless?

Trains the gated-FiLM cell at shrinking widths/depths under the exact protocol
of distill.py (same data, epochs, optimizer, seeds) and evaluates
single-step accuracy, rollout drift, partial-query retrieval, and STS-B. The
two largest sizes are NOT retrained here: (384, 4) is the default cell of
distill.json and (192, 3) is the width-matched cell of parity.json, both
already run under the identical protocol; this script covers the tail of the
curve. The Pareto sweep is skipped, as the size study needs drift and
downstream quality, not the budget frontier.

Usage:
    python size.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Dict, List

import numpy as np

from experiments.cells import build_cell
from experiments.distill_data import Teacher, build_trajectories, load_wikitext_docs
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)
from distill import (
    DistillConfig,
    _triples_from_trajectories,
    retrieval,
    rollout_drift,
    single_step,
    sts_eval,
    train_cell,
)

# (hidden_dim, num_layers), largest first. The (192, 3) = 0.96M default cell
# comes from the main distill run; here we add the (384, 4) = 4.29M large
# variant (to show scaling up barely helps) and the sub-default rungs, spanning
# a 117x range from 4.29M down to 0.037M.
SIZES = [(384, 4), (96, 3), (48, 2), (24, 2)]


def horizon_and_tail(per_seed: List[Dict]) -> Dict[str, float]:
    """First step of the seed-mean drift curve below 0.90, plus cos@64.

    ``rollout_drift`` keys are ints in memory but strings once the result has
    been through JSON, so both forms are accepted.
    """
    curves = [{int(k): v for k, v in s["rollout_drift"].items()} for s in per_seed]
    steps = sorted(curves[0])
    mu = np.array([[c[k] for k in steps] for c in curves]).mean(axis=0)
    horizon = next((k for k, v in zip(steps, mu) if v < 0.90), None)
    return {"horizon": horizon, "cos64": float(mu[steps.index(64)])}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--out", type=str, default="experiments/results/size.json")
    ap.add_argument("--force", action="store_true",
                    help="retrain every size, ignoring cached configs in --out")
    args = ap.parse_args()

    device = get_device()
    log_environment()
    base = DistillConfig()
    teacher = Teacher(base.teacher, device=device)
    docs = load_wikitext_docs(base.dataset_config, base.num_docs + 2000, base.min_words)

    # The teacher encoding of the training documents is identical for every
    # size and seed, so it is hoisted out of the sweep (distill.run_once
    # rebuilds it per run; here that would triple-triple the cost).
    trajectories = build_trajectories(teacher, docs[:base.num_docs],
                                      base.min_tokens, base.max_tokens)
    triples_ordered = _triples_from_trajectories(trajectories)
    eval_docs = docs[base.num_docs:]

    # Each size costs ~30 min, so completed configs are checkpointed to disk
    # and reused on a re-run rather than retrained.
    results: Dict = {"teacher": base.teacher, "seeds": args.seeds, "configs": {}}
    out_path = Path(args.out)
    if out_path.exists() and not args.force:
        try:
            prior = json.loads(out_path.read_text(encoding="utf-8"))
            if prior.get("seeds") == args.seeds:
                results["configs"] = prior.get("configs", {})
                if results["configs"]:
                    print(f"resuming: {', '.join(results['configs'])} already done")
        except (OSError, ValueError) as e:
            print(f"could not reuse {out_path} ({e}); starting fresh")

    for hidden, layers in SIZES:
        cfg = replace(base, hidden_dim=hidden, num_layers=layers)
        params = count_params(build_cell(cfg.cell, teacher.emb_dim, hidden, layers))
        key = f"h{hidden}_l{layers}"
        if key in results["configs"]:
            print(f"== {key}: cached, skipping ==")
            continue
        print(f"== {key}: {params:,} params ==")

        per_seed = []
        for seed in args.seeds:
            set_seed(seed)
            rng = np.random.default_rng(seed)
            order = rng.permutation(len(triples_ordered))
            triples = [triples_ordered[i] for i in order]
            n_train = int(len(triples) * cfg.train_split)

            model = train_cell(cfg, triples, device)
            r = {"single_step": single_step(model, triples[n_train:], device),
                 "rollout_drift": rollout_drift(model, teacher, eval_docs[:100], cfg, device),
                 "retrieval": retrieval(model, teacher, eval_docs, cfg, device),
                 "sts": sts_eval(model, teacher, cfg, device)}
            per_seed.append(r)
            print(f"  seed {seed}: 1step={r['single_step']['ise_cos']:.5f} "
                  f"retr={r['retrieval']['ise']:.3f} sts={r['sts']['ise_spearman']:.4f}")

        entry = {"hidden": hidden, "layers": layers, "params": params,
                 "per_seed": per_seed,
                 "single_step_cos": aggregate_seeds(
                     [p["single_step"]["ise_cos"] for p in per_seed]),
                 "retrieval_ise": aggregate_seeds(
                     [p["retrieval"]["ise"] for p in per_seed]),
                 "sts_ise": aggregate_seeds(
                     [p["sts"]["ise_spearman"] for p in per_seed])}
        entry.update(horizon_and_tail(per_seed))
        results["configs"][key] = entry
        save_results(args.out, results)   # checkpoint after every size

    save_results(args.out, results)
    for key, e in results["configs"].items():
        print(f"{key:<8} params={e['params']:>9,} "
              f"1step={e['single_step_cos']['mean']:.5f} horizon={e['horizon']} "
              f"cos64={e['cos64']:.3f} retr={e['retrieval_ise']['mean']:.3f} "
              f"sts={e['sts_ise']['mean']:.4f}")


if __name__ == "__main__":
    main()
