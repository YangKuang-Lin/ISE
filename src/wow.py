"""Streaming knowledge selection on Wizard of Wikipedia (review round 7).

Reviewers asked for one real streaming consumer where the between-reset floor
changes an end outcome. This is incremental knowledge selection: while the
apprentice's utterance streams in token by token, the system continuously
ranks the wizard's candidate knowledge sentences (the dataset's own
per-turn retrieved passages, with the human-annotated checked sentence as
gold), so the correct knowledge can be prefetched before the user finishes
typing. The dialogue context accumulates across turns, so rollouts run far
past the drift horizon, unlike single-query retrieval.

Metrics per policy: final selection accuracy@1 at utterance end, the mean
within-utterance step at which the top-1 first becomes correct and stays
correct (decision latency), and teacher calls per streamed token.

The candidate pool is per-turn (a few dozen sentences), so no corpus is
encoded and the whole evaluation runs in minutes per policy.

Usage:
    python wow.py --ckpt experiments/results/ckpt/gated_film_unroll1_seed41.pt ...
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from experiments.cells import build_default_cell
from experiments.distill_data import Teacher
from experiments.utils import aggregate_seeds, get_device, log_environment, save_results, set_seed

WOW_URL = "http://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz"
MAX_CONTEXT = 250        # teacher max sequence length is 256
N0 = 10                  # teacher-encoded seed, matching the paper's protocol


def load_wow(data_dir: Path, split: str = "test_random_split.json") -> List[Dict]:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / split
    if not path.exists():
        tgz = data_dir / "wizard_of_wikipedia.tgz"
        if not tgz.exists():
            print(f"  downloading {WOW_URL} ...")
            urllib.request.urlretrieve(WOW_URL, tgz)
        with tarfile.open(tgz) as tf:
            tf.extractall(data_dir)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_streams(dialogs: List[Dict], teacher: Teacher, max_dialogs: int
                  ) -> List[Dict]:
    """Token stream per dialogue plus knowledge-selection decision points.

    A decision point sits at every token of the apprentice utterance that
    precedes a wizard turn carrying a human-checked knowledge sentence; the
    candidates are that wizard turn's retrieved-passage sentences.
    """
    streams = []
    for dlg in dialogs:
        turns = dlg["dialog"]
        ids: List[int] = []
        decisions = []   # {"span": (start, end), "gold": str, "cands": [str]}
        ok = True
        for i, turn in enumerate(turns):
            text = turn["text"].strip()
            new_ids = teacher.token_ids((" " if ids else "") + text, 512)
            if len(ids) + len(new_ids) > MAX_CONTEXT:
                break
            is_apprentice = "Apprentice" in turn["speaker"]
            nxt = turns[i + 1] if i + 1 < len(turns) else None
            if (is_apprentice and nxt is not None
                    and "Wizard" in nxt["speaker"]):
                checked = nxt.get("checked_sentence") or {}
                gold = next((v for k, v in checked.items()
                             if k != "no_passages_used"), None)
                cands: List[str] = []
                for passage in nxt.get("retrieved_passages", []):
                    for sents in passage.values():
                        cands.extend(sents)
                if gold and gold in cands and len(cands) >= 5:
                    decisions.append({"span": (len(ids), len(ids) + len(new_ids)),
                                      "gold": cands.index(gold),
                                      "cands": cands})
            ids.extend(new_ids)
        if decisions and len(ids) > N0 + 4:
            # Teacher embeddings of every prefix, computed once and shared by
            # all policies (the teacher policy consumes them directly; the
            # others use them at reset steps), so no prefix is encoded twice.
            true = teacher.encode_prefixes(ids, list(range(N0, len(ids) + 1)))
            streams.append({"ids": ids, "decisions": decisions, "true": true})
        if len(streams) >= max_dialogs:
            break
    return streams


@torch.no_grad()
def run_policy(policy: str, streams: List[Dict], teacher: Teacher,
               model: Optional[torch.nn.Module], device: str, k: int = 8,
               tau: float = 0.6) -> Dict[str, float]:
    """Stream every dialogue under one policy and score its decisions."""
    final_hits, first_stable, calls_total, tokens_total = [], [], 0, 0
    first_correct, flips_all = [], []
    acc3_hits, rr_all, pool_sizes = [], [], []
    cand_cache: Dict[int, np.ndarray] = {}
    for si, s in enumerate(streams):
        ids = s["ids"]
        # Candidate embeddings are teacher-encoded once (they are the index).
        for di, d in enumerate(s["decisions"]):
            key = si * 1000 + di
            if key not in cand_cache:
                emb = teacher.encode_texts(d["cands"], role="passage")
                cand_cache[key] = emb / (np.linalg.norm(emb, axis=1,
                                                        keepdims=True) + 1e-9)
        rows = teacher.token_embed(ids)
        true = s["true"]                      # true[j] = E(ids[:N0 + j])
        e = true[0]
        calls, path = 1, 0.0
        state = torch.tensor(e, dtype=torch.float32, device=device)
        # Per-decision bookkeeping. "stable_from" is the step opening the
        # final correct run (biased toward frozen embeddings, which cannot
        # flip); "first_correct" is the unbiased latency metric, and "flips"
        # counts top-1 changes so stability is scored separately.
        pending = {i: {"stable_from": None, "first_correct": None,
                       "steps": 0, "flips": 0, "prev_top": None,
                       "last_rank": None}
                   for i in range(len(s["decisions"]))}
        span_of = {}
        for di, d in enumerate(s["decisions"]):
            for t in range(*d["span"]):
                span_of.setdefault(t, []).append(di)

        def rank(vec: np.ndarray, t: int) -> None:
            for di in span_of.get(t, []):
                d = s["decisions"][di]
                q = vec / (np.linalg.norm(vec) + 1e-9)
                sims = cand_cache[si * 1000 + di] @ q
                top = int(np.argmax(sims))
                info = pending[di]
                info["steps"] += 1
                if info["prev_top"] is not None and top != info["prev_top"]:
                    info["flips"] += 1
                info["prev_top"] = top
                # Review round 8 (reviewer Q5): acc@1 alone cannot be read
                # against a 13.8% teacher ceiling without knowing the pool
                # size, so the gold's full rank at the utterance's last step is
                # carried through to acc@3 and MRR.
                info["last_rank"] = int((sims > sims[d["gold"]]).sum())
                if top == d["gold"]:
                    if info["first_correct"] is None:
                        info["first_correct"] = info["steps"]
                    if info["stable_from"] is None:
                        info["stable_from"] = info["steps"]
                else:
                    info["stable_from"] = None

        for t in range(N0, len(ids)):
            step = t - N0 + 1
            if policy == "teacher":
                e = true[step]
                calls += 1
            elif policy == "stale":
                if step % k == 0:
                    e = true[step]
                    calls += 1
            elif policy == "seed":
                pass
            else:   # cell-based: ise / hybrid / adaptive
                row = torch.tensor(rows[t], dtype=torch.float32, device=device)
                delta = model(row.unsqueeze(0), state.unsqueeze(0)).squeeze(0)
                state = state + delta
                path += float(delta.norm())
                reset = (policy == "hybrid" and step % k == 0) or \
                        (policy == "adaptive" and path > tau)
                if reset:
                    e = true[step]
                    state = torch.tensor(e, dtype=torch.float32, device=device)
                    calls += 1
                    path = 0.0
                else:
                    e = state.cpu().numpy()
            rank(e, t)
        for di, d in enumerate(s["decisions"]):
            info = pending[di]
            hit = info["stable_from"] is not None
            final_hits.append(float(hit))
            if hit:
                first_stable.append(info["stable_from"])
            if info["first_correct"] is not None:
                first_correct.append(info["first_correct"])
            if info["steps"] > 1:
                flips_all.append(info["flips"] / (info["steps"] - 1))
            pool_sizes.append(len(d["cands"]))
            if info["last_rank"] is not None:
                acc3_hits.append(float(info["last_rank"] < 3))
                rr_all.append(1.0 / (info["last_rank"] + 1))
        calls_total += calls
        tokens_total += len(ids)
    pools = np.asarray(pool_sizes, dtype=np.float64)
    return {"acc1": float(np.mean(final_hits)) * 100,
            "acc3": float(np.mean(acc3_hits)) * 100 if acc3_hits else float("nan"),
            "mrr": float(np.mean(rr_all)) if rr_all else float("nan"),
            "n_decisions": len(final_hits),
            "mean_first_correct": float(np.mean(first_correct)),
            "frac_ever_correct": len(first_correct) / len(final_hits) * 100,
            "mean_first_stable": float(np.mean(first_stable)),
            "flip_rate": float(np.mean(flips_all)) * 100,
            "callfrac": calls_total / tokens_total * 100,
            # Pool statistics and the chance floor they imply, so the low
            # absolute accuracies are interpretable (reviewer Q5).
            "pool_mean": float(pools.mean()),
            "pool_median": float(np.median(pools)),
            "pool_min": float(pools.min()),
            "pool_max": float(pools.max()),
            "random_acc1": float(np.mean(1.0 / pools)) * 100,
            "random_acc3": float(np.mean(np.minimum(3.0, pools) / pools)) * 100,
            "random_mrr": float(np.mean(
                [sum(1.0 / r for r in range(1, int(n) + 1)) / n for n in pools])),
            "hits": final_hits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, nargs="+",
                    default=["experiments/results/ckpt/gated_film_unroll1.pt"])
    ap.add_argument("--teacher", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--data-dir", type=str, default="../data/wow")
    ap.add_argument("--max-dialogs", type=int, default=300)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tau", type=float, default=0.6)
    ap.add_argument("--out", type=str, default="experiments/results/wow.json")
    args = ap.parse_args()

    device = get_device()
    set_seed(42)
    log_environment()
    teacher = Teacher(args.teacher, device=device)
    dialogs = load_wow(Path(args.data_dir))
    streams = build_streams(dialogs, teacher, args.max_dialogs)
    n_dec = sum(len(s["decisions"]) for s in streams)
    mean_len = float(np.mean([len(s["ids"]) for s in streams]))
    print(f"  {len(streams)} dialogues, {n_dec} knowledge decisions, "
          f"mean stream {mean_len:.0f} tokens")

    results: Dict[str, Dict] = {}
    stds: Dict[str, Dict] = {}
    hits: Dict[str, np.ndarray] = {}
    # Teacher-only policies are cell-independent and run once.
    for policy in ("teacher", "stale", "seed"):
        r = run_policy(policy, streams, teacher, None, device,
                       k=args.k, tau=args.tau)
        hits[policy] = np.array(r.pop("hits"))
        results[policy] = r
        print(f"  {policy:>9}: " + "  ".join(f"{m}={v:.2f}" for m, v in r.items()))
    for policy in ("hybrid", "adaptive", "ise"):
        per_seed = []
        for ck in args.ckpt:
            model = build_default_cell("gated_film", teacher.emb_dim).to(device)
            model.load_state_dict(torch.load(ck, map_location=device))
            model.eval()
            per_seed.append(run_policy(policy, streams, teacher, model, device,
                                       k=args.k if policy == "hybrid" else 0,
                                       tau=args.tau))
        hits[policy] = np.mean([p.pop("hits") for p in per_seed], axis=0)
        keys = per_seed[0].keys()
        results[policy] = {m: aggregate_seeds([p[m] for p in per_seed])["mean"]
                           for m in keys}
        stds[policy] = {m: aggregate_seeds([p[m] for p in per_seed])["std"]
                        for m in keys}
        print(f"  {policy:>9}: " + "  ".join(f"{m}={v:.2f}" for m, v in
                                             results[policy].items()))

    # Paired bootstrap over decisions (seed-mean hits for cell policies),
    # matching the retrieval scripts' significance protocol.
    def paired_bootstrap(a: np.ndarray, b: np.ndarray, iters: int = 10000):
        rng = np.random.default_rng(0)
        diff = a - b
        n = len(diff)
        boots = np.array([diff[rng.integers(0, n, n)].mean()
                          for _ in range(iters)])
        p = float(2 * min((boots <= 0).mean(), (boots >= 0).mean()))
        return {"mean_diff": float(diff.mean()) * 100, "p": min(p, 1.0),
                "ci95": [float(np.percentile(boots, 2.5)) * 100,
                         float(np.percentile(boots, 97.5)) * 100],
                "n": int(n)}

    significance = {f"{a}_vs_{b}": paired_bootstrap(hits[a], hits[b])
                    for a, b in [("hybrid", "stale"), ("teacher", "hybrid"),
                                 ("hybrid", "ise")]}
    for kk, v in significance.items():
        print(f"  {kk}: diff={v['mean_diff']:+.2f} acc pts  p={v['p']:.4f}  "
              f"ci95=[{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]  n={v['n']}")

    save_results(args.out, {"teacher_model": args.teacher, "k": args.k,
                            "tau": args.tau, "n_dialogs": len(streams),
                            "n_decisions": n_dec,
                            "mean_stream_tokens": mean_len,
                            "results": results, "results_std": stds,
                            "significance": significance})


if __name__ == "__main__":
    main()
