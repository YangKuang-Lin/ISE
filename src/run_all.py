#!/usr/bin/env python3
# =============================================================================
# run_all.py -- reproduce every experiment for the ISE paper (Windows-friendly).
#
# Runs the experiment scripts with the *current* interpreter (sys.executable),
# streams output live, and tees a copy into experiments/results/logs/*.log.
#
# Stages are dependency-ordered and RESUMABLE: a stage whose output already
# exists is skipped unless --force is given, so a run killed halfway can be
# resumed by re-issuing the same command.
#
# Target hardware: single RTX 4070 Ti (12 GB VRAM) + ~46 GB system RAM.
#
# Usage:
#   python run_all.py                 # everything needed for the paper
#   python run_all.py --list          # show stages, outputs, and what would run
#   python run_all.py quick           # 1 seed, no ablations (fast sanity pass)
#   python run_all.py distill         # a single stage
#   python run_all.py --force         # ignore existing outputs, redo everything
#   python run_all.py distill --force # redo just one stage
#
# Results  -> experiments/results/*.json      Figures -> ../paper/assets/*.pdf
# Logs     -> experiments/results/logs/*.log  Ckpts   -> experiments/results/ckpt/
# =============================================================================
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# cd into this script's directory (src/) so relative paths resolve consistently.
os.chdir(Path(__file__).resolve().parent)

# The console echo in run_logged must never kill a stage: on Windows the
# active code page (e.g. cp950) cannot encode every character a child's
# progress bar may emit, and one bad byte would abort an hours-long run.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

RESULTS = Path("experiments/results")
LOGS = RESULTS / "logs"
CKPT = RESULTS / "ckpt"
ASSETS = Path("../paper/assets")

os.environ["PYTHONUNBUFFERED"] = "1"
# Harmless no-op on Windows; helps avoid fragmenting the 12 GB pool on Linux.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# (Optional) pin the HuggingFace cache to a large drive:
# os.environ["HF_HOME"] = r"D:\hf_cache"

PY = sys.executable


# --- helpers -----------------------------------------------------------------
def banner(msg: str) -> None:
    print(f"\n{'=' * 68}\n>>> {msg}\n{'=' * 68}", flush=True)


def stage_line(msg: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def run_logged(cmd: list[str], log_path: Path) -> None:
    """Run cmd, streaming merged stdout+stderr to console and log. Abort on failure."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding="utf-8", errors="replace", bufsize=1)
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            raise
    if proc.returncode != 0:
        raise SystemExit(f"\n*** Stage failed (exit {proc.returncode}): {' '.join(cmd)}")


def distill_json_is_current(path: Path) -> bool:
    """Detect result files written before the Pareto/retrieval metric fixes.

    Old runs stored a degenerate ``final_cos_mean`` (1.0 by construction) and a
    circular retrieval ``oracle_teacher``; those must not be reused.
    """
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        pareto = d["main"]["per_seed"][0]["pareto"][0]
        retr = d["main"]["per_seed"][0]["retrieval"]
        return "mean_cos" in pareto and "teacher_prefix" in retr
    except Exception:
        return False


# --- configuration -----------------------------------------------------------
@dataclass
class Config:
    seeds: list[str]
    scratch_cells: list[str]
    distill_ablate: list[str]
    unroll: list[str]
    adaptive: list[str]
    speed_gen: list[str]
    speed_batch: list[str]
    stream_lengths: list[str]
    stream_docs: str
    broader_limit: str          # max pairs per STS task (0 = all)
    adaptive_lengths: list[str] # rollout lengths for the tau sensitivity sweep


def make_config(quick: bool) -> Config:
    if quick:
        return Config(
            seeds=["42"],
            scratch_cells=["gated_residual", "transformer"],
            distill_ablate=[],
            unroll=["1"],
            adaptive=[],
            speed_gen=["512", "1024"],
            speed_batch=["1", "4"],
            stream_lengths=["128", "256"],
            stream_docs="10",
            broader_limit="200",
            adaptive_lengths=["32"],
        )
    return Config(
        seeds=["41", "42", "43"],
        scratch_cells=["gated_residual", "gru", "transformer"],
        # NOTE: --ablate-init intentionally absent. The `init` flag never
        # affected the training path (teacher/zero were bit-identical), so it
        # was a dead option, not a real ablation, and has been removed.
        distill_ablate=["--ablate-cells", "--ablate-loss"],
        unroll=["1", "4", "8"],          # multi-step (unrolled) training
        adaptive=["--adaptive"],         # path-length adaptive reset sweep
        speed_gen=["512", "1024", "2048"],
        speed_batch=["1", "4"],
        stream_lengths=["128", "256", "512"],
        stream_docs="20",
        # 0 == uncapped: full STS test splits (review round 4 flagged the
        # 800-pair cap as breaking comparability with published numbers).
        broader_limit="0",
        adaptive_lengths=["32", "64", "128"],
    )


# --- stages ------------------------------------------------------------------
@dataclass
class Stage:
    name: str
    desc: str
    est: str
    outputs: list[Path]
    needs: list[Path] = field(default_factory=list)
    fresh_check: bool = False        # apply distill_json_is_current() to outputs[0]
    # Substring that must appear in outputs[0]; if absent the file predates a
    # schema change and is re-run. Cheaper and more robust than versioning.
    requires_key: str = ""

    def satisfied(self) -> bool:
        if not all(o.exists() for o in self.outputs):
            return False
        if self.fresh_check and not distill_json_is_current(self.outputs[0]):
            return False
        if self.requires_key:
            try:
                if self.requires_key not in self.outputs[0].read_text(encoding="utf-8"):
                    return False
            except OSError:
                return False
        # Rebuild if any input is newer than the oldest output (e.g. figures
        # must be redrawn after distill.json is regenerated).
        if self.needs:
            oldest_out = min(o.stat().st_mtime for o in self.outputs)
            if any(n.exists() and n.stat().st_mtime > oldest_out for n in self.needs):
                return False
        return True


def build_stages(cfg: Config) -> dict[str, tuple[Stage, list[str]]]:
    """Return {name: (Stage, command)} in dependency order."""
    distill_out = RESULTS / "distill.json"
    speed_out = RESULTS / "speed.json"
    scratch_out = RESULTS / "scratch.json"
    stream_out = RESULTS / "streaming.json"
    ckpt = CKPT / "gated_film_unroll1.pt"

    def seed_ckpts(tag: str = "") -> list[str]:
        """Per-seed checkpoints for a cell tag, newest protocol first.

        distill saves one checkpoint per seed so the downstream
        evaluations can be averaged over the same three seeds as the
        distillation results. Falls back to the unsuffixed (seed-41) file if
        the per-seed ones are absent, which keeps older result trees usable.
        """
        paths = [CKPT / f"gated_film{tag}_unroll1_seed{s}.pt" for s in cfg.seeds]
        if all(q.exists() for q in paths):
            return [str(q) for q in paths]
        return [str(CKPT / f"gated_film{tag}_unroll1.pt")]

    return {
        # 1. Per-token update microbenchmark. No dataset download (random
        #    weights); architecture, not weights, determines throughput.
        # Timing is fragile: the no-cache baseline must be warmed at full
        # length or its first repetition pays a cold 1.3 GB allocation. Two
        # earlier protocols got this wrong (see speed.SpeedConfig); the
        # current one isolates each measurement. Run it on an idle machine --
        # the stage prints a warning for any cell with a wide spread.
        "speed": (
            Stage("speed", "Per-token update cost: ISE / GRU / TF(KV-cache) / TF(no-cache)",
                  "~15-20 min", [speed_out], requires_key="ISE_reps"),
            [PY, "speed.py", "--gen", *cfg.speed_gen, "--batch", *cfg.speed_batch,
             "--out", str(speed_out)],
        ),
        # 2. Core distillation. Writes distill.json AND the checkpoint that the
        #    streaming stage needs. Downloads WikiText-2 / STS-B / SST-2 once.
        "distill": (
            Stage("distill", "Embedding-shift distillation: drift, retrieval, STS, "
                             "Pareto, multi-step, adaptive resets, ablations",
                  "~5-6 h",
                  [distill_out, ckpt,
                   CKPT / f"gated_film_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True,
                  requires_key='"ise_cos_agg"'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--unroll", *cfg.unroll, *cfg.adaptive, *cfg.distill_ablate,
             "--ckpt-dir", str(CKPT), "--out", str(distill_out)],
        ),
        # 3. Streaming embedding (the target workload). Needs the checkpoint.
        #    Includes stale-cadence rows and per-step worst-case (min) fidelity
        #    (review round 4: matched wall clock, not only matched calls).
        "streaming": (
            Stage("streaming", "Streaming embedding end-to-end: latency / memory / fidelity",
                  "~15 min", [stream_out], needs=[ckpt],
                  requires_key='"floor_significance"'),
            [PY, "streaming.py", "--ckpt", str(ckpt), "--lengths", *cfg.stream_lengths,
             "--docs", cfg.stream_docs, "--stale-budgets", "4", "8",
             "--out", str(stream_out)],
        ),
        # 3b. Broader semantic evaluation: seven STS tasks, plus the teacher
        #     upper bound and a static token-lookup lower bound.
        "broader": (
            Stage("broader", "Seven-task STS suite vs teacher and static-lookup baselines",
                  "~45 min", [RESULTS / "broader.json"], needs=[ckpt],
                  requires_key=f'"limit": {cfg.broader_limit}'),
            [PY, "broader.py", "--ckpt", *seed_ckpts(), "--limit", cfg.broader_limit,
             "--out", str(RESULTS / "broader.json")],
        ),
        # 3c. CPU latency for the same streaming workload (on-device deployment).
        "streaming_cpu": (
            Stage("streaming_cpu", "Streaming embedding on CPU (on-device latency)",
                  "~15 min", [RESULTS / "streaming_cpu.json"], needs=[ckpt],
                  requires_key='"floor_significance"'),
            [PY, "streaming.py", "--ckpt", str(ckpt), "--lengths", "128",
             "--docs", "8", "--repeats", "2", "--device", "cpu",
             "--stale-budgets", "4", "8",
             "--out", str(RESULTS / "streaming_cpu.json")],
        ),
        # 3c-bis. tau sensitivity across rollout lengths (eval-only: reuses the
        #     trained checkpoint rather than retraining).
        "tau": (
            Stage("tau", "Adaptive-threshold sensitivity: rollout length x minimum-K",
                  "~25 min", [RESULTS / "tau.json"], needs=[ckpt], requires_key='"n_ckpt"'),
            [PY, "tau.py", "--ckpt", *seed_ckpts(), "--lengths", *cfg.adaptive_lengths,
             "--min-k", "0", "4", "8", "--out", str(RESULTS / "tau.json")],
        ),
        # 3c-ter. Reset-placement isolation: at a fixed teacher budget and fixed
        #     gap multiset, permute reset order to show placement matters on its
        #     own (eval-only; three seed checkpoints).
        "placement": (
            Stage("placement", "Reset placement vs spacing at a fixed budget",
                  "~15 min", [RESULTS / "placement.json"], needs=[ckpt], requires_key='"n_ckpt"'),
            [PY, "placement.py", "--ckpt", *seed_ckpts(),
             "--out", str(RESULTS / "placement.json")],
        ),
        # 3c-quater. Content-awareness probe: is the cell's update magnitude
        #     |delta| a content signal (large on content/pivot words, small on
        #     function words)? Motivates the adaptive reset rule. Eval-only.
        "delta_probe": (
            Stage("delta_probe", "Cell |delta| by token type (adaptive-rule content check)",
                  "~10 min", [RESULTS / "delta_probe.json"], needs=[ckpt], requires_key='"n_ckpt"'),
            [PY, "delta_probe.py", "--ckpt", *seed_ckpts(),
             "--out", str(RESULTS / "delta_probe.json")],
        ),
        # 3c-quinquies. The decisive content-vs-position reset test: at a fixed
        #     budget, compare resetting at content (high-|delta|) vs filler vs
        #     even vs the cumulative-path (equal-drift) rule. Shows content
        #     matters but must be used via cumulative spacing, not naive
        #     placement. Also dumps qualitative token-level examples. Eval-only.
        "content_reset": (
            Stage("content_reset", "Content-aware reset placement (decisive test)",
                  "~15 min", [RESULTS / "content_reset.json"], needs=[ckpt], requires_key='"n_ckpt"'),
            [PY, "content_reset.py", "--ckpt", *seed_ckpts(),
             "--out", str(RESULTS / "content_reset.json")],
        ),
        # 3d. Stability ablations (review round 2, Q4/Q5): does L2-normalizing
        #     the state, or a scale-invariant loss, extend the drift horizon?
        "stability": (
            Stage("stability", "State-normalization + scale-invariant-loss ablations",
                  "~75 min", [RESULTS / "stability.json"],
                  requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--normalize-state", "--ablate-loss",
             "--ckpt-tag", "_norm", "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "stability.json")],
        ),
        # 3e. Teacher transfer (review round 2, Q3): repeat the core metrics with
        #     a larger teacher to test whether the method depends on MiniLM.
        "teacher2": (
            # Measured 2026-07-20: ~12 min/seed (trajectory building batches
            # well); earlier "~3-4.5 h" was an unmeasured overestimate.
            Stage("teacher2", "Stronger teacher (all-mpnet-base-v2, 768-d) scaling",
                  "~40 min",
                  [RESULTS / "distill_mpnet.json",
                   CKPT / "gated_film_mpnet_unroll1.pt",
                   CKPT / f"gated_film_mpnet_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True, requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--teacher", "all-mpnet-base-v2", "--ckpt-tag", "_mpnet",
             "--ckpt-dir", str(CKPT), "--out", str(RESULTS / "distill_mpnet.json")],
        ),
        # 3f. STS for the second teacher, using its own checkpoint.
        "broader_mpnet": (
            Stage("broader_mpnet", "STS suite under the second teacher",
                  "~45 min", [RESULTS / "broader_mpnet.json"],
                  needs=[CKPT / "gated_film_mpnet_unroll1.pt"],
                  requires_key=f'"limit": {cfg.broader_limit}'),
            [PY, "broader.py", "--ckpt", *seed_ckpts("_mpnet"),
             "--teacher", "all-mpnet-base-v2", "--limit", cfg.broader_limit,
             "--out", str(RESULTS / "broader_mpnet.json")],
        ),
        # 3g. BEIR retrieval (review round 3, Q4): replaces the 500-doc toy
        #     setting with SciFact -- real corpus, real queries, human qrels.
        "beir": (
            Stage("beir", "SciFact (BEIR) retrieval with incrementally embedded queries",
                  "~25 min", [RESULTS / "beir.json"], needs=[ckpt],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--ckpt", *seed_ckpts(), "--n0", "4", "10",
             "--budgets", "8", "--max-queries", "300",
             "--out", str(RESULTS / "beir.json")],
        ),
        # 3g-bis. Stale-cadence baseline (reviews 5+6): re-encode every K and
        #     hold in between, no cell. Eval-only; the comparison that justifies
        #     the learned update.
        "stale": (
            Stage("stale", "Stale-cadence baseline vs cell at matched teacher budget",
                  "~15 min", [RESULTS / "stale.json"], needs=[ckpt], requires_key='"n_ckpt"'),
            [PY, "stale.py", "--ckpt", *seed_ckpts(),
             "--out", str(RESULTS / "stale.json")],
        ),
        # 3g-ter. Second BEIR dataset (review 5): FiQA, financial domain, 11x corpus.
        "beir_fiqa": (
            Stage("beir_fiqa", "FiQA (BEIR) retrieval, second domain (full 648 queries)",
                  "~55 min", [RESULTS / "beir_fiqa.json"], needs=[ckpt],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--dataset", "fiqa", "--ckpt", *seed_ckpts(),
             "--n0", "10", "--budgets", "8", "--max-queries", "648",
             "--out", str(RESULTS / "beir_fiqa.json")],
        ),
        # 3g-ter-a. Cost-matched adaptive point (review round 7): tau=0.9 puts
        #     the adaptive rule near hybrid-8's teacher-call budget on short
        #     queries, so the two policies can be compared at equal cost.
        "beir_tau09": (
            Stage("beir_tau09", "SciFact retrieval, adaptive at tau=0.9 (cost-matched)",
                  "~20 min", [RESULTS / "beir_tau09.json"], needs=[ckpt],
                  requires_key='"per_query"'),
            [PY, "beir.py", "--ckpt", *seed_ckpts(), "--n0", "4", "10",
             "--budgets", "8", "--max-queries", "300", "--tau", "0.9",
             "--out", str(RESULTS / "beir_tau09.json")],
        ),
        "beir_fiqa_tau09": (
            Stage("beir_fiqa_tau09", "FiQA retrieval, adaptive at tau=0.9 (cost-matched)",
                  "~50 min", [RESULTS / "beir_fiqa_tau09.json"], needs=[ckpt],
                  requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "fiqa", "--ckpt", *seed_ckpts(),
             "--n0", "10", "--budgets", "8", "--max-queries", "648",
             "--tau", "0.9",
             "--out", str(RESULTS / "beir_fiqa_tau09.json")],
        ),
        # 3g-ter-b. Full-rollout evaluation of the remaining cell baselines
        #     (review round 7): concat_mlp and gated_residual were previously
        #     scored at a single step only.
        "cells_rollout": (
            Stage("cells_rollout", "Concat/gated-residual cells under full rollout",
                  "~2-3 h", [RESULTS / "cells_rollout.json"]),
            [PY, "cells_rollout.py", "--seeds", *cfg.seeds,
             "--out", str(RESULTS / "cells_rollout.json")],
        ),
        # 3g-quater. SciFact under the mpnet teacher (review round 5): is the
        #     strong-teacher penalty as severe on the real benchmark?
        "beir_mpnet": (
            # requires_key bumped round 6: the round-4 file lacks per-query
            # scores, so the hybrid-vs-stale p-value reviewers asked for
            # cannot be computed from it.
            Stage("beir_mpnet", "SciFact (BEIR) under the mpnet teacher",
                  "~20 min", [RESULTS / "beir_mpnet.json"],
                  needs=[CKPT / "gated_film_mpnet_unroll1.pt"], requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "scifact",
             "--teacher", "all-mpnet-base-v2",
             "--ckpt", *seed_ckpts("_mpnet"),
             "--n0", "10", "--budgets", "8", "--max-queries", "300",
             "--out", str(RESULTS / "beir_mpnet.json")],
        ),
        # 3h. GRU cell under full rollout/hybrid evaluation (review round 3, Q1):
        #     does FiLM's single-step edge translate into a longer horizon?
        "gru_rollout": (
            Stage("gru_rollout", "GRU cell: rollout drift + hybrid, vs FiLM",
                  "~60 min", [RESULTS / "distill_gru.json"], fresh_check=True,
                  requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gru",
             "--ckpt-tag", "_gru", "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "distill_gru.json")],
        ),
        # 3i. Second teacher with a matched compute budget (review round 3, Q2):
        #     does more training close the mpnet gap, or is it intrinsic?
        "teacher2_tuned": (
            # Round 6: three seeds (reviews flagged the single-seed probe).
            Stage("teacher2_tuned", "Second teacher with 3x epochs (compute-matched, 3 seeds)",
                  "~6-9 h",
                  [RESULTS / "distill_mpnet_tuned.json",
                   CKPT / "gated_film_mpnet3x_unroll1.pt"], fresh_check=True,
                  requires_key='41,\n    42,\n    43'),
            # The first attempt at 180 epochs diverged (non-finite loss), so the
            # learning rate is halved for the longer schedule. distill.py now
            # aborts with a clear message if divergence recurs.
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--teacher", "all-mpnet-base-v2", "--epochs", "180", "--lr", "1.5e-4",
             "--ckpt-tag", "_mpnet3x", "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "distill_mpnet_tuned.json")],
        ),
        # 3j. Parameter-matched cell comparison (review round 4): the default
        #     FiLM cell has 4.1x the GRU's parameters, so the headline ablation
        #     conflates architecture with capacity. This shrinks FiLM to ~0.96M
        #     (H=192, L=3) to match the GRU's 1.03M.
        "parity": (
            Stage("parity", "Parameter-matched FiLM (0.96M) vs GRU (1.03M)",
                  "~60 min", [RESULTS / "parity.json"], fresh_check=True,
                  requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--hidden", "192", "--layers", "3", "--ckpt-tag", "_small",
             "--ckpt-dir", str(CKPT), "--out", str(RESULTS / "parity.json")],
        ),
        # 3j-bis. Cell-size sensitivity (reader question: how small can the
        #     cell be before it is useless?). Covers the tail below the 0.96M
        #     parity cell; (384,4) and (192,3) come from distill/parity.
        "size_sweep": (
            # Measured: ~3 min per (size, seed); completed sizes are cached in
            # size.json, so an interrupted sweep resumes rather than retrains.
            Stage("size_sweep", "Cell-size sensitivity: 0.30M / 0.09M / 0.04M cells",
                  "~30 min", [RESULTS / "size.json"], requires_key="h24_l2"),
            [PY, "size.py", "--seeds", *cfg.seeds,
             "--out", str(RESULTS / "size.json")],
        ),
        # 3j-ter. Over-proportional cell for the stronger teacher: width
        #     scaling already grows the mpnet cell to 3.84M (H=384, L=3, 4x the
        #     MiniLM cell); this grows it 14x further (H=1152, L=6, 53.1M)
        #     to test whether raw capacity closes the horizon collapse. Same
        #     recipe as teacher2; if training diverges, retry with --lr 1.5e-4.
        "teacher2_scaled": (
            Stage("teacher2_scaled", "mpnet teacher with an over-proportional 53M cell",
                  "~60-90 min",
                  [RESULTS / "distill_mpnet_scaled.json",
                   CKPT / "gated_film_mpnetXL_unroll1.pt"], fresh_check=True),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--teacher", "all-mpnet-base-v2", "--hidden", "1152", "--layers", "6",
             "--ckpt-tag", "_mpnetXL", "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "distill_mpnet_scaled.json")],
        ),
        # 3j-quater. Unrolled training at three seeds (consistency): the
        #     original U=4/U=8 runs were single-seed, which forced Table 8's
        #     U=1 row onto seed-41 values (horizon 13) against the paper's
        #     three-seed canon (horizon 12). Completed (U, seed) runs are
        #     cached in unroll3.json, so an interrupted sweep resumes.
        "unroll3": (
            Stage("unroll3", "Unrolled training (U=4, U=8) at three seeds",
                  "~2-3 h", [RESULTS / "unroll3.json"],
                  requires_key='"complete": true'),
            [PY, "unroll.py", "--seeds", *cfg.seeds,
             "--out", str(RESULTS / "unroll3.json")],
        ),
        # 3j-quinquies. The head-to-head a reader asks for when space
        #     preservation is NOT required: a GRU embedder trained end-to-end
        #     with a contrastive objective, encoding both corpus and queries in
        #     its own space. Contextualizes the trade-off rather than testing
        #     our claim, which is scoped to a fixed teacher space.
        "recurrent_baseline": (
            Stage("recurrent_baseline",
                  "Contrastively trained GRU embedder: SciFact + STS head-to-head",
                  "~1.5-2 h", [RESULTS / "recurrent_baseline.json"]),
            [PY, "recurrent_baseline.py", "--seeds", *cfg.seeds,
             "--limit", cfg.broader_limit,
             "--out", str(RESULTS / "recurrent_baseline.json")],
        ),
        # 3k. STS token-length statistics: how many pairs actually reach a reset.
        #     CPU-only and fast; makes the STS numbers interpretable.
        "sts_stats": (
            Stage("sts_stats", "STS token-length distribution vs reset schedule",
                  "~5 min", [RESULTS / "sts_stats.json"],
                  requires_key=f'"limit": {cfg.broader_limit}'),
            [PY, "sts_stats.py", "--limit", cfg.broader_limit,
             "--out", str(RESULTS / "sts_stats.json")],
        ),
        # 4. From-scratch comparison. Independent of the others; heaviest stage.
        "scratch": (
            Stage("scratch", "From-scratch ISE vs GRU vs Transformer (MLM -> SST-2)",
                  "~60-120 min", [scratch_out]),
            [PY, "scratch.py", "--seeds", *cfg.seeds, "--cells", *cfg.scratch_cells,
             "--out", str(scratch_out)],
        ),
        # 6. Review-round-4 additions.
        # 6a. Teacher-aligned student baselines: a causal KV-cached student and
        #     a bidirectional re-encoding student, both mapped into the
        #     teacher's space (the "why not a small aligned student?" answer).
        "students": (
            Stage("students", "Aligned students round 2: +re-grounding, +13M scaled",
                  "~3.5 h", [RESULTS / "student2.json"],
                  requires_key='"bi_scaled"'),
            [PY, "student.py", "--seeds", *cfg.seeds,
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "student2.json")],
        ),
        # 6b. State-bottleneck probes: aux carried state and token-window
        #     conditioning, all trained under the identical U=4 trajectory
        #     objective (baseline row = three-seed U=4 downstream numbers).
        "bottleneck": (
            Stage("bottleneck", "State-bottleneck probes: aux / token window / position vs baseline",
                  "~2 h", [RESULTS / "bottleneck.json"], requires_key='"pos"'),
            [PY, "bottleneck.py", "--seeds", *cfg.seeds,
             "--out", str(RESULTS / "bottleneck.json")],
        ),
        # 6c. Third teacher: gte-large (1024-d), turning the two-point teacher
        #     comparison into a three-point scaling result. Same recipe; the
        #     default cell geometry follows the teacher dim (512-hidden, 6.8M).
        "distill_gte": (
            Stage("distill_gte", "Third teacher: thenlper/gte-large (1024-d) scaling",
                  "~3-4 h",
                  [RESULTS / "distill_gte.json",
                   CKPT / "gated_film_gte_unroll1.pt",
                   CKPT / f"gated_film_gte_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True, requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--teacher", "thenlper/gte-large", "--ckpt-tag", "_gte",
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "distill_gte.json")],
        ),
        # 6d. Retrieval-while-streaming: token of first top-1/top-5 hit per
        #     policy -- the downstream *decision* metric review round 4 asked
        #     for. Eval-only on the existing seed checkpoints.
        "earlyhit": (
            Stage("earlyhit", "Retrieval-while-streaming: first-hit latency + paired bootstrap",
                  "~45 min", [RESULTS / "earlyhit.json"], needs=[ckpt],
                  requires_key='"significance"'),
            [PY, "earlyhit.py", "--ckpt", *seed_ckpts(),
             "--out", str(RESULTS / "earlyhit.json")],
        ),
        # 6e. Drift vs absolute prefix position (eval-only): does per-step
        #     drift decelerate as the prefix grows?
        "drift_position": (
            Stage("drift_position", "Drift vs absolute prefix position",
                  "~15 min", [RESULTS / "drift_position.json"], needs=[ckpt],
                  requires_key='"curves"'),
            [PY, "drift_position.py", "--ckpt", *seed_ckpts(),
             "--out", str(RESULTS / "drift_position.json")],
        ),
        # 7. Review-round-5 additions.
        # 7a. SciFact under the gte-large teacher: closes the fidelity/
        #     retention question across teachers (incl. stale + seed-only).
        "beir_gte": (
            Stage("beir_gte", "SciFact retrieval under the gte-large teacher",
                  "~40 min", [RESULTS / "beir_gte.json"],
                  needs=[CKPT / "gated_film_gte_unroll1.pt"],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--teacher", "thenlper/gte-large",
             "--ckpt", *seed_ckpts("_gte"), "--n0", "10", "--budgets", "8",
             "--max-queries", "300",
             "--out", str(RESULTS / "beir_gte.json")],
        ),
        # 6c (author response, ARR Aug 2026). A 4B-parameter teacher, run
        #     with the unchanged recipe, to answer "does the picture hold for
        #     larger embedders?". Loaded in bf16 (see Teacher.__init__).
        "distill_qwen3": (
            Stage("distill_qwen3", "Sixth teacher: Qwen/Qwen3-Embedding-4B (2560-d, bf16)",
                  "~2 h",
                  [RESULTS / "distill_qwen3_4b.json",
                   CKPT / f"gated_film_qwen3_4b_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True, requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--teacher", "Qwen/Qwen3-Embedding-4B", "--ckpt-tag", "_qwen3_4b",
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "distill_qwen3_4b.json")],
        ),
        "beir_qwen3": (
            Stage("beir_qwen3", "SciFact retrieval under the Qwen3-Embedding-4B teacher",
                  "~2 h", [RESULTS / "beir_qwen3_4b.json"],
                  needs=[CKPT / f"gated_film_qwen3_4b_unroll1_seed{cfg.seeds[-1]}.pt"],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--teacher", "Qwen/Qwen3-Embedding-4B",
             "--ckpt", *seed_ckpts("_qwen3_4b"), "--n0", "10", "--budgets", "8",
             "--max-queries", "300",
             "--out", str(RESULTS / "beir_qwen3_4b.json")],
        ),
        # 6d (camera-ready, ARR Aug 2026 reviewer uFGZ). Mamba2-, RWKV-, and
        #     xLSTM-style blocks as the update cell, parameter-matched to the
        #     0.96M FiLM cell (widths from temp/test_cells.py), under the full
        #     rollout protocol of the other Table 1 rows. One stage per cell
        #     because each needs its own --hidden; they share one result file.
        "cells_ssm_mamba2": (
            Stage("cells_ssm_mamba2", "Mamba2-style SSD update cell under full rollout",
                  "~1.5 h", [RESULTS / "cells_rollout_ssm.json"],
                  requires_key='"mamba2_ssd_h176"'),
            [PY, "cells_rollout.py", "--seeds", *cfg.seeds,
             "--cells", "mamba2_ssd", "--hidden", "176",
             "--out", str(RESULTS / "cells_rollout_ssm.json")],
        ),
        "cells_ssm_rwkv7": (
            Stage("cells_ssm_rwkv7", "RWKV-7 time-mix update cell under full rollout",
                  "~1.5 h", [RESULTS / "cells_rollout_ssm.json"],
                  requires_key='"rwkv7_h136"'),
            [PY, "cells_rollout.py", "--seeds", *cfg.seeds,
             "--cells", "rwkv7", "--hidden", "136",
             "--out", str(RESULTS / "cells_rollout_ssm.json")],
        ),
        "cells_ssm_xlstm": (
            Stage("cells_ssm_xlstm", "xLSTM sLSTM update cell under full rollout",
                  "~1.5 h", [RESULTS / "cells_rollout_ssm.json"],
                  requires_key='"xlstm_s_h160"'),
            [PY, "cells_rollout.py", "--seeds", *cfg.seeds,
             "--cells", "xlstm_s", "--hidden", "160",
             "--out", str(RESULTS / "cells_rollout_ssm.json")],
        ),
        # 7b. Long-stream retrieval with a task metric: accuracy@1/@5 at
        #     rollout depths {32,64,128,256}, incl. stale and window-64.
        "longstream": (
            Stage("longstream", "Long-stream retrieval accuracy at pre-reset depths",
                  "~45 min", [RESULTS / "longstream.json"], needs=[ckpt],
                  requires_key='"ci95"'),
            [PY, "longstream.py", "--ckpt", *seed_ckpts(),
             "--out", str(RESULTS / "longstream.json")],
        ),
        # 7c. Streaming latency at N=512/1024 over concatenated documents,
        #     where the quadratic term actually engages. This is the sample the
        #     0.87/0.75 worst-step floors are quoted from, so review 6 (W2)
        #     asks for a paired interval on the floor gap; 12 documents was too
        #     few to give one, hence 32 here.
        "streaming_long": (
            Stage("streaming_long", "Streaming embedding at N=512/1024 (concatenated docs)",
                  "~70 min", [RESULTS / "streaming_long.json"], needs=[ckpt],
                  requires_key='"floor_significance"'),
            [PY, "streaming.py", "--ckpt", str(ckpt),
             "--lengths", "512", "1024", "--docs", "32", "--repeats", "2",
             "--concat", "--stale-budgets", "4", "8",
             "--out", str(RESULTS / "streaming_long.json")],
        ),
        # 7d. State-bottleneck probes on the collapsing teacher (mpnet), where
        #     the diagnosis actually matters.
        "bottleneck_mpnet": (
            Stage("bottleneck_mpnet", "Aux / token-window / position probes under mpnet",
                  "~3 h", [RESULTS / "bottleneck_mpnet.json"],
                  requires_key='"pos"'),
            [PY, "bottleneck.py", "--seeds", *cfg.seeds,
             "--teacher", "all-mpnet-base-v2",
             "--out", str(RESULTS / "bottleneck_mpnet.json")],
        ),
        # 7e. Closed-form ridge shift predictor: how much of the residual a
        #     linear map recovers under the identical rollout protocol.
        "linear": (
            Stage("linear", "Ridge (linear) shift predictor + zero-param trajectory baselines",
                  "~20 min", [RESULTS / "linear.json"],
                  requires_key='"momentum_init"'),
            [PY, "linear.py", "--out", str(RESULTS / "linear.json")],
        ),
        # 7f. In-domain distillation ablation: SciFact abstracts as the
        #     distillation corpus, then SciFact retrieval with that cell.
        "distill_dom": (
            Stage("distill_dom", "In-domain (SciFact-corpus) distillation ablation",
                  "~45 min",
                  [RESULTS / "distill_scifact.json",
                   CKPT / "gated_film_dom_unroll1.pt",
                   CKPT / f"gated_film_dom_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True, requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--docs-source", "scifact", "--ckpt-tag", "_dom",
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "distill_scifact.json")],
        ),
        "beir_dom": (
            Stage("beir_dom", "SciFact retrieval with the in-domain cell",
                  "~25 min", [RESULTS / "beir_dom.json"],
                  needs=[CKPT / "gated_film_dom_unroll1.pt"],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--ckpt", *seed_ckpts("_dom"),
             "--n0", "10", "--budgets", "8", "--max-queries", "300",
             "--out", str(RESULTS / "beir_dom.json")],
        ),
        # 7g. U=4 with the full downstream suite, canonical implementation:
        #     settles whether unrolled training's fidelity gains transfer.
        "unroll_downstream": (
            Stage("unroll_downstream", "U=4 (canonical) with retrieval/STS + ckpts",
                  "~2 h",
                  [RESULTS / "unroll_downstream.json",
                   CKPT / f"gated_film_u4full_seed{cfg.seeds[-1]}.pt"],
                  requires_key='"retrieval_ise"'),
            [PY, "unroll_downstream.py", "--seeds", *cfg.seeds,
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "unroll_downstream.json")],
        ),
        "beir_u4": (
            Stage("beir_u4", "SciFact retrieval with the U=4 cell",
                  "~25 min", [RESULTS / "beir_u4.json"],
                  needs=[CKPT / f"gated_film_u4full_seed{cfg.seeds[-1]}.pt"],
                  requires_key='"seed_n10"'),
            [PY, "beir.py",
             "--ckpt", *[str(CKPT / f"gated_film_u4full_seed{s}.pt")
                         for s in cfg.seeds],
             "--n0", "10", "--budgets", "8", "--max-queries", "300",
             "--out", str(RESULTS / "beir_u4.json")],
        ),
        # 8. Review-round-6 additions.
        # 8a. Leakage control for the in-domain result: distill on NFCorpus
        #     (scientific abstracts DISJOINT from the SciFact index), then
        #     evaluate SciFact retrieval. Separates domain match from corpus
        #     overlap.
        "distill_dom_dis": (
            Stage("distill_dom_dis", "Disjoint in-domain distillation (NFCorpus abstracts)",
                  "~45 min",
                  [RESULTS / "distill_nfcorpus.json",
                   CKPT / "gated_film_domdis_unroll1.pt",
                   CKPT / f"gated_film_domdis_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True, requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--docs-source", "nfcorpus", "--ckpt-tag", "_domdis",
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "distill_nfcorpus.json")],
        ),
        "beir_dom_dis": (
            Stage("beir_dom_dis", "SciFact retrieval with the disjoint in-domain cell",
                  "~25 min", [RESULTS / "beir_dom_disjoint.json"],
                  needs=[CKPT / "gated_film_domdis_unroll1.pt"],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--ckpt", *seed_ckpts("_domdis"),
             "--n0", "10", "--budgets", "8", "--max-queries", "300",
             "--out", str(RESULTS / "beir_dom_disjoint.json")],
        ),
        # 8b. Symmetric in-domain control on the second dataset: distill on
        #     the FiQA corpus, evaluate FiQA retrieval.
        "distill_dom_fiqa": (
            Stage("distill_dom_fiqa", "In-domain (FiQA-corpus) distillation",
                  "~45 min",
                  [RESULTS / "distill_fiqadom.json",
                   CKPT / "gated_film_domfq_unroll1.pt",
                   CKPT / f"gated_film_domfq_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True, requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--docs-source", "fiqa", "--ckpt-tag", "_domfq",
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "distill_fiqadom.json")],
        ),
        "beir_dom_fiqa": (
            Stage("beir_dom_fiqa", "FiQA retrieval with the FiQA in-domain cell",
                  "~55 min", [RESULTS / "beir_dom_fiqa.json"],
                  needs=[CKPT / "gated_film_domfq_unroll1.pt"],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--dataset", "fiqa",
             "--ckpt", *seed_ckpts("_domfq"),
             "--n0", "10", "--budgets", "8", "--max-queries", "648",
             "--out", str(RESULTS / "beir_dom_fiqa.json")],
        ),
        # 8c. Seed-conditioned students: removes the conditioning asymmetry
        #     in the student comparison (the cell is handed a teacher seed;
        #     the plain student is not). The decisive fairness control.
        "students_seeded": (
            Stage("students_seeded", "Seed-conditioned students (2.7M + 13M, 3 seeds)",
                  "~12-14 h", [RESULTS / "student_seeded.json"],
                  requires_key='"bi_scaled_seeded"'),
            [PY, "student_seeded.py", "--seeds", *cfg.seeds,
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "student_seeded.json")],
        ),
        # 8h (round 7). The missing O(1) competitor: a teacher-aligned
        #     recurrent student, seeded like the cell, occupying exactly the
        #     cell's constant-cost niche.
        "students_rnn": (
            Stage("students_rnn", "Seeded recurrent (GRU) students (~1M + ~4M, 3 seeds)",
                  "~3-4 h", [RESULTS / "student_rnn.json"],
                  requires_key='"gru_scaled_seeded"'),
            [PY, "student_rnn.py", "--seeds", *cfg.seeds,
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "student_rnn.json")],
        ),
        # 8i (round 7). Long-query retrieval with real judgments: ArguAna
        #     queries are full argument passages (~190 subword tokens), so the
        #     mean rollout finally exceeds the drift horizon on a real task.
        "beir_arguana": (
            Stage("beir_arguana", "ArguAna (BEIR): long real queries past the horizon",
                  "~45 min", [RESULTS / "beir_arguana.json"], needs=[ckpt],
                  requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "arguana", "--ckpt", *seed_ckpts(),
             "--n0", "10", "--budgets", "8", "--max-queries", "300",
             "--out", str(RESULTS / "beir_arguana.json")],
        ),
        # 8i-bis (round 11). The cadence-vs-depth control both reviews call
        #     blocking. On ArguAna the learned update ties a training-free hold
        #     cadence at K=8 (+0.2, p=0.53), but K=8 was selected once on
        #     64-step MiniLM rollouts and the teacher's per-token shift at
        #     depth ~190 is a third of its value at depth ~20 (App. R). So the
        #     tie is equally consistent with "the schedule dominates at depth"
        #     and with "K=8 is simply over-dense for this depth". Sweeping K
        #     separates them: if the cell-over-stale margin reappears as the
        #     cadence sparsens -- as it does intrinsically, +0.016 at K=8 to
        #     +0.069 at K=32 -- then the attribution is set by cadence density
        #     relative to the local shift, not by depth as such.
        #     Round 11 review 3 asks for K=4 as well (a denser cadence bounds
        #     the sweep from below) and for the full ArguAna test split rather
        #     than a 300-query subsample, since the width of the [-0.47, +0.91]
        #     interval is what the equivalence argument leans on. ArguAna's
        #     test split is 1,406 queries; --max-queries 0 takes all of them.
        "arguana_ksweep": (
            Stage("arguana_ksweep",
                  "ArguAna K-sweep: is the tie depth or cadence density?",
                  "~9 h", [RESULTS / "beir_arguana_ksweep.json"], needs=[ckpt],
                  # K=4 exists only in the full-split re-run, so this
                  # invalidates the earlier 300-query, K>=8 file automatically.
                  requires_key='"hybrid4_n10"'),
            [PY, "beir.py", "--dataset", "arguana", "--ckpt", *seed_ckpts(),
             "--n0", "10", "--budgets", "4", "8", "16", "32", "64",
             "--max-queries", "0",
             "--out", str(RESULTS / "beir_arguana_ksweep.json")],
        ),
        # 8i-quinquies (round 11, review 4 row 4). The long-query finding rests
        #     on ArguAna alone, whose structure (query and document are both
        #     long passages of the same genre) is confounded with length: we
        #     cannot currently tell "long queries" from "ArguAna-like". This
        #     separates them within the data we already have, by truncating
        #     ArguAna queries to a ladder of depths at fixed K=8 and
        #     recomputing the schedule/update split at each. If the schedule
        #     share climbs monotonically toward 100% with truncation length on
        #     one fixed corpus and task, the finding is about depth; if it is
        #     flat at 100% from 32 tokens up, it is about the dataset.
        #     Truncation is applied to the query *text* at the teacher's token
        #     level, so the teacher baseline and the incremental policies see
        #     the same input at every depth and the 2x2 stays input-matched.
        # Reviews 9/11/13: does the ArguAna null hold under a teacher that
        #   drifts differently? gte-large (horizon >64) is the decisive case.
        "arguana_mpnet": (
            Stage("arguana_mpnet",
                  "ArguAna full split under the mpnet teacher",
                  "~2 h", [RESULTS / "beir_arguana_mpnet.json"],
                  needs=[CKPT / "gated_film_mpnet_unroll1.pt"], requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "arguana",
             "--teacher", "all-mpnet-base-v2",
             "--ckpt", *seed_ckpts("_mpnet"),
             "--n0", "10", "--budgets", "8", "--max-queries", "0",
             "--out", str(RESULTS / "beir_arguana_mpnet.json")],
        ),
        # Reviews 9/11/13: does the ArguAna null hold under a teacher that
        #   drifts differently? gte-large (horizon >64) is the decisive case.
        "arguana_bge": (
            Stage("arguana_bge",
                  "ArguAna full split under the bge teacher",
                  "~2 h", [RESULTS / "beir_arguana_bge.json"],
                  needs=[CKPT / "gated_film_bge_unroll1.pt"], requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "arguana",
             "--teacher", "BAAI/bge-base-en-v1.5",
             "--ckpt", *seed_ckpts("_bge"),
             "--n0", "10", "--budgets", "8", "--max-queries", "0",
             "--out", str(RESULTS / "beir_arguana_bge.json")],
        ),
        # Reviews 9/11/13: does the ArguAna null hold under a teacher that
        #   drifts differently? gte-large (horizon >64) is the decisive case.
        "arguana_e5": (
            Stage("arguana_e5",
                  "ArguAna full split under the e5 teacher",
                  "~2 h", [RESULTS / "beir_arguana_e5.json"],
                  needs=[CKPT / "gated_film_e5_unroll1.pt"], requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "arguana",
             "--teacher", "intfloat/e5-base-v2",
             "--ckpt", *seed_ckpts("_e5"),
             "--n0", "10", "--budgets", "8", "--max-queries", "0",
             "--out", str(RESULTS / "beir_arguana_e5.json")],
        ),
        # Reviews 9/11/13: does the ArguAna null hold under a teacher that
        #   drifts differently? gte-large (horizon >64) is the decisive case.
        "arguana_gte": (
            Stage("arguana_gte",
                  "ArguAna full split under the gte teacher",
                  "~2 h", [RESULTS / "beir_arguana_gte.json"],
                  needs=[CKPT / "gated_film_gte_unroll1.pt"], requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "arguana",
             "--teacher", "thenlper/gte-large",
             "--ckpt", *seed_ckpts("_gte"),
             "--n0", "10", "--budgets", "8", "--max-queries", "0",
             "--out", str(RESULTS / "beir_arguana_gte.json")],
        ),
        # Review 11 R4 / 12 R11: intervals on the SciFact and FiQA shares and
        #   on the hold-above-teacher residual. Pure re-analysis of stored
        #   per-query scores; seconds on CPU.
        "share_ci": (
            Stage("share_ci",
                  "Bootstrap CIs for the decomposition shares and residuals",
                  "~1 min", [RESULTS / "share_ci.json"],
                  needs=[RESULTS / "beir.json", RESULTS / "beir_fiqa.json",
                         RESULTS / "beir_arguana_ksweep.json"],
                  requires_key='"matched_margin_ci95"'),
            [PY, "share_ci.py", "--iters", "10000",
             "--out", str(RESULTS / "share_ci.json")],
        ),
        "arguana_depth": (
            Stage("arguana_depth",
                  "ArguAna depth ladder: is the null depth or dataset?",
                  "~3 h", [RESULTS / "beir_arguana_depth.json"], needs=[ckpt],
                  requires_key='"truncate_tokens"'),
            [PY, "beir.py", "--dataset", "arguana", "--ckpt", *seed_ckpts(),
             "--n0", "10", "--budgets", "8", "--max-queries", "0",
             "--truncate-queries", "32", "64", "96", "128", "172",
             "--out", str(RESULTS / "beir_arguana_depth.json")],
        ),
        # 9c-bis. Bootstrap intervals on the depth-ladder shares (review 6 W3).
        #     Pure re-analysis: the depth run already stores per-query nDCG at
        #     every truncation, so this resamples queries rather than re-scoring
        #     anything. Seconds on CPU, no GPU and no checkpoint needed.
        "depth_ci": (
            Stage("depth_ci",
                  "Bootstrap CIs for the depth-ladder schedule shares",
                  "~1 min", [RESULTS / "depth_ci.json"],
                  needs=[RESULTS / "beir_arguana_depth.json",
                         RESULTS / "beir_arguana_ksweep.json"],
                  requires_key='"sched_pct_ci95"'),
            [PY, "depth_ci.py", "--iters", "10000",
             "--out", str(RESULTS / "depth_ci.json")],
        ),
        # 8i-ter (round 11, review 3 row 4). Contribution 2 claims no better
        #     update closes the ArguAna gap, but the two updates that actually
        #     beat the cell on SciFact -- the seeded bidirectional student
        #     (62.9) and the seeded GRU (63.1) -- were never scored on ArguAna.
        #     They are the variants with headroom, so the claim currently rests
        #     on untested cases. This re-evaluates existing checkpoints at the
        #     identical call budget; no training.
        #     NOT YET RUNNABLE. student_seeded.py evaluates SciFact only
        #     (its args are --ckpt-dir/--out/--seeds/--teacher); adding ArguAna
        #     needs a --dataset switch threaded through its eval path, mirroring
        #     beir.py's. Left defined and out of ORDER so `full` does not
        #     invoke a command that would fail on unknown flags.
        "students_arguana": (
            Stage("students_arguana",
                  "Seeded bi + GRU students on ArguAna (NEEDS SCRIPT WORK)",
                  "~40 min", [RESULTS / "student_arguana.json"],
                  needs=[ckpt], requires_key='"gru_scaled_seeded"'),
            [PY, "student_seeded.py", "--dataset", "arguana",
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "student_arguana.json")],
        ),
        # 8i-quater (round 11, review 3 row 5). Contribution 3 says a *learned*
        #     update owns the between-call embedding, but the floor is measured
        #     only against hold cadences. Two training-free update rules bound
        #     that claim from the other side: the closed-form ridge map already
        #     in App. cellablate (scored so far only on final-embedding
        #     retrieval, never on worst-step fidelity), and a running mean over
        #     the static token rows the pipeline already loads.
        #     NOT YET RUNNABLE. streaming.py has no --policies switch; its
        #     policy list is built inline (around line 190) from ise/hybrid/
        #     stale/adaptive. Two rules must be added there: the closed-form
        #     ridge map of linear.py applied between calls, and a running
        #     mean over the teacher's static token rows scaled for mean
        #     pooling. Both are training-free and O(1), so they belong in the
        #     same fidelity/floor comparison as the cell.
        "trainfree_updates": (
            Stage("trainfree_updates",
                  "Training-free between-call updates (NEEDS SCRIPT WORK)",
                  "~30 min", [RESULTS / "trainfree_updates.json"],
                  needs=[ckpt], requires_key='"static_mean"'),
            [PY, "streaming.py", "--ckpt", *seed_ckpts(),
             "--lengths", "128", "256", "1024",
             "--out", str(RESULTS / "trainfree_updates.json")],
        ),
        # 8j (round 7). Cross-teacher margins on the second dataset, so the
        #     five-teacher significance story does not rest on SciFact alone.
        "beir_fiqa_mpnet": (
            Stage("beir_fiqa_mpnet", "FiQA under the mpnet teacher",
                  "~60 min", [RESULTS / "beir_fiqa_mpnet.json"],
                  needs=[CKPT / "gated_film_mpnet_unroll1.pt"],
                  requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "fiqa",
             "--teacher", "all-mpnet-base-v2", "--ckpt", *seed_ckpts("_mpnet"),
             "--n0", "10", "--budgets", "8", "--max-queries", "648",
             "--out", str(RESULTS / "beir_fiqa_mpnet.json")],
        ),
        "beir_fiqa_bge": (
            Stage("beir_fiqa_bge", "FiQA under the bge-base teacher",
                  "~60 min", [RESULTS / "beir_fiqa_bge.json"],
                  needs=[CKPT / "gated_film_bge_unroll1.pt"],
                  requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "fiqa",
             "--teacher", "BAAI/bge-base-en-v1.5", "--ckpt", *seed_ckpts("_bge"),
             "--n0", "10", "--budgets", "8", "--max-queries", "648",
             "--out", str(RESULTS / "beir_fiqa_bge.json")],
        ),
        "beir_fiqa_e5": (
            Stage("beir_fiqa_e5", "FiQA under the e5-base-v2 teacher",
                  "~60 min", [RESULTS / "beir_fiqa_e5.json"],
                  needs=[CKPT / "gated_film_e5_unroll1.pt"],
                  requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "fiqa",
             "--teacher", "intfloat/e5-base-v2", "--ckpt", *seed_ckpts("_e5"),
             "--n0", "10", "--budgets", "8", "--max-queries", "648",
             "--out", str(RESULTS / "beir_fiqa_e5.json")],
        ),
        "beir_fiqa_gte": (
            Stage("beir_fiqa_gte", "FiQA under the gte-large teacher",
                  "~90 min", [RESULTS / "beir_fiqa_gte.json"],
                  needs=[CKPT / "gated_film_gte_unroll1.pt"],
                  requires_key='"per_query"'),
            [PY, "beir.py", "--dataset", "fiqa",
             "--teacher", "thenlper/gte-large", "--ckpt", *seed_ckpts("_gte"),
             "--n0", "10", "--budgets", "8", "--max-queries", "648",
             "--out", str(RESULTS / "beir_fiqa_gte.json")],
        ),
        # 8k (round 7). A real streaming consumer: incremental knowledge
        #     selection on Wizard of Wikipedia -- the candidate knowledge is
        #     ranked while the user's utterance streams in, against the
        #     dataset's human-checked gold sentence.
        "wow": (
            Stage("wow", "Wizard-of-Wikipedia streaming knowledge selection",
                  "~40 min", [RESULTS / "wow.json"], needs=[ckpt],
                  requires_key='"random_acc1"'),
            [PY, "wow.py", "--ckpt", *seed_ckpts(),
             "--out", str(RESULTS / "wow.json")],
        ),
        # 8d. Long-stream retrieval v2: 3000-doc corpus so accuracy does not
        #     saturate at depth 255, plus paired bootstrap significance on
        #     every hybrid-vs-stale/adaptive gap (the round-5 table had
        #     neither). Writes a new file; longstream.json is superseded.
        "longstream2": (
            Stage("longstream2", "Long-stream retrieval v2: 3000 docs + significance",
                  "~1.5 h", [RESULTS / "longstream2.json"], needs=[ckpt],
                  requires_key='"significance"'),
            [PY, "longstream.py", "--ckpt", *seed_ckpts(),
             "--n-corpus", "3000", "--n-queries", "300",
             "--out", str(RESULTS / "longstream2.json")],
        ),
        # 8d-bis. Long-stream at n=1000 queries: settles whether the ~+1.3
        #     pre-reset edge over stale-8 (p=0.18 at n=300) is real.
        "longstream_n1000": (
            Stage("longstream_n1000", "Long-stream retrieval at 1000 queries (significance)",
                  "~5 h", [RESULTS / "longstream_n1000.json"], needs=[ckpt],
                  requires_key='"significance"'),
            [PY, "longstream.py", "--ckpt", *seed_ckpts(),
             "--n-corpus", "3000", "--n-queries", "1000",
             "--out", str(RESULTS / "longstream_n1000.json")],
        ),
        # 8d-ter. Copy-forward null horizons + residual domain-stability:
        #     the training-free null the cell's horizon must be separated
        #     from, and whether the residual diagnostic is domain-stable.
        "nullhorizon": (
            Stage("nullhorizon", "Null (copy-forward) horizons + residual stability, 5 teachers",
                  "~30 min", [RESULTS / "nullhorizon.json"],
                  needs=[RESULTS / "distill_e5.json", RESULTS / "distill_bge.json"],
                  requires_key='"residuals"'),
            [PY, "nullhorizon.py",
             "--out", str(RESULTS / "nullhorizon.json")],
        ),
        # 8e. Serving throughput: the regime where teacher calls, not
        #     single-stream wall clock, bind (many concurrent streams sharing
        #     one teacher GPU). Legitimizes the matched-calls axis.
        "serving": (
            Stage("serving", "Multi-stream serving: sustainable streams per policy",
                  "~15 min", [RESULTS / "serving.json"], needs=[ckpt],
                  requires_key='"streams_multiplier_hybrid8"'),
            [PY, "serving.py", "--ckpt", str(ckpt),
             "--out", str(RESULTS / "serving.json")],
        ),
        # 8f. Trained static embedder (Model2Vec potion-base-8M): replaces the
        #     untrained token-lookup proxy as the static-family comparator.
        "potion": (
            Stage("potion", "Trained static embedder baseline (potion-base-8M)",
                  "~15 min", [RESULTS / "potion.json"],
                  requires_key='"stsb_spearman"'),
            [PY, "potion.py", "--out", str(RESULTS / "potion.json")],
        ),
        # 8g. Teachers four and five: e5-base-v2 and bge-base-en-v1.5 take the
        #     horizon-vs-teacher-shift observation from n=3 to n=5.
        "distill_e5": (
            Stage("distill_e5", "Fourth teacher: intfloat/e5-base-v2 (768-d)",
                  "~40 min",
                  [RESULTS / "distill_e5.json",
                   CKPT / "gated_film_e5_unroll1.pt",
                   CKPT / f"gated_film_e5_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True, requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--teacher", "intfloat/e5-base-v2", "--ckpt-tag", "_e5",
             "--ckpt-dir", str(CKPT), "--out", str(RESULTS / "distill_e5.json")],
        ),
        "beir_e5": (
            Stage("beir_e5", "SciFact retrieval under the e5-base-v2 teacher",
                  "~25 min", [RESULTS / "beir_e5.json"],
                  needs=[CKPT / "gated_film_e5_unroll1.pt"],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--teacher", "intfloat/e5-base-v2",
             "--ckpt", *seed_ckpts("_e5"), "--n0", "10", "--budgets", "8",
             "--max-queries", "300",
             "--out", str(RESULTS / "beir_e5.json")],
        ),
        "distill_bge": (
            Stage("distill_bge", "Fifth teacher: BAAI/bge-base-en-v1.5 (768-d)",
                  "~40 min",
                  [RESULTS / "distill_bge.json",
                   CKPT / "gated_film_bge_unroll1.pt",
                   CKPT / f"gated_film_bge_unroll1_seed{cfg.seeds[-1]}.pt"],
                  fresh_check=True, requires_key='41,\n    42,\n    43'),
            [PY, "distill.py", "--seeds", *cfg.seeds, "--cell", "gated_film",
             "--teacher", "BAAI/bge-base-en-v1.5", "--ckpt-tag", "_bge",
             "--ckpt-dir", str(CKPT), "--out", str(RESULTS / "distill_bge.json")],
        ),
        "beir_bge": (
            Stage("beir_bge", "SciFact retrieval under the bge-base teacher",
                  "~25 min", [RESULTS / "beir_bge.json"],
                  needs=[CKPT / "gated_film_bge_unroll1.pt"],
                  requires_key='"seed_n10"'),
            [PY, "beir.py", "--teacher", "BAAI/bge-base-en-v1.5",
             "--ckpt", *seed_ckpts("_bge"), "--n0", "10", "--budgets", "8",
             "--max-queries", "300",
             "--out", str(RESULTS / "beir_bge.json")],
        ),
        # 9a (round 8). The decisive missing baseline: a causal student with a
        #     KV cache reuses its prefix exactly (no drift, no resets) at O(N)
        #     per step and a tiny constant, so it is the sharpest competitor to
        #     the cell in the long-stream regime. It is also the one student
        #     family that never received the seeded + re-grounded treatment,
        #     and its round-5 numbers came from a harness that gave it 8.7x
        #     fewer optimizer updates than the size-matched bidirectional
        #     student (see student_causal.py). Both are fixed here.
        "students_causal": (
            Stage("students_causal",
                  "Causal KV-cached students: step-matched, pooling ablation, "
                  "seeded + re-grounded (2.7M/2.8M/13.2M, 3 seeds)",
                  "~6-8 h", [RESULTS / "student_causal.json"],
                  requires_key='"causal_seeded_scaled"'),
            [PY, "student_causal.py", "--seeds", *cfg.seeds,
             "--ckpt-dir", str(CKPT),
             "--out", str(RESULTS / "student_causal.json")],
        ),
        # 9b (round 8). The regime map: at a matched teacher-call budget, what
        #     each candidate costs and delivers as a function of stream length,
        #     and the crossover length beyond which the cell's O(1) update is
        #     cheaper per stream than the causal student's O(N) cached step.
        #     Needs the seeded student checkpoints; missing ones are skipped.
        "crossover": (
            Stage("crossover", "Regime map: latency / fidelity vs stream length "
                               "at a matched call budget, and the O(1)-vs-O(N) crossover",
                  "~40 min", [RESULTS / "crossover.json"], needs=[ckpt],
                  requires_key='"crossover"'),
            [PY, "crossover.py", "--ckpt", *seed_ckpts(),
             "--ckpt-dir", str(CKPT), "--cpu",
             "--out", str(RESULTS / "crossover.json")],
        ),
        # 5. Figures from the JSON produced above.
        "figures": (
            Stage("figures", "Render vector-PDF figures into paper/assets/",
                  "~10 s", [ASSETS / "rollout_drift.pdf", ASSETS / "pareto_reencode.pdf",
                            ASSETS / "speed_scaling.pdf", ASSETS / "teacher_law.pdf"],
                  needs=[distill_out, speed_out, RESULTS / "linear.json"]),
            [PY, "make_figures.py", "--distill", str(distill_out), "--speed", str(speed_out),
             "--linear", str(RESULTS / "linear.json"),
             "--outdir", str(ASSETS)],
        ),
    }


ORDER = ["speed", "distill", "streaming", "streaming_cpu", "broader", "tau",
         "stability", "teacher2", "teacher2_scaled", "broader_mpnet", "beir",
         "beir_fiqa", "beir_mpnet", "stale", "gru_rollout", "parity",
         "size_sweep", "unroll3", "recurrent_baseline", "placement", "delta_probe", "content_reset", "sts_stats", "scratch",
         "beir_tau09", "beir_fiqa_tau09", "cells_rollout",
         "students", "bottleneck", "distill_gte", "earlyhit", "drift_position",
         "beir_gte", "distill_qwen3", "beir_qwen3", "cells_ssm_mamba2", "cells_ssm_rwkv7", "cells_ssm_xlstm", "longstream", "streaming_long", "bottleneck_mpnet",
         "linear", "distill_dom", "beir_dom", "unroll_downstream", "beir_u4",
         # round 6: cheap evals first, the two long trainings last, so an
         # interrupted night still lands the small results.
         "serving", "potion", "longstream2",
         "distill_dom_dis", "beir_dom_dis",
         "distill_dom_fiqa", "beir_dom_fiqa",
         "distill_e5", "beir_e5", "distill_bge", "beir_bge",
         "nullhorizon",
         "teacher2_tuned", "students_seeded", "longstream_n1000",
         # round 7: cheap evals first, the GRU-student training last.
         "beir_arguana", "wow",
         "beir_fiqa_mpnet", "beir_fiqa_bge", "beir_fiqa_e5", "beir_fiqa_gte",
         "students_rnn",
         # round 8: the two cheap re-runs are covered by their bumped
         # requires_key above; the causal students train before crossover,
         # which loads their checkpoints.
         "students_causal", "crossover",
         # round 11: the cadence-vs-depth control the reviews call blocking.
         # No training; reuses the deployed checkpoints. students_arguana and
         # trainfree_updates are defined above but held out of ORDER until
         # their runner scripts gain the switches noted there.
         "arguana_ksweep", "arguana_depth",
         # round 13: the null under the other four teachers, then the
         # CPU re-analyses that depend on the retrieval runs above.
         "arguana_mpnet", "arguana_bge", "arguana_e5", "arguana_gte",
         "share_ci",
         # round 12 (review 6 W3): re-analysis of the depth run, must follow it.
         "depth_ci",
         "figures"]

# Stages beyond the core paper results; run explicitly or via `full`.
EXTRA = {"tau", "stability", "teacher2", "teacher2_scaled", "broader_mpnet",
         "beir", "beir_fiqa", "beir_mpnet", "stale", "gru_rollout", "parity",
         "size_sweep", "unroll3", "recurrent_baseline", "placement", "delta_probe", "content_reset", "sts_stats", "teacher2_tuned",
         "beir_tau09", "beir_fiqa_tau09", "cells_rollout",
         "students", "bottleneck", "distill_gte", "earlyhit", "drift_position",
         "beir_gte", "distill_qwen3", "beir_qwen3", "cells_ssm_mamba2", "cells_ssm_rwkv7", "cells_ssm_xlstm", "longstream", "streaming_long", "bottleneck_mpnet",
         "linear", "distill_dom", "beir_dom", "unroll_downstream", "beir_u4",
         "serving", "potion", "longstream2", "longstream_n1000", "nullhorizon",
         "distill_dom_dis", "beir_dom_dis",
         "distill_dom_fiqa", "beir_dom_fiqa",
         "distill_e5", "beir_e5", "distill_bge", "beir_bge",
         "students_seeded", "students_rnn", "beir_arguana", "wow",
         "beir_fiqa_mpnet", "beir_fiqa_bge", "beir_fiqa_e5", "beir_fiqa_gte",
         "students_causal", "crossover"}


def check_env() -> None:
    banner("Environment")
    code = (
        "import torch\n"
        "print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())\n"
        "if torch.cuda.is_available():\n"
        "    p = torch.cuda.get_device_properties(0)\n"
        "    print('GPU  :', torch.cuda.get_device_name(0), f'| {p.total_memory/1e9:.1f} GB')\n"
    )
    if subprocess.run([PY, "-c", code]).returncode != 0:
        print("(!) torch/CUDA check failed; experiment scripts will error if the env isn't ready.")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Reproduce the ISE paper experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", default="full",
                    choices=["full", "quick", *ORDER],
                    help="stage to run, or full/quick for everything (default: full)")
    ap.add_argument("--force", action="store_true",
                    help="re-run stages even if their outputs already exist")
    ap.add_argument("--list", action="store_true",
                    help="show stages and what would run, then exit")
    args = ap.parse_args()

    for d in (RESULTS, LOGS, CKPT, ASSETS):
        d.mkdir(parents=True, exist_ok=True)

    cfg = make_config(quick=(args.mode == "quick"))
    stages = build_stages(cfg)
    selected = ORDER if args.mode in ("full", "quick") else [args.mode]

    if args.list:
        banner("Stages")
        will_refresh: set[Path] = set()   # outputs regenerated earlier in this run
        for name in ORDER:
            st, _ = stages[name]
            downstream = bool(set(st.needs) & will_refresh)
            runs = name in selected and (args.force or not st.satisfied() or downstream)
            if runs:
                will_refresh.update(st.outputs)
            sel = "*" if name in selected else " "
            note = "  (input refreshed upstream)" if downstream and st.satisfied() else ""
            print(f" {sel} [{'RUN ' if runs else 'skip'}] {name:<14} {st.est:>12}  {st.desc}{note}")
        print("\n  * = selected by the current mode; 'skip' = output already present and current.")
        print("  Stale distill.json (pre-metric-fix schema) counts as missing.")
        return

    check_env()
    start = time.time()
    ran, skipped = [], []

    for name in selected:
        # Re-resolve the stage table before each stage: seed_ckpts() falls back
        # to the single aggregate checkpoint when the per-seed files are absent,
        # and on a fresh tree they only appear after the distill stage runs. A
        # table baked once at startup would silently pin every downstream
        # evaluation to the one-checkpoint fallback.
        st, cmd = build_stages(cfg)[name]
        if not args.force and st.satisfied():
            stage_line(f"SKIP {name} (outputs present; use --force to redo)")
            skipped.append(name)
            continue
        missing = [p for p in st.needs if not p.exists()]
        if missing:
            raise SystemExit(
                f"*** Stage '{name}' needs {', '.join(map(str, missing))}, which is missing.\n"
                f"    Run:  python run_all.py {'distill' if 'ckpt' in str(missing[0]) else 'full'}")
        banner(f"{name} -- {st.desc}  ({st.est})")
        stage_line(" ".join(cmd))
        run_logged(cmd, LOGS / f"{name}.log")
        ran.append(name)

    banner(f"Done in {int((time.time() - start) / 60)} min")
    print(f"  ran:     {', '.join(ran) or '(nothing)'}")
    print(f"  skipped: {', '.join(skipped) or '(none)'}")
    print("  Verify the paper against these results with:")
    print(f"    {PY} check_paper_numbers.py")
    for f in ("distill.json", "streaming.json", "streaming_cpu.json",
              "broader.json", "scratch.json", "speed.json"):
        print(f"    {RESULTS / f}")


if __name__ == "__main__":
    main()
