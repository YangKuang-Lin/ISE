# Embedding-Shift Distillation: Incremental Sentence Embeddings with Periodic Re-grounding

## Abstract

Transformer sentence encoders recompute the *entire* prefix whenever a token is appended, which is wasteful in streaming settings that need an embedding after every token. A recurrent encoder removes that cost, but abandons the teacher's embedding space, which a populated nearest-neighbor index cannot afford to lose. We propose *embedding-shift distillation*: a compact gated cell learns the change each token induces in the frozen teacher's output, and an amortized hybrid encoder applies that update at constant cost between periodic re-encodings. Re-encoding once every eight tokens, it retains 93.7–97.8% of the teacher's retrieval quality across five teachers while spending calls on 9% of query tokens, and 98.3% when distilled on held-out in-domain text. Under every teacher it beats holding the last re-encoding at the same cadence. That budget, shared by any policy, sets serving capacity: 8× more concurrent streams per GPU. Between calls, the learned update holds a worst-step fidelity floor no hold cadence reaches, surfacing as earlier stable decisions. The advantage is regime-dependent: a seeded re-encoding student slightly exceeds the cell on short queries, and on long queries the final embedding is set by the last re-encoding alone, so the cell's regime is where constant per-token cost binds. These results indicate that a preserved embedding space can be streamed at constant cost between calls, once the budget is sized per teacher from its copy-forward residual.

## Project Layout

```
code/
├── README.md                     this file
├── requirements.txt              dependency bounds + the exact tested versions
└── src/
    ├── run_all.py                orchestrates every stage (resumable, logged)
    ├── make_figures.py           renders the four vector-PDF figures
    ├── distill.py                main distillation + ablations + adaptive sweep (3 seeds);
    │                             --teacher selects any of the six teachers,
    │                             --docs-source selects the distillation corpus
    ├── streaming.py              end-to-end streaming latency / memory / fidelity (GPU+CPU)
    ├── serving.py                multi-stream serving throughput: sustainable concurrent
    │                             streams per policy, with a closed-loop validation
    ├── broader.py                seven-task STS suite (teacher / ISE / hybrid / adaptive / static)
    ├── beir.py                   retrieval with incrementally embedded queries on
    │                             SciFact / FiQA / ArguAna (--dataset), for any teacher
    │                             (--teacher); policies teacher / seed-only / stale /
    │                             hybrid / adaptive
    ├── wow.py                    Wizard-of-Wikipedia streaming knowledge selection:
    │                             the end-to-end decision task (auto-downloads the data)
    ├── longstream.py             long-stream retrieval at pre-reset depths (31 to 255),
    │                             with paired-bootstrap significance
    ├── tau.py                    adaptive-threshold sensitivity across rollout lengths
    ├── stale.py                  stale-cadence (re-encode and hold) baseline
    ├── nullhorizon.py            copy-forward (training-free) null horizons for all five
    │                             teachers + residual domain-stability check
    ├── size.py                   cell-capacity ladder (0.037M to 4.29M)
    ├── cells_rollout.py          baseline cells (concat MLP / gated residual) and the
    │                             Mamba2-style SSD / sLSTM / RWKV-7 cells under full rollout
    ├── unroll.py                 multi-step (unrolled) training, 3 seeds
    ├── placement.py              reset-placement isolation (gap-multiset permutations)
    ├── delta_probe.py            cell update magnitude by token type
    ├── content_reset.py          content-aware vs positional reset policies
    ├── sts_stats.py              STS sentence-length statistics
    ├── scratch.py                from-scratch ISE / GRU / Transformer comparison (MLM + SST-2)
    ├── recurrent_baseline.py     end-to-end contrastive GRU embedder
    ├── potion.py                 trained static embedder baseline (Model2Vec potion-base-8M)
    ├── speed.py                  per-token microbenchmark (subprocess-isolated timing)
    ├── student.py                teacher-aligned student baselines (0.74M to 13.0M)
    │                             plus student re-grounding at the hybrid call budget
    ├── student_seeded.py         seed-conditioned bidirectional students (2.84M, 13.2M):
    │                             the conditioning-vs-architecture control
    ├── student_rnn.py            seeded recurrent (GRU) students (~1M, ~4M): the O(1)
    │                             competitor in the cell's own cost class
    ├── bottleneck.py             state-bottleneck probes (aux state, windowed input,
    │                             wider FiLM), also under mpnet (--teacher)
    ├── linear.py                 closed-form ridge shift predictor under full rollout
    ├── drift_position.py         drift vs absolute prefix position (fresh-start control)
    ├── earlyhit.py               first-hit latency probe on incremental queries
    ├── crossover.py              regime map: latency / fidelity vs stream length at a matched call
    │                             budget, and the O(1)-vs-O(N) crossover (students vs the cell)
    ├── student_causal.py         causal KV-cached students (step-matched, pooling ablation,
    │                             seeded + re-grounded), the Table 5 correction
    ├── depth_ci.py               ArguAna depth-ladder bootstrap intervals
    ├── share_ci.py               hold-vs-teacher share intervals on ArguAna
    ├── unroll_downstream.py      canonical U=4 unrolled cell + full downstream eval
    └── experiments/
        ├── cells.py              update-cell architectures + default geometry (single source of truth)
        ├── distill_data.py       frozen teacher wrapper (incl. per-teacher instruction
        │                         prefixes) + shift-triple / trajectory construction
        ├── encoders.py           from-scratch encoder wrappers (ISE / GRU / Transformer)
        ├── students.py           standalone and seeded student encoder architectures
        ├── data.py               MLM / classification datasets and tokenizer helpers
        └── utils.py              seeds, environment logging, aggregation, result IO
```

## Reproducing the paper

```bash
pip install -r requirements.txt
cd src
python run_all.py        # every stage needed for the paper (resumable)
python run_all.py --list # show stages, outputs, and what would run
python run_all.py quick  # 1-seed sanity pass
python run_all.py wow    # a single stage
python run_all.py cells_ssm_mamba2   # e.g. one of the linear-time update cells
```

Results are written to `src/experiments/results/*.json`, checkpoints to `src/experiments/results/ckpt/`, logs to `src/experiments/results/logs/`, and figures to `../paper/assets/` relative to `src/` (created on first run; pass `--outdir` to `make_figures.py` to put them elsewhere). Every trained configuration runs for seeds 41-43; downstream evaluations are run once per seed checkpoint and averaged. Each stage log opens with the Python, PyTorch, CUDA, GPU and dependency versions that stage ran under, so a re-run can be compared against the configuration recorded in `requirements.txt`.

All corpora and model weights are fetched automatically on first use (Hugging Face `datasets` / `sentence-transformers`, and a direct download for Wizard of Wikipedia), so no manual data preparation is needed.

Target hardware: a single 12 GB GPU and roughly 64 GB of system RAM. All results in the paper were produced on an RTX 4070 Ti with an Intel Core i7-13700K (16 cores / 24 threads); the CPU streaming latencies depend on that CPU and on PyTorch's default thread count, both of which each stage log records. `run_all.py` is stage-resumable, so an interrupted run continues where it stopped.


## Citation

```bibtex
@inproceedings{lin-2027-embedding,
    title = "Embedding-Shift Distillation: Incremental Sentence Embeddings with Periodic Re-grounding",
    author = "Lin, Yang-Kuang",
    editor = "Nissim, Malvina and
      Reichart, Roi and
      Tonelli, Sara",
    booktitle = "Proceedings of the 20th Conference of the {E}uropean Chapter of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = mar,
    year = "2027",
    address = "Athens, Greece",
    publisher = "Association for Computational Linguistics"
}
```
