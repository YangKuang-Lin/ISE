"""Head-to-head against a recurrent embedder trained end-to-end.

The paper argues that distilling a *fixed* teacher is the right move when an
embedding space must be preserved. The natural counter-question is what one
gives up when it need not be: a recurrent encoder trained directly with a
contrastive objective is also constant-cost per token, owes nothing to a
teacher, and needs no re-encoding at all -- but it defines its own space, so a
populated index built with the teacher would have to be rebuilt.

This script trains that baseline and evaluates it on the same tasks:

  * GRU sentence encoder, mean-pooled, over the teacher's WordPiece vocabulary
    so the comparison isolates architecture and objective rather than
    tokenization.
  * InfoNCE over in-batch negatives on SNLI entailment pairs, the standard
    Sentence-BERT recipe.
  * SciFact nDCG@10 with the model encoding BOTH corpus and queries, since it
    has no teacher to encode the corpus for it, and the seven-task STS suite.

Usage:
    python recurrent_baseline.py --seeds 41 42 43
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from torch import nn, optim

from experiments.distill_data import Teacher
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)
from beir import load_beir, ndcg_at_k
from broader import TASKS


@dataclass(frozen=True)
class ContrastiveConfig:
    tokenizer: str = "all-MiniLM-L6-v2"   # vocabulary shared with the teacher
    emb_dim: int = 384                     # output dim matches the teacher's
    hidden_dim: int = 384
    num_layers: int = 2
    max_tokens: int = 64
    batch_size: int = 256
    lr: float = 3e-4
    epochs: int = 3
    max_pairs: int = 200_000
    temperature: float = 0.05


class GRUEmbedder(nn.Module):
    """Mean-pooled GRU sentence encoder with an L2-normalized output.

    ``init_embeddings`` seeds the token table from the teacher's input
    embeddings. Published recurrent embedders are fine-tuned from pretrained
    language models rather than trained from random initialization, so a
    from-scratch table would understate the family; this gives the baseline the
    same pretrained token semantics our cell reads from.
    """

    def __init__(self, vocab_size: int, cfg: ContrastiveConfig,
                 init_embeddings: torch.Tensor | None = None):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, cfg.emb_dim, padding_idx=0)
        if init_embeddings is not None:
            n = min(vocab_size, init_embeddings.shape[0])
            with torch.no_grad():
                self.emb.weight[:n] = init_embeddings[:n].to(self.emb.weight.dtype)
        self.gru = nn.GRU(cfg.emb_dim, cfg.hidden_dim, cfg.num_layers,
                          batch_first=True)
        self.proj = nn.Linear(cfg.hidden_dim, cfg.emb_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = (ids != 0).float().unsqueeze(-1)
        h, _ = self.gru(self.emb(ids))
        pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1.0)
        return nn.functional.normalize(self.proj(pooled), dim=-1)


def encode_texts(model: GRUEmbedder, tok, texts: List[str], cfg: ContrastiveConfig,
                 device: str, batch: int = 256) -> np.ndarray:
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True,
                      max_length=cfg.max_tokens, return_tensors="pt")
            out.append(model(enc["input_ids"].to(device)).cpu().numpy())
    return np.concatenate(out, axis=0)


def load_pairs(cfg: ContrastiveConfig) -> List[tuple]:
    """SNLI entailment pairs: the standard contrastive sentence-embedding data."""
    from datasets import load_dataset

    ds = load_dataset("stanfordnlp/snli", split="train")
    pairs = []
    for row in ds:
        if row["label"] == 0 and row["premise"] and row["hypothesis"]:
            pairs.append((row["premise"], row["hypothesis"]))
        if len(pairs) >= cfg.max_pairs:
            break
    return pairs


def train(cfg: ContrastiveConfig, pairs: List[tuple], tok, vocab: int,
          device: str, init_emb=None) -> GRUEmbedder:
    model = GRUEmbedder(vocab, cfg, init_emb).to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    print(f"  contrastive GRU params={count_params(model):,} pairs={len(pairs):,}")
    for epoch in range(cfg.epochs):
        model.train()
        perm = np.random.permutation(len(pairs))
        total, nb = 0.0, 0
        for i in range(0, len(perm) - cfg.batch_size + 1, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            a = [pairs[j][0] for j in idx]
            b = [pairs[j][1] for j in idx]
            ea = model(tok(a, padding=True, truncation=True, max_length=cfg.max_tokens,
                           return_tensors="pt")["input_ids"].to(device))
            eb = model(tok(b, padding=True, truncation=True, max_length=cfg.max_tokens,
                           return_tensors="pt")["input_ids"].to(device))
            # InfoNCE with in-batch negatives; both directions.
            logits = ea @ eb.T / cfg.temperature
            target = torch.arange(len(idx), device=device)
            loss = 0.5 * (nn.functional.cross_entropy(logits, target)
                          + nn.functional.cross_entropy(logits.T, target))
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite contrastive loss; lower --lr.")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item()
            nb += 1
        print(f"    epoch {epoch + 1}/{cfg.epochs} loss={total / max(nb, 1):.4f}")
    return model


def eval_scifact(model, tok, cfg, device, max_queries: int,
                 max_corpus: int = 60000) -> Dict:
    """SciFact nDCG@10. The model encodes the corpus too: it has no teacher."""
    doc_ids, doc_texts, queries, qrels = load_beir("scifact", max_corpus)
    doc_emb = encode_texts(model, tok, doc_texts, cfg, device)
    qids = [q for q in queries if q in qrels][:max_queries]
    q_emb = encode_texts(model, tok, [queries[q] for q in qids], cfg, device)
    sims = q_emb @ doc_emb.T
    scores = []
    for i, qid in enumerate(qids):
        order = np.argsort(-sims[i])[:10]
        scores.append(ndcg_at_k([doc_ids[j] for j in order], qrels[qid], 10))
    return {"ndcg@10": float(np.mean(scores)) * 100, "n_queries": len(qids),
            "n_corpus": len(doc_ids)}


def eval_sts(model, tok, cfg, device, limit: int) -> Dict:
    from datasets import load_dataset
    from scipy.stats import spearmanr

    out = {}
    for name, ds_id, split in TASKS:
        try:
            data = load_dataset(ds_id, split=split)
        except Exception as e:  # a task that will not download must not abort
            print(f"    {name}: unavailable ({e})")
            continue
        if limit and len(data) > limit:
            data = data.select(range(limit))
        s1 = [r["sentence1"] for r in data]
        s2 = [r["sentence2"] for r in data]
        gold = [r["score"] for r in data]
        e1, e2 = (encode_texts(model, tok, s, cfg, device) for s in (s1, s2))
        pred = (e1 * e2).sum(1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-9)
        out[name] = float(spearmanr(pred, gold).correlation) * 100
        print(f"    {name}: {out[name]:.2f}")
    out["average"] = float(np.mean([v for k, v in out.items() if k != "average"]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--limit", type=int, default=800, help="max pairs per STS task")
    ap.add_argument("--max-queries", type=int, default=300)
    ap.add_argument("--random-init", action="store_true",
                    help="do not seed the token table from the teacher")
    ap.add_argument("--out", type=str,
                    default="experiments/results/recurrent_baseline.json")
    args = ap.parse_args()

    device = get_device()
    log_environment()
    cfg = ContrastiveConfig()
    teacher = Teacher(cfg.tokenizer, device=device)   # tokenizer only
    tok = teacher.tokenizer
    vocab = tok.vocab_size
    # Seed the token table from the teacher's input embeddings (see GRUEmbedder).
    init_emb = None if args.random_init else teacher.token_table.detach().cpu()
    print(f"  token table: {'random' if args.random_init else 'teacher-initialized'}")
    pairs = load_pairs(cfg)

    per_seed = []
    for seed in args.seeds:
        print(f"== seed {seed} ==")
        set_seed(seed)
        model = train(cfg, pairs, tok, vocab, device, init_emb)
        r = {"scifact": eval_scifact(model, tok, cfg, device, args.max_queries),
             "sts": eval_sts(model, tok, cfg, device, args.limit),
             "params": count_params(model)}
        print(f"  seed {seed}: SciFact nDCG@10={r['scifact']['ndcg@10']:.2f} "
              f"STS avg={r['sts']['average']:.2f}")
        per_seed.append(r)

    results = {
        "config": cfg, "seeds": args.seeds, "per_seed": per_seed,
        "teacher_initialized_embeddings": not args.random_init,
        "scifact_ndcg": aggregate_seeds([p["scifact"]["ndcg@10"] for p in per_seed]),
        "sts_average": aggregate_seeds([p["sts"]["average"] for p in per_seed]),
        "params": per_seed[0]["params"],
    }
    save_results(args.out, results)
    print(f"\nContrastive GRU: SciFact {results['scifact_ndcg']['mean']:.2f} "
          f"nDCG@10, STS {results['sts_average']['mean']:.2f}, "
          f"{results['params']:,} params")


if __name__ == "__main__":
    main()
