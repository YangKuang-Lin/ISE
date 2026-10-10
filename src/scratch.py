"""From-scratch comparison: ISE cells vs Transformer, trained fairly.

Addresses the confounds in the original ``ise_from_scratch.py``:
  * both models are evaluated under **both** pooling strategies (last / mean),
  * results are averaged over multiple seeds with mean +/- std,
  * a GRU cell is included as a recurrent baseline,
  * a shared subword tokenizer removes the word-level tokenization confound,
  * dataset and metric names in the output match what is actually run
    (WikiText-2 MLM pseudo-perplexity, GLUE SST-2 accuracy).

Usage:
    python scratch.py --seeds 41 42 43 --cells gated_residual gru
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from experiments.data import ClassificationDataset, MLMDataset, SubwordTokenizer
from experiments.encoders import PAD_ID, ISEEncoder, TFEncoder
from experiments.utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)


@dataclass(frozen=True)
class ScratchConfig:
    dataset: str = "wikitext"
    dataset_config: str = "wikitext-2-raw-v1"
    tokenizer: str = "bert-base-uncased"
    max_seq_len: int = 64
    mlm_prob: float = 0.15
    emb_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    nhead: int = 4
    ff_dim: int = 512
    mlm_hidden: int = 512
    batch_size: int = 128
    lr: float = 3e-4
    epochs: int = 10
    warmup_steps: int = 500
    ft_lr: float = 5e-5
    ft_epochs: int = 3
    ft_batch_size: int = 64
    grad_clip: float = 1.0
    poolings: tuple = ("last", "mean")


class MLMHead(nn.Module):
    """Identical decoder head used by every encoder for a fair MLM comparison."""

    def __init__(self, emb_dim: int, hidden_dim: int, vocab_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim), nn.GELU(),
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.net(hidden)


class ModelForMLM(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_ids, labels=None):
        logits = self.decoder(self.encoder(input_ids))
        if labels is None:
            return logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss, logits


class ModelForClassification(nn.Module):
    def __init__(self, encoder: nn.Module, emb_dim: int, num_classes: int, pool_type: str) -> None:
        super().__init__()
        self.encoder = encoder
        self.pool_type = pool_type
        self.classifier = nn.Linear(emb_dim, num_classes)

    def _pool(self, h, input_ids):
        if self.pool_type == "last":
            lens = (input_ids != PAD_ID).sum(dim=1).clamp(min=1) - 1
            idx = lens.view(-1, 1, 1).expand(-1, 1, h.size(2))
            return h.gather(1, idx).squeeze(1)
        mask = (input_ids != PAD_ID).float().unsqueeze(-1)
        return (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

    def forward(self, input_ids, labels=None):
        pooled = self._pool(self.encoder(input_ids), input_ids)
        logits = self.classifier(pooled)
        if labels is not None:
            return nn.functional.cross_entropy(logits, labels), logits
        return logits


def _make_encoder(kind: str, cfg: ScratchConfig, vocab_size: int) -> nn.Module:
    if kind == "transformer":
        return TFEncoder(vocab_size, cfg.emb_dim, cfg.nhead, cfg.num_layers, cfg.ff_dim)
    # otherwise ``kind`` is an ISE cell name
    return ISEEncoder(vocab_size, cfg.emb_dim, cfg.hidden_dim, cfg.num_layers, cell=kind)


def _train(model, train_loader, val_loader, device, epochs, lr, warmup, grad_clip, task):
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(train_loader)

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, total - warmup)
        return max(0.05, 0.5 * (1 + math.cos(math.pi * prog)))

    sched = optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best = float("inf") if task == "mlm" else 0.0
    for _ in range(epochs):
        model.train()
        for input_ids, labels in train_loader:
            input_ids, labels = input_ids.to(device), labels.to(device)
            loss, _ = model(input_ids, labels)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            sched.step()
        score = _evaluate(model, val_loader, device, task)
        best = min(best, score) if task == "mlm" else max(best, score)
    return best


@torch.no_grad()
def _evaluate(model, loader, device, task):
    model.eval()
    if task == "mlm":
        tot_loss, tot = 0.0, 0
        for input_ids, labels in loader:
            loss, _ = model(input_ids.to(device), labels.to(device))
            n = (labels != -100).sum().item()
            tot_loss += loss.item() * n
            tot += n
        return math.exp(tot_loss / max(tot, 1))  # pseudo-perplexity
    correct, tot = 0, 0
    for input_ids, labels in loader:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids)
        correct += (logits.argmax(-1) == labels).sum().item()
        tot += labels.size(0)
    return correct / max(tot, 1)


def run_seed(cfg: ScratchConfig, kinds: List[str], seed: int, device: str) -> Dict:
    from datasets import load_dataset
    set_seed(seed)

    raw = load_dataset(cfg.dataset, cfg.dataset_config)
    train_texts = [r["text"] for r in raw["train"] if len(r["text"].split()) > 10]
    val_texts = [r["text"] for r in raw["validation"] if len(r["text"].split()) > 10]
    tok = SubwordTokenizer(cfg.tokenizer)
    V = tok.vocab_size

    train_mlm = DataLoader(MLMDataset(train_texts, tok, cfg.max_seq_len, cfg.mlm_prob, seed),
                           batch_size=cfg.batch_size, shuffle=True)
    val_mlm = DataLoader(MLMDataset(val_texts, tok, cfg.max_seq_len, cfg.mlm_prob, seed),
                         batch_size=cfg.batch_size)

    cls = load_dataset("glue", "sst2")
    train_cls = DataLoader(ClassificationDataset(cls["train"]["sentence"], cls["train"]["label"],
                                                 tok, cfg.max_seq_len),
                           batch_size=cfg.ft_batch_size, shuffle=True)
    val_cls = DataLoader(ClassificationDataset(cls["validation"]["sentence"], cls["validation"]["label"],
                                               tok, cfg.max_seq_len),
                         batch_size=cfg.ft_batch_size)

    out: Dict = {}
    for kind in kinds:
        enc = _make_encoder(kind, cfg, V)
        mlm = ModelForMLM(enc, MLMHead(cfg.emb_dim, cfg.mlm_hidden, V))
        ppl = _train(mlm, train_mlm, val_mlm, device, cfg.epochs, cfg.lr,
                     cfg.warmup_steps, cfg.grad_clip, "mlm")
        entry: Dict = {"params": count_params(mlm), "mlm_ppl": ppl, "sst2": {}}
        for pool in cfg.poolings:
            clf = ModelForClassification(enc, cfg.emb_dim, 2, pool)
            acc = _train(clf, train_cls, val_cls, device, cfg.ft_epochs, cfg.ft_lr,
                         cfg.warmup_steps, cfg.grad_clip, "cls")
            entry["sst2"][pool] = acc
        out[kind] = entry
        print(f"  seed {seed} | {kind}: ppl={ppl:.2f} sst2={entry['sst2']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[41, 42, 43])
    ap.add_argument("--cells", type=str, nargs="+",
                    default=["gated_residual", "gru", "transformer"])
    ap.add_argument("--out", type=str, default="experiments/results/scratch.json")
    args = ap.parse_args()

    cfg = ScratchConfig()
    device = get_device()
    log_environment()

    per_seed = [run_seed(cfg, args.cells, s, device) for s in args.seeds]

    # Aggregate mean/std across seeds.
    agg: Dict = {}
    for kind in args.cells:
        agg[kind] = {
            "mlm_ppl": aggregate_seeds([p[kind]["mlm_ppl"] for p in per_seed]),
            "params": per_seed[0][kind]["params"],
            "sst2": {pool: aggregate_seeds([p[kind]["sst2"][pool] for p in per_seed])
                     for pool in cfg.poolings},
        }
    save_results(args.out, {"config": cfg, "seeds": args.seeds,
                            "per_seed": per_seed, "aggregate": agg})
    print("Aggregate:", agg)


if __name__ == "__main__":
    main()
