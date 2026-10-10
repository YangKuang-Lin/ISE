"""Tokenization and dataset construction for the from-scratch track.

Uses a subword (WordPiece) tokenizer shared by the ISE and Transformer models so
the comparison is not confounded by tokenization. MLM and classification
datasets mirror the original pipeline but are seed- and config-driven.
"""

from __future__ import annotations

import random
from typing import List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

from .encoders import PAD_ID

# BERT-style special ids (PAD == 0 matches ``encoders.PAD_ID``).
UNK_ID = 100
CLS_ID = 101
SEP_ID = 102
MASK_ID = 103


class SubwordTokenizer:
    """Thin wrapper over a HuggingFace WordPiece tokenizer.

    Kept as a class so the vocab size and special-token ids are discoverable by
    the models without importing ``transformers`` everywhere.
    """

    def __init__(self, name: str = "bert-base-uncased") -> None:
        from transformers import AutoTokenizer  # local import: heavy dependency
        self.tok = AutoTokenizer.from_pretrained(name)

    @property
    def vocab_size(self) -> int:
        return self.tok.vocab_size

    def encode(self, text: str, max_len: int | None = None) -> List[int]:
        ids = self.tok.encode(text, add_special_tokens=False)
        if max_len is not None:
            ids = ids[:max_len]
        return ids


class MLMDataset(Dataset):
    """Masked-language-modeling chunks with the standard 80/10/10 corruption."""

    def __init__(self, texts: Sequence[str], tokenizer: SubwordTokenizer,
                 max_len: int, mlm_prob: float, seed: int = 42) -> None:
        self.vocab_size = tokenizer.vocab_size
        self.max_len = max_len
        self.mlm_prob = mlm_prob
        self.rng = random.Random(seed)
        self.samples: List[List[int]] = []
        for text in texts:
            ids = tokenizer.encode(text)
            for start in range(0, max(0, len(ids) - max_len), max_len // 2):
                chunk = ids[start:start + max_len]
                if len(chunk) == max_len:
                    self.samples.append(chunk)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ids = list(self.samples[idx])
        labels = [-100] * len(ids)
        for i in range(len(ids)):
            if self.rng.random() < self.mlm_prob:
                labels[i] = ids[i]
                r = self.rng.random()
                if r < 0.80:
                    ids[i] = MASK_ID
                elif r < 0.90:
                    ids[i] = self.rng.randrange(1000, self.vocab_size)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class ClassificationDataset(Dataset):
    """Padded/truncated sequence-classification dataset."""

    def __init__(self, texts: Sequence[str], labels: Sequence[int],
                 tokenizer: SubwordTokenizer, max_len: int) -> None:
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for text, label in zip(texts, labels):
            ids = tokenizer.encode(text, max_len=max_len)
            ids = ids + [PAD_ID] * (max_len - len(ids))
            self.samples.append((torch.tensor(ids, dtype=torch.long),
                                 torch.tensor(label, dtype=torch.long)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.samples[idx]


__all__ = ["SubwordTokenizer", "MLMDataset", "ClassificationDataset",
           "UNK_ID", "CLS_ID", "SEP_ID", "MASK_ID"]
