"""Teacher encoder access and embedding-shift triple construction.

The distillation track learns to imitate a frozen sentence encoder. This module
wraps the teacher (all-MiniLM-L6-v2 by default), exposes its subword input
embedding table (the incoming-token representation), and builds
``(E_n, token_emb, ΔE)`` triples from causally re-encoded subword prefixes.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Models trained with instruction prefixes must be prompted the same way at
# inference; encoding E5 without its prefixes is out-of-distribution and
# collapses asymmetric (short query vs. long passage) retrieval.
TEACHER_PROMPTS: Dict[str, Dict[str, str]] = {
    "intfloat/e5-base-v2": {"query": "query: ", "passage": "passage: "},
    "intfloat/e5-large-v2": {"query": "query: ", "passage": "passage: "},
    "intfloat/e5-small-v2": {"query": "query: ", "passage": "passage: "},
    # Qwen3-Embedding: instruction on the query side only (model card default).
    "Qwen/Qwen3-Embedding-4B": {
        "query": "Instruct: Given a web search query, retrieve relevant passages "
                 "that answer the query\nQuery: ",
        "passage": "",
    },
}


class Teacher:
    """Frozen sentence-embedding teacher with subword-level prefix encoding.

    ``role`` selects the instruction prompt ("query" or "passage") for teachers
    that require one (see :data:`TEACHER_PROMPTS`); the prompt is prepended to
    the *text* handed to the teacher only, never to the token-id stream the
    incremental cell consumes, so the cell's inputs are unchanged and the
    teacher's embedding space is the prompted one throughout (triples,
    rollouts, resets, and corpus encoding alike).
    """

    def __init__(self, name: str = "all-MiniLM-L6-v2", device: str = "cpu",
                 normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer
        # Multi-billion-parameter teachers (Qwen3-Embedding) do not fit a
        # 12 GB GPU in fp32; load them in bf16 and cap the context. The
        # small teachers keep the fp32 default so their numbers are unchanged.
        kwargs = ({"model_kwargs": {"torch_dtype": torch.bfloat16}}
                  if "Qwen3" in name else {})
        self.model = SentenceTransformer(name, device=device, **kwargs)
        if "Qwen3" in name:
            self.model.max_seq_length = 512
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self.normalize = normalize
        self.prompts = TEACHER_PROMPTS.get(name, {})
        self.tokenizer = self.model.tokenizer
        table = self.model[0].auto_model.get_input_embeddings().weight.detach()
        self.token_table = table.to(device)          # (V, D)
        self.emb_dim = table.size(1)
        if self.prompts:
            logger.info("Teacher %s uses instruction prompts: %s", name, self.prompts)

    def _prompt(self, role: str) -> str:
        return self.prompts.get(role, "")

    def token_ids(self, text: str, max_tokens: int) -> List[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)[:max_tokens]

    def token_embed(self, ids: List[int]) -> np.ndarray:
        """Input-embedding-table rows for a list of token ids -> ``(len, D)``."""
        idx = torch.tensor(ids, dtype=torch.long, device=self.device)
        # .float(): a bf16 table (Qwen3) has no numpy dtype.
        return self.token_table[idx].float().cpu().numpy()

    @torch.no_grad()
    def encode_prefixes(self, ids: List[int], ks: List[int],
                        role: str = "query") -> np.ndarray:
        """Encode prefixes ``ids[:k]`` for each ``k`` in ``ks`` -> ``(len(ks), D)``."""
        prompt = self._prompt(role)
        texts = [prompt + self.tokenizer.decode(ids[:k]) for k in ks]
        embs = self.model.encode(texts, convert_to_numpy=True,
                                 normalize_embeddings=self.normalize,
                                 show_progress_bar=False)
        return embs

    @torch.no_grad()
    def encode_texts(self, texts: List[str], role: str = "query") -> np.ndarray:
        prompt = self._prompt(role)
        return self.model.encode([prompt + t for t in texts],
                                 convert_to_numpy=True,
                                 normalize_embeddings=self.normalize,
                                 show_progress_bar=False)


def build_triples(teacher: Teacher, docs: List[str], min_tokens: int, max_tokens: int
                  ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Return ``(E_n, token_emb, ΔE)`` triples across all documents.

    For each document we encode prefixes ``[:n]`` and ``[:n+1]`` for
    ``n in [min_tokens, max_tokens)`` and record the incoming token embedding.
    """
    triples: List[Tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for doc in docs:
        ids = teacher.token_ids(doc, max_tokens + 1)
        if len(ids) <= min_tokens + 1:
            continue
        ns = list(range(min_tokens, min(len(ids) - 1, max_tokens)))
        if not ns:
            continue
        e_n = teacher.encode_prefixes(ids, ns)              # (T, D)
        e_n1 = teacher.encode_prefixes(ids, [n + 1 for n in ns])
        delta = e_n1 - e_n
        tok = teacher.token_embed([ids[n] for n in ns])     # (T, D)
        for i in range(len(ns)):
            triples.append((e_n[i], tok[i], delta[i]))
    logger.info("Built %d triples from %d docs", len(triples), len(docs))
    return triples


def build_trajectories(teacher: Teacher, docs: List[str], min_tokens: int,
                       max_tokens: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return per-document ``(prefix_embeddings, token_embeddings)`` trajectories.

    Unlike :func:`build_triples`, consecutive steps are kept in order so the cell
    can be trained with a multi-step unrolled loss. ``prefix_embeddings`` has
    shape ``(T+1, D)`` and ``token_embeddings`` has shape ``(T, D)``, where step
    ``t`` consumes ``token_embeddings[t]`` to move from state ``t`` to ``t+1``.
    """
    trajectories: List[Tuple[np.ndarray, np.ndarray]] = []
    for doc in docs:
        ids = teacher.token_ids(doc, max_tokens + 1)
        if len(ids) <= min_tokens + 1:
            continue
        ks = list(range(min_tokens, min(len(ids), max_tokens + 1)))
        if len(ks) < 3:
            continue
        embs = teacher.encode_prefixes(ids, ks)                  # (T+1, D)
        toks = teacher.token_embed([ids[k] for k in ks[:-1]])    # (T, D)
        trajectories.append((embs, toks))
    logger.info("Built %d trajectories", len(trajectories))
    return trajectories


def load_wikitext_docs(config: str, num_docs: int, min_words: int) -> List[str]:
    """Load reasonably long WikiText paragraphs as raw strings."""
    from datasets import load_dataset
    raw = load_dataset("wikitext", config, split="train")
    docs: List[str] = []
    for row in raw:
        text = row["text"].strip()
        if len(text.split()) >= min_words:
            docs.append(text)
        if len(docs) >= num_docs:
            break
    return docs


__all__ = ["Teacher", "build_triples", "load_wikitext_docs"]
