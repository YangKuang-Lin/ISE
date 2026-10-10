"""Experiment package for Incremental Sentence Embedding (ISE).

Modules:
    utils     -- reproducibility, metrics, result I/O
    cells     -- incremental update cells (FiLM / GRU / concat-MLP / gated-residual)
    encoders  -- ISE / Transformer encoders and causal LMs (with KV cache)
    data      -- subword tokenizer and MLM / classification datasets
"""

from .cells import CELL_REGISTRY, build_cell, register_cell
from .encoders import (
    ISECausalLM,
    ISEEncoder,
    GRUCausalLM,
    TFCausalLMCached,
    TFCausalLMNoCache,
    TFEncoder,
)
from .utils import (
    aggregate_seeds,
    count_params,
    get_device,
    log_environment,
    save_results,
    set_seed,
)

__all__ = [
    "CELL_REGISTRY",
    "build_cell",
    "register_cell",
    "ISECausalLM",
    "ISEEncoder",
    "GRUCausalLM",
    "TFCausalLMCached",
    "TFCausalLMNoCache",
    "TFEncoder",
    "aggregate_seeds",
    "count_params",
    "get_device",
    "log_environment",
    "save_results",
    "set_seed",
]
