"""Shared utilities: reproducibility, metrics, and lightweight result I/O.

These helpers are used by every experiment script so that seeds, environment
recording, and result serialization stay consistent across runs.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import random
import subprocess
import sys
from dataclasses import asdict, is_dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Recorded alongside the hardware so a stage log identifies the exact
# dependency set it ran under; the bounds in requirements.txt do not.
TRACKED_PACKAGES = ("numpy", "scikit-learn", "scipy", "tqdm", "datasets",
                    "sentence-transformers", "transformers", "model2vec",
                    "matplotlib")


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """Set all relevant RNG seeds for reproducibility.

    Args:
        seed: The seed value shared across ``random``, NumPy and PyTorch.
        deterministic: When ``True`` force cuDNN into deterministic mode. This
            may reduce throughput but guarantees identical runs across seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Return ``"cuda"`` when a GPU is visible, otherwise ``"cpu"``."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def cpu_model() -> str:
    """Best-effort human-readable CPU model for the current platform."""
    try:
        if sys.platform == "win32":
            import winreg
            key = r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as handle:
                return str(winreg.QueryValueEx(handle, "ProcessorNameString")[0])
        if sys.platform == "linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if sys.platform == "darwin":
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return platform.processor() or "unknown"


def package_versions() -> Dict[str, str]:
    """Resolve the installed version of every tracked dependency."""
    versions: Dict[str, str] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def log_environment() -> Dict[str, Any]:
    """Collect environment metadata for the reproducibility appendix.

    Installs a stdout handler when the caller has configured none, so the
    record reaches the stage log instead of being discarded.

    Returns:
        The environment record, also emitted at INFO level.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(message)s",
                            stream=sys.stdout)
    env = {
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
        "cpu_model": cpu_model(),
        "cpu_count": os.cpu_count(),
        "torch_threads": torch.get_num_threads(),
        "platform": platform.platform(),
        "packages": package_versions(),
    }
    logger.info("Environment: %s", env)
    return env


def count_params(model: torch.nn.Module) -> int:
    """Count trainable parameters of a module."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Row-wise cosine similarity between two ``(B, D)`` tensors."""
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def save_results(path: str | Path, payload: Dict[str, Any]) -> None:
    """Serialize a results dictionary to JSON, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _default(obj: Any) -> Any:
        if is_dataclass(obj):
            return asdict(obj)
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_default)
    logger.info("Wrote results to %s", path)


def load_results(path: str | Path) -> Dict[str, Any]:
    """Load a JSON results file written by :func:`save_results`."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def aggregate_seeds(values: list[float]) -> Dict[str, float]:
    """Return mean/std/n for a list of per-seed scalar metrics."""
    arr = np.asarray(values, dtype=np.float64)
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=0)), "n": int(arr.size)}


def aggregate_over_ckpts(runs: list[dict]) -> tuple[dict, dict]:
    """Collapse per-checkpoint result dicts into (means, stds).

    ``runs`` are same-shaped dicts of floats, one per seed checkpoint. The mean
    dict keeps the original key layout and float leaves, so consumers written
    against a single-checkpoint run keep working; the std dict mirrors it.
    Non-numeric leaves are passed through from the first run.
    """
    if not runs:
        return {}, {}
    mean: Dict[str, object] = {}
    std: Dict[str, object] = {}
    for key, first in runs[0].items():
        vals = [r[key] for r in runs if key in r]
        if isinstance(first, dict):
            m, s = aggregate_over_ckpts([v for v in vals if isinstance(v, dict)])
            mean[key], std[key] = m, s
        elif isinstance(first, list) and first and isinstance(first[0], dict):
            # Element-wise over aligned lists of dicts (e.g. one row per
            # re-encoding budget), so the saved JSON keeps its list shape.
            pairs = [aggregate_over_ckpts([r[i] for r in vals])
                     for i in range(len(first))]
            mean[key] = [p[0] for p in pairs]
            std[key] = [p[1] for p in pairs]
        elif isinstance(first, (int, float)) and not isinstance(first, bool):
            arr = np.asarray(vals, dtype=np.float64)
            mean[key] = float(arr.mean())
            std[key] = float(arr.std(ddof=0))
        else:
            mean[key] = first
    return mean, std


__all__ = [
    "set_seed",
    "get_device",
    "log_environment",
    "count_params",
    "aggregate_over_ckpts",
    "cosine_sim",
    "save_results",
    "load_results",
    "aggregate_seeds",
]
