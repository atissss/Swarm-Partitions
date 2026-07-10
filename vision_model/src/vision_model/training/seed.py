"""Reproducible-training seeding.

Seeds Python's `random`, NumPy, and (if installed) PyTorch's CPU/CUDA RNGs
from a single seed, so `set_seed(42)` followed by the same training config
is expected to reproduce the same run.
"""

from __future__ import annotations

import os
import random

import numpy as np

from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed all relevant RNGs for reproducible training.

    Args:
        seed: Seed value applied to `random`, NumPy, and PyTorch (if
            installed).
        deterministic: If True and PyTorch is installed, also configure
            cuDNN for deterministic (if slower) convolution algorithms.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        logger.debug("torch not installed; skipped seeding torch RNGs.")

    logger.info("Seeded RNGs with seed=%d (deterministic=%s)", seed, deterministic)
