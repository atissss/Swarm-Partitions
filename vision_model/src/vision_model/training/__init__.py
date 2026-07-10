"""Training orchestration and reproducibility helpers.

Public entry points:

    from vision_model.training import Trainer, set_seed
"""

from vision_model.training.seed import set_seed
from vision_model.training.trainer import Trainer

__all__ = ["Trainer", "set_seed"]
