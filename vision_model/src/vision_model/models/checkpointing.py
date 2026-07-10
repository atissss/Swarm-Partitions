"""Checkpoint naming/bookkeeping helpers, backend-agnostic.

Backends handle their own weight serialization (`Detector.save_checkpoint` /
`load_checkpoint`); this module standardizes *where* and *how checkpoints
are named* so training runs are reproducible and the best/last checkpoint is
always discoverable by convention rather than by remembering a specific run's
file name.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from vision_model.utils.io import ensure_dir


@dataclass(frozen=True, slots=True)
class CheckpointPaths:
    """Resolved checkpoint file paths for one training run.

    Attributes:
        run_dir: Root directory for this run's checkpoints.
        last: Path to the most recent checkpoint.
        best: Path to the best-so-far checkpoint (by validation metric).
    """

    run_dir: Path
    last: Path
    best: Path


def resolve_checkpoint_paths(output_root: str | Path, run_name: str) -> CheckpointPaths:
    """Compute the standard `last.pt`/`best.pt` paths for a training run.

    Args:
        output_root: Root directory under which run directories live.
        run_name: Unique name for this training run (e.g. a timestamp or
            experiment id).

    Returns:
        Resolved `CheckpointPaths`, with `run_dir` created on disk.
    """
    run_dir = ensure_dir(Path(output_root) / run_name)
    return CheckpointPaths(
        run_dir=run_dir,
        last=run_dir / "last.pt",
        best=run_dir / "best.pt",
    )
