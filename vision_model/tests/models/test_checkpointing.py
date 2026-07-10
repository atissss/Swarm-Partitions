from __future__ import annotations

from pathlib import Path

from vision_model.models.checkpointing import resolve_checkpoint_paths


class TestResolveCheckpointPaths:
    def test_creates_run_dir_and_paths(self, tmp_path: Path) -> None:
        paths = resolve_checkpoint_paths(tmp_path / "runs", "exp001")
        assert paths.run_dir.is_dir()
        assert paths.last == paths.run_dir / "last.pt"
        assert paths.best == paths.run_dir / "best.pt"

    def test_idempotent_for_same_run_name(self, tmp_path: Path) -> None:
        p1 = resolve_checkpoint_paths(tmp_path / "runs", "exp001")
        p2 = resolve_checkpoint_paths(tmp_path / "runs", "exp001")
        assert p1 == p2
