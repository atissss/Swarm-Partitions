from __future__ import annotations

from pathlib import Path

import pytest

from vision_model.datasets.auair import AUAIRDataset
from vision_model.utils.exceptions import DatasetError


class TestAUAIRDataset:
    def test_indexes_only_images_present_on_disk(self, auair_root: Path) -> None:
        ds = AUAIRDataset(root=auair_root, split="train")
        # 3 records in fixture, 1 references a missing image -> skipped
        assert len(ds) == 2

    def test_telemetry_timestamp_is_threaded_through(self, auair_root: Path) -> None:
        ds = AUAIRDataset(root=auair_root, split="train")
        sample = next(s for s in ds.index if s.image_id == "frame_000001")
        assert sample.telemetry_timestamp == "2026-07-01T15:42:11Z"

    def test_out_of_scope_category_dropped(self, auair_root: Path) -> None:
        ds = AUAIRDataset(root=auair_root, split="train")
        sample = next(s for s in ds.index if s.image_id == "frame_000001")
        # fixture has car + human + bicycle; bicycle is out of scope
        class_names = sorted(a.class_name for a in sample.annotations)
        assert class_names == ["car", "person"]

    def test_missing_annotations_file_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "auair_empty"
        (root / "images").mkdir(parents=True)
        with pytest.raises(DatasetError):
            AUAIRDataset(root=root, split="train")
