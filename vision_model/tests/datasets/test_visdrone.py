from __future__ import annotations

from pathlib import Path

import pytest

from vision_model.datasets import build_dataset
from vision_model.datasets.visdrone import VisDroneDataset
from vision_model.utils.exceptions import AnnotationParseError, DatasetError


class TestVisDroneDataset:
    def test_indexes_only_images_with_annotations(self, visdrone_root: Path) -> None:
        ds = VisDroneDataset(root=visdrone_root, split="train")
        # image 0000003.jpg has no annotation file and should be skipped
        assert len(ds) == 2
        image_ids = {ds.index[i].image_id for i in range(len(ds))}
        assert image_ids == {"0000001", "0000002"}

    def test_out_of_scope_categories_are_dropped(self, visdrone_root: Path) -> None:
        ds = VisDroneDataset(root=visdrone_root, split="train")
        sample = next(s for s in ds.index if s.image_id == "0000001")
        # 3 raw lines in fixture, but the "van" line (category 5) is out of scope
        assert len(sample.annotations) == 2
        class_names = sorted(a.class_name for a in sample.annotations)
        assert class_names == ["car", "person"]

    def test_bbox_converted_from_xywh_to_xyxy(self, visdrone_root: Path) -> None:
        ds = VisDroneDataset(root=visdrone_root, split="train")
        sample = next(s for s in ds.index if s.image_id == "0000001")
        car = next(a for a in sample.annotations if a.class_name == "car")
        # raw line: 100,100,50,80,... -> left=100, top=100, w=50, h=80
        assert car.bbox == (100.0, 100.0, 150.0, 180.0)

    def test_getitem_returns_expected_keys(self, visdrone_root: Path) -> None:
        ds = VisDroneDataset(root=visdrone_root, split="train")
        item = ds[0]
        assert set(item) >= {"image_path", "image_id", "annotations", "telemetry_timestamp"}

    def test_missing_images_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(DatasetError):
            VisDroneDataset(root=tmp_path / "does_not_exist", split="train")

    def test_build_via_registry(self, visdrone_root: Path) -> None:
        ds = build_dataset("visdrone", root=visdrone_root, split="train")
        assert isinstance(ds, VisDroneDataset)
        assert len(ds) == 2

    def test_malformed_annotation_line_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "bad"
        (root / "images").mkdir(parents=True)
        (root / "annotations").mkdir(parents=True)
        (root / "images" / "x.jpg").write_bytes(b"fake")
        (root / "annotations" / "x.txt").write_text("not,enough,fields\n")
        with pytest.raises(AnnotationParseError):
            VisDroneDataset(root=root, split="train")
