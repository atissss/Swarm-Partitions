from __future__ import annotations

from pathlib import Path

import numpy as np

from vision_model.interfaces.detection import Detection
from vision_model.visualization.draw_boxes import draw_detections, save_annotated_image


def _det(class_name: str, bbox=(10, 10, 60, 60)) -> Detection:
    return Detection(
        class_name=class_name,  # type: ignore[arg-type]
        confidence=0.9,
        bbox=bbox,
        pixel_center=((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2),
        frame_id="f",
        timestamp="2026-07-01T15:42:11Z",
    )


class TestDrawDetections:
    def test_returns_new_array_without_mutating_input(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        original = image.copy()

        annotated = draw_detections(image, [_det("car")])

        assert np.array_equal(image, original)  # input untouched
        assert not np.array_equal(annotated, original)  # output has drawing

    def test_handles_empty_detections(self) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        annotated = draw_detections(image, [])
        assert np.array_equal(annotated, image)


class TestSaveAnnotatedImage:
    def test_writes_file_to_disk(self, tmp_path: Path) -> None:
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        dest = tmp_path / "out" / "annotated.jpg"

        result_path = save_annotated_image(image, [_det("person")], dest)

        assert result_path == dest
        assert dest.is_file()
        assert dest.stat().st_size > 0
