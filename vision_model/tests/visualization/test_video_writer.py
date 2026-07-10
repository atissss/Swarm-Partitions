from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import FakeDetector
from vision_model.inference.batch_runner import run_on_video
from vision_model.inference.predictor import Predictor
from vision_model.interfaces.detection import Detection
from vision_model.utils.exceptions import VisionModelError
from vision_model.visualization.video_writer import write_annotated_video


def _canned_detection() -> Detection:
    return Detection(
        class_name="person",
        confidence=0.8,
        bbox=(5.0, 5.0, 20.0, 20.0),
        pixel_center=(12.5, 12.5),
        frame_id="",
        timestamp="",
    )


class TestWriteAnnotatedVideo:
    def test_writes_expected_frame_count(self, make_synthetic_video, tmp_path: Path) -> None:
        video_path = make_synthetic_video(num_frames=4, fps=10.0)
        predictor = Predictor(FakeDetector(canned_detections=[_canned_detection()]))
        frames = run_on_video(predictor, video_path, start_timestamp="2026-07-01T00:00:00Z")

        summary = write_annotated_video(frames, tmp_path / "out.avi", fps=10.0, fourcc="MJPG")

        assert summary.num_frames == 4
        assert summary.num_detections == 4  # one detection per frame
        assert summary.output_path.is_file()
        assert summary.output_path.stat().st_size > 0

    def test_empty_frame_stream_raises(self, tmp_path: Path) -> None:
        with pytest.raises(VisionModelError):
            write_annotated_video(iter([]), tmp_path / "out.avi", fps=10.0)

    def test_output_video_is_readable(self, make_synthetic_video, tmp_path: Path) -> None:
        import cv2

        video_path = make_synthetic_video(num_frames=3, fps=5.0)
        predictor = Predictor(FakeDetector(canned_detections=[]))
        frames = run_on_video(predictor, video_path, start_timestamp="2026-07-01T00:00:00Z")

        out_path = tmp_path / "out.avi"
        write_annotated_video(frames, out_path, fps=5.0, fourcc="MJPG")

        cap = cv2.VideoCapture(str(out_path))
        try:
            assert cap.isOpened()
            assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        finally:
            cap.release()

    def test_creates_parent_directories(self, make_synthetic_video, tmp_path: Path) -> None:
        video_path = make_synthetic_video(num_frames=1)
        predictor = Predictor(FakeDetector(canned_detections=[]))
        frames = run_on_video(predictor, video_path, start_timestamp="2026-07-01T00:00:00Z")

        out_path = tmp_path / "nested" / "dir" / "out.avi"
        summary = write_annotated_video(frames, out_path, fps=10.0, fourcc="MJPG")

        assert summary.output_path.is_file()
