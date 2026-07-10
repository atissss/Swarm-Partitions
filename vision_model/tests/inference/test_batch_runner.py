from __future__ import annotations

from pathlib import Path

import pytest

from tests.fakes import FakeDetector
from vision_model.inference.batch_runner import (
    get_video_fps,
    iter_video_frames,
    run_on_image_directory,
    run_on_video,
)
from vision_model.inference.predictor import Predictor
from vision_model.interfaces.detection import Detection
from vision_model.utils.exceptions import InferenceError


def _canned_detection() -> Detection:
    return Detection(
        class_name="car",
        confidence=0.9,
        bbox=(5.0, 5.0, 20.0, 20.0),
        pixel_center=(12.5, 12.5),
        frame_id="",
        timestamp="",
    )


class TestGetVideoFps:
    def test_reads_fps_from_file(self, make_synthetic_video) -> None:
        path = make_synthetic_video(fps=15.0)
        assert get_video_fps(path) == pytest.approx(15.0)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InferenceError):
            get_video_fps(tmp_path / "nope.avi")


class TestIterVideoFrames:
    def test_yields_all_frames_in_order(self, make_synthetic_video) -> None:
        path = make_synthetic_video(num_frames=4)
        frames = list(iter_video_frames(path))
        assert [idx for idx, _ in frames] == [0, 1, 2, 3]
        assert all(frame.shape == (48, 64, 3) for _, frame in frames)

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InferenceError):
            list(iter_video_frames(tmp_path / "nope.avi"))


class TestRunOnVideo:
    def test_computes_sequential_timestamps(self, make_synthetic_video) -> None:
        path = make_synthetic_video(num_frames=3, fps=10.0)
        detector = FakeDetector(canned_detections=[_canned_detection()])
        predictor = Predictor(detector, confidence_threshold=0.5)

        results = list(run_on_video(predictor, path, start_timestamp="2026-07-01T00:00:00Z"))

        timestamps = [r.timestamp for _, r in results]
        assert timestamps == [
            "2026-07-01T00:00:00Z",
            "2026-07-01T00:00:00.100000Z",
            "2026-07-01T00:00:00.200000Z",
        ]

    def test_frame_ids_include_video_stem_and_index(self, make_synthetic_video) -> None:
        path = make_synthetic_video(name="myclip.avi", num_frames=2)
        detector = FakeDetector(canned_detections=[])
        predictor = Predictor(detector)

        results = list(run_on_video(predictor, path, start_timestamp="2026-07-01T00:00:00Z"))

        assert [r.frame_id for _, r in results] == ["myclip:0", "myclip:1"]

    def test_detections_returned_per_frame(self, make_synthetic_video) -> None:
        path = make_synthetic_video(num_frames=2)
        detector = FakeDetector(canned_detections=[_canned_detection()])
        predictor = Predictor(detector, confidence_threshold=0.5)

        results = list(run_on_video(predictor, path, start_timestamp="2026-07-01T00:00:00Z"))

        assert all(len(r.detections) == 1 for _, r in results)

    def test_explicit_fps_overrides_video_fps(self, make_synthetic_video) -> None:
        path = make_synthetic_video(num_frames=2, fps=10.0)
        detector = FakeDetector(canned_detections=[])
        predictor = Predictor(detector)

        results = list(
            run_on_video(predictor, path, start_timestamp="2026-07-01T00:00:00Z", fps=2.0)
        )

        assert results[1][1].timestamp == "2026-07-01T00:00:00.500000Z"


class TestRunOnImageDirectory:
    def test_runs_over_all_images_sorted(self, visdrone_root: Path) -> None:
        images_dir = visdrone_root / "images"
        detector = FakeDetector(canned_detections=[_canned_detection()])
        predictor = Predictor(detector, confidence_threshold=0.5)

        results = run_on_image_directory(
            predictor, images_dir, get_timestamp=lambda p: "2026-07-01T00:00:00Z"
        )

        assert len(results) == 3  # all 3 images in the visdrone fixture, annotated or not
        assert all(r.timestamp == "2026-07-01T00:00:00Z" for r in results)

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(InferenceError):
            run_on_image_directory(
                Predictor(FakeDetector()), tmp_path / "nope", get_timestamp=lambda p: "x"
            )

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(InferenceError):
            run_on_image_directory(Predictor(FakeDetector()), empty, get_timestamp=lambda p: "x")

    def test_get_timestamp_receives_path(self, visdrone_root: Path) -> None:
        seen_paths = []

        def _get_timestamp(p: Path) -> str:
            seen_paths.append(p.name)
            return "2026-07-01T00:00:00Z"

        run_on_image_directory(
            Predictor(FakeDetector()), visdrone_root / "images", get_timestamp=_get_timestamp
        )
        assert seen_paths == ["0000001.jpg", "0000002.jpg", "0000003.jpg"]
