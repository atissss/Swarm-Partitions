"""Shared pytest fixtures.

Fixtures here build small synthetic datasets/telemetry/camera configs so the
test suite never depends on downloading real VisDrone/AU-AIR data or real
flight logs.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from vision_model.interfaces.camera import CameraExtrinsics, CameraIntrinsics
from vision_model.interfaces.telemetry import Telemetry


@pytest.fixture
def visdrone_root(tmp_path: Path) -> Path:
    """Build a minimal on-disk VisDrone-DET split: 2 images, 2 annotation files."""
    root = tmp_path / "VisDrone2019-DET-train"
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True)
    annotations_dir.mkdir(parents=True)

    # image 1: one car, one pedestrian, one out-of-scope "van" (dropped)
    (images_dir / "0000001.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (annotations_dir / "0000001.txt").write_text(
        "100,100,50,80,1,4,0,0\n"  # car
        "300,200,20,60,1,1,0,1\n"  # pedestrian -> person
        "50,50,40,40,1,5,0,0\n"  # van -> dropped (out of scope)
    )

    # image 2: one "people" (-> person), no annotation errors
    (images_dir / "0000002.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (annotations_dir / "0000002.txt").write_text("10,10,30,90,1,2,1,0\n")

    # image 3: no matching annotation file -> should be skipped by loader
    (images_dir / "0000003.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    return root


@pytest.fixture
def auair_root(tmp_path: Path) -> Path:
    """Build a minimal on-disk AU-AIR split: 2 images + annotations.json."""
    root = tmp_path / "auair"
    images_dir = root / "images"
    images_dir.mkdir(parents=True)

    (images_dir / "frame_000001.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")
    (images_dir / "frame_000002.jpg").write_bytes(b"\xff\xd8\xff\xe0fakejpeg")

    records = [
        {
            "image_name": "frame_000001.jpg",
            "time": "2026-07-01T15:42:11Z",
            "bbox": [
                {"top": 10, "left": 20, "width": 40, "height": 60, "class": "car"},
                {"top": 100, "left": 100, "width": 15, "height": 30, "class": "human"},
                {"top": 5, "left": 5, "width": 20, "height": 20, "class": "bicycle"},
            ],
        },
        {
            "image_name": "frame_000002.jpg",
            "time": "2026-07-01T15:42:12Z",
            "bbox": [
                {"top": 50, "left": 50, "width": 30, "height": 30, "class": "car"},
            ],
        },
        {
            # references an image that doesn't exist on disk -> should be skipped
            "image_name": "frame_missing.jpg",
            "time": "2026-07-01T15:42:13Z",
            "bbox": [],
        },
    ]
    (root / "annotations.json").write_text(json.dumps(records))
    return root


@pytest.fixture
def make_synthetic_video(tmp_path: Path):
    """Factory fixture: build a small synthetic .avi (MJPG) video on disk.

    MJPG/AVI is used rather than mp4v/mp4 because it's reliably supported by
    OpenCV builds without extra system codec packages, keeping the test
    suite self-contained.
    """

    def _make(
        name: str = "clip.avi",
        num_frames: int = 5,
        fps: float = 10.0,
        size: tuple[int, int] = (64, 48),
    ) -> Path:
        import cv2

        path = tmp_path / name
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, fps, size)
        for i in range(num_frames):
            frame = np.full((size[1], size[0], 3), (i * 20) % 256, dtype=np.uint8)
            writer.write(frame)
        writer.release()
        return path

    return _make


@pytest.fixture
def nadir_camera_intrinsics() -> CameraIntrinsics:
    """A simple, symmetric pinhole camera with no distortion."""
    return CameraIntrinsics(
        focal_length_px=(1000.0, 1000.0),
        principal_point_px=(960.0, 540.0),
        image_size_px=(1920, 1080),
    )


@pytest.fixture
def rigid_mount_extrinsics() -> CameraExtrinsics:
    """Camera rigidly mounted with no offset (co-located with drone body origin)."""
    return CameraExtrinsics(translation_m=(0.0, 0.0, 0.0), rotation_deg=(0.0, 0.0, 0.0))


@pytest.fixture
def nadir_telemetry() -> Telemetry:
    """Drone hovering at 100m, pointed straight down, level attitude, no gimbal."""
    return Telemetry(
        timestamp="2026-07-01T15:42:11Z",
        latitude=30.352814,
        longitude=76.364921,
        altitude_msl=118.2,  # 118.2m MSL, ground assumed at 100m MSL -> 18.2m AGL
        yaw=0.0,
        pitch=0.0,
        roll=0.0,
        gimbal_yaw=0.0,
        gimbal_pitch=-90.0,  # pointed straight down
        gimbal_roll=0.0,
    )
