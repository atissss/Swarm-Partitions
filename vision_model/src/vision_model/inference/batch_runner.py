"""Video and directory batch inference.

`run_on_video` streams frames one at a time (via a generator) rather than
loading a whole video into memory, and computes each frame's capture
timestamp from the video's frame rate — the same pattern a future
`LiveStreamRunner` (Phase 6, RTSP/UDP feeds) would use, since both just need
something that yields `(frame, timestamp)` pairs into `Predictor.
predict_image`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision_model.inference.predictor import Predictor
from vision_model.interfaces.detection import Detection
from vision_model.utils.exceptions import InferenceError
from vision_model.utils.logging_setup import get_logger
from vision_model.utils.time_sync import frame_timestamp

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


@dataclass(frozen=True, slots=True)
class FrameResult:
    """Detections for one frame, plus the metadata needed to place it in time.

    Attributes:
        frame_index: Zero-based index of this frame within its source video
            (always 0 for directory/single-image runs).
        frame_id: Identifier stamped onto each `Detection.frame_id`.
        timestamp: ISO-8601 UTC timestamp of this frame.
        detections: Post-processed detections for this frame.
    """

    frame_index: int
    frame_id: str
    timestamp: str
    detections: list[Detection]


def _require_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - exercised without the optional dep
        raise InferenceError(
            "opencv-python is required for video inference. Install it with "
            "`pip install opencv-python-headless`."
        ) from exc


def get_video_fps(video_path: str | Path) -> float:
    """Read a video's frame rate.

    Args:
        video_path: Path to a video file.

    Returns:
        Frames per second, as reported by the container/codec.

    Raises:
        InferenceError: If the video can't be opened, or reports a
            non-positive frame rate (common for corrupt files or unusual
            containers OpenCV can open but not fully parse).
    """
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise InferenceError(f"Could not open video: {video_path}")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
    finally:
        cap.release()

    if fps <= 0:
        raise InferenceError(
            f"Video {video_path} reports a non-positive fps ({fps}); pass `fps=` explicitly."
        )
    return fps


def iter_video_frames(video_path: str | Path) -> Iterator[tuple[int, np.ndarray]]:
    """Stream `(frame_index, frame)` pairs from a video file.

    Args:
        video_path: Path to a video file.

    Yields:
        `(frame_index, frame)` for each frame, in order, where `frame` is a
        BGR `np.ndarray` (OpenCV convention).

    Raises:
        InferenceError: If the video can't be opened.
    """
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise InferenceError(f"Could not open video: {video_path}")

    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield index, frame
            index += 1
    finally:
        cap.release()


def run_on_video(
    predictor: Predictor,
    video_path: str | Path,
    start_timestamp: str,
    fps: float | None = None,
) -> Iterator[tuple[np.ndarray, FrameResult]]:
    """Run detection over every frame of a video, streaming results.

    Args:
        predictor: Configured `Predictor` to run inference with.
        video_path: Path to a video file.
        start_timestamp: ISO-8601 UTC timestamp of frame 0 (e.g. recording
            start time). Each subsequent frame's timestamp is derived from
            this plus `frame_index / fps` — see `utils.time_sync.
            frame_timestamp`.
        fps: Frame rate to use for timestamp computation. Defaults to the
            video's own reported frame rate (`get_video_fps`) if omitted.

    Yields:
        `(frame, FrameResult)` for each frame, in order — the raw frame is
        included so a caller (e.g. `visualization.video_writer`) can draw on
        it without re-reading the video.

    Raises:
        InferenceError: If the video can't be opened or has no usable frame
            rate and `fps` wasn't provided.
    """
    resolved_fps = fps if fps is not None else get_video_fps(video_path)
    video_id = Path(video_path).stem

    for index, frame in iter_video_frames(video_path):
        timestamp = frame_timestamp(start_timestamp, index, resolved_fps)
        frame_id = f"{video_id}:{index}"
        detections = predictor.predict_image(frame, timestamp=timestamp, frame_id=frame_id)
        yield frame, FrameResult(
            frame_index=index, frame_id=frame_id, timestamp=timestamp, detections=detections
        )


def run_on_image_directory(
    predictor: Predictor,
    images_dir: str | Path,
    get_timestamp: Callable[[Path], str],
) -> list[FrameResult]:
    """Run detection over every image in a directory.

    Args:
        predictor: Configured `Predictor` to run inference with.
        images_dir: Directory containing `.jpg`/`.jpeg`/`.png` images.
        get_timestamp: Callable resolving an image path to its ISO-8601
            capture timestamp (e.g. a lookup into an AU-AIR-style dataset's
            `telemetry_timestamp`, or a per-flight telemetry stream's
            coverage). Required explicitly — matching `Predictor.
            predict_image`'s own "no silent 'now' timestamp" rule, since
            geo-tagging accuracy depends on it.

    Returns:
        One `FrameResult` per image, sorted by file name.

    Raises:
        InferenceError: If `images_dir` doesn't exist or contains no
            supported image files.
    """
    directory = Path(images_dir)
    if not directory.is_dir():
        raise InferenceError(f"Images directory not found: {directory}")

    image_paths = sorted(p for p in directory.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS)
    if not image_paths:
        raise InferenceError(f"No supported images found in {directory}")

    results: list[FrameResult] = []
    for index, image_path in enumerate(image_paths):
        timestamp = get_timestamp(image_path)
        detections = predictor.predict_image(
            image_path, timestamp=timestamp, frame_id=image_path.stem
        )
        results.append(
            FrameResult(
                frame_index=index,
                frame_id=image_path.stem,
                timestamp=timestamp,
                detections=detections,
            )
        )
    return results
