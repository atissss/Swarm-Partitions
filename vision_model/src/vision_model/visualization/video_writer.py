"""Write annotated (boxes + labels drawn) video output.

Consumes the `(frame, FrameResult)` stream from `inference.batch_runner.
run_on_video` and writes each drawn frame straight to disk — frames are
never held in memory beyond one at a time, so this scales to long videos.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vision_model.inference.batch_runner import FrameResult
from vision_model.utils.exceptions import VisionModelError
from vision_model.utils.io import ensure_dir
from vision_model.utils.logging_setup import get_logger
from vision_model.visualization.draw_boxes import draw_detections

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class VideoAnnotationSummary:
    """Summary of an `annotate_video` run.

    Attributes:
        output_path: Path the annotated video was written to.
        num_frames: Total frames written.
        num_detections: Total detections drawn across all frames.
    """

    output_path: Path
    num_frames: int
    num_detections: int


def _require_cv2() -> Any:
    try:
        import cv2

        return cv2
    except ImportError as exc:  # pragma: no cover - exercised without the optional dep
        raise VisionModelError(
            "opencv-python is required for video writing. Install it with "
            "`pip install opencv-python-headless`."
        ) from exc


def write_annotated_video(
    frames: Iterable[tuple[np.ndarray, FrameResult]],
    output_path: str | Path,
    fps: float,
    fourcc: str = "mp4v",
    show_confidence: bool = True,
) -> VideoAnnotationSummary:
    """Draw detections onto each frame and write them out as a video.

    Args:
        frames: An iterable of `(frame, FrameResult)`, typically
            `inference.batch_runner.run_on_video(...)`.
        output_path: Destination video file path. The extension should
            match `fourcc` (e.g. `.mp4` for `"mp4v"`, `.avi` for `"MJPG"`).
        fps: Output video frame rate.
        fourcc: FourCC codec identifier. `"mp4v"` is broadly compatible;
            `"MJPG"` (with a `.avi` output path) is a safe fallback in
            environments without an MP4-capable OpenCV build.
        show_confidence: If True, append confidence scores to box labels.

    Returns:
        A `VideoAnnotationSummary` with the output path and frame/detection
        counts.

    Raises:
        VisionModelError: If `frames` is empty, or the output video file
            can't be opened for writing.
    """
    cv2 = _require_cv2()
    dest = Path(output_path)
    ensure_dir(dest.parent)

    writer = None
    num_frames = 0
    num_detections = 0

    try:
        for frame, result in frames:
            if writer is None:
                height, width = frame.shape[:2]
                writer = cv2.VideoWriter(
                    str(dest), cv2.VideoWriter_fourcc(*fourcc), fps, (width, height)
                )
                if not writer.isOpened():
                    raise VisionModelError(
                        f"Could not open {dest} for writing with fourcc='{fourcc}'. "
                        "Try a different codec/extension pair, e.g. fourcc='MJPG' with a "
                        ".avi output path."
                    )

            annotated = draw_detections(frame, result.detections, show_confidence=show_confidence)
            writer.write(annotated)
            num_frames += 1
            num_detections += len(result.detections)
    finally:
        if writer is not None:
            writer.release()

    if num_frames == 0:
        raise VisionModelError("No frames were provided to write_annotated_video.")

    logger.info(
        "Wrote annotated video: %d frames, %d detections -> %s",
        num_frames,
        num_detections,
        dest,
    )
    return VideoAnnotationSummary(
        output_path=dest, num_frames=num_frames, num_detections=num_detections
    )
