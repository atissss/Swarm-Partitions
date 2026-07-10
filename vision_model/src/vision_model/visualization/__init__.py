"""Presentation-only helpers: drawing detections on frames and video.

Public entry points:

    from vision_model.visualization import draw_detections, save_annotated_image
    from vision_model.visualization import write_annotated_video
"""

from vision_model.visualization.draw_boxes import draw_detections, save_annotated_image
from vision_model.visualization.video_writer import VideoAnnotationSummary, write_annotated_video

__all__ = [
    "VideoAnnotationSummary",
    "draw_detections",
    "save_annotated_image",
    "write_annotated_video",
]
