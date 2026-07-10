"""Stable data contracts shared across all vision_model packages.

Import from here, not from the submodules directly, e.g.:

    from vision_model.interfaces import Detection, Telemetry
"""

from vision_model.interfaces.camera import CameraExtrinsics, CameraIntrinsics
from vision_model.interfaces.detection import ClassName, Detection
from vision_model.interfaces.geotag import GeoTaggedDetection
from vision_model.interfaces.telemetry import Telemetry

__all__ = [
    "CameraExtrinsics",
    "CameraIntrinsics",
    "ClassName",
    "Detection",
    "GeoTaggedDetection",
    "Telemetry",
]
