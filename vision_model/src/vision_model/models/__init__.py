"""Detector backends behind the stable `Detector` interface.

Public entry points:

    from vision_model.models import build_detector, available_detectors
    from vision_model.models.base import Detector
"""

from vision_model.models.base import Detector
from vision_model.models.registry import available_detectors, build_detector, register

__all__ = ["Detector", "available_detectors", "build_detector", "register"]
