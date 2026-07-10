"""Frame -> `Detection` inference orchestration and post-processing.

Public entry points:

    from vision_model.inference import Predictor
    from vision_model.inference.batch_runner import run_on_video, run_on_image_directory
"""

from vision_model.inference.batch_runner import FrameResult, run_on_image_directory, run_on_video
from vision_model.inference.predictor import Predictor

__all__ = ["FrameResult", "Predictor", "run_on_image_directory", "run_on_video"]
