"""Dataset loading and annotation parsing for aerial detection sources.

Public entry points:

    from vision_model.datasets import build_dataset, available_datasets
    from vision_model.datasets.schema import BoxAnnotation, ImageSample
"""

from vision_model.datasets.registry import available_datasets, build_dataset, register
from vision_model.datasets.schema import BoxAnnotation, ImageSample

__all__ = [
    "BoxAnnotation",
    "ImageSample",
    "available_datasets",
    "build_dataset",
    "register",
]
