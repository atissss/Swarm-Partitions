"""Base dataset interface every aerial detection dataset backend implements.

New sources (a custom drone dataset, a new public benchmark) are added by
subclassing `AerialDetectionDataset` and registering the subclass in
`datasets.registry` — no changes to `models`, `training`, or `inference` are
required.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from vision_model.datasets.schema import ImageSample

try:  # pragma: no cover - exercised only when torch is unavailable
    from torch.utils.data import Dataset as _TorchDataset

    _HAS_TORCH = True
    _BASES: tuple[type, ...] = (_TorchDataset, ABC)
except ImportError:  # pragma: no cover
    _HAS_TORCH = False
    _BASES = (ABC,)


class AerialDetectionDataset(*_BASES):  # type: ignore[misc]
    """Abstract base for all aerial object-detection dataset backends.

    Subclasses implement `load_index()` to build a list of `ImageSample`
    (image path + ground-truth boxes) from a dataset's raw directory
    structure. `__len__`/`__getitem__` are implemented once here in terms of
    that index, so subclasses never need to touch PyTorch's `Dataset`
    protocol directly.

    Attributes:
        root: Root directory of the dataset on disk.
        split: Dataset split, e.g. "train", "val", "test".
        transform: Optional callable applied to `(image, annotations)`
            before returning them (typically an Albumentations pipeline from
            `datasets.augmentations`).
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform: Any | None = None,
    ) -> None:
        if not _HAS_TORCH:
            import warnings

            warnings.warn(
                "torch is not installed; AerialDetectionDataset will work for "
                "indexing/annotation-parsing use cases but cannot be handed "
                "to a torch DataLoader.",
                stacklevel=2,
            )
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self._index: list[ImageSample] = self.load_index()

    @abstractmethod
    def load_index(self) -> list[ImageSample]:
        """Build the list of `ImageSample` for this dataset/split.

        Returns:
            All samples (image path + ground-truth boxes) for `self.split`.
        """
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return a single sample.

        Args:
            index: Sample index in `[0, len(self))`.

        Returns:
            A dict with keys `"image_path"`, `"image_id"`, `"annotations"`
            (list of `BoxAnnotation`), and `"telemetry_timestamp"`. If
            `self.transform` is set, it is applied to
            `(image_path, annotations)` and its result is merged in under
            `"transformed"`.
        """
        sample = self._index[index]
        item: dict[str, Any] = {
            "image_path": sample.image_path,
            "image_id": sample.image_id,
            "annotations": sample.annotations,
            "telemetry_timestamp": sample.telemetry_timestamp,
        }
        if self.transform is not None:
            item["transformed"] = self.transform(sample)
        return item

    @property
    def index(self) -> list[ImageSample]:
        """Read-only view of the full sample index."""
        return list(self._index)
