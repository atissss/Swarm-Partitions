"""VisDrone-DET dataset backend.

Expects the standard VisDrone-DET directory layout for a given split, e.g.::

    <root>/
        images/
            0000001_00000_d_0000001.jpg
            ...
        annotations/
            0000001_00000_d_0000001.txt
            ...

Point `root` at the split-specific directory (e.g.
`.../VisDrone2019-DET-train`) — the dataset config (`configs/dataset/
visdrone.yaml`) is responsible for resolving which split's directory that
is, keeping this class oblivious to the on-disk convention for split names.
"""

from __future__ import annotations

from pathlib import Path

from vision_model.datasets.annotation_parsers import parse_visdrone_annotation
from vision_model.datasets.base import AerialDetectionDataset
from vision_model.datasets.schema import ImageSample
from vision_model.utils.exceptions import DatasetError
from vision_model.utils.logging_setup import get_logger

logger = get_logger(__name__)

_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


class VisDroneDataset(AerialDetectionDataset):
    """Aerial detection dataset backend for VisDrone-DET.

    Attributes:
        images_dir: Resolved `<root>/images` directory.
        annotations_dir: Resolved `<root>/annotations` directory.
    """

    def __init__(self, root: str | Path, split: str = "train", transform=None) -> None:
        self.images_dir = Path(root) / "images"
        self.annotations_dir = Path(root) / "annotations"
        super().__init__(root=root, split=split, transform=transform)

    def load_index(self) -> list[ImageSample]:
        """Scan `images_dir`/`annotations_dir` and build the sample index.

        Returns:
            One `ImageSample` per image file that has a matching annotation
            file. Images without a matching `.txt` annotation are skipped
            with a warning (VisDrone ships one annotation file per image, but
            partial/corrupted downloads can drop files).

        Raises:
            DatasetError: If `images_dir` does not exist.
        """
        if not self.images_dir.is_dir():
            raise DatasetError(f"VisDrone images directory not found: {self.images_dir}")

        image_paths = sorted(
            p for p in self.images_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
        )
        if not image_paths:
            logger.warning("No images found under %s", self.images_dir)

        samples: list[ImageSample] = []
        for image_path in image_paths:
            annotation_path = self.annotations_dir / f"{image_path.stem}.txt"
            if not annotation_path.is_file():
                logger.warning(
                    "Skipping %s: no matching annotation at %s", image_path.name, annotation_path
                )
                continue

            annotations = parse_visdrone_annotation(annotation_path)
            samples.append(
                ImageSample(
                    image_id=image_path.stem,
                    image_path=image_path,
                    annotations=annotations,
                )
            )

        logger.info(
            "VisDroneDataset[%s]: indexed %d samples from %s",
            self.split,
            len(samples),
            self.root,
        )
        return samples
