"""Name -> dataset class registry.

`configs/dataset/*.yaml` reference a dataset by name (e.g. `name: visdrone`);
this registry resolves that name to a concrete `AerialDetectionDataset`
subclass. Adding a custom dataset means writing a new subclass and calling
`register()` — no other module needs to change.
"""

from __future__ import annotations

from typing import Any

from vision_model.datasets.auair import AUAIRDataset
from vision_model.datasets.base import AerialDetectionDataset
from vision_model.datasets.visdrone import VisDroneDataset
from vision_model.utils.exceptions import ConfigError

_REGISTRY: dict[str, type[AerialDetectionDataset]] = {}


def register(name: str, dataset_cls: type[AerialDetectionDataset]) -> None:
    """Register a dataset class under `name`.

    Args:
        name: Lookup key used in dataset configs (case-insensitive).
        dataset_cls: `AerialDetectionDataset` subclass to register.

    Raises:
        ConfigError: If `name` is already registered to a different class.
    """
    key = name.lower()
    existing = _REGISTRY.get(key)
    if existing is not None and existing is not dataset_cls:
        raise ConfigError(
            f"Dataset name '{name}' is already registered to {existing.__name__}; "
            f"cannot re-register to {dataset_cls.__name__}."
        )
    _REGISTRY[key] = dataset_cls


def build_dataset(name: str, **kwargs: Any) -> AerialDetectionDataset:
    """Construct a registered dataset by name.

    Args:
        name: Registered dataset name, e.g. `"visdrone"` or `"auair"`.
        **kwargs: Forwarded to the dataset class constructor (`root`,
            `split`, `transform`, ...).

    Returns:
        An instantiated `AerialDetectionDataset` subclass.

    Raises:
        ConfigError: If `name` is not registered.
    """
    key = name.lower()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY)) or "<none registered>"
        raise ConfigError(f"Unknown dataset '{name}'. Available: {available}")
    return _REGISTRY[key](**kwargs)


def available_datasets() -> list[str]:
    """List all registered dataset names."""
    return sorted(_REGISTRY)


# Built-in registrations. Custom datasets register themselves the same way
# from their own module, e.g. at the bottom of a `datasets/my_dataset.py`:
#     from vision_model.datasets.registry import register
#     register("my_dataset", MyDataset)
register("visdrone", VisDroneDataset)
register("auair", AUAIRDataset)
