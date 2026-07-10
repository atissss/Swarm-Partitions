"""Config-loading tests: every shipped config combination should compose
without error via Hydra, and required keys should resolve to sane types.

Paths are relative to this test file's location so the suite doesn't depend
on the current working directory the test runner happens to be invoked from.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

CONFIG_DIR = str((Path(__file__).parent.parent.parent / "configs").resolve())


@pytest.mark.parametrize("dataset", ["visdrone", "auair"])
def test_root_config_composes_for_each_dataset(dataset: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"dataset={dataset}"])
    assert cfg.dataset.name == dataset
    assert cfg.model.name == "yolo"
    assert cfg.seed == 42


def test_default_composition_has_expected_top_level_groups() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")
    for key in ("dataset", "model", "camera", "train", "inference", "geotag"):
        assert key in cfg, f"missing config group: {key}"


def test_train_epochs_overridable_on_command_line() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=["train.epochs=5"])
    assert cfg.train.epochs == 5


def test_model_class_map_has_car_and_person() -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config")
    values = set(cfg.model.class_map.values())
    assert values == {"car", "person"}


@pytest.mark.parametrize("camera", ["generic_pinhole", "dji_mavic3"])
def test_camera_configs_compose(camera: str) -> None:
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        cfg = compose(config_name="config", overrides=[f"camera={camera}"])
    assert len(cfg.camera.intrinsics.focal_length_px) == 2
    assert len(cfg.camera.intrinsics.image_size_px) == 2
