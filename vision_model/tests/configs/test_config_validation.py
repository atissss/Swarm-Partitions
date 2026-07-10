from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from vision_model.utils.config_validation import validate_config
from vision_model.utils.exceptions import ConfigError

CONFIG_DIR = str((Path(__file__).parent.parent.parent / "configs").resolve())


def _compose(overrides: list[str] | None = None):
    with initialize_config_dir(version_base=None, config_dir=CONFIG_DIR):
        return compose(config_name="config", overrides=overrides or [])


class TestValidateConfigHappyPath:
    def test_default_config_is_valid(self) -> None:
        validate_config(_compose())

    @pytest.mark.parametrize("dataset", ["visdrone", "auair"])
    def test_all_datasets_valid(self, dataset: str) -> None:
        validate_config(_compose([f"dataset={dataset}"]))

    @pytest.mark.parametrize("camera", ["generic_pinhole", "dji_mavic3"])
    def test_all_cameras_valid(self, camera: str) -> None:
        validate_config(_compose([f"camera={camera}"]))


class TestValidateConfigRejectsBadValues:
    def test_out_of_range_model_confidence_threshold(self) -> None:
        with pytest.raises(ConfigError, match="model.confidence_threshold"):
            validate_config(_compose(["model.confidence_threshold=1.5"]))

    def test_negative_model_iou_threshold(self) -> None:
        with pytest.raises(ConfigError, match="model.iou_threshold"):
            validate_config(_compose(["model.iou_threshold=-0.1"]))

    def test_zero_epochs(self) -> None:
        with pytest.raises(ConfigError, match="train.epochs"):
            validate_config(_compose(["train.epochs=0"]))

    def test_zero_batch(self) -> None:
        with pytest.raises(ConfigError, match="train.batch"):
            validate_config(_compose(["train.batch=0"]))

    def test_negative_focal_length(self) -> None:
        with pytest.raises(ConfigError, match="focal_length_px"):
            validate_config(_compose(["camera.intrinsics.focal_length_px=[-100.0,1000.0]"]))

    def test_principal_point_outside_image(self) -> None:
        with pytest.raises(ConfigError, match="principal_point_px"):
            validate_config(_compose(["camera.intrinsics.principal_point_px=[9999.0,540.0]"]))

    def test_invalid_ground_model(self) -> None:
        with pytest.raises(ConfigError, match="ground_model"):
            validate_config(_compose(["geotag.ground_model=made_up_model"]))

    def test_negative_max_telemetry_delta(self) -> None:
        with pytest.raises(ConfigError, match="max_telemetry_delta_seconds"):
            validate_config(_compose(["geotag.max_telemetry_delta_seconds=-1.0"]))

    def test_unsupported_inference_allowed_class(self) -> None:
        with pytest.raises(ConfigError, match="allowed_classes"):
            validate_config(_compose(["inference.allowed_classes=[car,bicycle]"]))
