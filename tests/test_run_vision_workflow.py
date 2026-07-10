from types import SimpleNamespace

from scripts.run_vision_workflow import build_train_command


def test_train_command_uses_split_override_keys() -> None:
    args = SimpleNamespace(
        mode="train",
        dataset_root=None,
        train_root="custom/train",
        val_root=None,
        model=None,
        epochs=5,
        run_name="exp_test",
        telemetry=None,
        camera="dji_mavic3",
        python="python",
    )

    cmd = build_train_command(args)

    assert cmd[0:3] == ["python", "-m", "vision_model.cli.train"]
    assert "dataset=visdrone" in cmd
    assert "dataset.splits.train.root=custom/train" in cmd
    assert "dataset.splits.val.root=data/VisDrone2019-DET-val" in cmd
    assert "train.epochs=5" in cmd
    assert "train.run_name=exp_test" in cmd
