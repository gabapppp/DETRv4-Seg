import builtins
import json
from types import SimpleNamespace

import pytest
from test_train_seg_coco import synthetic_export

from detrseg.cli import build_parser
from detrseg.tracking import current_tracker, public_uri, tracked_training
from scripts.train_seg_coco import main as train_seg_main


@pytest.fixture
def tracking_store(tmp_path, monkeypatch):
    # Real SDK/storage, isolated from any tracking server configured by the user.
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    monkeypatch.setenv("MLFLOW_ENABLE_TELEMETRY", "false")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.delenv("MLFLOW_EXPERIMENT_NAME", raising=False)
    mlflow = pytest.importorskip("mlflow")
    uri = (tmp_path / "mlruns").as_uri()
    return uri, mlflow.MlflowClient(tracking_uri=uri)


def find_run(client, name="DETRv4-Seg"):
    experiment = client.get_experiment_by_name(name)
    runs = client.search_runs([experiment.experiment_id])
    assert len(runs) == 1
    return runs[0]


def test_disabled_tracking_never_imports_mlflow(tmp_path, monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    original = builtins.__import__

    def guard(name, *args, **kwargs):
        if name == "mlflow" or name.startswith("mlflow."):
            raise AssertionError("Disabled tracking imported MLflow")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guard)

    @tracked_training("seg")
    def dummy(output, mlflow_uri=None):
        tracker = current_tracker()
        tracker.params({"x": 1})
        tracker.metrics({"train/loss": 1.0}, 1)
        assert not tracker.should_log(1)
        return "ok"

    assert dummy(tmp_path / "unused") == "ok"
    assert not (tmp_path / "unused").exists()


def test_environment_url_and_explicit_override(tracking_store, tmp_path, monkeypatch):
    uri, client = tracking_store
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://should-not-be-contacted.invalid")

    @tracked_training("det")
    def dummy(output, mlflow_uri=None, mlflow_run_name=None):
        current_tracker().metrics({"val/AP": 0.25, "ignore_nonfinite": float("nan")}, 7)

    dummy(tmp_path / "unused", mlflow_uri=uri, mlflow_run_name="override")
    run = find_run(client)
    assert run.info.status == "FINISHED"
    assert run.data.tags["task"] == "det"
    assert run.data.tags["mlflow.runName"] == "override"
    assert run.data.metrics == {"val/AP": 0.25}
    assert "mlflow_uri" not in run.data.params
    assert client.get_metric_history(run.info.run_id, "val/AP")[0].step == 7


def test_missing_client_gives_install_command(tmp_path, monkeypatch):
    original = builtins.__import__

    def missing_client(name, *args, **kwargs):
        if name == "mlflow":
            raise ModuleNotFoundError("not installed")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_client)

    @tracked_training("seg")
    def dummy(output, mlflow_uri=None):
        pytest.fail("Training started without the requested tracking client")

    with pytest.raises(RuntimeError, match="uv pip install"):
        dummy(tmp_path / "unused", mlflow_uri="http://localhost:5000")


def test_server_connection_failure_stops_before_training(tracking_store, tmp_path, monkeypatch):
    import mlflow

    def disconnected(*args, **kwargs):
        raise ConnectionError("tracking server unavailable")

    monkeypatch.setattr(mlflow, "MlflowClient", disconnected)

    @tracked_training("det")
    def dummy(output, mlflow_uri=None):
        pytest.fail("Training started before MLflow connection succeeded")

    with pytest.raises(RuntimeError, match="Cannot start MLflow"):
        dummy(tmp_path / "unused", mlflow_uri="http://localhost:5000")
    assert not current_tracker().run_id


@pytest.mark.parametrize(
    "error,status", [(ValueError("bad batch"), "FAILED"), (KeyboardInterrupt(), "KILLED")]
)
def test_training_error_closes_run_and_restores_tracker(
    tracking_store, tmp_path, monkeypatch, error, status
):
    uri, client = tracking_store
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)

    @tracked_training("seg")
    def dummy(output):
        current_tracker().metrics({"train/loss": 1.5}, 1)
        raise error

    with pytest.raises(type(error)):
        dummy(tmp_path / "unused")
    run = find_run(client)
    assert run.info.status == status
    assert run.data.metrics["train/loss"] == 1.5
    assert not current_tracker().run_id


def test_async_tracking_failure_preserves_training_result(tracking_store, tmp_path, capsys):
    uri, client = tracking_store

    class FailedOperation:
        def wait(self):
            raise ConnectionError("server disconnected")

    @tracked_training("seg")
    def dummy(output, mlflow_uri=None):
        tracker = current_tracker()
        tracker.pending.append(FailedOperation())
        return {"checkpoint_saved": True}

    assert dummy(tmp_path / "unused", mlflow_uri=uri) == {"checkpoint_saved": True}
    assert find_run(client).info.status == "FINISHED"
    assert "training continues locally" in capsys.readouterr().out


def test_validation_flushes_and_is_visible_before_run_finishes(
    tracking_store, tmp_path, monkeypatch
):
    uri, client = tracking_store
    original_batch = type(client).log_batch
    validation_modes = []

    def log_batch(self, *args, **kwargs):
        if any(metric.key.startswith("val/") for metric in kwargs.get("metrics", [])):
            validation_modes.append(kwargs.get("synchronous"))
        return original_batch(self, *args, **kwargs)

    monkeypatch.setattr(type(client), "log_batch", log_batch)

    @tracked_training("seg")
    def dummy(output, mlflow_uri=None):
        tracker = current_tracker()
        tracker.metrics({"train/loss": 1.0}, 1)
        tracker.metrics({"val/iou": 0.8, "best/iou": 0.8}, 2, synchronous=True)
        assert not tracker.pending
        active = client.get_run(tracker.run_id)
        assert active.info.status == "RUNNING"
        assert active.data.metrics["train/loss"] == 1.0
        assert active.data.metrics["val/iou"] == active.data.metrics["best/iou"] == 0.8

    dummy(tmp_path / "unused", mlflow_uri=uri)
    assert validation_modes == [True]
    assert find_run(client).info.status == "FINISHED"


def test_seg_launcher_logs_real_training_and_validation(tracking_store, tmp_path, monkeypatch):
    from detrseg import train as mask_training

    uri, client = tracking_store
    original_validate = mask_training.validate
    live_checks = []

    def validate_with_live_check(*args, **kwargs):
        if live_checks:
            active = find_run(client)
            assert active.info.status == "RUNNING"
            assert "best/iou" in active.data.metrics
            assert [
                point.step for point in client.get_metric_history(active.info.run_id, "val/iou")
            ] == [2]
        live_checks.append(True)
        return original_validate(*args, **kwargs)

    monkeypatch.setattr(mask_training, "validate", validate_with_live_check)
    dataset, _, _, _ = synthetic_export(tmp_path)
    output = tmp_path / "seg_run"
    train_seg_main(
        [
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--architecture",
            "tiny",
            "--no-pretrained-encoder",
            "--device",
            "cpu",
            "--size",
            "32",
            "--steps",
            "3",
            "--batch-size",
            "2",
            "--val-every",
            "2",
            "--budget-seconds",
            "60",
            "--mlflow-uri",
            uri,
            "--mlflow-run-name",
            "seg-test",
            "--mlflow-log-every",
            "2",
        ]
    )
    run = find_run(client)
    assert run.info.status == "FINISHED"
    assert run.data.tags["task"] == "seg"
    assert run.data.params["steps"] == "3"
    assert run.data.params["roi_size"] == "32"
    assert {
        "train/loss",
        "train/lr",
        "val/loss",
        "val/iou",
        "val/dice",
        "best/iou",
    } <= run.data.metrics.keys()
    assert [point.step for point in client.get_metric_history(run.info.run_id, "train/loss")] == [
        1,
        2,
        3,
    ]
    assert [point.step for point in client.get_metric_history(run.info.run_id, "val/iou")] == [2, 3]
    assert [point.step for point in client.get_metric_history(run.info.run_id, "best/iou")] == [
        2,
        3,
    ]
    assert len(live_checks) == 2
    info = json.loads((output / "mask/mlflow.json").read_text())
    assert info["run_id"] == run.info.run_id
    assert (output / "mask/best.pt").is_file()


def test_detector_training_loop_logs_losses_and_ap(tracking_store, tmp_path, monkeypatch):
    import torch

    from detrseg import detector_train

    uri, client = tracking_store
    _, images, annotations, _ = synthetic_export(tmp_path)

    class SmallDetector(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 2)
            self.head = torch.nn.Linear(2, 1)

        def forward(self, images, targets=None):
            return {"prediction": self.head(self.backbone(images.mean((1, 2, 3))[:, None]))}

    class SmallCriterion(torch.nn.Module):
        def forward(self, outputs, targets):
            return {"loss_bbox": outputs["prediction"].square().mean() + 0.1}

    model = SmallDetector()
    checkpoint = tmp_path / "initial.pt"
    torch.save({"model": model.state_dict(), "model_size": "m"}, checkpoint)
    cfg = SimpleNamespace(
        model=model,
        criterion=SmallCriterion(),
        postprocessor=torch.nn.Identity(),
        yaml_cfg={"num_classes": 1},
    )
    monkeypatch.setattr(detector_train, "configuration", lambda *args: cfg)
    live_checks = []

    def validate_with_live_check(*args):
        if live_checks:
            active = find_run(client)
            assert active.info.status == "RUNNING"
            assert active.data.metrics["best/AP"] == 0.25
            assert [
                point.step for point in client.get_metric_history(active.info.run_id, "val/AP")
            ] == [2]
        live_checks.append(True)
        return {"AP": 0.25, "AP50": 0.5}, []

    monkeypatch.setattr(detector_train, "validate_detector", validate_with_live_check)
    output = tmp_path / "det_run"
    result = detector_train.train_detector(
        annotations,
        images,
        output,
        pretrained=checkpoint,
        repo=tmp_path / "fake_repo",
        steps=3,
        size=128,
        batch_size=2,
        val_every=2,
        device="cpu",
        threads=1,
        budget_seconds=60,
        mlflow_uri=uri,
        mlflow_log_every=2,
    )
    run = find_run(client)
    assert result["steps"] == 3 and run.info.status == "FINISHED"
    assert run.data.tags["task"] == "det"
    assert run.data.params["model_size"] == "m"
    assert {
        "train/loss",
        "train/loss_bbox",
        "train/lr_backbone",
        "val/AP50",
        "best/AP",
    } <= run.data.metrics.keys()
    assert run.data.metrics["val/AP"] == 0.25
    assert run.data.metrics["best/step"] == 2
    assert [point.step for point in client.get_metric_history(run.info.run_id, "val/AP")] == [2, 3]
    assert [point.step for point in client.get_metric_history(run.info.run_id, "best/AP")] == [2, 3]
    assert len(live_checks) == 2
    assert json.loads((output / "mlflow.json").read_text())["run_id"] == run.info.run_id
    assert (output / "best.pt").is_file()


@pytest.mark.parametrize(
    "command,required",
    [
        ("train", ["--cache", "fake_cache", "--output", "fake_run"]),
        (
            "train-detector",
            ["--annotations", "ann.json", "--images", "images", "--output", "fake_run"],
        ),
        ("fit", ["--annotations", "ann.json", "--images", "images", "--output", "fake_run"]),
    ],
)
def test_training_cli_accepts_tracking_url(command, required):
    args = build_parser().parse_args([command, *required, "--mlflow-uri", "http://localhost:5000"])
    assert args.mlflow_uri == "http://localhost:5000"


def test_tracking_uri_redacts_password_and_query():
    assert (
        public_uri("https://user:secret@host:5000/mlflow?token=secret")
        == "https://host:5000/mlflow"
    )
