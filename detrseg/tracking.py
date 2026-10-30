"""Optional MLflow tracking for explicit training runs, with no global active run."""

import inspect
import math
import os
import time
from contextvars import ContextVar
from functools import wraps
from numbers import Real
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from .coco import write_json

DEFAULT_EXPERIMENT = "DETRv4-Seg"
_current = ContextVar("detrseg_mlflow_tracker", default=None)


def add_tracking_arguments(parser):
    parser.add_argument("--mlflow-uri", help="MLflow server URL; also accepts MLFLOW_TRACKING_URI")
    parser.add_argument(
        "--mlflow-experiment", help=f"Experiment name (default: {DEFAULT_EXPERIMENT})"
    )
    parser.add_argument("--mlflow-run-name", help="Optional name for this training run")
    parser.add_argument(
        "--mlflow-log-every", type=int, default=20, help="Train metric interval in steps"
    )


def public_uri(uri):
    """Keep credentials and query tokens out of terminal output and run metadata."""
    parts = urlsplit(uri)
    if parts.scheme in ("http", "https"):
        return urlunsplit((parts.scheme, parts.netloc.rsplit("@", 1)[-1], parts.path, "", ""))
    return uri


class Tracker:
    def __init__(self, uri=None, experiment=None, run_name=None, log_every=20):
        if log_every < 1:
            raise ValueError("mlflow_log_every must be positive")
        self.uri = uri
        self.experiment = experiment or os.getenv("MLFLOW_EXPERIMENT_NAME") or DEFAULT_EXPERIMENT
        self.run_name = run_name
        self.log_every = log_every
        self.client = self.run_id = self.experiment_id = None
        self.pending = []
        self.error = False

    def start(self, task, params):
        if not self.uri:
            return
        try:
            from mlflow import MlflowClient
            from mlflow.entities import Metric, Param
        except ImportError as exc:
            raise RuntimeError(
                "MLflow tracking needs mlflow-skinny. Install with: "
                'uv pip install --python .venv-train/bin/python "mlflow-skinny>=2.17,<4"'
            ) from exc
        self.metric_type, self.param_type = Metric, Param
        try:
            self.client = MlflowClient(tracking_uri=self.uri)
            experiment = self.client.get_experiment_by_name(self.experiment)
            if experiment is None:
                try:
                    self.experiment_id = self.client.create_experiment(self.experiment)
                except Exception:
                    # Another training process may create the same experiment.
                    experiment = self.client.get_experiment_by_name(self.experiment)
                    if experiment is None:
                        raise
                    self.experiment_id = experiment.experiment_id
            else:
                self.experiment_id = experiment.experiment_id
            run = self.client.create_run(
                self.experiment_id,
                tags={"mlflow.runName": self.run_name, "task": task, "project": "DETRv4-Seg"},
            )
            self.run_id = run.info.run_id
            # Initial configuration must reach the server before expensive setup.
            self.client.log_batch(self.run_id, params=self._params(params), synchronous=True)
        except Exception as exc:
            raise RuntimeError(f"Cannot start MLflow tracking at {public_uri(self.uri)}") from exc
        print(f"MLflow {task}: {self.run_url or self.run_id}", flush=True)

    @property
    def run_url(self):
        if self.uri and self.run_id and urlsplit(self.uri).scheme in ("http", "https"):
            return f"{public_uri(self.uri).rstrip('/')}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
        return None

    def _params(self, values):
        return [self.param_type(str(key), str(value)[:6000]) for key, value in values.items()]

    def _failed(self, exc):
        if not self.error:
            print(
                f"MLflow logging failed ({type(exc).__name__}); training continues locally.",
                flush=True,
            )
        self.error = True

    def params(self, values):
        if self.run_id and not self.error:
            try:
                operation = self.client.log_batch(
                    self.run_id, params=self._params(values), synchronous=False
                )
                if operation is not None:
                    self.pending.append(operation)
            except Exception as exc:  # noqa: BLE001 - Tracking must not interrupt training.
                self._failed(exc)

    def should_log(self, step, last=False):
        return bool(
            self.run_id and not self.error and (step == 1 or step % self.log_every == 0 or last)
        )

    def metrics(self, values, step, synchronous=False):
        if not self.run_id or self.error:
            return
        if synchronous:
            self.flush()
            if self.error:
                return
        timestamp = int(time.time() * 1000)
        metrics = [
            self.metric_type(str(key), float(value), timestamp, int(step))
            for key, value in values.items()
            if isinstance(value, Real) and math.isfinite(value)
        ]
        if metrics:
            try:
                operation = self.client.log_batch(
                    self.run_id, metrics=metrics, synchronous=synchronous
                )
                if operation is not None:
                    self.pending.append(operation)
            except Exception as exc:  # noqa: BLE001 - Tracking must not interrupt training.
                self._failed(exc)

    def attach_output(self, output):
        if self.run_id:
            write_json(
                Path(output) / "mlflow.json",
                {
                    "tracking_uri": public_uri(self.uri),
                    "experiment": self.experiment,
                    "experiment_id": self.experiment_id,
                    "run_id": self.run_id,
                    "run_name": self.run_name,
                    "run_url": self.run_url,
                },
            )

    def flush(self):
        pending, self.pending = self.pending, []
        for operation in pending:
            try:
                operation.wait()
            except Exception as exc:  # noqa: BLE001 - Preserve the original training result.
                self._failed(exc)

    def finish(self, status):
        if self.run_id:
            self.flush()
            try:
                self.client.set_terminated(self.run_id, status=status)
            except Exception as exc:  # noqa: BLE001 - Preserve the original training result.
                self._failed(exc)


_disabled = Tracker()


def current_tracker():
    return _current.get() or _disabled


def tracked_training(task):
    """Cover setup, training, and failure cleanup in one isolated tracking run."""

    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            bound.apply_defaults()
            config = bound.arguments
            output = Path(config["output"])
            tracker = Tracker(
                uri=config.get("mlflow_uri") or os.getenv("MLFLOW_TRACKING_URI"),
                experiment=config.get("mlflow_experiment"),
                run_name=config.get("mlflow_run_name")
                or f"{task}-{output.parent.name}-{output.name}",
                log_every=config.get("mlflow_log_every", 20),
            )
            token = _current.set(tracker)
            try:
                tracker.start(
                    task,
                    {key: value for key, value in config.items() if not key.startswith("mlflow_")},
                )
                result = function(*args, **kwargs)
            except BaseException as exc:
                tracker.finish("KILLED" if isinstance(exc, KeyboardInterrupt) else "FAILED")
                raise
            else:
                tracker.finish("FINISHED")
                return result
            finally:
                _current.reset(token)

        return wrapped

    return decorate
