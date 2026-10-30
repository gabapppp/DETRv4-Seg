import json
import shutil
import zipfile
from types import SimpleNamespace

import pytest

from onnx_viewer.portable import run_viewer, update_viewer
from scripts.package_onnx_viewer import PACKAGE_NAME, PROJECT, package_viewer


def portable_package(tmp_path):
    models = tmp_path / "export"
    models.mkdir()
    for name in ("detector.onnx", "mask.onnx"):
        (models / name).write_bytes(b"synthetic model bytes for checksum tests")
    (models / "manifest.json").write_text(
        json.dumps({"detector": "detector.onnx", "mask": "mask.onnx"})
    )
    archive = tmp_path / "package.zip"
    package_viewer(models, archive)
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(tmp_path / "test_machine")
    root = tmp_path / "test_machine" / PACKAGE_NAME
    # Simulate rsync of application source into an existing portable install.
    shutil.copytree(
        PROJECT / "onnx_viewer",
        root / "onnx_viewer",
        dirs_exist_ok=True,
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    return root


def test_rsync_update_keeps_old_model_hashes_until_explicit_model_refresh(tmp_path, monkeypatch):
    root = portable_package(tmp_path)
    before = json.loads((root / "checksums.json").read_text())
    app = root / "onnx_viewer/app.py"
    app.write_text(app.read_text() + "\n# simulated updated app\n")
    (root / "models/detector.onnx").write_bytes(b"explicitly transferred new model")
    real_digest = update_viewer.digest

    def code_only(path):
        assert "models" not in path.parts and "sample_images" not in path.parts
        return real_digest(path)

    monkeypatch.setattr(update_viewer, "digest", code_only)
    update_viewer.update(root)
    after = json.loads((root / "checksums.json").read_text())
    assert before["models/detector.onnx"] == after["models/detector.onnx"]
    assert before["onnx_viewer/app.py"] != after["onnx_viewer/app.py"]
    monkeypatch.setattr(run_viewer, "ROOT", root)
    with pytest.raises(ValueError, match="checksum mismatch: models/detector.onnx"):
        run_viewer.verify_files()
    monkeypatch.setattr(update_viewer, "digest", real_digest)
    update_viewer.update(root, refresh_models=True)
    run_viewer.verify_files()
    assert (root / "run_viewer.py").read_bytes() == (
        root / "onnx_viewer/portable/run_viewer.py"
    ).read_bytes()


def test_portable_launcher_bootstraps_missing_pip_before_install(tmp_path, monkeypatch):
    (tmp_path / "checksums.json").write_text("{}")
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.touch()
    monkeypatch.setattr(run_viewer, "ROOT", tmp_path)
    monkeypatch.setattr(run_viewer.sys, "argv", ["run_viewer.py"])
    calls = []

    def subprocess(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=1 if "-c" in command else 0)

    monkeypatch.setattr(run_viewer.subprocess, "run", subprocess)
    assert run_viewer.main() == 0
    ensure = [str(python), "-m", "ensurepip", "--upgrade"]
    install = next(command for command in calls if "install" in command)
    assert calls.index(ensure) < calls.index(install)
    assert calls[-1] == [str(python), "-m", "onnx_viewer"]
