import hashlib
from pathlib import Path

import pytest

from detrseg import upstream
from detrseg.cli import build_parser
from detrseg.pyramid import p2_config


@pytest.mark.parametrize("command", ["setup-detector", "fit", "train-detector"])
def test_model_size_cli_defaults_and_s_override(command):
    required = (
        []
        if command == "setup-detector"
        else ["--annotations", "ann.json", "--images", "images", "--output", "run"]
    )
    parser = build_parser()
    args = parser.parse_args([command, *required])
    assert args.model_size == "m"
    assert getattr(args, "weights" if command == "setup-detector" else "pretrained") is None
    assert parser.parse_args([command, *required, "--model-size", "s"]).model_size == "s"


@pytest.mark.parametrize("model_size", ["s", "m"])
def test_bootstrap_download_selection_and_integrity(tmp_path, monkeypatch, model_size):
    gdown = pytest.importorskip("gdown")

    content = f"official-{model_size}".encode()
    digest = hashlib.sha256(content).hexdigest()
    monkeypatch.setitem(upstream.CHECKPOINTS[model_size], "sha256", digest)
    monkeypatch.setattr(upstream.subprocess, "check_output", lambda *a, **kw: upstream.UPSTREAM_REV)
    downloads = []

    def download(*, id, output, quiet):
        downloads.append(id)
        Path(output).write_bytes(content)
        return output

    monkeypatch.setattr(gdown, "download", download)
    monkeypatch.chdir(tmp_path)
    repo = tmp_path / "repo"
    repo.mkdir()
    report = upstream.bootstrap(repo, model_size=model_size)
    weights = Path(upstream.default_weights(model_size))
    assert downloads == [upstream.CHECKPOINTS[model_size]["id"]]
    assert weights.read_bytes() == content
    assert report["model_size"] == model_size
    assert report["weights_sha256"] == digest
    upstream.bootstrap(repo, model_size=model_size)
    assert len(downloads) == 1
    weights.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        upstream.bootstrap(repo, model_size=model_size)
    weights.unlink()
    monkeypatch.setitem(upstream.CHECKPOINTS[model_size], "sha256", "bad hash")
    with pytest.raises(ValueError, match="checksum mismatch"):
        upstream.bootstrap(repo, model_size=model_size)
    assert not weights.exists()  # A bad download must never become the final checkpoint.


@pytest.mark.parametrize(
    "model_size,channels", [("s", [64, 256, 512, 1024]), ("m", [96, 384, 768, 1536])]
)
def test_p2_channels_match_backbone(model_size, channels):
    assert p2_config(model_size)["HybridEncoder"]["in_channels"] == channels


def test_invalid_size_fails_before_setup(tmp_path):
    with pytest.raises(ValueError, match="Unsupported detector model size"):
        upstream.bootstrap(tmp_path / "absent", model_size="x")
    assert not (tmp_path / "absent").exists()
