"""Pinned upstream checkout and the official COCO checkpoint, kept outside Git."""

import hashlib
import subprocess
from pathlib import Path

from .coco import write_json

UPSTREAM_URL = "https://github.com/RT-DETRs/RT-DETRv4.git"
UPSTREAM_REV = "55fefaaed7efe2a5f72d0a18fd4e05965e35c292"
MODEL_SIZES = ("s", "m")
DEFAULT_MODEL_SIZE = "m"
CHECKPOINTS = {
    "s": {
        "id": "1jDAVxblqRPEWed7Hxm6GwcEl7zn72U6z",
        "sha256": "238a3f6537bf3b75b55e73f91f9d4cec8d21259b4908b3f21896f3e038b5a3ee",
    },
    "m": {
        "id": "1O-YpP4X-quuOXbi96y2TKkztbjroP5mX",
        "sha256": "8f1bd54437e9a227dc41ff3f65986f15097a4aab27e6a4e944c845e441f27501",
    },
}


def default_weights(model_size=DEFAULT_MODEL_SIZE):
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Unsupported detector model size: {model_size}")
    return f"weights/rtv4_{model_size}_coco.pth"


def bootstrap(repo="vendor/RT-DETRv4", weights=None, model_size=DEFAULT_MODEL_SIZE):
    selected_weights = default_weights(model_size)
    weights = weights if weights is not None else selected_weights
    checkpoint = CHECKPOINTS[model_size]
    repo, weights = Path(repo).resolve(), Path(weights).resolve()
    if not repo.exists():
        subprocess.run(["git", "clone", UPSTREAM_URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", UPSTREAM_REV], check=True)
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REV:
        raise ValueError(f"Expected upstream revision {UPSTREAM_REV}, got {revision}")
    if not weights.exists():
        import gdown

        weights.parent.mkdir(parents=True, exist_ok=True)
        temporary = weights.with_suffix(".download")
        if gdown.download(id=checkpoint["id"], output=str(temporary), quiet=False) is None:
            raise ValueError(
                "Could not download official weights; supply a local --pretrained file"
            )
        downloaded = temporary
    else:
        downloaded = weights
    with downloaded.open("rb") as file:
        digest = hashlib.file_digest(file, "sha256").hexdigest()
    if digest != checkpoint["sha256"]:
        raise ValueError(f"Official {model_size.upper()} pretrained checkpoint checksum mismatch")
    if downloaded != weights:
        downloaded.replace(weights)
    manifest = {
        "repo": str(repo),
        "revision": revision,
        "model_size": model_size,
        "weights": str(weights),
        "weights_sha256": digest,
        "weights_url": f"https://drive.google.com/file/d/{checkpoint['id']}",
    }
    write_json(weights.with_suffix(".json"), manifest)
    return manifest


def configuration(repo, config):
    """Import just one engine checkout per process; avoid mixing cached registries."""
    import sys

    repo = Path(repo).resolve()
    if not (repo / "engine/core").is_dir():
        raise ValueError("Missing RT-DETRv4 checkout. Run detrseg setup-detector first.")
    existing = sys.modules.get("engine")
    if existing is not None and not Path(existing.__file__).resolve().is_relative_to(repo):
        raise ValueError("A different engine package is already imported in this Python process")
    sys.path.insert(0, str(repo))
    from engine.core import YAMLConfig

    cfg = YAMLConfig(str(Path(config).resolve()))
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False
    return cfg


def move_model(model, device):
    """Upstream stores evaluation position embeddings as plain CPU attributes.

    Register them as nonpersistent buffers so .to() moves them without changing
    checkpoint keys or modifying the pinned upstream checkout.
    """
    import torch

    for module in model.modules():
        for name, value in list(vars(module).items()):
            if name.startswith("pos_embed") and isinstance(value, torch.Tensor):
                delattr(module, name)
                module.register_buffer(name, value, persistent=False)
    return model.to(device)
