"""Refresh a portable viewer's launch files/checksums after an explicit rsync update."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def update(root, refresh_models=False):
    root = Path(root).resolve()
    checksum_file = root / "checksums.json"
    if not checksum_file.is_file():
        raise ValueError("Choose the extracted portable package containing checksums.json")
    expected = json.loads(checksum_file.read_text(encoding="utf-8"))
    portable = root / "onnx_viewer/portable"
    for name in (
        "run_viewer.py",
        "run_linux.sh",
        "run_windows.bat",
        "verify_package.py",
        "README_VI.md",
    ):
        shutil.copy2(portable / name, root / name)
    shutil.copy2(root / "onnx_viewer/requirements.txt", root / "requirements.txt")
    code = [
        root / name
        for name in (
            "run_viewer.py",
            "run_linux.sh",
            "run_windows.bat",
            "verify_package.py",
            "README_VI.md",
            "requirements.txt",
        )
    ]
    code.extend(
        path
        for path in (root / "onnx_viewer").rglob("*")
        if path.is_file() and path.suffix in (".py", ".txt", ".sh", ".bat", ".md")
    )
    code.extend(root / "detrseg" / name for name in ("__init__.py", "roi.py"))
    for path in code:
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Code path escapes portable package: {path}")
        expected[path.relative_to(root).as_posix()] = digest(path)
    if refresh_models:
        models = root / "models"
        manifest = models / "manifest.json"
        config = json.loads(manifest.read_text(encoding="utf-8"))
        for path in [manifest, *(models / config[key] for key in ("detector", "mask"))]:
            if not path.resolve().is_relative_to(models.resolve()):
                raise ValueError(f"Model path escapes models directory: {path}")
            expected[path.relative_to(root).as_posix()] = digest(path)
    temporary = checksum_file.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(expected, indent=2) + "\n", encoding="utf-8")
    temporary.replace(checksum_file)
    return {
        "updated_code_files": len(code),
        "models_refreshed": refresh_models,
        "package": str(root),
        "next": "bash run_linux.sh",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Portable package folder after rsync")
    parser.add_argument(
        "--refresh-models",
        action="store_true",
        help="Accept checksums for explicitly transferred new models",
    )
    args = parser.parse_args(argv)
    try:
        print(json.dumps(update(args.root, args.refresh_models), indent=2))
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"Viewer update failed: {exc}\n")


if __name__ == "__main__":
    main()
