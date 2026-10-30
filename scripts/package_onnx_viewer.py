#!/usr/bin/env python3
"""Package the viewer, ONNX models and optional test images into a relocatable ZIP."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "DETRv4-Seg-ONNX-test"


def file_digest(path):
    with Path(path).open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def package_viewer(model_dir, output, images_root=None):
    model_dir, output = Path(model_dir).resolve(), Path(output).resolve()
    manifest = json.loads((model_dir / "manifest.json").read_text(encoding="utf-8"))
    files = {}
    for name in ("__init__.py", "__main__.py", "app.py", "runtime.py", "requirements.txt"):
        files[f"onnx_viewer/{name}"] = PROJECT / "onnx_viewer" / name
    # The shared geometry module imports only NumPy/OpenCV. Training code is excluded.
    for name in ("__init__.py", "roi.py"):
        files[f"detrseg/{name}"] = PROJECT / "detrseg" / name
    for name in (
        "run_viewer.py",
        "verify_package.py",
        "run_linux.sh",
        "run_windows.bat",
        "README_VI.md",
    ):
        files[name] = PROJECT / "onnx_viewer/portable" / name
    files["requirements.txt"] = PROJECT / "onnx_viewer/requirements.txt"
    for key in ("detector", "mask"):
        if key not in manifest:
            raise ValueError(f"Expected a complete detector + mask bundle, missing {key}")
        source = (model_dir / manifest[key]).resolve()
        if not source.is_relative_to(model_dir) or not source.is_file():
            raise ValueError(f"Invalid or missing model file: {manifest[key]}")
        files[f"models/{key}.onnx"] = source
        manifest[key] = f"{key}.onnx"
    files["models/manifest.json"] = (
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    ).encode()
    verification_path = model_dir / "check/verification.json"
    samples = []
    if verification_path.exists():
        files["model_verification.json"] = verification_path
        verification = json.loads(verification_path.read_text(encoding="utf-8"))
        if images_root:
            image_root = Path(images_root).resolve()
            for index, item in enumerate(verification.get("images", [])[:3], 1):
                source = (image_root / item["image"]).resolve()
                if not source.is_relative_to(image_root) or not source.is_file():
                    raise ValueError(f"Sample image missing or invalid: {item['image']}")
                name = f"sample_images/{index:02d}_{source.name}"
                files[name] = source
                samples.append({"file": name, "expected_objects_at_score_0.5": item["objects"]})
    files["PACKAGE_INFO.json"] = (
        json.dumps(
            {
                "format": "detrv4-seg-onnx-test-package-v1",
                "runtime": "ONNX Runtime CPU",
                "python": "3.11-3.13; recommended 3.12",
                "model": "models/manifest.json",
                "samples": samples,
            },
            indent=2,
        )
        + "\n"
    ).encode()
    hashes = {
        name: hashlib.sha256(source).hexdigest()
        if isinstance(source, bytes)
        else file_digest(source)
        for name, source in files.items()
    }
    files["checksums.json"] = (json.dumps(hashes, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, source in files.items():
            destination = f"{PACKAGE_NAME}/{name}"
            if isinstance(source, bytes):
                archive.writestr(destination, source)
            else:
                archive.write(source, destination)
    return {
        "zip": str(output),
        "size_mb": round(output.stat().st_size / 2**20, 2),
        "files": len(files),
        "samples": len(samples),
        "sha256": file_digest(output),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="Exported directory with manifest.json")
    parser.add_argument("--output", required=True, help="New ZIP path")
    parser.add_argument("--images-root", help="Image root for the 3 previously verified samples")
    args = parser.parse_args()
    try:
        print(json.dumps(package_viewer(args.model_dir, args.output, args.images_root), indent=2))
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"Packaging failed: {exc}\n")


if __name__ == "__main__":
    main()
