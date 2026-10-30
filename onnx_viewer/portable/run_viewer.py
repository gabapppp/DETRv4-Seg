"""Create an isolated runtime on first launch, then start the bundled viewer."""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def verify_files():
    expected = json.loads((ROOT / "checksums.json").read_text(encoding="utf-8"))
    for relative, digest in expected.items():
        path = (ROOT / relative).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f"Package file missing or invalid: {relative}")
        with path.open("rb") as file:
            actual = hashlib.file_digest(file, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"Package file checksum mismatch: {relative}")
    print(f"Package verified: {len(expected)} files", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true", help="Check models on a sample without opening GUI"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Verify package files without installing dependencies",
    )
    args = parser.parse_args()
    if not (3, 11) <= sys.version_info[:2] <= (3, 13):
        raise RuntimeError("Install Python 3.12 (Python 3.11 to 3.13 supported), then run again")
    verify_files()
    if args.verify_only:
        return 0
    environment = ROOT / ".venv"
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not python.is_file():
        print("Creating local Python environment...", flush=True)
        subprocess.run([sys.executable, "-m", "venv", str(environment)], check=True)
    probe = subprocess.run(
        [str(python), "-c", "import PyQt6.QtWidgets, onnxruntime, cv2, numpy, PIL"],
        cwd=ROOT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if probe.returncode:
        print("Installing viewer dependencies (Internet needed on first launch)...", flush=True)
        pip_probe = subprocess.run(
            [str(python), "-c", "import pip"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if pip_probe.returncode:
            print("Bootstrapping pip in the viewer environment...", flush=True)
            subprocess.run([str(python), "-m", "ensurepip", "--upgrade"], cwd=ROOT, check=True)
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--timeout",
                "600",
                "--retries",
                "5",
                "--index-url",
                "https://pypi.org/simple",
                "-r",
                str(ROOT / "requirements.txt"),
            ],
            cwd=ROOT,
            check=True,
        )
    command = (
        [str(python), str(ROOT / "verify_package.py")]
        if args.check
        else [str(python), "-m", "onnx_viewer"]
    )
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Cannot start viewer: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
