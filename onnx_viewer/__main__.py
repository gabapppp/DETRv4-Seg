"""Launch with ``python -m onnx_viewer`` from the repository root."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from .app import main as run
    except ImportError as exc:
        print(
            "Cannot start ONNX viewer. Install its dependencies first:\n"
            "  python -m pip install -r onnx_viewer/requirements.txt\n"
            f"Details: {exc}",
            file=sys.stderr,
        )
        return 1
    return run()


if __name__ == "__main__":
    raise SystemExit(main())
