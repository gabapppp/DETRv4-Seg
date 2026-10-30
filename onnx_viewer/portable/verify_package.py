"""Validate a relocated package and run its ONNX models on the first sample."""

import json
import sys
from pathlib import Path

from run_viewer import verify_files


def main():
    root = Path(__file__).resolve().parent
    verify_files()
    from onnx_viewer.runtime import OnnxPipeline, load_rgb

    pipeline = OnnxPipeline(root / "models/manifest.json")
    print(pipeline.model_description, flush=True)
    samples = sorted((root / "sample_images").glob("*"))
    if samples:
        result = pipeline.predict(load_rgb(samples[0]))
        print(
            json.dumps(
                {
                    "image": samples[0].name,
                    "objects": len(result["objects"]),
                    "circles": sum(obj["diameter_px"] is not None for obj in result["objects"]),
                    "elapsed_ms": result["elapsed_ms"],
                },
                indent=2,
            ),
            flush=True,
        )
    assert "torch" not in sys.modules, "Portable inference must not import PyTorch"
    print("ONNX Runtime CPU check: PASS", flush=True)


if __name__ == "__main__":
    main()
