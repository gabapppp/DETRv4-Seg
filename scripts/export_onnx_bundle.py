#!/usr/bin/env python3
"""Export a trained DETRv4-Seg bundle for the PyQt6 ONNX viewer."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Allow running this source script from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compare_outputs(expected, actual, output_names):
    """Compare DETR sets after matching queries; compare mask pixels in place."""
    import numpy as np

    if not len(expected) == len(actual) == len(output_names):
        raise AssertionError("ONNX output count differs from PyTorch")
    for name, before, after in zip(output_names, expected, actual):
        if before.shape != after.shape:
            raise AssertionError(f"ONNX shape mismatch for {name}: {after.shape} vs {before.shape}")
    matched = actual
    if output_names == ["logits", "pred_boxes"]:
        from scipy.optimize import linear_sum_assignment

        # Near-tied encoder Top-K scores can swap query order across backends.
        # Match each box and its class logits together, then apply the same strict
        # numeric checks. A different prediction must still fail verification.
        matched = [value.copy() for value in actual]
        for batch in range(expected[0].shape[0]):
            cost = np.abs(expected[1][batch, :, None] - actual[1][batch, None, :]).sum(-1)
            cost += 0.1 * np.abs(expected[0][batch, :, None] - actual[0][batch, None, :]).mean(-1)
            _, permutation = linear_sum_assignment(cost)
            for index, value in enumerate(actual):
                matched[index][batch] = value[batch, permutation]
    errors = {}
    for name, before, after in zip(output_names, expected, matched):
        np.testing.assert_allclose(
            after, before, rtol=0.003, atol=0.0005, err_msg=f"ONNX parity failed for {name}"
        )
        errors[name] = float(np.max(np.abs(after - before)))
    return errors


def export_bundle(bundle, output, *, mask_dynamic_batch=False):
    import onnx
    import onnxruntime as ort
    import torch

    from detrseg.pipeline import InstancePredictor, bundle_paths

    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be new or empty: {output}")
    state, _ = bundle_paths(bundle)
    predictor = InstancePredictor(bundle, "cpu")
    torch.set_num_threads(4)
    output.mkdir(parents=True, exist_ok=True)

    class Detector(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, images):
            result = self.model(images)
            return result["pred_logits"], result["pred_boxes"]

    def export_and_check(model, sample, path, input_name, output_names, dynamic_batch=False):
        model.eval()
        with torch.inference_mode():
            reference = model(sample)
        torch.onnx.export(
            model,
            sample,
            str(path),
            input_names=[input_name],
            output_names=output_names,
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
            dynamic_axes=(
                {name: {0: "batch"} for name in [input_name, *output_names]}
                if dynamic_batch
                else None
            ),
        )
        onnx.checker.check_model(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        actual = session.run(None, {input_name: sample.numpy()})
        reference = reference if isinstance(reference, tuple) else (reference,)
        expected = [value.detach().numpy() for value in reference]
        errors = compare_outputs(expected, actual, output_names)
        if dynamic_batch:
            for count in (2, 7, 32):
                batch = torch.rand(count, *sample.shape[1:])
                batch[:, 3] = (batch[:, 3] > 0.5).float()
                with torch.inference_mode():
                    expected_batch = model(batch).numpy()
                    individual = torch.cat([model(item[None]) for item in batch]).numpy()
                np_errors = compare_outputs([individual], [expected_batch], output_names)
                actual_batch = session.run(None, {input_name: batch.numpy()})
                batch_errors = compare_outputs([expected_batch], actual_batch, output_names)
                for name in errors:
                    errors[name] = max(errors[name], np_errors[name], batch_errors[name])
        return errors

    torch.manual_seed(42)
    detector_size = predictor.detector.size
    mask = predictor.mask_head
    detector_errors = export_and_check(
        Detector(predictor.detector.model),
        torch.rand(1, 3, detector_size, detector_size),
        output / "detector.onnx",
        "images",
        ["logits", "pred_boxes"],
    )
    mask_sample = torch.rand(1, 4, mask.size, mask.size)
    mask_sample[:, 3] = (mask_sample[:, 3] > 0.5).float()
    mask_errors = export_and_check(
        mask.model, mask_sample, output / "mask.onnx", "roi", ["mask_logits"], mask_dynamic_batch
    )
    categories = {int(item["id"]): item["name"] for item in state["categories"]}
    manifest = {
        "format": "detrv4-seg-onnx-v1",
        "detector": "detector.onnx",
        "mask": "mask.onnx",
        "detector_sha256": hashlib.sha256((output / "detector.onnx").read_bytes()).hexdigest(),
        "mask_sha256": hashlib.sha256((output / "mask.onnx").read_bytes()).hexdigest(),
        "detector_orientation": "display",
        "roi_size": mask.size,
        "mask_dynamic_batch": mask_dynamic_batch,
        "height": detector_size,
        "width": detector_size,
        "resize_mode": "resize",
        "do_rescale": True,
        "rescale_factor": 1 / 255,
        "do_normalize": False,
        "num_queries": 300,
        "id2label": {str(k): categories[v] for k, v in predictor.detector.mapping.items()},
        "padding": mask.padding,
        "mask_overflow": mask.mask_overflow,
        "fill_mode": mask.fill_mode,
        "mask_output": "logits",
        "roi_resize_mode": getattr(mask, "roi_resize_mode", "resize"),
        "mask_threshold": predictor.mask_threshold,
        "verification": {
            "detector_comparison": "matched unordered DETR query sets",
            "detector_max_abs_error": detector_errors,
            "mask_max_abs_error": mask_errors,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return {"manifest": str(output / "manifest.json"), "verification": manifest["verification"]}


def export_checkpoints(
    detector_dir, mask_checkpoint, output, repo="vendor/RT-DETRv4", *, mask_dynamic_batch=False
):
    """Build deployment metadata from training artifacts, then export on CPU."""
    from detrseg.pipeline import create_bundle

    detector_dir, output = Path(detector_dir).resolve(), Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"Output directory must be new or empty: {output}")
    bundle = output / "bundle.json"
    create_bundle(
        repo=repo,
        config=detector_dir / "config.yml",
        detector_checkpoint=detector_dir / "best.pt",
        mask_checkpoint=mask_checkpoint,
        label_map=detector_dir / "label_map.json",
        annotations=detector_dir / "train.json",
        output=bundle,
    )
    return {
        "bundle": str(bundle),
        **export_bundle(bundle, output / "onnx", mask_dynamic_batch=mask_dynamic_batch),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--bundle", help="Trained PyTorch bundle.json")
    source.add_argument(
        "--detector-dir",
        help="Train output containing best.pt, config.yml, label_map.json, train.json",
    )
    parser.add_argument("--mask-checkpoint", help="Seg mask/best.pt; required with --detector-dir")
    parser.add_argument("--repo", default="vendor/RT-DETRv4")
    parser.add_argument(
        "--mask-dynamic-batch",
        action="store_true",
        help="Export mask with dynamic batch; verify batches 1, 2, 7, 32",
    )
    parser.add_argument("--output", required=True, help="New or empty ONNX output directory")
    args = parser.parse_args(argv)
    if args.detector_dir and not args.mask_checkpoint:
        parser.error("--detector-dir requires --mask-checkpoint")
    if args.bundle and args.mask_checkpoint:
        parser.error("Use --mask-checkpoint with --detector-dir, not --bundle")
    try:
        result = (
            export_checkpoints(
                args.detector_dir,
                args.mask_checkpoint,
                args.output,
                args.repo,
                mask_dynamic_batch=args.mask_dynamic_batch,
            )
            if args.detector_dir
            else export_bundle(args.bundle, args.output, mask_dynamic_batch=args.mask_dynamic_batch)
        )
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, RuntimeError, ImportError, AssertionError, KeyError) as exc:
        parser.exit(1, f"Export failed: {exc}\n")


if __name__ == "__main__":
    main()
