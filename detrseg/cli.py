import argparse
import json
import sys

from .coco import convert_cvat, read_json, validate_coco, write_json


def build_parser():
    parser = argparse.ArgumentParser(description="COCO/CVAT → pseudo masks → tiny ROI mask head")
    commands = parser.add_subparsers(dest="command", required=True)
    convert = commands.add_parser("convert-cvat", help="Convert CVAT image XML to COCO")
    convert.add_argument("--xml", required=True)
    convert.add_argument("--images", required=True)
    convert.add_argument("--output", required=True)
    audit = commands.add_parser("audit", help="Validate boxes, image files, and decoded COCO masks")
    audit.add_argument("--annotations", required=True)
    audit.add_argument("--images")
    audit.add_argument("--output")
    prepare = commands.add_parser("prepare", help="Generate/cache labels; preserve existing masks")
    prepare.add_argument("--annotations", required=True)
    prepare.add_argument("--images", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--size", type=int, default=96)
    prepare.add_argument("--padding", type=float, default=0.25)
    prepare.add_argument(
        "--mask-overflow",
        type=float,
        default=0.25,
        help="Allowed mask extension on each side, as a fraction of original bbox size",
    )
    prepare.add_argument("--max-objects", type=int, default=3000)
    prepare.add_argument("--val-fraction", type=float, default=0.2)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--grabcut", action="store_true")
    prepare.add_argument(
        "--fill-mode",
        choices=("holes", "convex"),
        default="holes",
        help="convex fills bright cores behind open rims; only for near-convex particles",
    )
    prepare.add_argument("--budget-seconds", type=float, default=420)
    prepare.add_argument("--groups", help='JSON map {"image_id": "independent_source_group"}')
    train = commands.add_parser("train", help="Train just the ROI head with BCE + Dice")
    train.add_argument("--cache", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--steps", type=int, default=600)
    train.add_argument("--budget-seconds", type=float, default=1020)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--device", default="auto")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--threads", type=int, default=4)
    train.add_argument("--architecture", choices=("tiny", "resnet18_unet"), default="tiny")
    train.add_argument("--no-pretrained-encoder", action="store_true")
    train.add_argument("--initial-checkpoint")
    train.add_argument("--val-every", type=int, default=100)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--no-amp", action="store_true")
    infer = commands.add_parser("infer", help="Attach native-image RLE masks to detection results")
    infer.add_argument("--annotations", required=True)
    infer.add_argument("--images", required=True)
    infer.add_argument("--detections", required=True)
    infer.add_argument("--checkpoint", required=True)
    infer.add_argument("--output", required=True)
    infer.add_argument("--device", default="auto")
    infer.add_argument("--batch-size", type=int, default=32)
    infer.add_argument("--threshold", type=float, default=0.5)
    infer.add_argument("--score-threshold", type=float, default=0.25)
    infer.add_argument("--overlays")
    evaluate = commands.add_parser(
        "evaluate", help="Compare masks at native resolution with oracle boxes"
    )
    evaluate.add_argument("--annotations", required=True)
    evaluate.add_argument("--images", required=True)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--split", choices=("train", "val", "all"), default="val")
    evaluate.add_argument("--batch-size", type=int, default=32)
    evaluate.add_argument("--threshold", type=float, default=0.5)
    evaluate.add_argument("--preview")
    detect = commands.add_parser("detect", help="Run an installed official RT-DETRv4 checkout")
    for name in ("repo", "config", "checkpoint", "annotations", "images", "label-map", "output"):
        detect.add_argument(f"--{name}", required=True)
    detect.add_argument("--device", default="auto")
    detect.add_argument("--score-threshold", type=float, default=0.25)
    detect.add_argument("--size", type=int)
    setup = commands.add_parser(
        "setup-detector", help="Fetch pinned official RT-DETRv4 S and COCO weights"
    )
    setup.add_argument("--repo", default="vendor/RT-DETRv4")
    setup.add_argument("--weights", default="weights/rtv4_s_coco.pth")
    detector_train = commands.add_parser(
        "train-detector", help="Fine-tune the RT-DETRv4 student on custom boxes"
    )
    for name in ("annotations", "images", "output"):
        detector_train.add_argument(f"--{name}", required=True)
    detector_train.add_argument("--repo", default="vendor/RT-DETRv4")
    detector_train.add_argument("--pretrained", default="weights/rtv4_s_coco.pth")
    detector_train.add_argument("--split-report")
    detector_train.add_argument("--steps", type=int, default=600)
    detector_train.add_argument("--batch-size", type=int, default=4)
    detector_train.add_argument("--size", type=int, default=640)
    detector_train.add_argument("--lr", type=float, default=2e-4)
    detector_train.add_argument("--budget-seconds", type=float, default=900)
    detector_train.add_argument("--val-every", type=int, default=100)
    detector_train.add_argument("--device", default="auto")
    detector_train.add_argument("--seed", type=int, default=42)
    detector_train.add_argument("--threads", type=int, default=4)
    detector_train.add_argument(
        "--ema-decay",
        type=float,
        default=0.0,
        help="Average model weights for validation; 0 disables EMA",
    )
    detector_train.add_argument(
        "--p2", action="store_true", help="Use P2/P3/P4/P5, strides 4/8/16/32"
    )
    detector_train.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Stop after this many validations without higher AP; 0 disables",
    )
    fit = commands.add_parser(
        "fit", help="Prepare masks, train both models, validate, and write a pipeline bundle"
    )
    for name in ("annotations", "images", "output"):
        fit.add_argument(f"--{name}", required=True)
    fit.add_argument("--repo", default="vendor/RT-DETRv4")
    fit.add_argument("--pretrained", default="weights/rtv4_s_coco.pth")
    fit.add_argument("--detector-steps", type=int, default=600)
    fit.add_argument("--mask-steps", type=int, default=600)
    fit.add_argument("--budget-seconds", type=float, default=1800)
    fit.add_argument("--device", default="auto")
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--fill-mode", choices=("holes", "convex"), default="holes")
    fit.add_argument("--groups")
    fit.add_argument("--detector-size", type=int, default=640)
    fit.add_argument("--detector-batch-size", type=int, default=4)
    fit.add_argument("--max-objects", type=int, default=3000)
    fit.add_argument("--padding", type=float, default=0.25)
    fit.add_argument("--mask-overflow", type=float, default=0.25)
    fit.add_argument("--p2", action="store_true")
    fit.add_argument("--mask-architecture", choices=("tiny", "resnet18_unet"), default="tiny")
    predict = commands.add_parser(
        "predict", help="Image/folder → detector → masks, overlays, COCO JSON and CSV"
    )
    for name in ("bundle", "input", "output"):
        predict.add_argument(f"--{name}", required=True)
    predict.add_argument("--device", default="auto")
    predict.add_argument("--score-threshold", type=float, default=0.25)
    predict.add_argument(
        "--mask-threshold", type=float, help="Override the mask threshold saved in the bundle"
    )
    predict.add_argument("--batch-size", type=int, default=32)
    predict.add_argument("--no-overlays", action="store_true")
    bundle = commands.add_parser(
        "bundle", help="Package fine-tuned detector and mask checkpoints together"
    )
    for name in (
        "config",
        "detector-checkpoint",
        "mask-checkpoint",
        "label-map",
        "annotations",
        "output",
    ):
        bundle.add_argument(f"--{name}", required=True)
    bundle.add_argument("--repo", default="vendor/RT-DETRv4")
    bundle.add_argument("--mask-threshold", type=float, default=0.5)
    quality = commands.add_parser(
        "audit-masks", help="Local flip/bbox stability and instance-overlap diagnostics"
    )
    for name in ("bundle", "annotations", "images", "detections", "output"):
        quality.add_argument(f"--{name}", required=True)
    quality.add_argument("--device", default="auto")
    quality.add_argument("--score-threshold", type=float, default=0.5)
    quality.add_argument("--batch-size", type=int, default=32)
    quality.add_argument("--max-objects", type=int, default=1000)
    return parser


def main(argv=None):
    args = vars(build_parser().parse_args(argv))
    command = args.pop("command")
    if "images" in args:
        args["images_root"] = args.pop("images")
    try:
        if command == "convert-cvat":
            coco = convert_cvat(args["xml"], args["images_root"])
            write_json(args["output"], coco)
            result = validate_coco(coco)
        elif command == "audit":
            result = validate_coco(read_json(args["annotations"]), args["images_root"])
            if args["output"]:
                write_json(args["output"], result)
        elif command == "prepare":
            from .prepare import prepare

            result = prepare(**args)
            result = {
                key: value
                for key, value in result.items()
                if key not in ("rejected_annotations", "image_splits")
            }
        elif command == "train":
            from .train import train

            args["amp"] = not args.pop("no_amp")
            args["pretrained_encoder"] = not args.pop("no_pretrained_encoder")
            result = train(**args)
        elif command == "infer":
            from .infer import infer

            result = infer(**args)
        elif command == "detect":
            from .detector import detect

            result = detect(**args)
        elif command == "evaluate":
            from .evaluate import evaluate

            result = evaluate(**args)
        elif command == "setup-detector":
            from .upstream import bootstrap

            result = bootstrap(**args)
        elif command == "train-detector":
            from .detector_train import train_detector

            result = train_detector(**args)
        elif command == "fit":
            from .pipeline import fit

            result = fit(**args)
        elif command == "predict":
            from .pipeline import predict

            args["overlays"] = not args.pop("no_overlays")
            result = predict(**args)
        elif command == "bundle":
            from .pipeline import create_bundle

            result = create_bundle(**args)
        elif command == "audit-masks":
            from .quality import audit_masks

            result = audit_masks(**args)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, TypeError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
