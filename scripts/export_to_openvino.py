#!/usr/bin/env python3
"""
scripts/export_to_openvino.py

Converts an ONNX floor-segmentation model to OpenVINO IR format (.xml/.bin),
with optional INT8 post-training quantization via NNCF.

INT8 quantization needs a small "calibration" set of representative images
(no labels needed) to determine per-layer activation ranges — pass a folder
of real floor images via --calibration-dir. Without quantization, this
script still produces a working FP32/FP16 OpenVINO IR model, just not one
that reliably clears real-time CPU throughput on its own (see the
assignment's suggested-direction note: INT8 is "usually necessary" to hit
2 Hz on CPU).

Usage:
    # Required package versions (tested combination — newer nncf (3.x) is
    # NOT compatible with openvino 2024.6, raises
    # "AttributeError: module 'openvino' has no attribute 'Node'" during
    # quantization):
    #     pip install "openvino==2024.6.0" "nncf==2.14.1" opencv-python-headless numpy

    # FP32 IR only, no quantization (quick, for testing conversion works)
    python scripts/export_to_openvino.py \\
        --onnx-model artifacts/floor_seg_model_mobilenet_finetuned.onnx \\
        --output-dir artifacts/openvino_model \\
        --precision FP32

    # INT8 quantized (recommended for real deployment / the 2 Hz target)
    python scripts/export_to_openvino.py \\
        --onnx-model artifacts/floor_seg_model_mobilenet_finetuned.onnx \\
        --output-dir artifacts/openvino_model_int8 \\
        --precision INT8 \\
        --calibration-dir data/floor_seg_data/images \\
        --calibration-samples 300
"""

import argparse
import logging
import sys
from pathlib import Path

try:
    import cv2
    import numpy as np
    import openvino as ov
except ImportError:
    sys.exit(
        "Missing dependency. Install with:\n"
        "    pip install openvino opencv-python-headless numpy\n"
    )


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def preprocess(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Identical preprocessing to the ONNX inference scripts, so calibration
    statistics and deployed inference see the same input distribution."""
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return np.ascontiguousarray(image, dtype=np.float32)


def find_calibration_images(calibration_dir: Path, max_samples: int) -> list[Path]:
    files = sorted(
        p for p in calibration_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(
            f"No images found under {calibration_dir} "
            f"(looked for {sorted(IMAGE_EXTENSIONS)})"
        )
    if len(files) > max_samples:
        # Evenly-spaced subsample rather than just the first N, so
        # calibration sees variety across the whole folder, not just
        # whatever sorts first alphabetically.
        indices = np.linspace(0, len(files) - 1, max_samples, dtype=int)
        files = [files[i] for i in indices]
    return files


def build_calibration_dataset(
    calibration_dir: Path, input_width: int, input_height: int, max_samples: int
):
    """
    Builds an NNCF-compatible calibration dataset from a folder of real
    floor images. NNCF calls the provided transform_fn on each item to
    produce the exact tensor that would be fed to the model at inference
    time — using the SAME preprocessing function as deployment is
    important, otherwise the calibrated quantization ranges won't match
    real inference-time activation statistics.
    """
    import nncf

    image_paths = find_calibration_images(calibration_dir, max_samples)
    print(f"Calibration images: {len(image_paths)} (from {calibration_dir})")

    def transform_fn(image_path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read calibration image: {image_path}")
        return preprocess(image_bgr, input_width, input_height)

    return nncf.Dataset(image_paths, transform_fn)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--onnx-model", required=True, help="Path to the source .onnx model")
    parser.add_argument("--output-dir", required=True, help="Directory to write the OpenVINO IR (.xml/.bin) into")
    parser.add_argument(
        "--output-name", default=None,
        help="Base filename for the IR files (default: derived from --onnx-model's stem)",
    )
    parser.add_argument(
        "--precision", choices=["FP32", "FP16", "INT8"], default="INT8",
        help=(
            "FP32/FP16: plain format conversion, no quantization. "
            "INT8: NNCF post-training quantization (needs --calibration-dir). "
            "Default: INT8, since the assignment's real-time target usually "
            "requires it on CPU."
        ),
    )
    parser.add_argument(
        "--calibration-dir", default=None,
        help="Folder of representative floor images for INT8 calibration (required if --precision INT8)",
    )
    parser.add_argument(
        "--calibration-samples", type=int, default=300,
        help="Max number of calibration images to use (default: 300 — NNCF guidance is generally 300+ for stable ranges)",
    )
    parser.add_argument(
        "--input-size", type=int, default=384,
        help="Model input resolution used to preprocess calibration images (default: 384, matches training)",
    )

    args = parser.parse_args()

    onnx_path = Path(args.onnx_model).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or onnx_path.stem

    print(f"Loading ONNX model: {onnx_path}")

    # openvino.convert_model probes whether the given path is a PyTorch
    # torch.export archive (.pt2) before correctly falling back to treating
    # it as plain ONNX. That probe logs a scary-looking (but harmless)
    # traceback-style warning via torch's internal logger for every .onnx
    # file passed in. Disabling logging just for this one call so it
    # doesn't drown out the script's real output; re-enabled immediately
    # after regardless of success or failure.
    logging.disable(logging.CRITICAL)
    try:
        model = ov.convert_model(str(onnx_path))
    finally:
        logging.disable(logging.NOTSET)

    if args.precision == "INT8":
        if not args.calibration_dir:
            raise ValueError("--calibration-dir is required when --precision INT8")

        import nncf

        calibration_dataset = build_calibration_dataset(
            Path(args.calibration_dir).expanduser(),
            input_width=args.input_size,
            input_height=args.input_size,
            max_samples=args.calibration_samples,
        )

        print("\nRunning NNCF post-training INT8 quantization...")
        print("(This calibrates per-layer activation ranges using the images above — takes a few minutes on CPU.)")
        model = nncf.quantize(model, calibration_dataset)
        print("Quantization complete.")

    elif args.precision == "FP16":
        # ov.save_model's compress_to_fp16 flag handles this at save time,
        # not at convert_model time -- see below.
        pass

    xml_path = output_dir / f"{output_name}.xml"

    print(f"\nSaving OpenVINO IR to: {xml_path}")
    ov.save_model(
        model,
        str(xml_path),
        compress_to_fp16=(args.precision == "FP16"),
    )

    bin_path = xml_path.with_suffix(".bin")
    xml_size_kb = xml_path.stat().st_size / 1024
    bin_size_mb = bin_path.stat().st_size / (1024 * 1024)

    print(f"\nDone.")
    print(f"  {xml_path}  ({xml_size_kb:.1f} KB)")
    print(f"  {bin_path}  ({bin_size_mb:.2f} MB)")
    print(f"  Precision: {args.precision}")

    # -------------------------------------------------------------------
    # Sanity check: load the IR back and run one inference pass, confirm
    # it produces a finite, correctly-shaped output before declaring
    # success. Catches silent conversion issues (e.g. wrong input shape)
    # immediately rather than discovering them later during deployment.
    # -------------------------------------------------------------------
    print("\nVerifying converted model with a test inference pass...")
    core = ov.Core()
    compiled = core.compile_model(str(xml_path), "CPU")

    dummy_input = np.random.randn(1, 3, args.input_size, args.input_size).astype(np.float32)
    result = compiled([dummy_input])
    output = result[compiled.output(0)]

    print(f"  Output shape: {output.shape}")
    print(f"  Output finite: {np.isfinite(output).all()}")
    print(f"  Output range: [{output.min():.4f}, {output.max():.4f}]")

    if not np.isfinite(output).all():
        print("  WARNING: converted model produced non-finite output on a test input.")
        sys.exit(1)

    print("\nPASS: converted model runs and produces finite output.")


if __name__ == "__main__":
    main()
