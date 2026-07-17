#!/usr/bin/env python3
"""
scripts/infer_openvino.py

Runs floor segmentation inference via OpenVINO IR, with AUTOMATIC hardware
detection: uses the integrated NPU if present, falls back to CPU otherwise.
No manual device switch — detection happens in code via
openvino.Core().available_devices.

Works on either a single image, a folder of images, or a video file.
Prints per-frame latency (preprocess / inference / postprocess broken out
separately) and effective Hz, matching the assignment's required benchmark
report inputs.

Usage:
    # Single image or folder of images
    python scripts/infer_openvino.py \\
        --model artifacts/openvino_model_int8/floor_seg_model.xml \\
        --input path/to/image_or_folder \\
        --output-dir artifacts/openvino_test_output

    # Video file
    python scripts/infer_openvino.py \\
        --model artifacts/openvino_model_int8/floor_seg_model.xml \\
        --input artifacts/downloaded.mp4 \\
        --output artifacts/openvino_segmented.mp4

    # Force a specific device instead of auto-detecting (for comparison/debugging)
    python scripts/infer_openvino.py --model ... --input ... --output-dir ... --device CPU
"""

import argparse
import sys
import time
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
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def select_device(core: ov.Core, requested_device: str | None) -> str:
    """
    Automatic hardware detection: prefers NPU if present, falls back to
    CPU. This is the core requirement from the assignment — no manual
    switch, detection happens here via Core().available_devices.

    If --device is explicitly passed, that overrides auto-detection
    (useful for benchmarking CPU vs NPU side by side on NPU-equipped
    hardware, or for forcing CPU on a machine where NPU drivers are
    present but flaky).
    """
    available = core.available_devices
    print(f"Available OpenVINO devices on this machine: {available}")

    if requested_device is not None:
        if requested_device not in available and requested_device != "AUTO":
            print(
                f"WARNING: requested device '{requested_device}' not in "
                f"available devices {available} — attempting anyway, "
                f"OpenVINO will raise a clear error if it truly isn't usable."
            )
        print(f"Using explicitly requested device: {requested_device}")
        return requested_device

    npu_present = any(d.startswith("NPU") for d in available)

    if npu_present:
        print("NPU detected — using NPU for inference.")
        return "NPU"

    print("No NPU detected — falling back to CPU for inference.")
    return "CPU"


def preprocess(image_bgr: np.ndarray, width: int, height: int) -> tuple[np.ndarray, float]:
    start = time.perf_counter()

    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    tensor = np.ascontiguousarray(image, dtype=np.float32)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return tensor, elapsed_ms


def postprocess_floor_mask(
    probability_map: np.ndarray,
    original_width: int,
    original_height: int,
    threshold: float,
    open_kernel_size: int,
    close_kernel_size: int,
    min_component_area_fraction: float,
) -> tuple[np.ndarray, float]:
    start = time.perf_counter()

    binary = (probability_map >= threshold).astype(np.uint8) * 255

    if open_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_kernel_size, open_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if close_kernel_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel_size, close_kernel_size))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    height, width = binary.shape
    min_area = min_component_area_fraction * height * width
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    clean = np.zeros_like(binary)
    for label_id in range(1, num_labels):
        if stats[label_id, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == label_id] = 1

    mask = cv2.resize(
        clean.astype(np.uint8), (original_width, original_height), interpolation=cv2.INTER_NEAREST,
    ).astype(bool)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return mask, elapsed_ms


def run_inference(compiled_model, output_layer, tensor: np.ndarray) -> tuple[np.ndarray, float]:
    start = time.perf_counter()
    result = compiled_model([tensor])
    output = result[output_layer]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms


def find_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(
        p for p in input_path.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def process_images(args, compiled_model, output_layer, model_width, model_height):
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = find_input_images(input_path)
    if not image_paths:
        raise FileNotFoundError(f"No images found at {input_path}")

    if args.max_frames is not None:
        image_paths = image_paths[: args.max_frames]

    print(f"Images found: {len(image_paths)}")

    preprocess_times, inference_times, postprocess_times, total_times = [], [], [], []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            print(f"  SKIP (unreadable): {image_path}")
            continue

        original_height, original_width = image_bgr.shape[:2]
        total_start = time.perf_counter()

        tensor, preprocess_ms = preprocess(image_bgr, model_width, model_height)
        output, inference_ms = run_inference(compiled_model, output_layer, tensor)

        probability = sigmoid(output[0, 0])
        mask, postprocess_ms = postprocess_floor_mask(
            probability, original_width, original_height,
            args.threshold, args.open_kernel_size, args.close_kernel_size,
            args.min_component_area_fraction,
        )

        total_ms = (time.perf_counter() - total_start) * 1000.0

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        postprocess_times.append(postprocess_ms)
        total_times.append(total_ms)

        coloured = image_bgr.copy()
        coloured[mask] = (0, 255, 0)
        result = cv2.addWeighted(image_bgr, 1.0 - args.alpha, coloured, args.alpha, 0)

        cv2.putText(
            result, f"{total_ms:.1f} ms ({1000.0 / total_ms:.1f} Hz)", (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), result)
        print(f"  {image_path.name}: {total_ms:.1f} ms total "
              f"(pre {preprocess_ms:.1f} / inf {inference_ms:.1f} / post {postprocess_ms:.1f})")

    print_summary(preprocess_times, inference_times, postprocess_times, total_times, args.hz_target)


def process_video(args, compiled_model, output_layer, model_width, model_height):
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    original_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (original_width, original_height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    preprocess_times, inference_times, postprocess_times, total_times = [], [], [], []

    frame_index = -1
    while True:
        if args.max_frames is not None and frame_index + 1 >= args.max_frames:
            break

        ok, frame = capture.read()
        if not ok:
            break
        frame_index += 1

        total_start = time.perf_counter()

        tensor, preprocess_ms = preprocess(frame, model_width, model_height)
        output, inference_ms = run_inference(compiled_model, output_layer, tensor)

        probability = sigmoid(output[0, 0])
        mask, postprocess_ms = postprocess_floor_mask(
            probability, original_width, original_height,
            args.threshold, args.open_kernel_size, args.close_kernel_size,
            args.min_component_area_fraction,
        )

        total_ms = (time.perf_counter() - total_start) * 1000.0

        preprocess_times.append(preprocess_ms)
        inference_times.append(inference_ms)
        postprocess_times.append(postprocess_ms)
        total_times.append(total_ms)

        coloured = frame.copy()
        coloured[mask] = (0, 255, 0)
        result = cv2.addWeighted(frame, 1.0 - args.alpha, coloured, args.alpha, 0)

        cv2.putText(
            result, f"{total_ms:.1f} ms ({1000.0 / total_ms:.1f} Hz)", (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        writer.write(result)

        if frame_index % 30 == 0:
            print(f"  frame {frame_index}/{frame_count if frame_count > 0 else '?'}: "
                  f"{total_ms:.1f} ms ({1000.0 / total_ms:.1f} Hz)")

    capture.release()
    writer.release()

    print(f"\nSaved: {output_path}")
    print_summary(preprocess_times, inference_times, postprocess_times, total_times, args.hz_target)


def print_summary(preprocess_times, inference_times, postprocess_times, total_times, hz_target):
    if not total_times:
        print("\nNo frames processed.")
        return

    mean_total = float(np.mean(total_times))
    max_total = float(np.max(total_times))
    p95_total = float(np.percentile(total_times, 95))

    print(f"\n{'=' * 60}")
    print(f"Frames processed: {len(total_times)}")
    print(f"{'Stage':<15}{'Mean (ms)':>12}{'Max (ms)':>12}{'P95 (ms)':>12}")
    print(f"{'Preprocess':<15}{np.mean(preprocess_times):>12.2f}{np.max(preprocess_times):>12.2f}{np.percentile(preprocess_times, 95):>12.2f}")
    print(f"{'Inference':<15}{np.mean(inference_times):>12.2f}{np.max(inference_times):>12.2f}{np.percentile(inference_times, 95):>12.2f}")
    print(f"{'Postprocess':<15}{np.mean(postprocess_times):>12.2f}{np.max(postprocess_times):>12.2f}{np.percentile(postprocess_times, 95):>12.2f}")
    print(f"{'TOTAL':<15}{mean_total:>12.2f}{max_total:>12.2f}{p95_total:>12.2f}")
    print(f"{'=' * 60}")
    print(f"Mean effective rate: {1000.0 / mean_total:.2f} Hz")
    print(f"Worst-case rate (max latency): {1000.0 / max_total:.2f} Hz")

    target_ms = 1000.0 / hz_target
    meets_target = max_total <= target_ms
    print(f"\nTarget: >= {hz_target} Hz ({target_ms:.1f} ms/frame budget)")
    print(f"Mean meets target: {'YES' if mean_total <= target_ms else 'NO'} ({mean_total:.1f} ms)")
    print(f"Worst-case meets target: {'YES' if meets_target else 'NO'} ({max_total:.1f} ms)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR .xml file")
    parser.add_argument("--input", required=True, help="Image file, folder of images, or video file")
    parser.add_argument("--output-dir", default=None, help="Output folder for image-mode overlays")
    parser.add_argument("--output", default=None, help="Output video path for video-mode")
    parser.add_argument(
        "--device", default=None,
        help="Force a specific OpenVINO device (e.g. CPU, NPU, AUTO). Default: auto-detect NPU, fall back to CPU.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.45)
    parser.add_argument("--open-kernel-size", type=int, default=5)
    parser.add_argument("--close-kernel-size", type=int, default=7)
    parser.add_argument("--min-component-area-fraction", type=float, default=0.0015)
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames/images (both video and image-folder mode)")
    parser.add_argument(
        "--hz-target", type=float, default=2.0,
        help="Real-time target for the summary's pass/fail check (default: 2.0, per the assignment spec)",
    )

    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    input_path = Path(args.input).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    is_video = input_path.is_file() and input_path.suffix.lower() in VIDEO_EXTENSIONS

    if is_video and not args.output:
        raise ValueError("--output is required for video input")
    if not is_video and not args.output_dir:
        raise ValueError("--output-dir is required for image input")

    core = ov.Core()
    device = select_device(core, args.device)

    print(f"\nCompiling model for device: {device}")
    model = core.read_model(str(model_path))
    compiled_model = core.compile_model(model, device)
    output_layer = compiled_model.output(0)

    input_shape = model.inputs[0].get_partial_shape()
    model_height = int(input_shape[2].get_length()) if input_shape[2].is_static else 384
    model_width = int(input_shape[3].get_length()) if input_shape[3].is_static else 384

    print(f"Model input size: {model_width} x {model_height}")

    if is_video:
        process_video(args, compiled_model, output_layer, model_width, model_height)
    else:
        process_images(args, compiled_model, output_layer, model_width, model_height)


if __name__ == "__main__":
    main()
