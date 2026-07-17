import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from tqdm import tqdm


IMAGENET_MEAN = np.array(
    [0.485, 0.456, 0.406],
    dtype=np.float32,
).reshape(1, 1, 3)

IMAGENET_STD = np.array(
    [0.229, 0.224, 0.225],
    dtype=np.float32,
).reshape(1, 1, 3)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def postprocess_floor_mask(
    probability_map: np.ndarray,
    threshold: float,
    open_kernel_size: int,
    close_kernel_size: int,
    min_component_area_fraction: float,
    require_border_contact: bool,
    previous_probability_map: np.ndarray | None,
    temporal_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same cleanup as test_onnx_video.py. Included here (not imported) so
    this stays a single, self-contained file — and because timing this
    step is exactly the point of a benchmark script when --postprocess
    is requested, so it needs to be the real implementation, not a stub.
    """
    prob = probability_map.astype(np.float32)

    if temporal_alpha > 0.0 and previous_probability_map is not None:
        prob = (
            temporal_alpha * previous_probability_map
            + (1.0 - temporal_alpha) * prob
        )

    binary = (prob >= threshold).astype(np.uint8) * 255

    if open_kernel_size > 0:
        open_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (open_kernel_size, open_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)

    if close_kernel_size > 0:
        close_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (close_kernel_size, close_kernel_size),
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)

    height, width = binary.shape
    min_area = min_component_area_fraction * height * width

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    clean = np.zeros_like(binary)

    for label_id in range(1, num_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        if require_border_contact:
            x = stats[label_id, cv2.CC_STAT_LEFT]
            y = stats[label_id, cv2.CC_STAT_TOP]
            w = stats[label_id, cv2.CC_STAT_WIDTH]
            h = stats[label_id, cv2.CC_STAT_HEIGHT]
            touches_border = (
                x == 0 or y == 0 or (x + w) >= width or (y + h) >= height
            )
            if not touches_border:
                continue

        clean[labels == label_id] = 1

    return clean, prob


def preprocess(
    frame_bgr: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    image = cv2.cvtColor(
        frame_bgr,
        cv2.COLOR_BGR2RGB,
    )

    image = cv2.resize(
        image,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )

    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD

    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)

    return np.ascontiguousarray(image, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark-only ONNX inference timing on a video. Does NOT "
            "write any output video or frames — use test_onnx_video.py "
            "if you want to see/save the segmentation overlay. This is "
            "for measuring speed only, faster to run since it skips "
            "VideoWriter, mask compositing, and text overlay entirely."
        )
    )

    parser.add_argument("--model", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--fallback-size",
        type=int,
        default=384,
        help="Used when the ONNX model has dynamic spatial dimensions.",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help=(
            "Apply morphological cleanup + connected-component filtering "
            "to the mask, same as test_onnx_video.py. Include this if "
            "you want to benchmark the REAL deployed cost, since "
            "postprocessing adds real per-frame time on top of raw "
            "inference."
        ),
    )
    parser.add_argument(
        "--open-kernel-size", type=int, default=5,
        help="Morphological opening kernel size in pixels. 0 disables.",
    )
    parser.add_argument(
        "--close-kernel-size", type=int, default=7,
        help="Morphological closing kernel size in pixels. 0 disables.",
    )
    parser.add_argument(
        "--min-component-area-fraction", type=float, default=0.0015,
        help="Drop connected components smaller than this fraction of frame area.",
    )
    parser.add_argument(
        "--require-border-contact", action="store_true",
        help="Also drop floor blobs that don't touch the frame border.",
    )
    parser.add_argument(
        "--temporal-alpha", type=float, default=0.0,
        help="Exponential smoothing weight (0-1) for temporal mask smoothing. 0 disables.",
    )
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=1,
        help=(
            "Number of initial frames to exclude from timing statistics "
            "(default: 1). The first inference call always includes "
            "ONNX Runtime session warm-up cost, which isn't representative "
            "of sustained per-frame cost in deployment. Set to 0 to "
            "include every frame in the stats."
        ),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop after this many frames (default: process the whole video). Useful for a quick benchmark on a long video.",
    )
    parser.add_argument(
        "--max-latency-warn-ms",
        type=float,
        default=None,
        help=(
            "If set, print a warning line during the run for any frame "
            "whose end-to-end latency exceeds this many milliseconds."
        ),
    )

    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    input_path = Path(args.input).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    )

    session = ort.InferenceSession(
        str(model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )

    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]

    input_name = input_info.name
    output_name = output_info.name
    input_shape = input_info.shape

    model_height = (
        input_shape[2]
        if isinstance(input_shape[2], int)
        else args.fallback_size
    )

    model_width = (
        input_shape[3]
        if isinstance(input_shape[3], int)
        else args.fallback_size
    )

    print("Model input:", input_name, input_shape)
    print("Model output:", output_name, output_info.shape)
    print("Providers:", session.get_providers())
    print("Inference size:", model_width, "x", model_height)
    print("Postprocess:", "on" if args.postprocess else "off")
    print("Warmup frames excluded from stats:", args.warmup_frames)

    capture = cv2.VideoCapture(str(input_path))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    frame_count = int(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    total_to_process = frame_count if frame_count > 0 else None
    if args.max_frames is not None:
        total_to_process = (
            args.max_frames
            if total_to_process is None
            else min(total_to_process, args.max_frames)
        )

    inference_times = []
    total_times = []

    previous_probability_map = None  # used only if --temporal-alpha > 0

    progress = tqdm(
        total=total_to_process,
        desc="Benchmarking",
    )

    frame_index = -1

    while True:
        if args.max_frames is not None and frame_index + 1 >= args.max_frames:
            break

        ok, frame = capture.read()

        if not ok:
            break

        frame_index += 1
        frame_start = time.perf_counter()

        tensor = preprocess(
            frame,
            model_width,
            model_height,
        )

        inference_start = time.perf_counter()

        output = session.run(
            [output_name],
            {input_name: tensor},
        )[0]

        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000.0

        if output.ndim != 4:
            raise RuntimeError(
                f"Unexpected output shape: {output.shape}"
            )

        if output.shape[1] == 1:
            logits = output[0, 0]
            probability = sigmoid(logits)

            if args.postprocess:
                _, previous_probability_map = postprocess_floor_mask(
                    probability,
                    threshold=args.threshold,
                    open_kernel_size=args.open_kernel_size,
                    close_kernel_size=args.close_kernel_size,
                    min_component_area_fraction=args.min_component_area_fraction,
                    require_border_contact=args.require_border_contact,
                    previous_probability_map=previous_probability_map,
                    temporal_alpha=args.temporal_alpha,
                )
            # Deliberately NOT resizing the mask back up to original frame
            # size, compositing an overlay, or writing any output — those
            # steps are irrelevant to inference speed and this script only
            # measures timing.

        total_ms = (
            time.perf_counter() - frame_start
        ) * 1000.0

        if frame_index >= args.warmup_frames:
            inference_times.append(inference_ms)
            total_times.append(total_ms)

        if args.max_latency_warn_ms is not None and total_ms > args.max_latency_warn_ms:
            tqdm.write(
                f"  [frame {frame_index}] latency {total_ms:.1f} ms "
                f"exceeds {args.max_latency_warn_ms:.1f} ms budget"
            )

        progress.update(1)

    progress.close()
    capture.release()

    if not total_times:
        print("\nNo frames were timed (video shorter than --warmup-frames?).")
        return

    mean_inference = float(np.mean(inference_times))
    mean_total = float(np.mean(total_times))

    max_inference = float(np.max(inference_times))
    max_inference_frame = int(np.argmax(inference_times)) + args.warmup_frames

    max_total = float(np.max(total_times))
    max_total_frame = int(np.argmax(total_times)) + args.warmup_frames

    min_total = float(np.min(total_times))
    p95_total = float(np.percentile(total_times, 95))

    print()
    print(f"Frames timed: {len(total_times)} (of {frame_index + 1} processed, {args.warmup_frames} warm-up excluded)")
    print(f"Mean inference: {mean_inference:.2f} ms")
    print(f"Max inference:  {max_inference:.2f} ms (frame {max_inference_frame})")
    print(f"Mean end-to-end: {mean_total:.2f} ms")
    print(f"Min end-to-end:  {min_total:.2f} ms")
    print(f"P95 end-to-end:  {p95_total:.2f} ms")
    print(f"Max end-to-end:  {max_total:.2f} ms (frame {max_total_frame})")
    print(f"Effective rate (mean): {1000.0 / mean_total:.2f} Hz")
    print(f"Worst-case rate (max):  {1000.0 / max_total:.2f} Hz")


if __name__ == "__main__":
    main()
