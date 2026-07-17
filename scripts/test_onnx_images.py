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

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def postprocess_floor_mask(
    probability_map: np.ndarray,
    threshold: float,
    open_kernel_size: int,
    close_kernel_size: int,
    min_component_area_fraction: float,
    require_border_contact: bool,
) -> np.ndarray:
    """
    Same cleanup logic as test_onnx_video.py, minus temporal smoothing
    (no previous-frame concept for independent still images).
    Returns clean_binary_mask uint8 {0,1}.
    """
    prob = probability_map.astype(np.float32)
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

    return clean


def preprocess(
    image_bgr: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return np.ascontiguousarray(image, dtype=np.float32)


def find_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        files = sorted(
            p for p in input_path.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        return files

    raise FileNotFoundError(f"Input path not found: {input_path}")


def build_contact_sheet(
    overlay_paths: list[Path],
    output_path: Path,
    columns: int,
    thumb_width: int,
) -> None:
    """
    Tiles all overlay images into one grid image for fast eyeballing across
    a whole folder at once, instead of opening each file individually.
    Each tile is labeled with its source filename.
    """
    if not overlay_paths:
        return

    thumbs = []
    for path in overlay_paths:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        aspect = img.shape[0] / img.shape[1]
        thumb_height = int(thumb_width * aspect)
        thumb = cv2.resize(img, (thumb_width, thumb_height), interpolation=cv2.INTER_AREA)

        label_height = 22
        labeled = np.zeros((thumb_height + label_height, thumb_width, 3), dtype=np.uint8)
        labeled[label_height:, :, :] = thumb

        label_text = path.stem
        if len(label_text) > 28:
            label_text = label_text[:25] + "..."
        cv2.putText(
            labeled, label_text, (4, 16),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA,
        )
        thumbs.append(labeled)

    if not thumbs:
        return

    tile_height = max(t.shape[0] for t in thumbs)
    tile_width = thumb_width
    rows = (len(thumbs) + columns - 1) // columns

    sheet = np.zeros((tile_height * rows, tile_width * columns, 3), dtype=np.uint8)

    for index, thumb in enumerate(thumbs):
        row = index // columns
        col = index % columns
        y0 = row * tile_height
        x0 = col * tile_width
        sheet[y0:y0 + thumb.shape[0], x0:x0 + thumb.shape[1]] = thumb

    cv2.imwrite(str(output_path), sheet)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run ONNX floor segmentation on one image or a folder of images. "
            "Saves a green-overlay version of each, plus a contact-sheet grid "
            "for quickly eyeballing quality across many images at once. "
            "Purely visual — no ground truth / metrics, since none is assumed "
            "to exist for these test photos."
        )
    )

    parser.add_argument("--model", required=True, help="Path to .onnx model")
    parser.add_argument("--input", required=True, help="Path to a single image OR a folder of images")
    parser.add_argument("--output-dir", required=True, help="Folder to save overlay images + contact sheet into")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--alpha", type=float, default=0.45, help="Overlay blend strength (0-1)")
    parser.add_argument(
        "--fallback-size", type=int, default=384,
        help="Used when the ONNX model has dynamic spatial dimensions.",
    )
    parser.add_argument(
        "--postprocess", action="store_true",
        help="Apply morphological cleanup + connected-component filtering to the mask.",
    )
    parser.add_argument("--open-kernel-size", type=int, default=5)
    parser.add_argument("--close-kernel-size", type=int, default=7)
    parser.add_argument("--min-component-area-fraction", type=float, default=0.0015)
    parser.add_argument(
        "--require-border-contact", action="store_true",
        help="Also drop floor blobs that don't touch the image border.",
    )
    parser.add_argument(
        "--contact-sheet", action="store_true", default=True,
        help="Build a single tiled grid image of all overlays (default: on). Use --no-contact-sheet to disable.",
    )
    parser.add_argument("--no-contact-sheet", dest="contact_sheet", action="store_false")
    parser.add_argument(
        "--contact-sheet-columns", type=int, default=4,
        help="Number of columns in the contact sheet grid (default: 4).",
    )
    parser.add_argument(
        "--contact-sheet-thumb-width", type=int, default=320,
        help="Width in pixels of each tile in the contact sheet (default: 320).",
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Stop after this many images (default: process all found).",
    )

    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output_dir).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    image_paths = find_input_images(input_path)

    if not image_paths:
        raise FileNotFoundError(
            f"No images found at {input_path} "
            f"(looked for extensions: {sorted(IMAGE_EXTENSIONS)})"
        )

    if args.max_images is not None:
        image_paths = image_paths[: args.max_images]

    output_dir.mkdir(parents=True, exist_ok=True)

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

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

    model_height = input_shape[2] if isinstance(input_shape[2], int) else args.fallback_size
    model_width = input_shape[3] if isinstance(input_shape[3], int) else args.fallback_size

    print("Model input:", input_name, input_shape)
    print("Model output:", output_name, output_info.shape)
    print("Providers:", session.get_providers())
    print("Inference size:", model_width, "x", model_height)
    print("Images found:", len(image_paths))
    print()

    inference_times = []
    total_times = []
    overlay_paths = []
    floor_fractions = []
    failed_paths = []
    successful_paths = []  # tracked separately so argmax indexing below
                            # stays correct even when some images fail to
                            # read (failed images are never appended to
                            # inference_times/total_times, so indexing back
                            # into the original image_paths list would be
                            # misaligned by however many failures preceded
                            # the max)

    for image_path in tqdm(image_paths, desc="Testing"):
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

        if image_bgr is None:
            failed_paths.append(image_path)
            continue

        original_height, original_width = image_bgr.shape[:2]
        successful_paths.append(image_path)

        total_start = time.perf_counter()

        tensor = preprocess(image_bgr, model_width, model_height)

        inference_start = time.perf_counter()
        output = session.run([output_name], {input_name: tensor})[0]
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        if output.ndim != 4:
            raise RuntimeError(f"Unexpected output shape: {output.shape}")

        if output.shape[1] == 1:
            logits = output[0, 0]
            probability = sigmoid(logits)

            if args.postprocess:
                small_mask = postprocess_floor_mask(
                    probability,
                    threshold=args.threshold,
                    open_kernel_size=args.open_kernel_size,
                    close_kernel_size=args.close_kernel_size,
                    min_component_area_fraction=args.min_component_area_fraction,
                    require_border_contact=args.require_border_contact,
                )
                small_mask = small_mask.astype(bool)
            else:
                small_mask = probability >= args.threshold
        else:
            small_mask = np.argmax(output[0], axis=0) == 1

        mask = cv2.resize(
            small_mask.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        floor_fraction = float(mask.mean())
        floor_fractions.append(floor_fraction)

        coloured = image_bgr.copy()
        coloured[mask] = (0, 255, 0)

        result = cv2.addWeighted(image_bgr, 1.0 - args.alpha, coloured, args.alpha, 0)

        total_ms = (time.perf_counter() - total_start) * 1000.0
        inference_times.append(inference_ms)
        total_times.append(total_ms)

        cv2.putText(
            result, f"Inference: {inference_ms:.1f} ms", (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            result, f"Floor: {floor_fraction * 100:.1f}%", (16, 54),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )

        overlay_path = output_dir / f"{image_path.stem}_overlay.png"
        cv2.imwrite(str(overlay_path), result)
        overlay_paths.append(overlay_path)

    if failed_paths:
        print(f"\nWARNING: {len(failed_paths)} image(s) failed to read:")
        for path in failed_paths[:10]:
            print(" ", path)

    if not total_times:
        print("\nNo images were successfully processed.")
        return

    if args.contact_sheet:
        contact_sheet_path = output_dir / "_contact_sheet.png"
        build_contact_sheet(
            overlay_paths, contact_sheet_path,
            columns=args.contact_sheet_columns,
            thumb_width=args.contact_sheet_thumb_width,
        )
        print(f"\nContact sheet: {contact_sheet_path}")

    mean_inference = float(np.mean(inference_times))
    max_inference = float(np.max(inference_times))
    max_inference_path = successful_paths[int(np.argmax(inference_times))]

    mean_total = float(np.mean(total_times))
    max_total = float(np.max(total_times))
    max_total_path = successful_paths[int(np.argmax(total_times))]

    mean_floor_fraction = float(np.mean(floor_fractions))

    print(f"\nImages processed: {len(total_times)}")
    print(f"Mean inference: {mean_inference:.2f} ms")
    print(f"Max inference:  {max_inference:.2f} ms ({max_inference_path.name})")
    print(f"Mean end-to-end: {mean_total:.2f} ms")
    print(f"Max end-to-end:  {max_total:.2f} ms ({max_total_path.name})")
    print(f"Mean predicted floor coverage: {mean_floor_fraction * 100:.1f}%")
    print(f"\nOverlays saved to: {output_dir}")


if __name__ == "__main__":
    main()
