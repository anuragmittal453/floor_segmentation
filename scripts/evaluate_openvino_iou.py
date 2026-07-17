#!/usr/bin/env python3
"""
scripts/evaluate_openvino_iou.py

Computes IoU / Dice / precision / recall for an OpenVINO IR model against
a labeled held-out set, so quantization's effect on accuracy (not just
speed) can be measured directly — not just visually judged.

Reproduces the EXACT same held-out validation split used in
02_finetune_hand_annotated.ipynb (same SEED=42, same VAL_FRACTION=0.15,
same deterministic np.random.default_rng shuffle), so the resulting IoU
numbers are directly comparable to the fine-tuning notebook's reported
0.959 IoU for the PyTorch checkpoint.

Usage:
    # FP32
    python scripts/evaluate_openvino_iou.py \\
        --model artifacts/openvino_model_fp32_5/floor_seg_model_5.xml \\
        --data-root data/floor_seg_data

    # INT8
    python scripts/evaluate_openvino_iou.py \\
        --model artifacts/openvino_model_int8_5/floor_seg_model_5.xml \\
        --data-root data/floor_seg_data

Run both and compare the printed IoU to see quantization's real,
measured accuracy impact (not just the visual impression from looking at
overlay images).
"""

import argparse
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

# Must match 02_finetune_hand_annotated.ipynb exactly, so the held-out
# split reproduced here is the identical set of images.
SEED = 42
VAL_FRACTION = 0.15


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def preprocess(image_bgr: np.ndarray, width: int, height: int) -> np.ndarray:
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32) / 255.0
    image = (image - IMAGENET_MEAN) / IMAGENET_STD
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return np.ascontiguousarray(image, dtype=np.float32)


def build_held_out_split(data_root: Path) -> list[dict]:
    """
    Reproduces 02_finetune_hand_annotated.ipynb's exact validation split
    logic: match each image to its mask + ignore mask, shuffle
    deterministically with SEED, take the last VAL_FRACTION as "val".
    """
    image_dir = data_root / "images"
    mask_dir = data_root / "masks"
    ignore_dir = data_root / "ignore_masks"

    for name, path in [("images", image_dir), ("masks", mask_dir), ("ignore_masks", ignore_dir)]:
        assert path.is_dir(), f"{name} directory not found: {path}"

    image_files = sorted(image_dir.glob("*.jpg"))
    assert image_files, f"No JPEG images found in: {image_dir}"

    rows = []
    for image_path in image_files:
        sample_id = image_path.stem
        mask_path = mask_dir / f"{sample_id}.png"
        ignore_path = ignore_dir / f"{sample_id}.png"

        if not mask_path.is_file() or not ignore_path.is_file():
            continue

        rows.append({
            "sample_id": sample_id,
            "image_path": image_path,
            "mask_path": mask_path,
            "ignore_path": ignore_path,
        })

    assert rows, "No complete image/mask/ignore-mask triplets found."

    rng = np.random.default_rng(SEED)
    shuffled_indices = rng.permutation(len(rows))

    val_count = max(1, int(len(rows) * VAL_FRACTION))
    val_indices = set(shuffled_indices[:val_count].tolist())

    val_rows = [rows[i] for i in sorted(val_indices)]
    return val_rows


def evaluate(compiled_model, output_layer, model_width, model_height, val_rows, threshold=0.5):
    intersection = 0
    union = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    valid_pixel_total = 0

    per_sample_iou = []

    for row in val_rows:
        image_bgr = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(row["mask_path"]), cv2.IMREAD_GRAYSCALE)
        ignore = cv2.imread(str(row["ignore_path"]), cv2.IMREAD_GRAYSCALE)

        if image_bgr is None or mask is None or ignore is None:
            print(f"  SKIP (unreadable): {row['sample_id']}")
            continue

        original_height, original_width = image_bgr.shape[:2]

        tensor = preprocess(image_bgr, model_width, model_height)
        result = compiled_model([tensor])
        output = result[output_layer]

        probability = sigmoid(output[0, 0])
        small_prediction = probability >= threshold

        prediction = cv2.resize(
            small_prediction.astype(np.uint8),
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

        ground_truth = mask > 127
        valid = ignore <= 127  # exclude ignored pixels, same as training/eval

        prediction = prediction & valid
        ground_truth = ground_truth & valid

        sample_intersection = int((prediction & ground_truth).sum())
        sample_union = int((prediction | ground_truth).sum())
        sample_tp = sample_intersection
        sample_fp = int((prediction & ~ground_truth).sum())
        sample_fn = int((~prediction & ground_truth).sum())

        intersection += sample_intersection
        union += sample_union
        true_positive += sample_tp
        false_positive += sample_fp
        false_negative += sample_fn
        valid_pixel_total += int(valid.sum())

        sample_iou = sample_intersection / max(sample_union, 1)
        per_sample_iou.append((row["sample_id"], sample_iou))

    iou = intersection / max(union, 1)
    dice = 2 * true_positive / max(2 * true_positive + false_positive + false_negative, 1)
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    return {
        "iou": iou,
        "dice": dice,
        "precision": precision,
        "recall": recall,
        "valid_pixels": valid_pixel_total,
        "num_samples": len(per_sample_iou),
        "per_sample_iou": per_sample_iou,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="Path to OpenVINO IR .xml file")
    parser.add_argument("--data-root", required=True, help="Path to the floor_seg_data directory (images/masks/ignore_masks)")
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()

    model_path = Path(args.model).expanduser()
    data_root = Path(args.data_root).expanduser()

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    val_rows = build_held_out_split(data_root)
    print(f"Held-out validation set: {len(val_rows)} samples "
          f"(SEED={SEED}, VAL_FRACTION={VAL_FRACTION}, same split as fine-tuning)")

    core = ov.Core()
    available = core.available_devices
    device = "NPU" if any(d.startswith("NPU") for d in available) else "CPU"
    print(f"Available devices: {available} -> using {device}")

    model = core.read_model(str(model_path))
    compiled_model = core.compile_model(model, device)
    output_layer = compiled_model.output(0)

    input_shape = model.inputs[0].get_partial_shape()
    model_height = int(input_shape[2].get_length()) if input_shape[2].is_static else 384
    model_width = int(input_shape[3].get_length()) if input_shape[3].is_static else 384

    print(f"Model: {model_path.name} ({model_width}x{model_height})")
    print()

    metrics = evaluate(compiled_model, output_layer, model_width, model_height, val_rows, args.threshold)

    print(f"Evaluated: {metrics['num_samples']} / {len(val_rows)} samples")
    print()
    print(f"IoU:       {metrics['iou']:.4f}")
    print(f"Dice:      {metrics['dice']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")

    worst = sorted(metrics["per_sample_iou"], key=lambda item: item[1])[:5]
    print("\nLowest-IoU samples (worth a visual check):")
    for sample_id, sample_iou in worst:
        print(f"  {sample_id}: {sample_iou:.4f}")


if __name__ == "__main__":
    main()
