# Benchmark Report

## Hardware

- **CPU:** consumer laptop CPU, Intel Core i7-9750H.
  No NPU present on this machine (pre-dates Intel's Meteor Lake / Core
  Ultra generation, which is the first with an integrated NPU).
- **NPU:** not available on the test machine. The automatic detection
  logic in `scripts/infer_openvino.py` (`openvino.Core().available_devices`)
  was confirmed working — it correctly falls back to CPU when no NPU is
  present — but the NPU code path itself has not been exercised on real
  NPU hardware. This is a known, disclosed gap (see `writeup.md`).

## Segmentation quality (IoU)

Measured with 5-fold stratified group cross-validation on a mixed indoor
floor dataset (subsets of ADE20K, SUN RGB-D, NYUv2, filtered for
floor visibility). Mean ± standard deviation across 5 folds:

| Model | IoU | Dice | Precision | Recall |
|---|---|---|---|---|
| MobileNetV3-UNet | 0.7956 ± 0.0066 | 0.8862 | 0.8876 | 0.8848 |

### Fine-tuning on hand-annotated data

The cross-validated checkpoint was then fine-tuned on a small
(~190-image) hand-annotated set combining Hypersim synthetic renders and
stock photos, using a loss that penalizes false
positives (predicting floor where there is none) more heavily than false
negatives. This targeted a specific observed failure mode: the deployed
model was labeling parts of the robot's own chassis as floor in test
footage. Trained for a 40-epoch schedule; the best checkpoint by
validation IoU was at epoch 37.

Same-validation-split comparison (source checkpoint vs. fine-tuned, both
evaluated on the same held-out 15% split of the fine-tuning set):

| | IoU | Precision | Recall |
|---|---|---|---|
| Source checkpoint | 0.884 | 0.953 | 0.925 |
| Fine-tuned | 0.959 | 0.989 | 0.969 |

IoU improved by +0.074, precision by +0.036, and recall by +0.045 on
this split, consistent with the false-positive-biased loss improving all
three metrics together rather than trading recall away for precision.

**This comparison measures the complete fine-tuning procedure on a small,
narrow validation set drawn from the same distribution as the fine-tuning
data.** It is not a controlled ablation of the loss weighting in isolation,
and it is not evidence of improved (or unchanged) generalization to floor
types outside the fine-tuning distribution — see `writeup.md` for the full
discussion of this limitation.

## OpenVINO IR inference latency: FP32 vs. INT8

Measured with `scripts/infer_openvino.py` on the fine-tuned checkpoint
above, same CPU as above (no NPU present — automatic device detection
correctly falls back to CPU), converted to OpenVINO IR at both FP32 and
INT8 precision.

On a 25-image folder of stock photos (out-of-distribution relative to
the training/fine-tuning data):

| Precision | Mean (ms) | Mean Hz | Worst-case (ms) | Worst-case Hz |
|---|---|---|---|---|
| FP32 | 97.3 | 10.28 | 156.1 | 6.41 |
| INT8 | 67.5 | 14.82 | 96.9 | 10.32 |

**Both precisions clear the 2 Hz / 500 ms-per-frame target with
substantial margin**, including on the single worst-case frame — over 3x
the required rate even in the slower (FP32) configuration. INT8 is
~30% faster on mean latency and ~38% faster on worst-case latency than
FP32 on this test set. Model size also dropped as expected: 26.3 MB
(FP32) → 6.7 MB (INT8) on disk.

**INT8 quantization measurably reduces segmentation quality.** Evaluated
with `scripts/evaluate_openvino_iou.py` on the identical 24-image
held-out split used for the fine-tuning comparison above:

| Precision | IoU | Dice | Precision (metric) | Recall |
|---|---|---|---|---|
| FP32 | 0.9315 | 0.9646 | 0.9724 | 0.9569 |
| INT8 | 0.8583 | 0.9237 | 0.9483 | 0.9003 |

INT8 drops IoU by 0.073 relative to FP32 on this held-out split, plausibly explained by the INT8 calibration set (164 images) being smaller than NNCF's recommended minimum (300).

## Video inference latency (INT8, CPU)

Measured with `scripts/infer_openvino.py`, INT8 OpenVINO IR, same CPU as
above, on a 667-frame real video (with morphological postprocessing
enabled, no temporal smoothing):

| Metric | Value |
|---|---|
| Frames processed | 667 |
| Mean end-to-end latency | 41.49 ms |
| Mean effective rate | 24.10 Hz |
| Worst-case (max) latency | 117.81 ms |
| Worst-case rate | 8.49 Hz |

**Clears the 2 Hz / 500 ms-per-frame target with very large margin** —
over 11x the required rate on average, and still over 4x the required
rate on the single worst frame. This is consistent with (and slightly
faster than) the image-folder INT8 result above, as expected: consecutive
video frames are a fixed resolution and benefit from warm caches, while
the stock-photo folder had per-image resolution/decode variability.

## Not yet measured

- **NPU inference latency** — no NPU-equipped hardware was available for
  testing (see Hardware section above).
- **A larger INT8 calibration set** (300+ images, as NNCF recommends,
  rather than the 164 used here) has not yet been tried. Given the
  measured IoU drop is real and non-trivial (0.073), and concentrated on
  already-hard samples, this is now the most actionable next step to
  actually close the gap rather than just describe it.