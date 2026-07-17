# Writeup

## Architecture

**MobileNetV3-Large backbone + a custom U-Net-style decoder**, binary
(floor / not-floor) output, 384×384 input. The backbone stays
ImageNet-pretrained; a decoder is added on top with five skip connections
tapped from backbone stages at decreasing spatial resolution, mirroring
standard U-Net design.

Chosen over the alternatives listed in the assignment (Fast-SCNN,
BiSeNetV2, PP-LiteSeg) mainly for two reasons: MobileNetV3's ImageNet
pretraining gives the encoder a real head start on general visual features
before ever seeing a floor image, which matters a lot with a training set
in the low thousands of images rather than the tens of thousands typical
segmentation datasets use; and it exports cleanly to ONNX and OpenVINO IR
with no custom operators, which the suggested direction specifically calls
out as a practical constraint. A BiSeNetV2 (trained from scratch, no
pretrained weights) was also implemented and cross-validated as a
comparison point — it worked, but is not the model being submitted;
MobileNetV3-UNet was chosen as the primary result.

## Dataset

A mix of two sources, chosen to fit a one-week timeline rather than any
single source being ideal on its own:

1. **Public datasets** (subsets of ADE20K, SUN RGB-D, NYUv2)
   filtered for clear floor visibility, used for the initial
   cross-validated training. This gives broad scene diversity cheaply, at
   the cost of not matching the target deployment domain (an actual
   cleaning robot's camera) particularly closely.
2. **Stock photos and Hypersim synthetic renders**, hand-annotated
   (SAM-assisted manual annotation tool), added specifically to broaden
   visual diversity beyond the cross-validation dataset and target the
   false-positive failure mode described below. No real robot-camera
   footage was used for annotation/fine-tuning.

The public-dataset-first, hand-annotation-second approach was a deliberate
tradeoff: public data is fast to acquire and gives a real, defensible
cross-validated baseline number quickly; hand annotation is slow but lets
targeted fixes for specific observed failures (see Known Limitations)
rather than hoping a bigger generic dataset happens to cover them.

## Quantization / optimization plan

- **Export path:** PyTorch → ONNX (`torch.onnx.export`, static opset 17,
  forced legacy TorchScript-based exporter rather than the newer
  dynamo-based one, which was observed to behave less predictably on this
  architecture) → OpenVINO IR (`openvino.convert_model`).
- **Quantization:** post-training INT8 via NNCF, calibrated on a sample of
  real floor images (no labels needed for calibration). A test conversion
  produced a ~4x reduction in on-disk model size (26.3 MB FP32 → 6.7 MB
  INT8), consistent with expected INT8 quantization behavior.
- **Device selection:** automatic at runtime via
  `openvino.Core().available_devices` — uses NPU if present in that list,
  otherwise CPU. No manual device switch in the inference code path.

## Known limitations and failure cases

**Generalization beyond the fine-tuning distribution is not established.**
The fine-tuned checkpoint shows a strong same-split improvement (IoU 0.884
→ 0.959, precision 0.953 → 0.989, recall 0.925 → 0.969) on data drawn from
the same distribution as its ~190-image fine-tuning set. That comparison
does not by itself demonstrate improved or even unchanged behavior on
floor types visually distinct from the fine-tuning data — the fine-tuning
set, while deliberately built to include several different environment
types, is still small and does not cover every plausible deployment
scene. This is flagged as an open question rather than a claimed
strength: a same-split validation number is evidence the fine-tuning
procedure worked as intended on its own data, not evidence about
out-of-distribution robustness.

**The false-positive/false-negative loss weighting was not ablated in
isolation.** The reported fine-tuning improvement reflects the complete
procedure (new data + asymmetric loss weighting + partial-then-broader
unfreeze schedule) together. A neutral-weight ablation on the identical
data and schedule would be needed to attribute the improvement causally to
the loss weighting specifically, rather than to the additional data alone.

**INT8 quantization trades a real, measured amount of accuracy for
speed.** INT8 is ~30% faster than FP32 on mean latency (67.5 ms vs.
97.3 ms mean per-image on a 25-image test set), but drops IoU by 0.073
on the same held-out split used for the fine-tuning comparison (0.858
vs. 0.932), plausibly explained by the INT8 calibration set (164 images)
being smaller than NNCF's recommended minimum (300).

**NPU code path is implemented but untested on real NPU hardware.** No
NPU-equipped machine was available during development. The device-
detection and fallback-to-CPU logic itself was verified working (correctly
detects no NPU present and falls back), but the NPU execution path itself
has not been exercised.

**Cross-validated IoU (0.7956 ± 0.0066) has not been compared against an
unaugmented reference run.** The padding-safe resize and added geometric/
occlusion augmentation were applied together as a single change; whether
either piece individually helps, hurts, or is neutral relative to a
simpler transform pipeline is not established by this cross-validation
run alone.

**The fine-tuning source checkpoint used for this submission is the
highest-IoU fold from cross-validation** (fold 0, IoU 0.8020, the best of
the 5 folds). Earlier development used other checkpoints/fine-tuning runs
for iteration; the final reported numbers throughout this document are
from the fine-tune trained from fold 0.

## What's not yet done

- No NPU hardware benchmark — no NPU-equipped machine was available (see
  above).
- No formal out-of-distribution test set exists to quantify the
  generalization limitation described above — this would be the most
  valuable next addition to the evaluation setup.