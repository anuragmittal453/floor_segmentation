#!/usr/bin/env python3
"""
scripts/export_to_onnx.py

Exports the custom MobileNetV3UNet floor-segmentation checkpoint (the
model trained/fine-tuned in the CV and fine-tuning notebooks — a
MobileNetV3-Large backbone with a 5-skip-connection UNet decoder, binary
1-channel output) to ONNX format for downstream OpenVINO INT8 conversion.

This is NOT the same architecture as torchvision's built-in
lraspp_mobilenet_v3_large (2-class LR-ASPP head) used by an earlier,
separate export script — using that script against a MobileNetV3UNet
checkpoint will fail to load or silently load the wrong weights. This
script constructs the exact custom architecture from the CV/fine-tuning
notebooks and verifies the loaded state dict matches before exporting.

Usage:
    python scripts/export_to_onnx.py \\
        --checkpoint /mnt/c/Users/mitta/Downloads/checkpoints/mobilenet_unet/best.pt \\
        --output artifacts/floor_seg_model_mobilenet_unet.onnx \\
        --input-size 384

The output ONNX model has a FIXED input size (matches --input-size, default
384x384 to match training resolution) — not dynamic. This matches what
test_onnx_video.py already expects (it resizes each frame to the model's
reported input size before inference) and is the safer choice for the
downstream OpenVINO INT8 calibration step.
"""

import argparse
import sys
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models
except ImportError:
    sys.exit("Missing dependency. Install with:\n    pip install torch torchvision\n")


# =============================================================================
# Model definition — must exactly match the architecture used in training /
# fine-tuning (see the CV notebook's MobileNetV3UNet class). Copied verbatim
# rather than imported, since the notebooks aren't importable modules.
# =============================================================================

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, skip_channels, output_channels):
        super().__init__()
        self.fusion = ConvBNAct(input_channels + skip_channels, output_channels)

    def forward(self, decoder_feature, skip_feature):
        decoder_feature = F.interpolate(
            decoder_feature, size=skip_feature.shape[-2:], mode="bilinear", align_corners=False,
        )
        return self.fusion(torch.cat([decoder_feature, skip_feature], dim=1))


class MobileNetV3UNet(nn.Module):
    def __init__(self, image_size, pretrained=False):
        super().__init__()

        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights).features

        probe = torch.zeros(1, 3, image_size, image_size)
        stage_information = []
        backbone.eval()

        with torch.no_grad():
            for index, layer in enumerate(backbone):
                probe = layer(probe)
                stage_information.append(
                    (index, int(probe.shape[1]), int(probe.shape[2]), int(probe.shape[3]))
                )

        last_stage_per_size = {}
        for item in stage_information:
            last_stage_per_size[(item[2], item[3])] = item

        selected = sorted(last_stage_per_size.values(), key=lambda item: item[2], reverse=True)
        selected = [item for item in selected if item[2] < image_size][:5]
        assert len(selected) == 5, stage_information

        self.stage_ids = [item[0] for item in selected]
        channels = [item[1] for item in selected]

        self.backbone = backbone

        c1, c2, c3, c4, c5 = channels

        self.center = ConvBNAct(c5, 256)
        self.decoder4 = DecoderBlock(256, c4, 160)
        self.decoder3 = DecoderBlock(160, c3, 96)
        self.decoder2 = DecoderBlock(96, c2, 64)
        self.decoder1 = DecoderBlock(64, c1, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

        print("Selected MobileNet stages:", list(zip(self.stage_ids, channels)))

    def forward(self, image):
        input_size = image.shape[-2:]
        features = []
        feature = image
        wanted = set(self.stage_ids)

        for index, layer in enumerate(self.backbone):
            feature = layer(feature)
            if index in wanted:
                features.append(feature)

        assert len(features) == 5
        feature1, feature2, feature3, feature4, feature5 = features

        decoded = self.center(feature5)
        decoded = self.decoder4(decoded, feature4)
        decoded = self.decoder3(decoded, feature3)
        decoded = self.decoder2(decoded, feature2)
        decoded = self.decoder1(decoded, feature1)
        logits = self.head(decoded)

        return F.interpolate(logits, size=input_size, mode="bilinear", align_corners=False)


# =============================================================================
# Export logic
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt (or any checkpoint with a 'model' key or raw state dict)")
    parser.add_argument("--output", required=True, help="Output .onnx path")
    parser.add_argument("--input-size", type=int, default=384, help="Fixed square input resolution (default: 384, matches training)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser()
    output_path = Path(args.output).expanduser()

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    if isinstance(checkpoint, dict):
        for key in ("fold", "epoch", "metrics"):
            if key in checkpoint:
                print(f"  source {key}: {checkpoint[key]}")

    print(f"\nConstructing MobileNetV3UNet architecture at input_size={args.input_size}...")
    # pretrained=False: we're loading real trained weights next, no need to
    # download ImageNet init weights first (also avoids the corrupted-cache
    # EOFError failure mode entirely).
    model = MobileNetV3UNet(image_size=args.input_size, pretrained=False)

    print("\nLoading state dict into architecture...")
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print(f"  WARNING: {len(missing_keys)} missing keys (present in architecture, absent in checkpoint):")
        for key in missing_keys[:10]:
            print(f"    {key}")
        if len(missing_keys) > 10:
            print(f"    ... and {len(missing_keys) - 10} more")

    if unexpected_keys:
        print(f"  WARNING: {len(unexpected_keys)} unexpected keys (present in checkpoint, absent in architecture):")
        for key in unexpected_keys[:10]:
            print(f"    {key}")
        if len(unexpected_keys) > 10:
            print(f"    ... and {len(unexpected_keys) - 10} more")

    if not missing_keys and not unexpected_keys:
        print("  All keys matched exactly — checkpoint architecture confirmed correct.")
    else:
        print(
            "\n  ACTION NEEDED: key mismatches usually mean this checkpoint was NOT trained\n"
            "  with this exact MobileNetV3UNet class (e.g. it's from a different\n"
            "  architecture, or IMAGE_SIZE differs from --input-size and changed which\n"
            "  backbone stages got selected as skip connections). Do not trust this\n"
            "  export if there are mismatches — fix the mismatch first."
        )
        response = input("\nContinue exporting anyway? [y/N] ").strip().lower()
        if response != "y":
            print("Aborted.")
            sys.exit(1)

    model.eval()

    print(f"\nExporting to ONNX: {output_path}")
    dummy_input = torch.randn(1, 3, args.input_size, args.input_size)

    # dynamo=False forces the stable TorchScript-based exporter. The newer
    # dynamo-based exporter (torch's default in recent versions) depends on
    # onnxscript and has been observed to silently downgrade the opset
    # version instead of failing loudly when something isn't supported —
    # the legacy exporter is more predictable for a model like this
    # (plain convs, BatchNorm, bilinear interpolate — nothing exotic that
    # needs the newer exporter's capabilities).
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["image"],
        output_names=["logits"],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    print("Export complete.")

    # -------------------------------------------------------------------
    # Verification: run both PyTorch and ONNX Runtime on the same input,
    # confirm outputs match within numerical tolerance. This catches export
    # bugs (wrong opset behavior, silently dropped ops, etc.) instead of
    # discovering them later when the deployed model behaves differently
    # than testing suggested.
    # -------------------------------------------------------------------
    print("\nVerifying exported model against PyTorch output...")
    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime not installed — skipping verification.")
        print("  Install with: pip install onnxruntime")
        print(f"\nSaved: {output_path}")
        return

    with torch.no_grad():
        torch_output = model(dummy_input).numpy()

    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    onnx_output = session.run(None, {"image": dummy_input.numpy()})[0]

    import numpy as np
    max_abs_diff = float(np.max(np.abs(torch_output - onnx_output)))
    matches = np.allclose(torch_output, onnx_output, atol=1e-4, rtol=1e-3)

    print(f"  Max absolute difference (PyTorch vs ONNX): {max_abs_diff:.2e}")
    if matches:
        print("  PASS: outputs match within tolerance.")
    else:
        print("  WARNING: outputs differ more than expected — inspect before deploying.")

    print(f"\nModel input: image, shape (1, 3, {args.input_size}, {args.input_size})")
    print(f"Model output: logits, shape (1, 1, {args.input_size}, {args.input_size})")
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
