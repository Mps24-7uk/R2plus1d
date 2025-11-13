# tools/export_onnx.py
import os
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from model import R2Plus1DModel  # your video model


def strip_module_prefix(state_dict):
    """Remove a leading 'module.' from keys (from DataParallel)."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def infer_num_classes_from_state(state_dict, default=2):
    """
    Try to infer num_classes from the final Linear layer weight.
    Falls back to 'default' if nothing sensible is found.
    """
    # Common final-layer keys
    for key in ("backbone.fc.weight", "fc.weight", "classifier.5.weight"):
        v = state_dict.get(key, None)
        if isinstance(v, torch.Tensor) and v.ndim == 2:
            return int(v.shape[0])

    # Fallback: last 2D *.weight tensor (often the classifier head)
    candidates = [
        (k, v) for k, v in state_dict.items()
        if k.endswith(".weight") and isinstance(v, torch.Tensor) and v.ndim == 2
    ]
    if candidates:
        candidates.sort(key=lambda kv: kv[0])
        return int(candidates[-1][1].shape[0])

    return default


def main():
    parser = argparse.ArgumentParser(description="Export R2Plus1D model to ONNX (BS=1).")
    parser.add_argument("--ckpt", required=True,
                        help="Path to checkpoint (.pth) e.g. run/exp1/best_model_fp16.pth")
    parser.add_argument("--out", default=None,
                        help="Output ONNX path (default: <ckpt_dir>/best.onnx)")
    parser.add_argument("--num-frames", type=int, default=12,
                        help="Temporal length T used for dummy input")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Square input size H=W")
    parser.add_argument("--opset", type=int, default=17,
                        help="ONNX opset version")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                        help="Export graph in fp16 or fp32 (controls model + dummy dtype)")
    args = parser.parse_args()

    # Fixed batch size = 1, dynamic time dimension
    B = 1
    C = 3
    T = args.num_frames
    H = W = args.image_size

    # Use CUDA if available, but DO NOT silently change dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}, requested export dtype={args.dtype}")
    if args.dtype == "fp16" and device.type == "cpu":
        print("[WARN] Exporting fp16 ONNX on CPU – may fail for unsupported ops, "
              "but will NOT auto-fallback to fp32.")

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    out_path = Path(args.out) if args.out else ckpt_path.with_name("best.onnx")
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    state = strip_module_prefix(state)

    # Build model and load weights
    num_classes = infer_num_classes_from_state(state, default=2)
    print(f"[INFO] Inferred num_classes={num_classes}")
    model = R2Plus1DModel(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Set precision for export
    if args.dtype == "fp16":
        model.half()

    # Dummy input (B=1, C=3, T, H, W)
    dummy = torch.randn(B, C, T, H, W, device=device)
    if args.dtype == "fp16":
        dummy = dummy.half()

    # Names & dynamic axes: batch fixed, time dynamic
    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = {
        "input":  {2: "time"},  # only time is dynamic
        "logits": {}            # fixed batch=1
    }

    print(f"[INFO] Exporting to ONNX: {out_path}")
    torch.onnx.export(
        model,
        dummy,
        str(out_path),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"[OK] Saved ONNX: {out_path}")
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"[INFO] ONNX size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
