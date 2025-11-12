# tools/export_onnx.py
import os
import argparse
import torch
import torch.nn as nn

from model import R2Plus1DModel  # your model class

def strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def infer_num_classes_from_state(state_dict, default=2):
    # Try common heads
    for key in ("backbone.fc.weight", "fc.weight", "classifier.5.weight"):
        if key in state_dict and isinstance(state_dict[key], torch.Tensor):
            return int(state_dict[key].shape[0])
    return default

def main():
    ap = argparse.ArgumentParser(description="Export pruned/FP16 R(2+1)D to ONNX.")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth) e.g. run/exp1/best_model_f16.pth")
    ap.add_argument("--batch-size", type=int, default=1, help="Dummy batch size")
    ap.add_argument("--num-frames", type=int, default=12, help="Temporal length T")
    ap.add_argument("--image-size", type=int, default=224, help="Square input size H=W")
    ap.add_argument("--opset", type=int, default=17, help="ONNX opset")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16",
                    help="Export graph weights/inputs as fp16 (CUDA recommended) or fp32 (CPU safe).")
    ap.add_argument("--out", default=None, help="Output .onnx path (default: <ckpt_dir>/best.onnx)")
    args = ap.parse_args()

    # Device logic: prefer CUDA for fp16, fallback to fp32 on CPU
    cuda_ok = torch.cuda.is_available()
    if args.dtype == "fp16" and not cuda_ok:
        print("[WARN] CUDA not available; FP16 export on CPU is often unsupported. Falling back to FP32 export.")
        args.dtype = "fp32"

    device = torch.device("cuda" if cuda_ok else "cpu")
    print(f"[INFO] Using device: {device}, export dtype: {args.dtype}")

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    out_path = args.out or os.path.join(out_dir, "best.onnx")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    state = strip_module_prefix(state)

    # Build model & load weights
    num_classes = infer_num_classes_from_state(state, default=2)
    print(f"[INFO] Inferred num_classes: {num_classes}")
    model = R2Plus1DModel(num_classes=num_classes, pretrained=False)

    # If weights are FP16, PyTorch will upcast on load where needed; export dtype set below.
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    # Set precision for export
    if args.dtype == "fp16":
        model.half()

    # Dummy input (B, C=3, T, H, W)
    b = args.batch_size
    c = 3
    t = args.num_frames
    h = w = args.image_size

    dummy = torch.randn(b, c, t, h, w, device=device)
    if args.dtype == "fp16":
        dummy = dummy.half()

    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = {
        "input":  {0: "batch", 2: "time"},  # variable batch & time
        "logits": {0: "batch"},
    }

    print(f"[INFO] Exporting to ONNX: {out_path}")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )
    print(f"[OK] ONNX saved: {out_path}")

if __name__ == "__main__":
    main()


# GPU available → export as FP16 (recommended for smallest graph + later TensorRT FP16)
# python tools/export_onnx.py --ckpt run/exp1/best_model_f16.pth --dtype fp16 --num-frames 12 --image-size 224 --opset 17

# CPU only → safely export as FP32 (fallback)
# python tools/export_onnx.py --ckpt run/exp1/best_model_f16.pth --dtype fp32 --num-frames 12 --image-size 224 --opset 17
