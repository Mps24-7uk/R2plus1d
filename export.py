# export_onnx.py
import os
import argparse
import torch
import torch.nn as nn

# import your model class (from your repo)
from model import R2Plus1DModel


def strip_module_prefix(state_dict):
    """Remove 'module.' prefix from DataParallel checkpoints if present."""
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def infer_num_classes_from_state(state_dict, default=2):
    """
    Try to infer num_classes from the final FC weight if present.
    Falls back to 'default' if not found.
    """
    for key in ("backbone.fc.weight", "fc.weight"):
        if key in state_dict and isinstance(state_dict[key], torch.Tensor):
            return int(state_dict[key].shape[0])
    return default


def main():
    parser = argparse.ArgumentParser(description="Export R(2+1)D-18 .pth to ONNX.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint .pth")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size for dummy input")
    parser.add_argument("--num-frames", type=int, default=12, help="Temporal length T (default: 12)")
    parser.add_argument("--image-size", type=int, default=112, help="Square input size H=W (default: 112)")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (default: 17)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    # Output path = same directory as checkpoint, named best.onnx (as requested)
    out_dir = os.path.dirname(os.path.abspath(args.ckpt))
    out_path = os.path.join(out_dir, "best.onnx")

    # Load checkpoint
    print(f"[INFO] Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    state_dict = strip_module_prefix(state_dict)

    # Build model and load weights
    num_classes = infer_num_classes_from_state(state_dict, default=2)
    print(f"[INFO] Inferred num_classes: {num_classes}")
    model = R2Plus1DModel(num_classes=num_classes, pretrained=False).to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Dummy input: (B, C, T, H, W)
    b = args.batch_size
    c = 3
    t = args.num_frames
    h = w = args.image_size
    dummy = torch.randn(b, c, t, h, w, device=device)

    # Export
    input_names = ["input"]
    output_names = ["logits"]
    dynamic_axes = {
        "input":  {0: "batch", 2: "time"},  # allow variable batch and time lengths
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
    print(f"[OK] ONNX model saved at: {out_path}")


if __name__ == "__main__":
    main()



# python export_onnx.py --ckpt ./run/exp1/best_model.pth --batch-size 1 --num-frames 12 --image-size 112 --opset 17
