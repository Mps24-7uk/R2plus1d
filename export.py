# export_onnx.py
import os
import argparse
import torch

# import your model class from model.py in the same folder
from model import R2Plus1DModel


def resolve_paths(input_path: str):
    """
    If input_path is a folder, expect run/expX/best_model.pth inside it.
    If input_path is a .pth file, use it directly and save ONNX next to it.
    Returns (ckpt_path, onnx_path).
    """
    if os.path.isdir(input_path):
        ckpt_path = os.path.join(input_path, "best_model.pth")
        onnx_path = os.path.join(input_path, "best.onnx")
    else:
        if not input_path.endswith(".pth"):
            raise ValueError("If input_path is a file, it must be a .pth checkpoint.")
        ckpt_path = input_path
        onnx_path = os.path.join(os.path.dirname(input_path), "best.onnx")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    return ckpt_path, onnx_path


def load_model(ckpt_path: str, device: torch.device) -> torch.nn.Module:
    """
    Instantiate R2Plus1DModel and load weights from checkpoint.
    Handles both raw state_dict and dict with 'model_state_dict'.
    """
    model = R2Plus1DModel(num_classes=2, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location="cpu")

    state_dict = ckpt.get("model_state_dict", ckpt)  # support raw state_dict or wrapped
    # strict=False to be forgiving if extra keys exist
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Export R2Plus1DModel to ONNX")
    parser.add_argument(
        "--input_path",
        required=True,
        help="Path to experiment folder (e.g. run/exp1) OR a .pth file."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Dummy batch size for the exported graph (and for validation during export)."
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=12,
        help="Temporal length (T). Use the same T as training (default 12)."
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=112,
        help="Input spatial size (H=W). Use 112 if you trained with 112x112."
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (recommended 13–17)."
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path, onnx_path = resolve_paths(args.input_path)

    # Build and load model
    model = load_model(ckpt_path, device)

    # Create a dummy input: (B, C, T, H, W)
    dummy = torch.randn(
        args.batch_size, 3, args.num_frames, args.image_size, args.image_size,
        device=device
    )

    # Export
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {"0": "batch"},   # make batch dimension dynamic
            "output": {"0": "batch"}
        },
    )

    print(f"[OK] Exported ONNX to: {onnx_path}")
    print(f"      Device used: {device}")
    print(f"      Shapes -> input: (B, 3, {args.num_frames}, {args.image_size}, {args.image_size})")


if __name__ == "__main__":
    main()
