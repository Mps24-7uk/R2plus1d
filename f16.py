# tools/convert_to_fp16.py
import argparse
import os
from pathlib import Path
import torch

def to_fp16_state_dict(state):
    fp16 = {}
    for k, v in state.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            fp16[k] = v.half()
        else:
            fp16[k] = v
    return fp16

def main():
    ap = argparse.ArgumentParser(description="Convert a PyTorch checkpoint to FP16.")
    ap.add_argument("--ckpt", required=True, help="Path to FP32 checkpoint (e.g., run/exp1/best_model.pth)")
    ap.add_argument("--out", default=None, help="Output path; defaults to <ckpt>_fp16.pth")
    args = ap.parse_args()

    print(f"[load] {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu")

    # Support either {"model_state_dict": ...} or a raw state_dict
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        rest = {k: v for k, v in ckpt.items() if k != "model_state_dict"}
    else:
        state, rest = ckpt, {}

    state_fp16 = to_fp16_state_dict(state)

    out_path = args.out or (Path(args.ckpt).with_suffix("").as_posix() + "_fp16.pth")
    torch.save({"model_state_dict": state_fp16, **rest}, out_path)

    def mb(p): return os.path.getsize(p) / (1024 * 1024)
    print(f"[save] {out_path}  ({mb(out_path):.2f} MB)")
    print(f"[info] original={mb(args.ckpt):.2f} MB  ->  fp16={mb(out_path):.2f} MB")

if __name__ == "__main__":
    main()
