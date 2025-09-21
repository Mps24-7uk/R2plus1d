# single_folder_onnx_infer.py
import os, glob, argparse
from typing import Tuple, List
import numpy as np
from PIL import Image
import onnxruntime as ort

CLASS_NAMES = ["fall", "no_fall"]              # must match training/export
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

def pick_session(model_path: str, device: str = "auto") -> ort.InferenceSession:
    if device in ("auto", "cuda") and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        if device == "cuda":
            print("[WARN] CUDA EP unavailable; using CPU.")
        providers = ["CPUExecutionProvider"]
    sess = ort.InferenceSession(model_path, providers=providers)
    print(f"[INFO] Providers: {sess.get_providers()}")
    return sess

def load_first_12_frames(
    folder: str,
    image_size: Tuple[int, int] = (112, 112),
    num_frames: int = 12
) -> np.ndarray:
    """
    Loads the FIRST 12 frames from `folder`, preprocesses like training,
    and returns array of shape (1, C, T, H, W) float32.
    """
    paths = sorted(
        p for p in glob.glob(os.path.join(folder, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if len(paths) < num_frames:
        raise ValueError(f"Found {len(paths)} frames, need at least {num_frames} in {folder}")
    paths = paths[:num_frames]

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(image_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0          # (H,W,3), [0,1]
        arr = (arr - KINETICS_MEAN) / KINETICS_STD               # normalize
        arr = np.transpose(arr, (2, 0, 1))                       # (C,H,W)
        frames.append(arr)

    clip = np.stack(frames, axis=1)   # (C, T, H, W)
    clip = np.expand_dims(clip, 0)    # (1, C, T, H, W)
    return clip.astype(np.float32)

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def main():
    ap = argparse.ArgumentParser(description="Single-folder ONNX inference for R(2+1)D-18")
    ap.add_argument("--model", required=True, help="Path to best.onnx")
    ap.add_argument("--input", required=True, help="Folder with >=12 frames")
    ap.add_argument("--num-frames", type=int, default=12, help="Temporal length T")
    ap.add_argument("--image-size", type=int, default=112, help="Square resize (H=W)")
    ap.add_argument("--device", default="cuda", choices=["auto", "cuda", "cpu"], help="Execution device")
    args = ap.parse_args()

    sess = pick_session(args.model, device=args.device)
    clip = load_first_12_frames(
        args.input,
        image_size=(args.image_size, args.image_size),
        num_frames=args.num_frames
    )  # (1,C,T,H,W)

    input_name = sess.get_inputs()[0].name
    logits = sess.run(None, {input_name: clip})[0]   # (1, num_classes)
    probs = softmax(logits, axis=1)[0]
    pred = int(np.argmax(probs))
    print(f"{args.input} -> pred: {CLASS_NAMES[pred]}  "
          f"p(fall)={probs[0]:.4f}  p(no_fall)={probs[1]:.4f}  conf={probs[pred]:.4f}")

if __name__ == "__main__":
    main()