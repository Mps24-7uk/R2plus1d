# onnx_infer.py
import os, argparse, glob
from typing import List, Tuple
import numpy as np
from PIL import Image

import onnxruntime as ort

CLASS_NAMES = ["fall", "no_fall"]   # must match training/export
KINETICS_MEAN = np.array([0.43216, 0.394666, 0.37645], dtype=np.float32)
KINETICS_STD  = np.array([0.22803, 0.22145, 0.216989], dtype=np.float32)

def select_session(model_path: str, device: str = "auto") -> ort.InferenceSession:
    """
    Create an ONNXRuntime session. device: 'auto'|'cuda'|'cpu'
    """
    providers = []
    if device in ("auto", "cuda"):
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "cuda":
            print("[WARN] CUDA EP not available in onnxruntime. Falling back to CPU.")
            providers = ["CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(model_path, providers=providers)
    print(f"[INFO] Using providers: {sess.get_providers()}")
    return sess

def reflect_pad_paths(frame_files: List[str], target_len: int) -> List[str]:
    n = len(frame_files)
    if n >= target_len:
        return frame_files[:target_len]
    if n == 0:
        return frame_files
    if n == 1:
        return frame_files + [frame_files[0]] * (target_len - 1)
    out = frame_files[:]
    mirror_idx = list(range(n - 2, 0, -1))  # exclude endpoints
    while len(out) < target_len:
        for j in mirror_idx:
            out.append(frame_files[j])
            if len(out) >= target_len:
                break
    return out

def load_clip_numpy(
    seq_dir: str,
    num_frames: int = 12,
    image_size: Tuple[int, int] = (112, 112),
) -> np.ndarray:
    """
    Returns a numpy array shaped (C, T, H, W), float32, normalized with Kinetics stats.
    """
    paths = sorted(
        p for p in glob.glob(os.path.join(seq_dir, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not paths:
        raise FileNotFoundError(f"No frames found in: {seq_dir}")

    paths = reflect_pad_paths(paths, num_frames)[:num_frames]

    frames = []
    for p in paths:
        img = Image.open(p).convert("RGB").resize(image_size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0           # (H,W,3) in [0,1]
        arr = (arr - KINETICS_MEAN) / KINETICS_STD                # normalize
        arr = np.transpose(arr, (2, 0, 1))                        # (C,H,W)
        frames.append(arr)

    clip = np.stack(frames, axis=1)  # (C, T, H, W)
    return clip.astype(np.float32)

def find_sequence_dirs(path: str) -> List[str]:
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    has_frames = any(
        f.lower().endswith((".jpg", ".jpeg", ".png"))
        for f in os.listdir(path)
    )
    if has_frames:
        return [path]
    subdirs = sorted(
        os.path.join(path, d) for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )
    if not subdirs:
        raise FileNotFoundError(f"No subdirectories or frames under: {path}")
    return subdirs

def softmax(x: np.ndarray, axis=-1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def run_batch(session: ort.InferenceSession, batch: np.ndarray) -> np.ndarray:
    """
    batch: (B, C, T, H, W) -> returns logits (B, num_classes)
    """
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch})
    return outputs[0]

def prediction(frame_folder):
    ap = argparse.ArgumentParser(description="ONNX inference for R(2+1)D-18 best.onnx")
    ap.add_argument("--model", required=True, help="Path to best.onnx")
    ap.add_argument("--input", required=True, help="Dir with frames OR dir of many subfolders")
    ap.add_argument("--batch-size", type=int, default=4, help="Batch size for clips")
    ap.add_argument("--num-frames", type=int, default=12, help="Frames per clip (T)")
    ap.add_argument("--image-size", type=int, default=112, help="Square resize (H=W)")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"], help="Execution device")
    args = ap.parse_args()

    sess = select_session(args.model, device=args.device)
    seq_dirs = find_sequence_dirs(args.input)
    print(f"[INFO] Found {len(seq_dirs)} sequence(s).")

    B = args.batch_size
    H = W = args.image_size
    T = args.num_frames

    # Iterate in batches
    for start in range(0, len(seq_dirs), B):
        batch_dirs = seq_dirs[start:start+B]
        clips = []
        valid_idx = []
        for i, d in enumerate(batch_dirs):
            try:
                clip = load_clip_numpy(d, num_frames=T, image_size=(H, W))  # (C,T,H,W)
                clips.append(clip)
                valid_idx.append(i)
            except Exception as e:
                print(f"[ERROR] {d}: {e}")
        if not clips:
            continue

        batch = np.stack(clips, axis=0).astype(np.float32)  # (B,C,T,H,W)
        logits = run_batch(sess, batch)
        probs = softmax(logits, axis=1)
        preds = np.argmax(probs, axis=1)

        for i, d in zip(valid_idx, batch_dirs):
            p = probs[i]
            pred = int(preds[i])
            print(f"{d} -> pred: {CLASS_NAMES[pred]}  "
                  f"p(fall)={p[0]:.4f}  p(no_fall)={p[1]:.4f}  conf={p[pred]:.4f}")

# if __name__ == "__main__":
#     main()



#python onnx_infer.py --model ./run/exp1/best.onnx --input ./data/fall/vid01 --batch-size 1
#python onnx_infer.py --model ./run/exp1/best.onnx --input ./data/no_fall --batch-size 8