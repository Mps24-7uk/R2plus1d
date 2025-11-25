# inferencing_trt_opencv.py
import os
import glob
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context

CLASS_NAMES = ["fall", "no_fall"]
IMG_EXTS = (".jpg", ".jpeg", ".png")

# 🔸 Grayscale stats (same as training/export)
KINETICS_MEAN = 0.401092
KINETICS_STD  = 0.222156

# --------------------- utils / preprocessing ---------------------
def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

def _reflect_pad_paths(paths, target_len):
    n = len(paths)
    if n >= target_len:
        return paths[:target_len]
    if n == 0:
        return paths
    if n == 1:
        return paths + [paths[0]] * (target_len - 1)
    out = paths[:]
    mirror_idx = list(range(n - 2, 0, -1))  # exclude endpoints
    while len(out) < target_len:
        for j in mirror_idx:
            out.append(paths[j])
            if len(out) >= target_len:
                break
    return out

def _list_images(path):
    return sorted(
        p for p in glob.glob(os.path.join(path, "*"))
        if p.lower().endswith(IMG_EXTS)
    )

def _find_sequence_dirs(path: str):
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    has_frames = any(f.lower().endswith(IMG_EXTS) for f in os.listdir(path))
    if has_frames:
        return [path]
    return sorted(
        os.path.join(path, d) for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    )

def _imread_unicode(p: str):
    """Robust imread (Unicode/long paths) using np.fromfile + cv2.imdecode (grayscale)."""
    try:
        data = np.fromfile(p, dtype=np.uint8)
        if data.size == 0:
            return None
        # 🔸 Read as GRAYSCALE
        return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)  # (H, W), uint8
    except Exception:
        return None

def load_clip_from_folder_cv2(
    folder,
    seq_len=12,
    height=224,
    width=224,
    mean=KINETICS_MEAN,
    std=KINETICS_STD
):
    """
    Load a sequence of frames from `folder` as a single clip tensor:
    Returns np.ndarray of shape (1, 1, T, H, W) for grayscale.
    """
    paths = _list_images(folder)
    if not paths:
        raise ValueError(f"{folder}: no image files found")

    # reflect-pad to T, then take first T
    paths = _reflect_pad_paths(paths, seq_len)[:seq_len]

    frames = []
    for p in paths:
        img_gray = _imread_unicode(p)  # (H, W), uint8
        if img_gray is None:
            raise ValueError(f"Failed to read image: {p}")

        # Resize to (H, W)
        img_gray = cv2.resize(
            img_gray,
            (width, height),
            interpolation=cv2.INTER_LINEAR
        )

        # Normalize: [0,1] then (x - mean)/std
        arr = img_gray.astype(np.float32) / 255.0  # (H, W)
        arr = (arr - mean) / std                   # (H, W)
        arr = arr[None, :, :]                      # (1, H, W)  -> C=1

        frames.append(arr)

    # Stack into (C=1, T, H, W)
    clip = np.stack(frames, axis=1)    # (1, T, H, W)
    clip = np.expand_dims(clip, axis=0)  # (1, 1, T, H, W)
    return clip.astype(np.float32)

# --------------------- TensorRT runner ---------------------
class TRTInference:
    def __init__(self, engine_path):
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("Failed to load engine")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create execution context")
        self.stream = cuda.Stream()

        self.use_binding_api = hasattr(self.engine, "num_bindings")
        self.use_iotensor_api = hasattr(self.engine, "num_io_tensors")
        if not (self.use_binding_api or self.use_iotensor_api):
            raise RuntimeError("Unsupported TensorRT API")

        self.d_buffers, self.h_buffers = {}, {}

        if self.use_binding_api:
            self.binding_names = [
                self.engine.get_binding_name(i)
                for i in range(self.engine.num_bindings)
            ]
            self.input_indices = [
                i for i in range(self.engine.num_bindings)
                if self.engine.binding_is_input(i)
            ]
            self.output_indices = [
                i for i in range(self.engine.num_bindings)
                if not self.engine.binding_is_input(i)
            ]
        else:
            self.io_names = [
                self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)
            ]
            self.input_names = [
                n for n in self.io_names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT
            ]
            self.output_names = [
                n for n in self.io_names
                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT
            ]

        ver = getattr(trt, "__version__", "unknown")
        print(f"[TRT] version: {ver} | API: {'bindings' if self.use_binding_api else 'io-tensor'}")

    def _dtype_np(self, td: trt.DataType):
        m = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF:  np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT8:  np.int8,
            trt.DataType.BOOL:  np.bool_,
        }
        return m[td]

    # TRT ≤ 8.x
    def _alloc_bindings(self):
        self.bindings = [None] * self.engine.num_bindings
        for i in range(self.engine.num_bindings):
            shape = tuple(self.context.get_binding_shape(i))
            if -1 in shape:
                raise RuntimeError(f"Dynamic dims unset for {self.binding_names[i]}: {shape}")
            dtype = self._dtype_np(self.engine.get_binding_dtype(i))
            n = int(np.prod(shape))
            self.h_buffers[i] = cuda.pagelocked_empty(n, dtype=dtype)
            self.d_buffers[i] = cuda.mem_alloc(n * np.dtype(dtype).itemsize)
            self.bindings[i] = int(self.d_buffers[i])

    def _infer_bindings(self, x: np.ndarray, in_idx: int):
        # x: (B, 1, T, H, W) float32
        self.context.set_binding_shape(in_idx, x.shape)
        self._alloc_bindings()

        expected = self._dtype_np(self.engine.get_binding_dtype(in_idx))
        if x.dtype != expected:
            x = x.astype(expected, copy=False)

        np.copyto(self.h_buffers[in_idx], x.ravel())
        cuda.memcpy_htod_async(self.d_buffers[in_idx], self.h_buffers[in_idx], self.stream)

        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v2 failed")

        outs = []
        for oi in self.output_indices:
            host = self.h_buffers[oi]
            cuda.memcpy_dtoh_async(host, self.d_buffers[oi], self.stream)
            shape = tuple(self.context.get_binding_shape(oi))
            dtype = self._dtype_np(self.engine.get_binding_dtype(oi))
            outs.append(host.reshape(shape).astype(dtype, copy=False))

        self.stream.synchronize()
        return outs

    # TRT ≥ 10.x
    def _alloc_iotensors(self):
        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if -1 in shape:
                raise RuntimeError(f"Dynamic dims unset for {name}: {shape}")
            dtype = self._dtype_np(self.engine.get_tensor_dtype(name))
            n = int(np.prod(shape))
            self.h_buffers[name] = cuda.pagelocked_empty(n, dtype=dtype)
            self.d_buffers[name] = cuda.mem_alloc(n * np.dtype(dtype).itemsize)
            self.context.set_tensor_address(name, int(self.d_buffers[name]))

    def _infer_iotensors(self, x: np.ndarray, in_name: str):
        self.context.set_input_shape(in_name, tuple(x.shape))
        self._alloc_iotensors()

        expected = self._dtype_np(self.engine.get_tensor_dtype(in_name))
        if x.dtype != expected:
            x = x.astype(expected, copy=False)

        np.copyto(self.h_buffers[in_name], x.ravel())
        cuda.memcpy_htod_async(self.d_buffers[in_name], self.h_buffers[in_name], self.stream)

        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v3 failed")

        outs = []
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.h_buffers[name], self.d_buffers[name], self.stream)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._dtype_np(self.engine.get_tensor_dtype(name))
            outs.append(self.h_buffers[name].reshape(shape).astype(dtype, copy=False))

        self.stream.synchronize()
        return outs

    def infer(self, x: np.ndarray):
        if self.use_binding_api:
            return self._infer_bindings(x, self.input_indices[0])
        else:
            return self._infer_iotensors(x, self.input_names[0])

# --------------------- main ---------------------
def main():
    ap = argparse.ArgumentParser(description="TensorRT inference (OpenCV preprocessing, grayscale)")
    ap.add_argument("--engine", required=True, help="Path to best.engine")
    ap.add_argument("--input",  required=True,
                    help="Dir of frames OR root dir with many clip folders")
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--height",  type=int, default=224)
    ap.add_argument("--width",   type=int, default=224)
    ap.add_argument("--save-npy", action="store_true",
                    help="Save first clip tensor as clip.npy for parity check")
    args = ap.parse_args()

    runner = TRTInference(args.engine)
    seq_dirs = _find_sequence_dirs(args.input)
    processed = skipped = 0

    for i, d in enumerate(seq_dirs):
        try:
            clip = load_clip_from_folder_cv2(
                d,
                seq_len=args.seq_len,
                height=args.height,
                width=args.width,
                mean=KINETICS_MEAN,
                std=KINETICS_STD,
            )
            if i == 0 and args.save_npy:
                np.save("clip.npy", clip)

            outputs = runner.infer(clip)
            logits = outputs[0]                  # (B, num_classes)
            probs = softmax(logits, axis=-1)[0]  # (num_classes,)
            pred = int(np.argmax(probs))
            p = probs
            print(
                f"{d} -> pred: {CLASS_NAMES[pred]}  "
                f"p(fall)={p[0]:.4f}  p(no_fall)={p[1]:.4f}  conf={p[pred]:.4f}"
            )
            processed += 1
        except Exception as e:
            print(f"{d} -> SKIPPED: {e}")
            skipped += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Total: {len(seq_dirs)}")


if __name__ == "__main__":
    main()
