# inferencing_trt.py
# TensorRT inference for best.engine with ONNX-matching preprocessing
# - Sequence: reflect-pad to T frames
# - Resize: 224x224, PIL bilinear
# - Color: RGB
# - Scale: /255
# - Normalize: Kinetics-400 stats (default)
# - API: auto-detects TRT 8.x (bindings + execute_async_v2) vs TRT 10.x (IO-tensor + execute_async_v3)

import os
import glob
import argparse
import numpy as np
from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context

CLASS_NAMES = ["fall", "no_fall"]  # adjust if needed
IMG_EXTS = (".jpg", ".jpeg", ".png")

# --------------------- Pre/Post ---------------------
KINETICS_MEAN = (0.43216, 0.394666, 0.37645)
KINETICS_STD  = (0.22803, 0.22145, 0.216989)

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
    return sorted(p for p in glob.glob(os.path.join(path, "*"))
                  if p.lower().endswith(IMG_EXTS))

def _find_sequence_dirs(path: str):
    """If path directly contains frames, treat it as one sequence.
       Otherwise, return subdirectories (each is a sequence)."""
    if not os.path.isdir(path):
        raise NotADirectoryError(f"{path} is not a directory.")
    has_frames = any(f.lower().endswith(IMG_EXTS) for f in os.listdir(path))
    if has_frames:
        return [path]
    return sorted(os.path.join(path, d) for d in os.listdir(path)
                  if os.path.isdir(os.path.join(path, d)))

def load_clip_from_folder(folder, seq_len=12, height=224, width=224,
                          mean=KINETICS_MEAN, std=KINETICS_STD):
    paths = _list_images(folder)
    if not paths:
        raise ValueError(f"{folder}: no image files found")
    # reflect-pad to seq_len, then take first T in sorted order
    paths = _reflect_pad_paths(paths, seq_len)[:seq_len]

    frames = []
    for p in paths:
        # ONNX path: PIL -> RGB -> bilinear resize -> /255 -> (x-mean)/std
        img = Image.open(p).convert("RGB").resize((width, height), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))               # (C,H,W)
        frames.append(arr)

    clip = np.stack(frames, axis=1)          # (C,T,H,W)
    clip = np.expand_dims(clip, axis=0)      # (1,C,T,H,W)
    return clip.astype(np.float32)

# --------------------- TensorRT Runner ---------------------
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

        # API detection
        self.use_binding_api = hasattr(self.engine, "num_bindings")
        self.use_iotensor_api = hasattr(self.engine, "num_io_tensors")
        if not (self.use_binding_api or self.use_iotensor_api):
            raise RuntimeError("Unsupported TensorRT Python API (no bindings/IO-tensor).")

        # Buffers
        self.d_buffers = {}
        self.h_buffers = {}

        if self.use_binding_api:
            self.binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            self.input_indices = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
            self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        else:
            self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names = [n for n in self.io_names
                                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in self.io_names
                                 if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        ver = getattr(trt, "__version__", "unknown")
        api = "bindings" if self.use_binding_api else "io-tensor"
        print(f"[TRT] version: {ver} | API: {api}")

    def _dtype_np(self, trt_dtype: trt.DataType):
        m = {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF:  np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT8:  np.int8,
            trt.DataType.BOOL:  np.bool_,
        }
        if trt_dtype not in m:
            raise TypeError(f"Unsupported dtype: {trt_dtype}")
        return m[trt_dtype]

    # -------- Bindings path (TRT ≤ 8.x) --------
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

    def _infer_bindings(self, input_array: np.ndarray, input_index: int = 0):
        self.context.set_binding_shape(input_index, input_array.shape)
        self._alloc_bindings()

        expected = self._dtype_np(self.engine.get_binding_dtype(input_index))
        if input_array.dtype != expected:
            input_array = input_array.astype(expected, copy=False)

        # H2D
        np.copyto(self.h_buffers[input_index], input_array.ravel())
        cuda.memcpy_htod_async(self.d_buffers[input_index], self.h_buffers[input_index], self.stream)

        # Exec
        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v2 failed")

        # D2H outputs
        outputs = []
        for oi in self.output_indices:
            out_host = self.h_buffers[oi]
            cuda.memcpy_dtoh_async(out_host, self.d_buffers[oi], self.stream)
            shape = tuple(self.context.get_binding_shape(oi))
            dtype = self._dtype_np(self.engine.get_binding_dtype(oi))
            outputs.append(out_host.reshape(shape).astype(dtype, copy=False))

        self.stream.synchronize()
        return outputs

    # -------- IO-tensor path (TRT ≥ 10.x) --------
    def _alloc_iotensors(self):
        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if -1 in shape:
                raise RuntimeError(f"Dynamic dims unset for tensor {name}: {shape}")
            dtype = self._dtype_np(self.engine.get_tensor_dtype(name))
            n = int(np.prod(shape))
            self.h_buffers[name] = cuda.pagelocked_empty(n, dtype=dtype)
            self.d_buffers[name] = cuda.mem_alloc(n * np.dtype(dtype).itemsize)
            self.context.set_tensor_address(name, int(self.d_buffers[name]))

    def _infer_iotensors(self, input_array: np.ndarray, input_name: str):
        self.context.set_input_shape(input_name, tuple(input_array.shape))
        self._alloc_iotensors()

        expected = self._dtype_np(self.engine.get_tensor_dtype(input_name))
        if input_array.dtype != expected:
            input_array = input_array.astype(expected, copy=False)

        # H2D
        np.copyto(self.h_buffers[input_name], input_array.ravel())
        cuda.memcpy_htod_async(self.d_buffers[input_name], self.h_buffers[input_name], self.stream)

        # Exec
        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v3 failed")

        # D2H outputs
        outputs = []
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.h_buffers[name], self.d_buffers[name], self.stream)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._dtype_np(self.engine.get_tensor_dtype(name))
            outputs.append(self.h_buffers[name].reshape(shape).astype(dtype, copy=False))

        self.stream.synchronize()
        return outputs

    def infer(self, input_array: np.ndarray):
        if self.use_binding_api:
            return self._infer_bindings(input_array, input_index=self.input_indices[0])
        else:
            return self._infer_iotensors(input_array, input_name=self.input_names[0])

# --------------------- Main ---------------------
def main():
    ap = argparse.ArgumentParser(description="TensorRT inference (ONNX-parity preprocessing)")
    ap.add_argument("--engine", required=True, help="Path to best.engine")
    ap.add_argument("--input",  required=True, help="Directory of frames OR root directory of many clip folders")
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--height",  type=int, default=224)
    ap.add_argument("--width",   type=int, default=224)
    ap.add_argument("--mean",    type=float, nargs=3, default=KINETICS_MEAN)
    ap.add_argument("--std",     type=float, nargs=3, default=KINETICS_STD)
    ap.add_argument("--save-npy", action="store_true", help="Save first clip tensor as clip.npy for parity checks")
    args = ap.parse_args()

    runner = TRTInference(args.engine)

    # Build list of sequences to evaluate
    seq_dirs = _find_sequence_dirs(args.input)
    processed, skipped = 0, 0

    for idx, d in enumerate(seq_dirs):
        try:
            clip = load_clip_from_folder(
                d,
                seq_len=args.seq_len,
                height=args.height,
                width=args.width,
                mean=tuple(args.mean),
                std=tuple(args.std),
            )
            if idx == 0 and args.save_npy:
                np.save("clip.npy", clip)

            outputs = runner.infer(clip)
            logits = outputs[0]                  # (B, num_classes)
            probs = softmax(logits, axis=-1)[0]  # (num_classes,)
            pred = int(np.argmax(probs))
            p = probs
            print(f"{d} -> pred: {CLASS_NAMES[pred]}  "
                  f"p(fall)={p[0]:.4f}  p(no_fall)={p[1]:.4f}  conf={p[pred]:.4f}")
            processed += 1
        except Exception as e:
            print(f"{d} -> SKIPPED: {e}")
            skipped += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Total: {len(seq_dirs)}")

if __name__ == "__main__":
    main()
