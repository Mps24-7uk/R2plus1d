# infer_trt_engine_folders_v2.py
import os
import glob
import argparse
import numpy as np
import cv2
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # initializes CUDA context

CLASS_NAMES = ["fall", "no_fall"]  # index 0 -> fall, index 1 -> no_fall

# ---------- Pre/Post ----------
def load_clip_from_folder(folder, seq_len=12, height=224, width=224,
                          mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    paths = sorted(glob.glob(os.path.join(folder, "*")))
    if len(paths) < seq_len:
        raise ValueError(f"{folder}: need at least {seq_len} frames, found {len(paths)}")
    frames = []
    for p in paths[:seq_len]:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        frames.append(img)
    clip = np.stack(frames, axis=0)                 # (T,H,W,C)
    clip = (clip - mean) / std
    clip = np.transpose(clip, (3, 0, 1, 2))         # (C,T,H,W)
    clip = np.expand_dims(clip, axis=0)             # (1,C,T,H,W)
    return clip  # float32

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# ---------- TensorRT ----------
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

        # Detect API flavor
        self.use_binding_api = hasattr(self.engine, "num_bindings")
        self.use_iotensor_api = hasattr(self.engine, "num_io_tensors")

        if not (self.use_binding_api or self.use_iotensor_api):
            raise RuntimeError("Unsupported TensorRT Python API (no bindings and no IO-tensor interfaces).")

        # Buffers
        self.d_buffers = {}   # name_or_index -> device mem
        self.h_buffers = {}   # name_or_index -> host pinned mem

        if self.use_binding_api:
            # Classic bindings
            self.binding_names = [self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            self.input_indices = [i for i in range(self.engine.num_bindings) if self.engine.binding_is_input(i)]
            self.output_indices = [i for i in range(self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        else:
            # IO-tensor API (TRT ≥ 10)
            self.io_names = [self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.input_names = [n for n in self.io_names
                                if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            self.output_names = [n for n in self.io_names
                                 if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]

        ver = getattr(trt, "__version__", "unknown")
        mode = "bindings" if self.use_binding_api else "io-tensor"
        print(f"[TRT] version: {ver} | using {mode} API")

    # dtype helpers
    def _dtype_np(self, trt_dtype: trt.DataType):
        return {
            trt.DataType.FLOAT: np.float32,
            trt.DataType.HALF:  np.float16,
            trt.DataType.INT32: np.int32,
            trt.DataType.INT8:  np.int8,
            trt.DataType.BOOL:  np.bool_,
        }[trt_dtype]

    # ----- Bindings path (TRT ≤ 8.x) -----
    def _alloc_bindings(self):
        self.bindings = [None] * self.engine.num_bindings  # raw device ptrs
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
        # Set dynamic shape
        self.context.set_binding_shape(input_index, input_array.shape)
        # Allocate according to current shapes
        self._alloc_bindings()

        # Cast input as needed
        expected = self._dtype_np(self.engine.get_binding_dtype(input_index))
        if input_array.dtype != expected:
            input_array = input_array.astype(expected, copy=False)

        # HtoD input
        np.copyto(self.h_buffers[input_index], input_array.ravel())
        cuda.memcpy_htod_async(self.d_buffers[input_index], self.h_buffers[input_index], self.stream)

        # Exec (v2)
        ok = self.context.execute_async_v2(self.bindings, self.stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v2 failed")

        # DtoH outputs
        outputs = []
        for oi in self.output_indices:
            out_host = self.h_buffers[oi]
            cuda.memcpy_dtoh_async(out_host, self.d_buffers[oi], self.stream)
            shape = tuple(self.context.get_binding_shape(oi))
            dtype = self._dtype_np(self.engine.get_binding_dtype(oi))
            outputs.append(out_host.reshape(shape).astype(dtype, copy=False))

        self.stream.synchronize()
        return outputs

    # ----- IO-tensor path (TRT ≥ 10.x) -----
    def _alloc_iotensors(self):
        # Allocate for every input & output tensor
        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            if -1 in shape:
                raise RuntimeError(f"Dynamic dims unset for tensor {name}: {shape}")
            dtype = self._dtype_np(self.engine.get_tensor_dtype(name))
            n = int(np.prod(shape))
            self.h_buffers[name] = cuda.pagelocked_empty(n, dtype=dtype)
            self.d_buffers[name] = cuda.mem_alloc(n * np.dtype(dtype).itemsize)
            # Give TensorRT the device pointer for this tensor
            self.context.set_tensor_address(name, int(self.d_buffers[name]))

    def _infer_iotensors(self, input_array: np.ndarray, input_name: str):
        # Tell TRT the runtime input shape
        self.context.set_input_shape(input_name, tuple(input_array.shape))

        # After shapes are known for all I/O, allocate and bind pointers
        self._alloc_iotensors()

        # Cast input to expected dtype
        expected = self._dtype_np(self.engine.get_tensor_dtype(input_name))
        if input_array.dtype != expected:
            input_array = input_array.astype(expected, copy=False)

        # HtoD input
        np.copyto(self.h_buffers[input_name], input_array.ravel())
        cuda.memcpy_htod_async(self.d_buffers[input_name], self.h_buffers[input_name], self.stream)

        # Exec (v3)
        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("execute_async_v3 failed")

        # DtoH outputs
        outputs = []
        for name in self.output_names:
            cuda.memcpy_dtoh_async(self.h_buffers[name], self.d_buffers[name], self.stream)
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._dtype_np(self.engine.get_tensor_dtype(name))
            outputs.append(self.h_buffers[name].reshape(shape).astype(dtype, copy=False))

        self.stream.synchronize()
        return outputs

    # ----- Public infer -----
    def infer(self, input_array: np.ndarray):
        """
        input_array: shape (B, C, T, H, W), dtype float32/float16
        returns: list of output arrays
        """
        if self.use_binding_api:
            # assume first binding is input
            return self._infer_bindings(input_array, input_index=self.input_indices[0])
        else:
            # use the first input tensor name
            return self._infer_iotensors(input_array, input_name=self.input_names[0])

# ---------- CLI / Runner ----------
def main():
    ap = argparse.ArgumentParser(description="Run best.engine on all subfolders")
    ap.add_argument("--engine", required=True, help="Path to best.engine")
    ap.add_argument("--input", default="D:/fall/", help="Root folder containing subfolders (each with frames)")
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--mean", type=float, nargs=3, default=(0.485, 0.456, 0.406))
    ap.add_argument("--std", type=float, nargs=3, default=(0.229, 0.224, 0.225))
    args = ap.parse_args()

    runner = TRTInference(args.engine)

    if not os.path.isdir(args.root):
        raise NotADirectoryError(f"Root path not found: {args.root}")

    subfolders = [f.path for f in os.scandir(args.root) if f.is_dir()]
    subfolders.sort()

    processed = 0
    skipped = 0

    for d in subfolders:
        try:
            clip = load_clip_from_folder(
                d, seq_len=args.seq_len, height=args.height, width=args.width,
                mean=tuple(args.mean), std=tuple(args.std)
            )
            outputs = runner.infer(clip)
            logits = outputs[0]  # assume (B, num_classes)
            probs = softmax(logits, axis=-1)[0]
            pred = int(np.argmax(probs))
            p = probs  # alias for printing
            print(f"{d} -> pred: {CLASS_NAMES[pred]}  "
                  f"p(fall)={p[0]:.4f}  p(no_fall)={p[1]:.4f}  conf={p[pred]:.4f}")
            processed += 1
        except Exception as e:
            print(f"{d} -> SKIPPED: {e}")
            skipped += 1

    print(f"\nDone. Processed: {processed}, Skipped: {skipped}, Total: {len(subfolders)}")

if __name__ == "__main__":
    main()
