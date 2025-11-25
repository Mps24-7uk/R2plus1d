# export_trt.py  (previously convert_onnx_to_engine.py)
import os
import argparse
import tensorrt as trt


def build_engine(
    onnx_path: str,
    engine_path: str,
    precision: str = "fp16",   # "fp32" or "fp16"
    min_batch: int = 1,
    opt_batch: int = 2,
    max_batch: int = 4,
    channels: int = 1,         # 🔸 default 1 for grayscale
    seq_len: int = 12,
    height: int = 224,
    width: int = 224,
    workspace_gb: float = 2.0
):
    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")

    with trt.Builder(logger) as builder, \
         builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, logger) as parser:

        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")

        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("Failed to parse ONNX. Errors:")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                raise RuntimeError("ONNX parse failed")

        config = builder.create_builder_config()

        # Workspace limit (TRT >= 8.6) or legacy (<= 8.5)
        workspace_bytes = int(workspace_gb * (1 << 30))
        try:
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)
        except Exception:
            # Older TensorRT
            config.max_workspace_size = workspace_bytes  # type: ignore[attr-defined]

        # Precision selection
        wanted_fp16 = (precision.lower() == "fp16")
        using_fp16 = False
        if wanted_fp16:
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                using_fp16 = True
            else:
                logger.log(trt.Logger.WARNING, "FP16 requested but not supported; falling back to FP32.")

        # Dynamic profile for (B, C, T, H, W)
        if network.num_inputs < 1:
            raise RuntimeError("Network has no inputs.")
        input_tensor = network.get_input(0)
        input_name = input_tensor.name

        # 🔸 Make sure C matches your ONNX (1 for grayscale)
        min_shape = (min_batch, channels, seq_len, height, width)
        opt_shape = (opt_batch, channels, seq_len, height, width)
        max_shape = (max_batch, channels, seq_len, height, width)

        profile = builder.create_optimization_profile()
        profile.set_shape(input_name, min=min_shape, opt=opt_shape, max=max_shape)
        config.add_optimization_profile(profile)

        # Build engine
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            raise RuntimeError("Engine build returned None.")

        os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)

        print(f"[OK] Engine written to: {engine_path}")
        print(f"[INFO] Precision: {'FP16' if using_fp16 else 'FP32'} (requested: {precision.upper()})")
        print(f"[INFO] Profile min/opt/max: {min_shape} / {opt_shape} / {max_shape}")
        print(f"[INFO] Workspace: {workspace_gb} GB")


def main():
    ap = argparse.ArgumentParser(description="Convert ONNX to TensorRT engine (grayscale-ready)")
    ap.add_argument("--onnx", required=True, help="Path to best.onnx")
    ap.add_argument("--engine", default="best.engine", help="Output engine path")
    ap.add_argument("--precision", choices=["fp32", "fp16"], default="fp16",
                    help="Build precision")
    ap.add_argument("--min-batch", type=int, default=1)
    ap.add_argument("--opt-batch", type=int, default=2)
    ap.add_argument("--max-batch", type=int, default=4)
    # 🔸 default 1 channel for grayscale R2+1D
    ap.add_argument("--channels", type=int, default=1,
                    help="Number of input channels (1 for grayscale, 3 for RGB)")
    ap.add_argument("--seq-len", type=int, default=12)
    ap.add_argument("--height", type=int, default=224)
    ap.add_argument("--width", type=int, default=224)
    ap.add_argument("--workspace-gb", type=float, default=2.0)
    args = ap.parse_args()

    build_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        precision=args.precision,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        channels=args.channels,
        seq_len=args.seq_len,
        height=args.height,
        width=args.width,
        workspace_gb=args.workspace_gb
    )


if __name__ == "__main__":
    main()





#python export_trt.py --onnx best.onnx --engine best.engine --precision fp16 --min-batch 1 --opt-batch 2 --max-batch 4
  # --channels defaults to 1, so you can omit it
