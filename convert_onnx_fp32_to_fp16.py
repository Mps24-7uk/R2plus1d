# convert_onnx_fp32_to_fp16.py
import argparse
import onnx
from onnx import TensorProto

def convert_model_to_fp16(input_path, output_path):
    print(f"[INFO] Loading ONNX model: {input_path}")
    model = onnx.load(input_path)
    graph = model.graph

    # 1) Convert initializers (weights) from FP32 -> FP16
    converted_tensors = 0
    for tensor in graph.initializer:
        if tensor.data_type == TensorProto.FLOAT:  # FP32
            # Convert raw_data bytes to float32 array
            float_data = onnx.numpy_helper.to_array(tensor).astype("float16")

            # Make new tensor in FP16
            new_tensor = onnx.numpy_helper.from_array(float_data, name=tensor.name)
            tensor.CopyFrom(new_tensor)
            converted_tensors += 1

    print(f"[INFO] Converted {converted_tensors} initializers to FP16")

    # 2) Update tensor types in graph (inputs / outputs / value_info)
    def _convert_value_info_dtype(value_info):
        ttype = value_info.type.tensor_type
        if ttype.elem_type == TensorProto.FLOAT:
            ttype.elem_type = TensorProto.FLOAT16

    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        _convert_value_info_dtype(vi)

    # 3) (Optional) Update metadata
    if model.metadata_props is None:
        model.metadata_props = []
    model.producer_name = "fp32_to_fp16_converter"
    model.producer_version = "1.0"

    print(f"[INFO] Saving FP16 model to: {output_path}")
    onnx.save(model, output_path)
    print("[OK] Done.")

def main():
    parser = argparse.ArgumentParser(description="Convert FP32 ONNX model to FP16.")
    parser.add_argument("--input",  required=True, help="Path to FP32 ONNX model (e.g. best.onnx)")
    parser.add_argument("--output", required=True, help="Path to save FP16 ONNX model (e.g. best_fp16.onnx)")
    args = parser.parse_args()

    convert_model_to_fp16(args.input, args.output)

if __name__ == "__main__":
    main()



#python convert_onnx_fp32_to_fp16.py --input ./run/exp1/best.onnx --output ./run/exp1/best_fp16.onnx
