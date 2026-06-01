"""ONNX 量化检测工具函数"""

from typing import Optional
from onnx import ModelProto, NodeProto, TensorProto
from onnx import numpy_helper


QUANT_OPS = {
    "QuantizeLinear",
    "DequantizeLinear",
    "DynamicQuantizeLinear",
    "QLinearConv",
    "QLinearMatMul",
    "QLinearAdd",
    "QLinearMul",
    "QLinearDiv",
    "QLinearSub",
    "QLinearSigmoid",
    "QLinearRelu",
    "QLinearLeakyRelu",
    "QLinearPRelu",
    "QLinearAveragePool",
    "QLinearMaxPool",
}

# Whitelist of FP32 op sequences that map to quantized ops
# Only 2 core patterns: Conv+[BN]+[ReLU] -> QLinearConv and MatMul+[ReLU] -> QLinearMatMul
QUANTIZED_OP_PAIRS = {
    # Conv + [optional BN] + [optional ReLU] -> QLinearConv
    ("Conv",): "QLinearConv",
    ("Conv", "BatchNormalization"): "QLinearConv",
    ("Conv", "Relu"): "QLinearConv",
    ("Conv", "BatchNormalization", "Relu"): "QLinearConv",
    ("BatchNormalization",): "QLinearConv",  # BN can appear as separate fused node
    ("Relu",): "QLinearConv",
    # Linear + [optional ReLU] -> QLinearMatMul
    ("MatMul",): "QLinearMatMul",
    ("MatMul", "Relu"): "QLinearMatMul",
}


def is_quant_op(op_type: str) -> bool:
    return op_type in QUANT_OPS


def get_initializer_dtype(initializer: TensorProto) -> int:
    return initializer.data_type


def is_quantized_initializer(initializer: TensorProto) -> bool:
    dtype = initializer.data_type
    return dtype in {2, 3}


def get_initializer_tensor_name(initializer: TensorProto) -> str:
    return initializer.name


def get_node_tensors(node: NodeProto) -> set[str]:
    return set(node.input) | set(node.output)


def get_node_attributes(node: NodeProto) -> dict[str, any]:
    attr_map = {}
    for attr in node.attribute:
        attr_type = attr.type
        if attr_type == 1:
            attr_map[attr.name] = attr.f
        elif attr_type == 2:
            attr_map[attr.name] = attr.i
        elif attr_type == 6:
            attr_map[attr.name] = list(attr.floats)
        elif attr_type == 7:
            attr_map[attr.name] = list(attr.ints)
        elif attr_type == 4:
            attr_map[attr.name] = "Tensor"
        elif attr_type == 3:
            attr_map[attr.name] = list(attr.strings)
    return attr_map


def get_quantization_info(node: NodeProto) -> Optional[dict]:
    if node.op_type not in {"QuantizeLinear", "DequantizeLinear"}:
        return None

    attr_map = {}
    for attr in node.attribute:
        attr_map[attr.name] = attr

    info = {"op_type": node.op_type}

    if node.op_type == "QuantizeLinear":
        info["axis"] = attr_map.get("axis", None)
        info["block_size"] = attr_map.get("block_size", None)
        info["saturate"] = attr_map.get("saturate", None)
        info["output_dtype"] = attr_map.get("output_dtype", None)
    elif node.op_type == "DequantizeLinear":
        info["axis"] = attr_map.get("axis", None)
        info["block_size"] = attr_map.get("block_size", None)
        info["output_dtype"] = attr_map.get("output_dtype", None)

    return info


def compare_node_attributes(
    node_a: NodeProto, node_b: NodeProto
) -> dict[str, tuple[any, any]]:
    attrs_a = get_node_attributes(node_a)
    attrs_b = get_node_attributes(node_b)

    all_keys = set(attrs_a.keys()) | set(attrs_b.keys())
    changes = {}

    for key in all_keys:
        val_a = attrs_a.get(key)
        val_b = attrs_b.get(key)
        if val_a != val_b:
            changes[key] = (val_a, val_b)

    return changes


def has_quantization_in_model(model: ModelProto) -> bool:
    for node in model.graph.node:
        if is_quant_op(node.op_type):
            return True
    for init in model.graph.initializer:
        if is_quantized_initializer(init):
            return True
    return False


def count_quant_nodes(model: ModelProto) -> int:
    count = 0
    for node in model.graph.node:
        if is_quant_op(node.op_type):
            count += 1
    return count


def get_tensor_name_to_node_map(model: ModelProto) -> dict[str, NodeProto]:
    result = {}
    for node in model.graph.node:
        for tensor_name in node.output:
            if tensor_name:
                result[tensor_name] = node
    return result


def find_tensor_consumers(model: ModelProto, tensor_name: str) -> list[NodeProto]:
    consumers = []
    for node in model.graph.node:
        if tensor_name in node.input:
            consumers.append(node)
    return consumers


def find_tensor_producer(model: ModelProto, tensor_name: str) -> Optional[NodeProto]:
    for node in model.graph.node:
        if tensor_name in node.output:
            return node
    return None


def detect_quantization_pattern(fp32_sequence: tuple[str, ...]) -> Optional[str]:
    """Check if FP32 op sequence matches a known quantization fusion pattern.

    Args:
        fp32_sequence: Tuple of op types (e.g., ("Conv", "BatchNorm", "Relu"))

    Returns:
        Quantized op type if matched, None otherwise
    """
    return QUANTIZED_OP_PAIRS.get(fp32_sequence)


def get_all_fp32_sequences() -> list[tuple[str, ...]]:
    """Return all whitelisted FP32 op sequences for pattern matching."""
    return list(QUANTIZED_OP_PAIRS.keys())