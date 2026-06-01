import pytest
from pathlib import Path
import tempfile
import onnx
from onnx import helper, TensorProto

from app.services.diff_service import DiffService
from app.services.onnx_utils import QUANTIZED_OP_PAIRS, detect_quantization_pattern


def create_simple_onnx_model(name: str, nodes: list, inputs=None, outputs=None) -> Path:
    if inputs is None:
        inputs = [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 32, 32])]
    if outputs is None:
        outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 32, 32, 32])]
    graph = helper.make_graph(
        nodes=nodes,
        name=name,
        inputs=inputs,
        outputs=outputs,
        initializer=[],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    path = Path(tempfile.gettempdir()) / f"{name}.onnx"
    onnx.save(model, str(path))
    return path


class TestQuantizedOpPairs:
    """Test QUANTIZED_OP_PAIRS whitelist and detect_quantization_pattern function"""

    def test_whitelist_has_correct_entries(self):
        """Verify the whitelist contains expected entries"""
        assert ("Conv",) in QUANTIZED_OP_PAIRS
        assert ("Conv", "BatchNorm", "Relu") in QUANTIZED_OP_PAIRS
        assert ("MatMul",) in QUANTIZED_OP_PAIRS
        assert ("MatMul", "Relu") in QUANTIZED_OP_PAIRS

    def test_detect_conv_bn_relu_fusion(self):
        """Conv + BN + ReLU should map to QLinearConv"""
        result = detect_quantization_pattern(("Conv", "BatchNorm", "Relu"))
        assert result == "QLinearConv"

    def test_detect_conv_relu_fusion(self):
        """Conv + ReLU should map to QLinearConv"""
        result = detect_quantization_pattern(("Conv", "Relu"))
        assert result == "QLinearConv"

    def test_detect_conv_only(self):
        """Conv alone should map to QLinearConv"""
        result = detect_quantization_pattern(("Conv",))
        assert result == "QLinearConv"

    def test_detect_matmul_relu_fusion(self):
        """MatMul + ReLU should map to QLinearMatMul"""
        result = detect_quantization_pattern(("MatMul", "Relu"))
        assert result == "QLinearMatMul"

    def test_detect_matmul_only(self):
        """MatMul alone should map to QLinearMatMul"""
        result = detect_quantization_pattern(("MatMul",))
        assert result == "QLinearMatMul"

    def test_detect_unknown_pattern(self):
        """Unknown patterns should return None (graceful fallback)"""
        result = detect_quantization_pattern(("Unknown", "Op"))
        assert result is None


class TestConservativeMode:
    """Test conservative (exact/tensor) matching"""

    def test_compare_identical_models(self):
        service = DiffService()
        node = helper.make_node("Relu", ["x"], ["y"], name="test_node")
        model_path = create_simple_onnx_model("test_model", [node])

        diff_id, result = service.compare_models(model_path, model_path, match_mode="conservative")

        assert diff_id is not None
        assert result.summary.total_nodes_a == result.summary.total_nodes_b
        assert result.summary.added_count == 0
        assert result.summary.removed_count == 0

    def test_compare_different_op_types(self):
        service = DiffService()

        node_a = helper.make_node("Relu", ["x"], ["y"], name="node_a")
        node_b = helper.make_node("Sigmoid", ["x"], ["y"], name="node_b")

        model_a_path = create_simple_onnx_model("model_a", [node_a])
        model_b_path = create_simple_onnx_model("model_b", [node_b])

        diff_id, result = service.compare_models(model_a_path, model_b_path, match_mode="conservative")

        assert result.summary.modified_count >= 1


class TestAggressiveMode:
    """Test aggressive mode with fusion pattern detection"""

    def test_aggressive_mode_exists(self):
        """Verify aggressive mode is accepted"""
        service = DiffService()
        node = helper.make_node("Relu", ["x"], ["y"], name="test_node")
        model_path = create_simple_onnx_model("test_model", [node])

        # Should not raise an error
        diff_id, result = service.compare_models(model_path, model_path, match_mode="aggressive")
        assert diff_id is not None


class TestFusionInfo:
    """Test that fusion_info is properly included in results"""

    def test_fusion_info_structure(self):
        """Verify FusionInfoResponse can be created"""
        from app.models.schemas import FusionInfoResponse, DiffNodeResponse

        fusion = FusionInfoResponse(
            is_fusion=True,
            fp32_components=["conv1", "bn1", "relu1"],
            confidence=0.85,
        )

        node = DiffNodeResponse(
            id="conv1",
            name="conv1",
            op_type="QLinearConv",
            diff_type="modified",
            details="Fused from: conv1 -> bn1 -> relu1 (confidence: 0.85)",
            fusion_info=fusion,
        )

        assert node.fusion_info is not None
        assert node.fusion_info.is_fusion is True
        assert node.fusion_info.fp32_components == ["conv1", "bn1", "relu1"]
        assert node.fusion_info.confidence == 0.85