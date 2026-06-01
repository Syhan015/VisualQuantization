import numpy as np
from pathlib import Path
from typing import Optional
import onnx
from onnx import ModelProto
from onnx import numpy_helper

from app.core.exceptions import ONNXParseError
from app.models.schemas import WeightAnalysisResponse


class WeightService:
    def analyze_weight(
        self, model_path: Path, node_name: str
    ) -> Optional[WeightAnalysisResponse]:
        try:
            model = onnx.load(str(model_path))
        except Exception as e:
            raise ONNXParseError(f"Failed to parse ONNX model: {e}")

        initializer_map = {init.name: init for init in model.graph.initializer}

        if node_name not in initializer_map:
            return None

        init = initializer_map[node_name]
        weights = numpy_helper.to_array(init).flatten()

        distribution = self._compute_distribution(weights, bins=50)

        return WeightAnalysisResponse(
            node_name=node_name,
            cosine_similarity=1.0,
            l2_error=0.0,
            mae=0.0,
            mean_diff=0.0,
            std_diff=0.0,
            distribution_a=distribution,
            distribution_b=distribution,
        )

    def compare_weights(
        self,
        model_a_path: Path,
        model_b_path: Path,
        node_name: str,
        model_b_node_name: Optional[str] = None,
    ) -> Optional[WeightAnalysisResponse]:
        try:
            model_a = onnx.load(str(model_a_path))
            model_b = onnx.load(str(model_b_path))
        except Exception as e:
            raise ONNXParseError(f"Failed to parse ONNX model: {e}")

        # Extract FP32 weights from model A
        weights_a = self._extract_fp32_weight(model_a, node_name)
        if weights_a is None:
            return None

        # Extract and dequantize weights from model B
        target_name = model_b_node_name or node_name
        weights_b = self._extract_int8_weight_and_dequantize(model_b, target_name, node_name)
        if weights_b is None:
            return None

        # Ensure shape match
        if weights_a.shape != weights_b.shape:
            return None

        # Compute all 5 metrics
        cosine_sim = self._cosine_similarity(weights_a, weights_b)
        l2_error = float(np.linalg.norm(weights_a - weights_b))
        mae = float(np.mean(np.abs(weights_a - weights_b)))
        mean_diff = float(np.mean(weights_a - weights_b))
        std_diff = float(np.std(weights_a) - np.std(weights_b))

        # Compute distributions with shared bins and return edges
        bins = 50
        all_weights = np.concatenate([weights_a, weights_b])
        bin_edges = np.histogram_bin_edges(all_weights, bins=bins)
        dist_a, _ = np.histogram(weights_a, bins=bin_edges)
        dist_b, _ = np.histogram(weights_b, bins=bin_edges)
        # Normalize to probability
        total_a = dist_a.sum()
        total_b = dist_b.sum()
        dist_a = (dist_a / total_a).tolist() if total_a > 0 else [0.0] * bins
        dist_b = (dist_b / total_b).tolist() if total_b > 0 else [0.0] * bins

        return WeightAnalysisResponse(
            node_name=node_name or model_b_node_name or "unknown",
            cosine_similarity=cosine_sim,
            l2_error=l2_error,
            mae=mae,
            mean_diff=mean_diff,
            std_diff=std_diff,
            distribution_a=dist_a,
            distribution_b=dist_b,
            bin_edges=bin_edges.tolist(),
        )

    def _extract_fp32_weight(self, model: ModelProto, node_name: str) -> Optional[np.ndarray]:
        """Extract FP32 weight for a node by node name.

        For a Conv node with name='conv1', the weight initializer is typically 'conv1_w'.
        """
        if not node_name:
            return None

        # Find the node by name
        node = self._find_node_by_name(model, node_name)
        if node is None:
            return None

        # For Conv: inputs are [input, weight, bias] - weight is usually second input
        if node.op_type == 'Conv':
            if len(node.input) >= 2 and node.input[1]:
                init = self._get_initializer(model, node.input[1])
                if init is not None:
                    return numpy_helper.to_array(init).flatten()

        # Fallback: try node_name + '_w' as weight name
        weight_name = node_name + '_w'
        init = self._get_initializer(model, weight_name)
        if init is not None:
            return numpy_helper.to_array(init).flatten()

        # Fallback: try finding any initializer that could be related
        for inp in node.input:
            if inp:
                init = self._get_initializer(model, inp)
                if init is not None:
                    return numpy_helper.to_array(init).flatten()

        return None

    def _extract_int8_weight_and_dequantize(
        self, model: ModelProto, node_name: str, fallback_name: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """Extract INT8 weight and dequantize to FP32.

        For QLinearConv: inputs are x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp
        Weight is input 3, scale is input 4, zero_point is input 5.
        """
        # Try to find node by name first
        node = self._find_node_by_name(model, node_name) if node_name else None

        if node is None and fallback_name:
            node = self._find_node_by_name(model, fallback_name)

        # If still not found and node_name was empty, try first quantized node
        if node is None and not node_name:
            node = self._find_first_quantized_node(model)

        if node is None:
            return None

        # All supported quantized ops - extended from QUANT_OPS in onnx_utils.py
        quant_ops = {
            "QLinearConv",
            "QLinearMatMul",
            "ConvInteger",
            "MatMulInteger",
            "QuantizeLinear",
            "DequantizeLinear",
            "DynamicQuantizeLinear",
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

        if node.op_type not in quant_ops:
            # Not quantized, try direct weight extraction
            return self._extract_fp32_weight(model, node_name)

        # For QLinearConv and ConvInteger:
        # QLinearConv inputs: [x, w, b, x_scale, x_zp, w_scale, w_zp, y_scale, y_zp]
        # Weight is input 1, w_scale is input 5, w_zp is input 6
        if node.op_type in {"QLinearConv", "ConvInteger"}:
            if len(node.input) < 9:
                return None

            weight_input_name = node.input[1]  # w (INT8 weight)
            scale_input_name = node.input[5]   # w_scale
            zp_input_name = node.input[6]      # w_zp

            weight_init = self._get_initializer(model, weight_input_name)
            scale_init = self._get_initializer(model, scale_input_name)
            zp_init = self._get_initializer(model, zp_input_name)

            if weight_init is None or scale_init is None or zp_init is None:
                return None

            int8_weights = numpy_helper.to_array(weight_init).astype(np.float32)
            scale = numpy_helper.to_array(scale_init).flatten()
            zero_point = numpy_helper.to_array(zp_init).flatten()

            # Dequantize: float = (int8 - zero_point) * scale
            scale_val = scale[0]
            zp_val = int(zero_point[0])
            dequantized = (int8_weights - zp_val) * scale_val
            print(f"  [INT8 dequantized] size={dequantized.size}, shape={dequantized.shape}")
            return dequantized.flatten()

        # For QLinearMatMul and MatMulInteger:
        # QLinearMatMul inputs: [A, A_scale, A_zp, B, B_scale, B_zp, C_scale, C_zp]
        # Weight is input 3, w_scale is input 5, w_zp is input 6
        if node.op_type in {"QLinearMatMul", "MatMulInteger"}:
            if len(node.input) < 7:
                return None

            weight_input_name = node.input[3]  # B (INT8 weight)
            scale_input_name = node.input[5]   # B_scale
            zp_input_name = node.input[6]      # B_zp

            weight_init = self._get_initializer(model, weight_input_name)
            scale_init = self._get_initializer(model, scale_input_name)
            zp_init = self._get_initializer(model, zp_input_name)

            if weight_init is None or scale_init is None or zp_init is None:
                return None

            int8_weights = numpy_helper.to_array(weight_init).astype(np.float32)
            scale = numpy_helper.to_array(scale_init).flatten()
            zero_point = numpy_helper.to_array(zp_init).flatten()

            scale_val = scale[0]
            zp_val = int(zero_point[0])
            dequantized = (int8_weights - zp_val) * scale_val

            return dequantized.flatten()

        # For other quantized ops, fall back to direct weight extraction
        return self._extract_fp32_weight(model, node_name)

    def _find_node_by_name(self, model: ModelProto, name: str):
        """Find node by node name."""
        for node in model.graph.node:
            if node.name == name:
                return node
        return None

    def _find_first_quantized_node(self, model: ModelProto):
        """Find the first quantized op node in the model."""
        quant_ops = {"QLinearConv", "QLinearMatMul", "ConvInteger", "MatMulInteger"}
        for node in model.graph.node:
            if node.op_type in quant_ops:
                return node
        return None

    def _get_initializer(self, model: ModelProto, name: str):
        for init in model.graph.initializer:
            if init.name == name:
                return init
        return None

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _compute_distribution(self, weights: np.ndarray, bins: int) -> list[float]:
        hist, _ = np.histogram(weights, bins=bins)
        total = hist.sum()
        if total == 0:
            return [0.0] * bins
        return (hist / total).tolist()


weight_service = WeightService()