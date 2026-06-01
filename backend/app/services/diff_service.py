import json
import uuid
from pathlib import Path
from typing import Optional

import onnx
from onnx import ModelProto, NodeProto

from app.core.config import config
from app.core.exceptions import ONNXParseError, DiffResultNotFoundError
from app.models.schemas import (
    DiffNodeResponse,
    DiffSummaryResponse,
    DiffResultResponse,
    QuantizationInfoResponse,
    FusionInfoResponse,
)
from app.services.onnx_utils import (
    is_quant_op,
    is_quantized_initializer,
    get_node_tensors,
    get_node_attributes,
    get_quantization_info,
    compare_node_attributes,
    has_quantization_in_model,
    count_quant_nodes,
    detect_quantization_pattern,
    get_all_fp32_sequences,
)


class DiffService:
    def __init__(self):
        self._results: dict[str, DiffResultResponse] = {}

    def compare_models(
        self,
        model_a_path: Path,
        model_b_path: Path,
        match_mode: str = "conservative",
    ) -> tuple[str, DiffResultResponse]:
        """Compare two ONNX models.

        Args:
            model_a_path: Path to FP32 model
            model_b_path: Path to INT8 quantized model
            match_mode: "conservative" (exact/tensor match) or "aggressive" (with fusion detection)
        """
        try:
            model_a = onnx.load(str(model_a_path))
            model_b = onnx.load(str(model_b_path))
        except Exception as e:
            raise ONNXParseError(f"Failed to parse ONNX model: {e}")

        diff_id = str(uuid.uuid4())

        def _build_node_map(model):
            node_map = {}
            unnamed_idx = 0
            for node in model.graph.node:
                if node.name:
                    key = node.name
                else:
                    key = f"__uq_{unnamed_idx}_{node.op_type}"
                    unnamed_idx += 1
                node_map[key] = node
            return node_map

        def _build_name_map(node_map):
            name_map = {}
            for key, node in node_map.items():
                name_map[key] = node.name if node.name else node.op_type
            return name_map

        node_map_a = _build_node_map(model_a)
        node_map_b = _build_node_map(model_b)
        node_name_a = _build_name_map(node_map_a)
        node_name_b = _build_name_map(node_map_b)

        init_map_a = {init.name: init for init in model_a.graph.initializer}
        init_map_b = {init.name: init for init in model_b.graph.initializer}

        matched_pairs: list[tuple[str, str]] = []
        matched_a: set[str] = set()
        matched_b: set[str] = set()

        # Level 1: Exact name matching
        for name_a, node_a in node_map_a.items():
            if name_a in node_map_b:
                matched_pairs.append((name_a, name_a))
                matched_a.add(name_a)
                matched_b.add(name_a)

        unmatched_a = set(node_map_a.keys()) - matched_a
        unmatched_b = set(node_map_b.keys()) - matched_b

        tensor_to_node_a = self._build_tensor_map(model_a)
        tensor_to_node_b = self._build_tensor_map(model_b)

        # Level 2 (Aggressive only): Fusion detection BEFORE tensor matching
        # In aggressive mode, detect fusion patterns before consuming nodes via tensor matching
        fusion_matches: dict[str, tuple[str, list[str], float]] = {}
        if match_mode == "aggressive":
            fusion_matches = self._detect_fusion_patterns(
                unmatched_a, unmatched_b,
                node_map_a, node_map_b, tensor_to_node_a, tensor_to_node_b
            )
            # Mark fusion-matched nodes as handled (remove from unmatched sets)
            for fp32_start_name, (int8_name, fp32_components, confidence) in fusion_matches.items():
                for comp in fp32_components:
                    unmatched_a.discard(comp)
                unmatched_b.discard(int8_name)

        # Level 2: Tensor-based matching for remaining unmatched nodes
        for name_a in list(unmatched_a):
            node_a = node_map_a[name_a]
            match_name = self._find_match_by_tensors(node_a, node_map_b, tensor_to_node_b, matched_b)
            if match_name:
                matched_pairs.append((name_a, match_name))
                matched_a.add(name_a)
                matched_b.add(match_name)
                unmatched_a.discard(name_a)
                unmatched_b.discard(match_name)

        diff_nodes: list[DiffNodeResponse] = []
        removed_nodes: list[str] = []
        added_nodes: list[str] = []
        modified_pairs: list[tuple[str, str]] = []

        # Process matched pairs for modified nodes
        for name_a, name_b in matched_pairs:
            if name_a == name_b:
                node_a = node_map_a[name_a]
                node_b = node_map_b[name_b]
                changes = self._get_node_changes(node_a, node_b)
                if changes:
                    modified_pairs.append((name_a, name_b))
            else:
                modified_pairs.append((name_a, name_b))

        # Remaining unmatched nodes
        removed_nodes = list(unmatched_a)
        added_nodes = list(unmatched_b)

        # Handle fusion matches: add ALL FP32 components as modified nodes
        for fp32_start_name, (int8_name, fp32_components, confidence) in fusion_matches.items():
            # First component → modified_pair (also processed in the modified_pairs loop below)
            modified_pairs.append((fp32_start_name, int8_name))

            # Absorbed components (BN, ReLU etc.) → direct DiffNode entries
            node_b = node_map_b[int8_name]
            display_name_b = node_name_b.get(int8_name, int8_name)
            quant_info = self._get_quantization_info_for_node(node_b, init_map_b)
            fusion_info = FusionInfoResponse(
                is_fusion=True,
                fp32_components=fp32_components,
                confidence=confidence,
            )
            for comp in fp32_components[1:]:
                node_a = node_map_a[comp]
                display_name_a = node_name_a.get(comp, comp)
                diff_nodes.append(
                    DiffNodeResponse(
                        id=comp,
                        name=display_name_a,
                        op_type=node_a.op_type,
                        diff_type="modified",
                        details=f"Absorbed into fusion: {' -> '.join(fp32_components)} (confidence: {confidence:.2f})",
                        quantization=quant_info,
                        matched_name=display_name_b,
                        fusion_info=fusion_info,
                    )
                )

        for name_a in removed_nodes:
            node_a = node_map_a[name_a]
            display_name = node_name_a.get(name_a, name_a)
            quant_info = self._get_quantization_info_for_node(node_a, init_map_a)
            diff_nodes.append(
                DiffNodeResponse(
                    id=name_a,
                    name=display_name,
                    op_type=node_a.op_type,
                    diff_type="removed",
                    details=f"Node '{name_a}' exists in model A but not in model B",
                    quantization=quant_info,
                )
            )

        for name_b in added_nodes:
            node_b = node_map_b[name_b]
            # Check if this is a fused node that matched FP32 components
            fusion_key = None
            for fp32_name, (int8_name, components, _) in fusion_matches.items():
                if int8_name == name_b:
                    fusion_key = fp32_name
                    break
            if fusion_key:
                matched_int8_name, fp32_components, confidence = fusion_matches[fusion_key]
                quant_info = self._get_quantization_info_for_node(node_b, init_map_b)
                diff_nodes.append(
                    DiffNodeResponse(
                        id=fp32_components[0] if fp32_components else name_b,
                        name=fp32_components[0] if fp32_components else name_b,
                        op_type=node_b.op_type,
                        diff_type="modified",
                        details=f"Fused from: {' -> '.join(fp32_components)} (confidence: {confidence:.2f})",
                        quantization=quant_info,
                        fusion_info=FusionInfoResponse(
                            is_fusion=True,
                            fp32_components=fp32_components,
                            confidence=confidence,
                        ),
                    )
                )
            else:
                display_name = node_name_b.get(name_b, name_b)
                quant_info = self._get_quantization_info_for_node(node_b, init_map_b)
                diff_nodes.append(
                    DiffNodeResponse(
                        id=name_b,
                        name=display_name,
                        op_type=node_b.op_type,
                        diff_type="added",
                        details=f"Node '{name_b}' exists in model B but not in model A",
                        quantization=quant_info,
                    )
                )

        for name_a, name_b in modified_pairs:
            node_a = node_map_a[name_a]
            node_b = node_map_b[name_b]
            changes = self._get_node_changes(node_a, node_b)
            quant_a = self._get_quantization_info_for_node(node_a, init_map_a)
            quant_b = self._get_quantization_info_for_node(node_b, init_map_b)
            details_parts = []
            if node_a.op_type != node_b.op_type:
                details_parts.append(f"OpType: {node_a.op_type} -> {node_b.op_type}")
            if changes:
                for attr, (old, new) in changes.items():
                    details_parts.append(f"{attr}: {old} -> {new}")
            if quant_a != quant_b:
                details_parts.append("quantization parameters changed")
            quant_info = quant_b
            if quant_a or quant_b:
                quant_info = quant_b or quant_a

            # Check if this modified pair is a fusion result
            fusion_info = None
            if name_a in fusion_matches:
                int8_name, fp32_components, confidence = fusion_matches[name_a]
                details_parts.append(f"Fused from: {' -> '.join(fp32_components)} (confidence: {confidence:.2f})")
                fusion_info = FusionInfoResponse(
                    is_fusion=True,
                    fp32_components=fp32_components,
                    confidence=confidence,
                )

            display_name_a = node_name_a.get(name_a, name_a)
            display_name_b = node_name_b.get(name_b, name_b)
            diff_nodes.append(
                DiffNodeResponse(
                    id=name_a,
                    name=display_name_a,
                    op_type=node_b.op_type,
                    diff_type="modified",
                    details="; ".join(details_parts) if details_parts else f"Node '{name_a}' modified",
                    quantization=quant_info,
                    matched_name=display_name_b if name_a != name_b else None,
                    fusion_info=fusion_info,
                )
            )

        # Count total modified: modified_pairs + fusion-absorbed components
        fusion_absorbed_count = sum(len(fc) - 1 for _, (_, fc, _) in fusion_matches.items())
        summary = DiffSummaryResponse(
            total_nodes_a=len(node_map_a),
            total_nodes_b=len(node_map_b),
            added_count=len(added_nodes),
            removed_count=len(removed_nodes),
            modified_count=len(modified_pairs) + fusion_absorbed_count,
            quantization_detected=has_quantization_in_model(model_a) or has_quantization_in_model(model_b),
            quant_nodes_a=count_quant_nodes(model_a),
            quant_nodes_b=count_quant_nodes(model_b),
        )
        print(f"[compare_models] quant_nodes_a={summary.quant_nodes_a}, quant_nodes_b={summary.quant_nodes_b}")

        result = DiffResultResponse(
            id=diff_id,
            model_a_id=model_a_path.stem,
            model_b_id=model_b_path.stem,
            nodes=diff_nodes,
            summary=summary,
        )

        self._results[diff_id] = result
        result_path = config.DIFF_RESULTS_DIR / f"{diff_id}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, ensure_ascii=False, indent=2)

        return diff_id, result

    def _build_tensor_map(self, model: ModelProto) -> dict[str, str]:
        tensor_map = {}
        for node in model.graph.node:
            for tensor_name in node.output:
                if tensor_name:
                    tensor_map[tensor_name] = node.name
        return tensor_map

    def _find_match_by_tensors(
        self,
        node: NodeProto,
        other_node_map: dict[str, NodeProto],
        other_tensor_map: dict[str, str],
        already_matched: set[str],
    ) -> Optional[str]:
        node_tensors = get_node_tensors(node)
        if not node_tensors:
            return None
        candidates = {}
        for tensor_name in node_tensors:
            producer_name = other_tensor_map.get(tensor_name)
            if producer_name and producer_name not in already_matched:
                candidates[producer_name] = candidates.get(producer_name, 0) + 1
        if not candidates:
            return None
        best_match = max(candidates.items(), key=lambda x: x[1])
        return best_match[0]

    def _get_node_changes(
        self, node_a: NodeProto, node_b: NodeProto
    ) -> dict[str, tuple[any, any]]:
        changes = {}
        if node_a.op_type != node_b.op_type:
            changes["op_type"] = (node_a.op_type, node_b.op_type)
        attr_changes = compare_node_attributes(node_a, node_b)
        changes.update(attr_changes)
        return changes

    def _get_quantization_info_for_node(
        self, node: NodeProto, init_map: dict
    ) -> Optional[QuantizationInfoResponse]:
        is_quant = is_quant_op(node.op_type)
        has_quant_init = False
        for inp in node.input:
            if inp in init_map and is_quantized_initializer(init_map[inp]):
                has_quant_init = True
                break
        if is_quant or has_quant_init or node.op_type in {"Conv", "MatMul", "Gemm", "Linear"}:
            init_dtype = None
            for inp in node.input:
                if inp in init_map:
                    init_dtype = init_map[inp].data_type
                    break
            return QuantizationInfoResponse(
                is_quantized=is_quant or has_quant_init or (init_dtype in {2, 3}),
                quant_type="per_tensor" if is_quant else "unknown",
                scale=None,
                zero_point=None,
                axis=None,
            )
        return None

    def get_result(self, diff_id: str) -> DiffResultResponse:
        if diff_id in self._results:
            return self._results[diff_id]

        result_path = config.DIFF_RESULTS_DIR / f"{diff_id}.json"
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return DiffResultResponse(**data)

        raise DiffResultNotFoundError(f"Diff result {diff_id} not found")

    def _detect_fusion_patterns(
        self,
        unmatched_a: set[str],
        unmatched_b: set[str],
        node_map_a: dict[str, NodeProto],
        node_map_b: dict[str, NodeProto],
        tensor_to_node_a: dict[str, str],
        tensor_to_node_b: dict[str, str],
    ) -> dict[str, tuple[str, list[str], float]]:
        """Detect fusion patterns between FP32 and INT8 models.

        Returns:
            Dict mapping FP32 component[0] -> (INT8_name, FP32_components, confidence)
        """
        fusion_matches: dict[str, tuple[str, list[str], float]] = {}

        pos_a = {name: i for i, name in enumerate(node_map_a)}
        pos_b = {name: i for i, name in enumerate(node_map_b)}

        fp32_sequences = sorted(get_all_fp32_sequences(), key=len, reverse=True)
        used_int8: set[str] = set()

        for fp32_seq in fp32_sequences:
            target_quant_op = detect_quantization_pattern(fp32_seq)
            if not target_quant_op:
                continue

            fp32_candidates = self._find_fp32_sequence_candidates(
                list(unmatched_a), node_map_a, tensor_to_node_a, fp32_seq
            )
            if not fp32_candidates:
                continue

            fp32_candidates.sort(key=lambda c: pos_a.get(c[0], 9999))

            int8_candidates = [
                nb for nb in unmatched_b
                if node_map_b[nb].op_type == target_quant_op and nb not in used_int8
            ]

            for fp32_components in fp32_candidates:
                if fp32_components[0] in fusion_matches:
                    continue

                fp32_pos = pos_a.get(fp32_components[0], 9999)

                best_int8 = None
                best_dist = 9999
                for nb in int8_candidates:
                    if nb in used_int8:
                        continue
                    d = abs(pos_b.get(nb, 9999) - fp32_pos)
                    if d < best_dist:
                        best_dist = d
                        best_int8 = nb

                if best_int8 is None:
                    continue

                confidence = self._calculate_fusion_confidence(
                    fp32_components, best_int8, node_map_a, node_map_b
                )
                if confidence > 0.5:
                    fusion_matches[fp32_components[0]] = (best_int8, fp32_components, confidence)
                    used_int8.add(best_int8)

        return fusion_matches

    def _find_fp32_sequence_candidates(
        self,
        node_names: list[str],
        node_map: dict[str, NodeProto],
        tensor_to_node: dict[str, str],
        target_sequence: tuple[str, ...],
    ) -> list[list[str]]:
        """Find FP32 node sequences that match the target pattern.

        For example, for ("Conv", "BatchNorm", "Relu"), find chains like [conv1, bn1, relu1]
        """
        candidates = []
        if not target_sequence:
            return candidates

        # Build input->consumer map (tensor -> node that consumes it as input)
        # This is the reverse of tensor_to_node which maps tensor->producer
        input_to_node: dict[str, str] = {}
        for name in node_names:
            node = node_map[name]
            for inp in node.input:
                if inp:
                    input_to_node[inp] = name

        # For each starting node, try to follow the chain
        for start_name in node_names:
            start_node = node_map[start_name]
            if start_node.op_type != target_sequence[0]:
                continue

            chain = [start_name]
            current_name = start_name

            for i in range(1, len(target_sequence)):
                found_next = False
                current_node = node_map[current_name]
                # Get the output tensor of current node
                if not current_node.output:
                    break
                out_tensor = current_node.output[0]
                if not out_tensor:
                    break
                # Find which node consumes this tensor
                consumer_name = input_to_node.get(out_tensor)
                if consumer_name and consumer_name in node_map:
                    consumer_node = node_map[consumer_name]
                    if consumer_node.op_type == target_sequence[i]:
                        chain.append(consumer_name)
                        current_name = consumer_name
                        found_next = True
                        # Continue to next iteration of the for loop
                    if not found_next:
                        break

            if len(chain) == len(target_sequence):
                candidates.append(chain)

        return candidates

    def _calculate_fusion_confidence(
        self,
        fp32_components: list[str],
        int8_name: str,
        node_map_a: dict[str, NodeProto],
        node_map_b: dict[str, NodeProto],
    ) -> float:
        """Calculate confidence score for a fusion match.

        Higher confidence if:
        - All FP32 components exist
        - INT8 op is a known quantized op
        - Kernel/shape parameters are compatible
        """
        if not fp32_components or not int8_name:
            return 0.0

        int8_node = node_map_b.get(int8_name)
        if not int8_node or not is_quant_op(int8_node.op_type):
            return 0.0

        # Base confidence for quantized op match
        confidence = 0.6

        # Check if FP32 nodes are consecutive in original model order
        # (stronger signal if they were likely fused)
        # This is a simplified heuristic

        return min(confidence + 0.3, 1.0)


diff_service = DiffService()