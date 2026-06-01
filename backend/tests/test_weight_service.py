import pytest
from pathlib import Path
import tempfile
import numpy as np
import onnx
from onnx import helper, numpy_helper

from app.services.weight_service import WeightService


def create_model_with_weight(weight_data: np.ndarray) -> Path:
    init = numpy_helper.from_array(weight_data, name="test_weight")
    node = helper.make_node("Relu", ["x"], ["y"], name="test_node")

    graph = helper.make_graph(
        nodes=[node],
        name="test_model",
        inputs=[],
        outputs=[],
        initializer=[init],
    )
    model = helper.make_model(graph)
    path = Path(tempfile.gettempdir()) / "weight_test.onnx"
    onnx.save(model, str(path))
    return path


def test_analyze_weight():
    service = WeightService()

    weight_data = np.random.randn(100).astype(np.float32)
    model_path = create_model_with_weight(weight_data)

    result = service.analyze_weight(model_path, "test_weight")

    assert result is not None
    assert result.node_name == "test_weight"
    assert len(result.distribution_a) == 50


def test_analyze_nonexistent_weight():
    service = WeightService()

    weight_data = np.random.randn(10).astype(np.float32)
    model_path = create_model_with_weight(weight_data)

    result = service.analyze_weight(model_path, "nonexistent")

    assert result is None
