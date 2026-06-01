from pydantic import BaseModel
from typing import Optional


class ModelUploadResponse(BaseModel):
    id: str
    name: str
    message: str


class ModelMetadataResponse(BaseModel):
    id: str
    name: str
    node_count: int
    input_shapes: list[str]
    output_shapes: list[str]
    size_bytes: int


class DiffCompareRequest(BaseModel):
    model_a_id: str
    model_b_id: str


class DiffCompareResponse(BaseModel):
    id: str
    message: str


class QuantizationInfoResponse(BaseModel):
    is_quantized: bool
    quant_type: Optional[str] = None
    scale: Optional[float] = None
    zero_point: Optional[int] = None
    axis: Optional[int] = None


class FusionInfoResponse(BaseModel):
    is_fusion: bool
    fp32_components: Optional[list[str]] = None
    confidence: Optional[float] = None


class DiffNodeResponse(BaseModel):
    id: str
    name: str
    op_type: str
    diff_type: str
    details: Optional[str] = None
    quantization: Optional[QuantizationInfoResponse] = None
    matched_name: Optional[str] = None
    fusion_info: Optional[FusionInfoResponse] = None


class DiffSummaryResponse(BaseModel):
    total_nodes_a: int
    total_nodes_b: int
    added_count: int
    removed_count: int
    modified_count: int
    quantization_detected: bool = False
    quant_nodes_a: int = 0
    quant_nodes_b: int = 0


class DiffResultResponse(BaseModel):
    id: str
    model_a_id: str
    model_b_id: str
    nodes: list[DiffNodeResponse]
    summary: DiffSummaryResponse


class WeightAnalyzeRequest(BaseModel):
    model_id: str
    node_name: str


class WeightAnalysisResponse(BaseModel):
    node_name: str
    cosine_similarity: float
    l2_error: float
    mae: float
    mean_diff: float
    std_diff: float
    distribution_a: list[float]
    distribution_b: list[float]
    bin_edges: Optional[list[float]] = None


class WeightCompareRequest(BaseModel):
    model_a_id: str
    model_b_id: str
    node_name: str
    model_b_node_name: Optional[str] = None
