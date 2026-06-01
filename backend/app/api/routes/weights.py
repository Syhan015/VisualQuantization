from fastapi import APIRouter, HTTPException, status

from app.core.config import config
from app.models.schemas import (
    WeightAnalyzeRequest,
    WeightAnalysisResponse,
    WeightCompareRequest,
)
from app.services.weight_service import weight_service

router = APIRouter()


@router.post("/analyze", response_model=WeightAnalysisResponse)
async def analyze_weight(request: WeightAnalyzeRequest):
    model_path = config.MODELS_DIR / f"{request.model_id}.onnx"

    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {request.model_id} not found",
        )

    try:
        result = weight_service.analyze_weight(model_path, request.node_name)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to analyze weight: {e}",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Node '{request.node_name}' not found or has no weights",
        )

    return result


@router.post("/compare", response_model=WeightAnalysisResponse)
async def compare_weights(request: WeightCompareRequest):
    model_a_path = config.MODELS_DIR / f"{request.model_a_id}.onnx"
    model_b_path = config.MODELS_DIR / f"{request.model_b_id}.onnx"

    if not model_a_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model A ({request.model_a_id}) not found",
        )
    if not model_b_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model B ({request.model_b_id}) not found",
        )

    try:
        result = weight_service.compare_weights(
            model_a_path,
            model_b_path,
            request.node_name,
            request.model_b_node_name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare weights: {e}",
        )

    if not request.node_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="node_name is empty",
        )

    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No weights found for node '{request.node_name}' (node may not have trainable weights)",
        )

    return result
