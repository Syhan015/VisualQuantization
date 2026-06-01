from fastapi import APIRouter, HTTPException, status, Query

from app.core.config import config
from app.core.exceptions import ModelNotFoundError
from app.models.schemas import (
    DiffCompareRequest,
    DiffCompareResponse,
    DiffResultResponse,
)
from app.services.diff_service import diff_service

router = APIRouter()


@router.post("/compare", response_model=DiffCompareResponse)
async def compare_models(
    request: DiffCompareRequest,
    match_mode: str = Query("conservative", regex="^(conservative|aggressive)$"),
):
    """Compare two models.

    Args:
        request: Contains model_a_id and model_b_id
        match_mode: "conservative" (exact/tensor match) or "aggressive" (with fusion detection)
    """
    model_a_path = config.MODELS_DIR / f"{request.model_a_id}.onnx"
    model_b_path = config.MODELS_DIR / f"{request.model_b_id}.onnx"

    if not model_a_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model A {request.model_a_id} not found",
        )
    if not model_b_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model B {request.model_b_id} not found",
        )

    try:
        diff_id, _ = diff_service.compare_models(model_a_path, model_b_path, match_mode=match_mode)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to compare models: {e}",
        )

    return DiffCompareResponse(id=diff_id, message=f"Diff comparison started ({match_mode} mode)")


@router.get("/{diff_id}/result", response_model=DiffResultResponse)
async def get_diff_result(diff_id: str):
    try:
        result = diff_service.get_result(diff_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Diff result {diff_id} not found: {e}",
        )

    return result
