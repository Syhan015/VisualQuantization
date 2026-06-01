import uuid
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse
import onnx

from app.core.config import config
from app.core.exceptions import ONNXParseError
from app.models.schemas import (
    ModelUploadResponse,
    ModelMetadataResponse,
)

router = APIRouter()


@router.post("/upload", response_model=ModelUploadResponse)
async def upload_model(file: UploadFile = File(...)):
    if not file.filename.endswith(".onnx"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .onnx files are supported",
        )

    config.ensure_dirs()

    model_id = str(uuid.uuid4())
    file_path = config.MODELS_DIR / f"{model_id}.onnx"

    try:
        contents = await file.read()
        onnx.load_from_string(contents)
        with open(file_path, "wb") as f:
            f.write(contents)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid ONNX file: {e}",
        )

    return ModelUploadResponse(
        id=model_id,
        name=file.filename,
        message="Model uploaded successfully",
    )


@router.get("/{model_id}", response_model=ModelMetadataResponse)
async def get_model_metadata(model_id: str):
    model_path = config.MODELS_DIR / f"{model_id}.onnx"

    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    try:
        model = onnx.load(str(model_path))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse ONNX model: {e}",
        )

    input_shapes = []
    for inp in model.graph.input:
        shape = []
        for dim in inp.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else -1)
        input_shapes.append(f"{inp.name}: {shape}")

    output_shapes = []
    for out in model.graph.output:
        shape = []
        for dim in out.type.tensor_type.shape.dim:
            shape.append(dim.dim_value if dim.dim_value > 0 else -1)
        output_shapes.append(f"{out.name}: {shape}")

    return ModelMetadataResponse(
        id=model_id,
        name=model_path.name,
        node_count=len(model.graph.node),
        input_shapes=input_shapes,
        output_shapes=output_shapes,
        size_bytes=model_path.stat().st_size,
    )


@router.get("/{model_id}/download")
async def download_model(model_id: str):
    """Download the model file - used by Netron iframe"""
    model_path = config.MODELS_DIR / f"{model_id}.onnx"

    if not model_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )

    return FileResponse(
        path=str(model_path),
        media_type="application/octet-stream",
        filename=model_path.name,
    )
