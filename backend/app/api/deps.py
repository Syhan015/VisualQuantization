from fastapi import Depends, HTTPException, status
from pathlib import Path
from app.core.config import config
from app.core.exceptions import ModelNotFoundError


def get_model_path(model_id: str) -> Path:
    path = config.MODELS_DIR / f"{model_id}.onnx"
    if not path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Model {model_id} not found",
        )
    return path
