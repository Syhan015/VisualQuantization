from pathlib import Path
import os


class Config:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    MODELS_DIR = PROJECT_ROOT / "models"
    DIFF_RESULTS_DIR = PROJECT_ROOT / "diff_results"

    @classmethod
    def ensure_dirs(cls) -> None:
        cls.MODELS_DIR.mkdir(exist_ok=True)
        cls.DIFF_RESULTS_DIR.mkdir(exist_ok=True)


config = Config()
