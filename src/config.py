"""
Configuration module for the attendance system.
Loads environment variables and provides centralized configuration.
"""

import os
from pathlib import Path
from typing import Tuple, Optional

from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
LOGS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)
EMBEDDINGS_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)


class CameraConfig:
    """Camera configuration settings."""
    
    RTSP_URL: str = os.getenv("RTSP_URL", "")
    WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1280"))
    HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "720"))
    FPS: int = int(os.getenv("CAMERA_FPS", "30"))
    
    @classmethod
    def validate(cls) -> bool:
        """Validate camera configuration."""
        if not cls.RTSP_URL:
            logger.warning("RTSP_URL is not set in environment variables")
            return False
        return True


class InsightFaceConfig:
    """InsightFace model configuration."""
    
    MODEL_NAME: str = os.getenv("INSIGHTFACE_MODEL", "buffalo_s")
    DET_SIZE: Tuple[int, int] = tuple(
        map(int, os.getenv("INSIGHTFACE_DET_SIZE", "640,640").split(","))
    )
    DET_THRESH: float = float(os.getenv("INSIGHTFACE_DET_THRESH", "0.5"))


class RecognitionConfig:
    """Face recognition configuration."""
    
    THRESHOLD: float = float(os.getenv("RECOGNITION_THRESHOLD", "0.4"))
    EMBEDDING_DIM: int = int(os.getenv("EMBEDDING_DIM", "512"))


class QdrantConfig:
    """Qdrant Vector DB configuration."""
    
    URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY", None)
    LOCATION: str = os.getenv("QDRANT_LOCATION", ":memory:") # Default to memory for easy testing
    COLLECTION_NAME: str = os.getenv("QDRANT_COLLECTION", "face_attendance")


class ApiConfig:
    """FastAPI configuration."""
    
    HOST: str = os.getenv("API_HOST", "0.0.0.0")
    PORT: int = int(os.getenv("API_PORT", "8000"))


class LogConfig:
    """Logging configuration."""
    
    LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    FILE: Path = LOGS_DIR / os.getenv("LOG_FILE", "app.log")
    ENABLE_CONSOLE: bool = os.getenv("ENABLE_CONSOLE_LOG", "true").lower() == "true"


# Export all configs
__all__ = [
    "PROJECT_ROOT",
    "LOGS_DIR",
    "DATA_DIR",
    "EMBEDDINGS_DIR",
    "MODELS_DIR",
    "CameraConfig",
    "InsightFaceConfig",
    "RecognitionConfig",
    "QdrantConfig",
    "ApiConfig",
    "LogConfig",
]
